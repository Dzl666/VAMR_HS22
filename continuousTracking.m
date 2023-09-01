function [P_total, X_total, C_total, F_total, Tao_total, C_cnt_total, T_i_W2C] = continuousTracking(img_prev, img,...
    camParams, cfgs, P_prev, X_prev, C_prev, F_prev, Tao_prev, C_cnt_prev)
% P_i [2, n] - kpt in the i-th frame that has correspongding landmark X_i [3, n]
% C_i [2, m] - kpt in the i-th frame that doesn't match a landmark
% F_i [2, m]- kpt at the first frame that a tracked candidate appeared, corres. to each C_i
% Tao_i [12, m] - cam pose of first frame that F_i occur, W2C
% C_cnt_i [m] - count successful tracked frames of a candidate

    %% Track all key points
    % use P_i-1 to track P_i by KLT
    kptTracker = vision.PointTracker('MaxBidirectionalError', cfgs.max_KLT_bidir_err,...
        'NumPyramidLevels', cfgs.KLT_pyrmid_level);
    initialize(kptTracker, P_prev', img_prev);
    [P_i, valid_P] = kptTracker(img);
    release(kptTracker);
    P_i = P_i(valid_P, :);
    if cfgs.ds == 2
        P_i = round(P_i);
    end
    X_i = X_prev(:, valid_P)';
    fprintf('Kpts tracked: %d\n', nnz(valid_P));

    %% RANSAC P3P + NL pose optimization
    % use P_i and X_i to estimate cam pose T_CW in the current frame
    [R_W2C, t_W2C, inliers_kpt] = estimateWorldCameraPose(P_i, X_i, camParams, 'Confidence', cfgs.ransac_conf,...
        'MaxNumTrials', cfgs.max_ransac_iters, 'MaxReprojectionError', cfgs.max_ransac_reproj_err);
    P_i = P_i(inliers_kpt, :);
    X_i = X_i(inliers_kpt, :);

    % for kpt not passing the test, store them in C_t ?

    % pose optimization
    T_rigid = rigidtform3d(R_W2C, t_W2C);
    T_rigid_refine = bundleAdjustmentMotion(X_i, P_i, T_rigid, camParams, 'PointsUndistorted', true);
    R_i_W2C = T_rigid_refine.Rotation';
    t_i_W2C = -R_i_W2C * T_rigid_refine.Translation';
    T_i_W2C = [R_i_W2C, t_i_W2C];

    %% Track all candidates
    C_i = [];
    F_i = F_prev';
    Tao_i = Tao_prev';
    C_cnt_i = C_cnt_prev;
    % use KLT to track kpt in C_i-1
    if not(isempty(C_prev))
        candTracker = vision.PointTracker('MaxBidirectionalError', cfgs.max_KLT_bidir_err,...
            'NumPyramidLevels', cfgs.KLT_pyrmid_level);
        initialize(candTracker, C_prev', img_prev);   
        [C_i, valid_C] = candTracker(img);
        release(candTracker);
        C_i = C_i(valid_C, :);
        if cfgs.ds == 2
            C_i = round(C_i);
        end
        F_i = F_prev(:, valid_C)';
        Tao_i = Tao_prev(:, valid_C)';
        C_cnt_i = C_cnt_prev(valid_C) + 1;
    end
    fprintf('Cands tracked: %d\n', size(C_i, 1));

    %% Find new candidates
    harris_features = detectHarrisFeatures(img, 'MinQuality', cfgs.min_harris_q);
    img_corners = selectStrongest(harris_features, cfgs.max_corners).Location;

    exist_fea = [P_i; C_i];
    dist_new_exist_fea = min(pdist2(img_corners, exist_fea), [], 2);
    % take care about rudundant points !!!!!
    C_add = round(img_corners(dist_new_exist_fea > cfgs.min_track_displm, :));
    Tao_add = repmat(reshape(T_i_W2C, [1, 12]), [size(C_add, 1), 1]);
    C_cnt_add = ones(1, size(C_add, 1));
    
    %% Triangulate new points and landmarks from candidates
    P_add = [];
    X_add = [];
    F_corres = [];
    % perform triangulate check for each kpt in C_i
    if not(isempty(C_i))
        % [num_C, 1]
        angles = calculateCandidateAngle(C_i', T_i_W2C, F_i', Tao_i', camParams.IntrinsicMatrix');

        valid_angle = abs(angles) > cfgs.min_triangulate_angle;
        valid_cnt = C_cnt_i > cfgs.min_cons_frames;
        valid_new_kpt = (valid_cnt' + valid_angle) > 0;

        P_add = C_i(valid_new_kpt, :);
        F_corres = F_i(valid_new_kpt, :);
        Tao_corres = Tao_i(valid_new_kpt, :);
        C_cnt_corres = C_cnt_i(valid_new_kpt);

        X_add = zeros(size(P_add, 1), 3);
        valid_add = zeros(size(P_add, 1), 1);

        % triangulate & checking
        M2 = cameraMatrix(camParams, R_i_W2C, t_i_W2C);
        max_triangulate_reproj_err = cfgs.max_reproj_err;
        max_triangulate_P3d_dist = cfgs.max_dist_P3d;
        parfor k = 1 : size(P_add, 1)
            Tao_temp = reshape(Tao_corres(k, :), [3,4]); % W2C
            M1 = cameraMatrix(camParams, Tao_temp(:, 1:3), Tao_temp(:, 4));
            
            [X_add(k,:), reproj_err, valid_add(k)] = triangulate(F_corres(k,:), P_add(k,:), M1, M2);
            % delete kpts with too large reprojection err / P3d too far away
            if reproj_err > max_triangulate_reproj_err || norm(abs(X_add(k,:)') - abs(t_i_W2C)) > max_triangulate_P3d_dist
                valid_add(k) = 0;
            end
        end

        % re-assign all points
        valid_add = valid_add > 0;
        if not(isempty(P_add))
            % clean C, F, Tao, C_cnt
            invalid_add = not(valid_add);
            invalid_new_kpt = not(valid_new_kpt);
            
            C_i = [P_add(invalid_add, :); C_i(invalid_new_kpt, :)];
            F_i = [F_corres(invalid_add, :); F_i(invalid_new_kpt, :)];
            Tao_i = [Tao_corres(invalid_add, :); Tao_i(invalid_new_kpt, :)];
            C_cnt_i = [C_cnt_corres(invalid_add), C_cnt_i(invalid_new_kpt)];
            % clean P_add, X_add
            P_add = P_add(valid_add, :);
            F_corres = F_corres(valid_add, :);
            X_add = X_add(valid_add, :);
        end
        fprintf('Pairs triangulated: %d\n', nnz(valid_add));

        % optimize
        if not(isempty(P_add))
            u = P_add(:, 1);
            v = P_add(:, 2);
            kpt_array = [pointTrack(1, [u(1), v(1)])];
            parfor k = 2 : size(P_add, 1)
                kpt_array(k) = pointTrack(1, [u(k), v(k)]);
            end
            ViewId = uint32(1);
            AbsolutePose = rigidtform3d(R_i_W2C', (-R_i_W2C' * t_i_W2C)');
            tab = table(ViewId, AbsolutePose);

            X_add = bundleAdjustmentStructure(X_add, kpt_array, tab, camParams, 'PointsUndistorted', true);
        end
    end


    %% collect all particals
    P_total = [P_i; P_add]';
    X_total = [X_i; X_add]';
    C_total = [C_i; C_add]';
    F_total = [F_i; C_add]';
    Tao_total = [Tao_i; Tao_add]';
    C_cnt_total = [C_cnt_i, C_cnt_add];

    %% plotting
    if cfgs.plot_kpts_cands
        figure(3)
        subplot(2,3,1:3)
        imshow(img); hold on;
        % plot all tracked kpts
        plot(P_i(:,1), P_i(:,2), '.g');
        % plot all candidates that are tracked & not converted to kpts
        if not(isempty(C_i))
            plot(C_i(:,1), C_i(:,2), '.r');
        end
        % plot all new extracted candidates
        if not(isempty(C_add))
            plot(C_add(:,1), C_add(:,2), '.m')
        end
        %  plot new added P and corres. F
        if not(isempty(P_add))
            plot(P_add(:,1), P_add(:,2), 'cx'); 
            plot(F_corres(:,1), F_corres(:,2), 'bx');
            plot([P_add(:,1)'; F_corres(:,1)'], [P_add(:,2)'; F_corres(:,2)'], 'y-', 'Linewidth', 1);
        end
        title("kpt (g), cand (r), new cand (m), triangulate pair (y)"); hold off;
    end

    if cfgs.plot_cam_pose
        figure(3)
        subplot(2,3,4)
%         plot3(X_i(:,1), X_i(:,2), X_i(:,3), 'bo'); 
%         if not(isempty(X_add))
%             plot3(X_add(:,1), X_add(:,2), X_add(:,3), 'ro');
%         end
        axis equal; view(0,0); grid on; rotate3d on; hold on;
        t_C2W = -R_i_W2C' * t_i_W2C;
        plot3(t_C2W(1), t_C2W(2), t_C2W(3)); 
        plotCoordinateFrame(R_i_W2C', t_C2W, 1);
        xlim([t_C2W(1)-3, t_C2W(1)+3]);
        ylim([t_C2W(2)-3, t_C2W(2)+3]);
        zlim([t_C2W(3)-3, t_C2W(3)+3]);
        title('Cameras poses');
    end
 
end