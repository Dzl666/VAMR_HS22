function [P_total, X_total, C_total, F_total, Tao_total, C_cnt_total, T_i_W2C] = continuousTracking(img_prev, img,...
    camParams, cfgs, P_prev, X_prev, C_prev, F_prev, Tao_prev, C_cnt_prev, R_W2C_prev, t_W2C_prev)

% P_i [2, n] - kpt in the i-th frame that has correspongding landmark X_i [3, n]
% C_i [2, m] - kpt in the i-th frame that doesn't match a landmark
% F_i [2, m]- kpt at the first frame that a tracked candidate appeared, corres. to each C_i
% Tao_i [12, m] - cam pose of first frame that F_i occur
% C_cnt_i [m] - count successful tracked frames of a candidate

    %% Track all key points
    % use P_i-1 to track P_i by KLT
    kptTracker = vision.PointTracker('MaxBidirectionalError', cfgs.max_track_bidir_error);  
    initialize(kptTracker, P_prev', img_prev);
    [P_new, valid_P] = kptTracker(img);
    release(kptTracker);
    P_i = P_new(valid_P, :);
    if cfgs.ds == 2
        P_i = round(P_i);
    end
    X_i = X_prev(:, valid_P)';

    fprintf('Keypoints success tracked: %d\n', nnz(valid_P));


    %% RANSAC P3P + NL pose optimization
    % use P_i and X_i to estimate cam pose T_CW = [R_CW | t_CW] in the current frame
    % record the P_i and correspongding X_i-1(X_i) that pass the RANSAC
    % for kpt not passing the test, store them in C_t
    [R_W2C, t_W2C, inliers_kpt] = estimateWorldCameraPose(P_i, X_i, camParams, 'Confidence', cfgs.ransac_conf,...
        'MaxNumTrials', cfgs.max_ransac_iters, 'MaxReprojectionError', cgfs.max_ransac_err);
    P_i = P_i(inliers_kpt, :);
    X_i = X_i(inliers_kpt, :);

    fprintf('Keypoints success to localize: %d\n', nnz(inliers_kpt));

    T_rigid = rigid3d(R_W2C, t_W2C)
    T_rigid_refine = bundleAdjustmentMotion(X_i, P_i, T_rigid, camParams, 'PointsUndistorted', true);
    R_i_W2C = T_rigid_refine.Rotation';
    t_i_W2C = -R_i_W2C * T_rigid_refine.Translation';
    T_i_W2C = [R_i_W2C, t_i_W2C];

    %% Track all candidates
    % use KLT to track kpt in C_i-1, process triangulate check 
    % in each tracked kpt in frame i
    if not(isempty(C_prev))
        candTracker = vision.PointTracker('MaxBidirectionalError', cfgs.max_track_bidir_error);
        initialize(candTracker, C_prev', img_prev);   
        [C_new, valid_C] = candTracker(img);
        C_i = C_new(valid_C, :);
        if cfgs.ds == 2
            C_i = round(C_i);
        end
        F_i = F_prev(:, valid_C)';
        Tao_i = Tao_prev(:, valid_C)';
        C_cnt_i = C_cnt_prev(valid_C) + 1;

        fprintf('Candidate kpts success tracked: %d\n', nnz(valid_C));
    end

    %% Find new candidates
    % extract features
    harris_features = detectHarrisFeatures(img, 'MinQuality', cfgs.min_harris_q);
    img_corners = selectStrongest(harris_features, cfgs.max_corners).Location;

    % update the newly occur candidate kpt C_t into C_i and F_i, as well as Tao_i
    % take care about rudundant!!!!!
    exist_fea = [P_i; C_i];
    dist_new_exist_fea = min(pdist2(img_corners, exist_fea), [], 2);
    C_add = round(img_corners(dist_new_exist_fea > cfgs.min_track_displm, :));
    C_cnt_add = ones(1, size(C_add, 1));
    Tao_add = repmat(reshape(T_i_W2C, [1, 12]), [size(C_add, 1), 1]);
    
    %% Triangulate new points and landmarks from candidates
    if not(isempty(C_i))
        angles = calculateCandidateAngle(C_i, T_i_W2C, F_i, Tao_i, K)

        valid_angle = abs(angles) > cfgs.min_triangulate_angle;
        valid_cnt = C_cnt_i > cfgs.min_cons_frames;
        valid_new_kpt = (valid_cnt' + valid_angle) > 0;

        P_add = C_i(valid_new_kpt, :);
        F_corres = F_i(valid_new_kpt, :);
        Tao_corres = Tao_i(valid_new_kpt, :);
        cnt_corres = C_cnt_i(valid_new_kpt);
        X_add = zeros(size(P_add, 1), 3);
        reproj_err = []; valid_add = [];

        % triangulate & checking
        for k = 1 : size(P_add, 1)
            Tao_temp = reshape(Tao_corres(k, :), [3,4]);
            R_f = Tao_temp(:, 1:3);
            t_f = Tao_temp(:, 4);
            M1 = cameraMatrix(camParams, R_f, t_f);
            M2 = cameraMatrix(camParams, R_i_W2C, t_i_W2C);

            [X_add(k,:), reproj_err(k), valid_add(k)] = triangulate(F_corres(k,:), P_add(k,:), M1, M2);

            % checking
            if reproj_err(k) > cfgs.max_reproj_err || norm(X_add(k,:)' - t_i_W2C) > cfgs.max_dist_P3d
                valid_add(k) = 0
            end
        end

        % re-assign all points
        valid_add = valid_add > 0;
        if not(isempty(P_add))
            % clean C, F, Tao, C_cnt
            invalid_add = not(valid_add)
            invalid_new_kpt = not(valid_new_kpt)
            C_i = [P_add(invalid_add, :); C_i(invalid_new_kpt, :)];
            F_i = [F_corres(invalid_add, :); F_i(invalid_new_kpt, :)];
            Tao_i = [Tao_corres(invalid_add, :); Tao_i(invalid_new_kpt, :)];
            C_cnt_i = [cnt_corres(invalid_add, :); C_cnt_i(invalid_new_kpt, :)];
            % clean P_add, X_add
            P_add = P_add(valid_add, :);
            F_corres = F_corres(valid_add, :);
            X_add = X_add(valid_add, :);
        end
        fprintf('New 2D-3D pairs triangulated: %d\n', nnz(valid_add));

        % optimize
        if not(isempty(P_add))
            absPose = rigid3d(R_i_W2C', (-R_i_W2C' * t_i_W2C)');
            viewId = uint32(1);
            tab = table(viewId, absPose);
            u = P_add(:, 1); v = P_add(:, 2);
            kpt_array = [pointTrack(1, [u(1), v(1)])];
            parfor k = 2 : size(P_add, 1)
                kpt_array(k) = pointTrack(1, [u(k), v(k)]);
            end
            X_add = bundleAdjustmentStructure(X_add, kpt_array, tab, camParams, 'PointsUndistorted', true);
        end
    end


    %% collect all particals
    P_total = [P_i; P_add];
    X_total = [X_i; X_add];
    C_total = [C_i; C_add];
    F_total = [F_i; C_add];
    Tao_total = [Tao_i; Tao_add];
    C_cnt_total = [C_cnt_i, C_cnt_add];

    %% plotting
    if cfgs.plot_kpts_cands
        figure(3)
        subplot(2,2,2)
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
        %  
        if not(isempty(P_add))
            plot(P_add(:,1), P_add(:,2), 'cs'); 
            plot(F_corres(:,1), F_corres(:,2), 'bs');
            plot([P_add(:,1)'; F_corres(:,1)'], [P_add(:,2)'; F_corres(:,2)'], 'y-', 'Linewidth', 1);
        end
        title("kpt (green), cand (red), new cand (magenta)"); hold off;
    end

    if cfgs.plot_cam_pose
        figure(3)
        subplot(2,2,3)
        plot3(X_i(:,1), X_i(:,2), X_i(:,3), 'bo'); hold on;
        if not(isempty(X_add))
            plot3(X_add(:,1), X_add(:,2), X_add(:,3), 'ro');
        end

        t_C2W = -R_i_W2C' * t_i_W2C;
        plot3(t_C2W(1), t_C2W(2), t_C2W(3))
        plotCoordinateFrame(R_W2C_prev', -R_W2C_prev' * t_W2C_prev, 0.8);
        plotCoordinateFrame(R_i_W2C', t_C2W, 0.8);
        text(t_C2W(1)-0.1, t_C2W(2)-0.1, t_C2W(3)-0.1,'Cam','fontsize',10,'color','k','FontWeight','bold');
        axis equal; rotate3d on; grid on; view(0,0);
        title('Cameras poses'); hold off;
    end
 
end