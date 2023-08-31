function [R_C2W, t_C2W, kpt_init, P3d_init] = bootstrapKLT(img0, img1, camParams, cfgs)

    %% Feature tracking
    % find kpts from both imgs using Harris detector
    harris_features = detectHarrisFeatures(img0, 'MinQuality', cfgs.min_harris_q);
    % use [N, 2] (x, y) keypoints
    img0_corners = selectStrongest(harris_features, cfgs.max_corners_boot).Location;

    kptTracker = vision.PointTracker('MaxBidirectionalError', cfgs.max_KLT_bidir_err,...
        'NumPyramidLevels', cfgs.KLT_pyrmid_level, 'MaxIterations', 40);
    initialize(kptTracker, img0_corners, img0);
    [kpt_temp, valid_temp] = kptTracker(img1);
    release(kptTracker);

    % matched kpts stored in shape [2, N]
    matched_kpt0 = img0_corners(valid_temp, :)';
    matched_kpt1 = kpt_temp(valid_temp, :)';
    if cfgs.ds == 2
        matched_kpt0 = round(matched_kpt0);
        matched_kpt1 = round(matched_kpt1);
    end

    % delete matches with small displacement !!!
    dists = vecnorm(matched_kpt0 - matched_kpt1);
    thres = cfgs.min_KLT_displm;
    matched_kpt0 = matched_kpt0(:, dists > thres)';
    matched_kpt1 = matched_kpt1(:, dists > thres)';
    fprintf('kpts extracted in bootstrap: %d\n', size(matched_kpt1, 1));

    %% Solving relative pose

    % estimate F using lib func (with RANSAC)
    [F_RANSAC, inliers_kpt] = estimateFundamentalMatrix(matched_kpt0, matched_kpt1,...
        'Method','RANSAC', 'NumTrials', cfgs.max_ransac_iters,...
        'DistanceThreshold', cfgs.max_ransac_dist_err, 'Confidence', cfgs.ransac_conf);
        
    K = camParams.IntrinsicMatrix';
    E = K' * F_RANSAC * K;
    % matched kpts after RANSAC [N, 2]
    matched_p1 = matched_kpt0(inliers_kpt, :);
    matched_p2 = matched_kpt1(inliers_kpt, :);
    
    % get relative pose of 2 cams using lib func
    % pose of cam2 related to cam 1
    [R_C2W, t_C2W] = relativeCameraPose(E, camParams, matched_p1, matched_p2);
    t_C2W = t_C2W';
    
    % triangulate the first landmarks from the bootstrap imgs
    M1 = cameraMatrix(camParams, eye(3), [0,0,0]');
    M2 = cameraMatrix(camParams, R_C2W', -R_C2W' * t_C2W);
 
    [P, ~, valid_P] = triangulate(matched_p1, matched_p2, M1, M2);
    
    dist_thres = cfgs.max_dist_P3d;
    parfor k = 1:size(P, 1)
        if norm(P(k, :)) > dist_thres
            valid_P(k) = false;
        end
    end

    % [3, N] & [2, N]
    P3d_init = P(valid_P, :)';
    kpt_init = matched_p2(valid_P, :)';
    matched_p1 = matched_p1(valid_P, :)';

    if cfgs.plot_init
        figure(1)
        subplot(2,2,1)
        plot3(P(1,:), P(2,:), P(3,:), 'o'); hold on;
        plotCoordinateFrame(eye(3), zeros(3,1), 2);
        text(-0.1,-0.1,-0.1, 'Cam 1', 'fontsize',10, 'FontWeight','bold');
        plotCoordinateFrame(R_C2W, t_C2W, 2);
        text(t_C2W(1)-0.1, t_C2W(2)-0.1, t_C2W(3)-0.1, 'Cam 2', 'fontsize',10, 'FontWeight','bold');
        axis equal; rotate3d on; grid on; view(0,0);
        title('3d point cloud and cameras'); hold off;

        subplot(2,2, 3:4)
        showMatchedFeatures(img0, img1, matched_p1', kpt_init', "montage");
        title("Point Matched");

        pause()
    end
end