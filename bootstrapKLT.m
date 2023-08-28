function [R_C2W, t_C2W, kpt_init, P3d_init] = bootstrapKLT(img0, img1, cfgs, camParams)

    %% Feature tracking
    % find kpts from both imgs using Harris detector
    harris_features = detectHarrisFeatures(img0, 'MinQuality', cfgs.min_harris_q);
    img0_corners = selectStrongest(harris_features, cfgs.max_corners_boot).Location;

    kptTracker = vision.PointTracker('MaxBidirectionalError', cfg.max_track_bidir_error);
    
    % use [N, 2] (x, y) keypoints
    initialize(kptTracker, img0_corners, img0);
    [kpt1_temp, valid1_temp] = kptTracker(img1);
    kpt_temp = kpt_temp(valid_temp, :);
    release(kptTracker);

    % matched kpts in shape [2, N]
    matched_kpt0 = img0_corners(valid1_temp, :)';
    matched_kpt1 = kpt1_temp(valid1_temp, :)';
    if cfgs.ds == 2
        matched_kpt0 = round(matched_kpt0);
        matched_kpt1 = round(matched_kpt1);

    % delete matches with small displacement
    dists = vecnorm(matched_kpt0 - matched_kpt1);
    thres = cfgs.min_KLT_displm
    matched_kpt0 = matched_kpt0(:, dists > thres)';
    matched_kpt1 = matched_kpt1(:, dists > thres)';

    %% Solving relative pose

    % estimate F using lib func (with RANSAC)
    thres_estFmat = 1e-2;
    [F_RANSAC, inliers_kpt] = estimateFundamentalMatrix(matched_kpt0, matched_kpt1,...
        'Method','RANSAC', 'NumTrials', cfgs.max_ransac_iters,...
        'DistanceThreshold', cgfs.max_ransac_err, 'Confidence', cfgs.ransac_conf);
        
    K = camParams.IntrinsicMatrix';
    E = K' * F_RANSAC * K;
    % matched kpts after RANSAC [N, 2]
    matched_p1 = matched_kpt0(inliers, :);
    matched_p2 = matched_kpt1(inliers, :);
    
    % get relative pose of 2 cams using lib func
    % pose of cam2 related to cam 1
    [R_C2W, t_C2W] = relativeCameraPose(E, camParams, matched_p1,matched_p2);
    
    % triangulate the first landmarks from the bootstrap imgs
    M1 = cameraMatrix(camParams, eye(3), [0,0,0]);
    M2 = cameraMatrix(camParams, R_C2W', -R_C2W' * t_C2W');
 
    [P, ~, valid_P] = triangulate(matched_p1, matched_p2, M1, M2);
    % ========== self-implementation ==========
    % P = linearTriangulation(homo_matched_p1, homo_matched_p2, M1, M2);
    % valid_P = P(3, :) > 0;
    
    dist_thres = cfgs.max_dist_P3d;
    parfor k = 1:size(P, 1)
        if norm(P(k, :)) > dist_thres
            valid_P(k) = false;
        end
    end

    % [3, N] & [2, N]
    P3d_init = P(valid_P, :)';
    kpt_init = matched_p2(valid_P, :)';

    if cfgs.plot_init
        figure(1)
        subplot(2,2,1)
        plot3(P(1,:), P(2,:), P(3,:), 'o');
        plotCoordinateFrame(eye(3),zeros(3,1), 0.8);
        text(-0.1,-0.1,-0.1,'Cam 1','fontsize',10,'color','k','FontWeight','bold');
        plotCoordinateFrame(R_C2W, t_C2W', 0.8);
        text(t_C2W(1)-0.1, t_C2W(2)-0.1, t_C2W(3)-0.1,'Cam 2','fontsize',10,'color','k','FontWeight','bold');
        axis equal
        rotate3d on;
        grid
        title('3d point cloud and cameras')

        subplot(2,2, 3:4)
        showMatchedFeatures(img0, img1, matched_p1, matched_p2, "montage");
        title("Point Matched");

        pause()
    end
end