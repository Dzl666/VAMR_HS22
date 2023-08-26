function [matched_p1, matched_p2, inliers_kpt, F_RANSAC] = bootstrapKLT(img0, img1, num_kpts)
    % find kpts from both imgs using Harris detector
%     harris_patch_size = 9;
%     harris_kappa = 0.08;
%     NMS_radius = 8;
%     img0_harris = harris(img0, harris_patch_size, harris_kappa);
%     % [2, N] in (x, y)
%     img0_keypoints = selectKeypoints(img0_harris, num_kpts, NMS_radius);
%     img0_keypoints = img0_keypoints';

    % Malaga 1e-2, 5 
    % KITTI  1e-2, 5
    harris_features = detectHarrisFeatures(img0, 'MinQuality', 1e-2, 'FilterSize', 5);
    img0_keypoints = harris_features.selectStrongest(num_kpts).Location;

    kptTracker = vision.PointTracker( ...
        'NumPyramidLevels', 5, 'MaxBidirectionalError', 1, ...
        'BlockSize', [31, 31], 'MaxIterations', 40);
    % use [N, 2] (x, y) keypoints
    initialize(kptTracker, img0_keypoints, img0);
    [kpt, valid] = step(kptTracker, img1);
    matched_p1 = img0_keypoints(valid, :);
    matched_p2 = kpt(valid, :);

%     figure(1)
%     showMatchedFeatures(img0, img1, matched_p1, matched_p2, "montage");
%     title("Point Matches");

    % estimate F using lib function (with RANSAC)
    thres_estFmat = 1e-2;
    [F_RANSAC, inliers_kpt] = estimateFundamentalMatrix(...
        matched_p1, matched_p2,...
        'Method','RANSAC', 'NumTrials',500,...
        'DistanceThreshold',thres_estFmat,...
        'Confidence', 99);
end