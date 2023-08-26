%% Setup dataset
close all;clear;
ds = 1; % 0: KITTI, 1: Malaga, 2: parking
kitti_path = 'datasets/kitti';
malaga_path = 'datasets/malaga-urban';
parking_path = 'datasets/parking';

addpath('keypointDetect/')
addpath('poseEstimate/');
addpath('triangulation/');
addpath('KLT/')

% ground_truth: T_WC real pose of the cam in each frame [num_frames, 8]
if ds == 0
    % need to set kitti_path to folder containing "05" and "poses"
    assert(exist('kitti_path', 'var') ~= 0);
    ground_truth = load([kitti_path '/poses/05.txt']);
    ground_truth = ground_truth(:, [end-8 end]);
    last_frame = 4540;
    K = [7.188560e+02, 0, 6.071928e+02;
        0, 7.188560e+02, 1.852157e+02;
        0, 0, 1];
elseif ds == 1
    % Path containing the many files of Malaga 7.
    assert(exist('malaga_path', 'var') ~= 0);
    images = dir([malaga_path ...
        '/malaga-urban-dataset-extract-07_rectified_800x600_Images']);
    left_images = images(3:2:end);
    last_frame = length(left_images);
    K = [621.18428, 0, 404.0076;
            0, 621.18428, 309.05989;
            0, 0, 1];
elseif ds == 2
    % Path containing images, depths and all...
    assert(exist('parking_path', 'var') ~= 0);
    last_frame = 598;
    ground_truth = load([parking_path '/poses.txt']);
    ground_truth = ground_truth(:, [end-8 end]);
    K = [331.37, 0, 320;
            0, 369.568, 240;
            0,      0,   1];
else
    assert(false);
end

%% Bootstrap
% set bootstrap frames with appropriate baseline
if ds == 0
    bootstrap_frames = [1, 3];
    img0 = imread([kitti_path '/05/image_0/' ...
        sprintf('%06d.png', bootstrap_frames(1))]);
    img1 = imread([kitti_path '/05/image_0/' ...
        sprintf('%06d.png', bootstrap_frames(2))]);
elseif ds == 1
    bootstrap_frames = [1, 3];
    img0 = rgb2gray(imread([malaga_path ...
        '/malaga-urban-dataset-extract-07_rectified_800x600_Images/' ...
        left_images(bootstrap_frames(1)).name]));
    img1 = rgb2gray(imread([malaga_path ...
        '/malaga-urban-dataset-extract-07_rectified_800x600_Images/' ...
        left_images(bootstrap_frames(2)).name]));
elseif ds == 2
    bootstrap_frames = [1, 4];
    img0 = rgb2gray(imread([parking_path ...
        sprintf('/images/img_%05d.png', bootstrap_frames(1))]));
    img1 = rgb2gray(imread([parking_path ...
        sprintf('/images/img_%05d.png', bootstrap_frames(2))]));
else
    assert(false);
end

rng(2);
num_kpts = 800;
[matched_p1, matched_p2, inliers, F_RANSAC] = bootstrapKLT(img0, img1, num_kpts);
E = K' * F_RANSAC * K;
matched_p1 = matched_p1(inliers, :);
invalid_kpt2 = matched_p2((1-inliers) > 0, :);
matched_p2 = matched_p2(inliers, :);
%     matched_p1 = fliplr(img0_keypoints);
%     matched_p2 = fliplr(img1_keypoints);
% figure(2)
% showMatchedFeatures(img0, img1, matched_p1, matched_p2, "montage");
% title("Point Matches After Outliers Are Removed");

% decompose E and check possible pose [3, 3] -> [3,3,2], [3,1]
[Rots, u3] = decomposeEssentialMatrix(E);
homo_matched_p1 = [matched_p1'; ones(1, length(matched_p1))];
homo_matched_p2 = [matched_p2'; ones(1, length(matched_p2))];
% [3,3,2], [3,1], [3, N], [3, N], [3,3], [3,3] -> [3,3], [3,1]
% care about the dir of the R & t
[R_C2W, t_C2W] = disambiguateRelativePose(...
    Rots, u3, homo_matched_p1, homo_matched_p2, K, K);

% triangulate the first landmarks from the bootstrap imgs
M_C2W = [R_C2W, t_C2W];
M_WC2 = [R_C2W', -R_C2W' * t_C2W];
disp("First M_CW");
disp(M_C2W);
% [3,N]
first_P = linearTriangulation(...
    homo_matched_p1, homo_matched_p2,...
    K *eye(3, 4), K *M_C2W);
valid_P = first_P(3, :) > 0;
first_P = first_P(:, valid_P);
matched_p2 = matched_p2(valid_P, :);

% figure(3)
% imshow(img1); hold on;
% plot(matched_p2(:, 1), matched_p2(: ,2), 'gx', 'Linewidth', 2);
% plot(invalid_kpt2(:, 1), invalid_kpt2(: ,2), 'rx', 'Linewidth', 1);hold off;
% pause(0.01);

%% Continuous operation
% setup the initial state S_0
% State - Struct includes: P_i, X_i, C_i, F_i, Tao_i
% P_i [2, n] - kpt in the i-th frame that has correspongding landmark X_i [3, n]
% C_i [2, m] - kpt that doesn't match a landmark
% F_i [2, m]- vaild tracked kpt, each from the fist frame they occur, corresponging to a kpt in C_i
% Tao_i [12, m] - cam pose in the the first frame the tracking kpt occur
[height, width] = size(img1);
prev_img = img1;
P_prev = matched_p2'; X_prev = first_P;
C_prev = invalid_kpt2'; F_prev = invalid_kpt2';
Tao_prev = repmat(reshape(M_C2W, [12, 1]), 1, size(C_prev, 2));
% history array
num_landmarks_hist = zeros(length(range)+1);
poseWC_hist = zeros(12, length(range) + 1);
num_landmarks_hist(1) = size(X_prev, 2);
poseWC_hist(:, 1) = reshape(M_WC2, [12, 1]);


% looping
range = (bootstrap_frames(2) + 1) : last_frame;
waitforbuttonpress
for i = range
    fprintf('\n=========== Processing frame %d ==========\n', i);
    img = loadImage(ds, i);
    
    % use P_i-1 to track P_i by KLT
    kptTracker = vision.PointTracker(...
        'NumPyramidLevels', 5, 'MaxBidirectionalError', 1,...
        'BlockSize', [31, 31], 'MaxIterations', 40);  
    initialize(kptTracker, P_prev', prev_img);
    [new_P, valid_P] = step(kptTracker, img); % [N, 2]
    new_P = new_P';

    % P_i = round(P_i)';
    % keep_x = (P_i(1,:) < 1 | P_i(1,:) > width);
    % keep_P(keep_x) = 0;
    % keep_y = (P_i(2,:) < 1 | P_i(2,:) > height);
    % keep_P(keep_y) = 0;
    
    P_i = new_P(:, valid_P);
    C_KLT = new_P(:, ~valid_P); % candidates [2, c1]
    X_i = X_prev(:, valid_P);
    fprintf('Keypoints success tracked: %d\n', nnz(valid_P));
    
    % use P_i and X_i to estimate cam pose T_CW = [R_CW | t_CW] in the current frame
    % record the P_i and correspongding X_i-1(X_i) that pass the RANSAC
    % for kpt not passing the test, store them in C_t
    P3P_confidence = 0.95;
    P3P_reprojectErr_thres = 10;
    [R_CnewW, t_CW, inlier_mask] = localizationP3P(P_i, X_i, K, P3P_confidence, P3P_reprojectErr_thres);
    M_frame = [R_CnewW, t_CW];

    % optimization of pose from P3P
    pose_error = @(pose) reprojectError(X_i, P_i, K* pose, false);
    options = optimoptions(@lsqnonlin, 'Algorithm', 'levenberg-marquardt',...
        'MaxIter', 30, 'display', 'off');
    M_frame = lsqnonlin(pose_error, double(M_frame), [], [], options);    
    
    disp("New M_CW");
    disp(M_frame);
    M_frame_WC = [R_CnewW', -R_CnewW' *  t_CW];
    % candidates from Localization [2, c2]   C_localize = P_i(:, (1 - inlier_mask) > 0);
    % fprintf('Keypoints failed to track: %d\n', nnz(1 - keep_P));
    fprintf('Reprojection Error of P3P (with Optimization): %d\n', mean(reprojectError(X_i, P_i, K* M_frame, false), 2))
    fprintf('Keypoints failed to localize: %d\n', nnz(1 - inlier_mask));

%     figure(4)
%     showMatchedFeatures(prev_img, img, P_prev(:, inlier_mask)', P_i(:, inlier_mask)', "montage");
%     pause(0.01);
%     figure(5)
%     imshow(img); hold on;
%     plot(P_i(1, inlier_mask), P_i(2, inlier_mask), 'gx', 'Linewidth', 2);




    % use KLT to track kpt in C_i-1, process triangulate check in each tracked kpt in frame i
    candidatekptTracker = vision.PointTracker(...
        'NumPyramidLevels', 5, 'MaxBidirectionalError', 1,...
        'BlockSize', [31, 31], 'MaxIterations', 40);
    initialize(candidatekptTracker, C_prev', prev_img);   
    [C_i, keep_C] = step(candidatekptTracker, img);

    C_i = round(C_i)';
    keep_t = (C_i(1,:) < 1 | C_i(1,:) > width);
    keep_C(keep_t) = 0;
    keep_t = (C_i(2,:) < 1 | C_i(2,:) > height);
    keep_C(keep_t) = 0;
    % discard candidate points that failed to track
    C_i = C_i(:, keep_C);
    F_i = F_prev(:, keep_C);
    Tao_i = Tao_prev(:, keep_C);
    fprintf('Candidate kpts success tracked: %d\n', nnz(keep_C));
    
    % triangulate new points and landmarks
    alpha_thres = 1;
    % [2, p], [3, p], [q]
    [new_P, new_X, keep_candidate] =...
        triangulateTrackingPoints(C_i, M_frame, F_i, Tao_i, K, alpha_thres);
    % re-projection checking
    reproject_err = reprojectError(new_X, new_P, K*M_frame, false);
    reproject_keep = reproject_err < 0.5;
    P_i = [P_i(:, inlier_mask), new_P(:, reproject_keep)];
    X_i = [X_i(:, inlier_mask), new_X(:, reproject_keep)];
%     plot(new_P(1, :), new_P(2, :), 'yx', 'Linewidth', 2);
    fprintf('New 2D-3D pairs triangulated: %d\n', nnz(reproject_keep));
    
    % update the newly occur candidate kpt C_t into C_i and F_i, as well as Tao_i
    % take care about rudundant!!!!!
    num_kpts = 500;
    [~, new_kpt, inliers, ~] = bootstrapKLT(prev_img, img, num_kpts);
%     invalid_new_kpt = new_kpt((1 - inliers) > 0, :)';
    new_kpt = new_kpt(inliers, :)';
    
    new_C = [new_kpt, new_P(:, 1-reproject_keep > 0), C_KLT];
    keep_new_C = true(size(new_C, 2), 1);
    keep_t = (new_C(1,:) < 1 | new_C(1,:) > width);
    keep_new_C(keep_t) = 0;
    keep_t = (new_C(2,:) < 1 | new_C(2,:) > height);
    keep_new_C(keep_t) = 0;
    new_C = new_C(:, keep_new_C);
    num_new_candidate = size(new_C, 2);
    C_i = [C_i(:, keep_candidate), new_C];
    F_i = [F_i(:, keep_candidate), new_C];
    frame_pose = reshape(M_frame, [12, 1]);
    Tao_i = [Tao_i(:, keep_candidate), repmat(frame_pose, 1, num_new_candidate)];
    fprintf('Current candidate number: %d\n', size(C_i, 2));
    
%     plot(C_i(1, :), C_i(2, :), 'cx', 'Linewidth', 1);
%     plot(C_KLT(1, :), C_KLT(2, :), 'rx', 'Linewidth', 1.2);
%     plot(C_localize(1, :), C_localize(2, :), 'mx', 'Linewidth', 1.2);
%     hold off;
%     pause(0.01);

    % update state
    prev_img = img;
    P_prev = P_i; X_prev = X_i;
    C_prev = C_i; F_prev = F_i; Tao_prev = Tao_i;

    % save log
    iter = i - bootstrap_frames(2);
    num_landmarks_hist(iter+1) = size(X_i, 2);
    poseWC_hist(:, iter + 1) = reshape(M_frame_WC, [12,1]);
    
    % display 
    figure(6)
    displayTracking(img, iter, P_i, X_i, C_i, num_landmarks_hist, poseWC_hist);
end
