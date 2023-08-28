close all; clear;

%% Setup
addpath("initialization");
addpath("keypointDetect/");
addpath("poseEstimate/");
addpath("triangulation/");
addpath("KLT/");

cfgs = getConfig();

%% Load dataset
ds = cfgs.ds; % 0: KITTI, 1: Malaga, 2: parking

% ground_truth: T_WC real pose of the cam in each frame [num_frames, 8]
if ds == 0
    % need to set kitti_path to folder containing "05" and "poses"
    kitti_path = 'datasets/kitti';
    assert(exist('kitti_path', 'var') ~= 0);
    ground_truth = load([kitti_path '/poses/05.txt']);
    ground_truth = ground_truth(:, [end-8 end]);
    last_frame = 4540;
    K = [7.188560e+02, 0, 6.071928e+02;
        0, 7.188560e+02, 1.852157e+02;
        0, 0, 1];
elseif ds == 1
    % Path containing the many files of Malaga 7.
    malaga_path = 'datasets/malaga-urban';
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
    parking_path = 'datasets/parking';
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
bootstrap_frames = cfgs.bootstrap_frames;

if ds == 0
    img0 = imread([kitti_path '/05/image_0/' ...
        sprintf('%06d.png', bootstrap_frames(1))]);
    img1 = imread([kitti_path '/05/image_0/' ...
        sprintf('%06d.png', bootstrap_frames(2))]);
elseif ds == 1
    img0 = rgb2gray(imread([malaga_path ...
        '/malaga-urban-dataset-extract-07_rectified_800x600_Images/' ...
        left_images(bootstrap_frames(1)).name]));
    img1 = rgb2gray(imread([malaga_path ...
        '/malaga-urban-dataset-extract-07_rectified_800x600_Images/' ...
        left_images(bootstrap_frames(2)).name]));
elseif ds == 2
    img0 = rgb2gray(imread([parking_path ...
        sprintf('/images/img_%05d.png', bootstrap_frames(1))]));
    img1 = rgb2gray(imread([parking_path ...
        sprintf('/images/img_%05d.png', bootstrap_frames(2))]));
else
    assert(false);
end

rng(2);

camParams = cameraParameters("IntrinsicMatrix", K');
[R_C2W, t_C2W, kpt_init, P3d_init] = bootstrapKLT(img0, img1, cfgs, camParams);

%% Continuous operation
% setup the initial state S_0
% State - Struct includes: P_i, X_i, C_i, F_i, Tao_i
% P_i [2, n] - kpt in the i-th frame that has correspongding landmark X_i [3, n]
% C_i [2, m] - kpt in the i-th frame that doesn't match a landmark
% F_i [2, m]- vaild kpt that still can be tracked from the fist frame 
%               they occur, corresponging to a kpt in C_i
% Tao_i [12, m] - cam pose of first frame that F_i occur
% cand_cnt_i - 
 
[height, width] = size(img1);
range = (bootstrap_frames(end) + 1) : last_frame;

% state
prev_img = img1;
P_prev = kpt_init;
X_prev = P3d_init;
C_prev = []; F_prev = [];
Tao_prev = []; C_cnt_prev = [];

% history array
num_landmarks_hist = zeros(last_frame);
num_landmarks_hist(bootstrap_frames(end)) = size(X_prev, 2);

poseW2C_hist = cell(2, last_frame);
for i = 1:bootstrap_frames(end)-1
    poseW2C_hist{1, i} = eye(3);
    poseW2C_hist{2, i} = [0,0,0]';
end
poseW2C_hist{1, bootstrap_frames(end)} = R_C2W';
poseW2C_hist{2, bootstrap_frames(end)} = -R_C2W' * t_C2W;

% looping
for iter = range
    fprintf('\n=========== Processing frame %d ==========\n', iter);
    img = loadImage(ds, iter);

    % if number of kpts too small, reboot
    if size(P_prev, 2) < cfgs.min_track_kpts
        prev_prev_img = loadImage(ds, iter-2);
        [R, T, kpt_init, P3d_init] = bootstrapKLT(prev_prev_img, prev_img, cfgs, camParams);
        P_prev = [P_prev, kpt_init];
        % conver the reboot P3d from prev_img coord. to world coord. R_CW * P + (-R_WC * t_WC)
        P3d_W = poseW2C_hist{1, iter-2}' * P3d_init - poseW2C_hist{1, iter-2}' * poseW2C_hist{2, iter-2};
        X_prev = [X_prev, P3d_W];
        
        % optimize the kpts & P3ds according to old pose and new pose
        absPose = rigid3d(poseW2C_hist{1, iter-2}', (-poseW2C_hist{1, iter-2}' * poseW2C_hist{2, iter-2})');
        viewId = uint32(1);
        tab = table(viewId, absPose);
        u = P_prev(1, :); v = P_prev(2, :);
        kpt_array = [pointTrack(1, [u(1), v(1)])];
        parfor k = 2 : size(P_prev, 2)
            kpt_array(k) = pointTrack(1, [u(k), v(k)]);
        end
        X_prev = bundleAdjustmentStructure(X_prev', kpt_array, tab, camParams)';
    end


    %% Track all key points
    % use P_i-1 to track P_i by KLT
    kptTracker = vision.PointTracker('MaxBidirectionalError', cfg.max_track_bidir_error);  
    initialize(kptTracker, P_prev', prev_img);
    [P_new, valid_P] = kptTracker(img); % [N, 2]
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
        candTracker = vision.PointTracker('MaxBidirectionalError', cfg.max_track_bidir_error);
        initialize(candTracker, C_prev', prev_img);   
        [C_new, valid_C] = step(candidatekptTracker, img);
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

        valid_add = valid_add > 0;
        if not(isempty(P_add))
            % clean C, F, Tao, C_cnt
            invalid_add = not(valid_add)
            invalid_new_kpt = not(valid_new_kpt)
            C_i = [P_add(invalid_add, :); C_i(invalid_new_kpt, :)];
            F_i = [F_corres(invalid_add, :); F_i(invalid_new_kpt, :)];
            F_i = [Tao_corres(invalid_add, :); Tao_i(invalid_new_kpt, :)];
            C_cnt_i = [cnt_corres(invalid_add, :); C_cnt_i(invalid_new_kpt, :)];
            % clean P_add, X_add
            P_add = P_add(valid_add, :);
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


    %% update state
    P_total = [P_i; P_add];
    X_total = [X_i; X_add];
    C_total = [C_i; C_add];
    F_total = [F_i; C_add];
    Tao_total = [Tao_i; Tao_add];
    C_cnt_total = [C_cnt_i, C_cnt_add];

    prev_img = img;
    P_prev = P_total'; X_prev = X_total';
    C_prev = C_total'; F_prev = F_total';
    Tao_prev = Tao_total'; C_cnt_prev = C_cnt_total;

    if cfgs.plot_kpts_cands
        figure(4)
        subplot(1,2,1)
        imshow(img_i);
        hold on;
        plot(P_next(:,1), P_next(:,2), '.g');
        if not(isempty(C_next))
            plot(C_next(:,1), C_next(:,2), '.r');
        end
        if not(isempty(C_new))
            plot(C_new(:,1), C_new(:,2), '.m')
        end
        
        if not(isempty(P_new))
            plot(P_new(:,1), P_new(:,2), 'cs');
            plot(first_obser_P_new(:,1), first_obser_P_new(:,2), 'bs');
            plot([P_new(:,1)'; first_obser_P_new(:,1)'], [P_new(:,2)'; first_obser_P_new(:,2)'], 'y-', 'Linewidth', 1);
        end
        title("keypoints (green), candidates (red), new candidates (magenta)");
        hold off;
    end

    if cfgs.plot_cam_pose
        figure(4)
        subplot(1,2,2)
        plot3(X_prev_good(:,1), X_prev_good(:,2), X_prev_good(:,3), 'bo');
        hold on
        if not(isempty(X_new))
            plot3(X_new(:,1), X_new(:,2), X_new(:,3), 'ro');
        end
        center_cam2_W = -R_C_W'*t_C_W;
        plot3(center_cam2_W(1), center_cam2_W(2), center_cam2_W(3))
        plotCoordinateFrame(R_prev,-R_prev'*T_prev, 0.8);
        plotCoordinateFrame(R_C_W,center_cam2_W, 0.8);
        text(center_cam2_W(1)-0.1, center_cam2_W(2)-0.1, center_cam2_W(3)-0.1,'Cam','fontsize',10,'color','k','FontWeight','bold');
        axis equal
        rotate3d on;
        grid
        title('Cameras poses')
        view(0,0)
        hold off;
    end

    max_cands = cfgs.max_track_cands
    if size(C_prev, 2) >= max_cands
        C_prev = C_prev(:, :max_cands);
        F_prev = F_prev(:, :max_cands);
        Tao_prev = Tao_prev(:, :max_cands);
        C_cnt_prev = C_cnt_prev(:max_cands);
    end
    
    %% update log and plot together with trajetory
    % save log
    num_landmarks_hist(iter) = size(X_prev, 2);
    poseW2C_hist{1, iter} = R_i_W2C;
    poseW2C_hist{2, iter} = t_i_W2C;
    
    % display
    figure(5)
    displayTracking(img, iter, P_i, X_i, C_i, num_landmarks_hist, poseW2C_hist);
end

if cfgs.plot_final_path
    figure(6)
    x = zeros(i,1);
    y = x;
    z = x;
    for k = [1, bootstrap_frames(end):i]
        cam_center = - T_i_wc_history{1,k}'* T_i_wc_history{2,k};
        x(k) = cam_center(1);
        y(k) = cam_center(2);
        z(k) = cam_center(3);
    end
    plot3(x,y,z, 'r-');
    hold on
    plot3(x,y,z, 's');
    if ds == 0
        xlim([-100,200])
        zlim([-60,200])
    end
    view(0,0)
    axis equal
    title("trajectory");
end


