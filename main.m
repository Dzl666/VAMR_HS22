close all; clear;

%% Setup
% addpath("initialization");
% addpath("keypointDetect/");
% addpath("poseEstimate/");
% addpath("triangulation/");
% addpath("KLT/");

cfgs = getConfig();

%% Load dataset
ds = cfgs.ds; % 0: KITTI, 1: Malaga, 2: parking

% ground_truth: T_WC real pose of the cam in each frame [num_frames, 8]
if ds == 0
    % need to set kitti_path to folder containing "05" and "poses"
    kitti_path = 'datasets/kitti';
    assert(exist('kitti_path', 'var') ~= 0);
    last_frame = 300; %4540
    K = [7.188560e+02, 0, 6.071928e+02;
        0, 7.188560e+02, 1.852157e+02;
        0, 0, 1];
    ground_truth = load([kitti_path '/poses/05.txt']);
    ground_truth = ground_truth(:, [end-8 end]);
elseif ds == 1
    % Path containing the many files of Malaga 7.
    malaga_path = 'datasets/malaga-urban';
    assert(exist('malaga_path', 'var') ~= 0);
    images = dir([malaga_path '/malaga-urban-dataset-extract-07_rectified_800x600_Images']);
    left_images = images(3:2:end);
    last_frame = 300; %length(left_images);
    K = [621.18428, 0, 404.0076;
        0, 621.18428, 309.05989;
        0, 0, 1];
elseif ds == 2
    % Path containing images, depths and all...
    parking_path = 'datasets/parking';
    assert(exist('parking_path', 'var') ~= 0);
    last_frame = 598;
    K = [331.37, 0, 320;
        0, 369.568, 240;
        0,      0,   1];
    ground_truth = load([parking_path '/poses.txt']);
    ground_truth = ground_truth(:, [end-8 end]);
else
    assert(false);
end

%% Bootstrap
% set bootstrap frames with appropriate baseline
bootstrap_frames = cfgs.bootstrap_frames;

if ds == 0
    img0 = imread([kitti_path '/05/image_0/' sprintf('%06d.png', bootstrap_frames(1))]);
    img1 = imread([kitti_path '/05/image_0/' sprintf('%06d.png', bootstrap_frames(2))]);
elseif ds == 1
    img0 = rgb2gray(imread([malaga_path '/malaga-urban-dataset-extract-07_rectified_800x600_Images/' ...
        left_images(bootstrap_frames(1)).name]));
    img1 = rgb2gray(imread([malaga_path '/malaga-urban-dataset-extract-07_rectified_800x600_Images/' ...
        left_images(bootstrap_frames(2)).name]));
elseif ds == 2
    img0 = rgb2gray(imread([parking_path sprintf('/images/img_%05d.png', bootstrap_frames(1))]));
    img1 = rgb2gray(imread([parking_path sprintf('/images/img_%05d.png', bootstrap_frames(2))]));
else
    assert(false);
end

rng(2);

camParams = cameraParameters("IntrinsicMatrix", K');
[R_C2W, t_C2W, kpt_init, P3d_init] = bootstrapKLT(img0, img1, camParams, cfgs);

%% Continuous operation 
range = (bootstrap_frames(end) + 1) : last_frame;

% setup the initial state S_0
% State - Struct includes: P_i, X_i, C_i, F_i, Tao_i, C_cnt_i
prev_img = img1;
P_t = kpt_init;
X_t = P3d_init;
C_t = []; F_t = [];
Tao_t = []; C_cnt_t = [];

% history array
% num_X_hist = zeros(last_frame);
% num_X_hist(bootstrap_frames(end)) = size(X_prev, 2);
num_kpt = 0; num_cand = 0;

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
    if size(P_t, 2) < cfgs.min_track_kpts
        img0 = loadImage(ds, iter-2); % prev_prev_img
        [R, T, kpt_init, P3d_init] = bootstrapKLT(img0, prev_img, camParams, cfgs);
        P_t = [P_t, kpt_init];
        % conver the reboot P3d from prev_prev_img coord. to world coord. R_CW * P + (-R_WC' * t_WC)
        cam_pos_prev_prev = -poseW2C_hist{1, iter-2}' * poseW2C_hist{2, iter-2};
        P3d_W = poseW2C_hist{1, iter-2}' * P3d_init + cam_pos_prev_prev;
        X_t = [X_t, P3d_W];
        
        % optimize the P3ds according to previous pose
        u = P_t(1, :);
        v = P_t(2, :);
        kpt_array = [pointTrack(1, [u(1), v(1)])];
        parfor k = 2 : size(P_t, 2)
            kpt_array(k) = pointTrack(1, [u(k), v(k)]);
        end
        ViewId = uint32(1);
        R_C2W_prev = poseW2C_hist{1, iter-1}';
        t_C2W_prev = -R_C2W_prev  * poseW2C_hist{2, iter-1};
        AbsolutePose = rigid3d(R_C2W_prev, t_C2W_prev');
        tab = table(ViewId,AbsolutePose);
        X_t = bundleAdjustmentStructure(X_t', kpt_array, tab, camParams,  'PointsUndistorted', true)';
    end


    %% tracking & match & pose estimation & triangulation
    [P_t, X_t, C_t, F_t, Tao_t, C_cnt_t, T_W2C] = continuousTracking(prev_img, img, camParams, cfgs,...
        P_t, X_t, C_t, F_t, Tao_t, C_cnt_t, poseW2C_hist{1,iter-1}, poseW2C_hist{2,iter-1});
    
    prev_img = img;

    max_cands = cfgs.max_track_cands;
    if size(C_t, 2) >= max_cands
        C_t = C_t(:, 1:max_cands);
        F_t = F_t(:, 1:max_cands);
        Tao_t = Tao_t(:, 1:max_cands);
        C_cnt_t = C_cnt_t(1:max_cands);
    end
    
    %% update pose history and plot together with trajetory
    poseW2C_hist{1, iter} = T_W2C(1:3, 1:3);
    poseW2C_hist{2, iter} = T_W2C(1:3, 4);
    
    % display
    figure(3)
    subplot(2,3,5);
    axis equal; view(0,0); hold on;
    
    traj = [-poseW2C_hist{1,iter}' * poseW2C_hist{2,iter}, ...
            -poseW2C_hist{1,iter-1}' * poseW2C_hist{2,iter-1}]';
    plot3(traj(1,1), traj(1,2), traj(1,3), 'r*');
    plot3(traj(:,1), traj(:,2), traj(:,3), '-b'); 

%     if exist('h', 'var') == 1
%         set(h,'Visible','off');
%     end
%     h = plot3(X_prev(1,:), X_prev(2,:), X_prev(3,:), 'ko');

    xlim([traj(1,1)-15, traj(1,1)+15])
    ylim([traj(1,2)-15, traj(1,2)+15])
    zlim([traj(1,3)-15, traj(1,3)+15])
    title("Trajectory and 3D landmarks");

    if cfgs.plot_num_stat
        subplot(2,3,6)
        hold on;
        kpts_line = [num_kpt, size(P_t,2)];
        cands_line = [num_cand, size(C_t,2)];
        plot([iter-1, iter], kpts_line, 'b-');
        plot([iter-1, iter], cands_line, 'r-');
        xlim([iter-20, iter+1])
        ylim([0, 1900])
        num_kpt = size(P_t,2);
        num_cand = size(C_t,2);
        legend('kpts num', 'cands num', 'Location', 'northeast');
        % title("keypoints (blue), candidates (red)");
    end
    % displayTracking(img, iter, P_i, X_i, C_i, poseW2C_hist);

    pause(0.1);
end

%% plot final results
if cfgs.plot_final_path
    figure(4)
    x = zeros(iter, 1);
    y = x;
    z = x;
    for k = [1, bootstrap_frames(end):iter]
        cam_center = -poseW2C_hist{1,k}' * poseW2C_hist{2,k};
        x(k) = cam_center(1);
        y(k) = cam_center(2);
        z(k) = cam_center(3);
    end
    plot3(x,y,z, 'r-'); hold on;
    plot3(x,y,z, 's');
    if ds == 0
        xlim([-100,200])
        zlim([-60,200])
    end
    view(0,0); axis equal;
    title("final trajectory");
end


