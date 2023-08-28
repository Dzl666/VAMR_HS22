function [args] = getConfig()
    %   function to get data structure with all paramters 
    
    ds = 2; % 0: KITTI, 1: Malaga, 2: parking
    bootstrap_frames = [1, 3]; %frames for bootstrap
    
    % feature extraction
    min_harris_q = 1e-5; % harris minimum quality (weakest kept score / strongest score)
    max_detect_corners_boot = 800; % maximum number of corner detected
    max_detect_corners = 500; % maximum number of corner detected
    
    % KLT tracking
    max_KLT_bidir_err = 1; % maximum bidirectional error KLT tracker
    min_KLT_displm = 3; % if below, treat new feature as old keypoint/candidate (discard)
    
    % RANSAC
    max_ransac_iters = 8000;
    ransac_conf = 99.999;
    max_ransac_err = 0.2;

    % keypoints stream
    min_track_kpts = 200; % minimum number of keypoints, when below reinitialize
    max_track_cands = 500; % maximum number of candidates to track
    min_track_displm = 3; 
    
    % Triangulation
    max_dist_P3d = 100; % filter out 3d points that are too far from the cam
    min_cons_frames = 8; % minimum consecutive frames
    max_reproj_err = 2; % max reprojection error
    min_triangulate_angle = 10 * pi / 180; % min angle for triangulation
    
    % Debug and ploting
    plot_init = false; % plot initial pointcloud
    plot_kpts_cands = true; % plot image with keypoints (green), candidates (red) and new candidates (magenta)
    plot_cam_pose = true; % plot poses of previous and current frame cameras
    plot_traj = true; % plot local trajectory
    plot_pcd = true; % plot current pointcloud on trajectory
    plot_num_stat = true; % plot number keypoints and candidates
    plot_final_path = true; % plot complete path
    
    configparams = struct(...
        "ds", ds, "bootstrap_frames", bootstrap_frames, ...
        "min_harris_q", min_harris_q, "max_corners_boot", max_corners_boot, "max_corners", max_corners, ...
        "max_KLT_bidir_err", max_KLT_bidir_err, "min_KLT_displm", min_KLT_displm, ...
        "max_ransac_iters", max_ransac_iters, "ransac_conf", ransac_conf, "max_ransac_err", max_ransac_err, ...
        "min_track_kpts", min_track_kpts, "max_track_cands", max_track_cands, "min_track_displm", min_track_displm, ...
        "max_dist_P3d", max_dist_P3d, "min_cons_frames", min_cons_frames, ...
        "max_reproj_err", max_reproj_err, "min_triangulate_angle", min_triangulate_angle, ...
        "plot_init", plot_init, "plot_kpts_cands", plot_kpts_cands, "plot_cam_pose", plot_cam_pose, ...
        "plot_traj", plot_traj, "plot_pcd", plot_pcd, "plot_num_stat", plot_num_stat, ...
        "plot_final_path", plot_final_path);
    end
    