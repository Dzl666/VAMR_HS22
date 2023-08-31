function [args] = getConfig()
    %   function to get data structure with all paramters 
    
    ds = 1; % 0: KITTI, 1: Malaga, 2: parking
    bootstrap_frames = [1, 3]; %frames for bootstrap
    
    % feature extraction
    min_harris_q = 1e-5; % harris minimum quality (weakest kept score / strongest score)
    max_corners_boot = 1000; % maximum number of corner detected in bootstrap
    max_corners = 500; % maximum number of corner detected
    
    % KLT tracking
    max_KLT_bidir_err = 1; % maximum bidirectional error KLT tracker
    KLT_pyrmid_level = 5;
    min_KLT_displm = 3; % drop too close pair
    
    % RANSAC
    max_ransac_iters = 6000;
    ransac_conf = 99.99;
    max_ransac_dist_err = 0.3;
    max_ransac_reproj_err = 2;

    % keypoints stream
    min_track_kpts = 100; % minimum number of keypoints, when below reinitialize
    max_track_cands = 600; % maximum number of candidates to track
    min_track_displm = 3; % if below, treat new feature as old keypoint/candidate (discard)
    
    % Triangulation
    max_dist_P3d = 100; % filter out 3d points that are too far from the cam
    max_reproj_err = 3; % max reprojection error
    min_cons_frames = 4; % min consecutive frames between Fi_j & Ci_j that can triangulate
    min_triangulate_angle = 10 * pi / 180; % min angle for triangulation
    
    % Debug and ploting
    plot_init = false; % plot initial pointcloud
    plot_kpts_cands = true; % plot image with keypoints (green), candidates (red) and new candidates (magenta)
    plot_cam_pose = true; % plot poses of previous and current frame cameras
    plot_num_stat = false; % plot number keypoints and candidates
    plot_final_path = true; % plot complete path
    
    args = struct(...
        "ds", ds, "bootstrap_frames", bootstrap_frames, ...
        "min_harris_q", min_harris_q, "max_corners_boot", max_corners_boot, "max_corners", max_corners, ...
        "max_KLT_bidir_err", max_KLT_bidir_err, "KLT_pyrmid_level", KLT_pyrmid_level, "min_KLT_displm", min_KLT_displm, ...
        "max_ransac_iters", max_ransac_iters, "ransac_conf", ransac_conf,...
        "max_ransac_dist_err", max_ransac_dist_err, "max_ransac_reproj_err", max_ransac_reproj_err,...
        "min_track_kpts", min_track_kpts, "max_track_cands", max_track_cands, "min_track_displm", min_track_displm, ...
        "max_dist_P3d", max_dist_P3d, "max_reproj_err", max_reproj_err, ...
        "min_cons_frames", min_cons_frames, "min_triangulate_angle", min_triangulate_angle, ...
        "plot_init", plot_init, "plot_num_stat", plot_num_stat, "plot_final_path", plot_final_path,...
        "plot_kpts_cands", plot_kpts_cands, "plot_cam_pose", plot_cam_pose);
    end
    