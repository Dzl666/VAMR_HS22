function [] = displayTracking(...
    cur_Img, iter, P, X, C, num_landmarks_hist, pose_hist)
    % plot img with inliners and outliners
    subplot('Position', [0.05, 0.55, 0.45, 0.3])
    imshow(cur_Img); hold on;
    plot(P(1, :), P(2, :), 'gx', 'Linewidth', 2);
    plot(C(1, :), C(2, :), 'rx', 'Linewidth', 1);
    hold off;
    title(['Current Image ', num2str(iter+3)]);
    
    frame_list = -20:0;
    start_frame = iter - 19;
    if (iter < 20)
        frame_list = -iter:0;
        start_frame = 1;
    end
    pose_list = pose_hist([10 12], :);
    pose_list_20 = pose_list(:, start_frame:iter + 1);
    % plot history landmarks number in past 20 frames + cur frame
    subplot('Position', [0.05, 0.15, 0.2, 0.3])
    plot(frame_list, num_landmarks_hist(start_frame:iter + 1), 'b-', 'Linewidth', 1.5);
    title("Landmarks number (past 20)");
    % plot trajectory
    subplot('Position', [0.3, 0.15, 0.2, 0.3])
    plot(pose_list(1, 1:iter+1), pose_list(2, 1:iter+1), 'bx-', 'Linewidth', 2); 
    title("Full trajectory")
    %
    subplot('Position', [0.55, 0.15, 0.4, 0.7])
    plot(pose_list_20(1, :), pose_list_20(2, :), 'bs', 'Linewidth', 2); hold on;
    plot(X(1, :), X(3, :), 'ko', 'Linewidth', 2); hold off;
    title("Trajectory (past 20) & Landmarks")
    
    % Makes sure that plots refresh.
    pause(0.01);
    
end