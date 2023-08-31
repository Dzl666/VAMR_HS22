function angle = calculateCandidateAngle(C, T, F, Tao, K)
% C [2 ,c]: C_i, candidate kpts in current frame
% F [2 ,c]: F_i, tracked kpts corresponding to candidate kpts
% T [3, 4]: pose of current frame, Cam to World 
% Tao [12, c]: Tao_i, frame pose corresponding to F_i

    num_candidates = size(F, 1);
    C = K \ [C'; ones(1, num_candidates)]; % 3xc
    F = K \ [F'; ones(1, num_candidates)]; % 3xc
    angle = zeros(num_candidates, 1);

    R_t = T(1:3, 1:3)';
    parfor i = 1:num_candidates
        T_f = reshape(Tao(i, :), [3, 4]);
        R_f = T_f(1:3, 1:3)'; % C2W
        Tro =  R_f' * R_t ;
        
        angle(i) = acos(dot(C(:,i), Tro * F(:,i)) / (norm(C(:,i)) * norm(Tro * F(:,i))));

        % ========== another implementation ==========
        % M_track = reshape(track_T(:, c), [3, 4]);
        % p1 = track_C(:, c);
        % p2 = cur_C(:, c);
        % norm_p1 = K \ [p1; 1];
        % norm_p2 = K \ [p2; 1];
        % norm_p1 = norm_p1 ./ repmat(norm_p1(3, :), 3, 1);
        % norm_p2 = norm_p2 ./ repmat(norm_p2(3, :), 3, 1);
        % dir1 =  M_track(:, 1:3)' * norm_p1;
        % dir2 =  cur_T(:, 1:3)' * norm_p2;
        % alpha = atan2d(norm(cross(dir1, dir2)), dot(dir1, dir2));
    end
end