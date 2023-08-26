% keypoints: 2 * n, landmarks: 3 * n
% best_inlier_mask 1 * n, if the match is an outlier, 1 otherwise.
function [R_CW, t_CW, best_inmask] ...
    = localizationP3P(matched_keypoints, landmarks, K, confidence, reprojectErr_thres)
%     matched_keypoints = flipud(matched_keypoints);
    
    if_adaptive = true;
    % num_iters = 200;
    max_iters = 10000;
    num_sample = 3;
    if(if_adaptive)
        num_iters = inf;
    end
    
    best_inliners = 0;
    min_inliners = 6;

    num_match = size(matched_keypoints, 2);
    best_M = eye(3, 4);
    best_inmask = zeros(num_match, 1);
    
    iter = 1;
    while(num_iters > iter)
        % sample from data
        [sample_P, idx] = datasample(landmarks, num_sample, 2, 'Replace', false);
        sample_p = matched_keypoints(:, idx); % [2, k]
        
        % solve 2D-3D and validate
        % normalize
        p_norm = K \ [sample_p; ones(1, num_sample)]; % [3, k]
        for point = 1:3
            p_norm(:, point) = p_norm(:, point) / norm(p_norm(:, point), 2);
        end

        M_multi = p3p(sample_P, p_norm);
        inliners = false(num_match, 1);
        num_in_solu = -1;
        for solu = 1:4
            R_WC = real(M_multi(:, solu*4 - 2 : solu*4));
            t_WC = real(M_multi(:, solu*4 - 3));
            M_CW = [R_WC', -R_WC' * t_WC];
            % check all landmarks
            p_reproj = reprojectPoints(landmarks, K* M_CW);
            err = sum((p_reproj - matched_keypoints).^2, 1);
            inliners_solu = (err <= reprojectErr_thres ^ 2);
            if(nnz(inliners_solu) > num_in_solu)
                inliners = inliners_solu;
                num_in_solu = nnz(inliners);
                best_M_solu = M_CW;
            end
        end

        % update best inliners
        num_inliners = nnz(inliners);
        if(num_inliners > min_inliners && num_inliners > best_inliners)
            best_inliners = num_inliners;
            best_inmask = inliners;
            best_M = best_M_solu;
        end
        
        % update iterations
        if(if_adaptive)
            outlier_ratio = 1 - best_inliners / num_match;
            % set a upper bound
            outlier_ratio = min(0.9, outlier_ratio);
            num_iters = log(1-confidence) / log(1 - (1-outlier_ratio)^num_sample);
            num_iters = min(max_iters, num_iters);
        end
        iter = iter + 1;
    end
    
    assert(best_inliners > 0, 'Localization failed.')
    R_CW = best_M(:, 1:3);
    t_CW = best_M(:, 4);

end
