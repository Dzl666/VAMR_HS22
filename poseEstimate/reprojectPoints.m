function p_reproj = reprojectPoints(P, KM)
% Reproject 3D points given a projection matrix
% ==== Parameter ====
% P: [3, n] coordinates of the 3d points in the world frame
% M_tilde: [3x4] projection matrix
% K: [3x3] camera matrix
% ==== Return ====
% p_reproj: [2, n] coordinates of the reprojected 2d points
    p_homo = KM * [P; ones(1, size(P, 2))];
    p_reproj = p_homo(1:2, :) ./ repmat(p_homo(3, :), 2, 1);
end

