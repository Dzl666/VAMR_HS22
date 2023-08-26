function P = linearTriangulation(p1, p2, KM1, KM2)
% LINEARTRIANGULATION  Linear Triangulation
%
% Input:
%  - p1, p2 (3,N): homogeneous coordinates of points in image 1, 2
%  - M1, M2 (3,4): proj matrix corresponding to image 1, 2, from C to W
% Output:
%  - P(3,N): homogeneous coordinates of 3-D points

    % Sanity checks
    [dim,nPoints] = size(p1);
    [dim2,nPoints2] = size(p2);
    assert(dim==dim2,'Size mismatch of input points');
    assert(nPoints == nPoints2, 'Size mismatch of input points');
    assert(dim==3,'p1, p2 should be 3xN matrices (homogeneous coords)');

    [rows,cols] = size(KM1);
    assert(rows==3 && cols==4,'Projection matrices should be of size 3x4');
    [rows,cols] = size(KM2);
    assert(rows==3 && cols==4,'Projection matrices should be of size 3x4');

    P = zeros(4, nPoints);
    % DLT
    for j = 1:nPoints
        % Built matrix of linear homogeneous system of equations
        A1 = cross2Matrix(p1(:,j)) * KM1;
        A2 = cross2Matrix(p2(:,j)) * KM2;
        % Solve the linear homogeneous system of equations
        [~,~,v] = svd([A1; A2], 0);
        P(:,j) = v(:,4);
    end
    % Dehomogeneize
    P = P(1:3, :) ./ repmat(P(4,:), 3,1);
    
    p1 = p1(1:2, :);
    p2 = p2(1:2, :);
    error_terms = @(P_3d) ...
        reprojectError(P_3d, p1, KM1, true) + reprojectError(P_3d, p2, KM2, true);
    options = optimoptions(@lsqnonlin,...
        'Algorithm', 'levenberg-marquardt', 'MaxIter', 30, 'display','off');
    P = lsqnonlin(error_terms, double(P), [], [], options);
end


