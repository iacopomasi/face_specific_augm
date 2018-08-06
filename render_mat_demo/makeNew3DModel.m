function model3D = makeNew3DModel(A, U, xy, threedee, indbad)
% Produces a new struct with information mapping 2D-3D facial feature
% points for face pose estimation. This is part of the distribution for
% face image frontalization ("frontalization" software). described in [1].
%
% If you find this code useful and use it in your own work, please add
% reference to [1].
%
% Please see project page for more details:
% http://www.openu.ac.il/home/hassner/projects/frontalize
% 
% This function is used whenever a new facial feature detector is employed. 
%
% Example usage:
%
% >> load model3DSDM Model3D % use existing struct as basis; any previous detector will do
% >> imref = imread('reference_320_320.png');
% >> %...% run new facial feature detector on reference face imref
% >> %...% assuming detected features returned in an n x 2 matrix of coordinates ref_XY
% >> model3D = makeNew3DModel(Model3D.outA, Model3D.refU, ref_XY);
% >> save model3DNewDetector Model3D
%
%   Copyright 2014, Tal Hassner
%   http://www.openu.ac.il/home/hassner/projects/frontalize
%
%  References:
%   [1] Tal Hassner, Shai Harel, Eran Paz, Roee Enbar, "Effective Face
%   Frontalization in Unconstrained Images," forthcoming. 
%   See project page for more details: 
%   http://www.openu.ac.il/home/hassner/projects/frontalize

%   The SOFTWARE ("frontalization" and all included files) is provided "as is", without any
%   guarantee made as to its suitability or fitness for any particular use.
%   It may contain bugs, so use of this tool is at your own risk.
%   We take no responsibility for any damage that may unintentionally be caused
%   through its use.
%
%   ver 1.3, 8-Dec-2015
%

model3D.refU = U;
model3D.outA = A;
model3D.ref_XY = xy;
model3D.render_width = size(U,2);
model3D.render_height = size(U,1);
model3D.sizeU = [size(U,1),size(U,2)];

ind = sub2ind([size(U,1), size(U,2)], round(xy(:,2)), round(xy(:,1)));
threedee = zeros(numel(ind),3);
tmp = U(:,:,1);
threedee(:,1) = tmp(ind);
tmp = U(:,:,2);
threedee(:,2) = tmp(ind);
tmp = U(:,:,3);
threedee(:,3) = tmp(ind);

model3D.indbad = indbad;
model3D.threedee = threedee;








