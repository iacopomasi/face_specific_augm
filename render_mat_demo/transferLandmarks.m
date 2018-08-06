function [nmodel3D] = transferLandmarks(U,A,land,xy)
%% unprojectPT here refers to the new model which you want to transfer the points
unprojectPt = reshape(U,size(U,1)*size(U,2),3);
dist = pdist2(land,unprojectPt);
[distSort, idxSort] = sort(dist,2);

%% Finding the visible landmarks
idx = find(distSort(:,1)<4);
newland = zeros(68,3);
for i=1:size(idx)
    newland(idx(i),:) = unprojectPt(idxSort(idx(i),1),:);
end
indbad = setdiff(1:68,idx);
newland_all=newland;
newland(indbad,:)=[];
clear model3D
%% Building new model
nmodel3D.refU = U;
nmodel3D.outA = A;
nmodel3D.ref_XY = xy;
nmodel3D.ref_XY_all = xy;
nmodel3D.ref_XY(indbad,:)=[];
nmodel3D.render_width = size(U,2);
nmodel3D.render_height = size(U,1);
nmodel3D.sizeU = [size(U,1),size(U,2)];
nmodel3D.indbad = indbad;
nmodel3D.threedee = newland;
nmodel3D.threedee_all = newland_all;
end