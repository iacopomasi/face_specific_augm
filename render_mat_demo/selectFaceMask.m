function [ model3D ] = selectFaceMask( model3D )
refU = model3D.refU;
land = model3D.threedee;
indba = model3D.indbad;
X = refU(:,:,1);
Y = refU(:,:,3);
Z = refU(:,:,2);
%% Full landmarks
newland = zeros(68,3);
newland(setdiff(1:68,indba),:)=land;
%% picking the nose
nose = newland(31,:); %% 75
model = [X(:) Z(:) Y(:);];
dist = pdist2(model,nose);
idxInside = find(dist<120);
%plot3(model(idxInside,1),model(idxInside,3),model(idxInside,2),'r.')
model3D.facemask = idxInside;
end

