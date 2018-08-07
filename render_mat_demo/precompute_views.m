set(0,'DefaultFigureWindowStyle','docked')
%% Yaws to render to recompute
yaws = [0, -22, -40, -55 , -70, -75 ];
%% You can loop over tilt in order to render tilt as well
tilt = 0;
crop_size = 224;
%% Loop over subjects
for m=1:10
    for yaw=yaws
        disp(['> Running yaw ' num2str(yaw) ' subj ' num2str(m)])
        model_name=['data/bm'  sprintf('%02d',m) '_plane.wrl'];
        [depth, rendered, ~, A, R, T] = renderer(crop_size,crop_size,model_name,0,0, 0, -90+tilt, yaw, 0,'xyz');
        sim = adjust_translation(yaw);
        %% Zooming a bit and fixing translation
        A(1,1)=2880;A(2,2)=2880;
        A(1,3)=280+sim.tx; A(2,3)=348;
        %% Now doing the final rendering that we want
        [depth_new, rendered_new, U, A, R, T] = renderer(crop_size, crop_size, model_name,0,0,A,R,T);
        %% Showing
        figure(1); subplot(121), imshow(rendered_new);
        imwrite(rendered_new,['imgs_render/rendered_new_' sprintf('%02d',yaw) '.png'])
        subplot(122), imagesc(depth_new), axis equal off;
        %% Getting landmarks
        landmarks = load(['data/landmarks/landmarks_'  sprintf('%02d',m) '.mat']);
        landmarks = landmarks.landmarks;   
        unprojectPt = reshape(U,size(U,1)*size(U,2),3);
        dist = pdist2(landmarks,unprojectPt);
        [distSort, idxSort] = sort(dist,2);
        %% Finding correspondences
        indbad = [];
        [lay,lax] = ind2sub([size(U,1), size(U,2)], idxSort(:,1));
        xy=[lax,lay];
        %% Creating models
        model3D = makeNew3DModel(A, U,xy, [], indbad);
        strinY = sprintf('%02d',yaw);
        %% In case of side views remove occluded landmarks
        if yaw ~= 0
            [model3D] = transferLandmarks(U,A,landmarks,xy);
            strinY(1)=[];
        end
        %% Keep track of indexes that select only the face part (3D ball around the nose)
        [ model3D ] = selectFaceMask( model3D );
        save(['models10_new/model3D_aug_-' strinY '_' sprintf('%02d',tilt) '_'  sprintf('%02d',m)],'model3D','-v7')
        hold on,
        pause();
        clf;
    end
end
