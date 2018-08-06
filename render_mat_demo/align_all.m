function [fim, tform, basePts] = align_all(img,fidu, prof)
%% Frontal
if ~prof
    basePts = [  73.5122  100.8849  130.9949  160.6487  119.1334   89.0234  144.2251 117.1334;
                 108.4308 , 108.4308 , 108.4308 , 108.4308 , 128.1212 , 162.2637 , 162.7200 188.1212 ];
    enabled = [37,40,43,46,34,60,56,9];   
%% Profile
else
    basePts = [    130.6487  109.1334  114.2251 112.1334;
                94.1497-10 , 128.1212 , 152.7200 188.1212 ];
    basePts(2,:)= basePts(2,:)+15;
    basePts(1,:)= basePts(1,:)-15;
    enabled = [46,36,56,9];
end
basePts(2,:)= basePts(2,:)-10;
x0 = 1;
y0 = 1;
x1 = 224;
y1 = 224;
basePts = basePts';
fidu = fidu(enabled,:);
tform = cp2tform(fidu, basePts, 'similarity');
fim = imtransform(img, tform, 'bicubic', 'XData', [x0 x1], 'YData', [y0 y1], 'XYScale', 1);
end
