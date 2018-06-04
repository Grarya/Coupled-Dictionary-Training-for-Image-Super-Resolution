function [HP, LP] = sample_patches(im, patch_size, patch_num_img, upscale)

%Generate a grayscale image if needed
if size(im, 3) == 3,
    hIm = rgb2gray(im);
else
    hIm = im;
end
[nrow, ncol] = size(hIm);

%Generate low resolution counter parts
lIm = imresize(hIm, 1/upscale); %downsample per 2
lIm = imresize(lIm, size(hIm)); %resize to the original image size using bicubic

%Random patches
x = randperm(nrow-patch_size-1);
y = randperm(ncol-patch_size-1);

[X,Y] = meshgrid(x,y);

xrow = X(:);
ycol = Y(:);

if patch_num_img < length(xrow),
    xrow = xrow(1:patch_num_img);
    ycol = ycol(1:patch_num_img);
end

patch_num_img = length(xrow);

%Cast hIm and lIm to double
hIm = double(hIm);
lIm = double(lIm);

H = zeros(patch_size^2,  length(xrow)); %For the patch
L = zeros(4*patch_size^2, length(xrow));%For the gradients
 
%Filters to extract the derivatives: compute first and second order gradients
f1 = [-1,0,1]; %f1
f2 = [-1,0,1]';%f2

lImG11 = conv2(lIm, f1,'same');
lImG12 = conv2(lIm, f2,'same');
 
f3 = [1,0,-2,0,1]; %f3
f4 = [1,0,-2,0,1]'; %f4
 
lImG13 = conv2(lIm,f3,'same');
lImG14 = conv2(lIm,f4,'same');

for ii = 1:patch_num_img,    
   
    row = xrow(ii);
    col = ycol(ii);
    
    %Get patch from the image
    Hpatch = hIm(row:row+patch_size-1,col:col+patch_size-1);
    
    %Get corresponding derivatives
    Lpatch1 = lImG11(row:row+patch_size-1,col:col+patch_size-1);
    Lpatch2 = lImG12(row:row+patch_size-1,col:col+patch_size-1);
    Lpatch3 = lImG13(row:row+patch_size-1,col:col+patch_size-1);
    Lpatch4 = lImG14(row:row+patch_size-1,col:col+patch_size-1);
     
    %Concatenating the four feature vectors into one vector: feature vector
    Lpatch = [Lpatch1(:),Lpatch2(:),Lpatch3(:),Lpatch4(:)];
    Lpatch = Lpatch(:);
    
    %Substract the mean to every patch
    HP(:,ii) = Hpatch(:)-mean(Hpatch(:));
    
    %Compile the feature vector for every image
    LP(:,ii) = Lpatch;
end
