function [im_h_y] = backprojection(im_h_y, im_l_y, maxIter)

[row_l, col_l] = size(im_l_y);
[row_h, col_h] = size(im_h_y);

p = fspecial('gaussian', 5, 1); %create a filter size 5 and sigma 1
p = p.^2;
p = p./sum(p(:));

im_l_y = double(im_l_y);
im_h_y = double(im_h_y);

for ii = 1:maxIter
    im_l_s = imresize(im_h_y, [row_l, col_l], 'bicubic');
    im_diff = im_l_y - im_l_s;
    
    im_diff = imresize(im_diff, [row_h, col_h], 'bicubic');
    im_h_y = im_h_y + conv2(im_diff, p, 'same');
end
end
    