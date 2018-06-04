% =========================================================================
% % Reference
%   J. Yang et al. Image super-resolution as sparse representation of raw
%   image patches. CVPR 2008.
%   J. Yang et al. Image super-resolution via sparse representation. IEEE 
%   Transactions on Image Processing, Vol 19, Issue 11, pp2861-2873, 2010
%   J. Yang et al. Coupled Dictionary Training for Image Super-Resolution. 
%   Transactions on Image Processing, Vol 21, Issue 08, pp3467-3478, 2012
% Jianchao Yang
% ECE Department, University of Illinois at Urbana-Champaign
% For any questions, send email to jyang29@uiuc.edu
% =========================================================================

clear all; clc;

start_time= cputime;
% read test image
im_l = imread('Data/Testing/input.bmp');

% set parameters
lambda = 0.2;                   % sparsity regularization
overlap = 4;                    % the more overlap the better (patch size 5x5)
up_scale = 2;                   % scaling factor, depending on the trained dictionary
maxIter = 20;                   % if 0, do not use backprojection

% load dictionary
load('Dictionary/CD_512_0.15_5.mat');

% change color space, work on illuminance only
im_l_ycbcr = rgb2ycbcr(im_l);
% get the components Y, Cb, Cr
im_l_y = im_l_ycbcr(:, :, 1);
im_l_cb = im_l_ycbcr(:, :, 2);
im_l_cr = im_l_ycbcr(:, :, 3);

% image super-resolution based on sparse representation
[im_h_y] = ScSR(im_l_y, up_scale, Dh, Dl, lambda, overlap);
[im_h_y] = backprojection(im_h_y, im_l_y, maxIter); %Use gaussian filter

% upscale the chrominance(Cb and Cr) simply by "bicubic" 
[nrow, ncol] = size(im_h_y);
im_h_cb = imresize(im_l_cb, [nrow, ncol], 'bicubic');
im_h_cr = imresize(im_l_cr, [nrow, ncol], 'bicubic');

% change back color space to RGB with the super-resolution image
im_h_ycbcr = zeros([nrow, ncol, 3]);
im_h_ycbcr(:, :, 1) = im_h_y;
im_h_ycbcr(:, :, 2) = im_h_cb;
im_h_ycbcr(:, :, 3) = im_h_cr;
im_h = ycbcr2rgb(uint8(im_h_ycbcr));

elapsed_time  = (cputime - start_time);
%Comparison of SR and Bicubic interpolation
% bicubic interpolation for reference
im_b = imresize(im_l, [nrow, ncol], 'bicubic');

% read ground truth image
im = imread('Data/Testing/gnd.bmp');

% compute PSNR for the illuminance channel
bb_rmse = compute_rmse(im, im_b);
sp_rmse = compute_rmse(im, im_h);

bb_psnr = 20*log10(255/bb_rmse);
sp_psnr = 20*log10(255/sp_rmse);

%fprintf('PSNR for Bicubic Interpolation: %f dB\n', bb_psnr);
%fprintf('PSNR for Sparse Representation Recovery: %f dB\n', sp_psnr);

% show the images
figure;
subplot(1,2,1);imshow(im_h);
title({'Sparse Recovery';['PSNR: ', num2str(sp_psnr),'dB']})
hold on
subplot(1,2,2);imshow(im_b);
title({'Bicubic Interpolation';['PSNR: ', num2str(bb_psnr),'dB']});
set(gcf,'color','white')
