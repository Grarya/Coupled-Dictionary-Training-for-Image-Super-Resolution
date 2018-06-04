clear all; clc; close all;
addpath(genpath('RegularizedSC'));

img_path = 'Data/Training';

dict_size   = 512;        % dictionary size
lambda      = 0.15;       % sparsity regularization
patch_size  = 5;          % image patch size
num_patch   = 100000;     % number of patches to sample
upscale     = 2;          % upscaling factor
img_type    = '*.bmp';    % image type
threshold   = 10;         % threshold based on the training data for prunning
gamma       = 0.5;        % balance reconstruction error

%%1:Preparing data
%Randomly sample image patches
[x, y] = rnd_smp_patch(img_path, img_type, patch_size, num_patch, upscale);

%Prune patches with small variances, threshould chosen based on the
% training data
[x, y] = patch_pruning(x, y, threshold);

%%2: Coupled dictionary training algorithm
% load dictionary from joint sparse coding for initialize the dictionary
load(['Dictionary/D_',num2str(dict_size),'_',num2str(lambda),'_',num2str(patch_size),'.mat']);
Dx = Dh; %X is high resolution image
Dy = Dl; %Y is low resolution image

[Dx, Dy, error] = train_coupled_dict_SR(x, y, dict_size, lambda, upscale, Dx, Dy, gamma);
dict_path = ['Dictionary/CD_' num2str(dict_size) '_' num2str(lambda) '_' num2str(patch_size) '.mat' ];
save(dict_path, 'Dh', 'Dl');

% ========================================================================
% Reference
%   J. Yang et al. Image super-resolution as sparse representation of raw
%   image patches. CVPR 2008.
%   J. Yang et al. Image super-resolution via sparse representation. IEEE 
%   Transactions on Image Processing, Vol 19, Issue 11, pp2861-2873, 2010
%   J. Yang et al. Coupled Dictionary Training for Image Super-Resolution. 
%   Transactions on Image Processing, Vol 21, Issue 08, pp3467-3478, 2012
%
% Jianchao Yang
% ECE Department, University of Illinois at Urbana-Champaign
% For any questions, send email to jyang29@uiuc.edu
% =========================================================================