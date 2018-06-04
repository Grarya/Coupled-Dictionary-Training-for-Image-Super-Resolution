%Results comparison

%% Overlapping pixels

overlap= [0, 1, 2, 3, 4];
bb_psnr_lena = 32.7958*ones(1, 5);
bb_psnr_flw = 32.7958*ones(1, 5);
sp_psnr_lena = [33.656608859955020, 34.3986, 34.5456,34.9144 ,35.0119];

figure;
plot(overlap, bb_psnr_lena, '--','LineWidth', 1, 'Color', 'k');
hold on
plot(overlap, sp_psnr_lena, '-o', 'LineWidth', 1, 'Color' , 'r');
axis([0 4, 32 35.3]);
xticks([0 1 2 3 4])
xticklabels({'0','1','2','3', '4'})
xlabel('Overlapping pixels'); ylabel('PSNR')
legend('Bicubic','Our results', 'Location','southeast');
set(gca,'fontsize',13)
title('Lena','fontsize',16 );
set(gcf,'color','white')

%% Basis dictionary trained
n=5;
Dh_img = ToBlock(Dh, n);
M = size(Dh_img,1);
N = size(Dh_img,2);

indx = (Dh_img(:,:)~=0);
Dh_img = uint8(250 * mat2gray(Dh_img).*indx);
figure;
imshow(Dh_img, []); title('HR image patch dictionary trained');
set(gcf,'color','white')
hold on;
for k = 1:n:M;
    x = [1 N];
    y = [k k];
    plot(x,y,'Color','k','LineStyle','-');
end
for k = 1:n:N;
    x = [k k];
    y = [1 M];
    plot(x,y,'Color','k','LineStyle','-');
end
hold off;


Dl_1 = Dl(1:25, :);
Dl_2 = Dl(26:50, :);
Dl_3 = Dl(51:75, :);
Dl_4 = Dl(76:100, :);

Dl_img_1 = ToBlock(Dl_1, 5);
Dl_img_2 = ToBlock(Dl_2, 5);
Dl_img_3 = ToBlock(Dl_3, 5);
Dl_img_4 = ToBlock(Dl_4, 5);
figure;
subplot(2,2,1); imshow(Dl_img_1,[]); title('Dl');
subplot(2,2,2); imshow(Dl_img_2,[]); title('Dh');
subplot(2,2,3); imshow(Dl_img_3,[]); title('Dh');
subplot(2,2,4); imshow(Dl_img_4,[]); title('Dh');

%% Iterations
% Load saved figures
c=hgload('It2_100k.fig');
k=hgload('It3_100k.fig');
l=hgload('It4_100k.fig');
% Prepare subplots
figure
h(1)=subplot(1,1,1);
h(2)=subplot(1,1,1);
h(3)=subplot(1,1,1);
% Paste figures on the subplots
copyobj(allchild(get(c,'CurrentAxes')),h(1));
copyobj(allchild(get(k,'CurrentAxes')),h(2));
copyobj(allchild(get(l,'CurrentAxes')),h(3));

% Add legends
l(1)=legend(h(1),'Error per iteration');

%% Dictionary size

size = [256, 512, 1024];
sp_psnr = [34.4817,35.0119 ,35.038009711645294];
time = [1.466562500000000e+02, 1.957031250000000e+02,2.758125000000000e+02];
train_time = [0.4566, 1.2755, 2.8476];
figure;
plot(size, sp_psnr);
xticks([256 512 1024])
xticklabels({'256','512','1024'})
grid on;
xlabel('Dictionary size'); ylabel('PSNR')
title('Lena','fontsize',16 );
set(gcf,'color','white')

figure;
plot(size, time);
xticks([256 512 1024])
xticklabels({'256','512','1024'})
grid on;
xlabel('Dictionary size'); ylabel('time(s)')
title('Lena','fontsize',16 );
set(gcf,'color','white')

figure;
plot(size, train_time);
xticks([256 512 1024])
xticklabels({'256','512','1024'})
grid on;
xlabel('Dictionary size'); ylabel('time(hours)')
title('Training time','fontsize',16 );
set(gcf,'color','white')


imdiff = im(:,:,1)  - im_h(:,:,1);
imshow(((imcomplement(imdiff))), [])
