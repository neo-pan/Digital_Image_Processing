clear;
close all;

I = imread('rice.png');
figure; imshow(I);
% 创建半径为12的圆盘形结构元
se = strel('disk',12);
% 用该结构元对图像进行顶帽操作
I_tophat = imtophat(I,se);
% 将图像根据其全局Otsu阈值二值化
bw = imbinarize(I_tophat,"global");
% 消除孤立的亮点
bw = bwmorph(bw,'clean');

% 对二值图像进行联通区域标记
[L, num] = bwlabel(bw);

figure; imshow(L,[]);
disp(['米粒的数量是:',num2str(num)]);
