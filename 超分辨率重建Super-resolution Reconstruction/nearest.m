clear
clc

%读入待处理的图像
srcImg = imread('input.bmp');

%读入放大倍数
T = 3;

%最邻近插值法(nearest)
srcWidth = size(srcImg, 2);
srcHeight = size(srcImg, 1);
dstWidth = size(srcImg, 2) * T;
dstHeight = size(srcImg, 1) * T;

%创建一个空的目标图像
dstImg1 = uint8(zeros(dstHeight, dstWidth, size(srcImg, 3)));

%最邻近插值法
%遍历目标图像的每个像素
for y = 1 : dstHeight
    for x = 1 : dstWidth
        %计算目标图像像素对应的原图像像素坐标
        srcX = ceil(x/T);
        srcY = ceil(y/T);
        %将原图像像素的值赋给目标图像像素
        dstImg1(y, x, :) = srcImg(srcY, srcX, :);
    end
end

%显示原图和最邻近插值法的结果
figure;
subplot(1,2,1),imshow(srcImg),title('原图');
subplot(1,2,2),imshow(dstImg1),title('最邻近插值法');

