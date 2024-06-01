clear
clc

% 读入待处理的图像
srcImg = imread('input.bmp');

% 设置放大倍数
T = 3;

srcWidth = size(srcImg, 2);
srcHeight = size(srcImg, 1);
dstWidth = size(srcImg, 2) * T;
dstHeight = size(srcImg, 1) * T;

dstImg2 = uint8(zeros(dstWidth, dstHeight,size(srcImg, 3)));

% 双线性插值
for i = 1:dstHeight % 遍历目标图像的每个像素
    for j = 1:dstWidth
        % 找到原图像中最近的四个像素坐标
        % 得到Q11的坐标点
        x = floor(i / T); % 目标像素在原图像中的行坐标
        y = floor(j / T); % 目标像素在原图像中的列坐标
        
        % 计算目标像素在最近邻四个像素中的位置权重
        % i/T和j/T为待求像素的位置x
        dx = i / T - x; 
        dy = j / T - y; 
        
        % 超出边界的像素，使用最近边界像素进行填充
        if x < 1 
            x = 1;
            dx = 0;
        elseif x >= srcHeight 
            x = srcHeight - 1; 
            dx = 1; 
        end
        
        if y < 1 
            y = 1;
            dy = 0; 
        elseif y >= srcWidth 
            y = srcWidth - 1;
            dy = 1; 
        end
        
        % 双线性插值计算
        A = double(srcImg(x, y,:)); % 最近邻像素Q11
        B = double(srcImg(x, y+1,:)); % 最近邻像素Q12
        C = double(srcImg(x+1, y,:)); % 最近邻像素Q21
        D = double(srcImg(x+1, y+1,:)); % 最近邻像素Q22
        
        fR1 = (1-dx)*A + dx*C; % 在x方向上进行线性插值
        fR2 = (1-dx)*B + dx*D; % 在x方向上进行线性插值
        fP = (1-dy)*fR1 + dy*fR2; % 在y方向上进行线性插值
        dstImg2(i, j,:) = fP; % 将插值后的像素值赋给目标像素
    end
end

figure;
subplot(1,2,1),imshow(srcImg),title('原图');
subplot(1,2,2),imshow(dstImg2),title('双线性插值');