clear
clc

% 读入待处理的图像
srcImg = imread('input.bmp');

% 设置放大倍数
T = 3;

% 双三次插值
dstImg3 = bicubic_interpolation(srcImg, T);

figure;
subplot(1, 2, 1), imshow(srcImg), title('原图');
subplot(1, 2, 2), imshow(dstImg3), title('双三次插值法');

function  res = W(x)
%% 权重函数
a = -0.5; %一般是-1或者-0.5
if abs(x)<=1
    res = (a+2)*abs(x)^3 - (a+3)*abs(x)^2 + 1;
elseif abs(x)>1 && abs(x)<2
    res = a*abs(x)^3 - 5*a*abs(x)^2 + 8*a*abs(x) - 4*a;
else
    res = 0.;
end
end

function dstImg3 = bicubic_interpolation(srcImg, T)
%% 双三次插值法
[srcHeight, srcWidth,~] = size(srcImg);
% 新图像的大小
dstHeight = round(srcHeight*T);
dstWidth = round(srcWidth*T);

% 创建新图像的矩阵
dstImg3 = uint8(zeros(dstHeight, dstWidth,size(srcImg, 3)));

% 双三次插值法插值
for i = 1:dstHeight
    for j = 1:dstWidth
        % 找到原图像中最近的那个像素坐标
        x = ceil(i / T);
        y = floor(j / T);
        
        % 超出边界的像素，使用最近边界像素进行填充
        if x < 2
            x = 2;
        elseif x > srcHeight-2
            x = srcHeight-2;
        end

        if y < 2
            y = 2;
        elseif y > srcWidth-2
            y = srcWidth-2;
        end
        
        BXY = 0;
        % 进行插值
        for m = -1:1:2  % 横坐标
            for n = -1:1:2  %纵坐标
                % 计算权重
                weight_x = W(x+m-i/T);
                weight_y = W(y+n-j/T);
                % 计算插值
                BXY = BXY + double(srcImg(x+m,y+n,:))*weight_x*weight_y;
            end
        end
        dstImg3(i,j,:) = uint8(BXY);
    end
end
end
