clear
clc

% 读取图像
img = imread('./3.jpg');

% 转为灰度图像
gray_img = rgb2gray(img);

% 高斯滤波
gauss_img = imgaussfilt(gray_img, 2.5);

% Sobel算子计算像素梯度
sobel_x = [-1 0 1; -2 0 2; -1 0 1];
sobel_y = [-1 -2 -1; 0 0 0; 1 2 1];
grad_x = imfilter(double(gauss_img), sobel_x);
grad_y = imfilter(double(gauss_img), sobel_y);
grad_mag = sqrt(grad_x.^2 + grad_y.^2);
grad_dir = atan2(grad_y, grad_x);

% 非极大值抑制
nms_mag = zeros(size(grad_mag));
for i = 2:size(grad_mag, 1)-1
    for j = 2:size(grad_mag, 2)-1
        dir = grad_dir(i,j);
        mag = grad_mag(i,j);
        if (dir > -22.5 && dir <= 22.5) || (dir > 157.5 && dir <= -157.5)  % 水平方向
            if (mag >= grad_mag(i,j-1) && mag >= grad_mag(i,j+1))
                nms_mag(i,j) = mag;
            end
        elseif (dir > 112.5 && dir <= 157.5) || (dir > -67.5 && dir <= -22.5)  % 45度方向
            if (mag >= grad_mag(i-1,j-1) && mag >= grad_mag(i+1,j+1))
                nms_mag(i,j) = mag;
            end
        elseif (dir > 67.5 && dir <= 112.5) || (dir > -112.5 && dir <= -67.5)  % 垂直方向
            if (mag >= grad_mag(i-1,j) && mag >= grad_mag(i+1,j))
                nms_mag(i,j) = mag;
            end
        elseif (dir > 22.5 && dir <= 67.5) || (dir > -157.5 && dir <= -112.5)  % -45度方向
            if (mag >= grad_mag(i-1,j+1) && mag >= grad_mag(i+1,j-1))
                nms_mag(i,j) = mag;
            end
        end
    end
end


% 阈值滞后处理
th_high = 20;
th_low = 10;
edge_img = zeros(size(nms_mag));
edge_img(nms_mag >= th_high) = 1;
edge_img(nms_mag < th_low) = 0;

% 孤立弱边缘抑制
for i = 2:size(nms_mag, 1)-1
    for j = 2:size(nms_mag, 2)-1
        if (nms_mag(i,j) >= th_low && nms_mag(i,j) < th_high)
            if (any(edge_img(i-1:i+1, j-1:j+1) == 1))
                edge_img(i,j) = 1;
            end
        end
    end
end

% 使用高斯滤波强化图像，使边缘更明显
edge_img = imgaussfilt(edge_img, 0.3);
figure,imshow(edge_img);

[x, y] = ginput(4);
% 选取 ROI 区域生成蒙版图像
roi_mask = poly2mask(x, y, size(edge_img, 1), size(edge_img, 2));
roi_img = edge_img & roi_mask;
figure,imshow(roi_img);


% 霍夫变换识别直线边缘
[h, theta, rho] = hough(roi_img);
peaks = houghpeaks(h, 10);
lines = houghlines(roi_img, theta, rho, peaks, 'FillGap', 2, 'MinLength', 5);


% 根据长度排序
lineLengths = zeros(length(lines), 1);
for i = 1:length(lines)
    x1 = lines(i).point1(1);
    y1 = lines(i).point1(2);
    x2 = lines(i).point2(1);
    y2 = lines(i).point2(2);
    lineLengths(i) = sqrt((x2-x1)^2 + (y2-y1)^2);
end
[sortedLengths, sortedIndices] = sort(lineLengths, 'descend');
sortedLines = lines(sortedIndices);

figure,imshow(gray_img);
hold on
for k = 1:2
   xy =  [sortedLines(k).point1; sortedLines(k).point2];
   plot (xy (:,1), xy (:,2), 'Linewidth' ,2, 'Color', 'green');
   plot (xy (1,1), xy (1,2), 'x', 'Linewidth',2, 'Color','yellow');
   plot (xy (2,1), xy (2,2), 'x', 'Linewidth', 2, 'Color', 'red');
end
hold off

