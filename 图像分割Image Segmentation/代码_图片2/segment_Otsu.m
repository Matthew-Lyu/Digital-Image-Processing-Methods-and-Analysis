clc;
p = plate(1,:);
rgb_plate_img = rgb_img(p(2)+0.5:p(2) + p(4) - 0.5, p(1)+0.5:p(1) + p(3) - 0.5,:);
plate_img = rgb2gray(rgb_img(p(2)+0.5:p(2) + p(4) - 0.5, p(1)+0.5:p(1) + p(3) - 0.5,:));

% 获取灰度图像的直方图
num_pixel = numel(plate_img);
histogram = imhist(plate_img) / num_pixel;

% 计算所有可能的阈值的类间方差，并选择最大值
max_variance = 0;
threshold = 0;

for i = 1:256
    % 计算类别概率
    P1 = sum(histogram(1:i));
    P2 = 1 - P1;

    % 计算均值
    mean_1 = sum((0:i-1)'.*histogram(1:i)) / P1;
    mean_2 = sum((i:255)'.*histogram(i+1:end)) / P2;

    % 计算类间方差
    variance = P1*P2*(mean_1 - mean_2)^2;

    % 更新阈值和最大方差
    if (variance > max_variance)
        max_variance = variance;
        threshold = i-1;
    end
end

% 根据阈值进行图像分割
segmented_img = plate_img > threshold;

% 显示原始图像和分割结果
subplot(1,2,1), imshow(plate_img), title('原图');
subplot(1,2,2), imshow(segmented_img), title('大津法');
