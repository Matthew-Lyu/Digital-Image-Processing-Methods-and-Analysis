clc;
p = plate(1,:);
rgb_plate_img = rgb_img(p(2)+0.5:p(2) + p(4) - 0.5, p(1)+0.5:p(1) + p(3) - 0.5,:);
plate_img = rgb2gray(rgb_img(p(2)+0.5:p(2) + p(4) - 0.5, p(1)+0.5:p(1) + p(3) - 0.5,:));

% 定义局部窗口大小和亮度阈值
window_size = 1;  % 窗口大小
brightness_threshold = 50;  % 亮度阈值

% 获取图像尺寸
[rows, cols] = size(plate_img);

% 初始化二值化图像
binary_image = zeros(rows, cols);

% 遍历图像中的每个像素
for i = 1:rows
    for j = 1:cols
        % 获取当前像素的局部窗口
        row_start = max(i - floor(window_size/2), 1);
        row_end = min(i + floor(window_size/2), rows);
        col_start = max(j - floor(window_size/2), 1);
        col_end = min(j + floor(window_size/2), cols);
        window = plate_img(row_start:row_end, col_start:col_end);
        
        % 计算局部窗口中的最小和最大像素值
        min_val = min(window(:));
        max_val = max(window(:));
        
        % 计算局部窗口的动态阈值
        threshold = (max_val + min_val) / 2;
        
        % 检查当前像素的亮度是否超过动态阈值
        if plate_img(i, j) >= threshold - brightness_threshold && plate_img(i, j) <= threshold + brightness_threshold
            binary_image(i, j) = 255;
        end
    end
end

binary_image  = imcomplement(imbinarize(binary_image));

% 显示原始图像和处理后的图像
figure;
subplot(1, 2, 1),imshow(plate_img),title('原图');
subplot(1, 2, 2),imshow(binary_image),title('bernsen');

x1 = [];
x2 = [];
X = [];
for x = 1:cols
    if sum(binary_image(:,x)) == 0
        X = [X, x];
    end
end

for i = 1:length(X)-1
    if X(i) + 1 ~= X(i+1)
        x1 = [x1; X(i)];
        x2 = [x2; X(i+1)];
    end
end

figure;
path = './iden_num/';
for n = 1:length(x1)
    subplot(1,7,n),imshow(rgb_plate_img(1:rows, x1(n):x2(n),:));
    imwrite(rgb_plate_img(1:rows, x1(n):x2(n)),[path,num2str(n),'.bmp']);
end