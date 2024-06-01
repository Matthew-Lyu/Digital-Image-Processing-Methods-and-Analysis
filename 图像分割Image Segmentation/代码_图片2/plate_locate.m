clear
clc

rgb_img = imread('test2.jpg');
img = rgb2gray(imread('test2.jpg'));

figure;

% 边缘检测，使用roberts算子
img_edge = edge(img,'roberts');
subplot(2,2,1),imshow(img_edge);

% 灰度反转
inverted_img = imcomplement(img_edge);

% 开闭运算
se1= strel("rectangle",[8 8]);
open_img = imopen(inverted_img,se1);
subplot(2,2,2),imshow(open_img);
se2= strel("rectangle",[10 15]);
close_img = imclose(open_img,se2);
subplot(2,2,3),imshow(open_img);

% 标记连通域
labeled_img = bwlabel(close_img);

% 获取连通域的属性
stats = regionprops(labeled_img, 'Area');

% 设定阈值，用于删除过小的连通域
threshold = 1000; % 根据需要调整阈值的大小

% 删除过小的连通域
for i = 1:length(stats)
    if stats(i).Area < threshold
        labeled_img(labeled_img == i) = 0;
    end
end
labeled_img = imbinarize(labeled_img);
subplot(2,2,4),imshow(labeled_img);

% 矩形化
% 获取尺寸
[m, n] = size(labeled_img);

% 遍历图像中的每个像素（左上角）
for i = 2:m-1
    for j = 2:n-1
        % 获取当前像素的值
        pixel = labeled_img(i, j);
        
        % 获取4-邻域的像素值
        neighbors = [labeled_img(i-1, j), labeled_img(i+1, j), labeled_img(i, j-1),labeled_img(i, j+1)];
        
        % 统计4-邻域中的黑色像素数量
        black_count = sum(neighbors(:) == 0);

        % 统计4-邻域中的白色像素数量
        white_count = sum(neighbors(:) == 1);

        if pixel == 0 % 当前像素为黑色
            if white_count > 2 % 如果至少有三个邻域像素为白色
                labeled_img(i, j) = 1; % 将当前像素设置为白色
            end
        else % 当前像素为白色
            if black_count >= 2 % 如果所有邻域像素都为黑色
                labeled_img(i, j) = 0; % 将当前像素设置为黑色
            end
        end
    end
end

% 右上角
for i = m-1:-1:2
    for j = 2:n-1
        pixel = labeled_img(i, j);
        neighbors = [labeled_img(i-1, j), labeled_img(i+1, j), labeled_img(i, j-1),labeled_img(i, j+1)];
        black_count = sum(neighbors(:) == 0);
        white_count = sum(neighbors(:) == 1);

        if pixel == 0 % 当前像素为黑色
            if white_count > 2 % 如果至少有两个邻域像素为白色
                labeled_img(i, j) = 1; % 将当前像素设置为白色
            end
        else % 当前像素为白色
            if black_count >= 2 % 如果所有邻域像素都为黑色
                labeled_img(i, j) = 0; % 将当前像素设置为黑色
            end
        end
    end
end

% 左下角
for i = 2:m-1
    for j = n-1:-1:2
        pixel = labeled_img(i, j);
        neighbors = [labeled_img(i-1, j), labeled_img(i+1, j), labeled_img(i, j-1),labeled_img(i, j+1)];
        black_count = sum(neighbors(:) == 0);
        white_count = sum(neighbors(:) == 1);
        if pixel == 0 % 当前像素为黑色
            if white_count > 2 % 如果至少有三个邻域像素为白色
                labeled_img(i, j) = 1; % 将当前像素设置为白色
            end
        else % 当前像素为白色
            if black_count >= 2 % 如果所有邻域像素都为黑色
                labeled_img(i, j) = 0; % 将当前像素设置为黑色
            end
        end
    end
end

% 右下角
for i = m-1:-1:2
    for j = n-1:-1:2
        pixel = labeled_img(i, j);
        neighbors = [labeled_img(i-1, j), labeled_img(i+1, j), labeled_img(i, j-1),labeled_img(i, j+1)];
        black_count = sum(neighbors(:) == 0);
        white_count = sum(neighbors(:) == 1);
        if pixel == 0 % 当前像素为黑色
            if white_count > 2 % 如果至少有三个邻域像素为白色
                labeled_img(i, j) = 1; % 将当前像素设置为白色
            end
        else % 当前像素为白色
            if black_count >= 2 % 如果所有邻域像素都为黑色
                labeled_img(i, j) = 0; % 将当前像素设置为黑色
            end
        end
    end
end

figure;
imshow(labeled_img);

hsv_img = rgb2hsv(rgb_img);

% 定义筛选条件的阈值
% 矩形长宽比阈值
ratio = [0, 10];
% 蓝色像素比例阈值
blue_ratio = 0.5;
% 交点个数阈值
count = [5, 15];

% 获取图像的尺寸
[height, width] = size(labeled_img);

% 存储符合条件的车牌位置
plate = [];

% 标记图像中的对象
L = bwlabel(imcomplement(labeled_img));

% 计算区域属性
stats = regionprops(L, 'BoundingBox');

for i = 1:numel(stats)
    % 获取当前矩形区域的边界框信息
    bbox = stats(i).BoundingBox;
    
    % 计算矩形的长宽比
    wh = bbox(3) / bbox(4);
    % 检查长宽比是否符合约束条件
    if (wh > ratio(1) && wh < ratio(2))
        % 统计矩形区域中蓝色像素点的数量
        blue_count = 0;
        for y = bbox(2)+0.5:bbox(2) + bbox(4) - 0.5
            for x = bbox(1)+0.5:bbox(1) + bbox(3) - 0.5
                if hsv_img(y, x, 1) <= 0.6667 && hsv_img(y, x, 1) >= 0.5833
                    blue_count = blue_count + 1;
                end
            end
        end
    

        % 计算蓝色像素点的比例
        br = blue_count / (bbox(3) * bbox(4));
    
        % 检查蓝色像素比例是否符合约束条件
        if br > blue_ratio
            % 计算交点个数
            pj_count = Verticalprojection(img(bbox(2)+0.5:bbox(2) + bbox(4) - 0.5, bbox(1)+0.5:bbox(1) + bbox(3) - 0.5));
            % 检查交点个数是否符合约束条件
            if pj_count > count(1) && pj_count < count(2)
                % 将符合条件的车牌位置添加到结果中
                plate = [plate; bbox];
            end
        end
    end
end

% 在原始图像中标记车牌位置
figure,imshow(rgb_img);
hold on;
for i = 1:size(plate, 1)
    rectangle('Position', plate(i, :), 'EdgeColor', 'r', 'LineWidth', 2);
end
hold off;