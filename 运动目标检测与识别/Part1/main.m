clear
clc

load('background.mat');

% 导入视频
videoDir = './视频数据集/001.mp4';
videoReader = VideoReader(videoDir);

% 获取视频的宽度和高度
width = videoReader.Width;
height = videoReader.Height;

% 计算视频帧的总数
numFrames = videoReader.NumFrames;

% 创建一个新的VideoWriter对象来保存输出视频
outputFile = 'result2';
videoWriter = VideoWriter(outputFile);
open(videoWriter);

for index = 1:numFrames
    % 读取当前帧
    frame = read(videoReader, index);

    % 与背景帧进行差分
    diffFrame = imabsdiff(frame, background);

    % 将图像二值化
    gray_img = rgb2gray(diffFrame);
    threshold = 25;
    binary_img = gray_img >= threshold;
    % 高斯滤波，处理车道线
    kernel = fspecial('gaussian', 6,6); 
    binary_img = imfilter(binary_img, kernel, 'replicate'); 
    
    % 删除过小的联通区域
    binary_img = bwareaopen(binary_img, 600);
    % 形态学处理
    SE1 = strel('rectangle',[2,2]);
    after_img = imerode(binary_img, SE1);
    SE2 = strel('disk', 9);
    % 闭运算
    im_close = imclose(binary_img, SE2);

    % 获取连通区域属性
    stats = regionprops(im_close, 'BoundingBox', 'Area');

    % 在原始图像上绘制矩形
    imshow(frame)
    hold on;
    for i = 1:length(stats)
        area = stats(i).Area;  % 获取当前连通区域的面积
        if area >= 300
            rectangle('Position', stats(i).BoundingBox, 'EdgeColor', 'r', 'LineWidth', 3);  % 绘制框
            hold on;
        end
    end
        h = getframe(gcf);
    hold off;
    writeVideo(videoWriter, h);
end

% 关闭VideoWriter对象
close(videoWriter);