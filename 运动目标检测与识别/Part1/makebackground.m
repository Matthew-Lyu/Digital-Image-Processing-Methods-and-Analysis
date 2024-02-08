clear
clc

videoReader = VideoReader("/Users/LYU/Downloads/图像处理期末大作业/参考文件/视频数据集/001.mp4");
numFrames = videoReader.NumFrames;

% 初始化背景帧变量
background = zeros(videoReader.Height, videoReader.Width, 3);

% 循环遍历每一帧并累加
for i = 1:numFrames
    % 读取当前帧
    frame = read(videoReader, i);
    
    % 累加当前帧到背景帧
    background = background + double(frame);
end

% 计算均值，作为背景帧
background = uint8(background / numFrames);

save("background.mat",'background');