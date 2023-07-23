clear
clc

img = imread('demo1.jpg');
gray_img = rgb2gray(img);

figure;
subplot(2, 2, 1), imshow(gray_img);
title("原图");

% 创建一个运动模糊滤波器
motion_kernel = fspecial('motion', 30, 45);

% 为灰度图添加运动模糊
blurred_img = imfilter(gray_img, motion_kernel, 'conv', 'circular');
subplot(2, 2, 2), imshow(blurred_img);
title("运动模糊");

% 添加高斯噪声
noisy_img = imnoise(blurred_img, 'gaussian', 0, 0.01);
subplot(2, 2, 3), imshow(noisy_img);
title("运动模糊+高斯噪声");

% 运动模糊滤波器的傅里叶变化
H = fft2(motion_kernel, size(gray_img, 1), size(gray_img, 2));

% 获取噪声分量N(u, v)
noisy = noisy_img - blurred_img;
N = fft2(noisy);

% 获取未退化的图片F(u, v)
F = fft2(gray_img);

% 计算信噪比NSR
NSR = abs(N).^2 ./ abs(F).^2;

% 搭建维纳滤波器F_hat(u, v)
F_hat = conj(H) ./(abs(H).^2 + NSR);

% 获取模糊图片G(u, v)
G = fft2(blurred_img);

% 还原图片
restored_img = real(ifft2(F_hat .* G));

subplot(2, 2, 4), imshow(restored_img,[]);
title("恢复图像");