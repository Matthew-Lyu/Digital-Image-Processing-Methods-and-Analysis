clear
clc

%读入低分辨率图
srcImg = imread('input.bmp');

%设置放大倍数
T = 3;

% 1. 载入字典，归一化 Dl
load('D_1024_0.15_5.mat');
Dl = Dl ./ sqrt(sum(Dl.^2,1));


% 2. 获取特征块大小 patch_size，自定义重叠域 overlap，超分系数 lambda  
patch_size = sqrt(size(Dh, 1));
overlap = 3;
lambda = 0.2;


% 3. 利用插值法，把低分辨率图变大(与目标高分辨率图大小一致)，可用 imresize
im_l = imresize(srcImg, T);
im_l_ycbcr = rgb2ycbcr(im_l);
im_l_y = im_l_ycbcr(:,:,1);
im_l_cb = im_l_ycbcr(:,:,2);
im_l_cr = im_l_ycbcr(:,:,3);


% 4. 提取 resize 后的低分辨率图特征（一阶导和二阶导）
% 第一层和第二层采用一阶导算子
h1 = [-1, 0, 1];
h2 = [-1; 0; 1];
img_c1 = conv2(im_l_y, h1,'full');
img_c2 = conv2(im_l_y, h2,'full');
% 第三层和第四层采用二阶导算子
h3 = [1, 0, -2, 0, 1];
h4 = [1; 0; -2; 0; 1];
img_c3 = conv2(im_l_y, h3,'full');
img_c4 = conv2(im_l_y, h4,'full');


% 5. 对每个特征块求最优高分辨率块(循环)
[m, n] = size(im_l_y);
im_h_y = zeros([m, n]);

% 计算像素块加的次数
flag = zeros([m, n]);
for i = 1:overlap:(m - patch_size)
    for j = 1:overlap:(n - patch_size)
        % 计算图像块的均值 m
        idx_i = i: i + patch_size -1;
        idx_j = j: j + patch_size -1;
        patch = im_l_y(idx_i, idx_j);
        m_patch = mean(patch(:));

        % 找到对应位置的特征(5*5*4)向量，展开为一维向量(100*1)，并且归一化，得到 Fy
        sub_img_c1 = img_c1(idx_i, idx_j);
        sub_img_c2 = img_c2(idx_i, idx_j);
        sub_img_c3 = img_c3(idx_i, idx_j);
        sub_img_c4 = img_c4(idx_i, idx_j);
        Fy = [sub_img_c1(:); sub_img_c2(:); sub_img_c3(:); sub_img_c4(:)];
        Fy = Fy./norm(Fy);

        % 利用 Dl，Fy，求得 A，b，代入函数求得该块的最优稀疏系数 a
        A = Dl'*Dl;
        b = -Dl'*Fy;
        a = L1QP(lambda, A, b);
        x = Dh*a;
        im_h_y(idx_i, idx_j) = im_h_y(idx_i, idx_j)+ reshape(x+m_patch, [5 5]);
        flag(idx_i,idx_j) = flag(idx_i,idx_j) + 1;
    end
end

im_h_y = uint8(im_h_y./flag);
% 超分重建只对Y域操作，所以Cb和Cr域直接取resize后的值
im_h_cb = im_l_cb;
im_h_cr = im_l_cr;

im_h_ycbcr = cat(3, im_h_y,im_h_cb,im_l_cr);
im_h = ycbcr2rgb(uint8(im_h_ycbcr));

figure;
subplot(1,2,1),imshow(srcImg),title('原图');
subplot(1,2,2),imshow(im_h),title('基于字典的超分辨率重建');

