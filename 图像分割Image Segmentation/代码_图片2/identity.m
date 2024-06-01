clc;
% 识别汉字
img1 = imread('./iden_num/1.bmp');

dir1 = './5-carNumber/汉字';

files = dir(fullfile(dir1, '*.bmp'));

img1_score = {};

for i = 1:length(files)
    filename = fullfile(dir1, files(i).name);
    I = rgb2gray(imread(filename));
    Img = imresize(I, size(img1));
    img1_score{i,1} = files(i).name;
    img1_score{i,2} = cosine_similarity(img1, Img);
end

img1_score = sortrows(img1_score, -2);

disp(['省份简称：',img1_score{1,1}])

% 识别字母和数字
img2 = imread('./iden_num/2.bmp');
img3 = imread('./iden_num/3.bmp');
img4 = imread('./iden_num/4.bmp');
img5 = imread('./iden_num/5.bmp');
img6 = imread('./iden_num/6.bmp');
img7 = imread('./iden_num/7.bmp');

lm_img = {img2,img3,img4,img5,img6,img7};

dir2 = './5-carNumber/字母和数字';

files2 = dir(fullfile(dir2, '*.bmp'));

for n = 1:length(lm_img)
    img_score = {};
    for i = 1:length(files2)
        filename = fullfile(dir2, files2(i).name);
        I = rgb2gray(imread(filename));
        Img = imresize(I, size(lm_img{n}));
        img_score{i,1} = files2(i).name;
        img_score{i,2} = cosine_similarity(lm_img{n}, Img);
    end
    img_score = sortrows(img_score, -2);
    disp(['字符：',img_score{1,1}])
end


function [similarity] = cosine_similarity(img1, img2)

% 将二值图像转换为向量
vec1 = double(img1(:)');
vec2 = double(img2(:)');

% 计算余弦相似度
similarity = dot(vec1, vec2) / (norm(vec1) * norm(vec2));
end