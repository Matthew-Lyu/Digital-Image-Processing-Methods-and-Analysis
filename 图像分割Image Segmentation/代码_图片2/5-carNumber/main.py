import os

# 指定要重命名的文件夹路径
folder_path = "/Users/LYU/Downloads/图像处理实验8/5-carNumber/数字"

# 获取文件夹中所有文件的文件名列表
file_names = os.listdir(folder_path)

# 对文件名按照字母顺序排序
file_names.sort()

# 循环重命名所有文件
for i in range(len(file_names)):
    # 获取当前文件名和文件扩展名
    old_name, ext = os.path.splitext(file_names[i])
    # 构建新文件名
    new_name = str(i) + ext
    # 重命名文件
    os.rename(os.path.join(folder_path, file_names[i]), os.path.join(folder_path, new_name))
    # 输出重命名信息
    print(f"{file_names[i]} → {new_name}")
