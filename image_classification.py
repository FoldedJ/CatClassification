import os
import shutil

# 源文件夹路径，这里需要替换为你实际的大文件夹路径
source_folder = r'F:\迅雷下载\oxford-iiit-pet\images'
# 目标文件夹路径，这里需要替换为你实际想要存储分类后图片的文件夹路径
target_folder = './dataset'

# 检查目标文件夹是否存在，如果不存在则创建
if not os.path.exists(target_folder):
    os.makedirs(target_folder)

# 遍历源文件夹中的所有文件
for filename in os.listdir(source_folder):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        # 提取文件名中的类别信息
        category = filename.split('_')[-1]

        # 创建对应类别的目标子文件夹
        category_folder = os.path.join(target_folder, category)
        if not os.path.exists(category_folder):
            os.makedirs(category_folder)

        # 源文件的完整路径
        source_file = os.path.join(source_folder, filename)
        # 目标文件的完整路径
        target_file = os.path.join(category_folder, filename)

        # 复制文件到目标文件夹
        shutil.copy2(source_file, target_file)

print("图片分类完成！")
