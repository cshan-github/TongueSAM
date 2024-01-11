
import os
import shutil

# 源文件夹路径
src_folder = '/home/disk1/cs/project/dataset/COCO/val_gt/'

# 目标文件夹路径
dst_folder = '/home/disk1/cs/project/dataset/COCO/split/val_gt/'

# 每组文件数
group_size = 8

# 遍历文件夹中所有图片文件，按顺序分组
file_groups = []
for i, file_name in enumerate(sorted(os.listdir(src_folder))):
    # if file_name.endswith('.jpg'):
    group_idx = i // group_size
    if len(file_groups) <= group_idx:
        file_groups.append([])
    file_groups[group_idx].append(file_name)

# 创建目标文件夹
os.makedirs(dst_folder, exist_ok=True)

# 将每组文件复制到目标文件夹
for i, group in enumerate(file_groups):
    group_folder = os.path.join(dst_folder, f'group_{i}')
    os.makedirs(group_folder, exist_ok=True)
    for file_name in group:
        src_file = os.path.join(src_folder, file_name)
        dst_file = os.path.join(group_folder, file_name)
        shutil.copy(src_file, dst_file)
