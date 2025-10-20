import os
import shutil

def extract_jpg_in_range(source_dir, target_dir, start_file, end_file):
    """
    从 source_dir 中提取 start_file 和 end_file 之间的 .jpg 文件，复制到 target_dir。

    :param source_dir: 源文件夹路径
    :param target_dir: 目标文件夹路径
    :param start_file: 起始文件名（包含路径）
    :param end_file: 结束文件名（包含路径）
    """
    # 确保目标文件夹存在
    os.makedirs(target_dir, exist_ok=True)

    # 提取文件名
    start_name = os.path.basename(start_file)
    end_name = os.path.basename(end_file)

    # 获取文件列表并排序
    files = sorted(f for f in os.listdir(source_dir) if f.endswith('.jpg'))

    # 确定范围
    try:
        start_index = files.index(start_name)
        end_index = files.index(end_name)
    except ValueError:
        print("起始或结束文件不在源文件夹中")
        return

    # 提取范围内的文件
    for file in files[start_index:end_index + 1]:
        src_path = os.path.join(source_dir, file)
        dst_path = os.path.join(target_dir, file)
        shutil.copy(src_path, dst_path)
        print(f"已复制: {file}")

# 示例调用
source_directory = "/home/zwk/dataset/nuscenes/CPG-LCF-main/data/nuscenes/sweeps/CAM_FRONT"
target_directory = "/home/zwk/dataset/nus"
start_image = "file:///home/zwk/dataset/nuscenes/CPG-LCF-main/data/nuscenes/sweeps/CAM_FRONT/n008-2018-08-31-11-19-57-0400__CAM_FRONT__1535728830612404.jpg"
end_image = "file:///home/zwk/dataset/nuscenes/CPG-LCF-main/data/nuscenes/sweeps/CAM_FRONT/n008-2018-08-31-11-19-57-0400__CAM_FRONT__1535728837862404.jpg"

# 去掉路径前的 "file://"
start_image = start_image.replace("file://", "")
end_image = end_image.replace("file://", "")

extract_jpg_in_range(source_directory, target_directory, start_image, end_image)