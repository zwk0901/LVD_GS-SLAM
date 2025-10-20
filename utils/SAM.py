#!/usr/bin/env python3
"""
一键修复SAM模型问题
运行此脚本自动下载和配置SAM模型
"""

import os
import sys
import urllib.request
import yaml


def download_with_progress(url, filename):
    """带进度条的下载"""

    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, (downloaded / total_size) * 100)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            print(f"\r下载进度: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end='', flush=True)

    try:
        print(f"开始下载: {filename}")
        urllib.request.urlretrieve(url, filename, progress_hook)
        print(f"\n✅ 下载完成: {filename}")
        return True
    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        return False


def check_and_fix_sam():
    """检查并修复SAM模型问题"""
    print("🔍 检查SAM模型状态...")

    # 1. 检查segment-anything包
    try:
        import segment_anything
        print("✅ segment-anything 包已安装")
    except ImportError:
        print("❌ segment-anything 包未安装")
        print("正在安装...")
        os.system("pip install segment-anything")

    # 2. 检查现有模型文件
    sam_models = {
        'sam_vit_b_01ec64.pth': {
            'url': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth',
            'size_mb': 375,
            'description': '小模型 (推荐)'
        },
        'sam_vit_l_0b3195.pth': {
            'url': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth',
            'size_mb': 1249,
            'description': '中等模型'
        },
        'sam_vit_h_4b8939.pth': {
            'url': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',
            'size_mb': 2560,
            'description': '大模型 (最高精度)'
        }
    }

    # 检查是否已存在模型
    existing_models = []
    for model_name in sam_models.keys():
        if os.path.exists(model_name):
            size_mb = os.path.getsize(model_name) / (1024 * 1024)
            existing_models.append((model_name, size_mb))

    if existing_models:
        print("✅ 找到现有SAM模型:")
        for name, size in existing_models:
            print(f"  - {name} ({size:.1f}MB)")
        return existing_models[0][0]  # 返回第一个找到的模型

    # 3. 如果没有模型，下载小模型
    print("📥 未找到SAM模型，正在下载小模型...")
    model_name = 'sam_vit_b_01ec64.pth'
    model_info = sam_models[model_name]

    if download_with_progress(model_info['url'], model_name):
        return model_name
    else:
        return None


def update_config_file():
    """更新配置文件中的SAM路径"""
    config_files = ['config.yaml', 'slam_config.yaml', 'example_config.yaml']

    model_path = check_and_fix_sam()
    if not model_path:
        print("❌ 无法获取SAM模型，建议禁用SAM")
        return False

    abs_model_path = os.path.abspath(model_path)
    print(f"📝 模型路径: {abs_model_path}")

    # 查找并更新配置文件
    updated_files = []
    for config_file in config_files:
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)

                # 更新SAM配置
                if 'dynamic_filtering' not in config:
                    config['dynamic_filtering'] = {}

                config['dynamic_filtering']['use_sam'] = True
                config['dynamic_filtering']['sam_checkpoint'] = abs_model_path

                with open(config_file, 'w', encoding='utf-8') as f:
                    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

                updated_files.append(config_file)
                print(f"✅ 已更新配置文件: {config_file}")

            except Exception as e:
                print(f"⚠️  无法更新 {config_file}: {e}")

    if not updated_files:
        # 创建新的配置文件
        config = {
            'dynamic_filtering': {
                'enabled': True,
                'use_sam': True,
                'sam_checkpoint': abs_model_path,
                'save_masked_images': True,
                'save_dir': './masked_images',
                'yolo': {
                    'model': 'yolov8n.pt',
                    'confidence_threshold': 0.3,
                    'dynamic_classes': [0, 2, 3, 5, 7]
                }
            }
        }

        with open('sam_fixed_config.yaml', 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

        print("✅ 已创建新配置文件: sam_fixed_config.yaml")
        updated_files.append('sam_fixed_config.yaml')

    return updated_files


def test_sam_setup():
    """测试SAM设置是否正确"""
    print("🧪 测试SAM设置...")

    try:
        from segment_anything import sam_model_registry, SamPredictor
        import torch
        import numpy as np

        # 找到模型文件
        model_files = ['sam_vit_b_01ec64.pth', 'sam_vit_l_0b3195.pth', 'sam_vit_h_4b8939.pth']
        model_path = None
        model_type = None

        for model_file in model_files:
            if os.path.exists(model_file):
                model_path = model_file
                if 'vit_b' in model_file:
                    model_type = 'vit_b'
                elif 'vit_l' in model_file:
                    model_type = 'vit_l'
                elif 'vit_h' in model_file:
                    model_type = 'vit_h'
                break

        if not model_path:
            print("❌ 未找到SAM模型文件")
            return False

        # 加载模型
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry[model_type](checkpoint=model_path)
        sam.to(device=device)
        predictor = SamPredictor(sam)

        # 简单测试
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        predictor.set_image(test_image)

        input_box = np.array([100, 100, 200, 200])
        masks, scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
        )

        print(f"✅ SAM测试成功! 模型: {model_type}, 设备: {device}")
        return True

    except Exception as e:
        print(f"❌ SAM测试失败: {e}")
        return False


def main():
    """主修复流程"""
    print("🚀 SAM模型一键修复工具")
    print("=" * 50)

    # 1. 检查并下载模型
    print("\n第1步: 检查和下载SAM模型")
    model_path = check_and_fix_sam()

    if not model_path:
        print("❌ 模型下载失败，建议手动下载或禁用SAM")
        return False

    # 2. 更新配置文件
    print("\n第2步: 更新配置文件")
    updated_configs = update_config_file()

    # 3. 测试设置


    # 4. 总结
    print("\n" + "=" * 50)
    print("🎉 修复完成!")
    print(f"SAM模型: {os.path.abspath(model_path)}")

    if updated_configs:
        print("已更新的配置文件:")
        for config in updated_configs:
            print(f"  - {config}")


    print("\n使用方法:")
    if updated_configs:
        print(f"python your_slam.py --config {updated_configs[0]}")
    print("或者在现有配置中设置:")
    print(f"sam_checkpoint: '{os.path.abspath(model_path)}'")

    return True


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n操作取消")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        print("请手动下载SAM模型或在配置中禁用SAM")