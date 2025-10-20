# enhanced_grounding_dino_sam_detection.py
import torch
import cv2
import numpy as np
from groundingdino.util.inference import load_model, load_image, predict
from groundingdino.util import box_ops
from PIL import Image
import torchvision
import time
import os
import colorsys
import glob
from pathlib import Path
import json
from datetime import datetime

# SAM imports
try:
    from segment_anything import SamPredictor, sam_model_registry

    SAM_AVAILABLE = True
    print("✅ SAM is available")
except ImportError:
    SAM_AVAILABLE = False
    print("❌ SAM not available. Install with: pip install segment-anything")


class EnhancedDynamicStaticDetector:
    """增强的动态和静态物体分类检测器 + SAM精确分割 + 批量处理"""

    def __init__(self, grounding_dino_config, grounding_dino_checkpoint,
                 sam_checkpoint=None, device="cuda"):
        self.device = device
        self.grounding_dino_model = None
        self.sam_predictor = None

        # 加载Grounding DINO
        self.load_grounding_dino(grounding_dino_config, grounding_dino_checkpoint)

        # 加载SAM
        if sam_checkpoint and SAM_AVAILABLE:
            self.load_sam(sam_checkpoint)
        else:
            print("⚠️  SAM未启用，将仅使用边界框")

        # 定义动态和静态物体类别
        self.dynamic_objects = {
            "primary": ["car", "truck", "bus", "motorcycle", "bicycle", "person", "pedestrian"],
            "secondary": ["vehicle", "automobile", "people", "bike", "motorbike", "van", "suv"],
            "specific": ["walking person", "moving car", "driving vehicle", "riding bicycle"]
        }

        self.static_objects = {
            "primary": ["building", "tree", "road", "sidewalk", "wall", "fence", "pole"],
            "secondary": ["house", "street", "pavement", "barrier", "traffic light", "street lamp"],
            "specific": ["traffic sign", "stop sign", "road sign", "fire hydrant", "mailbox", "bench"]
        }

        # 彩色可视化配置
        self.setup_colors()

        # 检测参数
        self.box_threshold = 0.15
        self.text_threshold = 0.15
        self.min_confidence = 0.15

    def load_grounding_dino(self, config_path, checkpoint_path):
        """加载Grounding DINO模型"""
        print("⏳ 加载Grounding DINO模型...")
        try:
            self.grounding_dino_model = load_model(config_path, checkpoint_path, device=self.device)
            print("✅ Grounding DINO模型加载成功")
        except Exception as e:
            print(f"❌ Grounding DINO模型加载失败: {e}")
            raise e

    def load_sam(self, sam_checkpoint):
        """加载SAM模型"""
        print("⏳ 加载SAM模型...")
        try:
            if "vit_h" in sam_checkpoint:
                model_type = "vit_h"
            elif "vit_l" in sam_checkpoint:
                model_type = "vit_l"
            else:
                model_type = "vit_b"

            print(f"   使用SAM模型类型: {model_type}")
            sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
            sam.to(device=self.device)
            self.sam_predictor = SamPredictor(sam)
            print("✅ SAM模型加载成功")
        except Exception as e:
            print(f"❌ SAM模型加载失败: {e}")
            self.sam_predictor = None

    def setup_colors(self):
        """设置彩色可视化配置"""
        # 动态对象颜色（暖色调）
        self.dynamic_colors = {
            'person': [0, 0, 255],  # 红色
            'people': [0, 0, 255],
            'pedestrian': [0, 0, 255],

            'car': [0, 165, 255],  # 橙色
            'vehicle': [0, 165, 255],
            'automobile': [0, 165, 255],

            'truck': [0, 255, 255],  # 黄色
            'bus': [255, 0, 255],  # 品红色
            'motorcycle': [255, 0, 128],  # 粉红色
            'bicycle': [128, 0, 255],  # 紫色
            'bike': [128, 0, 255],
        }

        # 静态对象颜色（冷色调）
        self.static_colors = {
            'building': [128, 128, 64],  # 橄榄色
            'house': [128, 128, 64],
            'wall': [96, 96, 96],  # 灰色
            'road': [64, 64, 64],  # 深灰色
            'street': [64, 64, 64],
            'sidewalk': [160, 160, 160],  # 浅灰色
            'pavement': [128, 128, 128],

            'tree': [0, 100, 0],  # 深绿色
            'fence': [150, 150, 0],  # 深黄色
            'pole': [100, 50, 0],  # 棕色
            'traffic light': [0, 255, 0],  # 绿色
            'street lamp': [100, 50, 0],
        }

        self.color_cache = {}
        self.class_count = 0

    def get_color_for_class(self, class_name, is_dynamic=True):
        """为类别获取颜色（BGR格式）"""
        if class_name is None:
            class_name = f"{'dynamic' if is_dynamic else 'static'}_unknown_{self.class_count}"

        class_name = str(class_name).lower().strip()

        # 选择预定义颜色
        predefined_colors = self.dynamic_colors if is_dynamic else self.static_colors

        # 精确匹配
        if class_name in predefined_colors:
            return predefined_colors[class_name]

        # 部分匹配
        for key, color in predefined_colors.items():
            if key in class_name or class_name in key:
                return color

        # 检查缓存
        cache_key = f"{class_name}_{is_dynamic}"
        if cache_key in self.color_cache:
            return self.color_cache[cache_key]

        # 生成新颜色
        if is_dynamic:
            hue_base = 0  # 红色基础
            hue_range = 60  # 到黄色的范围
        else:
            hue_base = 180  # 青色基础
            hue_range = 120  # 到绿色的范围

        hue = (hue_base + (self.class_count * 137.5) % hue_range) % 360
        saturation = 0.6 + (self.class_count % 4) * 0.1
        value = 0.7 + (self.class_count % 3) * 0.15

        rgb = colorsys.hsv_to_rgb(hue / 360, saturation, value)
        bgr = [int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255)]

        self.color_cache[cache_key] = bgr
        self.class_count += 1
        return bgr

    # ================================
    # 🚀 批量处理功能
    # ================================

    def detect_folder_images(self, input_folder, output_base_dir="./batch_detection_results"):
        """
        批量检测文件夹中的所有图片，为每张图片创建独立的结果文件夹

        Args:
            input_folder: 输入图片文件夹路径
            output_base_dir: 输出基础目录路径
        """
        print("🚀 开始批量检测文件夹图片")
        print("=" * 60)

        # 支持的图片格式
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']

        # 获取所有图片文件
        image_files = []
        input_path = Path(input_folder)

        for ext in image_extensions:
            image_files.extend(glob.glob(str(input_path / ext)))
            image_files.extend(glob.glob(str(input_path / ext.upper())))

        if not image_files:
            print(f"❌ 在文件夹 {input_folder} 中未找到任何图片文件")
            return []

        print(f"📁 找到 {len(image_files)} 张图片")
        print(f"📂 输出目录: {output_base_dir}")

        # 创建输出基础目录
        output_base_path = Path(output_base_dir)
        output_base_path.mkdir(parents=True, exist_ok=True)

        # 批量检测统计
        batch_results = {
            'total_images': len(image_files),
            'processed_images': 0,
            'failed_images': 0,
            'total_detections': 0,
            'total_dynamic': 0,
            'total_static': 0,
            'total_sam_success': 0,
            'processing_time': 0,
            'results_per_image': []
        }

        start_time = datetime.now()

        # 处理每张图片
        for i, image_path in enumerate(image_files, 1):
            image_name = Path(image_path).stem
            print(f"\n🔍 处理图片 [{i}/{len(image_files)}]: {image_name}")
            print("-" * 50)

            try:
                # 为每张图片创建独立的输出文件夹
                image_output_dir = output_base_path / image_name
                image_output_dir.mkdir(parents=True, exist_ok=True)

                # 检测当前图片
                detections = self.detect_objects_with_sam(image_path, str(image_output_dir))

                # 统计当前图片的检测结果
                dynamic_count = len([d for d in detections if d['type'] == 'dynamic'])
                static_count = len([d for d in detections if d['type'] == 'static'])
                sam_count = len([d for d in detections if d.get('sam_mask') is not None])

                # 保存当前图片的检测结果到JSON
                image_result = {
                    'image_name': image_name,
                    'image_path': image_path,
                    'output_dir': str(image_output_dir),
                    'total_detections': len(detections),
                    'dynamic_count': dynamic_count,
                    'static_count': static_count,
                    'sam_success_count': sam_count,
                    'detections': []
                }

                # 详细检测结果
                for det in detections:
                    detection_info = {
                        'phrase': det['phrase'],
                        'type': det['type'],
                        'confidence': float(det['confidence']),
                        'bbox': det['box_xyxy'].cpu().numpy().tolist(),
                        'sam_success': det.get('sam_success', False)
                    }
                    image_result['detections'].append(detection_info)

                # 保存JSON结果
                json_path = image_output_dir / f"{image_name}_detection_results.json"
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(image_result, f, indent=2, ensure_ascii=False)

                # 更新批量统计
                batch_results['processed_images'] += 1
                batch_results['total_detections'] += len(detections)
                batch_results['total_dynamic'] += dynamic_count
                batch_results['total_static'] += static_count
                batch_results['total_sam_success'] += sam_count
                batch_results['results_per_image'].append(image_result)

                print(f"✅ {image_name} 处理完成:")
                print(f"   📊 检测总数: {len(detections)}")
                print(f"   🔴 动态物体: {dynamic_count}")
                print(f"   🟢 静态物体: {static_count}")
                print(f"   🎯 SAM成功: {sam_count}")
                print(f"   💾 结果保存至: {image_output_dir}")

            except Exception as e:
                print(f"❌ 处理 {image_name} 时出错: {e}")
                batch_results['failed_images'] += 1
                continue

        # 计算总处理时间
        end_time = datetime.now()
        batch_results['processing_time'] = (end_time - start_time).total_seconds()

        # 保存批量处理总结报告
        self._save_batch_summary(batch_results, output_base_path, start_time, end_time)

        # 创建批量可视化概览
        self._create_batch_overview(batch_results, output_base_path)

        print("\n" + "=" * 60)
        print("🎉 批量检测完成!")
        print(f"📊 处理统计:")
        print(f"   📷 总图片数: {batch_results['total_images']}")
        print(f"   ✅ 成功处理: {batch_results['processed_images']}")
        print(f"   ❌ 处理失败: {batch_results['failed_images']}")
        print(f"   🔍 总检测数: {batch_results['total_detections']}")
        print(f"   🔴 总动态物体: {batch_results['total_dynamic']}")
        print(f"   🟢 总静态物体: {batch_results['total_static']}")
        print(f"   🎯 总SAM成功: {batch_results['total_sam_success']}")
        print(f"   ⏱️  总耗时: {batch_results['processing_time']:.2f} 秒")
        print(f"   📁 结果目录: {output_base_path}")

        return batch_results

    def _save_batch_summary(self, batch_results, output_base_path, start_time, end_time):
        """保存批量处理的总结报告"""

        # 保存详细的JSON报告
        summary_json = {
            **batch_results,
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'average_time_per_image': batch_results['processing_time'] / max(1, batch_results['processed_images']),
            'success_rate': batch_results['processed_images'] / batch_results['total_images'] * 100
        }

        json_summary_path = output_base_path / "batch_processing_summary.json"
        with open(json_summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_json, f, indent=2, ensure_ascii=False)

        # 创建可读的文本报告
        txt_summary_path = output_base_path / "batch_processing_report.txt"
        with open(txt_summary_path, 'w', encoding='utf-8') as f:
            f.write("🚀 批量图片检测处理报告\n")
            f.write("=" * 50 + "\n\n")
            f.write(
                f"处理时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')} - {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总处理时长: {batch_results['processing_time']:.2f} 秒\n")
            f.write(
                f"平均每张图片: {batch_results['processing_time'] / max(1, batch_results['processed_images']):.2f} 秒\n\n")

            f.write("📊 处理统计:\n")
            f.write(f"  总图片数量: {batch_results['total_images']}\n")
            f.write(f"  成功处理: {batch_results['processed_images']}\n")
            f.write(f"  处理失败: {batch_results['failed_images']}\n")
            f.write(f"  成功率: {batch_results['processed_images'] / batch_results['total_images'] * 100:.1f}%\n\n")

            f.write("🔍 检测统计:\n")
            f.write(f"  总检测数量: {batch_results['total_detections']}\n")
            f.write(f"  动态物体: {batch_results['total_dynamic']}\n")
            f.write(f"  静态物体: {batch_results['total_static']}\n")
            f.write(f"  SAM分割成功: {batch_results['total_sam_success']}\n")
            f.write(
                f"  平均每张图片检测数: {batch_results['total_detections'] / max(1, batch_results['processed_images']):.1f}\n\n")

            f.write("📷 各图片详细结果:\n")
            f.write("-" * 50 + "\n")
            for result in batch_results['results_per_image']:
                f.write(f"图片: {result['image_name']}\n")
                f.write(
                    f"  总检测: {result['total_detections']} (动态: {result['dynamic_count']}, 静态: {result['static_count']})\n")
                f.write(f"  SAM成功: {result['sam_success_count']}\n")
                f.write(f"  输出目录: {result['output_dir']}\n\n")

        print(f"📋 批量处理报告已保存:")
        print(f"   📄 JSON报告: {json_summary_path}")
        print(f"   📄 文本报告: {txt_summary_path}")

    def _create_batch_overview(self, batch_results, output_base_path):
        """创建批量处理的可视化概览"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            # 设置中文字体（如果有的话）
            plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False

            # 创建概览图
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('批量图片检测处理概览', fontsize=16, fontweight='bold')

            # 1. 处理成功率饼图
            success_data = [batch_results['processed_images'], batch_results['failed_images']]
            success_labels = ['成功处理', '处理失败']
            colors1 = ['#2ecc71', '#e74c3c']
            ax1.pie(success_data, labels=success_labels, autopct='%1.1f%%', colors=colors1, startangle=90)
            ax1.set_title('处理成功率')

            # 2. 检测类型分布
            detection_data = [batch_results['total_dynamic'], batch_results['total_static']]
            detection_labels = ['动态物体', '静态物体']
            colors2 = ['#e67e22', '#3498db']
            ax2.pie(detection_data, labels=detection_labels, autopct='%1.1f%%', colors=colors2, startangle=90)
            ax2.set_title('检测物体类型分布')

            # 3. 每张图片检测数量分布
            if batch_results['results_per_image']:
                detection_counts = [r['total_detections'] for r in batch_results['results_per_image']]
                ax3.hist(detection_counts, bins=10, color='#9b59b6', alpha=0.7, edgecolor='black')
                ax3.set_xlabel('检测数量')
                ax3.set_ylabel('图片数量')
                ax3.set_title('每张图片检测数量分布')
                ax3.grid(True, alpha=0.3)

            # 4. SAM成功率
            if batch_results['total_detections'] > 0:
                sam_success_rate = batch_results['total_sam_success'] / batch_results['total_detections'] * 100
                sam_fail_rate = 100 - sam_success_rate
                sam_data = [sam_success_rate, sam_fail_rate]
                sam_labels = ['SAM成功', 'SAM失败']
                colors4 = ['#1abc9c', '#95a5a6']
                ax4.pie(sam_data, labels=sam_labels, autopct='%1.1f%%', colors=colors4, startangle=90)
                ax4.set_title('SAM分割成功率')
            else:
                ax4.text(0.5, 0.5, '无检测数据', ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('SAM分割成功率')

            plt.tight_layout()

            # 保存概览图
            overview_path = output_base_path / "batch_processing_overview.png"
            plt.savefig(overview_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"   📊 可视化概览: {overview_path}")

        except ImportError:
            print("   ⚠️  matplotlib未安装，跳过可视化概览生成")
        except Exception as e:
            print(f"   ⚠️  生成可视化概览时出错: {e}")

    # ================================
    # 原有的所有方法保持不变
    # ================================

    def create_mask_overlay_on_original(self, image_path, detections, save_dir="./", image_name="result"):
        """在原图上创建mask叠加可视化（保持原图背景）"""

        # 读取原图
        original_image = cv2.imread(image_path)
        if original_image is None:
            print(f"❌ 无法读取图像: {image_path}")
            return None

        h, w, _ = original_image.shape
        base_name = os.path.splitext(image_name)[0]

        # 创建保存目录
        overlay_dir = os.path.join(save_dir, "mask_overlays")
        os.makedirs(overlay_dir, exist_ok=True)

        print("🎨 在原图上创建mask叠加可视化...")

        # 创建不同版本的叠加图像（都基于原图）
        complete_overlay = original_image.copy()
        dynamic_overlay = original_image.copy()
        static_overlay = original_image.copy()

        # 透明度设置
        alpha_objects = 0.4

        # 统计计数
        dynamic_count = 0
        static_count = 0

        # 处理每个检测结果
        for det in detections:
            obj_type = det['type']
            phrase = det['phrase']
            confidence = det['confidence']
            sam_mask = det.get('sam_mask')
            box_xyxy = det['box_xyxy']

            # 获取颜色
            color = self.get_color_for_class(phrase, is_dynamic=(obj_type == 'dynamic'))

            # 创建mask
            if sam_mask is not None:
                # 使用SAM精确mask
                mask_bool = sam_mask.astype(bool)
            else:
                # 使用边界框作为mask
                x1, y1, x2, y2 = box_xyxy.cpu().numpy().astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                mask_bool = np.zeros((h, w), dtype=bool)
                mask_bool[y1:y2, x1:x2] = True

            # 创建彩色叠加层
            overlay_layer = np.zeros_like(original_image)
            overlay_layer[mask_bool] = color

            # 在相应的图像上叠加（保持原图背景）
            complete_overlay = cv2.addWeighted(complete_overlay, 1 - alpha_objects, overlay_layer, alpha_objects, 0)

            if obj_type == 'dynamic':
                dynamic_overlay = cv2.addWeighted(dynamic_overlay, 1 - alpha_objects, overlay_layer, alpha_objects, 0)
                dynamic_count += 1
            else:
                static_overlay = cv2.addWeighted(static_overlay, 1 - alpha_objects, overlay_layer, alpha_objects, 0)
                static_count += 1

            # 在完整叠加图上添加边界框和标签
            # x1, y1, x2, y2 = box_xyxy.cpu().numpy().astype(int)
            # thickness = 3 if obj_type == 'dynamic' else 2
            # line_color = tuple(int(c * 0.8) for c in color)
            #
            # cv2.rectangle(complete_overlay, (x1, y1), (x2, y2), line_color, thickness)

            # 添加标签
            label = f"[{'D' if obj_type == 'dynamic' else 'S'}] {phrase}: {confidence:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 2

            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)

            # 文本背景
            # bg_color = tuple(int(c * 0.6) for c in color)
            # cv2.rectangle(complete_overlay, (x1, y1 - text_height - 10),
            #               (x1 + text_width + 10, y1), bg_color, -1)

            # 文本
            # cv2.putText(complete_overlay, label, (x1 + 5, y1 - 5),
            #             font, font_scale, (255, 255, 255), font_thickness)

        # 添加统计信息
        self._add_overlay_statistics(complete_overlay, dynamic_count, static_count, "COMPLETE OVERLAY")
        self._add_overlay_statistics(dynamic_overlay, dynamic_count, 0, "DYNAMIC OBJECTS")
        self._add_overlay_statistics(static_overlay, 0, static_count, "STATIC OBJECTS")

        # 保存结果
        complete_path = os.path.join(overlay_dir, f"{base_name}_complete_overlay.jpg")
        dynamic_path = os.path.join(overlay_dir, f"{base_name}_dynamic_overlay.jpg")
        static_path = os.path.join(overlay_dir, f"{base_name}_static_overlay.jpg")

        cv2.imwrite(complete_path, complete_overlay)
        cv2.imwrite(dynamic_path, dynamic_overlay)
        cv2.imwrite(static_path, static_overlay)

        # 创建2x2对比网格
        self._create_overlay_grid(original_image, dynamic_overlay, static_overlay,
                                  complete_overlay, overlay_dir, base_name)

        print(f"✅ Mask叠加可视化完成:")
        print(f"   🌈 完整叠加: {complete_path}")
        print(f"   🔴 动态叠加: {dynamic_path}")
        print(f"   🟢 静态叠加: {static_path}")

        return complete_overlay, dynamic_overlay, static_overlay

    def _add_overlay_statistics(self, image, dynamic_count, static_count, title):
        """为叠加图像添加统计信息"""
        h, w = image.shape[:2]

        stats_lines = [
            title,
            f"Dynamic: {dynamic_count}",
            f"Static: {static_count}",
            f"Total: {dynamic_count + static_count}"
        ]

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 2
        line_height = 30

        # 计算背景大小
        max_width = 0
        for line in stats_lines:
            (text_width, _), _ = cv2.getTextSize(line, font, font_scale, font_thickness)
            max_width = max(max_width, text_width)

        bg_height = len(stats_lines) * line_height + 20

        # 半透明背景
        # overlay = image.copy()
        # cv2.rectangle(overlay, (10, 10), (max_width + 30, bg_height), (0, 0, 0), -1)
        # cv2.addWeighted(image, 0.7, overlay, 0.3, 0, image)

        # 边框
        # cv2.rectangle(image, (10, 10), (max_width + 30, bg_height), (255, 255, 255), 2)

        # 文本
        # for i, line in enumerate(stats_lines):
        #     y = 35 + i * line_height
        #     color = (255, 255, 255) if i == 0 else (0, 255, 255)
        #     cv2.putText(image, line, (20, y), font, font_scale, color, font_thickness)

    def _create_overlay_grid(self, original, dynamic_overlay, static_overlay, complete_overlay, save_dir, base_name):
        """创建2x2对比网格"""

        # 调整所有图像到相同大小
        h, w = original.shape[:2]
        target_size = (w // 2, h // 2)  # 缩小一半

        # 调整大小并添加标题
        def resize_and_add_title(img, title):
            resized = cv2.resize(img, target_size)
            cv2.putText(resized, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 255), 3)
            cv2.putText(resized, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 0), 2)
            return resized

        orig_small = resize_and_add_title(original, "Original")
        dyn_small = resize_and_add_title(dynamic_overlay, "Dynamic Objects")
        stat_small = resize_and_add_title(static_overlay, "Static Objects")
        comp_small = resize_and_add_title(complete_overlay, "Complete")

        # 拼接
        top_row = np.hstack([orig_small, dyn_small])
        bottom_row = np.hstack([stat_small, comp_small])
        grid = np.vstack([top_row, bottom_row])

        # 保存网格
        grid_path = os.path.join(save_dir, f"{base_name}_overlay_grid.jpg")
        cv2.imwrite(grid_path, grid)
        print(f"   📊 对比网格: {grid_path}")

        return grid

    def create_prompts(self):
        """创建检测提示词"""
        # 动态物体提示词
        dynamic_prompt = " . ".join(
            self.dynamic_objects["primary"] +
            self.dynamic_objects["secondary"][:3]
        ) + " ."

        # 静态物体提示词
        static_prompt = " . ".join(
            self.static_objects["primary"] +
            self.static_objects["secondary"][:3]
        ) + " ."

        return dynamic_prompt, static_prompt

    def classify_detection(self, phrase):
        """分类检测结果是动态还是静态"""
        phrase_lower = phrase.lower().strip()

        # 检查动态物体
        all_dynamic = (self.dynamic_objects["primary"] +
                       self.dynamic_objects["secondary"] +
                       self.dynamic_objects["specific"])

        for dynamic_word in all_dynamic:
            if dynamic_word.lower() in phrase_lower:
                return "dynamic"

        # 检查静态物体
        all_static = (self.static_objects["primary"] +
                      self.static_objects["secondary"] +
                      self.static_objects["specific"])

        for static_word in all_static:
            if static_word.lower() in phrase_lower:
                return "static"

        # 默认分类逻辑
        moving_keywords = ["car", "truck", "bus", "person", "bike", "vehicle", "people"]
        static_keywords = ["building", "tree", "road", "wall", "house", "street"]

        if any(keyword in phrase_lower for keyword in moving_keywords):
            return "dynamic"
        elif any(keyword in phrase_lower for keyword in static_keywords):
            return "static"

        return "unknown"

    def detect_objects_with_sam(self, image_path, save_dir="./detection_results"):
        """检测图像中的动态和静态物体，并使用SAM进行精确分割"""
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, "masks"), exist_ok=True)

        # 加载图像
        print(f"📷 加载图像: {image_path}")
        image_source, image = load_image(image_path)
        h, w, _ = image_source.shape
        print(f"   图像尺寸: {w}x{h}")

        # 设置SAM图像
        if self.sam_predictor:
            self.sam_predictor.set_image(image_source)
            print("🎯 SAM图像设置完成")

        # 创建提示词
        dynamic_prompt, static_prompt = self.create_prompts()
        print(f"🎯 动态物体提示词: {dynamic_prompt[:50]}...")
        print(f"🏗️  静态物体提示词: {static_prompt[:50]}...")

        all_detections = []

        # 检测动态物体
        print("\n🔍 检测动态物体...")
        dynamic_detections = self._detect_with_prompt_and_sam(image, dynamic_prompt, "dynamic", w, h)
        all_detections.extend(dynamic_detections)

        # 检测静态物体
        print("🔍 检测静态物体...")
        static_detections = self._detect_with_prompt_and_sam(image, static_prompt, "static", w, h)
        all_detections.extend(static_detections)

        # 后处理和去重
        filtered_detections = self._post_process_detections_with_sam(all_detections, w, h)

        # 分类统计
        dynamic_count = len([d for d in filtered_detections if d['type'] == 'dynamic'])
        static_count = len([d for d in filtered_detections if d['type'] == 'static'])
        sam_count = len([d for d in filtered_detections if d.get('sam_mask') is not None])

        print(f"\n📊 检测结果统计:")
        print(f"   动态物体: {dynamic_count} 个")
        print(f"   静态物体: {static_count} 个")
        print(f"   总计: {len(filtered_detections)} 个")
        print(f"   SAM分割成功: {sam_count} 个")

        # 🎨 创建可视化
        self._create_comprehensive_visualization(image_source, filtered_detections, save_dir,
                                                 os.path.basename(image_path))

        # 🎨 创建原版mask叠加可视化（在原图上）
        print("\n🎨 创建mask叠加可视化...")
        overlay_results = self.create_mask_overlay_on_original(
            image_path, filtered_detections, save_dir, os.path.basename(image_path)
        )

        # 🎨 创建增强的mask叠加可视化（新增）
        print("\n🎨 创建增强mask叠加可视化...")
        enhanced_results = self.create_enhanced_mask_overlay(
            image_path, filtered_detections, save_dir, os.path.basename(image_path)
        )

        # 🎨 创建纯轮廓叠加可视化（新增）
        print("🎨 创建轮廓叠加可视化...")
        contour_result = self.create_contour_only_overlay(
            image_path, filtered_detections, save_dir, os.path.basename(image_path)
        )

        print("\n✅ 所有可视化完成!")
        print("📁 生成的文件夹:")
        print("   📂 mask_overlays/ - 原版mask叠加")
        print("   📂 enhanced_overlays/ - 增强mask叠加 (推荐)")
        print("   📂 contour_overlays/ - 纯轮廓叠加")
        print("   📂 masks/ - 单独的mask文件")

        return filtered_detections

    # 改进的mask在原图上的可视化方法

    def create_enhanced_mask_overlay(self, image_path, detections, save_dir="./", image_name="result"):
        """改进的在原图上创建mask叠加可视化（更清晰的原图背景）"""

        # 读取原图
        original_image = cv2.imread(image_path)
        if original_image is None:
            print(f"❌ 无法读取图像: {image_path}")
            return None

        h, w, _ = original_image.shape
        base_name = os.path.splitext(image_name)[0]

        # 创建保存目录
        overlay_dir = os.path.join(save_dir, "enhanced_overlays")
        os.makedirs(overlay_dir, exist_ok=True)

        print("🎨 创建改进的mask叠加可视化...")

        # 创建不同版本的叠加图像（都基于原图）
        complete_overlay = original_image.copy()
        dynamic_overlay = original_image.copy()
        static_overlay = original_image.copy()

        # 🎯 关键改进：更好的透明度设置
        alpha_mask = 0.3  # mask透明度降低，让原图更清晰
        alpha_original = 0.9  # 原图透明度提高

        # 🎨 改进的颜色方案（更亮的颜色）
        enhanced_dynamic_colors = {
            'person': [0, 100, 255],  # 亮红色
            'car': [0, 200, 255],  # 亮橙色
            'truck': [0, 255, 255],  # 黄色
            'bus': [128, 0, 255],  # 紫色
            'bicycle': [255, 100, 255],  # 亮粉色
            'motorcycle': [255, 150, 0],  # 蓝橙色
        }

        enhanced_static_colors = {
            'building': [200, 200, 100],  # 亮橄榄色
            'tree': [100, 255, 100],  # 亮绿色
            'road': [150, 150, 150],  # 亮灰色
            'wall': [180, 180, 180],  # 浅灰色
            'fence': [200, 200, 0],  # 亮黄色
            'pole': [150, 100, 50],  # 亮棕色
        }

        # 统计计数
        dynamic_count = 0
        static_count = 0

        # 处理每个检测结果
        for det in detections:
            obj_type = det['type']
            phrase = det['phrase']
            confidence = det['confidence']
            sam_mask = det.get('sam_mask')
            box_xyxy = det['box_xyxy']

            # 🎨 获取增强的颜色
            phrase_lower = phrase.lower()
            if obj_type == 'dynamic':
                color = None
                for key, enhanced_color in enhanced_dynamic_colors.items():
                    if key in phrase_lower:
                        color = enhanced_color
                        break
                if color is None:
                    color = self.get_color_for_class(phrase, is_dynamic=True)
                    # 增亮颜色
                    color = [min(255, int(c * 1.5)) for c in color]
            else:
                color = None
                for key, enhanced_color in enhanced_static_colors.items():
                    if key in phrase_lower:
                        color = enhanced_color
                        break
                if color is None:
                    color = self.get_color_for_class(phrase, is_dynamic=False)
                    # 增亮颜色
                    color = [min(255, int(c * 1.3)) for c in color]

            # 创建mask
            if sam_mask is not None:
                # 使用SAM精确mask
                mask_bool = sam_mask.astype(bool)
            else:
                # 使用边界框作为mask
                x1, y1, x2, y2 = box_xyxy.cpu().numpy().astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                mask_bool = np.zeros((h, w), dtype=bool)
                mask_bool[y1:y2, x1:x2] = True

            # 🎯 改进的叠加方法：只对mask区域进行混合
            mask_area = np.zeros_like(original_image)
            mask_area[mask_bool] = color

            # 只在mask区域进行颜色混合，保持原图其他区域不变
            mask_indices = np.where(mask_bool)
            if len(mask_indices[0]) > 0:
                # 在完整叠加图上应用
                complete_overlay[mask_indices] = (
                        alpha_original * complete_overlay[mask_indices] +
                        alpha_mask * np.array(color)
                ).astype(np.uint8)

                if obj_type == 'dynamic':
                    dynamic_overlay[mask_indices] = (
                            alpha_original * dynamic_overlay[mask_indices] +
                            alpha_mask * np.array(color)
                    ).astype(np.uint8)
                    dynamic_count += 1
                else:
                    static_overlay[mask_indices] = (
                            alpha_original * static_overlay[mask_indices] +
                            alpha_mask * np.array(color)
                    ).astype(np.uint8)
                    static_count += 1

            # 🎯 添加轮廓线增强可见性
            if sam_mask is not None:
                # 找到mask轮廓
                contours, _ = cv2.findContours(sam_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # 绘制轮廓
                cv2.drawContours(complete_overlay, contours, -1, color, 2)
                if obj_type == 'dynamic':
                    cv2.drawContours(dynamic_overlay, contours, -1, color, 2)
                else:
                    cv2.drawContours(static_overlay, contours, -1, color, 2)

            # 添加边界框（细线）
            # x1, y1, x2, y2 = box_xyxy.cpu().numpy().astype(int)
            # thickness = 2 if obj_type == 'dynamic' else 1
            # line_color = tuple(int(c * 0.8) for c in color)
            #
            # cv2.rectangle(complete_overlay, (x1, y1), (x2, y2), line_color, thickness)

            # 🎯 改进的标签显示
            label = f"{phrase}: {confidence:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1

            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)

            # 半透明文本背景
            # overlay_bg = complete_overlay.copy()
            # cv2.rectangle(overlay_bg, (x1, y1 - text_height - 8),
            #               (x1 + text_width + 8, y1), color, -1)
            # cv2.addWeighted(complete_overlay, 0.7, overlay_bg, 0.3, 0, complete_overlay)

            # 白色文本
            # cv2.putText(complete_overlay, label, (x1 + 4, y1 - 4),
            #             font, font_scale, (255, 255, 255), font_thickness)

        # 添加改进的统计信息
        self._add_enhanced_statistics(complete_overlay, dynamic_count, static_count, "ENHANCED OVERLAY")
        self._add_enhanced_statistics(dynamic_overlay, dynamic_count, 0, "DYNAMIC OBJECTS")
        self._add_enhanced_statistics(static_overlay, 0, static_count, "STATIC OBJECTS")

        # 保存结果
        complete_path = os.path.join(overlay_dir, f"{base_name}_enhanced_complete.jpg")
        dynamic_path = os.path.join(overlay_dir, f"{base_name}_enhanced_dynamic.jpg")
        static_path = os.path.join(overlay_dir, f"{base_name}_enhanced_static.jpg")

        cv2.imwrite(complete_path, complete_overlay)
        cv2.imwrite(dynamic_path, dynamic_overlay)
        cv2.imwrite(static_path, static_overlay)

        # 🎯 创建原图对比
        self._create_before_after_comparison(original_image, complete_overlay, overlay_dir, base_name)

        print(f"✅ 增强mask叠加可视化完成:")
        print(f"   🌈 增强完整: {complete_path}")
        print(f"   🔴 增强动态: {dynamic_path}")
        print(f"   🟢 增强静态: {static_path}")

        return complete_overlay, dynamic_overlay, static_overlay

    def _add_enhanced_statistics(self, image, dynamic_count, static_count, title):
        """添加增强的统计信息（更清晰的显示）"""
        h, w = image.shape[:2]

        stats_lines = [
            title,
            f"Dynamic: {dynamic_count}",
            f"Static: {static_count}",
            f"Total: {dynamic_count + static_count}"
        ]

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        line_height = 25

        # 计算背景大小
        max_width = 0
        for line in stats_lines:
            (text_width, _), _ = cv2.getTextSize(line, font, font_scale, font_thickness)
            max_width = max(max_width, text_width)

        bg_height = len(stats_lines) * line_height + 15

        # 🎯 更清晰的半透明背景
        # overlay = image.copy()
        # cv2.rectangle(overlay, (10, 10), (max_width + 25, bg_height), (0, 0, 0), -1)
        # cv2.addWeighted(image, 0.8, overlay, 0.2, 0, image)  # 降低背景透明度

        # 亮色边框
        # cv2.rectangle(image, (10, 10), (max_width + 25, bg_height), (255, 255, 255), 2)

        # 文本
        # for i, line in enumerate(stats_lines):
        #     y = 30 + i * line_height
        #     color = (255, 255, 255) if i == 0 else (0, 255, 255)
        #     cv2.putText(image, line, (15, y), font, font_scale, color, font_thickness)

    def _create_before_after_comparison(self, original, overlay_result, save_dir, base_name):
        """创建原图与叠加结果的对比"""

        # 调整大小
        h, w = original.shape[:2]
        target_size = (w // 2, h // 2)

        orig_small = cv2.resize(original, target_size)
        overlay_small = cv2.resize(overlay_result, target_size)

        # 添加标题
        # cv2.putText(orig_small, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
        #             1.0, (255, 255, 255), 3)
        # cv2.putText(orig_small, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
        #             1.0, (0, 0, 0), 2)
        #
        # cv2.putText(overlay_small, "With Masks", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
        #             1.0, (255, 255, 255), 3)
        # cv2.putText(overlay_small, "With Masks", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
        #             1.0, (0, 0, 0), 2)

        # 水平拼接
        comparison = np.hstack([orig_small, overlay_small])

        # 保存对比图
        comparison_path = os.path.join(save_dir, f"{base_name}_before_after.jpg")
        cv2.imwrite(comparison_path, comparison)
        print(f"   📊 前后对比: {comparison_path}")

        return comparison

    # 🎯 另一种纯轮廓显示方法
    def create_contour_only_overlay(self, image_path, detections, save_dir="./", image_name="result"):
        """创建只显示轮廓的叠加（保持原图完全清晰）"""

        original_image = cv2.imread(image_path)
        if original_image is None:
            return None

        h, w, _ = original_image.shape
        base_name = os.path.splitext(image_name)[0]

        # 创建保存目录
        contour_dir = os.path.join(save_dir, "contour_overlays")
        os.makedirs(contour_dir, exist_ok=True)

        print("🎨 创建轮廓叠加可视化...")

        # 完全保持原图不变，只添加轮廓
        contour_overlay = original_image.copy()

        for det in detections:
            obj_type = det['type']
            phrase = det['phrase']
            confidence = det['confidence']
            sam_mask = det.get('sam_mask')
            box_xyxy = det['box_xyxy']

            # 获取颜色
            color = self.get_color_for_class(phrase, is_dynamic=(obj_type == 'dynamic'))
            # 增强颜色亮度
            color = [min(255, int(c * 1.8)) for c in color]

            # 绘制轮廓
            if sam_mask is not None:
                contours, _ = cv2.findContours(sam_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # 粗轮廓
                cv2.drawContours(contour_overlay, contours, -1, color, 3)
                # 细内轮廓增强对比
                cv2.drawContours(contour_overlay, contours, -1, (255, 255, 255), 1)
            # else:
            #     # 边界框轮廓
            #     # x1, y1, x2, y2 = box_xyxy.cpu().numpy().astype(int)
            #     # cv2.rectangle(contour_overlay, (x1, y1), (x2, y2), color, 3)
            #     # cv2.rectangle(contour_overlay, (x1 + 1, y1 + 1), (x2 - 1, y2 - 1), (255, 255, 255), 1)

            # 简洁标签
            x1, y1, x2, y2 = box_xyxy.cpu().numpy().astype(int)
            label = f"{phrase}"

            # 标签背景
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1

            (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, font_thickness)

            # 半透明标签背景
            # label_bg = contour_overlay.copy()
            # cv2.rectangle(label_bg, (x1, y1 - text_height - 6), (x1 + text_width + 6, y1), color, -1)
            # cv2.addWeighted(contour_overlay, 0.8, label_bg, 0.2, 0, contour_overlay)

            # 白色文字
            # cv2.putText(contour_overlay, label, (x1 + 3, y1 - 3), font, font_scale, (255, 255, 255), font_thickness)

        # 保存轮廓叠加
        contour_path = os.path.join(contour_dir, f"{base_name}_contour_only.jpg")
        cv2.imwrite(contour_path, contour_overlay)

        print(f"✅ 轮廓叠加完成: {contour_path}")
        return contour_overlay

    def _detect_with_prompt_and_sam(self, image, prompt, object_type, w, h):
        """使用指定提示词检测物体并应用SAM分割"""
        detections = []

        try:
            # Grounding DINO检测
            with torch.no_grad():
                boxes, logits, phrases = predict(
                    model=self.grounding_dino_model,
                    image=image,
                    caption=prompt,
                    box_threshold=self.box_threshold,
                    text_threshold=self.text_threshold,
                    device=self.device
                )

            print(f"   Grounding DINO检测到 {len(boxes)} 个{object_type}物体")

            # 处理每个检测结果
            for i, (box, logit, phrase) in enumerate(zip(boxes, logits, phrases)):
                confidence = logit.item()

                # 置信度过滤
                if confidence < self.min_confidence:
                    continue

                # 转换坐标
                box_xyxy = box_ops.box_cxcywh_to_xyxy(box.unsqueeze(0)) * torch.tensor([w, h, w, h])
                x1, y1, x2, y2 = box_xyxy[0].cpu().numpy().astype(int)

                # 确保坐标在图像范围内
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                # 重新分类
                actual_type = self.classify_detection(phrase)
                if actual_type == "unknown":
                    actual_type = object_type

                detection = {
                    'box': box,
                    'box_xyxy': box_xyxy[0],
                    'confidence': confidence,
                    'phrase': phrase,
                    'type': actual_type,
                    'original_type': object_type,
                    'sam_mask': None,
                    'sam_success': False
                }

                # 🎯 SAM分割
                if self.sam_predictor:
                    try:
                        input_box = np.array([x1, y1, x2, y2])
                        masks, sam_scores, _ = self.sam_predictor.predict(
                            point_coords=None,
                            point_labels=None,
                            box=input_box[None, :],
                            multimask_output=False,
                        )

                        if len(masks) > 0 and masks[0] is not None:
                            sam_mask = masks[0].astype(np.uint8)
                            detection['sam_mask'] = sam_mask
                            detection['sam_success'] = True
                            detection['sam_score'] = sam_scores[0] if len(sam_scores) > 0 else 0.0
                            print(f"   ✅ SAM分割成功: {phrase} (mask pixels: {np.sum(sam_mask)})")
                        else:
                            print(f"   ❌ SAM分割失败: {phrase}")

                    except Exception as e:
                        print(f"   ❌ SAM分割出错: {phrase} - {e}")

                detections.append(detection)

            # 按置信度排序
            detections.sort(key=lambda x: x['confidence'], reverse=True)
            print(f"   过滤后保留 {len(detections)} 个{object_type}物体")

        except Exception as e:
            print(f"❌ {object_type}物体检测失败: {e}")

        return detections

    def _post_process_detections_with_sam(self, all_detections, w, h):
        """后处理检测结果，包括SAM mask的NMS"""
        if not all_detections:
            return []

        # 尺寸过滤
        processed_detections = []
        for det in all_detections:
            x1, y1, x2, y2 = det['box_xyxy'].cpu().numpy()
            box_area = (x2 - x1) * (y2 - y1)
            image_area = h * w

            if box_area < 0.001 * image_area or box_area > 0.8 * image_area:
                continue

            processed_detections.append(det)

        if not processed_detections:
            return []

        # 分别对动态和静态物体应用NMS
        dynamic_dets = [d for d in processed_detections if d['type'] == 'dynamic']
        static_dets = [d for d in processed_detections if d['type'] == 'static']

        final_detections = []

        # 对动态物体应用NMS
        if dynamic_dets:
            dynamic_nms = self._apply_nms_with_sam(dynamic_dets, iou_threshold=0.5)
            final_detections.extend(dynamic_nms)

        # 对静态物体应用NMS
        if static_dets:
            static_nms = self._apply_nms_with_sam(static_dets, iou_threshold=0.6)
            final_detections.extend(static_nms)

        return final_detections

    def _apply_nms_with_sam(self, detections, iou_threshold=0.5):
        """应用考虑SAM mask的NMS"""
        if len(detections) <= 1:
            return detections

        boxes = torch.stack([det['box_xyxy'] for det in detections])
        scores = torch.tensor([det['confidence'] for det in detections])

        keep_indices = torchvision.ops.nms(boxes, scores, iou_threshold)
        return [detections[i] for i in keep_indices.cpu().numpy()]

    def _create_comprehensive_visualization(self, image_source, detections, save_dir, image_name):
        """🎨 创建完整的可视化"""
        base_name = os.path.splitext(image_name)[0]
        h, w = image_source.shape[:2]

        # 1. 创建基础图像副本
        combined_image = image_source.copy()
        dynamic_only_image = image_source.copy()
        static_only_image = image_source.copy()
        sam_visualization = image_source.copy()

        dynamic_count = 0
        static_count = 0
        sam_success_count = 0

        # 2. 处理每个检测结果
        for det in detections:
            x1, y1, x2, y2 = det['box_xyxy'].cpu().numpy().astype(int)
            confidence = det['confidence']
            phrase = det['phrase']
            obj_type = det['type']
            sam_mask = det.get('sam_mask')
            sam_success = det.get('sam_success', False)

            # 获取颜色
            color = self.get_color_for_class(phrase, is_dynamic=(obj_type == 'dynamic'))

            if obj_type == 'dynamic':
                dynamic_count += 1
            else:
                static_count += 1

            if sam_success:
                sam_success_count += 1

            label = f"{phrase}: {confidence:.2f}"

            # 3. 绘制边界框在各种图像上
            self._draw_detection_box(combined_image, (x1, y1, x2, y2), label, color, obj_type)

            if obj_type == 'dynamic':
                self._draw_detection_box(dynamic_only_image, (x1, y1, x2, y2), label, color, obj_type)
            else:
                self._draw_detection_box(static_only_image, (x1, y1, x2, y2), label, color, obj_type)

            # 4. 🎯 处理SAM mask可视化
            if sam_mask is not None and sam_success:
                mask_bool = sam_mask.astype(bool)

                # SAM可视化 - 半透明叠加
                overlay_layer = np.zeros_like(sam_visualization)
                overlay_layer[mask_bool] = color
                sam_visualization = cv2.addWeighted(sam_visualization, 0.7, overlay_layer, 0.3, 0)

        # 5. 添加统计信息
        self._add_statistics_text(combined_image, dynamic_count, static_count, sam_success_count)
        self._add_statistics_text(dynamic_only_image, dynamic_count, 0, 0, "DYNAMIC OBJECTS ONLY")
        self._add_statistics_text(static_only_image, 0, static_count, 0, "STATIC OBJECTS ONLY")
        self._add_statistics_text(sam_visualization, dynamic_count, static_count, sam_success_count, "SAM SEGMENTATION")

        # 6. 保存所有可视化结果
        save_paths = {
            'combined': f"{base_name}_combined_detection.jpg",
            'dynamic_only': f"{base_name}_dynamic_only.jpg",
            'static_only': f"{base_name}_static_only.jpg",
            'sam_overlay': f"{base_name}_sam_overlay.jpg",
        }

        # 保存图像
        cv2.imwrite(os.path.join(save_dir, save_paths['combined']), combined_image)
        cv2.imwrite(os.path.join(save_dir, save_paths['dynamic_only']), dynamic_only_image)
        cv2.imwrite(os.path.join(save_dir, save_paths['static_only']), static_only_image)
        cv2.imwrite(os.path.join(save_dir, save_paths['sam_overlay']), sam_visualization)

        # 保存原始SAM masks
        self._save_individual_sam_masks(detections, save_dir, base_name)

        print(f"✅ 完整可视化结果已保存到 {save_dir}")
        print(f"   🎨 综合检测: {save_paths['combined']}")
        print(f"   🔴 动态物体: {save_paths['dynamic_only']}")
        print(f"   🟢 静态物体: {save_paths['static_only']}")
        print(f"   🎯 SAM叠加: {save_paths['sam_overlay']}")

    def _draw_detection_box(self, image, box, label, color, obj_type):
        """在图像上绘制检测框"""
        x1, y1, x2, y2 = box

        # 绘制边界框
        thickness = 3 if obj_type == 'dynamic' else 2
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        # 准备标签文本
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2

        # 添加类型标识
        type_prefix = "[D] " if obj_type == 'dynamic' else "[S] "
        full_label = type_prefix + label

        # 获取文本大小
        (text_width, text_height), baseline = cv2.getTextSize(
            full_label, font, font_scale, font_thickness
        )

        # 绘制文本背景
        # bg_color = tuple(int(c * 0.8) for c in color)
        # cv2.rectangle(
        #     image,
        #     (x1, y1 - text_height - 10),
        #     (x1 + text_width + 10, y1),
        #     bg_color,
        #     -1
        # )

        # 绘制文本
        # cv2.putText(
        #     image,
        #     full_label,
        #     (x1 + 5, y1 - 5),
        #     font,
        #     font_scale,
        #     (255, 255, 255),
        #     font_thickness
        # )

    def _add_statistics_text(self, image, dynamic_count, static_count, sam_count=0, title="DETECTION RESULTS"):
        """在图像上添加统计信息"""
        h, w = image.shape[:2]

        # 统计文本
        stats_lines = [
            title,
            f"Dynamic Objects: {dynamic_count}",
            f"Static Objects: {static_count}",
            f"Total Objects: {dynamic_count + static_count}"
        ]

        if sam_count > 0:
            stats_lines.append(f"SAM Success: {sam_count}")

        # 文本参数
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 2
        line_height = 30

        # 计算文本区域大小
        max_width = 0
        for line in stats_lines:
            (text_width, _), _ = cv2.getTextSize(line, font, font_scale, font_thickness)
            max_width = max(max_width, text_width)

        # 绘制背景
        # bg_height = len(stats_lines) * line_height + 20
        # cv2.rectangle(image, (10, 10), (max_width + 30, bg_height), (0, 0, 0), -1)
        # cv2.rectangle(image, (10, 10), (max_width + 30, bg_height), (255, 255, 255), 2)

        # 绘制文本
        # for i, line in enumerate(stats_lines):
        #     y = 35 + i * line_height
        #     color = (255, 255, 255) if i == 0 else (0, 255, 255)
        #     cv2.putText(image, line, (20, y), font, font_scale, color, font_thickness)

    def _save_individual_sam_masks(self, detections, save_dir, base_name):
        """保存单独的SAM masks"""
        masks_dir = os.path.join(save_dir, "masks")

        for i, det in enumerate(detections):
            if det.get('sam_mask') is not None:
                mask = det['sam_mask']
                phrase = det['phrase'].replace(' ', '_').replace('/', '_')
                obj_type = det['type']

                # 保存二值mask
                mask_path = os.path.join(masks_dir, f"{base_name}_{obj_type}_{i:02d}_{phrase}_mask.png")
                cv2.imwrite(mask_path, mask * 255)

                # 保存彩色mask
                color = self.get_color_for_class(det['phrase'], is_dynamic=(obj_type == 'dynamic'))
                colored_individual_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
                colored_individual_mask[mask.astype(bool)] = color

                colored_path = os.path.join(masks_dir, f"{base_name}_{obj_type}_{i:02d}_{phrase}_colored.png")
                cv2.imwrite(colored_path, colored_individual_mask)


# ================================
# 主函数和批量处理示例
# ================================

def main():
    """单张图片处理主函数"""
    print("🚀 启动动态静态物体检测系统")
    print("=" * 50)

    # 配置路径
    grounding_dino_config = "/home/zwk/下载/S3PO-GS-main/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    grounding_dino_checkpoint = "/home/zwk/下载/S3PO-GS-main/groundingdino_swint_ogc.pth"
    sam_checkpoint = "/home/zwk/下载/S3PO-GS-main/sam_vit_h_4b8939.pth"

    # 初始化检测器
    detector = EnhancedDynamicStaticDetector(
        grounding_dino_config=grounding_dino_config,
        grounding_dino_checkpoint=grounding_dino_checkpoint,
        sam_checkpoint=sam_checkpoint,
        device="cuda"
    )

    # 测试图像
    test_image = "/home/zwk/dataset/SemanticKITTI/SemanticKITTI/dataset/sequences/04/image_2/000004.png"

    if os.path.exists(test_image):
        print("\n🎯 开始检测...")
        detections = detector.detect_objects_with_sam(test_image, "./detection_results")

        print(f"\n📊 检测结果:")
        print(f"   总检测数: {len(detections)}")
        print(f"   动态物体: {len([d for d in detections if d['type'] == 'dynamic'])}")
        print(f"   静态物体: {len([d for d in detections if d['type'] == 'static'])}")

    else:
        print(f"⚠️ 测试图像不存在: {test_image}")

    print("\n✅ 检测完成!")


def main_batch():
    """批量处理主函数"""
    print("🚀 启动批量动态静态物体检测系统")
    print("=" * 50)

    # 配置路径
    grounding_dino_config = "/home/zwk/下载/S3PO-GS-main/GroundingDINO-main/groundingdino/config/GroundingDINO_SwinB_cfg.py"
    grounding_dino_checkpoint = "/home/zwk/下载/groundingdino_swinb_cogcoor.pth"
    sam_checkpoint = "/home/zwk/下载/S3PO-GS-main/sam_vit_h_4b8939.pth"

    # 初始化检测器
    detector = EnhancedDynamicStaticDetector(
        grounding_dino_config=grounding_dino_config,
        grounding_dino_checkpoint=grounding_dino_checkpoint,
        sam_checkpoint=sam_checkpoint,
        device="cuda"
    )

    # 批量处理文件夹
    input_folder = "/home/zwk/dataset/SemanticKITTI/SemanticKITTI/dataset/sequences/08/image_2/"
    output_folder = "./batch_detection_results_ours"

    if os.path.exists(input_folder):
        print("\n🎯 开始批量检测...")
        batch_results = detector.detect_folder_images(input_folder, output_folder)
        print("✅ 批量检测完成!")
    else:
        print(f"⚠️ 输入文件夹不存在: {input_folder}")


if __name__ == "__main__":
    # 可以选择单张图片处理或批量处理
    print("📋 请选择处理模式:")
    print("1. 单张图片处理")
    print("2. 批量处理文件夹")

    try:
        choice = input("请输入选择 (1 或 2): ").strip()
        if choice == "2":
            main_batch()
        elif choice == "1":
            main()
        else:
            print("⚠️ 无效选择，默认运行单张图片处理")
            main()
    except KeyboardInterrupt:
        print("\n👋 程序被用户中断")
    except Exception as e:
        print(f"❌ 程序运行出错: {e}")