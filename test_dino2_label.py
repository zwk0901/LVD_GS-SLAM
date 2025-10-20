# enhanced_semantic_segmentation_detector.py
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


class EnhancedSemanticSegmentationDetector:
    """增强的语义分割检测器 - 生成语义类别mask图"""

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

        # 定义语义类别和颜色
        self.setup_semantic_categories()

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

    def setup_semantic_categories(self):
        """设置语义类别和颜色"""

        # 定义详细的语义类别
        self.semantic_categories = {
            # 地面相关 - 紫红色系
            "ground": {
                "keywords": ["road", "street", "pavement", "sidewalk", "ground", "asphalt",
                             "concrete", "pathway", "walkway", "lane", "highway"],
                "color": [128, 0, 128],  # 紫红色
                "id": 1
            },

            # 建筑物 - 灰色系
            "building": {
                "keywords": ["building", "house", "structure", "architecture", "edifice",
                             "construction", "facility", "office", "residential"],
                "color": [128, 128, 128],  # 灰色
                "id": 2
            },

            # 墙体和围栏 - 褐色系
            "wall_fence": {
                "keywords": ["wall", "fence", "barrier", "railing", "guardrail", "partition"],
                "color": [139, 69, 19],  # 褐色
                "id": 3
            },

            # 植被 - 绿色系
            "vegetation": {
                "keywords": ["tree", "bush", "plant", "vegetation", "grass", "shrub",
                             "foliage", "garden", "park", "lawn"],
                "color": [0, 128, 0],  # 绿色
                "id": 4
            },

            # 车辆 - 红色系
            "vehicle": {
                "keywords": ["car", "truck", "bus", "van", "suv", "vehicle", "automobile",
                             "motorcycle", "motorbike", "taxi", "sedan"],
                "color": [255, 0, 0],  # 红色
                "id": 5
            },

            # 人 - 蓝色系
            "person": {
                "keywords": ["person", "people", "pedestrian", "human", "individual",
                             "man", "woman", "child"],
                "color": [0, 0, 255],  # 蓝色
                "id": 6
            },

            # 自行车 - 橙色系
            "bicycle": {
                "keywords": ["bicycle", "bike", "cycling", "cyclist"],
                "color": [255, 165, 0],  # 橙色
                "id": 7
            },

            # 交通设施 - 黄色系
            "traffic": {
                "keywords": ["traffic light", "traffic sign", "sign", "signal", "pole",
                             "street lamp", "light pole", "stop sign", "traffic signal"],
                "color": [255, 255, 0],  # 黄色
                "id": 8
            },

            # 天空 - 青色系
            "sky": {
                "keywords": ["sky", "cloud", "air", "atmosphere"],
                "color": [0, 255, 255],  # 青色
                "id": 9
            },

            # 其他物体 - 白色
            "other": {
                "keywords": ["object", "item", "thing"],
                "color": [255, 255, 255],  # 白色
                "id": 10
            }
        }

        # 创建颜色映射字典
        self.color_map = {}
        self.id_to_color = {}
        for category, info in self.semantic_categories.items():
            for keyword in info["keywords"]:
                self.color_map[keyword.lower()] = info["color"]
            self.id_to_color[info["id"]] = info["color"]

        print("🎨 语义类别设置完成:")
        for category, info in self.semantic_categories.items():
            print(f"   {category}: {info['color']} (ID: {info['id']})")

    def classify_semantic_category(self, phrase):
        """将检测到的物体分类到语义类别"""
        phrase_lower = phrase.lower().strip()

        # 遍历所有语义类别
        for category, info in self.semantic_categories.items():
            for keyword in info["keywords"]:
                if keyword in phrase_lower:
                    return category, info["color"], info["id"]

        # 默认返回other类别
        return "other", self.semantic_categories["other"]["color"], self.semantic_categories["other"]["id"]

    def create_semantic_prompts(self):
        """创建语义分割的提示词"""
        # 收集所有关键词
        all_keywords = []
        for category, info in self.semantic_categories.items():
            all_keywords.extend(info["keywords"][:3])  # 每个类别取前3个关键词

        # 创建提示词
        prompt = " . ".join(all_keywords) + " ."
        return prompt

    def generate_semantic_masks(self, image_path, save_dir="./semantic_results"):
        """生成语义分割mask图"""
        print(f"🎯 开始生成语义分割mask: {image_path}")

        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, "semantic_masks"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "overlay_results"), exist_ok=True)

        # 加载图像
        image_source, image = load_image(image_path)
        h, w, _ = image_source.shape
        print(f"   图像尺寸: {w}x{h}")

        # 设置SAM图像
        if self.sam_predictor:
            self.sam_predictor.set_image(image_source)

        # 创建语义提示词
        semantic_prompt = self.create_semantic_prompts()
        print(f"🎯 语义提示词: {semantic_prompt[:100]}...")

        # 执行检测
        detections = self._detect_semantic_objects(image, semantic_prompt, w, h)

        # 生成语义mask
        semantic_mask, overlay_image = self._create_semantic_mask(image_source, detections, w, h)

        # 保存结果
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        self._save_semantic_results(semantic_mask, overlay_image, image_source, save_dir, base_name, detections)

        return semantic_mask, overlay_image, detections

    def _detect_semantic_objects(self, image, prompt, w, h):
        """检测语义物体"""
        detections = []

        try:
            with torch.no_grad():
                boxes, logits, phrases = predict(
                    model=self.grounding_dino_model,
                    image=image,
                    caption=prompt,
                    box_threshold=self.box_threshold,
                    text_threshold=self.text_threshold,
                    device=self.device
                )

            print(f"   检测到 {len(boxes)} 个物体")

            # 处理每个检测结果
            for i, (box, logit, phrase) in enumerate(zip(boxes, logits, phrases)):
                confidence = logit.item()

                if confidence < self.min_confidence:
                    continue

                # 转换坐标
                box_xyxy = box_ops.box_cxcywh_to_xyxy(box.unsqueeze(0)) * torch.tensor([w, h, w, h])
                x1, y1, x2, y2 = box_xyxy[0].cpu().numpy().astype(int)

                # 确保坐标在图像范围内
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                # 语义分类
                category, color, category_id = self.classify_semantic_category(phrase)

                detection = {
                    'box': box,
                    'box_xyxy': box_xyxy[0],
                    'confidence': confidence,
                    'phrase': phrase,
                    'category': category,
                    'color': color,
                    'category_id': category_id,
                    'sam_mask': None,
                    'sam_success': False
                }

                # SAM分割
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

                    except Exception as e:
                        print(f"   ❌ SAM分割出错: {phrase} - {e}")

                detections.append(detection)

            # 按置信度排序
            detections.sort(key=lambda x: x['confidence'], reverse=True)
            print(f"   保留 {len(detections)} 个有效检测")

        except Exception as e:
            print(f"❌ 语义检测失败: {e}")

        return detections

    def _create_semantic_mask(self, image_source, detections, w, h):
        """创建语义分割mask"""
        print("🎨 生成语义分割mask...")

        # 创建语义mask (H, W)，初始化为0（背景）
        semantic_mask = np.zeros((h, w), dtype=np.uint8)

        # 创建彩色叠加图像
        overlay_image = image_source.copy()

        # 统计各类别数量
        category_counts = {}

        # 应用NMS去重
        filtered_detections = self._apply_semantic_nms(detections)

        # 按类别ID排序，确保重要类别（如地面）优先处理
        priority_order = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # 地面优先
        sorted_detections = []

        for priority_id in priority_order:
            for det in filtered_detections:
                if det['category_id'] == priority_id:
                    sorted_detections.append(det)

        # 处理每个检测结果
        for det in sorted_detections:
            category = det['category']
            category_id = det['category_id']
            color = det['color']
            sam_mask = det.get('sam_mask')
            box_xyxy = det['box_xyxy']

            # 统计类别
            category_counts[category] = category_counts.get(category, 0) + 1

            # 创建mask区域
            if sam_mask is not None and det['sam_success']:
                # 使用SAM精确mask
                mask_bool = sam_mask.astype(bool)
            else:
                # 使用边界框作为mask
                x1, y1, x2, y2 = box_xyxy.cpu().numpy().astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                mask_bool = np.zeros((h, w), dtype=bool)
                mask_bool[y1:y2, x1:x2] = True

            # 更新语义mask（只在当前位置为背景时更新，避免覆盖重要类别）
            update_mask = mask_bool & (semantic_mask == 0)
            semantic_mask[update_mask] = category_id

            # 创建半透明叠加
            alpha = 0.6
            overlay_layer = np.zeros_like(overlay_image)
            overlay_layer[mask_bool] = color
            overlay_image = cv2.addWeighted(overlay_image, 1 - alpha, overlay_layer, alpha, 0)

            # 添加轮廓
            if sam_mask is not None:
                contours, _ = cv2.findContours(sam_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay_image, contours, -1, color, 2)

        print(f"✅ 语义mask生成完成，包含类别: {category_counts}")
        return semantic_mask, overlay_image

    def _apply_semantic_nms(self, detections, iou_threshold=0.5):
        """对语义检测结果应用NMS"""
        if len(detections) <= 1:
            return detections

        # 按类别分组应用NMS
        category_groups = {}
        for det in detections:
            category = det['category']
            if category not in category_groups:
                category_groups[category] = []
            category_groups[category].append(det)

        filtered_detections = []

        for category, dets in category_groups.items():
            if len(dets) <= 1:
                filtered_detections.extend(dets)
                continue

            # 对每个类别应用NMS
            boxes = torch.stack([det['box_xyxy'] for det in dets])
            scores = torch.tensor([det['confidence'] for det in dets])

            keep_indices = torchvision.ops.nms(boxes, scores, iou_threshold)
            category_filtered = [dets[i] for i in keep_indices.cpu().numpy()]
            filtered_detections.extend(category_filtered)

        return filtered_detections

    def _save_semantic_results(self, semantic_mask, overlay_image, original_image, save_dir, base_name, detections):
        """保存语义分割结果"""
        print("💾 保存语义分割结果...")

        # 1. 保存语义mask (灰度图，像素值代表类别ID)
        semantic_mask_path = os.path.join(save_dir, "semantic_masks", f"{base_name}_semantic_mask.png")
        cv2.imwrite(semantic_mask_path, semantic_mask)

        # 2. 创建彩色语义mask
        colored_semantic_mask = self._create_colored_semantic_mask(semantic_mask)
        colored_mask_path = os.path.join(save_dir, "semantic_masks", f"{base_name}_colored_semantic.png")
        cv2.imwrite(colored_mask_path, colored_semantic_mask)

        # 3. 保存叠加结果
        overlay_path = os.path.join(save_dir, "overlay_results", f"{base_name}_overlay.jpg")
        cv2.imwrite(overlay_path, overlay_image)

        # 4. 创建对比图
        comparison = self._create_comparison_grid(original_image, colored_semantic_mask, overlay_image)
        comparison_path = os.path.join(save_dir, f"{base_name}_comparison.jpg")
        cv2.imwrite(comparison_path, comparison)

        # 5. 保存检测信息JSON
        detection_info = {
            'image_name': base_name,
            'total_detections': len(detections),
            'categories': {},
            'detections': []
        }

        for det in detections:
            category = det['category']
            detection_info['categories'][category] = detection_info['categories'].get(category, 0) + 1

            detection_info['detections'].append({
                'phrase': det['phrase'],
                'category': det['category'],
                'category_id': det['category_id'],
                'confidence': float(det['confidence']),
                'color': det['color'],
                'sam_success': det.get('sam_success', False)
            })

        json_path = os.path.join(save_dir, f"{base_name}_semantic_info.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(detection_info, f, indent=2, ensure_ascii=False)

        print(f"✅ 语义分割结果保存完成:")
        print(f"   🎯 语义mask: {semantic_mask_path}")
        print(f"   🌈 彩色mask: {colored_mask_path}")
        print(f"   📊 叠加结果: {overlay_path}")
        print(f"   📋 对比图: {comparison_path}")
        print(f"   📄 检测信息: {json_path}")

    def _create_colored_semantic_mask(self, semantic_mask):
        """创建彩色语义mask"""
        h, w = semantic_mask.shape
        colored_mask = np.zeros((h, w, 3), dtype=np.uint8)

        # 为每个类别ID分配颜色
        for category_id, color in self.id_to_color.items():
            mask_bool = (semantic_mask == category_id)
            colored_mask[mask_bool] = color

        return colored_mask

    def _create_comparison_grid(self, original, colored_mask, overlay):
        """创建三图对比网格"""
        h, w = original.shape[:2]
        target_size = (w // 3, h // 3)

        # 调整大小
        orig_small = cv2.resize(original, target_size)
        mask_small = cv2.resize(colored_mask, target_size)
        overlay_small = cv2.resize(overlay, target_size)

        # 添加标题
        def add_title(img, title):
            cv2.putText(img, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 255), 3)
            cv2.putText(img, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 0), 2)
            return img

        orig_small = add_title(orig_small, "Original")
        mask_small = add_title(mask_small, "Semantic Mask")
        overlay_small = add_title(overlay_small, "Overlay")

        # 水平拼接
        comparison = np.hstack([orig_small, mask_small, overlay_small])
        return comparison

    def batch_generate_semantic_masks(self, input_folder, output_base_dir="./batch_semantic_results"):
        """批量生成语义分割mask"""
        print("🚀 开始批量语义分割处理")
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

        # 创建输出目录
        output_base_path = Path(output_base_dir)
        output_base_path.mkdir(parents=True, exist_ok=True)

        # 批量处理统计
        batch_results = {
            'total_images': len(image_files),
            'processed_images': 0,
            'failed_images': 0,
            'total_categories': {},
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
                # 为每张图片创建输出文件夹
                image_output_dir = output_base_path / image_name
                image_output_dir.mkdir(parents=True, exist_ok=True)

                # 生成语义分割
                semantic_mask, overlay_image, detections = self.generate_semantic_masks(
                    image_path, str(image_output_dir)
                )

                # 统计类别
                image_categories = {}
                for det in detections:
                    category = det['category']
                    image_categories[category] = image_categories.get(category, 0) + 1
                    batch_results['total_categories'][category] = batch_results['total_categories'].get(category, 0) + 1

                # 保存图片结果信息
                image_result = {
                    'image_name': image_name,
                    'image_path': image_path,
                    'output_dir': str(image_output_dir),
                    'total_detections': len(detections),
                    'categories': image_categories
                }

                batch_results['results_per_image'].append(image_result)
                batch_results['processed_images'] += 1

                print(f"✅ {image_name} 处理完成:")
                print(f"   📊 检测总数: {len(detections)}")
                print(f"   🎨 类别统计: {image_categories}")

            except Exception as e:
                print(f"❌ 处理 {image_name} 时出错: {e}")
                batch_results['failed_images'] += 1
                continue

        # 计算处理时间
        end_time = datetime.now()
        batch_results['processing_time'] = (end_time - start_time).total_seconds()

        # 保存批量处理报告
        self._save_batch_semantic_report(batch_results, output_base_path, start_time, end_time)

        print("\n" + "=" * 60)
        print("🎉 批量语义分割完成!")
        print(f"📊 处理统计:")
        print(f"   📷 总图片数: {batch_results['total_images']}")
        print(f"   ✅ 成功处理: {batch_results['processed_images']}")
        print(f"   ❌ 处理失败: {batch_results['failed_images']}")
        print(f"   🎨 总类别统计: {batch_results['total_categories']}")
        print(f"   ⏱️  总耗时: {batch_results['processing_time']:.2f} 秒")

        return batch_results

    def _save_batch_semantic_report(self, batch_results, output_base_path, start_time, end_time):
        """保存批量语义分割报告"""

        # 保存JSON报告
        summary_json = {
            **batch_results,
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'average_time_per_image': batch_results['processing_time'] / max(1, batch_results['processed_images']),
            'success_rate': batch_results['processed_images'] / batch_results['total_images'] * 100
        }

        json_summary_path = output_base_path / "batch_semantic_summary.json"
        with open(json_summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_json, f, indent=2, ensure_ascii=False)

        # 创建文本报告
        txt_summary_path = output_base_path / "batch_semantic_report.txt"
        with open(txt_summary_path, 'w', encoding='utf-8') as f:
            f.write("🎨 批量语义分割处理报告\n")
            f.write("=" * 50 + "\n\n")
            f.write(
                f"处理时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')} - {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总处理时长: {batch_results['processing_time']:.2f} 秒\n\n")

            f.write("📊 处理统计:\n")
            f.write(f"  总图片数量: {batch_results['total_images']}\n")
            f.write(f"  成功处理: {batch_results['processed_images']}\n")
            f.write(f"  处理失败: {batch_results['failed_images']}\n\n")

            f.write("🎨 语义类别统计:\n")
            for category, count in batch_results['total_categories'].items():
                f.write(f"  {category}: {count}\n")

        print(f"📋 批量语义分割报告已保存:")
        print(f"   📄 JSON报告: {json_summary_path}")
        print(f"   📄 文本报告: {txt_summary_path}")


# 主函数
def main_semantic_single():
    """单张图片语义分割主函数"""
    print("🎨 启动语义分割检测系统")
    print("=" * 50)

    # 配置路径
    grounding_dino_config = "/home/zwk/下载/S3PO-GS-main/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    grounding_dino_checkpoint = "/home/zwk/下载/S3PO-GS-main/groundingdino_swint_ogc.pth"
    sam_checkpoint = "/home/zwk/下载/S3PO-GS-main/sam_vit_h_4b8939.pth"

    # 初始化检测器
    detector = EnhancedSemanticSegmentationDetector(
        grounding_dino_config=grounding_dino_config,
        grounding_dino_checkpoint=grounding_dino_checkpoint,
        sam_checkpoint=sam_checkpoint,
        device="cuda"
    )

    # 测试图像
    test_image = "/home/zwk/dataset/SemanticKITTI/SemanticKITTI/dataset/sequences/04/image_2/000004.png"

    if os.path.exists(test_image):
        print("\n🎯 开始语义分割...")
        semantic_mask, overlay_image, detections = detector.generate_semantic_masks(test_image, "./semantic_results")

        print(f"\n📊 语义分割结果:")
        category_counts = {}
        for det in detections:
            category = det['category']
            category_counts[category] = category_counts.get(category, 0) + 1

        print(f"   总检测数: {len(detections)}")
        print(f"   语义类别: {category_counts}")

    else:
        print(f"⚠️ 测试图像不存在: {test_image}")

    print("\n✅ 语义分割完成!")


def main_semantic_batch():
    """批量语义分割主函数"""
    print("🎨 启动批量语义分割系统")
    print("=" * 50)

    # 配置路径
    grounding_dino_config = "/home/zwk/下载/S3PO-GS-main/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    grounding_dino_checkpoint = "/home/zwk/下载/S3PO-GS-main/groundingdino_swint_ogc.pth"
    sam_checkpoint = "/home/zwk/下载/S3PO-GS-main/sam_vit_h_4b8939.pth"

    # 初始化检测器
    detector = EnhancedSemanticSegmentationDetector(
        grounding_dino_config=grounding_dino_config,
        grounding_dino_checkpoint=grounding_dino_checkpoint,
        sam_checkpoint=sam_checkpoint,
        device="cuda"
    )

    # 批量处理文件夹
    input_folder = "/home/zwk/dataset/SemanticKITTI/SemanticKITTI/dataset/sequences/04/image_2/"
    output_folder = "./batch_semantic_results"

    if os.path.exists(input_folder):
        print("\n🎯 开始批量语义分割...")
        batch_results = detector.batch_generate_semantic_masks(input_folder, output_folder)
        print("✅ 批量语义分割完成!")
    else:
        print(f"⚠️ 输入文件夹不存在: {input_folder}")


if __name__ == "__main__":
    print("📋 请选择处理模式:")
    print("1. 单张图片语义分割")
    print("2. 批量语义分割")

    try:
        choice = input("请输入选择 (1 或 2): ").strip()
        if choice == "2":
            main_semantic_batch()
        elif choice == "1":
            main_semantic_single()
        else:
            print("⚠️ 无效选择，默认运行单张图片处理")
            main_semantic_single()
    except KeyboardInterrupt:
        print("\n👋 程序被用户中断")
    except Exception as e:
        print(f"❌ 程序运行出错: {e}")