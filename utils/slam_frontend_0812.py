import time
import numpy as np
import torch
import torch.multiprocessing as mp
import os
import cv2
from PIL import Image
import torch.nn.functional as F
import colorsys
from typing import Dict, List, Tuple, Optional
from groundingdino.util import box_ops
# Grounding DINO imports with fallback handling
try:
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    GROUNDING_DINO_AVAILABLE = True
except ImportError:
    print("⚠️  Warning: transformers not available. Grounding DINO will be disabled.")
    GROUNDING_DINO_AVAILABLE = False

# SAM imports with fallback handling
try:
    from segment_anything import SamPredictor, sam_model_registry

    SAM_AVAILABLE = True
except ImportError:
    print("⚠️  Warning: segment_anything not available. SAM will be disabled.")
    SAM_AVAILABLE = False

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
from gui import gui_utils
from utils.camera_utils import Camera
from utils.eval_utils import eval_ate, save_gaussians
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj
from utils.pose_utils import update_pose
from utils.slam_utils import get_loss_tracking, get_median_depth
from utils.init_pose import get_pose, get_depth
from utils.depth_utils import process_depth

try:
    from groundingdino.util.inference import load_model, load_image, predict, annotate
    import groundingdino.datasets.transforms as T
    from groundingdino.models import build_model
    from groundingdino.util.slconfig import SLConfig
    from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

    GROUNDING_DINO_ORIGINAL = True
    print("✅ Original Grounding DINO package available")
except ImportError:
    print("⚠️  Original Grounding DINO not installed")
    print("   Install with: pip install groundingdino")

    # 尝试transformers作为备选
    try:
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

        GROUNDING_DINO_AVAILABLE = True
        print("✅ Transformers available as fallback")
    except ImportError:
        print("⚠️  Transformers not available either")


class ColorfulSegmentationVisualizer:
    """彩色类别分割可视化器 - 完全修复版，解决边界框坐标问题"""

    def __init__(self):
        # 预定义动态对象颜色（暖色调，表示运动）
        self.dynamic_colors = {
            'person': [0, 0, 255],  # 红色
            'people': [0, 0, 255],
            'pedestrian': [0, 0, 255],
            'pedestrians': [0, 0, 255],
            'human': [0, 0, 255],

            'car': [0, 165, 255],  # 橙色
            'cars': [0, 165, 255],
            'vehicle': [0, 165, 255],
            'vehicles': [0, 165, 255],

            'truck': [0, 255, 255],  # 黄色
            'trucks': [0, 255, 255],

            'bus': [255, 0, 255],  # 品红色
            'buses': [255, 0, 255],

            'bicycle': [128, 0, 255],  # 紫色
            'bike': [128, 0, 255],
            'bicycles': [128, 0, 255],

            'motorcycle': [255, 0, 128],  # 粉红色
            'motorcycles': [255, 0, 128],
            'motorbike': [255, 0, 128],

            'scooter': [0, 128, 255],  # 浅蓝色
            'e-scooter': [0, 128, 255],
            'skateboard': [64, 0, 255],  # 深紫色
        }

        # 预定义静态对象颜色（冷色调，表示静止）
        self.static_colors = {
            'building': [128, 128, 64],  # 橄榄色
            'wall': [96, 96, 96],  # 灰色
            'road': [64, 64, 64],  # 深灰色
            'street': [64, 64, 64],
            'pavement': [128, 128, 128],  # 浅灰色
            'sidewalk': [160, 160, 160],
            'crosswalk': [192, 192, 192],

            'traffic light': [0, 255, 0],  # 绿色
            'stop sign': [0, 100, 200],  # 深橙色
            'street sign': [0, 150, 150],  # 青色
            'road sign': [0, 150, 150],
            'traffic sign': [0, 150, 150],
            'traffic _ sign': [0, 150, 150],
            'traffic * sign': [0, 150, 150],

            'lamp post': [100, 50, 0],  # 棕色
            'street lamp': [100, 50, 0],
            'lamp': [100, 50, 0],
            'pole': [100, 50, 0],
            'traffic cone': [0, 200, 255],  # 亮橙色

            'tree': [0, 100, 0],  # 深绿色
            'fence': [150, 150, 0],  # 深黄色
            'barrier': [100, 100, 0],
            'guardrail': [100, 100, 0],

            'fire hydrant': [255, 100, 0],  # 橙红色
            'firent': [255, 100, 0],
            'mailbox': [200, 0, 100],  # 深红色
            'bench': [150, 100, 50],  # 棕褐色
            'curb': [200, 200, 200],  # 白色
            'parking meter': [100, 100, 100],  # 中灰色
            'bollard': [150, 150, 100],  # 暗黄色
            'street furniture': [120, 120, 120],
            'street _ furniture': [120, 120, 120],
            'furniture': [120, 120, 120],
            'manhole cover': [50, 50, 50],  # 很深的灰色
            'guard': [80, 80, 80],
        }

        self.color_cache = {}
        self.class_count = 0

    def get_color_for_class(self, class_name: str, is_dynamic: bool = True) -> List[int]:
        """为类别获取颜色（BGR格式）"""
        if class_name is None:
            class_name = f"{'dynamic' if is_dynamic else 'static'}_unknown_{self.class_count}"

        class_name = str(class_name).lower().strip()
        class_name_clean = class_name.replace('_', ' ').replace('*', ' ').replace('  ', ' ')

        predefined_colors = self.dynamic_colors if is_dynamic else self.static_colors

        # 精确匹配
        if class_name in predefined_colors:
            return predefined_colors[class_name]

        if class_name_clean in predefined_colors:
            return predefined_colors[class_name_clean]

        # 包含匹配
        for key, color in predefined_colors.items():
            if key in class_name or class_name in key:
                return color
            if key in class_name_clean or class_name_clean in key:
                return color

        # 词语匹配
        class_words = class_name_clean.split()
        for key, color in predefined_colors.items():
            key_words = key.split()
            if any(word in class_words for word in key_words):
                return color
            if any(class_word in key_words for class_word in class_words):
                return color

        # 检查缓存
        cache_key = f"{class_name}_{is_dynamic}"
        if cache_key in self.color_cache:
            return self.color_cache[cache_key]

        # 生成新颜色
        try:
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

        except Exception as e:
            print(f"⚠️  生成颜色时出错: {e}，使用默认颜色")
            return [0, 0, 255] if is_dynamic else [0, 255, 0]

    def _convert_box_coordinates(self, box, w, h):
        """🔧 核心修复：智能转换边界框坐标格式"""
        try:
            # 转换为numpy数组
            if hasattr(box, 'cpu'):
                box_np = box.cpu().numpy()
            elif hasattr(box, 'numpy'):
                box_np = box.numpy()
            else:
                box_np = np.array(box)

            print(f"🔍 原始box坐标: {box_np}, 图像尺寸: {w}x{h}")

            # 检查坐标格式和范围
            if len(box_np) != 4:
                raise ValueError(f"无效的box格式，应该有4个值，实际: {len(box_np)}")

            # 判断坐标格式
            if np.all(box_np <= 1.0) and np.all(box_np >= 0.0):
                # 归一化坐标 (0-1范围)
                print(f"  📌 检测到归一化坐标")

                # 检查是否是 cxcywh 格式 (center_x, center_y, width, height)
                cx, cy, bw, bh = box_np

                # 转换为像素坐标的 xyxy 格式
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)

                print(f"  📌 cxcywh -> xyxy: ({cx:.3f},{cy:.3f},{bw:.3f},{bh:.3f}) -> ({x1},{y1},{x2},{y2})")

            else:
                # 像素坐标
                # print(f"  📌 检测到像素坐标")

                # 检查是否已经是 xyxy 格式
                if (box_np[2] > box_np[0] and box_np[3] > box_np[1] and
                        box_np[0] >= 0 and box_np[1] >= 0):
                    # 已经是 xyxy 格式
                    x1, y1, x2, y2 = box_np.astype(int)
                    # print(f"  📌 已是xyxy格式: ({x1},{y1},{x2},{y2})")
                else:
                    # 可能是 cxcywh 像素格式
                    cx, cy, bw, bh = box_np
                    x1 = int(cx - bw / 2)
                    y1 = int(cy - bh / 2)
                    x2 = int(cx + bw / 2)
                    y2 = int(cy + bh / 2)
                    # print(f"  📌 像素cxcywh -> xyxy: ({cx},{cy},{bw},{bh}) -> ({x1},{y1},{x2},{y2})")

            # 确保坐标在图像范围内
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(x1 + 1, min(x2, w))
            y2 = max(y1 + 1, min(y2, h))

            # print(f"  ✅ 最终坐标: ({x1},{y1},{x2},{y2})")

            return x1, y1, x2, y2

        except Exception as e:
            print(f"❌ 坐标转换失败: {e}")
            # 返回默认的小区域
            return 0, 0, min(50, w), min(50, h)

    def debug_detection_results(self, dynamic_boxes, dynamic_labels, static_boxes, static_labels, image_shape):
        """🔍 调试检测结果的坐标信息"""
        h, w = image_shape[:2]
        print(f"\n🔍 === 调试检测结果 ===")
        print(f"图像尺寸: {w} x {h}")

        print(f"\n🎯 动态对象 ({len(dynamic_boxes)} 个):")
        for i, (box, label) in enumerate(zip(dynamic_boxes[:3], dynamic_labels[:3])):  # 只显示前3个
            try:
                x1, y1, x2, y2 = self._convert_box_coordinates(box, w, h)
                area = (x2 - x1) * (y2 - y1)
                print(f"  {i + 1}. {label}: ({x1},{y1},{x2},{y2}) 面积={area}")
            except Exception as e:
                print(f"  {i + 1}. {label}: 坐标转换失败 - {e}")

        print(f"\n🏗️ 静态对象 ({len(static_boxes)} 个):")
        for i, (box, label) in enumerate(zip(static_boxes[:3], static_labels[:3])):  # 只显示前3个
            try:
                x1, y1, x2, y2 = self._convert_box_coordinates(box, w, h)
                area = (x2 - x1) * (y2 - y1)
                print(f"  {i + 1}. {label}: ({x1},{y1},{x2},{y2}) 面积={area}")
            except Exception as e:
                print(f"  {i + 1}. {label}: 坐标转换失败 - {e}")

        print(f"=========================\n")

    def create_combined_segmentation_mask(self, image_shape: Tuple[int, int],
                                          dynamic_boxes: np.ndarray, dynamic_labels: List[str],
                                          dynamic_scores: np.ndarray,
                                          static_boxes: np.ndarray, static_labels: List[str],
                                          static_scores: np.ndarray,
                                          dynamic_sam_masks: Optional[List[np.ndarray]] = None,
                                          static_sam_masks: Optional[List[np.ndarray]] = None) -> Tuple[
        np.ndarray, Dict, Dict, np.ndarray]:
        """创建动态和静态对象的组合彩色分割mask"""
        h, w = image_shape
        colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
        class_mask = np.zeros((h, w), dtype=np.int32)
        dynamic_color_map = {}
        static_color_map = {}

        class_id = 1

        # 处理动态对象
        try:
            if len(dynamic_boxes) > 0:
                print(f"🎯 处理 {len(dynamic_boxes)} 个动态对象")
                for i, (box, label, score) in enumerate(zip(dynamic_boxes, dynamic_labels, dynamic_scores)):
                    try:
                        if label is None:
                            label = f"dynamic_object_{i}"

                        color = self.get_color_for_class(label, is_dynamic=True)
                        dynamic_color_map[str(label)] = color

                        # 检查SAM mask
                        use_sam_mask = False
                        if (dynamic_sam_masks is not None and
                                len(dynamic_sam_masks) > i and
                                dynamic_sam_masks[i] is not None):

                            try:
                                mask = dynamic_sam_masks[i]
                                if hasattr(mask, 'shape') and len(mask.shape) >= 2:
                                    if mask.shape[0] == h and mask.shape[1] == w:
                                        mask_bool = mask.astype(bool)
                                        colored_mask[mask_bool] = color
                                        class_mask[mask_bool] = class_id
                                        use_sam_mask = True
                                        print(f"  ✅ 使用SAM mask for {label}")
                                    else:
                                        print(f"⚠️  动态SAM mask {i} 尺寸不匹配: {mask.shape} vs ({h}, {w})")
                            except Exception as e:
                                print(f"⚠️  处理动态SAM mask {i} 时出错: {e}")

                        # 如果SAM mask不可用，使用边界框
                        if not use_sam_mask:
                            try:
                                x1, y1, x2, y2 = self._convert_box_coordinates(box, w, h)

                                if x2 > x1 and y2 > y1:
                                    colored_mask[y1:y2, x1:x2] = color
                                    class_mask[y1:y2, x1:x2] = class_id
                                    print(f"  ✅ 使用边界框 for {label}: ({x1},{y1},{x2},{y2})")
                                else:
                                    print(f"  ❌ 无效边界框 for {label}")

                            except Exception as e:
                                print(f"❌ 处理动态边界框 {i} 时出错: {e}")
                                continue

                        class_id += 1

                    except Exception as e:
                        print(f"❌ 处理动态对象 {i} 时出错: {e}")
                        continue
        except Exception as e:
            print(f"❌ 处理动态对象列表时出错: {e}")

        # 处理静态对象
        try:
            if len(static_boxes) > 0:
                print(f"🏗️ 处理 {len(static_boxes)} 个静态对象")
                for i, (box, label, score) in enumerate(zip(static_boxes, static_labels, static_scores)):
                    try:
                        if label is None:
                            label = f"static_object_{i}"

                        color = self.get_color_for_class(label, is_dynamic=False)
                        static_color_map[str(label)] = color

                        # 检查SAM mask
                        use_sam_mask = False
                        if (static_sam_masks is not None and
                                len(static_sam_masks) > i and
                                static_sam_masks[i] is not None):

                            try:
                                mask = static_sam_masks[i]
                                if hasattr(mask, 'shape') and len(mask.shape) >= 2:
                                    if mask.shape[0] == h and mask.shape[1] == w:
                                        mask_bool = mask.astype(bool)
                                        colored_mask[mask_bool] = color
                                        class_mask[mask_bool] = class_id
                                        use_sam_mask = True
                                        print(f"  ✅ 使用SAM mask for {label}")
                                    else:
                                        print(f"⚠️  静态SAM mask {i} 尺寸不匹配: {mask.shape} vs ({h}, {w})")
                            except Exception as e:
                                print(f"⚠️  处理静态SAM mask {i} 时出错: {e}")

                        # 如果SAM mask不可用，使用边界框
                        if not use_sam_mask:
                            try:
                                x1, y1, x2, y2 = self._convert_box_coordinates(box, w, h)

                                if x2 > x1 and y2 > y1:
                                    colored_mask[y1:y2, x1:x2] = color
                                    class_mask[y1:y2, x1:x2] = class_id
                                    print(f"  ✅ 使用边界框 for {label}: ({x1},{y1},{x2},{y2})")
                                else:
                                    print(f"  ❌ 无效边界框 for {label}")

                            except Exception as e:
                                print(f"❌ 处理静态边界框 {i} 时出错: {e}")
                                continue

                        class_id += 1

                    except Exception as e:
                        print(f"❌ 处理静态对象 {i} 时出错: {e}")
                        continue
        except Exception as e:
            print(f"❌ 处理静态对象列表时出错: {e}")

        return colored_mask, dynamic_color_map, static_color_map, class_mask

    def create_overlay_visualization(self, original_image: np.ndarray,
                                     colored_mask: np.ndarray,
                                     alpha: float = 0.6) -> np.ndarray:
        """创建叠加可视化"""
        try:
            if len(original_image.shape) == 3:
                original_bgr = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
            else:
                original_bgr = original_image

            mask_area = (colored_mask.sum(axis=2) > 0)
            overlay = original_bgr.copy()

            if np.any(mask_area):
                overlay[mask_area] = cv2.addWeighted(
                    original_bgr[mask_area], 1 - alpha,
                    colored_mask[mask_area], alpha, 0
                )

            return overlay
        except Exception as e:
            print(f"❌ 创建叠加可视化时出错: {e}")
            return original_image if original_image is not None else np.zeros((480, 640, 3), dtype=np.uint8)

    def add_combined_legend(self, image: np.ndarray,
                            dynamic_color_map: Dict[str, List[int]],
                            static_color_map: Dict[str, List[int]],
                            dynamic_scores: Optional[np.ndarray] = None,
                            dynamic_labels: Optional[List[str]] = None,
                            static_scores: Optional[np.ndarray] = None,
                            static_labels: Optional[List[str]] = None) -> np.ndarray:
        """在图像上添加动态和静态对象的组合图例"""
        try:
            legend_image = image.copy()
            h, w = image.shape[:2]

            total_classes = len(dynamic_color_map) + len(static_color_map)
            if total_classes == 0:
                return legend_image

            legend_width = min(350, w // 3)
            legend_height = min(h - 20, total_classes * 25 + 100)
            legend_x = max(0, w - legend_width - 10)
            legend_y = 10

            # 绘制图例背景
            cv2.rectangle(legend_image,
                          (legend_x, legend_y),
                          (legend_x + legend_width, legend_y + legend_height),
                          (255, 255, 255), -1)
            cv2.rectangle(legend_image,
                          (legend_x, legend_y),
                          (legend_x + legend_width, legend_y + legend_height),
                          (0, 0, 0), 2)

            y_offset = legend_y + 20

            # 动态对象标题
            if dynamic_color_map:
                cv2.putText(legend_image, "Dynamic Objects",
                            (legend_x + 10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                y_offset += 25

                # 绘制动态对象
                for class_name, color in dynamic_color_map.items():
                    try:
                        if y_offset > legend_y + legend_height - 30:
                            break

                        cv2.rectangle(legend_image,
                                      (legend_x + 10, y_offset - 8),
                                      (legend_x + 25, y_offset + 8),
                                      color, -1)

                        display_name = str(class_name)
                        if len(display_name) > 25:
                            display_name = display_name[:22] + "..."

                        text = display_name
                        if dynamic_scores is not None and dynamic_labels is not None:
                            try:
                                for j, label in enumerate(dynamic_labels):
                                    if str(label).lower() == str(class_name).lower():
                                        text += f" ({dynamic_scores[j]:.2f})"
                                        break
                            except Exception as e:
                                print(f"⚠️  处理动态对象置信度时出错: {e}")

                        cv2.putText(legend_image, text,
                                    (legend_x + 30, y_offset + 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                        y_offset += 22
                    except Exception as e:
                        print(f"❌ 绘制动态图例项时出错: {e}")
                        continue

            # 静态对象标题
            if static_color_map:
                y_offset += 5
                if y_offset < legend_y + legend_height - 50:
                    cv2.putText(legend_image, "Static Objects",
                                (legend_x + 10, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 0), 2)
                    y_offset += 25

                    # 绘制静态对象
                    for class_name, color in static_color_map.items():
                        try:
                            if y_offset > legend_y + legend_height - 30:
                                break

                            cv2.rectangle(legend_image,
                                          (legend_x + 10, y_offset - 8),
                                          (legend_x + 25, y_offset + 8),
                                          color, -1)

                            display_name = str(class_name)
                            if len(display_name) > 25:
                                display_name = display_name[:22] + "..."

                            text = display_name
                            if static_scores is not None and static_labels is not None:
                                try:
                                    for j, label in enumerate(static_labels):
                                        if str(label).lower() == str(class_name).lower():
                                            text += f" ({static_scores[j]:.2f})"
                                            break
                                except Exception as e:
                                    print(f"⚠️  处理静态对象置信度时出错: {e}")

                            cv2.putText(legend_image, text,
                                        (legend_x + 30, y_offset + 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                            y_offset += 22
                        except Exception as e:
                            print(f"❌ 绘制静态图例项时出错: {e}")
                            continue

            return legend_image

        except Exception as e:
            print(f"❌ 添加图例时出错: {e}")
            return image if image is not None else np.zeros((480, 640, 3), dtype=np.uint8)

    def create_separate_visualizations(self, image_shape: Tuple[int, int],
                                       dynamic_boxes: np.ndarray, dynamic_labels: List[str],
                                       dynamic_scores: np.ndarray,
                                       static_boxes: np.ndarray, static_labels: List[str],
                                       static_scores: np.ndarray,
                                       dynamic_sam_masks: Optional[List[np.ndarray]] = None,
                                       static_sam_masks: Optional[List[np.ndarray]] = None) -> Tuple[
        np.ndarray, np.ndarray]:
        """创建分离的动态和静态对象可视化"""
        try:
            h, w = image_shape

            # 动态对象可视化
            dynamic_mask = np.zeros((h, w, 3), dtype=np.uint8)
            try:
                if len(dynamic_boxes) > 0:
                    for i, (box, label, score) in enumerate(zip(dynamic_boxes, dynamic_labels, dynamic_scores)):
                        try:
                            if label is None:
                                label = f"dynamic_{i}"

                            color = self.get_color_for_class(label, is_dynamic=True)

                            # 处理SAM mask
                            if (dynamic_sam_masks is not None and
                                    len(dynamic_sam_masks) > i and
                                    dynamic_sam_masks[i] is not None):
                                try:
                                    mask = dynamic_sam_masks[i]
                                    if hasattr(mask, 'shape') and mask.shape[0] == h and mask.shape[1] == w:
                                        mask_bool = mask.astype(bool)
                                        dynamic_mask[mask_bool] = color
                                    else:
                                        self._draw_box_mask(dynamic_mask, box, color, w, h)
                                except Exception as e:
                                    print(f"⚠️  处理动态SAM mask {i}: {e}")
                                    self._draw_box_mask(dynamic_mask, box, color, w, h)
                            else:
                                self._draw_box_mask(dynamic_mask, box, color, w, h)
                        except Exception as e:
                            print(f"❌ 处理动态对象 {i}: {e}")
                            continue
            except Exception as e:
                print(f"❌ 处理动态对象列表: {e}")

            # 静态对象可视化
            static_mask = np.zeros((h, w, 3), dtype=np.uint8)
            try:
                if len(static_boxes) > 0:
                    for i, (box, label, score) in enumerate(zip(static_boxes, static_labels, static_scores)):
                        try:
                            if label is None:
                                label = f"static_{i}"

                            color = self.get_color_for_class(label, is_dynamic=False)

                            # 处理SAM mask
                            if (static_sam_masks is not None and
                                    len(static_sam_masks) > i and
                                    static_sam_masks[i] is not None):
                                try:
                                    mask = static_sam_masks[i]
                                    if hasattr(mask, 'shape') and mask.shape[0] == h and mask.shape[1] == w:
                                        mask_bool = mask.astype(bool)
                                        static_mask[mask_bool] = color
                                    else:
                                        self._draw_box_mask(static_mask, box, color, w, h)
                                except Exception as e:
                                    print(f"⚠️  处理静态SAM mask {i}: {e}")
                                    self._draw_box_mask(static_mask, box, color, w, h)
                            else:
                                self._draw_box_mask(static_mask, box, color, w, h)
                        except Exception as e:
                            print(f"❌ 处理静态对象 {i}: {e}")
                            continue
            except Exception as e:
                print(f"❌ 处理静态对象列表: {e}")

            return dynamic_mask, static_mask

        except Exception as e:
            print(f"❌ 创建分离可视化时出错: {e}")
            h, w = image_shape
            return np.zeros((h, w, 3), dtype=np.uint8), np.zeros((h, w, 3), dtype=np.uint8)

    def _draw_box_mask(self, mask, box, color, w, h):
        """安全地绘制边界框mask"""
        try:
            x1, y1, x2, y2 = self._convert_box_coordinates(box, w, h)

            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = color
            else:
                print(f"❌ 无效的边界框尺寸: ({x1},{y1},{x2},{y2})")

        except Exception as e:
            print(f"❌ 绘制边界框时出错: {e}")


class ScenePromptManager:
    """场景检测和Prompt管理器"""

    def __init__(self, default_scene="outdoor_street"):
        self.current_scene = default_scene
        self.scene_prompts = {

            "outdoor_street": {
                "dynamic_objects": [
                    "car", "cars", "vehicle", "vehicles", "truck", "trucks",
                    "bus", "buses", "motorcycle", "motorcycles",
                    "bicycle", "bicycles", "person", "people", "pedestrian",
                    "pedestrians"
                ],
                "static_objects": [
                    "road", "sidewalk", "building", "traffic_sign", "traffic_light",
                    "pole", "tree", "street_furniture", "barrier", "guardrail",
                    "traffic light", "stop sign", "street sign", "road sign",
                    "lamp post", "street lamp", "pole", "traffic cone",
                    "fire hydrant", "mailbox", "bench", "tree", "curb",
                    "bollard"
                ],
                "confidence_threshold": 0.20,
                "description": "Complete urban street scene with all dynamic and static elements"
            },

            "parking_lot": {
                "dynamic_objects": [
                    "car", "cars", "parked car", "moving car",
                    "truck", "trucks", "van", "vans",
                    "suv", "sedan", "hatchback",
                    "person", "people", "pedestrian", "walking person",
                    "shopping cart", "trolley",
                    "motorcycle", "bike"
                ],
                "static_objects": [
                    "parking sign", "parking meter", "light pole",
                    "building", "wall", "fence", "barrier",
                    "parking line", "curb", "pavement"
                ],
                "confidence_threshold": 0.2,
                "description": "Parking lot with stationary and moving vehicles"
            },

            "highway": {
                "dynamic_objects": [
                    "car", "cars", "vehicle", "vehicles",
                    "truck", "trucks", "semi truck", "trailer",
                    "bus", "coach", "van", "suv",
                    "motorcycle", "motorbike"
                ],
                "static_objects": [
                    "road", "highway", "guardrail", "barrier",
                    "highway sign", "road sign", "light pole",
                    "bridge", "overpass"
                ],
                "confidence_threshold": 0.25,
                "description": "Highway scene with fast-moving vehicles"
            },

            "residential": {
                "dynamic_objects": [
                    "car", "cars", "parked car",
                    "person", "people", "child", "children", "adult",
                    "bicycle", "bike", "scooter", "skateboard",
                    "dog", "cat", "pet", "animal",
                    "stroller", "wheelchair"
                ],
                "static_objects": [
                    "house", "building", "fence", "gate",
                    "mailbox", "tree", "garden", "lawn",
                    "sidewalk", "driveway", "street lamp"
                ],
                "confidence_threshold": 0.18,
                "description": "Residential area with people and pets"
            },

            "indoor": {
                "dynamic_objects": [
                    "person", "people", "human", "visitor",
                    "chair", "rolling chair", "office chair",
                    "robot", "cleaning robot", "vacuum robot",
                    "cart", "trolley", "wheelchair",
                    "door", "opening door", "moving door"
                ],
                "static_objects": [
                    "wall", "desk", "table", "cabinet",
                    "window", "door", "ceiling", "floor",
                    "computer", "monitor", "lamp"
                ],
                "confidence_threshold": 0.3,
                "description": "Indoor environment with people and movable objects"
            },

            "construction": {
                "dynamic_objects": [
                    "construction vehicle", "excavator", "bulldozer",
                    "dump truck", "crane", "forklift",
                    "worker", "construction worker", "person",
                    "vehicle", "truck", "van"
                ],
                "static_objects": [
                    "building", "construction site", "scaffold",
                    "barrier", "fence", "sign", "pole",
                    "container", "material pile"
                ],
                "confidence_threshold": 0.2,
                "description": "Construction site with heavy machinery"
            },

            "campus": {
                "dynamic_objects": [
                    "person", "people", "student", "students",
                    "bicycle", "bike", "scooter", "skateboard",
                    "car", "vehicle", "bus", "shuttle bus",
                    "delivery robot", "robot", "cart"
                ],
                "static_objects": [
                    "building", "library", "classroom", "tree",
                    "bench", "sign", "lamp post", "statue",
                    "fountain", "walkway", "lawn"
                ],
                "confidence_threshold": 0.2,
                "description": "University campus with students and vehicles"
            }
        }

        # 场景自动检测关键词
        self.scene_keywords = {
            "highway": ["highway", "freeway", "motorway", "interstate"],
            "parking_lot": ["parking", "garage", "lot"],
            "residential": ["residential", "neighborhood", "suburb"],
            "indoor": ["indoor", "inside", "interior", "office", "building"],
            "construction": ["construction", "building", "work", "site"],
            "campus": ["campus", "university", "college", "school"]
        }

    def detect_scene_from_config(self, config_scene_hint=None):
        """从配置或其他信息检测场景类型"""
        if config_scene_hint and config_scene_hint in self.scene_prompts:
            self.current_scene = config_scene_hint
            return self.current_scene

        # 可以扩展为基于图像内容的场景检测
        return self.current_scene

    def detect_scene_from_path(self, data_path):
        """从数据路径检测场景类型"""
        path_lower = data_path.lower()

        for scene_type, keywords in self.scene_keywords.items():
            if any(keyword in path_lower for keyword in keywords):
                self.current_scene = scene_type
                print(f"🎯 Auto-detected scene type: {scene_type} from path: {data_path}")
                return scene_type

        print(f"🔍 Using default scene type: {self.current_scene}")
        return self.current_scene

    def get_current_prompt(self):
        """获取当前场景的动态对象prompt"""
        scene_info = self.scene_prompts[self.current_scene]
        prompt = ". ".join(scene_info["dynamic_objects"])
        return prompt, scene_info["confidence_threshold"]

    def get_static_prompt(self):
        """获取当前场景的静态对象prompt"""
        scene_info = self.scene_prompts[self.current_scene]
        if "static_objects" in scene_info:
            prompt = ". ".join(scene_info["static_objects"])
            return prompt, scene_info["confidence_threshold"]
        return "", scene_info["confidence_threshold"]

    def get_combined_prompt(self):
        """获取动态和静态对象的组合prompt"""
        scene_info = self.scene_prompts[self.current_scene]
        dynamic_prompt = ". ".join(scene_info["dynamic_objects"])
        static_prompt = ". ".join(scene_info.get("static_objects", []))
        combined_prompt = dynamic_prompt
        if static_prompt:
            combined_prompt += ". " + static_prompt
        return combined_prompt, scene_info["confidence_threshold"]

    def get_detailed_prompt(self):
        """获取详细的prompt信息"""
        scene_info = self.scene_prompts[self.current_scene]
        return {
            "dynamic_prompt": ". ".join(scene_info["dynamic_objects"]),
            "static_prompt": ". ".join(scene_info.get("static_objects", [])),
            "confidence_threshold": scene_info["confidence_threshold"],
            "description": scene_info["description"],
            "dynamic_classes": scene_info["dynamic_objects"],
            "static_classes": scene_info.get("static_objects", [])
        }

    def set_scene(self, scene_type):
        """手动设置场景类型"""
        if scene_type in self.scene_prompts:
            self.current_scene = scene_type
            print(f"🎬 Scene type set to: {scene_type}")
        else:
            available_scenes = list(self.scene_prompts.keys())
            print(f"❌ Unknown scene type: {scene_type}. Available: {available_scenes}")

    def add_custom_scene(self, scene_name, dynamic_objects, static_objects=None, confidence_threshold=0.2,
                         description=""):
        """添加自定义场景配置"""
        self.scene_prompts[scene_name] = {
            "dynamic_objects": dynamic_objects,
            "static_objects": static_objects or [],
            "confidence_threshold": confidence_threshold,
            "description": description
        }
        print(f"✅ Added custom scene: {scene_name}")


class GroundingDINODetector:
    """Grounding DINO 检测器封装，支持本地.pth文件"""

    def __init__(self, model_path=None, config_path=None, device="cuda"):
        self.device = device
        self.model = None
        self.processor = None
        self.use_original = False
        self.use_transformers = False
        self.use_yolo = True

        # 如果提供了本地.pth文件路径
        if model_path and model_path.endswith('.pth') and os.path.exists(model_path):
            self._load_original_grounding_dino(model_path, config_path)
        else:
            self._load_transformers_model(model_path)

    def _load_original_grounding_dino(self, model_path, config_path=None):
        """加载原始的Grounding DINO模型（.pth文件）"""
        if not GROUNDING_DINO_ORIGINAL:
            print("❌ Original Grounding DINO package not installed")
            print("💡 Install with: pip install groundingdino")
            self._try_load_yolo()
            return

        try:
            print(f"🔄 Loading Grounding DINO from .pth file: {model_path}")

            # 如果没有提供config路径，尝试查找
            if config_path is None:
                # 尝试几个常见的配置文件位置
                possible_configs = [
                    "/home/zwk/下载/S3PO-GS-main/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
                ]

                for cfg in possible_configs:
                    if os.path.exists(cfg):
                        config_path = cfg
                        print(f"✅ Found config file: {config_path}")
                        break

                if config_path is None:
                    print("❌ Config file not found. Please provide config_path")
                    print("💡 Download from: https://github.com/IDEA-Research/GroundingDINO")
                    self._try_load_yolo()
                    return

            # 加载模型
            self.model = load_model(config_path, model_path, device=self.device)
            self.use_original = True
            print("✅ Successfully loaded Grounding DINO from .pth file")

        except Exception as e:
            print(f"❌ Failed to load Grounding DINO .pth: {e}")
            self._try_load_yolo()

    def _load_transformers_model(self, model_id):
        """加载Hugging Face transformers版本的模型"""
        if not GROUNDING_DINO_AVAILABLE:
            print("❌ Transformers not available")
            self._try_load_yolo()
            return

        if model_id is None:
            model_id = "IDEA-Research/grounding-dino-tiny"

        try:
            print(f"🔄 Loading Grounding DINO from Hugging Face: {model_id}")
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)
            self.use_transformers = True
            print("✅ Successfully loaded Grounding DINO from Hugging Face")
        except Exception as e:
            print(f"❌ Failed to load from Hugging Face: {e}")
            self._try_load_yolo()

    def _try_load_yolo(self):
        """尝试加载YOLO作为备选检测器"""
        try:
            import ultralytics
            from ultralytics import YOLO

            print("🔄 Trying to load YOLOv8 as fallback...")
            self.yolo_model = YOLO('/home/zwk/下载/S3PO-GS-main/yolo11x.pt')
            self.use_yolo = True
            print("✅ YOLOv8 loaded as fallback detector")
        except Exception as e:
            print(f"❌ Failed to load YOLO: {e}")
            print("⚠️  No detector available! Detection will be disabled.")

    def detect(self, image, text_prompt, confidence_threshold=0.2):
        """统一的检测接口"""
        if self.use_original:
            return self._detect_original(image, text_prompt, confidence_threshold)
        elif self.use_transformers:
            return self._detect_transformers(image, text_prompt, confidence_threshold)
        elif self.use_yolo:
            return self._detect_yolo(image, text_prompt, confidence_threshold)
        else:
            return np.array([]), np.array([]), []

    def _detect_original(self, image, text_prompt, confidence_threshold):
        """使用原始Grounding DINO进行检测"""
        try:
            # 确保输入格式正确
            if isinstance(image, np.ndarray):
                if image.dtype != np.uint8:
                    image = (image * 255).astype(np.uint8)
                # 原始Grounding DINO需要PIL Image
                image_pil = Image.fromarray(image)
            else:
                image_pil = image

            # 图像预处理
            transform = T.Compose([
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            image_tensor, _ = transform(image_pil, None)

            # 预测
            boxes, logits, phrases = predict(
                model=self.model,
                image=image_tensor,
                caption=text_prompt,
                box_threshold=confidence_threshold,
                text_threshold=confidence_threshold,
                device=self.device
            )

            # 转换输出格式
            h, w = image.shape[:2] if isinstance(image, np.ndarray) else image_pil.size[::-1]
            boxes_scaled = boxes * torch.tensor([w, h, w, h], dtype=torch.float32)
            boxes_np = boxes_scaled.cpu().numpy()
            scores_np = logits.cpu().numpy()

            # 解析标签
            labels = []
            for phrase in phrases:
                # phrase格式可能是 "object(0.95)" 这样的
                label = phrase.split('(')[0].strip()
                labels.append(label)

            print(f"🎯 Grounding DINO detected {len(boxes_np)} objects")

            return boxes_np, scores_np, labels

        except Exception as e:
            print(f"❌ Original Grounding DINO detection failed: {e}")
            return np.array([]), np.array([]), []

    def _detect_transformers(self, image, text_prompt, confidence_threshold):
        """使用transformers版本进行检测"""
        try:
            if isinstance(image, np.ndarray):
                if image.dtype != np.uint8:
                    image = (image * 255).astype(np.uint8)
                image_pil = Image.fromarray(image)
            else:
                image_pil = image

            inputs = self.processor(images=image_pil, text=text_prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=confidence_threshold,
                text_threshold=confidence_threshold,
                target_sizes=[image_pil.size[::-1]]
            )[0]

            boxes = results["boxes"].cpu().numpy() if len(results["boxes"]) > 0 else np.array([])
            scores = results["scores"].cpu().numpy() if len(results["scores"]) > 0 else np.array([])
            labels = results["labels"] if "labels" in results else []

            print(f"🎯 Grounding DINO detected {len(boxes)} objects")
            return boxes, scores, labels

        except Exception as e:
            print(f"❌ Transformers detection failed: {e}")
            return np.array([]), np.array([]), []

    def _detect_yolo(self, image, text_prompt, confidence_threshold):
        """使用YOLO进行检测"""
        if not hasattr(self, 'yolo_model'):
            return np.array([]), np.array([]), []

        try:
            # YOLO类别映射
            target_classes = {
                'person': 0, 'bicycle': 1, 'car': 2, 'motorcycle': 3,
                'bus': 5, 'truck': 7, 'traffic light': 9,
                'stop sign': 11, 'bench': 13, 'bird': 14,
                'cat': 15, 'dog': 16, 'horse': 17
            }

            # 解析prompt
            prompt_words = text_prompt.lower().replace('.', '').split()
            relevant_classes = []
            for word in prompt_words:
                if word in target_classes:
                    relevant_classes.append(target_classes[word])
                elif word.rstrip('s') in target_classes:
                    relevant_classes.append(target_classes[word.rstrip('s')])

            # 运行检测
            results = self.yolo_model(image, conf=confidence_threshold)

            boxes = []
            scores = []
            labels = []

            for r in results:
                if r.boxes is not None:
                    for box in r.boxes:
                        class_id = int(box.cls)
                        if not relevant_classes or class_id in relevant_classes:
                            xyxy = box.xyxy[0].cpu().numpy()
                            boxes.append(xyxy)
                            scores.append(float(box.conf))
                            labels.append(self.yolo_model.names[class_id])

            boxes = np.array(boxes) if boxes else np.array([])
            scores = np.array(scores) if scores else np.array([])

            print(f"🎯 YOLO detected {len(boxes)} objects")
            return boxes, scores, labels

        except Exception as e:
            print(f"❌ YOLO detection failed: {e}")
            return np.array([]), np.array([]), []


class EnhancedDynamicObjectMasker:
    """增强的动态物体遮罩器，支持动态和静态对象的完整可视化"""

    def __init__(self, device="cuda", use_sam=True,
                 sam_checkpoint="/home/zwk/下载/S3PO-GS-main/sam_vit_h_4b8939.pth",
                 save_dir=None, save_images=True, use_ground_segmentation=True,
                 scene_type="outdoor_street",
                 grounding_dino_model="/home/zwk/下载/S3PO-GS-main/groundingdino_swint_ogc.pth",
                 grounding_dino_config=None,
                 enable_colorful_vis=True,
                 detect_static_objects=True):
        """
        初始化增强的动态物体遮罩器

        Args:
            detect_static_objects: 是否检测和可视化静态对象
            enable_colorful_vis: 是否启用彩色分割可视化
        """
        self.device = device
        self.initialization_success = True
        self.enable_colorful_vis = enable_colorful_vis
        self.detect_static_objects = detect_static_objects

        # 场景和Prompt管理
        try:
            self.prompt_manager = ScenePromptManager(default_scene=scene_type)
            print(f"✅ Scene prompt manager initialized")
        except Exception as e:
            print(f"❌ Failed to initialize scene manager: {e}")
            self.initialization_success = False

        # Grounding DINO检测器 - 支持.pth文件
        print(f"🔄 Initializing Grounding DINO detector...")
        try:
            # 如果是.pth文件，使用原始加载方式
            if False:
                self.grounding_detector = GroundingDINODetector(
                    model_path='/home/zwk/下载/S3PO-GS-main/groundingdino_swint_ogc.pth',
                    config_path='/home/zwk/下载/S3PO-GS-main/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py',
                    device=device
                )
            else:
                # 否则使用Hugging Face模型
                self.grounding_detector = GroundingDINODetector(
                    model_path=grounding_dino_model,
                    device=device
                )

            print(f"✅ Grounding DINO detector initialized")
        except Exception as e:
            print(f"❌ Failed to initialize Grounding DINO: {e}")
            self.grounding_detector = None
            self.initialization_success = False

        # SAM分割器
        self.use_sam = use_sam
        if use_sam:
            try:
                if os.path.exists(sam_checkpoint):
                    sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
                    sam.to(device=device)
                    self.sam_predictor = SamPredictor(sam)
                    print("✅ SAM model loaded successfully")
                else:
                    print(f"⚠️  SAM checkpoint not found at {sam_checkpoint}")
                    print("⚠️  SAM will be disabled")
                    self.use_sam = False
            except Exception as e:
                print(f"⚠️  Warning: SAM model failed to load ({e})")
                print("⚠️  Will use Grounding DINO boxes only")
                self.use_sam = False

        # 🎨 彩色可视化器
        if self.enable_colorful_vis:
            try:
                self.colorful_visualizer = ColorfulSegmentationVisualizer()
                print("✅ Colorful segmentation visualizer initialized")
            except Exception as e:
                print(f"⚠️  Colorful visualizer failed to initialize: {e}")
                self.enable_colorful_vis = False

        # 地面分割功能
        self.use_ground_segmentation = use_ground_segmentation
        if use_ground_segmentation:
            try:
                self._init_ground_segmentation()
            except Exception as e:
                print(f"⚠️  Ground segmentation initialization failed: {e}")
                self.use_ground_segmentation = False

        # 运动检测参数
        self.prev_frame = None
        self.prev_mask = None
        self.motion_threshold = 3.0

        # 时间一致性参数
        self.mask_history = []
        self.history_length = 5

        # 地面修复参数
        self.inpaint_radius = 3
        self.ground_dilation_kernel = np.ones((7, 7), np.uint8)

        # 图像保存设置
        self.save_images = save_images
        self.save_dir = save_dir if save_dir else "./masked_images"
        if self.save_images:
            try:
                self._create_save_directories()
            except Exception as e:
                print(f"⚠️  Failed to create save directories: {e}")
                self.save_images = False

        # 打印配置信息
        self._print_configuration(grounding_dino_model)
        print(f"🎯 Mode: Dynamic objects will be MASKED OUT (not reconstructed)")
        print(f"✅ Mode: Static objects will be PRESERVED (reconstructed)")
        if self.detect_static_objects:
            print(f"🎨 Mode: Static objects will also be VISUALIZED")

    def _create_save_directories(self):
        """创建保存图像的目录结构，包括动态和静态对象的可视化目录"""
        directories = [
            self.save_dir,
            os.path.join(self.save_dir, "original"),
            os.path.join(self.save_dir, "grounding_dino_detections"),
            os.path.join(self.save_dir, "grounding_dino_masks"),
            os.path.join(self.save_dir, "sam_masks"),
            os.path.join(self.save_dir, "motion_masks"),
            os.path.join(self.save_dir, "ground_masks"),
            os.path.join(self.save_dir, "shadow_regions"),
            os.path.join(self.save_dir, "inpainted_ground"),
            os.path.join(self.save_dir, "final_masks"),
            os.path.join(self.save_dir, "masked_overlay"),
            os.path.join(self.save_dir, "static_only"),
            os.path.join(self.save_dir, "repaired_images"),
            os.path.join(self.save_dir, "keyframes"),
            os.path.join(self.save_dir, "detection_analysis"),

            # 🎨 彩色可视化目录
            os.path.join(self.save_dir, "colorful_combined"),  # 动态+静态组合
            os.path.join(self.save_dir, "colorful_dynamic_only"),  # 仅动态对象
            os.path.join(self.save_dir, "colorful_static_only"),  # 仅静态对象
            os.path.join(self.save_dir, "colorful_overlay"),  # 叠加显示
            os.path.join(self.save_dir, "colorful_legend"),  # 带图例
            os.path.join(self.save_dir, "colorful_analysis"),  # 分析结果
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)

        print(f"📁 Created all directories including colorful visualization in: {self.save_dir}")

    def _print_configuration(self, grounding_dino_model):
        """打印配置信息"""
        try:
            if self.prompt_manager:
                prompt_info = self.prompt_manager.get_detailed_prompt()
                print(f"🎯 Enhanced Dynamic Object Masker Configuration:")
                print(f"  - Scene type: {self.prompt_manager.current_scene}")
                print(f"  - Scene description: {prompt_info['description']}")
                print(f"  - Dynamic objects: {', '.join(prompt_info['dynamic_classes'][:5])}...")
                if self.detect_static_objects:
                    print(f"  - Static objects: {', '.join(prompt_info['static_classes'][:5])}...")
                print(f"  - Confidence threshold: {prompt_info['confidence_threshold']}")
            else:
                print(f"🎯 Enhanced Dynamic Object Masker Configuration:")
                print(f"  - Scene manager: FAILED")

            print(f"  - Grounding DINO model: {grounding_dino_model}")
            print(f"  - Grounding DINO status: {'✅ OK' if self.grounding_detector else '⚠️  FAILED'}")
            print(f"  - SAM enabled: {self.use_sam}")
            print(f"  - Ground segmentation: {self.use_ground_segmentation}")
            print(f"  - Colorful visualization: {self.enable_colorful_vis}")
            print(f"  - Detect static objects: {self.detect_static_objects}")
            print(f"  - Save images: {self.save_images}")

            if not self.initialization_success:
                print(f"⚠️  WARNING: Some components failed to initialize!")
                print(f"  - The system will continue with reduced functionality")
                print(f"  - Consider checking dependencies and model files")

        except Exception as e:
            print(f"❌ Failed to print configuration: {e}")

    def _init_ground_segmentation(self):
        """初始化地面分割模型"""
        try:
            self.ground_segmentation_method = "traditional"
            print("✅ Ground segmentation initialized with traditional method")
        except Exception as e:
            print(f"Warning: Ground segmentation failed: {e}")
            self.ground_segmentation_method = "traditional"

    def set_scene_from_config(self, config):
        """从配置中设置场景类型"""
        scene_hint = config.get("dynamic_filtering", {}).get("scene_type", None)
        data_path = config.get("Dataset", {}).get("dataset_path", "")

        # 优先使用配置中的场景类型
        if scene_hint:
            self.prompt_manager.set_scene(scene_hint)
        # 其次尝试从数据路径推断
        elif data_path:
            self.prompt_manager.detect_scene_from_path(data_path)

        # 更新检测阈值
        prompt_info = self.prompt_manager.get_detailed_prompt()
        print(f"🎬 Scene configuration updated:")
        print(f"  - Active scene: {self.prompt_manager.current_scene}")
        print(f"  - Confidence threshold: {prompt_info['confidence_threshold']}")

    def segment_ground(self, image):
        """分割图像中的地面区域"""
        if self.ground_segmentation_method == "traditional":
            return self._traditional_ground_segmentation(image)
        else:
            return self._ml_ground_segmentation(image)

    def _traditional_ground_segmentation(self, image):
        """基于传统方法的地面分割"""
        h, w = image.shape[:2]
        ground_mask = np.zeros((h, w), dtype=np.uint8)

        # 转换到HSV空间进行颜色分析
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # 1. 基于位置的先验：地面通常在图像下半部分
        ground_region_y_start = int(h * 0.6)

        # 2. 在下半部分进行颜色聚类
        lower_region = image[ground_region_y_start:, :, :]
        lower_gray = gray[ground_region_y_start:, :]

        # 3. 基于颜色一致性检测地面
        kernel_size = 15
        blur_gray = cv2.GaussianBlur(lower_gray, (kernel_size, kernel_size), 0)
        grad_x = cv2.Sobel(blur_gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blur_gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

        # 地面区域通常梯度较小
        texture_threshold = np.percentile(gradient_magnitude, 30)
        smooth_regions = (gradient_magnitude < texture_threshold).astype(np.uint8)

        # 4. 基于颜色聚类检测主要地面颜色
        mean_color = np.mean(lower_region, axis=(0, 1))
        color_distances = np.linalg.norm(lower_region - mean_color, axis=2)
        color_threshold = np.std(color_distances) * 1.5
        color_mask = (color_distances < color_threshold).astype(np.uint8)

        # 5. 结合纹理和颜色信息
        combined_mask = np.logical_and(smooth_regions, color_mask).astype(np.uint8)

        # 6. 形态学操作清理mask
        kernel = np.ones((5, 5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

        # 7. 将结果映射回完整图像
        ground_mask[ground_region_y_start:, :] = combined_mask

        # 8. 向上扩展地面区域
        if np.sum(combined_mask) > 0:
            ground_mask = self._extend_ground_upward(image, ground_mask, ground_region_y_start)

        return ground_mask

    def _extend_ground_upward(self, image, initial_ground_mask, start_y):
        """向上扩展地面区域，基于颜色相似性"""
        h, w = image.shape[:2]
        extended_mask = initial_ground_mask.copy()

        # 获取已知地面区域的平均颜色
        ground_pixels = image[initial_ground_mask > 0]
        if len(ground_pixels) == 0:
            return initial_ground_mask

        mean_ground_color = np.mean(ground_pixels, axis=0)
        color_std = np.std(ground_pixels, axis=0)

        # 向上逐行检查
        for y in range(start_y - 1, max(int(h * 0.3), 0), -1):
            row_colors = image[y, :, :]
            color_distances = np.linalg.norm(row_colors - mean_ground_color, axis=1)

            threshold = np.linalg.norm(color_std) * 2
            similar_pixels = color_distances < threshold

            # 只保留与下方地面区域连通的像素
            if y < h - 1:
                below_mask = extended_mask[y + 1, :]
                # 确保below_mask是一维的，并进行膨胀操作
                below_mask_2d = below_mask.reshape(1, -1)  # 转换为 [1, W] 形状进行膨胀
                dilated_below = cv2.dilate(below_mask_2d.astype(np.uint8), np.ones((1, 3), np.uint8))
                dilated_below_1d = dilated_below.reshape(-1) > 0  # 转回一维并转为布尔值

                # 确保维度匹配
                connected_pixels = np.logical_and(similar_pixels, dilated_below_1d)
                extended_mask[y, :] = connected_pixels.astype(np.uint8)

        return extended_mask

    def _ml_ground_segmentation(self, image):
        """基于机器学习模型的地面分割（预留接口）"""
        h, w = image.shape[:2]
        return np.zeros((h, w), dtype=np.uint8)

    def repair_ground_shadows(self, image, vehicle_mask, ground_mask):
        """修复车辆在地面上的阴影/鬼影"""
        # 1. 检测车辆mask与地面的交集（潜在阴影区域）
        shadow_regions = np.logical_and(vehicle_mask, ground_mask).astype(np.uint8)

        if np.sum(shadow_regions) == 0:
            return image.copy(), shadow_regions

        # 2. 扩展阴影区域，包含可能的边缘效应
        kernel = self.ground_dilation_kernel
        expanded_shadow = cv2.dilate(shadow_regions, kernel, iterations=1)

        # 3. 确保扩展区域仍在地面内
        final_shadow_regions = np.logical_and(expanded_shadow, ground_mask).astype(np.uint8)

        # 4. 创建修复mask
        inpaint_mask = final_shadow_regions.astype(np.uint8) * 255

        # 5. 使用图像修复算法
        repaired_image = self._inpaint_ground_region(image, inpaint_mask, ground_mask)

        return repaired_image, final_shadow_regions

    def _inpaint_ground_region(self, image, inpaint_mask, ground_mask):
        """对地面区域进行图像修复"""
        try:
            repaired = cv2.inpaint(image, inpaint_mask, self.inpaint_radius, cv2.INPAINT_TELEA)
            return repaired
        except:
            return self._simple_ground_inpaint(image, inpaint_mask, ground_mask)

    def _simple_ground_inpaint(self, image, inpaint_mask, ground_mask):
        """简单的地面修复：用周围地面像素的均值填充"""
        repaired = image.copy()
        h, w = image.shape[:2]

        repair_coords = np.where(inpaint_mask > 0)
        if len(repair_coords[0]) == 0:
            return repaired

        for i in range(len(repair_coords[0])):
            y, x = repair_coords[0][i], repair_coords[1][i]

            search_radius = 10
            y1 = max(0, y - search_radius)
            y2 = min(h, y + search_radius + 1)
            x1 = max(0, x - search_radius)
            x2 = min(w, x + search_radius + 1)

            search_region_ground = ground_mask[y1:y2, x1:x2]
            search_region_inpaint = inpaint_mask[y1:y2, x1:x2]

            valid_ground = np.logical_and(search_region_ground, search_region_inpaint == 0)

            if np.sum(valid_ground) > 0:
                search_region_image = image[y1:y2, x1:x2]
                valid_pixels = search_region_image[valid_ground]
                mean_color = np.mean(valid_pixels, axis=0)

                noise = np.random.normal(0, 5, 3)
                final_color = np.clip(mean_color + noise, 0, 255)
                repaired[y, x] = final_color.astype(np.uint8)

        return repaired

    def create_colorful_visualization(self, image, frame_idx,
                                      dynamic_boxes, dynamic_labels, dynamic_scores,
                                      static_boxes, static_labels, static_scores,
                                      dynamic_sam_masks=None, static_sam_masks=None):
        """🎨 创建动态和静态对象的彩色可视化 - 使用修复版"""
        if not self.save_images or not self.enable_colorful_vis:
            return {}

        try:
            h, w = image.shape[:2]

            # 🔍 添加调试信息
            print(f"\n🎨 开始创建彩色可视化 - Frame {frame_idx}")
            self.colorful_visualizer.debug_detection_results(
                dynamic_boxes, dynamic_labels, static_boxes, static_labels, (h, w)
            )

            # 1. 创建组合彩色分割mask（使用修复版的坐标转换）
            combined_mask, dynamic_color_map, static_color_map, class_mask = self.colorful_visualizer.create_combined_segmentation_mask(
                (h, w), dynamic_boxes, dynamic_labels, dynamic_scores,
                static_boxes, static_labels, static_scores,
                dynamic_sam_masks, static_sam_masks
            )

            # 2. 验证生成的mask
            dynamic_pixels = np.sum(combined_mask[:, :, 0] > 0) if len(dynamic_color_map) > 0 else 0
            static_pixels = np.sum(combined_mask[:, :, 1] > 0) if len(static_color_map) > 0 else 0
            total_colored_pixels = np.sum(np.any(combined_mask > 0, axis=2))

            print(f"  📊 生成的mask统计:")
            print(f"     - 总彩色像素: {total_colored_pixels}")
            print(f"     - 动态区域像素: {dynamic_pixels}")
            print(f"     - 静态区域像素: {static_pixels}")
            print(f"     - 覆盖率: {total_colored_pixels / (h * w) * 100:.1f}%")

            # 3. 创建分离的可视化
            dynamic_only_mask, static_only_mask = self.colorful_visualizer.create_separate_visualizations(
                (h, w), dynamic_boxes, dynamic_labels, dynamic_scores,
                static_boxes, static_labels, static_scores,
                dynamic_sam_masks, static_sam_masks
            )

            # 4. 创建叠加可视化
            overlay_image = self.colorful_visualizer.create_overlay_visualization(
                image, combined_mask, alpha=0.6
            )

            # 5. 创建带图例的可视化
            legend_image = self.colorful_visualizer.add_combined_legend(
                overlay_image, dynamic_color_map, static_color_map,
                dynamic_scores, dynamic_labels, static_scores, static_labels
            )

            # 6. 保存所有可视化结果
            self._save_colorful_results(
                frame_idx, image, combined_mask, dynamic_only_mask, static_only_mask,
                overlay_image, legend_image, dynamic_color_map, static_color_map,
                dynamic_labels, dynamic_scores, static_labels, static_scores, class_mask
            )

            print(f"✅ 彩色可视化创建完成 - Frame {frame_idx}")
            print(f"   - 动态类别: {list(dynamic_color_map.keys())}")
            print(f"   - 静态类别: {list(static_color_map.keys())}")

            return {
                'combined_mask': combined_mask,
                'dynamic_only_mask': dynamic_only_mask,
                'static_only_mask': static_only_mask,
                'overlay_image': overlay_image,
                'legend_image': legend_image,
                'dynamic_color_map': dynamic_color_map,
                'static_color_map': static_color_map,
                'class_mask': class_mask
            }

        except Exception as e:
            print(f"❌ 彩色可视化创建失败 - Frame {frame_idx}: {e}")
            import traceback
            traceback.print_exc()
            return {}
    def _save_colorful_results(self, frame_idx, original_image, combined_mask,
                               dynamic_only_mask, static_only_mask, overlay_image,
                               legend_image, dynamic_color_map, static_color_map,
                               dynamic_labels, dynamic_scores, static_labels, static_scores, class_mask):
        """保存彩色可视化结果"""
        try:
            # 保存组合彩色mask
            cv2.imwrite(
                os.path.join(self.save_dir, "colorful_combined", f"frame_{frame_idx:06d}_combined.jpg"),
                combined_mask
            )

            # 保存动态对象mask
            cv2.imwrite(
                os.path.join(self.save_dir, "colorful_dynamic_only", f"frame_{frame_idx:06d}_dynamic.jpg"),
                dynamic_only_mask
            )

            # 保存静态对象mask
            cv2.imwrite(
                os.path.join(self.save_dir, "colorful_static_only", f"frame_{frame_idx:06d}_static.jpg"),
                static_only_mask
            )

            # 保存叠加图像
            cv2.imwrite(
                os.path.join(self.save_dir, "colorful_overlay", f"frame_{frame_idx:06d}_overlay.jpg"),
                overlay_image
            )

            # 保存带图例的图像
            cv2.imwrite(
                os.path.join(self.save_dir, "colorful_legend", f"frame_{frame_idx:06d}_legend.jpg"),
                legend_image
            )

            # 保存详细分析
            self._create_detailed_analysis(frame_idx, dynamic_color_map, static_color_map,
                                           dynamic_labels, dynamic_scores, static_labels, static_scores, class_mask)

        except Exception as e:
            print(f"❌ Failed to save colorful results for frame {frame_idx}: {e}")

    def _create_detailed_analysis(self, frame_idx, dynamic_color_map, static_color_map,
                                  dynamic_labels, dynamic_scores, static_labels, static_scores, class_mask):
        """创建详细的颜色分析报告"""
        try:
            analysis_path = os.path.join(self.save_dir, "colorful_analysis", f"frame_{frame_idx:06d}_analysis.txt")

            # 计算每个类别的像素统计
            total_pixels = class_mask.size
            class_stats = {}

            # 动态对象统计
            for i, label in enumerate(dynamic_labels):
                class_id = i + 1  # 类别ID从1开始
                pixel_count = np.sum(class_mask == class_id)
                percentage = (pixel_count / total_pixels) * 100
                class_stats[f"dynamic_{label}"] = {
                    'pixel_count': pixel_count,
                    'percentage': percentage,
                    'confidence': dynamic_scores[i],
                    'type': 'dynamic'
                }

            # 静态对象统计
            start_id = len(dynamic_labels) + 1
            for i, label in enumerate(static_labels):
                class_id = start_id + i
                pixel_count = np.sum(class_mask == class_id)
                percentage = (pixel_count / total_pixels) * 100
                class_stats[f"static_{label}"] = {
                    'pixel_count': pixel_count,
                    'percentage': percentage,
                    'confidence': static_scores[i],
                    'type': 'static'
                }

            # 保存分析报告
            with open(analysis_path, 'w') as f:
                f.write(f"Frame {frame_idx} Colorful Segmentation Analysis\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Scene Type: {self.prompt_manager.current_scene}\n")
                f.write(f"Total Dynamic Classes: {len(dynamic_color_map)}\n")
                f.write(f"Total Static Classes: {len(static_color_map)}\n\n")

                # 动态对象颜色映射
                f.write("Dynamic Objects Color Mapping (BGR):\n")
                f.write("-" * 40 + "\n")
                for class_name, color in dynamic_color_map.items():
                    f.write(f"{class_name:20}: BGR({color[0]:3d}, {color[1]:3d}, {color[2]:3d})\n")

                f.write("\nStatic Objects Color Mapping (BGR):\n")
                f.write("-" * 40 + "\n")
                for class_name, color in static_color_map.items():
                    f.write(f"{class_name:20}: BGR({color[0]:3d}, {color[1]:3d}, {color[2]:3d})\n")

                # 像素覆盖统计
                f.write("\nPixel Coverage Statistics:\n")
                f.write("-" * 40 + "\n")
                sorted_stats = sorted(class_stats.items(), key=lambda x: x[1]['pixel_count'], reverse=True)

                for rank, (class_name, stats) in enumerate(sorted_stats, 1):
                    type_indicator = "🔴" if stats['type'] == 'dynamic' else "🔵"
                    f.write(f"{rank:2d}. {type_indicator} {class_name:20}: {stats['pixel_count']:6d} pixels "
                            f"({stats['percentage']:5.2f}%) - Conf: {stats['confidence']:.3f}\n")

                # 总体统计
                dynamic_pixels = sum(s['pixel_count'] for s in class_stats.values() if s['type'] == 'dynamic')
                static_pixels = sum(s['pixel_count'] for s in class_stats.values() if s['type'] == 'static')

                f.write(f"\nOverall Statistics:\n")
                f.write("-" * 20 + "\n")
                f.write(f"Total Dynamic Pixels: {dynamic_pixels} ({dynamic_pixels / total_pixels * 100:.2f}%)\n")
                f.write(f"Total Static Pixels: {static_pixels} ({static_pixels / total_pixels * 100:.2f}%)\n")
                f.write(
                    f"Total Segmented: {dynamic_pixels + static_pixels} ({(dynamic_pixels + static_pixels) / total_pixels * 100:.2f}%)\n")
                f.write(f"Background Pixels: {total_pixels - dynamic_pixels - static_pixels} "
                        f"({(total_pixels - dynamic_pixels - static_pixels) / total_pixels * 100:.2f}%)\n")

        except Exception as e:
            print(f"❌ Failed to create detailed analysis for frame {frame_idx}: {e}")

    def detect_and_segment(self, image, frame_idx=None):
        """
        使用Grounding DINO检测动态和静态物体并生成精确分割mask，包含完整彩色可视化

        Returns:
            final_mask: 动态物体mask (1=dynamic, 0=static)
            max_confidence: 最高检测置信度
            repaired_image: 修复后的图像
            colorful_results: 彩色可视化结果字典
        """
        h, w = image.shape[:2]
        final_mask = np.zeros((h, w), dtype=np.uint8)
        grounding_dino_mask = np.zeros((h, w), dtype=np.uint8)
        repaired_image = image.copy()
        max_confidence = 0.0
        colorful_results = {}
        self._last_detected_objects = []

        # 检查是否应该使用fallback模式
        if not self.grounding_detector or not self.initialization_success:
            print(f"⚠️  Using fallback mode for frame {frame_idx}")
            return final_mask, 0.0, repaired_image, colorful_results

        # 1. 地面分割（可选）
        ground_mask = None
        if self.use_ground_segmentation:
            try:
                ground_mask = self.segment_ground(image)
                print(f"🌍 Ground segmentation: {np.sum(ground_mask)} pixels detected as ground")
            except Exception as e:
                print(f"❌ Ground segmentation failed: {e}")

        # 2. 获取当前场景的prompt和阈值
        try:
            dynamic_prompt, confidence_threshold = self.prompt_manager.get_current_prompt()
            static_prompt, _ = self.prompt_manager.get_static_prompt()
            print(f"🎯 Dynamic prompt: '{dynamic_prompt[:30]}...' (confidence: {confidence_threshold})")
            if self.detect_static_objects and static_prompt:
                print(f"🏗️  Static prompt: '{static_prompt[:30]}...'")
        except Exception as e:
            print(f"❌ Failed to get prompts: {e}")
            dynamic_prompt = "car. truck. person. bicycle"
            static_prompt = "building. tree. road. wall"
            confidence_threshold = 0.2

        # 3. 检测动态对象
        dynamic_boxes, dynamic_scores, dynamic_labels = np.array([]), np.array([]), []
        try:
            dynamic_boxes, dynamic_scores, dynamic_labels = self.grounding_detector.detect(
                image, dynamic_prompt, confidence_threshold
            )

            # 🔍 添加坐标调试
            if len(dynamic_boxes) > 0:
                print(f"\n🔍 动态对象检测结果调试:")
                for i, (box, label, score) in enumerate(zip(dynamic_boxes[:3], dynamic_labels[:3], dynamic_scores[:3])):
                    print(f"  {i + 1}. {label} (conf: {score:.3f})")
                    if hasattr(box, 'shape'):
                        print(f"     原始box: {box}")
                    else:
                        print(f"     原始box: {np.array(box)}")

        except Exception as e:
            print(f"❌ 动态对象检测失败: {e}")

        # 4. 检测静态对象（如果启用）
        static_boxes, static_scores, static_labels = np.array([]), np.array([]), []
        if self.detect_static_objects and static_prompt:
            try:
                static_boxes, static_scores, static_labels = self.grounding_detector.detect(
                    image, static_prompt, confidence_threshold
                )
                print(f"🏗️  检测到 {len(static_boxes)} 个静态对象")

                # 调试静态对象坐标
                if len(static_boxes) > 0:
                    print(f"🔍 静态对象检测结果调试:")
                    for i, (box, label, score) in enumerate(
                            zip(static_boxes[:3], static_labels[:3], static_scores[:3])):
                        print(f"  {i + 1}. {label} (conf: {score:.3f})")
                        print(f"     原始box: {np.array(box) if not hasattr(box, 'shape') else box}")

            except Exception as e:
                print(f"❌ 静态对象检测失败: {e}")

        if len(dynamic_boxes) == 0 and len(static_boxes) == 0:
            print(f"ℹ️  No objects detected in frame {frame_idx}")
            if frame_idx is not None and self.save_images:
                self.save_detection_results(
                    image, frame_idx,
                    grounding_dino_mask=grounding_dino_mask,
                    final_mask=final_mask,
                    boxes=[], labels=[], scores=[],
                    ground_mask=ground_mask,
                    repaired_image=repaired_image
                )
            return final_mask, 0.0, repaired_image, colorful_results

        # 5. 处理动态对象检测结果，创建动态物体mask
        vehicle_detected = False
        if len(dynamic_boxes) > 0:
            print(f"🎯 Detected {len(dynamic_boxes)} dynamic objects:")

            for i, (box, score, label) in enumerate(zip(dynamic_boxes, dynamic_scores, dynamic_labels)):
                box_tensor = torch.tensor(box, dtype=torch.float32)
                box_xyxy = box_ops.box_cxcywh_to_xyxy(box_tensor.unsqueeze(0)) * torch.tensor([w, h, w, h])
                x1, y1, x2, y2 = box_xyxy[0].cpu().numpy().astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                max_confidence = max(max_confidence, score)

                self._last_detected_objects.append({
                    'label': label,
                    'box': box,
                    'score': score,
                    'frame': frame_idx,
                    'type': 'dynamic'
                })

                # 检查是否是车辆
                vehicle_keywords = ["car", "truck", "bus", "vehicle", "van", "suv", "motorcycle", "bike"]
                if any(keyword in label.lower() for keyword in vehicle_keywords):
                    vehicle_detected = True
                    # 对车辆扩展边界框
                    width = x2 - x1
                    height = y2 - y1
                    expand_w = int(width * 0.1)
                    expand_h = int(height * 0.1)

                    x1 = max(0, x1 - expand_w)
                    y1 = max(0, y1 - expand_h)
                    x2 = min(w, x2 + expand_w)
                    y2 = min(h, y2 + expand_h)

                    print(f"  🚗 Vehicle '{label}' (conf={score:.3f}) - expanded mask")
                else:
                    print(f"  👤 Object '{label}' (conf={score:.3f})")

                # 标记动态区域
                grounding_dino_mask[y1:y2, x1:x2] = 1

            final_mask = grounding_dino_mask.copy()

        # 6. 处理静态对象检测结果（用于可视化，不影响final_mask）
        if len(static_boxes) > 0:
            print(f"🏗️  Processing {len(static_boxes)} static objects for visualization:")
            for i, (box, score, label) in enumerate(zip(static_boxes, static_scores, static_labels)):
                self._last_detected_objects.append({
                    'label': label,
                    'box': box,
                    'score': score,
                    'frame': frame_idx,
                    'type': 'static'
                })
                print(f"  🏢 Static '{label}' (conf={score:.3f})")

        # 7. SAM精确分割（处理动态和静态对象）
        dynamic_sam_masks = []
        static_sam_masks = []

        if self.use_sam:
            self.sam_predictor.set_image(image)

            # SAM处理动态对象
            if len(dynamic_boxes) > 0:
                try:
                    sam_combined_mask = np.zeros((h, w), dtype=np.uint8)

                    for i, (box, score, label) in enumerate(zip(dynamic_boxes, dynamic_scores, dynamic_labels)):
                        try:
                            box_tensor = torch.tensor(box, dtype=torch.float32)
                            box_xyxy = box_ops.box_cxcywh_to_xyxy(box_tensor.unsqueeze(0)) * torch.tensor([w, h, w, h])
                            x1, y1, x2, y2 = box_xyxy[0].cpu().numpy().astype(int)

                            # 车辆扩展处理
                            vehicle_keywords = ["car", "truck", "bus", "vehicle", "van", "suv", "motorcycle", "bike"]
                            if any(keyword in label.lower() for keyword in vehicle_keywords):
                                width = x2 - x1
                                height = y2 - y1
                                expand_w = int(width * 0.1)
                                expand_h = int(height * 0.1)
                                x1 = max(0, x1 - expand_w)
                                y1 = max(0, y1 - expand_h)
                                x2 = min(w, x2 + expand_w)
                                y2 = min(h, y2 + expand_h)

                            input_box = np.array([x1, y1, x2, y2])
                            masks, sam_scores, _ = self.sam_predictor.predict(
                                point_coords=None,
                                point_labels=None,
                                box=input_box[None, :],
                                multimask_output=False,
                            )

                            if len(masks) > 0:
                                best_mask = masks[0].astype(np.uint8)
                                dynamic_sam_masks.append(best_mask)
                                sam_combined_mask = np.logical_or(sam_combined_mask, best_mask).astype(np.uint8)
                                print(f"  ✅ SAM refined dynamic '{label}' mask")
                            else:
                                dynamic_sam_masks.append(None)
                        except Exception as e:
                            print(f"  ❌ SAM failed for dynamic '{label}': {e}")
                            dynamic_sam_masks.append(None)

                    if sam_combined_mask.sum() > 0:
                        final_mask = sam_combined_mask
                        print(f"🎯 SAM refinement complete for dynamic objects")

                except Exception as e:
                    print(f"❌ SAM processing failed for dynamic objects: {e}")

            # SAM处理静态对象（仅用于可视化）
            if len(static_boxes) > 0:
                try:
                    for i, (box, score, label) in enumerate(zip(static_boxes, static_scores, static_labels)):
                        try:
                            box_tensor = torch.tensor(box, dtype=torch.float32)
                            box_xyxy = box_ops.box_cxcywh_to_xyxy(box_tensor.unsqueeze(0)) * torch.tensor([w, h, w, h])
                            x1, y1, x2, y2 = box_xyxy[0].cpu().numpy().astype(int)
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(w, x2), min(h, y2)

                            input_box = np.array([x1, y1, x2, y2])
                            masks, sam_scores, _ = self.sam_predictor.predict(
                                point_coords=None,
                                point_labels=None,
                                box=input_box[None, :],
                                multimask_output=False,
                            )

                            if len(masks) > 0:
                                best_mask = masks[0].astype(np.uint8)
                                static_sam_masks.append(best_mask)
                                print(f"  ✅ SAM refined static '{label}' mask")
                            else:
                                static_sam_masks.append(None)
                        except Exception as e:
                            print(f"  ❌ SAM failed for static '{label}': {e}")
                            static_sam_masks.append(None)

                except Exception as e:
                    print(f"❌ SAM processing failed for static objects: {e}")

        # 8. 🎨 创建彩色可视化（包含动态和静态对象）
        if self.enable_colorful_vis:
            try:
                colorful_results = self.create_colorful_visualization(
                    image, frame_idx,
                    dynamic_boxes, dynamic_labels, dynamic_scores,
                    static_boxes, static_labels, static_scores,
                    dynamic_sam_masks if dynamic_sam_masks else None,
                    static_sam_masks if static_sam_masks else None
                )
            except Exception as e:
                print(f"❌ 彩色可视化失败: {e}")
                import traceback
                traceback.print_exc()

        # 9. 地面修复（如果检测到车辆）
        shadow_regions = None
        if self.use_ground_segmentation and ground_mask is not None and vehicle_detected:
            try:
                repaired_image, shadow_regions = self.repair_ground_shadows(image, final_mask, ground_mask)
                print(f"🔧 Ground repair: {np.sum(shadow_regions) if shadow_regions is not None else 0} pixels")
            except Exception as e:
                print(f"❌ Ground repair failed: {e}")

        # 10. 时间一致性滤波
        try:
            final_mask = self._temporal_consistency(final_mask)
        except Exception as e:
            print(f"❌ Temporal consistency failed: {e}")

        # 11. 最终的膨胀处理
        if final_mask.sum() > 0:
            kernel = np.ones((5, 5), np.uint8)
            final_mask = cv2.dilate(final_mask, kernel, iterations=1)
            print(f"🛡️  Applied final dilation to dynamic mask")

        # 12. 保存结果
        if frame_idx is not None and self.save_images:
            self.save_detection_results(
                image, frame_idx,
                grounding_dino_mask=grounding_dino_mask,
                final_mask=final_mask,
                boxes=dynamic_boxes,  # 保存动态检测框
                labels=dynamic_labels,
                scores=dynamic_scores,
                ground_mask=ground_mask,
                shadow_regions=shadow_regions,
                repaired_image=repaired_image
            )

        # 输出统计
        print(f"📊 Frame {frame_idx} final statistics:")
        print(f"  - Dynamic objects: {len(dynamic_boxes)} detected")
        print(f"  - Static objects: {len(static_boxes)} detected")
        print(f"  - Total dynamic pixels: {final_mask.sum()} ({np.mean(final_mask) * 100:.1f}% of image)")
        print(f"  - Max confidence: {max_confidence:.3f}")
        if self.enable_colorful_vis and colorful_results:
            total_classes = len(colorful_results.get('dynamic_color_map', {})) + len(
                colorful_results.get('static_color_map', {}))
            print(f"  - Colorful visualization: ✅ Created with {total_classes} classes")

        return final_mask, max_confidence, repaired_image, colorful_results

    def debug_coordinate_conversion():
        """调试坐标转换的独立测试函数"""
        visualizer = ColorfulSegmentationVisualizer()

        # 测试各种坐标格式
        test_cases = [
            # 归一化 cxcywh
            ([0.5, 0.5, 0.2, 0.3], "归一化 cxcywh"),
            # 像素 xyxy
            ([100, 100, 200, 200], "像素 xyxy"),
            # 像素 cxcywh
            ([150, 150, 100, 100], "像素 cxcywh"),
            # torch tensor
            (torch.tensor([0.3, 0.4, 0.1, 0.2]), "torch归一化"),
        ]

        w, h = 640, 480
        print(f"🔍 测试图像尺寸: {w} x {h}")

        for box, desc in test_cases:
            try:
                x1, y1, x2, y2 = visualizer._convert_box_coordinates(box, w, h)
                print(f"✅ {desc}: {box} -> ({x1},{y1},{x2},{y2})")
            except Exception as e:
                print(f"❌ {desc}: {box} -> 转换失败: {e}")
    def get_static_mask_for_gaussian_init(self, image, frame_idx=None):
        """
        为高斯体初始化获取静态区域mask，现在包含完整的动静态可视化
        返回的static_mask中：
        - 1 表示静态区域（应该被重建）
        - 0 表示动态区域（应该被mask掉）

        Returns:
            static_mask: 静态区域mask (1=static, 0=dynamic)
            repaired_image: 用于初始化的修复图像
            detected_objects: 检测到的所有物体列表（动态+静态）
            colorful_results: 彩色可视化结果
        """
        # 获取动态物体mask和修复图像，以及完整的可视化结果
        dynamic_mask, confidence, repaired_image, colorful_results = self.detect_and_segment(image, frame_idx)

        # 静态mask是动态mask的反向
        # dynamic_mask中 1=动态物体，0=静态背景
        # static_mask中 1=静态背景，0=动态物体
        static_mask = (1 - dynamic_mask).astype(np.uint8)

        # 获取检测到的物体信息（包含动态和静态）
        detected_objects = []
        if hasattr(self, '_last_detected_objects'):
            detected_objects = self._last_detected_objects

        print(f"📊 Frame {frame_idx} mask summary:")
        print(f"  - Dynamic pixels (masked out): {np.sum(dynamic_mask > 0)} ({np.mean(dynamic_mask) * 100:.1f}%)")
        print(f"  - Static pixels (reconstructed): {np.sum(static_mask > 0)} ({np.mean(static_mask) * 100:.1f}%)")

        # 统计动态和静态对象
        dynamic_count = len([obj for obj in detected_objects if obj.get('type') == 'dynamic'])
        static_count = len([obj for obj in detected_objects if obj.get('type') == 'static'])
        print(f"  - Total objects detected: {len(detected_objects)} (Dynamic: {dynamic_count}, Static: {static_count})")

        return static_mask, repaired_image, detected_objects, colorful_results

    def save_detection_results(self, image, frame_idx, grounding_dino_mask=None, sam_mask=None,
                               motion_mask=None, final_mask=None, boxes=None, labels=None, scores=None,
                               ground_mask=None, shadow_regions=None, repaired_image=None):
        """保存Grounding DINO检测和分割的各种结果"""
        if not self.save_images:
            return

        try:
            # 确保图像是正确的格式
            if isinstance(image, np.ndarray):
                if image.dtype != np.uint8:
                    image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
                img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if len(image.shape) == 3 else image
            else:
                return

            # 1. 保存原始图像
            original_path = os.path.join(self.save_dir, "original", f"frame_{frame_idx:06d}.jpg")
            cv2.imwrite(original_path, img_bgr)

            # 2. 保存Grounding DINO检测框和标签
            if boxes is not None and len(boxes) > 0:
                detection_img = img_bgr.copy()
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box)
                    # 绘制检测框
                    cv2.rectangle(detection_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # 添加标签和置信度
                    if labels and i < len(labels):
                        label_text = labels[i]
                        if scores is not None and i < len(scores):
                            label_text += f" {scores[i]:.2f}"

                        # 绘制标签背景
                        (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(detection_img, (x1, y1 - text_height - 5), (x1 + text_width, y1), (0, 255, 0), -1)
                        cv2.putText(detection_img, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0),
                                    1)

                detection_path = os.path.join(self.save_dir, "grounding_dino_detections", f"frame_{frame_idx:06d}.jpg")
                cv2.imwrite(detection_path, detection_img)

            # 3. 保存各种mask
            if grounding_dino_mask is not None:
                grounding_mask_img = (grounding_dino_mask * 255).astype(np.uint8)
                grounding_path = os.path.join(self.save_dir, "grounding_dino_masks", f"frame_{frame_idx:06d}.jpg")
                cv2.imwrite(grounding_path, grounding_mask_img)

            if sam_mask is not None:
                sam_mask_img = (sam_mask * 255).astype(np.uint8)
                sam_path = os.path.join(self.save_dir, "sam_masks", f"frame_{frame_idx:06d}.jpg")
                cv2.imwrite(sam_path, sam_mask_img)

            if motion_mask is not None:
                motion_mask_img = (motion_mask * 255).astype(np.uint8)
                motion_path = os.path.join(self.save_dir, "motion_masks", f"frame_{frame_idx:06d}.jpg")
                cv2.imwrite(motion_path, motion_mask_img)

            # 4. 保存地面相关结果
            if ground_mask is not None:
                ground_mask_img = (ground_mask * 255).astype(np.uint8)
                ground_path = os.path.join(self.save_dir, "ground_masks", f"frame_{frame_idx:06d}.jpg")
                cv2.imwrite(ground_path, ground_mask_img)

            if shadow_regions is not None:
                shadow_mask_img = (shadow_regions * 255).astype(np.uint8)
                shadow_path = os.path.join(self.save_dir, "shadow_regions", f"frame_{frame_idx:06d}.jpg")
                cv2.imwrite(shadow_path, shadow_mask_img)

            if repaired_image is not None:
                repaired_bgr = cv2.cvtColor(repaired_image, cv2.COLOR_RGB2BGR)
                repaired_path = os.path.join(self.save_dir, "repaired_images", f"frame_{frame_idx:06d}.jpg")
                cv2.imwrite(repaired_path, repaired_bgr)

            # 5. 保存最终mask和组合显示
            if final_mask is not None:
                final_mask_img = (final_mask * 255).astype(np.uint8)
                final_path = os.path.join(self.save_dir, "final_masks", f"frame_{frame_idx:06d}.jpg")
                cv2.imwrite(final_path, final_mask_img)

                # 叠加显示（动态区域用红色，地面用绿色）
                overlay_img = img_bgr.copy()
                if ground_mask is not None:
                    overlay_img[ground_mask > 0] = [0, 255, 0]  # 地面用绿色
                overlay_img[final_mask > 0] = [0, 0, 255]  # 动态物体用红色
                overlay_path = os.path.join(self.save_dir, "masked_overlay", f"frame_{frame_idx:06d}.jpg")
                cv2.imwrite(overlay_path, overlay_img)

                # 静态区域图像
                static_img = repaired_bgr.copy() if repaired_image is not None else img_bgr.copy()
                static_img[final_mask > 0] = [0, 0, 0]
                static_path = os.path.join(self.save_dir, "static_only", f"frame_{frame_idx:06d}.jpg")
                cv2.imwrite(static_path, static_img)

            # 6. 保存检测分析结果
            if boxes is not None and labels is not None:
                analysis_path = os.path.join(self.save_dir, "detection_analysis", f"frame_{frame_idx:06d}.txt")
                with open(analysis_path, 'w') as f:
                    f.write(f"Frame {frame_idx} Detection Analysis\n")
                    f.write(f"Scene Type: {self.prompt_manager.current_scene}\n")
                    f.write(f"Dynamic Prompt Used: {self.prompt_manager.get_current_prompt()[0]}\n")
                    f.write(f"Total Detections: {len(boxes)}\n\n")

                    for i, (box, label) in enumerate(zip(boxes, labels)):
                        score = scores[i] if scores is not None and i < len(scores) else 0.0
                        f.write(f"Detection {i + 1}:\n")
                        f.write(f"  Label: {label}\n")
                        f.write(f"  Confidence: {score:.3f}\n")
                        f.write(f"  Box: [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]\n\n")

            print(f"💾 Saved all detection results for frame {frame_idx}")

        except Exception as e:
            print(f"❌ Warning: Failed to save detection results for frame {frame_idx}: {e}")

    def _temporal_consistency(self, current_mask):
        """时间一致性滤波，减少mask的闪烁"""
        self.mask_history.append(current_mask.copy())

        if len(self.mask_history) > self.history_length:
            self.mask_history.pop(0)

        if len(self.mask_history) < 3:
            return current_mask

        # 使用历史mask的中位数滤波
        mask_stack = np.stack(self.mask_history, axis=0)
        consistent_mask = np.median(mask_stack, axis=0).astype(np.uint8)

        return consistent_mask


# 修改FrontEnd类以使用新的检测器
class FrontEnd(mp.Process):
    def __init__(self, config, model, save_dir=None):
        super().__init__()
        self.config = config
        self.background = None
        self.pipeline_params = None
        self.frontend_queue = None
        self.backend_queue = None
        self.q_main2vis = None
        self.q_vis2main = None
        self.save_dir = save_dir

        self.initialized = False
        self.kf_indices = []
        self.monocular = config["Training"]["monocular"]
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.current_window = []

        self.reset = True
        self.requested_init = False
        self.requested_keyframe = 0
        self.use_every_n_frames = 1

        self.gaussians = None
        self.cameras = dict()
        self.device = "cuda:0"
        self.pause = False

        self.model = model  # MASt3R Model

        # 🔧 修复：将theta初始化为tensor而不是整数
        self.theta = torch.tensor(0.0, device=self.device)

        # 初始化增强的动态物体遮罩器（包含完整的动静态可视化）
        self.enable_dynamic_filtering = config.get("dynamic_filtering", {}).get("enabled", True)
        self.filter_initialization = config.get("dynamic_filtering", {}).get("filter_initialization", True)
        self.save_masked_images = config.get("dynamic_filtering", {}).get("save_masked_images", True)
        self.use_ground_segmentation = config.get("dynamic_filtering", {}).get("use_ground_segmentation", True)

        if self.enable_dynamic_filtering and self.filter_initialization:
            # 设置保存目录
            mask_save_dir = config.get("dynamic_filtering", {}).get("save_dir", "./masked_images")
            scene_type = config.get("dynamic_filtering", {}).get("scene_type", "outdoor_street")
            grounding_dino_model = config.get("dynamic_filtering", {}).get("grounding_dino_model",
                                                                           "IDEA-Research/grounding-dino-tiny")

            # 🎨 新增：彩色可视化和静态对象检测配置
            enable_colorful_vis = config.get("dynamic_filtering", {}).get("enable_colorful_visualization", True)
            detect_static_objects = config.get("dynamic_filtering", {}).get("detect_static_objects", True)

            self.dynamic_masker = EnhancedDynamicObjectMasker(
                device=self.device,
                use_sam=config.get("dynamic_filtering", {}).get("use_sam", True),
                save_dir=mask_save_dir,
                save_images=self.save_masked_images,
                use_ground_segmentation=self.use_ground_segmentation,
                scene_type=scene_type,
                grounding_dino_model=grounding_dino_model,
                enable_colorful_vis=enable_colorful_vis,
                detect_static_objects=detect_static_objects
            )

            # 从配置设置场景
            self.dynamic_masker.set_scene_from_config(config)

            print(f"🎯 Enhanced Dynamic Filtering with Complete Visualization:")
            print(f"  - Enabled: {self.enable_dynamic_filtering}")
            print(f"  - Filter initialization: {self.filter_initialization}")
            print(f"  - SAM: {config.get('dynamic_filtering', {}).get('use_sam', True)}")
            print(f"  - Ground segmentation: {self.use_ground_segmentation}")
            print(f"  - Model: {grounding_dino_model}")
            print(f"  - Colorful visualization: {enable_colorful_vis}")
            print(f"  - Detect static objects: {detect_static_objects}")
            print(f"  - Save images: {self.save_masked_images}")
        else:
            print("❌ Dynamic filtering is DISABLED - dynamic objects will appear in reconstruction")

    def set_hyperparams(self):
        self.save_dir = self.config["Results"]["save_dir"]
        self.save_results = self.config["Results"]["save_results"]
        self.save_trj = self.config["Results"]["save_trj"]
        self.save_trj_kf_intv = self.config["Results"]["save_trj_kf_intv"]

        self.tracking_itr_num = self.config["Training"]["tracking_itr_num"]
        self.kf_interval = self.config["Training"]["kf_interval"]
        self.window_size = self.config["Training"]["window_size"]
        self.single_thread = self.config["Training"]["single_thread"]

    def _expand_dynamic_mask(self, dynamic_mask, kernel_size=5):
        """扩展动态物体mask，避免边界处的高斯体生成"""
        mask_np = dynamic_mask.cpu().numpy().astype(np.uint8)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        expanded_mask_np = cv2.dilate(mask_np, kernel, iterations=1)
        expanded_mask = torch.from_numpy(expanded_mask_np).to(dynamic_mask.device).bool()
        return expanded_mask

    def add_new_keyframe(self, cur_frame_idx, depth=None, opacity=None, init=False):
        """添加新关键帧，使用完整的动静态物体检测和可视化"""
        rgb_boundary_threshold = self.config["Training"]["rgb_boundary_threshold"]
        if len(self.kf_indices) > 0:
            last_kf = self.kf_indices[-1]
            viewpoint_last = self.cameras[last_kf]
            R_last = viewpoint_last.R

        self.kf_indices.append(cur_frame_idx)
        viewpoint = self.cameras[cur_frame_idx]

        gt_img = viewpoint.original_image.cuda()
        valid_rgb = (gt_img.sum(dim=0) > rgb_boundary_threshold)[None]

        # ===== 核心：使用完整的动静态物体检测和可视化 =====
        if self.enable_dynamic_filtering and (not init or self.filter_initialization):
            # 转换图像格式
            img_np = gt_img.permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)

            # 获取静态mask、修复图像和完整的检测结果（包含彩色可视化）
            static_mask_np, repaired_image_np, detected_objects, colorful_results = self.dynamic_masker.get_static_mask_for_gaussian_init(
                img_np, frame_idx=cur_frame_idx
            )

            # 转换为torch tensors
            static_mask = torch.from_numpy(static_mask_np).to(self.device).bool()
            dynamic_mask = ~static_mask  # 动态mask是静态mask的反向
            repaired_image = torch.from_numpy(repaired_image_np).to(self.device).float() / 255.0

            # 扩展动态mask边界，确保动态物体完全被排除
            expanded_dynamic_mask = self._expand_dynamic_mask(dynamic_mask, kernel_size=7)
            expanded_static_mask = ~expanded_dynamic_mask

            # 关键：从valid_rgb中排除动态区域
            valid_rgb = valid_rgb & expanded_static_mask[None]

            # 存储信息
            viewpoint.dynamic_mask = dynamic_mask
            viewpoint.expanded_dynamic_mask = expanded_dynamic_mask
            viewpoint.static_mask = static_mask
            viewpoint.expanded_static_mask = expanded_static_mask
            viewpoint.detected_objects = detected_objects
            viewpoint.repaired_image = repaired_image
            viewpoint.colorful_results = colorful_results  # 🎨 新增：存储彩色可视化结果

            # 打印统计
            static_ratio = static_mask.float().mean().item()
            expanded_static_ratio = expanded_static_mask.float().mean().item()
            dynamic_count = len([obj for obj in detected_objects if obj.get('type') == 'dynamic'])
            static_count = len([obj for obj in detected_objects if obj.get('type') == 'static'])

            print(f"🔧 Frame {cur_frame_idx} keyframe processing:")
            print(f"  - Detected {dynamic_count} dynamic objects, {static_count} static objects")
            print(f"  - Original static ratio: {static_ratio:.1%}")
            print(f"  - After expansion: {expanded_static_ratio:.1%}")
            print(f"  - Dynamic objects MASKED OUT from reconstruction")
            print(f"  - Static objects PRESERVED for reconstruction")

            if colorful_results:
                total_visualized = len(colorful_results.get('dynamic_color_map', {})) + len(
                    colorful_results.get('static_color_map', {}))
                print(f"  - Colorful visualization: ✅ {total_visualized} classes visualized")

            if expanded_static_ratio < 0.3:
                print(f"⚠️  WARNING: Only {expanded_static_ratio:.1%} of frame will be reconstructed!")
                print("    Most of the frame contains dynamic objects.")
        # ============================================================

        if self.monocular:
            if depth is None:
                initial_depth = torch.from_numpy(viewpoint.mono_depth).unsqueeze(0)
                print(f"Initial depth map stats for frame {cur_frame_idx}:",
                      f"Max: {torch.max(initial_depth).item():.3f}",
                      f"Min: {torch.min(initial_depth).item():.3f}",
                      f"Mean: {torch.mean(initial_depth).item():.3f}")

                # 将无效区域（包括动态区域）深度设为0
                # 由于valid_rgb已经排除了动态区域，这里会自动处理
                initial_depth[~valid_rgb.cpu()] = 0

                # 额外的统计信息
                if 'dynamic_mask' in locals():
                    total_pixels = initial_depth[0].numel()
                    valid_pixels = (initial_depth[0] > 0).sum().item()
                    print(
                        f"  Valid depth pixels: {valid_pixels}/{total_pixels} ({valid_pixels / total_pixels * 100:.1f}%)")

                return initial_depth[0].numpy()
            else:
                depth = depth.detach().clone()
                opacity = opacity.detach()
                initial_depth = depth

                # 深度处理
                render_depth = initial_depth.cpu().numpy()[0]
                initial_depth, scale_factor, error_mask, num_accurate_pixels = process_depth(
                    render_depth, viewpoint.mono_depth,
                    last_depth=viewpoint_last.mono_depth,
                    im1=viewpoint_last.original_image,
                    im2=viewpoint.original_image,
                    model=self.model,
                    patch_size=self.config["depth"]["patch_size"],
                    mean_threshold=self.config["depth"]["mean_threshold"],
                    std_threshold=self.config["depth"]["std_threshold"],
                    error_threshold=self.config["depth"]["error_threshold"],
                    final_error_threshold=self.config["depth"]["final_error_threshold"],
                    min_accurate_pixels_ratio=self.config["depth"]["min_accurate_pixels_ratio"]
                )

                viewpoint.mono_depth = viewpoint.mono_depth * scale_factor

                # 应用完整的掩码（包括RGB边界和动态物体）
                valid_rgb_np = valid_rgb.cpu().numpy() if isinstance(valid_rgb, torch.Tensor) else valid_rgb
                if initial_depth.shape == valid_rgb_np.shape[1:]:
                    initial_depth[~valid_rgb_np[0]] = 0

            return initial_depth

        # 使用ground truth深度
        initial_depth = torch.from_numpy(viewpoint.depth).unsqueeze(0)
        # 应用掩码（valid_rgb已经包含了动态物体排除）
        initial_depth[~valid_rgb.cpu()] = 0

        return initial_depth[0].numpy() if 'initial_depth' in locals() else None

    def tracking(self, cur_frame_idx, viewpoint):
        """跟踪函数，包含完整的动静态物体检测和可视化"""
        # 生成动态物体遮罩和完整的可视化（主要用于统计和可视化）
        if self.enable_dynamic_filtering:
            gt_img = viewpoint.original_image
            img_np = gt_img.permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)

            # 获取完整的检测和可视化结果
            static_mask_np, repaired_image_np, detected_objects, colorful_results = self.dynamic_masker.get_static_mask_for_gaussian_init(
                img_np, frame_idx=cur_frame_idx
            )

            dynamic_mask = torch.from_numpy(1 - static_mask_np).to(self.device).bool()
            static_mask = torch.from_numpy(static_mask_np).to(self.device).bool()

            viewpoint.dynamic_mask = dynamic_mask
            viewpoint.static_mask = static_mask
            viewpoint.colorful_results = colorful_results  # 🎨 存储彩色可视化结果

            # 存储检测到的物体信息
            if detected_objects:
                viewpoint.detected_objects = detected_objects

            static_ratio = viewpoint.static_mask.float().mean().item()
            dynamic_count = len([obj for obj in detected_objects if obj.get('type') == 'dynamic'])
            static_count = len([obj for obj in detected_objects if obj.get('type') == 'static'])

            print(f"🎬 Tracking frame {cur_frame_idx}: Static ratio={static_ratio:.1%}")
            print(f"    Objects: {dynamic_count} dynamic, {static_count} static detected")

        # 原有的跟踪逻辑
        prev = self.cameras[cur_frame_idx - self.use_every_n_frames]
        pose_prev = getWorld2View2(prev.R, prev.T)

        last_keyframe_idx = self.current_window[0]
        last_kf = self.cameras[last_keyframe_idx]
        pose_last_kf = getWorld2View2(last_kf.R, last_kf.T)
        img1 = last_kf.original_image

        img2 = viewpoint.original_image
        rel_pose, render_depth = get_pose(
            img1=img1, img2=img2, model=self.model,
            dist_coeffs=self.dataset.dist_coeffs,
            viewpoint=last_kf, gaussians=self.gaussians,
            pipeline_params=self.pipeline_params, background=self.background
        )

        viewpoint.mono_depth = get_depth(img2, img2, self.model)

        identity_matrix = torch.eye(4, device=self.device)
        rel_pose = torch.from_numpy(rel_pose).to(self.device).float()

        if torch.allclose(rel_pose, identity_matrix, atol=1e-6):
            pose_init = rel_pose @ pose_last_kf
            viewpoint.update_RT(prev.R, prev.T)
        else:
            pose_init = rel_pose @ pose_last_kf
            viewpoint.update_RT(pose_init[:3, :3], pose_init[:3, 3])

        # 优化参数
        opt_params = []
        opt_params.append({
            "params": [viewpoint.cam_rot_delta],
            "lr": self.config["Training"]["lr"]["cam_rot_delta"],
            "name": "rot_{}".format(viewpoint.uid),
        })
        opt_params.append({
            "params": [viewpoint.cam_trans_delta],
            "lr": self.config["Training"]["lr"]["cam_trans_delta"],
            "name": "trans_{}".format(viewpoint.uid),
        })
        opt_params.append({
            "params": [viewpoint.exposure_a],
            "lr": 0.01,
            "name": "exposure_a_{}".format(viewpoint.uid),
        })
        opt_params.append({
            "params": [viewpoint.exposure_b],
            "lr": 0.01,
            "name": "exposure_b_{}".format(viewpoint.uid),
        })

        pose_optimizer = torch.optim.Adam(opt_params)

        for tracking_itr in range(self.tracking_itr_num):
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background
            )
            image, depth, opacity = (
                render_pkg["render"],
                render_pkg["depth"],
                render_pkg["opacity"],
            )

            pose_optimizer.zero_grad()

            # ===== 动态感知的损失计算 =====
            # 损失函数现在会自动排除动态区域，与高斯体生成保持一致
            loss_tracking = get_loss_tracking(self.config, image, depth, opacity, viewpoint)
            # ===================================

            loss_tracking.backward()

            with torch.no_grad():
                pose_optimizer.step()
                converged = update_pose(viewpoint)

            if tracking_itr % 10 == 0:
                self.q_main2vis.put(
                    gui_utils.GaussianPacket(
                        current_frame=viewpoint,
                        gtcolor=viewpoint.original_image,
                        gtdepth=viewpoint.depth if not self.monocular
                        else np.zeros((viewpoint.image_height, viewpoint.image_width)),
                    )
                )
            if converged:
                break

        self.median_depth = get_median_depth(depth, opacity)
        return render_pkg

    def save_keyframe_mask(self, viewpoint, cur_frame_idx):
        """保存关键帧的掩码可视化，现在包含完整的动静态可视化"""
        if (not self.enable_dynamic_filtering or
                not self.save_masked_images or
                not hasattr(viewpoint, 'dynamic_mask')):
            return

        try:
            # 创建关键帧目录
            kf_dir = os.path.join(self.dynamic_masker.save_dir, "keyframes")
            os.makedirs(kf_dir, exist_ok=True)

            # 获取原始图像
            gt_image = viewpoint.original_image  # [3, H, W] tensor
            img_np = gt_image.permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            # 获取masks
            dynamic_mask = viewpoint.dynamic_mask.cpu().numpy().astype(np.uint8)
            static_mask = viewpoint.static_mask.cpu().numpy().astype(np.uint8)

            if hasattr(viewpoint, 'expanded_dynamic_mask'):
                expanded_mask = viewpoint.expanded_dynamic_mask.cpu().numpy().astype(np.uint8)
            else:
                expanded_mask = dynamic_mask

            # 创建可视化
            vis_img = img_bgr.copy()

            # 静态区域（将被重建）：绿色半透明叠加
            static_overlay = np.zeros_like(vis_img)
            static_overlay[static_mask > 0] = [0, 255, 0]  # 绿色
            vis_img = cv2.addWeighted(vis_img, 0.7, static_overlay, 0.3, 0)

            # 动态区域（被mask掉）：红色
            vis_img[dynamic_mask > 0] = [0, 0, 255]  # 红色

            # 扩展区域（安全边界）：黄色
            vis_img[(expanded_mask > 0) & (dynamic_mask == 0)] = [0, 255, 255]  # 黄色

            # 添加文字说明
            cv2.putText(vis_img, f"Frame {cur_frame_idx} - Keyframe",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(vis_img, "Green: Static (Reconstructed)",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(vis_img, "Red: Dynamic (Masked Out)",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # 如果有检测到的物体，添加标签
            if hasattr(viewpoint, 'detected_objects'):
                y_offset = 120
                dynamic_count = len([obj for obj in viewpoint.detected_objects if obj.get('type') == 'dynamic'])
                static_count = len([obj for obj in viewpoint.detected_objects if obj.get('type') == 'static'])

                cv2.putText(vis_img, f"Dynamic Objects: {dynamic_count}",
                            (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
                y_offset += 25
                cv2.putText(vis_img, f"Static Objects: {static_count}",
                            (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                y_offset += 25

                # 显示前几个检测到的对象
                for obj in viewpoint.detected_objects[:5]:  # 最多显示5个
                    type_icon = "🔴" if obj.get('type') == 'dynamic' else "🔵"
                    label = f"{type_icon} {obj['label']}: {obj['score']:.2f}"
                    cv2.putText(vis_img, f"  {label}",
                                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    y_offset += 20

            # 保存图像
            kf_path = os.path.join(kf_dir, f"keyframe_{cur_frame_idx:06d}_mask_vis.jpg")
            cv2.imwrite(kf_path, vis_img)

            print(f"💾 Saved enhanced keyframe visualization for frame {cur_frame_idx}")

        except Exception as e:
            print(f"Warning: Failed to save keyframe mask for frame {cur_frame_idx}: {e}")

    def is_keyframe(self, cur_frame_idx, last_keyframe_idx, cur_frame_visibility_filter, occ_aware_visibility):
        """考虑完整动静态物体检测的关键帧选择策略"""
        kf_translation = self.config["Training"]["kf_translation"]
        kf_min_translation = self.config["Training"]["kf_min_translation"]
        kf_overlap = self.config["Training"]["kf_overlap"]

        curr_frame = self.cameras[cur_frame_idx]
        last_kf = self.cameras[last_keyframe_idx]

        # 计算相机运动
        pose_CW = getWorld2View2(curr_frame.R, curr_frame.T)
        last_kf_CW = getWorld2View2(last_kf.R, last_kf.T)
        last_kf_WC = torch.linalg.inv(last_kf_CW)
        dist = torch.norm((pose_CW @ last_kf_WC)[0:3, 3])

        dist_check = dist > kf_translation * self.median_depth
        dist_check2 = dist > kf_min_translation * self.median_depth

        # 计算重叠度
        union = torch.logical_or(
            cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
        ).count_nonzero()
        intersection = torch.logical_and(
            cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
        ).count_nonzero()

        # ===== 动态场景的关键帧策略调整 =====
        adjusted_overlap = kf_overlap

        if hasattr(curr_frame, 'expanded_static_mask'):
            # 检查扩展后的静态区域比例
            static_ratio = curr_frame.expanded_static_mask.float().mean().item()
            if static_ratio < 0.3:
                # 静态区域太少（包括动静态完整检测后），更积极创建关键帧
                adjusted_overlap = kf_overlap * 0.7
                print(
                    f"🔄 Limited static region ({static_ratio:.1%}) after complete detection, adjusted overlap: {adjusted_overlap:.3f}")
        # ==========================================

        point_ratio_2 = intersection / union
        return (point_ratio_2 < adjusted_overlap and dist_check2) or dist_check

    def add_to_window(self, cur_frame_idx, cur_frame_visibility_filter, occ_aware_visibility, window):
        # 原有实现保持不变
        N_dont_touch = 2
        window = [cur_frame_idx] + window
        curr_frame = self.cameras[cur_frame_idx]
        to_remove = []
        removed_frame = None

        for i in range(N_dont_touch, len(window)):
            kf_idx = window[i]
            intersection = torch.logical_and(
                cur_frame_visibility_filter, occ_aware_visibility[kf_idx]
            ).count_nonzero()
            denom = min(
                cur_frame_visibility_filter.count_nonzero(),
                occ_aware_visibility[kf_idx].count_nonzero(),
            )
            point_ratio_2 = intersection / denom
            cut_off = self.config["Training"].get("kf_cutoff", 0.4)
            if not self.initialized:
                cut_off = 0.4
            if (point_ratio_2 <= cut_off) and (len(window) > self.config["Training"]["window_size"]):
                to_remove.append(kf_idx)

        if to_remove:
            window.remove(to_remove[-1])
            removed_frame = to_remove[-1]

        kf_0_WC = torch.linalg.inv(getWorld2View2(curr_frame.R, curr_frame.T))

        if len(window) > self.config["Training"]["window_size"]:
            inv_dist = []
            for i in range(N_dont_touch, len(window)):
                inv_dists = []
                kf_i_idx = window[i]
                kf_i = self.cameras[kf_i_idx]
                kf_i_CW = getWorld2View2(kf_i.R, kf_i.T)
                for j in range(N_dont_touch, len(window)):
                    if i == j:
                        continue
                    kf_j_idx = window[j]
                    kf_j = self.cameras[kf_j_idx]
                    kf_j_WC = torch.linalg.inv(getWorld2View2(kf_j.R, kf_j.T))
                    T_CiCj = kf_i_CW @ kf_j_WC
                    inv_dists.append(1.0 / (torch.norm(T_CiCj[0:3, 3]) + 1e-6).item())
                T_CiC0 = kf_i_CW @ kf_0_WC
                k = torch.sqrt(torch.norm(T_CiC0[0:3, 3])).item()
                inv_dist.append(k * sum(inv_dists))

            idx = np.argmax(inv_dist)
            removed_frame = window[N_dont_touch + idx]
            window.remove(removed_frame)

        return window, removed_frame

    def request_keyframe(self, cur_frame_idx, viewpoint, current_window, depthmap):
        msg = ["keyframe", cur_frame_idx, viewpoint, current_window, depthmap, self.theta]
        self.backend_queue.put(msg)
        self.requested_keyframe += 1

    def reqeust_mapping(self, cur_frame_idx, viewpoint):
        msg = ["map", cur_frame_idx, viewpoint]
        self.backend_queue.put(msg)

    def request_init(self, cur_frame_idx, viewpoint, depth_map):
        msg = ["init", cur_frame_idx, viewpoint, depth_map]
        self.backend_queue.put(msg)
        self.requested_init = True

    def sync_backend(self, data):
        self.gaussians = data[1]
        occ_aware_visibility = data[2]
        keyframes = data[3]
        self.occ_aware_visibility = occ_aware_visibility

        for kf_id, kf_R, kf_T in keyframes:
            self.cameras[kf_id].update_RT(kf_R.clone(), kf_T.clone())

    def cleanup(self, cur_frame_idx):
        self.cameras[cur_frame_idx].clean()
        if cur_frame_idx % 10 == 0:
            torch.cuda.empty_cache()

    def initialize(self, cur_frame_idx, viewpoint):
        self.initialized = not self.monocular
        self.kf_indices = []
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.current_window = []

        while not self.backend_queue.empty():
            self.backend_queue.get()

        viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)

        img = viewpoint.original_image
        viewpoint.mono_depth = get_depth(img, img, self.model)

        # 在初始化阶段就应用完整的动静态物体检测和可视化
        print(f"🔄 INITIALIZING with frame {cur_frame_idx}")
        if self.enable_dynamic_filtering and self.filter_initialization:
            print("  ✅ Complete Dynamic/Static Object Detection ENABLED for initialization")
            print("  🎨 Colorful visualization for both dynamic and static objects")
            print("  🛠️  Ground shadows will be repaired automatically")
            print(f"  🎯 Scene type: {self.dynamic_masker.prompt_manager.current_scene}")
        elif self.enable_dynamic_filtering and not self.filter_initialization:
            print("  ⚠️  Dynamic filtering enabled but SKIPPING initialization frame")
        else:
            print("  ❌ Dynamic filtering DISABLED - cars may appear as ghosts!")

        self.kf_indices = []
        depth_map = self.add_new_keyframe(cur_frame_idx, init=True)
        self.request_init(cur_frame_idx, viewpoint, depth_map)
        self.reset = False

    def run(self):
        # 主执行循环集成了完整的动静态物体检测和彩色可视化
        cur_frame_idx = 0
        projection_matrix = getProjectionMatrix2(
            znear=0.01, zfar=100.0,
            fx=self.dataset.fx, fy=self.dataset.fy,
            cx=self.dataset.cx, cy=self.dataset.cy,
            W=self.dataset.width, H=self.dataset.height,
        ).transpose(0, 1)
        projection_matrix = projection_matrix.to(device=self.device)

        tic = torch.cuda.Event(enable_timing=True)
        toc = torch.cuda.Event(enable_timing=True)

        while True:
            # 处理GUI消息
            if self.q_vis2main.empty():
                if self.pause:
                    continue
            else:
                data_vis2main = self.q_vis2main.get()
                self.pause = data_vis2main.flag_pause
                if self.pause:
                    self.backend_queue.put(["pause"])
                    continue
                else:
                    self.backend_queue.put(["unpause"])

            if self.frontend_queue.empty():
                tic.record()

                # 检查是否完成
                if cur_frame_idx >= len(self.dataset):
                    if self.save_results:
                        eval_ate(self.cameras, self.kf_indices, self.save_dir, 0,
                                 final=True, monocular=self.monocular)
                        save_gaussians(self.gaussians, self.save_dir, "final", final=True)

                    # 🎨 保存最终的彩色可视化统计报告
                    if self.enable_dynamic_filtering and self.save_masked_images:
                        self._save_final_colorful_statistics()

                    break

                # 等待初始化
                if self.requested_init:
                    time.sleep(0.01)
                    continue

                if self.single_thread and self.requested_keyframe > 0:
                    time.sleep(0.01)
                    continue

                if not self.initialized and self.requested_keyframe > 0:
                    time.sleep(0.01)
                    continue

                # 处理当前帧
                viewpoint = Camera.init_from_dataset(
                    self.dataset, cur_frame_idx, projection_matrix
                )
                viewpoint.compute_grad_mask(self.config)
                self.cameras[cur_frame_idx] = viewpoint

                # 初始化
                if self.reset:
                    self.initialize(cur_frame_idx, viewpoint)
                    self.current_window.append(cur_frame_idx)
                    cur_frame_idx += 1
                    continue

                self.initialized = self.initialized or (len(self.current_window) == self.window_size)

                # 跟踪
                render_pkg = self.tracking(cur_frame_idx, viewpoint)
                current_window_dict = {}
                current_window_dict[self.current_window[0]] = self.current_window[1:]
                keyframes = [self.cameras[kf_idx] for kf_idx in self.current_window]

                self.q_main2vis.put(
                    gui_utils.GaussianPacket(
                        gaussians=clone_obj(self.gaussians),
                        current_frame=viewpoint,
                        keyframes=keyframes,
                        kf_window=current_window_dict,
                    )
                )

                if self.requested_keyframe > 0:
                    self.cleanup(cur_frame_idx)
                    cur_frame_idx += 1
                    continue

                # 关键帧选择
                last_keyframe_idx = self.current_window[0]
                check_time = (cur_frame_idx - last_keyframe_idx) >= self.kf_interval
                curr_visibility = (render_pkg["n_touched"] > 0).long()

                create_kf = self.is_keyframe(
                    cur_frame_idx, last_keyframe_idx,
                    curr_visibility, self.occ_aware_visibility,
                )

                if len(self.current_window) < self.window_size:
                    union = torch.logical_or(
                        curr_visibility, self.occ_aware_visibility[last_keyframe_idx]
                    ).count_nonzero()
                    intersection = torch.logical_and(
                        curr_visibility, self.occ_aware_visibility[last_keyframe_idx]
                    ).count_nonzero()
                    point_ratio = intersection / union
                    create_kf = (check_time and
                                 point_ratio < self.config["Training"]["kf_overlap"])

                if self.single_thread:
                    create_kf = check_time and create_kf

                if create_kf:
                    self.current_window, removed = self.add_to_window(
                        cur_frame_idx, curr_visibility,
                        self.occ_aware_visibility, self.current_window,
                    )
                    depth_map = self.add_new_keyframe(
                        cur_frame_idx,
                        depth=render_pkg["depth"],
                        opacity=render_pkg["opacity"],
                        init=False,
                    )
                    self.request_keyframe(
                        cur_frame_idx, viewpoint, self.current_window, depth_map
                    )

                    # 为关键帧保存特殊标记的掩码图像（包含完整的动静态可视化）
                    self.save_keyframe_mask(viewpoint, cur_frame_idx)
                else:
                    self.cleanup(cur_frame_idx)

                cur_frame_idx += 1

                # 评估
                if (self.save_results and self.save_trj and create_kf and
                        len(self.kf_indices) % self.save_trj_kf_intv == 0):
                    Log("Evaluating ATE at frame: ", cur_frame_idx)
                    eval_ate(self.cameras, self.kf_indices, self.save_dir,
                             cur_frame_idx, monocular=self.monocular)

                toc.record()
                torch.cuda.synchronize()
                if create_kf:
                    duration = tic.elapsed_time(toc)
                    time.sleep(max(0.01, 1.0 / 3.0 - duration / 1000))

            else:
                # 处理后端消息
                data = self.frontend_queue.get()
                if data[0] == "sync_backend":
                    self.sync_backend(data)
                elif data[0] == "keyframe":
                    self.sync_backend(data)
                    self.requested_keyframe -= 1
                elif data[0] == "init":
                    self.sync_backend(data)
                    self.requested_init = False
                elif data[0] == "stop":
                    Log("Frontend Stopped.")
                    break

    def _save_final_colorful_statistics(self):
        """保存最终的彩色可视化统计报告"""
        try:
            stats_path = os.path.join(self.dynamic_masker.save_dir, "final_colorful_statistics.txt")

            # 收集所有检测到的对象统计
            all_dynamic_objects = {}
            all_static_objects = {}
            total_frames = 0

            for frame_idx, camera in self.cameras.items():
                if hasattr(camera, 'detected_objects'):
                    total_frames += 1
                    for obj in camera.detected_objects:
                        obj_type = obj.get('type', 'unknown')
                        label = obj['label']

                        if obj_type == 'dynamic':
                            if label not in all_dynamic_objects:
                                all_dynamic_objects[label] = {'count': 0, 'total_confidence': 0.0}
                            all_dynamic_objects[label]['count'] += 1
                            all_dynamic_objects[label]['total_confidence'] += obj['score']
                        elif obj_type == 'static':
                            if label not in all_static_objects:
                                all_static_objects[label] = {'count': 0, 'total_confidence': 0.0}
                            all_static_objects[label]['count'] += 1
                            all_static_objects[label]['total_confidence'] += obj['score']

            with open(stats_path, 'w') as f:
                f.write("Final Colorful Segmentation Statistics\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Total Processed Frames: {total_frames}\n")
                f.write(f"Scene Type: {self.dynamic_masker.prompt_manager.current_scene}\n\n")

                # 动态对象统计
                f.write("Dynamic Objects Summary:\n")
                f.write("-" * 30 + "\n")
                if all_dynamic_objects:
                    for label, stats in sorted(all_dynamic_objects.items(), key=lambda x: x[1]['count'], reverse=True):
                        avg_conf = stats['total_confidence'] / stats['count']
                        freq = stats['count'] / total_frames * 100
                        f.write(
                            f"{label:20}: {stats['count']:4d} detections ({freq:5.1f}% frames) - Avg Conf: {avg_conf:.3f}\n")
                else:
                    f.write("No dynamic objects detected.\n")

                f.write("\nStatic Objects Summary:\n")
                f.write("-" * 30 + "\n")
                if all_static_objects:
                    for label, stats in sorted(all_static_objects.items(), key=lambda x: x[1]['count'], reverse=True):
                        avg_conf = stats['total_confidence'] / stats['count']
                        freq = stats['count'] / total_frames * 100
                        f.write(
                            f"{label:20}: {stats['count']:4d} detections ({freq:5.1f}% frames) - Avg Conf: {avg_conf:.3f}\n")
                else:
                    f.write("No static objects detected.\n")

                # 总体统计
                total_dynamic_detections = sum(stats['count'] for stats in all_dynamic_objects.values())
                total_static_detections = sum(stats['count'] for stats in all_static_objects.values())

                f.write("\nOverall Summary:\n")
                f.write("-" * 20 + "\n")
                f.write(f"Total Dynamic Object Classes: {len(all_dynamic_objects)}\n")
                f.write(f"Total Static Object Classes: {len(all_static_objects)}\n")
                f.write(f"Total Dynamic Detections: {total_dynamic_detections}\n")
                f.write(f"Total Static Detections: {total_static_detections}\n")
                f.write(
                    f"Average Detections per Frame: {(total_dynamic_detections + total_static_detections) / total_frames:.2f}\n")

            print(f"📊 Saved final colorful statistics to: {stats_path}")

        except Exception as e:
            print(f"❌ Failed to save final colorful statistics: {e}")


# 使用示例和配置建议
def create_enhanced_config_example():
    """
    创建增强配置文件示例，包含完整的动静态物体检测和彩色可视化功能
    """
    config_example = {
        "dynamic_filtering": {
            "enabled": True,
            "filter_initialization": True,
            "save_masked_images": True,
            "use_ground_segmentation": True,
            "scene_type": "outdoor_street",  # 可选: parking_lot, highway, residential, indoor, construction, campus

            # Grounding DINO 配置
            "grounding_dino_model": "/path/to/your/groundingdino_swint_ogc.pth",  # 本地模型路径
            # "grounding_dino_model": "IDEA-Research/grounding-dino-tiny",  # 或使用 Hugging Face 模型
            "grounding_dino_config": "/path/to/config/GroundingDINO_SwinT_OGC.py",  # 配置文件路径

            # SAM 配置
            "use_sam": True,
            "sam_checkpoint": "/home/zwk/下载/S3PO-GS-main/sam_vit_h_4b8939.pth",

            # 保存和可视化配置
            "save_dir": "./enhanced_detection_results",

            # 🎨 新增：完整的彩色可视化配置
            "enable_colorful_visualization": True,
            "detect_static_objects": True,
            "colorful_overlay_alpha": 0.6,
            "save_class_separated": True,
            "create_legend": True,
        },

        # 其他原有配置保持不变
        "Training": {
            "monocular": True,
            "rgb_boundary_threshold": 0.01,
            # ... 其他训练参数
        },

        "Results": {
            "save_results": True,
            "save_dir": "./results",
            # ... 其他结果参数
        }
    }

    return config_example


# 主函数示例
def main():
    """主函数示例，展示如何使用增强的动态物体遮罩器"""

    print("🚀 Starting Enhanced Dynamic Object Masking with Complete Visualization")
    print("=" * 70)

    # 1. 加载配置
    config = create_enhanced_config_example()

    # 2. 初始化模型和数据集
    model = None  # 加载你的 MASt3R 模型
    dataset = None  # 加载你的数据集

    # 3. 创建 FrontEnd 实例
    frontend = FrontEnd(config, model, save_dir=config["Results"]["save_dir"])

    print("🎯 Configuration Summary:")
    print(f"  - Dynamic filtering: {config['dynamic_filtering']['enabled']}")
    print(f"  - Colorful visualization: {config['dynamic_filtering']['enable_colorful_visualization']}")
    print(f"  - Static object detection: {config['dynamic_filtering']['detect_static_objects']}")
    print(f"  - Scene type: {config['dynamic_filtering']['scene_type']}")
    print(f"  - Ground segmentation: {config['dynamic_filtering']['use_ground_segmentation']}")
    print(f"  - SAM enabled: {config['dynamic_filtering']['use_sam']}")

    # 4. 运行处理
    try:
        frontend.run()
        print("✅ Processing completed successfully!")

        # 5. 输出结果路径
        results_dir = config["dynamic_filtering"]["save_dir"]
        print(f"\n📁 Results saved to: {results_dir}")
        print("📊 Generated visualizations:")
        print(f"  - Colorful combined: {results_dir}/colorful_combined/")
        print(f"  - Dynamic objects only: {results_dir}/colorful_dynamic_only/")
        print(f"  - Static objects only: {results_dir}/colorful_static_only/")
        print(f"  - Overlay visualization: {results_dir}/colorful_overlay/")
        print(f"  - With legends: {results_dir}/colorful_legend/")
        print(f"  - Analysis reports: {results_dir}/colorful_analysis/")
        print(f"  - Keyframe visualizations: {results_dir}/keyframes/")

    except Exception as e:
        print(f"❌ Error during processing: {e}")
        raise


if __name__ == "__main__":
    main()