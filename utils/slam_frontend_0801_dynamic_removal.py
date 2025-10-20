import time
import numpy as np
import torch
import torch.multiprocessing as mp
import os
import cv2
from ultralytics import YOLO
from segment_anything import SamPredictor, sam_model_registry

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


class EnhancedDynamicObjectMasker:
    def __init__(self, device="cuda", use_sam=True,
                 sam_checkpoint="/home/zwk/下载/S3PO-GS-main/utils/sam_vit_b_01ec64.pth",
                 save_dir=None, save_images=True, use_ground_segmentation=True):
        """
        增强的动态物体遮罩器，结合YOLO、SAM和地面分割
        """
        # YOLO检测器
        self.yolo_model = YOLO("yolo11x.pt").to(device)
        self.device = device
        self.dynamic_class_ids = [0, 2, 3, 5, 7]  # 人、车、摩托车、公交车、卡车

        # SAM分割器
        self.use_sam = use_sam
        if use_sam:
            try:
                sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
                sam.to(device=device)
                self.sam_predictor = SamPredictor(sam)
            except:
                print("Warning: SAM model not found, using YOLO boxes only")
                self.use_sam = False

        # 地面分割功能
        self.use_ground_segmentation = use_ground_segmentation
        if use_ground_segmentation:
            self._init_ground_segmentation()

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
            self._create_save_directories()

    def _init_ground_segmentation(self):
        """
        初始化地面分割模型
        """
        try:
            # 使用传统方法进行地面分割
            self.ground_segmentation_method = "traditional"
            print("✅ Ground segmentation initialized with traditional method")
        except Exception as e:
            print(f"Warning: Ground segmentation failed: {e}")
            self.ground_segmentation_method = "traditional"

    def _create_save_directories(self):
        """创建保存图像的目录结构"""
        # 创建主目录和子目录
        directories = [
            self.save_dir,
            os.path.join(self.save_dir, "original"),
            os.path.join(self.save_dir, "detections"),  # YOLO检测框
            os.path.join(self.save_dir, "yolo_masks"),  # YOLO生成的mask
            os.path.join(self.save_dir, "sam_masks"),  # SAM精确分割的mask
            os.path.join(self.save_dir, "motion_masks"),  # 运动检测mask
            os.path.join(self.save_dir, "ground_masks"),  # 地面分割mask
            os.path.join(self.save_dir, "shadow_regions"),  # 车辆阴影区域
            os.path.join(self.save_dir, "inpainted_ground"),  # 修复后的地面
            os.path.join(self.save_dir, "final_masks"),  # 最终组合mask
            os.path.join(self.save_dir, "masked_overlay"),  # 叠加显示
            os.path.join(self.save_dir, "static_only"),  # 只显示静态区域
            os.path.join(self.save_dir, "repaired_images"),  # 地面修复后的图像
            os.path.join(self.save_dir, "keyframes"),  # 关键帧特殊保存
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)

        print(f"Created mask image directories in: {self.save_dir}")

    def segment_ground(self, image):
        """
        分割图像中的地面区域

        Args:
            image: RGB图像 [H, W, 3]

        Returns:
            ground_mask: 地面mask [H, W], 1表示地面，0表示非地面
        """
        h, w = image.shape[:2]

        if self.ground_segmentation_method == "traditional":
            return self._traditional_ground_segmentation(image)
        else:
            return self._ml_ground_segmentation(image)

    def _traditional_ground_segmentation(self, image):
        """
        基于传统方法的地面分割
        假设地面通常在图像下半部分，具有相对一致的颜色和纹理
        """
        h, w = image.shape[:2]
        ground_mask = np.zeros((h, w), dtype=np.uint8)

        # 转换到HSV空间进行颜色分析
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # 1. 基于位置的先验：地面通常在图像下半部分
        ground_region_y_start = int(h * 0.6)  # 从60%高度开始认为可能是地面

        # 2. 在下半部分进行颜色聚类
        lower_region = image[ground_region_y_start:, :, :]
        lower_hsv = hsv[ground_region_y_start:, :, :]
        lower_gray = gray[ground_region_y_start:, :]

        # 3. 基于颜色一致性检测地面
        # 检测相对均匀的区域（低标准差）
        kernel_size = 15
        blur_gray = cv2.GaussianBlur(lower_gray, (kernel_size, kernel_size), 0)
        grad_x = cv2.Sobel(blur_gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blur_gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

        # 地面区域通常梯度较小（纹理相对均匀）
        texture_threshold = np.percentile(gradient_magnitude, 30)  # 取30%分位数作为阈值
        smooth_regions = (gradient_magnitude < texture_threshold).astype(np.uint8)

        # 4. 基于颜色聚类检测主要地面颜色
        lower_region_reshaped = lower_region.reshape(-1, 3)

        # 使用简单的颜色一致性检测
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

        # 8. 向上扩展地面区域（基于颜色相似性）
        if np.sum(combined_mask) > 0:
            ground_mask = self._extend_ground_upward(image, ground_mask, ground_region_y_start)

        return ground_mask

    def _extend_ground_upward(self, image, initial_ground_mask, start_y):
        """
        向上扩展地面区域，基于颜色相似性
        """
        h, w = image.shape[:2]
        extended_mask = initial_ground_mask.copy()

        # 获取已知地面区域的平均颜色
        ground_pixels = image[initial_ground_mask > 0]
        if len(ground_pixels) == 0:
            return initial_ground_mask

        mean_ground_color = np.mean(ground_pixels, axis=0)
        color_std = np.std(ground_pixels, axis=0)

        # 向上逐行检查
        for y in range(start_y - 1, max(int(h * 0.3), 0), -1):  # 不超过图像的30%高度
            row_colors = image[y, :, :]
            color_distances = np.linalg.norm(row_colors - mean_ground_color, axis=1)

            # 基于颜色相似性和现有mask的连通性
            threshold = np.linalg.norm(color_std) * 2
            similar_pixels = color_distances < threshold

            # 只保留与下方地面区域连通的像素
            if y < h - 1:
                below_mask = extended_mask[y + 1, :]
                connected_pixels = np.logical_and(similar_pixels,
                                                  cv2.dilate(below_mask, np.ones((1, 3), np.uint8)) > 0)
                extended_mask[y, :] = connected_pixels.astype(np.uint8)

        return extended_mask

    def _ml_ground_segmentation(self, image):
        """
        基于机器学习模型的地面分割（预留接口）
        """
        h, w = image.shape[:2]
        return np.zeros((h, w), dtype=np.uint8)

    def repair_ground_shadows(self, image, vehicle_mask, ground_mask):
        """
        修复车辆在地面上的阴影/鬼影

        Args:
            image: 原始图像 [H, W, 3]
            vehicle_mask: 车辆mask [H, W]
            ground_mask: 地面mask [H, W]

        Returns:
            repaired_image: 修复后的图像
            shadow_regions: 检测到的阴影区域mask
        """
        # 1. 检测车辆mask与地面的交集（潜在阴影区域）
        shadow_regions = np.logical_and(vehicle_mask, ground_mask).astype(np.uint8)

        if np.sum(shadow_regions) == 0:
            return image.copy(), shadow_regions

        # 2. 扩展阴影区域，包含可能的边缘效应
        kernel = self.ground_dilation_kernel
        expanded_shadow = cv2.dilate(shadow_regions, kernel, iterations=1)

        # 3. 确保扩展区域仍在地面内
        final_shadow_regions = np.logical_and(expanded_shadow, ground_mask).astype(np.uint8)

        # 4. 创建修复mask（需要被修复的区域）
        inpaint_mask = final_shadow_regions.astype(np.uint8) * 255

        # 5. 使用图像修复算法
        repaired_image = self._inpaint_ground_region(image, inpaint_mask, ground_mask)

        return repaired_image, final_shadow_regions

    def _inpaint_ground_region(self, image, inpaint_mask, ground_mask):
        """
        对地面区域进行图像修复
        """
        # 方法1: 使用OpenCV的图像修复
        try:
            repaired = cv2.inpaint(image, inpaint_mask, self.inpaint_radius, cv2.INPAINT_TELEA)
            return repaired
        except:
            pass

        # 方法2: 基于邻近像素的简单修复
        return self._simple_ground_inpaint(image, inpaint_mask, ground_mask)

    def _simple_ground_inpaint(self, image, inpaint_mask, ground_mask):
        """
        简单的地面修复：用周围地面像素的均值填充
        """
        repaired = image.copy()
        h, w = image.shape[:2]

        # 找到需要修复的区域
        repair_coords = np.where(inpaint_mask > 0)

        if len(repair_coords[0]) == 0:
            return repaired

        # 对每个需要修复的像素，寻找周围的地面像素进行填充
        for i in range(len(repair_coords[0])):
            y, x = repair_coords[0][i], repair_coords[1][i]

            # 在周围区域寻找地面像素
            search_radius = 10
            y1 = max(0, y - search_radius)
            y2 = min(h, y + search_radius + 1)
            x1 = max(0, x - search_radius)
            x2 = min(w, x + search_radius + 1)

            # 获取搜索区域内的地面像素
            search_region_ground = ground_mask[y1:y2, x1:x2]
            search_region_inpaint = inpaint_mask[y1:y2, x1:x2]

            # 排除需要修复的区域，只使用有效的地面像素
            valid_ground = np.logical_and(search_region_ground, search_region_inpaint == 0)

            if np.sum(valid_ground) > 0:
                # 计算有效地面像素的平均颜色
                search_region_image = image[y1:y2, x1:x2]
                valid_pixels = search_region_image[valid_ground]
                mean_color = np.mean(valid_pixels, axis=0)

                # 添加一些随机性以避免过于均匀
                noise = np.random.normal(0, 5, 3)  # 小幅随机噪声
                final_color = np.clip(mean_color + noise, 0, 255)

                repaired[y, x] = final_color.astype(np.uint8)

        return repaired

    def save_detection_results(self, image, frame_idx, yolo_mask=None, sam_mask=None,
                               motion_mask=None, final_mask=None, boxes=None,
                               ground_mask=None, shadow_regions=None, repaired_image=None):
        """
        保存检测和分割的各种结果，包括地面相关的结果
        """
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

            # 2. 保存YOLO检测框
            if boxes is not None and len(boxes) > 0:
                detection_img = img_bgr.copy()
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(detection_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                detection_path = os.path.join(self.save_dir, "detections", f"frame_{frame_idx:06d}.jpg")
                cv2.imwrite(detection_path, detection_img)

            # 3. 保存各种mask
            if yolo_mask is not None:
                yolo_mask_img = (yolo_mask * 255).astype(np.uint8)
                yolo_path = os.path.join(self.save_dir, "yolo_masks", f"frame_{frame_idx:06d}.jpg")
                cv2.imwrite(yolo_path, yolo_mask_img)

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

            print(f"Saved all mask images for frame {frame_idx}")

        except Exception as e:
            print(f"Warning: Failed to save mask images for frame {frame_idx}: {e}")

    def detect_and_segment(self, image, frame_idx=None):
        """
        检测动态物体并生成精确分割mask，包含地面修复
        """
        h, w = image.shape[:2]
        final_mask = np.zeros((h, w), dtype=np.uint8)
        yolo_mask = np.zeros((h, w), dtype=np.uint8)
        sam_mask = None
        motion_mask = None
        ground_mask = None
        shadow_regions = None
        repaired_image = image.copy()
        max_confidence = 0.0

        # 1. 地面分割
        if self.use_ground_segmentation:
            ground_mask = self.segment_ground(image)
            print(f"Ground segmentation: {np.sum(ground_mask)} pixels detected as ground")

        # 2. YOLO检测 - 降低置信度阈值以捕获更多车辆
        results = self.yolo_model(image, imgsz=640, conf=0.2, classes=self.dynamic_class_ids, verbose=False)

        boxes_with_scores = []
        vehicle_detected = False

        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls)
                    if cls_id in self.dynamic_class_ids:
                        xyxy = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        boxes_with_scores.append((xyxy, conf, cls_id))
                        max_confidence = max(max_confidence, conf)

                        # 检查是否检测到车辆
                        if cls_id in [2, 3, 5, 7]:  # 车、摩托车、公交车、卡车
                            vehicle_detected = True

        if not boxes_with_scores:
            # 即使没有检测到，也保存原始图像和地面mask
            if frame_idx is not None:
                self.save_detection_results(image, frame_idx, yolo_mask=yolo_mask,
                                            final_mask=final_mask, boxes=[],
                                            ground_mask=ground_mask,
                                            repaired_image=repaired_image)
            return final_mask, 0.0, repaired_image

        # 3. 创建YOLO基础mask
        for box, conf, cls_id in boxes_with_scores:
            x1, y1, x2, y2 = box.astype(int)

            # 对车辆类别扩展边界框
            if cls_id in [2, 3, 5, 7]:  # 车辆类别
                width = x2 - x1
                height = y2 - y1
                expand_w = int(width * 0.1)
                expand_h = int(height * 0.1)

                x1 = max(0, x1 - expand_w)
                y1 = max(0, y1 - expand_h)
                x2 = min(w, x2 + expand_w)
                y2 = min(h, y2 + expand_h)

                print(f"  Expanded vehicle bbox (cls={cls_id}, conf={conf:.3f}): {box} -> [{x1},{y1},{x2},{y2}]")

            yolo_mask[y1:y2, x1:x2] = 1

        final_mask = yolo_mask.copy()

        # 4. SAM精确分割（如果启用）
        if self.use_sam:
            sam_combined_mask = np.zeros((h, w), dtype=np.uint8)
            self.sam_predictor.set_image(image)

            for box, conf, cls_id in boxes_with_scores:
                x1, y1, x2, y2 = box.astype(int)

                if cls_id in [2, 3, 5, 7]:
                    width = x2 - x1
                    height = y2 - y1
                    expand_w = int(width * 0.1)
                    expand_h = int(height * 0.1)

                    x1 = max(0, x1 - expand_w)
                    y1 = max(0, y1 - expand_h)
                    x2 = min(w, x2 + expand_w)
                    y2 = min(h, y2 + expand_h)

                input_box = np.array([x1, y1, x2, y2])

                try:
                    masks, scores, _ = self.sam_predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=input_box[None, :],
                        multimask_output=False,
                    )

                    if len(masks) > 0:
                        best_mask = masks[0].astype(np.uint8)
                        sam_combined_mask = np.logical_or(sam_combined_mask, best_mask).astype(np.uint8)
                except Exception as e:
                    print(f"  SAM failed for box {input_box}: {e}")

            if sam_combined_mask.sum() > 0:
                final_mask = sam_combined_mask
                sam_mask = sam_combined_mask

        # 5. 运动检测增强
        motion_refined_mask = self._refine_with_motion(image, final_mask)
        if motion_refined_mask is not None:
            motion_mask = motion_refined_mask
            final_mask = motion_refined_mask

        # 6. 地面修复：处理车辆在地面上的阴影
        if self.use_ground_segmentation and ground_mask is not None and vehicle_detected:
            repaired_image, shadow_regions = self.repair_ground_shadows(image, final_mask, ground_mask)
            print(
                f"  Ground repair: {np.sum(shadow_regions) if shadow_regions is not None else 0} shadow pixels repaired")

        # 7. 时间一致性滤波
        final_mask = self._temporal_consistency(final_mask)

        # 8. 车辆特殊处理
        if vehicle_detected and final_mask.sum() > 0:
            kernel = np.ones((5, 5), np.uint8)
            final_mask = cv2.dilate(final_mask, kernel, iterations=1)
            print(f"  Applied additional dilation for vehicle detection")

        # 9. 保存所有结果
        if frame_idx is not None:
            boxes = [box for box, _, _ in boxes_with_scores]
            self.save_detection_results(
                image, frame_idx,
                yolo_mask=yolo_mask,
                sam_mask=sam_mask,
                motion_mask=motion_mask,
                final_mask=final_mask,
                boxes=boxes,
                ground_mask=ground_mask,
                shadow_regions=shadow_regions,
                repaired_image=repaired_image
            )

        if vehicle_detected:
            print(f"🚗 Vehicle detected in frame {frame_idx}: conf={max_confidence:.3f}, mask_pixels={final_mask.sum()}")

        return final_mask, max_confidence, repaired_image

    def get_static_mask_for_gaussian_init(self, image, frame_idx=None):
        """
        为高斯体初始化获取静态区域mask
        这个函数会返回可以用于初始化高斯体的区域（排除动态物体，修复地面阴影）

        Returns:
            static_mask: 静态区域mask
            repaired_image: 用于初始化的修复图像
        """
        # 获取动态物体mask和修复图像
        dynamic_mask, confidence, repaired_image = self.detect_and_segment(image, frame_idx)

        # 静态mask是动态mask的反向
        static_mask = (1 - dynamic_mask).astype(np.uint8)

        return static_mask, repaired_image

    def _refine_with_motion(self, current_frame, detection_mask):
        """
        使用光流运动信息优化mask
        """
        if self.prev_frame is None:
            self.prev_frame = cv2.cvtColor(current_frame, cv2.COLOR_RGB2GRAY)
            return detection_mask

        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_RGB2GRAY)

        try:
            # 计算光流
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_frame, current_gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )

            # 计算运动幅度
            magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)

            # 运动mask
            motion_mask = (magnitude > self.motion_threshold).astype(np.uint8)

            # 结合检测mask和运动mask
            refined_mask = np.logical_and(detection_mask, motion_mask).astype(np.uint8)

            # 对静止的检测物体保留部分区域（可能是暂时静止的动态物体）
            static_detection = np.logical_and(detection_mask, ~motion_mask)
            refined_mask = np.logical_or(refined_mask, static_detection * 0.5).astype(np.uint8)

            self.prev_frame = current_gray
            return refined_mask

        except Exception as e:
            print(f"Motion detection failed: {e}")
            self.prev_frame = current_gray
            return detection_mask

    def _temporal_consistency(self, current_mask):
        """
        时间一致性滤波，减少mask的闪烁
        """
        self.mask_history.append(current_mask.copy())

        if len(self.mask_history) > self.history_length:
            self.mask_history.pop(0)

        if len(self.mask_history) < 3:
            return current_mask

        # 使用历史mask的中位数滤波
        mask_stack = np.stack(self.mask_history, axis=0)
        consistent_mask = np.median(mask_stack, axis=0).astype(np.uint8)

        return consistent_mask


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
        self.theta = 0

        # 初始化动态物体遮罩器（带地面分割）
        self.enable_dynamic_filtering = config.get("dynamic_filtering", {}).get("enabled", True)
        self.filter_initialization = config.get("dynamic_filtering", {}).get("filter_initialization", True)
        self.save_masked_images = config.get("dynamic_filtering", {}).get("save_masked_images", True)
        self.use_ground_segmentation = config.get("dynamic_filtering", {}).get("use_ground_segmentation", True)

        if self.enable_dynamic_filtering and (self.filter_initialization):
            # 设置保存目录
            mask_save_dir = config.get("dynamic_filtering", {}).get("save_dir", "./masked_images")

            self.dynamic_masker = EnhancedDynamicObjectMasker(
                device=self.device,
                use_sam=config.get("dynamic_filtering", {}).get("use_sam", True),
                save_dir=mask_save_dir,
                save_images=self.save_masked_images,
                use_ground_segmentation=self.use_ground_segmentation  # 启用地面分割
            )

            print(f"🎯 Enhanced dynamic filtering configuration:")
            print(f"  - Enabled: {self.enable_dynamic_filtering}")
            print(f"  - Filter initialization frame: {self.filter_initialization}")
            print(f"  - Use SAM: {config.get('dynamic_filtering', {}).get('use_sam', True)}")
            print(f"  - Use ground segmentation: {self.use_ground_segmentation}")
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
        """
        扩展动态物体mask，避免边界处的高斯体生成
        """
        # 转换为numpy进行形态学操作
        mask_np = dynamic_mask.cpu().numpy().astype(np.uint8)

        # 膨胀操作扩展mask
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        expanded_mask_np = cv2.dilate(mask_np, kernel, iterations=1)

        # 转换回tensor
        expanded_mask = torch.from_numpy(expanded_mask_np).to(dynamic_mask.device).bool()

        return expanded_mask

    def add_new_keyframe(self, cur_frame_idx, depth=None, opacity=None, init=False):
        """
        添加新关键帧，使用动态物体掩码和地面修复
        """
        rgb_boundary_threshold = self.config["Training"]["rgb_boundary_threshold"]
        if len(self.kf_indices) > 0:
            last_kf = self.kf_indices[-1]
            viewpoint_last = self.cameras[last_kf]
            R_last = viewpoint_last.R

        self.kf_indices.append(cur_frame_idx)
        viewpoint = self.cameras[cur_frame_idx]

        # 计算角度差
        R_now = viewpoint.R
        if len(self.kf_indices) > 1:
            R_now = R_now.to(torch.float32)
            R_last = R_last.to(torch.float32)
            R_diff = torch.matmul(R_last.T, R_now)
            trace_R_diff = torch.trace(R_diff)
            theta_rad = torch.acos((trace_R_diff - 1) / 2)
            theta_deg = torch.rad2deg(theta_rad)
            self.theta = theta_deg

        gt_img = viewpoint.original_image.cuda()
        valid_rgb = (gt_img.sum(dim=0) > rgb_boundary_threshold)[None]

        # ===== 核心修改：生成动态物体遮罩并从有效区域排除，同时修复地面 =====
        dynamic_mask = None
        static_mask = None
        repaired_image = None

        if self.enable_dynamic_filtering and (not init or self.filter_initialization):
            # 转换图像格式用于YOLO/SAM和地面分割
            img_np = gt_img.permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
            img_np = (img_np * 255).astype(np.uint8)

            # 使用新的方法获取静态mask和修复图像
            static_mask_np, repaired_image_np = self.dynamic_masker.get_static_mask_for_gaussian_init(
                img_np, frame_idx=cur_frame_idx
            )

            dynamic_mask = torch.from_numpy(1 - static_mask_np).to(self.device).bool()
            static_mask = torch.from_numpy(static_mask_np).to(self.device).bool()
            repaired_image = torch.from_numpy(repaired_image_np).to(self.device).float() / 255.0

            # 关键改动：从有效区域中完全排除动态物体
            # 扩展动态mask边界，确保边界处也不生成高斯体
            expanded_dynamic_mask = self._expand_dynamic_mask(dynamic_mask, kernel_size=7)
            expanded_static_mask = ~expanded_dynamic_mask

            # 应用扩展后的静态mask
            valid_rgb = valid_rgb & expanded_static_mask[None]

            # 存储mask信息和修复图像
            viewpoint.dynamic_mask = dynamic_mask
            viewpoint.expanded_dynamic_mask = expanded_dynamic_mask
            viewpoint.static_mask = static_mask
            viewpoint.expanded_static_mask = expanded_static_mask
            viewpoint.repaired_image = repaired_image  # 存储修复后的图像

            # 打印统计信息
            static_ratio = static_mask.float().mean().item()
            expanded_static_ratio = expanded_static_mask.float().mean().item()
            print(f"🔧 Frame {cur_frame_idx} processing:")
            print(f"  ✅ Ground repair applied")
            print(f"  📊 Original static ratio: {static_ratio:.1%}")
            print(f"  📊 Expanded static ratio: {expanded_static_ratio:.1%}")
            print(f"  🛡️  Excluded {(static_ratio - expanded_static_ratio) * 100:.1f}% additional pixels for safety")

            # 如果静态区域太少，发出警告
            if expanded_static_ratio < 0.2:
                print(f"⚠️  WARNING: Expanded static region only {expanded_static_ratio:.1%}!")
                print("    This may severely affect reconstruction quality.")
                print("    Consider reducing mask expansion or disabling dynamic filtering.")
            elif expanded_static_ratio < 0.4:
                print(f"⚠️  CAUTION: Limited static region {expanded_static_ratio:.1%}")
                print("    Tracking may be less stable.")
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
                if dynamic_mask is not None:
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

        return initial_depth[0].numpy()

    def tracking(self, cur_frame_idx, viewpoint):
        """
        跟踪函数，包含地面修复的动态物体过滤
        """
        # 生成动态物体遮罩（主要用于统计和可视化）
        if self.enable_dynamic_filtering:
            gt_img = viewpoint.original_image
            img_np = gt_img.permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)

            # 使用带地面修复的检测方法
            static_mask_np, repaired_image_np = self.dynamic_masker.get_static_mask_for_gaussian_init(
                img_np, frame_idx=cur_frame_idx
            )

            dynamic_mask = torch.from_numpy(1 - static_mask_np).to(self.device).bool()
            static_mask = torch.from_numpy(static_mask_np).to(self.device).bool()

            viewpoint.dynamic_mask = dynamic_mask
            viewpoint.static_mask = static_mask

            static_ratio = viewpoint.static_mask.float().mean().item()
            print(f"🎬 Tracking frame {cur_frame_idx}: Static ratio={static_ratio:.1%} (with ground repair)")

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

            # 调试信息：每50次迭代输出一次mask覆盖情况
            if tracking_itr == 0 and self.enable_dynamic_filtering:
                try:
                    from utils.slam_utils import debug_loss_mask_coverage
                    debug_loss_mask_coverage(self.config, viewpoint, verbose=False)
                except:
                    pass
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
        """
        为关键帧保存特殊标记的掩码图像，包括地面修复信息
        """
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

            # 获取动态mask和扩展mask
            dynamic_mask = viewpoint.dynamic_mask.cpu().numpy().astype(np.uint8)
            if hasattr(viewpoint, 'expanded_dynamic_mask'):
                expanded_mask = viewpoint.expanded_dynamic_mask.cpu().numpy().astype(np.uint8)
            else:
                expanded_mask = dynamic_mask

            # 创建关键帧特殊标记图像
            # 原始动态区域用红色，扩展区域用黄色
            kf_img = img_bgr.copy()
            kf_img[dynamic_mask > 0] = [0, 0, 255]  # 红色：原始动态区域
            kf_img[(expanded_mask > 0) & (dynamic_mask == 0)] = [0, 255, 255]  # 黄色：扩展区域

            # 保存关键帧图像
            kf_path = os.path.join(kf_dir, f"keyframe_{cur_frame_idx:06d}.jpg")
            cv2.imwrite(kf_path, kf_img)

            print(
                f"💾 Saved keyframe mask for frame {cur_frame_idx} (Red=Dynamic, Yellow=Expanded, includes ground repair)")

        except Exception as e:
            print(f"Warning: Failed to save keyframe mask for frame {cur_frame_idx}: {e}")

    def is_keyframe(self, cur_frame_idx, last_keyframe_idx, cur_frame_visibility_filter, occ_aware_visibility):
        """
        考虑动态物体的关键帧选择策略
        """
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
                # 静态区域太少（包括地面修复后），更积极创建关键帧
                adjusted_overlap = kf_overlap * 0.7
                print(
                    f"🔄 Limited static region ({static_ratio:.1%}) after ground repair, adjusted overlap: {adjusted_overlap:.3f}")
        # ==========================================

        point_ratio_2 = intersection / union
        return (point_ratio_2 < adjusted_overlap and dist_check2) or dist_check

    # 其他方法保持不变...
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

        # 在初始化阶段就应用动态过滤和地面修复
        print(f"🔄 INITIALIZING with frame {cur_frame_idx}")
        if self.enable_dynamic_filtering and self.filter_initialization and self.use_ground_segmentation:
            print("  ✅ Dynamic filtering + Ground segmentation ENABLED for initialization")
            print("  🛠️  Ground shadows will be repaired automatically")
        elif self.enable_dynamic_filtering and self.filter_initialization:
            print("  ✅ Dynamic filtering ENABLED (no ground repair)")
        elif self.enable_dynamic_filtering and not self.filter_initialization:
            print("  ⚠️  Dynamic filtering enabled but SKIPPING initialization frame")
        else:
            print("  ❌ Dynamic filtering DISABLED - cars may appear as ghosts!")

        self.kf_indices = []
        depth_map = self.add_new_keyframe(cur_frame_idx, init=True)
        self.request_init(cur_frame_idx, viewpoint, depth_map)
        self.reset = False

    def run(self):
        # 主执行循环保持原有逻辑，但集成了动态物体过滤和地面修复
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

                    # 为关键帧保存特殊标记的掩码图像
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