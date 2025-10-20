import time
import numpy as np
import torch
import torch.multiprocessing as mp
import os
import cv2
from PIL import Image
import torch.nn.functional as F

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


class ScenePromptManager:
    """场景检测和Prompt管理器"""

    def __init__(self, default_scene="outdoor_street"):
        self.current_scene = default_scene
        self.scene_prompts = {
            "outdoor_street": {
                "dynamic_objects": [
                    "car", "cars", "vehicle", "vehicles",
                    "truck", "trucks", "bus", "buses",
                    "motorcycle", "motorcycles", "bike", "bicycle", "bicycles",
                    "person", "people", "pedestrian", "pedestrians", "human",
                    "scooter", "e-scooter", "skateboard",
                    "delivery robot", "mobile robot"
                ],
                "confidence_threshold": 0.35,
                "description": "Urban street scene with vehicles and pedestrians"
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
        """获取当前场景的prompt"""
        scene_info = self.scene_prompts[self.current_scene]
        prompt = ". ".join(scene_info["dynamic_objects"])
        return prompt, scene_info["confidence_threshold"]

    def get_detailed_prompt(self):
        """获取详细的prompt信息"""
        scene_info = self.scene_prompts[self.current_scene]
        return {
            "prompt": ". ".join(scene_info["dynamic_objects"]),
            "confidence_threshold": scene_info["confidence_threshold"],
            "description": scene_info["description"],
            "object_classes": scene_info["dynamic_objects"]
        }

    def set_scene(self, scene_type):
        """手动设置场景类型"""
        if scene_type in self.scene_prompts:
            self.current_scene = scene_type
            print(f"🎬 Scene type set to: {scene_type}")
        else:
            available_scenes = list(self.scene_prompts.keys())
            print(f"❌ Unknown scene type: {scene_type}. Available: {available_scenes}")

    def add_custom_scene(self, scene_name, dynamic_objects, confidence_threshold=0.2, description=""):
        """添加自定义场景配置"""
        self.scene_prompts[scene_name] = {
            "dynamic_objects": dynamic_objects,
            "confidence_threshold": confidence_threshold,
            "description": description
        }
        print(f"✅ Added custom scene: {scene_name}")


class GroundingDINODetector:
    """直接使用本地.pth文件的Grounding DINO检测器"""

    def __init__(self, model_path="/home/zwk/下载/S3PO-GS-main/groundingdino_swint_ogc.pth", device="cuda"):
        self.device = device
        self.model_path = model_path
        self.model = None
        self.warmup_completed = False

        # 直接加载本地权重
        self._load_local_model()

        # 预热模型确保第一帧正常工作
        if self.model is not None:
            self._warmup_detector()

    def _load_local_model(self):
        """直接加载本地的.pth文件"""
        try:
            print(f"🔄 Loading local Grounding DINO model: {self.model_path}")

            # 检查文件是否存在
            if not os.path.exists(self.model_path):
                print(f"❌ Model file not found: {self.model_path}")
                self._use_simple_detection()
                return

            # 尝试导入groundingdino
            try:
                from groundingdino.util.inference import load_model, predict
                from groundingdino.util.slconfig import SLConfig
                from groundingdino.models import build_model
                import groundingdino.datasets.transforms as T

                # 查找配置文件
                config_path = self._find_config_file()
                if config_path:
                    self.model = load_model(config_path, self.model_path, device=self.device)
                    self.use_groundingdino = True
                    print(f"✅ Successfully loaded Grounding DINO from {self.model_path}")
                else:
                    print("❌ Config file not found")
                    self._use_simple_detection()

            except ImportError:
                print("❌ groundingdino package not installed")
                print("💡 Install with: pip install groundingdino")
                self._use_simple_detection()

        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            self._use_simple_detection()

    def _warmup_detector(self):
        """预热检测器，确保第一帧能正常工作"""
        try:
            print("🔥 Warming up Grounding DINO detector...")
            # 创建一个小的测试图像进行预热
            test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            test_prompt = "car. person"

            # 执行一次测试检测
            boxes, scores, labels = self.detect(test_image, test_prompt, confidence_threshold=0.5)

            self.warmup_completed = True
            print(f"✅ Detector warmup completed - ready for first frame")

            # 测试是否能正常工作
            if len(boxes) == 0:
                print("📝 Warmup note: No detections in random image (expected)")
            else:
                print(f"📝 Warmup note: Got {len(boxes)} detections in random image")

            return True

        except Exception as e:
            print(f"⚠️  Detector warmup failed: {e}")
            self.warmup_completed = False
            return False

    def _find_config_file(self):
        """查找配置文件"""
        possible_paths = [
            "/home/zwk/下载/S3PO-GS-main/GroundingDINO-main/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            os.path.join(os.path.dirname(self.model_path), "GroundingDINO_SwinT_OGC.cfg.py"),
            "./GroundingDINO_SwinT_OGC.cfg.py",
        ]

        for path in possible_paths:
            if os.path.exists(path):
                print(f"✅ Found config file: {path}")
                return path

        # 如果找不到，尝试下载
        print("⚠️  Config file not found, attempting to download...")
        try:
            import urllib.request
            config_url = "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.cfg.py"
            config_path = os.path.join(os.path.dirname(self.model_path), "GroundingDINO_SwinT_OGC.cfg.py")
            urllib.request.urlretrieve(config_url, config_path)
            print(f"✅ Downloaded config to: {config_path}")
            return config_path
        except:
            return None

    def _use_simple_detection(self):
        """使用简单的检测方法作为备选"""
        self.use_groundingdino = False
        print("⚠️  Using simple detection mode (no real detection)")

    def detect(self, image, text_prompt, confidence_threshold=0.2):
        """检测接口 - 修复坐标转换问题"""
        if not hasattr(self, 'use_groundingdino') or not self.use_groundingdino:
            # 返回空结果，让系统继续运行
            return np.array([]), np.array([]), []

        try:
            from groundingdino.util.inference import predict
            import groundingdino.datasets.transforms as T

            # 准备图像并记录原始尺寸
            if isinstance(image, np.ndarray):
                if image.dtype != np.uint8:
                    image = (image * 255).astype(np.uint8)
                image_pil = Image.fromarray(image)
                orig_h, orig_w = image.shape[:2]  # 原始图像尺寸
            else:
                image_pil = image
                orig_w, orig_h = image_pil.size  # PIL格式：(width, height)

            print(f"🖼️  Original image size: {orig_w}x{orig_h}")

            # 使用不resize的transform，避免复杂的坐标转换
            transform = T.Compose([
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

            # 应用变换
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

            if len(boxes) == 0:
                print(f"🔍 No objects detected")
                return np.array([]), np.array([]), []

            # 关键修复：处理Grounding DINO返回的坐标格式
            # Grounding DINO返回的是归一化的 [cx, cy, w, h] 格式
            boxes_np = boxes.cpu().numpy()
            scores_np = logits.cpu().numpy()

            # 调试：查看原始boxes格式
            if len(boxes_np) > 0:
                print(f"📦 Raw box format (first box): {boxes_np[0]} (shape: {boxes_np.shape})")

            # 转换 [cx, cy, w, h] 到 [x1, y1, x2, y2]
            # boxes是归一化坐标 (0-1之间)
            cx = boxes_np[:, 0] * orig_w  # 中心点x坐标
            cy = boxes_np[:, 1] * orig_h  # 中心点y坐标
            w = boxes_np[:, 2] * orig_w  # 宽度
            h = boxes_np[:, 3] * orig_h  # 高度

            # 转换为左上角和右下角坐标
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2

            # 组合成新的boxes数组 [x1, y1, x2, y2]
            boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

            # 确保坐标在有效范围内
            boxes_xyxy[:, 0] = np.clip(boxes_xyxy[:, 0], 0, orig_w)  # x1
            boxes_xyxy[:, 1] = np.clip(boxes_xyxy[:, 1], 0, orig_h)  # y1
            boxes_xyxy[:, 2] = np.clip(boxes_xyxy[:, 2], 0, orig_w)  # x2
            boxes_xyxy[:, 3] = np.clip(boxes_xyxy[:, 3], 0, orig_h)  # y2

            labels = []
            for phrase in phrases:
                label = phrase.split('(')[0].strip()
                labels.append(label)

            print(f"🎯 Detected {len(boxes_xyxy)} objects (converted from cxcywh to xyxy)")
            for i, (box, score, label) in enumerate(zip(boxes_xyxy, scores_np, labels)):
                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1
                print(f"  {i + 1}. {label}: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}] "
                      f"(w={width:.1f}, h={height:.1f}) conf={score:.3f}")

            return boxes_xyxy, scores_np, labels

        except Exception as e:
            print(f"❌ Detection failed: {e}")
            import traceback
            traceback.print_exc()

            # 如果失败，尝试使用resize方法作为备选
            print("🔄 Trying fallback detection with resize...")
            return self._detect_with_resize(image, text_prompt, confidence_threshold)

    def _detect_with_resize(self, image, text_prompt, confidence_threshold):
        """备选方法：使用resize的检测"""
        try:
            from groundingdino.util.inference import predict
            import groundingdino.datasets.transforms as T

            if isinstance(image, np.ndarray):
                if image.dtype != np.uint8:
                    image = (image * 255).astype(np.uint8)
                image_pil = Image.fromarray(image)
                orig_h, orig_w = image.shape[:2]
            else:
                image_pil = image
                orig_w, orig_h = image_pil.size

            print(f"🔄 Fallback: Using resize detection for {orig_w}x{orig_h} image")

            # 使用resize的transform
            transform = T.Compose([
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

            image_tensor, _ = transform(image_pil, None)

            # 获取实际的tensor尺寸
            _, tensor_h, tensor_w = image_tensor.shape
            print(f"📐 Resized tensor dimensions: {tensor_w}x{tensor_h}")

            boxes, logits, phrases = predict(
                model=self.model,
                image=image_tensor,
                caption=text_prompt,
                box_threshold=confidence_threshold,
                text_threshold=confidence_threshold,
                device=self.device
            )

            if len(boxes) == 0:
                return np.array([]), np.array([]), []

            # 处理boxes格式转换
            boxes_np = boxes.cpu().numpy()
            scores_np = logits.cpu().numpy()

            # 调试：查看原始boxes格式
            if len(boxes_np) > 0:
                print(f"📦 Raw box format in resize (first box): {boxes_np[0]}")

            # Grounding DINO 返回的是归一化的 [cx, cy, w, h] 格式
            # 需要转换为 [x1, y1, x2, y2] 格式
            cx = boxes_np[:, 0] * orig_w
            cy = boxes_np[:, 1] * orig_h
            w = boxes_np[:, 2] * orig_w
            h = boxes_np[:, 3] * orig_h

            # 转换为 [x1, y1, x2, y2]
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2

            # 组合成新的boxes数组
            boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

            # 确保坐标在有效范围内
            boxes_xyxy[:, 0] = np.clip(boxes_xyxy[:, 0], 0, orig_w)
            boxes_xyxy[:, 1] = np.clip(boxes_xyxy[:, 1], 0, orig_h)
            boxes_xyxy[:, 2] = np.clip(boxes_xyxy[:, 2], 0, orig_w)
            boxes_xyxy[:, 3] = np.clip(boxes_xyxy[:, 3], 0, orig_h)

            labels = []
            for phrase in phrases:
                label = phrase.split('(')[0].strip()
                labels.append(label)

            print(f"🎯 Resize fallback detected {len(boxes_xyxy)} objects")
            for i, (box, score, label) in enumerate(zip(boxes_xyxy, scores_np, labels)):
                x1, y1, x2, y2 = box
                print(f"  {i + 1}. {label}: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}] "
                      f"(w={x2 - x1:.1f}, h={y2 - y1:.1f}) conf={score:.3f}")

            return boxes_xyxy, scores_np, labels

        except Exception as e:
            print(f"❌ Resize detection also failed: {e}")
            import traceback
            traceback.print_exc()
            return np.array([]), np.array([]), []


class EnhancedDynamicObjectMasker:
    def __init__(self, device="cuda", use_sam=True,
                 sam_checkpoint="/home/zwk/下载/S3PO-GS-main/utils/sam_vit_b_01ec64.pth",
                 save_dir=None, save_images=True, scene_type="outdoor_street"):
        """
        使用本地Grounding DINO权重的动态物体遮罩器（移除地面功能）
        """
        self.device = device
        self.initialization_success = True
        self.first_frame_processed = False  # 添加第一帧标记

        # 场景和Prompt管理
        try:
            self.prompt_manager = ScenePromptManager(default_scene=scene_type)
            print(f"✅ Scene prompt manager initialized")
        except Exception as e:
            print(f"❌ Failed to initialize scene manager: {e}")
            self.initialization_success = False

        # 直接使用本地的Grounding DINO权重
        print(f"🔄 Initializing Grounding DINO detector with local weights...")
        try:
            self.grounding_detector = GroundingDINODetector(
                model_path="/home/zwk/下载/S3PO-GS-main/groundingdino_swint_ogc.pth",
                device=device
            )
            print(f"✅ Grounding DINO detector initialized")
        except Exception as e:
            print(f"❌ Failed to initialize Grounding DINO: {e}")
            self.grounding_detector = None
            self.initialization_success = False

        # SAM分割器（保持不变）
        self.use_sam = use_sam
        if use_sam and SAM_AVAILABLE:
            try:
                if os.path.exists(sam_checkpoint):
                    sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
                    sam.to(device=device)
                    self.sam_predictor = SamPredictor(sam)
                    print("✅ SAM model loaded successfully")
                else:
                    print(f"⚠️  SAM checkpoint not found at {sam_checkpoint}")
                    self.use_sam = False
            except Exception as e:
                print(f"⚠️  Warning: SAM model failed to load ({e})")
                self.use_sam = False

        # 其他初始化保持不变...
        self.prev_frame = None
        self.prev_mask = None
        self.motion_threshold = 3.0
        self.mask_history = []
        self.history_length = 5
        self.save_images = save_images
        self.save_dir = save_dir if save_dir else "./masked_images"

        if self.save_images:
            try:
                self._create_save_directories()
            except Exception as e:
                print(f"⚠️  Failed to create save directories: {e}")
                self.save_images = False

        # 打印配置信息
        print(f"🎯 Dynamic Object Masker Configuration:")
        print(f"  - Using local model: /home/zwk/下载/S3PO-GS-main/groundingdino_swint_ogc.pth")
        print(f"  - SAM enabled: {self.use_sam}")
        print(f"  - Save images: {self.save_images}")

    def _create_conservative_first_frame_mask(self, image):
        """为第一帧创建保守的动态物体mask"""
        h, w = image.shape[:2]

        # 基于颜色和位置的启发式检测
        # 1. 检测车辆常见颜色
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # 车辆常见颜色范围
        vehicle_color_ranges = [
            # 白色车辆
            ([0, 0, 180], [180, 30, 255]),
            # 黑色车辆
            ([0, 0, 0], [180, 255, 50]),
            # 灰色车辆
            ([0, 0, 50], [180, 50, 150]),
            # 红色车辆
            ([0, 100, 100], [10, 255, 255]),
            ([170, 100, 100], [180, 255, 255]),
            # 蓝色车辆
            ([100, 100, 100], [130, 255, 255]),
        ]

        color_mask = np.zeros((h, w), dtype=np.uint8)
        for lower, upper in vehicle_color_ranges:
            lower = np.array(lower)
            upper = np.array(upper)
            mask = cv2.inRange(hsv, lower, upper)
            color_mask = cv2.bitwise_or(color_mask, mask)

        # 2. 形态学操作
        kernel = np.ones((5, 5), np.uint8)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)

        # 3. 只保留较大的连通区域（可能是车辆）
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_mask = np.zeros((h, w), dtype=np.uint8)

        min_area = (h * w) * 0.01  # 至少占图像1%的区域
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                cv2.drawContours(filtered_mask, [contour], -1, 1, -1)

        # 4. 在图像下半部分更保守（车辆更可能出现在这里）
        conservative_mask = filtered_mask.copy()
        bottom_half_start = int(h * 0.4)

        # 对下半部分应用更宽松的检测
        bottom_region = image[bottom_half_start:, :, :]
        bottom_gray = cv2.cvtColor(bottom_region, cv2.COLOR_RGB2GRAY)

        # 检测边缘较强的区域（车辆轮廓）
        edges = cv2.Canny(bottom_gray, 50, 150)
        dilated_edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)

        conservative_mask[bottom_half_start:, :] = np.maximum(
            conservative_mask[bottom_half_start:, :],
            (dilated_edges > 0).astype(np.uint8)
        )

        print(f"🛡️  Created conservative first-frame mask: {np.sum(conservative_mask)} pixels")
        return conservative_mask

    def _fallback_detection(self, image, frame_idx=None):
        """改进的fallback检测，为第一帧提供保守的动态物体检测"""
        print(f"🔄 Using improved fallback detection for frame {frame_idx}")
        h, w = image.shape[:2]

        # 如果是第一帧或前几帧，使用保守的启发式检测
        if frame_idx is not None and frame_idx < 5:
            print(f"🔰 Applying conservative detection for early frame {frame_idx}")
            conservative_mask = self._create_conservative_first_frame_mask(image)

            # 扩展mask以确保安全
            kernel = np.ones((9, 9), np.uint8)
            expanded_mask = cv2.dilate(conservative_mask, kernel, iterations=2)

            return expanded_mask, 0.5, image.copy()

        # 对于后续帧，如果有历史mask，可以使用运动检测
        if hasattr(self, 'prev_frame') and self.prev_frame is not None:
            current_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

            try:
                # 计算光流
                flow = cv2.calcOpticalFlowFarneback(
                    self.prev_frame, current_gray, None,
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0
                )

                magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
                motion_mask = (magnitude > self.motion_threshold).astype(np.uint8)

                # 扩展运动区域
                kernel = np.ones((7, 7), np.uint8)
                expanded_motion = cv2.dilate(motion_mask, kernel, iterations=1)

                print(f"🏃 Motion-based fallback detection: {np.sum(expanded_motion)} pixels")
                return expanded_motion, 0.3, image.copy()

            except Exception as e:
                print(f"❌ Motion detection failed: {e}")

        # 最后的保底方案：返回空mask，但记录警告
        print(f"⚠️  WARNING: No dynamic object detection for frame {frame_idx}")
        return np.zeros((h, w), dtype=np.uint8), 0.0, image.copy()

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

    def _create_save_directories(self):
        """创建保存图像的目录结构"""
        directories = [
            self.save_dir,
            os.path.join(self.save_dir, "original"),
            os.path.join(self.save_dir, "grounding_dino_detections"),  # Grounding DINO检测框
            os.path.join(self.save_dir, "grounding_dino_masks"),  # Grounding DINO生成的mask
            os.path.join(self.save_dir, "sam_masks"),  # SAM精确分割的mask
            os.path.join(self.save_dir, "motion_masks"),  # 运动检测mask
            os.path.join(self.save_dir, "final_masks"),  # 最终组合mask
            os.path.join(self.save_dir, "masked_overlay"),  # 叠加显示
            os.path.join(self.save_dir, "static_only"),  # 只显示静态区域
            os.path.join(self.save_dir, "keyframes"),  # 关键帧特殊保存
            os.path.join(self.save_dir, "detection_analysis"),  # 检测分析结果
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)

        print(f"📁 Created mask image directories in: {self.save_dir}")

    def save_detection_results(self, image, frame_idx, grounding_dino_mask=None, sam_mask=None,
                               motion_mask=None, final_mask=None, boxes=None, labels=None, scores=None):
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
                print(
                    f"🖼️  Drawing {len(boxes)} detection boxes on {detection_img.shape[1]}x{detection_img.shape[0]} image")

                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box)

                    # 确保坐标在图像范围内
                    x1 = max(0, min(x1, detection_img.shape[1] - 1))
                    y1 = max(0, min(y1, detection_img.shape[0] - 1))
                    x2 = max(0, min(x2, detection_img.shape[1] - 1))
                    y2 = max(0, min(y2, detection_img.shape[0] - 1))

                    print(f"  Box {i + 1}: [{x1}, {y1}, {x2}, {y2}]")

                    # 绘制检测框
                    cv2.rectangle(detection_img, (x1, y1), (x2, y2), (0, 255, 0), 3)

                    # 添加标签和置信度
                    if labels and i < len(labels):
                        label_text = labels[i]
                        if scores is not None and i < len(scores):
                            label_text += f" {scores[i]:.2f}"

                        # 绘制标签背景
                        (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                        cv2.rectangle(detection_img, (x1, y1 - text_height - 10), (x1 + text_width + 5, y1),
                                      (0, 255, 0), -1)
                        cv2.putText(detection_img, label_text, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (0, 0, 0), 2)

                detection_path = os.path.join(self.save_dir, "grounding_dino_detections", f"frame_{frame_idx:06d}.jpg")
                cv2.imwrite(detection_path, detection_img)
                print(f"✅ Saved detection visualization to {detection_path}")

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

            # 4. 保存最终mask和组合显示
            if final_mask is not None:
                final_mask_img = (final_mask * 255).astype(np.uint8)
                final_path = os.path.join(self.save_dir, "final_masks", f"frame_{frame_idx:06d}.jpg")
                cv2.imwrite(final_path, final_mask_img)

                # 叠加显示（动态区域用红色）
                overlay_img = img_bgr.copy()
                overlay_img[final_mask > 0] = [0, 0, 255]  # 动态物体用红色
                overlay_path = os.path.join(self.save_dir, "masked_overlay", f"frame_{frame_idx:06d}.jpg")
                cv2.imwrite(overlay_path, overlay_img)

                # 静态区域图像
                static_img = img_bgr.copy()
                static_img[final_mask > 0] = [0, 0, 0]
                static_path = os.path.join(self.save_dir, "static_only", f"frame_{frame_idx:06d}.jpg")
                cv2.imwrite(static_path, static_img)

            # 5. 保存检测分析结果
            if boxes is not None and labels is not None:
                analysis_path = os.path.join(self.save_dir, "detection_analysis", f"frame_{frame_idx:06d}.txt")
                with open(analysis_path, 'w') as f:
                    f.write(f"Frame {frame_idx} Detection Analysis\n")
                    f.write(f"Scene Type: {self.prompt_manager.current_scene}\n")
                    f.write(f"Prompt Used: {self.prompt_manager.get_current_prompt()[0]}\n")
                    f.write(f"Total Detections: {len(boxes)}\n\n")

                    for i, (box, label) in enumerate(zip(boxes, labels)):
                        score = scores[i] if scores is not None and i < len(scores) else 0.0
                        f.write(f"Detection {i + 1}:\n")
                        f.write(f"  Label: {label}\n")
                        f.write(f"  Confidence: {score:.3f}\n")
                        f.write(f"  Box: [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]\n\n")

            print(f"💾 Saved all Grounding DINO detection results for frame {frame_idx}")

        except Exception as e:
            print(f"❌ Warning: Failed to save detection results for frame {frame_idx}: {e}")
            import traceback
            traceback.print_exc()

    def detect_and_segment(self, image, frame_idx=None):
        """
        改进的检测方法，特别优化第一帧的处理和坐标修复（移除地面功能）
        """
        h, w = image.shape[:2]
        final_mask = np.zeros((h, w), dtype=np.uint8)
        grounding_dino_mask = np.zeros((h, w), dtype=np.uint8)
        sam_mask = None
        motion_mask = None
        max_confidence = 0.0

        # 第一帧特殊处理标记
        is_first_frame = (frame_idx == 0) or not self.first_frame_processed

        if is_first_frame:
            print(f"🔰 Processing FIRST FRAME {frame_idx} with enhanced detection")

        # 1. 获取当前场景的prompt和阈值
        try:
            text_prompt, confidence_threshold = self.prompt_manager.get_current_prompt()

            # 第一帧使用更低的置信度阈值，更保守的检测
            if is_first_frame:
                confidence_threshold = max(0.15, confidence_threshold * 0.7)
                print(f"🔰 First frame: lowered confidence threshold to {confidence_threshold}")

            print(f"🎯 Using prompt: '{text_prompt[:100]}...' (confidence: {confidence_threshold})")
        except Exception as e:
            print(f"❌ Failed to get prompt: {e}")
            text_prompt = "car. truck. person. bicycle"
            confidence_threshold = 0.15 if is_first_frame else 0.2

        # 2. Grounding DINO检测 - 多次尝试确保第一帧成功
        detection_success = False
        max_attempts = 3 if is_first_frame else 1

        for attempt in range(max_attempts):
            try:
                if is_first_frame and attempt > 0:
                    print(f"🔄 First frame detection attempt {attempt + 1}/{max_attempts}")

                boxes, scores, labels = self.grounding_detector.detect(
                    image, text_prompt, confidence_threshold
                )

                if len(boxes) > 0 or not is_first_frame:
                    detection_success = True
                    break

            except Exception as e:
                print(f"❌ Grounding DINO detection attempt {attempt + 1} failed: {e}")
                if attempt == max_attempts - 1:
                    print(f"❌ All detection attempts failed for frame {frame_idx}")

        # 如果检测完全失败，使用fallback
        if not detection_success or len(boxes) == 0:
            print(f"🆘 Using fallback detection for frame {frame_idx}")
            fallback_mask, fallback_conf, fallback_image = self._fallback_detection(image, frame_idx)

            # 保存fallback结果
            if frame_idx is not None and self.save_images:
                try:
                    self.save_detection_results(
                        image, frame_idx,
                        grounding_dino_mask=fallback_mask,
                        final_mask=fallback_mask,
                        boxes=[], labels=['fallback'], scores=[fallback_conf]
                    )
                except Exception as e:
                    print(f"⚠️  Failed to save fallback results: {e}")

            self.first_frame_processed = True
            return fallback_mask, fallback_conf, fallback_image

        # 3. 处理检测结果，创建基础mask
        vehicle_detected = False

        try:
            for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
                x1, y1, x2, y2 = box.astype(int)
                max_confidence = max(max_confidence, score)

                # 确保坐标在图像范围内
                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(0, min(x2, w - 1))
                y2 = max(0, min(y2, h - 1))

                # 确保x2 > x1, y2 > y1
                if x2 <= x1 or y2 <= y1:
                    print(f"⚠️  Invalid box coordinates for '{label}': [{x1}, {y1}, {x2}, {y2}]")
                    continue

                # 检查是否检测到车辆类别
                vehicle_keywords = ["car", "truck", "bus", "vehicle", "van", "suv", "motorcycle", "bike"]
                if any(keyword in label.lower() for keyword in vehicle_keywords):
                    vehicle_detected = True

                    # 对车辆类别扩展边界框 - 第一帧更保守
                    width = x2 - x1
                    height = y2 - y1
                    expand_ratio = 0.15 if is_first_frame else 0.1
                    expand_w = int(width * expand_ratio)
                    expand_h = int(height * expand_ratio)

                    x1 = max(0, x1 - expand_w)
                    y1 = max(0, y1 - expand_h)
                    x2 = min(w, x2 + expand_w)
                    y2 = min(h, y2 + expand_h)

                    print(f"🚗 {'[FIRST FRAME] ' if is_first_frame else ''}Expanded vehicle '{label}' "
                          f"(conf={score:.3f}): expanded by {expand_w}x{expand_h}")
                else:
                    print(f"👤 Non-vehicle detection '{label}' (conf={score:.3f})")

                # 创建基础mask
                grounding_dino_mask[y1:y2, x1:x2] = 1

            final_mask = grounding_dino_mask.copy()
        except Exception as e:
            print(f"❌ Failed to process detection results: {e}")

        # 4. SAM精确分割（如果启用）
        use_sam_result = False  # 标记是否使用SAM结果
        if self.use_sam and len(boxes) > 0:
            try:
                sam_combined_mask = np.zeros((h, w), dtype=np.uint8)
                self.sam_predictor.set_image(image)

                successful_sam_count = 0
                for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
                    try:
                        x1, y1, x2, y2 = box.astype(int)

                        # 确保坐标有效
                        x1 = max(0, min(x1, w - 1))
                        y1 = max(0, min(y1, h - 1))
                        x2 = max(0, min(x2, w - 1))
                        y2 = max(0, min(y2, h - 1))

                        if x2 <= x1 or y2 <= y1:
                            continue

                        # 对车辆进行扩展（与上面保持一致）
                        vehicle_keywords = ["car", "truck", "bus", "vehicle", "van", "suv", "motorcycle", "bike"]
                        if any(keyword in label.lower() for keyword in vehicle_keywords):
                            width = x2 - x1
                            height = y2 - y1
                            expand_ratio = 0.15 if is_first_frame else 0.1
                            expand_w = int(width * expand_ratio)
                            expand_h = int(height * expand_ratio)
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
                            sam_combined_mask = np.logical_or(sam_combined_mask, best_mask).astype(np.uint8)
                            successful_sam_count += 1
                            print(f"✅ SAM refined '{label}' detection")
                    except Exception as e:
                        print(f"❌ SAM failed for '{label}': {e}")

                if sam_combined_mask.sum() > 0:
                    final_mask = sam_combined_mask
                    sam_mask = sam_combined_mask
                    use_sam_result = True  # 标记使用了SAM结果
                    print(f"🎯 SAM successfully refined {successful_sam_count}/{len(boxes)} detections")
                    print(
                        f"📊 SAM mask pixels: {sam_mask.sum()}, Grounding DINO mask pixels: {grounding_dino_mask.sum()}")
            except Exception as e:
                print(f"❌ SAM processing failed: {e}")

        # 5. 运动检测增强 - 修复：不要覆盖SAM mask，而是组合
        if not is_first_frame:  # 第一帧不使用运动检测
            try:
                motion_refined_mask = self._refine_with_motion(image, final_mask)
                if motion_refined_mask is not None:
                    motion_mask = motion_refined_mask

                    # 修复：如果使用了SAM，将运动检测作为补充而不是替换
                    if use_sam_result:
                        # 只在SAM没有检测到的区域使用运动检测
                        # 或者扩展SAM检测到的区域
                        final_mask = np.logical_or(final_mask, motion_refined_mask).astype(np.uint8)
                        print(f"🏃 Motion detection added to SAM mask (combined)")
                    else:
                        # 如果没有SAM，可以使用运动检测结果
                        final_mask = motion_refined_mask
                        print(f"🏃 Motion detection used as primary mask")
            except Exception as e:
                print(f"❌ Motion detection failed: {e}")

        # 6. 时间一致性滤波 - 修复：对于第一帧或使用SAM的情况要更保守
        if not is_first_frame and not use_sam_result:
            # 只有在没有使用SAM且不是第一帧时才使用时间一致性
            try:
                final_mask = self._temporal_consistency(final_mask)
                print(f"⏱️  Applied temporal consistency filter")
            except Exception as e:
                print(f"❌ Temporal consistency failed: {e}")
        elif use_sam_result:
            print(f"⏭️  Skipping temporal consistency (SAM result is more reliable)")

        # 7. 车辆特殊处理 - 第一帧更保守
        if vehicle_detected and final_mask.sum() > 0:
            try:
                kernel_size = 7 if is_first_frame else 5
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                final_mask = cv2.dilate(final_mask, kernel, iterations=1)
                print(
                    f"🛡️  {'[FIRST FRAME] ' if is_first_frame else ''}Applied additional dilation for vehicle detection")
            except Exception as e:
                print(f"❌ Dilation failed: {e}")

        # 调试信息：比较各个mask的差异
        if sam_mask is not None:
            print(f"🔍 Mask comparison:")
            print(f"  - Grounding DINO mask: {grounding_dino_mask.sum()} pixels")
            print(f"  - SAM mask: {sam_mask.sum()} pixels")
            print(f"  - Final mask: {final_mask.sum()} pixels")
            if motion_mask is not None:
                print(f"  - Motion mask: {motion_mask.sum()} pixels")

        # 8. 保存所有结果
        if frame_idx is not None and self.save_images:
            try:
                self.save_detection_results(
                    image, frame_idx,
                    grounding_dino_mask=grounding_dino_mask,
                    sam_mask=sam_mask,
                    motion_mask=motion_mask,
                    final_mask=final_mask,
                    boxes=boxes,
                    labels=labels,
                    scores=scores
                )
            except Exception as e:
                print(f"⚠️  Failed to save detection results: {e}")

        # 9. 输出检测统计
        try:
            detection_stats = {}
            if labels:
                for label in labels:
                    detection_stats[label] = detection_stats.get(label, 0) + 1

            print(f"📊 {'[FIRST FRAME] ' if is_first_frame else ''}Frame {frame_idx} summary:")
            print(f"  - Scene: {getattr(self.prompt_manager, 'current_scene', 'unknown')}")
            print(f"  - Detections: {len(boxes)} objects")
            print(f"  - Vehicle detected: {vehicle_detected}")
            print(f"  - Max confidence: {max_confidence:.3f}")
            print(f"  - Final mask pixels: {final_mask.sum()}")
            print(f"  - Detection breakdown: {detection_stats}")
        except Exception as e:
            print(f"❌ Failed to print statistics: {e}")

        # 标记第一帧已处理
        if is_first_frame:
            self.first_frame_processed = True
            print(f"✅ First frame {frame_idx} processing completed")

        return final_mask, max_confidence, image.copy()

    def _refine_with_motion(self, current_frame, detection_mask):
        """使用光流运动信息优化mask - 修复版本"""
        if self.prev_frame is None:
            self.prev_frame = cv2.cvtColor(current_frame, cv2.COLOR_RGB2GRAY)
            return None  # 第一次调用时返回None，不修改mask

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

            # 修复：正确处理运动和静止区域的组合
            # 移动的物体保留
            moving_objects = np.logical_and(detection_mask, motion_mask).astype(np.uint8)

            # 对静止的检测物体，如果它们在检测mask中，也要保留一部分
            # 修复：避免使用0.5的浮点数
            static_detection = np.logical_and(detection_mask, ~motion_mask).astype(np.uint8)

            # 组合：移动物体完全保留，静止检测物体也保留（可能是暂时静止的车）
            refined_mask = np.logical_or(moving_objects, static_detection).astype(np.uint8)

            self.prev_frame = current_gray

            print(
                f"🏃 Motion refinement: moving={moving_objects.sum()}, static={static_detection.sum()}, refined={refined_mask.sum()}")

            return refined_mask

        except Exception as e:
            print(f"❌ Motion detection failed: {e}")
            self.prev_frame = current_gray
            return None  # 失败时返回None，不修改原mask

    def get_static_mask_for_gaussian_init(self, image, frame_idx=None):
        """
        为高斯体初始化获取静态区域mask（移除地面修复功能）
        这个函数会返回可以用于初始化高斯体的区域（排除动态物体）

        Returns:
            static_mask: 静态区域mask
            image: 原始图像（无修复）
        """
        # 获取动态物体mask
        dynamic_mask, confidence, original_image = self.detect_and_segment(image, frame_idx)

        # 静态mask是动态mask的反向
        static_mask = (1 - dynamic_mask).astype(np.uint8)

        return static_mask, original_image

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

        # 初始化增强的动态物体遮罩器（移除地面功能）
        self.enable_dynamic_filtering = config.get("dynamic_filtering", {}).get("enabled", True)
        self.filter_initialization = config.get("dynamic_filtering", {}).get("filter_initialization", True)
        self.save_masked_images = config.get("dynamic_filtering", {}).get("save_masked_images", True)

        if self.enable_dynamic_filtering and self.filter_initialization:
            # 设置保存目录
            mask_save_dir = config.get("dynamic_filtering", {}).get("save_dir", "./masked_images")
            scene_type = config.get("dynamic_filtering", {}).get("scene_type", "outdoor_street")

            # 创建动态遮罩器（移除地面相关参数）
            self.dynamic_masker = EnhancedDynamicObjectMasker(
                device=self.device,
                use_sam=config.get("dynamic_filtering", {}).get("use_sam", True),
                sam_checkpoint="/home/zwk/下载/S3PO-GS-main/utils/sam_vit_b_01ec64.pth",
                save_dir=mask_save_dir,
                save_images=self.save_masked_images,
                scene_type=scene_type
            )

            # 从配置设置场景
            self.dynamic_masker.set_scene_from_config(config)

            print(f"🎯 Enhanced Dynamic Filtering with local Grounding DINO:")
            print(f"  - Model: /home/zwk/下载/S3PO-GS-main/groundingdino_swint_ogc.pth")
            print(f"  - Enabled: {self.enable_dynamic_filtering}")
            print(f"  - Filter initialization: {self.filter_initialization}")
            print(f"  - SAM: {config.get('dynamic_filtering', {}).get('use_sam', True)}")
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
        """改进的关键帧添加，特别处理第一帧（移除地面功能）"""
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

        # ===== 核心修改：使用Grounding DINO生成动态物体遮罩（移除地面功能） =====
        dynamic_mask = None
        static_mask = None
        is_first_frame = (cur_frame_idx == 0)

        if self.enable_dynamic_filtering and (not init or self.filter_initialization):
            # 转换图像格式用于Grounding DINO
            img_np = gt_img.permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
            img_np = (img_np * 255).astype(np.uint8)

            if is_first_frame:
                print(f"🔰 Processing FIRST keyframe {cur_frame_idx} with enhanced detection")
                # 给第一帧一些额外时间确保模型完全加载
                time.sleep(0.2)

            # 使用新的Grounding DINO方法获取静态mask（无地面修复）
            static_mask_np, original_image_np = self.dynamic_masker.get_static_mask_for_gaussian_init(
                img_np, frame_idx=cur_frame_idx
            )

            dynamic_mask = torch.from_numpy(1 - static_mask_np).to(self.device).bool()
            static_mask = torch.from_numpy(static_mask_np).to(self.device).bool()

            # 关键改动：从有效区域中完全排除动态物体
            # 扩展动态mask边界，确保边界处也不生成高斯体 - 第一帧更保守
            kernel_size = 9 if is_first_frame else 7
            expanded_dynamic_mask = self._expand_dynamic_mask(dynamic_mask, kernel_size=kernel_size)
            expanded_static_mask = ~expanded_dynamic_mask

            # 应用扩展后的静态mask
            valid_rgb = valid_rgb & expanded_static_mask[None]

            # 存储mask信息
            viewpoint.dynamic_mask = dynamic_mask
            viewpoint.expanded_dynamic_mask = expanded_dynamic_mask
            viewpoint.static_mask = static_mask
            viewpoint.expanded_static_mask = expanded_static_mask

            # 打印统计信息
            static_ratio = static_mask.float().mean().item()
            expanded_static_ratio = expanded_static_mask.float().mean().item()

            print(f"🔧 {'[FIRST FRAME] ' if is_first_frame else ''}Frame {cur_frame_idx} processing:")
            print(f"  📊 Original static ratio: {static_ratio:.1%}")
            print(f"  📊 Expanded static ratio: {expanded_static_ratio:.1%}")
            print(f"  🛡️  Safety margin: {(static_ratio - expanded_static_ratio) * 100:.1f}%")

            # 第一帧检查
            if is_first_frame and expanded_static_ratio < 0.15:
                print(f"🚨 CRITICAL: First frame static region only {expanded_static_ratio:.1%}!")
                print("    This will severely affect initialization!")
                print("    Consider:")
                print("    1. Checking if scene type is correct")
                print("    2. Reducing confidence threshold")
                print("    3. Reducing mask expansion")

            # 如果静态区域太少，发出警告
            if expanded_static_ratio < 0.2:
                print(f"⚠️  WARNING: Expanded static region only {expanded_static_ratio:.1%}!")
                print("    This may severely affect reconstruction quality.")
                print("    Consider reducing mask expansion or adjusting scene prompts.")
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
        """跟踪函数，包含Grounding DINO动态物体过滤（移除地面功能）"""
        # 生成动态物体遮罩（主要用于统计和可视化）
        if self.enable_dynamic_filtering:
            gt_img = viewpoint.original_image
            img_np = gt_img.permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)

            # 使用Grounding DINO检测方法（无地面修复）
            static_mask_np, original_image_np = self.dynamic_masker.get_static_mask_for_gaussian_init(
                img_np, frame_idx=cur_frame_idx
            )

            dynamic_mask = torch.from_numpy(1 - static_mask_np).to(self.device).bool()
            static_mask = torch.from_numpy(static_mask_np).to(self.device).bool()

            viewpoint.dynamic_mask = dynamic_mask
            viewpoint.static_mask = static_mask

            static_ratio = viewpoint.static_mask.float().mean().item()
            print(f"🎬 Tracking frame {cur_frame_idx}: Static ratio={static_ratio:.1%} (Grounding DINO)")

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
        """为关键帧保存特殊标记的掩码图像（移除地面功能）"""
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
            kf_path = os.path.join(kf_dir, f"keyframe_{cur_frame_idx:06d}_grounding_dino.jpg")
            cv2.imwrite(kf_path, kf_img)

            print(
                f"💾 Saved Grounding DINO keyframe mask for frame {cur_frame_idx} (Red=Dynamic, Yellow=Expanded)")

        except Exception as e:
            print(f"Warning: Failed to save keyframe mask for frame {cur_frame_idx}: {e}")

    def is_keyframe(self, cur_frame_idx, last_keyframe_idx, cur_frame_visibility_filter, occ_aware_visibility):
        """考虑Grounding DINO动态物体检测的关键帧选择策略（移除地面功能）"""
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
                # 静态区域太少，更积极创建关键帧
                adjusted_overlap = kf_overlap * 0.7
                print(
                    f"🔄 Limited static region ({static_ratio:.1%}) after Grounding DINO, adjusted overlap: {adjusted_overlap:.3f}")
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
        """改进的初始化方法，确保第一帧正确处理（移除地面功能）"""
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

        # 在初始化阶段就应用Grounding DINO动态过滤（移除地面功能）
        print(f"🔄 INITIALIZING with frame {cur_frame_idx}")
        if self.enable_dynamic_filtering and self.filter_initialization:
            print("  ✅ Enhanced first-frame Grounding DINO filtering ENABLED")
            print("  🔰 Using conservative detection for initialization")
            print(f"  🎯 Scene type: {self.dynamic_masker.prompt_manager.current_scene}")

            # 给动态遮罩器一些时间确保模型完全加载
            time.sleep(0.1)

        elif self.enable_dynamic_filtering and not self.filter_initialization:
            print("  ⚠️  Grounding DINO filtering enabled but SKIPPING initialization frame")
        else:
            print("  ❌ Dynamic filtering DISABLED - cars may appear as ghosts!")

        self.kf_indices = []
        depth_map = self.add_new_keyframe(cur_frame_idx, init=True)
        self.request_init(cur_frame_idx, viewpoint, depth_map)
        self.reset = False

    def run(self):
        # 主执行循环集成了Grounding DINO动态物体过滤（移除地面功能）
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