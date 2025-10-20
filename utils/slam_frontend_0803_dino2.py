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
                "confidence_threshold": 0.15,
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
    """Grounding DINO 检测器封装，支持本地.pth文件"""

    def __init__(self, model_path=None, config_path=None, device="cuda"):
        self.device = device
        self.model = None
        self.processor = None
        self.use_original = False
        self.use_transformers = False
        self.use_yolo = False

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
                    os.path.join(os.path.dirname(model_path), "GroundingDINO_SwinT_OGC.cfg.py"),
                    os.path.join(os.path.dirname(model_path), "config.py"),
                    "/home/zwk/下载/S3PO-GS-main/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
                    "./GroundingDINO_SwinT_OGC.cfg.py"
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
            self.yolo_model = YOLO('yolov8m.pt')
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

# 在 FrontEnd.__init__ 中使用更合适的默认模型
class FrontEnd(mp.Process):
    def __init__(self, config, model, save_dir=None):
        super().__init__()
        # ... (其他初始化代码保持不变)

        if self.enable_dynamic_filtering and self.filter_initialization:
            # 设置保存目录
            mask_save_dir = config.get("dynamic_filtering", {}).get("save_dir", "./masked_images")
            scene_type = config.get("dynamic_filtering", {}).get("scene_type", "outdoor_street")

            # 使用更合适的默认模型
            grounding_dino_model = config.get("dynamic_filtering", {}).get(
                "grounding_dino_model",
                "IDEA-Research/grounding-dino-tiny"  # 使用tiny作为默认
            )

            self.dynamic_masker = EnhancedDynamicObjectMasker(
                device=self.device,
                use_sam=config.get("dynamic_filtering", {}).get("use_sam", True),
                save_dir=mask_save_dir,
                save_images=self.save_masked_images,
                use_ground_segmentation=self.use_ground_segmentation,
                scene_type=scene_type,
                grounding_dino_model=grounding_dino_model
            )

            # ... (其余代码保持不变)

# EnhancedDynamicObjectMasker 类的修改部分
class EnhancedDynamicObjectMasker:
    def __init__(self, device="cuda", use_sam=True,
                 sam_checkpoint="/home/zwk/下载/S3PO-GS-main/utils/sam_vit_b_01ec64.pth",
                 save_dir=None, save_images=True, use_ground_segmentation=True,
                 scene_type="outdoor_street",
                 grounding_dino_model="/home/zwk/下载/S3PO-GS-main/groundingdino_swint_ogc.pth",
                 grounding_dino_config=None):
        """
        增强的动态物体遮罩器，支持本地.pth文件
        """
        self.device = device
        self.initialization_success = True

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
            if grounding_dino_model and grounding_dino_model.endswith('.pth'):
                self.grounding_detector = GroundingDINODetector(
                    model_path=grounding_dino_model,
                    config_path=grounding_dino_config,
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
                    sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
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

    def _print_configuration(self, grounding_dino_model):
        """打印配置信息"""
        try:
            if self.prompt_manager:
                prompt_info = self.prompt_manager.get_detailed_prompt()
                print(f"🎯 Enhanced Dynamic Object Masker Configuration:")
                print(f"  - Scene type: {self.prompt_manager.current_scene}")
                print(f"  - Scene description: {prompt_info['description']}")
                print(f"  - Target objects: {', '.join(prompt_info['object_classes'][:8])}...")
                print(f"  - Confidence threshold: {prompt_info['confidence_threshold']}")
            else:
                print(f"🎯 Enhanced Dynamic Object Masker Configuration:")
                print(f"  - Scene manager: FAILED")

            print(f"  - Grounding DINO model: {grounding_dino_model}")
            print(
                f"  - Grounding DINO status: {'✅ OK' if self.grounding_detector and not self.grounding_detector.fallback_mode else '⚠️  FALLBACK'}")
            print(f"  - SAM enabled: {self.use_sam}")
            print(f"  - Ground segmentation: {self.use_ground_segmentation}")
            print(f"  - Save images: {self.save_images}")

            if not self.initialization_success:
                print(f"⚠️  WARNING: Some components failed to initialize!")
                print(f"  - The system will continue with reduced functionality")
                print(f"  - Consider checking dependencies and model files")

        except Exception as e:
            print(f"❌ Failed to print configuration: {e}")

    def _fallback_detection(self, image, frame_idx=None):
        """
        当Grounding DINO不可用时的fallback检测
        可以在这里集成其他检测方法
        """
        print(f"🔄 Using fallback detection for frame {frame_idx}")
        h, w = image.shape[:2]

        # 返回空的检测结果，但保持系统运行
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

    def _init_ground_segmentation(self):
        """初始化地面分割模型"""
        try:
            self.ground_segmentation_method = "traditional"
            print("✅ Ground segmentation initialized with traditional method")
        except Exception as e:
            print(f"Warning: Ground segmentation failed: {e}")
            self.ground_segmentation_method = "traditional"

    def _create_save_directories(self):
        """创建保存图像的目录结构"""
        directories = [
            self.save_dir,
            os.path.join(self.save_dir, "original"),
            os.path.join(self.save_dir, "grounding_dino_detections"),  # Grounding DINO检测框
            os.path.join(self.save_dir, "grounding_dino_masks"),  # Grounding DINO生成的mask
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
            os.path.join(self.save_dir, "detection_analysis"),  # 检测分析结果
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)

        print(f"📁 Created mask image directories in: {self.save_dir}")

    def segment_ground(self, image):
        """
        分割图像中的地面区域

        Args:
            image: RGB图像 [H, W, 3]

        Returns:
            ground_mask: 地面mask [H, W], 1表示地面，0表示非地面
        """
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

    def detect_and_segment(self, image, frame_idx=None):
        """
        使用Grounding DINO检测动态物体并生成精确分割mask

        Returns:
            final_mask: 动态物体mask (1=dynamic, 0=static)
            max_confidence: 最高检测置信度
            repaired_image: 修复后的图像
        """
        h, w = image.shape[:2]
        final_mask = np.zeros((h, w), dtype=np.uint8)
        grounding_dino_mask = np.zeros((h, w), dtype=np.uint8)
        repaired_image = image.copy()
        max_confidence = 0.0
        self._last_detected_objects = []

        # 检查是否应该使用fallback模式
        if not self.grounding_detector or not self.initialization_success:
            print(f"⚠️  Using fallback mode for frame {frame_idx}")
            return final_mask, 0.0, repaired_image

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
            text_prompt, confidence_threshold = self.prompt_manager.get_current_prompt()
            print(f"🎯 Using prompt: '{text_prompt[:50]}...' (confidence: {confidence_threshold})")
        except Exception as e:
            print(f"❌ Failed to get prompt: {e}")
            text_prompt = "car. truck. person. bicycle"
            confidence_threshold = 0.2

        # 3. Grounding DINO检测
        try:
            boxes, scores, labels = self.grounding_detector.detect(
                image, text_prompt, confidence_threshold
            )
        except Exception as e:
            print(f"❌ Grounding DINO detection failed: {e}")
            boxes, scores, labels = np.array([]), np.array([]), []

        if len(boxes) == 0:
            print(f"ℹ️  No dynamic objects detected in frame {frame_idx}")
            if frame_idx is not None and self.save_images:
                self.save_detection_results(
                    image, frame_idx,
                    grounding_dino_mask=grounding_dino_mask,
                    final_mask=final_mask,
                    boxes=[], labels=[], scores=[],
                    ground_mask=ground_mask,
                    repaired_image=repaired_image
                )
            return final_mask, 0.0, repaired_image

        # 4. 处理检测结果，创建动态物体mask
        vehicle_detected = False
        print(f"🎯 Detected {len(boxes)} dynamic objects:")

        for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            max_confidence = max(max_confidence, score)

            # 记录检测到的物体
            self._last_detected_objects.append({
                'label': label,
                'box': box,
                'score': score,
                'frame': frame_idx
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

        # 5. SAM精确分割（如果启用）
        if self.use_sam and len(boxes) > 0:
            try:
                sam_combined_mask = np.zeros((h, w), dtype=np.uint8)
                self.sam_predictor.set_image(image)

                for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
                    try:
                        x1, y1, x2, y2 = box.astype(int)

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
                            sam_combined_mask = np.logical_or(sam_combined_mask, best_mask).astype(np.uint8)
                            print(f"  ✅ SAM refined '{label}' mask")
                    except Exception as e:
                        print(f"  ❌ SAM failed for '{label}': {e}")

                if sam_combined_mask.sum() > 0:
                    final_mask = sam_combined_mask
                    print(f"🎯 SAM refinement complete")
            except Exception as e:
                print(f"❌ SAM processing failed: {e}")

        # 6. 地面修复（如果检测到车辆）
        shadow_regions = None
        if self.use_ground_segmentation and ground_mask is not None and vehicle_detected:
            try:
                repaired_image, shadow_regions = self.repair_ground_shadows(image, final_mask, ground_mask)
                print(f"🔧 Ground repair: {np.sum(shadow_regions) if shadow_regions is not None else 0} pixels")
            except Exception as e:
                print(f"❌ Ground repair failed: {e}")

        # 7. 时间一致性滤波
        try:
            final_mask = self._temporal_consistency(final_mask)
        except Exception as e:
            print(f"❌ Temporal consistency failed: {e}")

        # 8. 最终的膨胀处理
        if final_mask.sum() > 0:
            kernel = np.ones((5, 5), np.uint8)
            final_mask = cv2.dilate(final_mask, kernel, iterations=1)
            print(f"🛡️  Applied final dilation to dynamic mask")

        # 9. 保存结果
        if frame_idx is not None and self.save_images:
            self.save_detection_results(
                image, frame_idx,
                grounding_dino_mask=grounding_dino_mask,
                final_mask=final_mask,
                boxes=boxes,
                labels=labels,
                scores=scores,
                ground_mask=ground_mask,
                shadow_regions=shadow_regions,
                repaired_image=repaired_image
            )

        # 输出统计
        print(f"📊 Frame {frame_idx} final statistics:")
        print(f"  - Total dynamic pixels: {final_mask.sum()} ({np.mean(final_mask) * 100:.1f}% of image)")
        print(f"  - Max confidence: {max_confidence:.3f}")

        return final_mask, max_confidence, repaired_image

    def get_static_mask_for_gaussian_init(self, image, frame_idx=None):
        """
        为高斯体初始化获取静态区域mask
        返回的static_mask中：
        - 1 表示静态区域（应该被重建）
        - 0 表示动态区域（应该被mask掉）

        Returns:
            static_mask: 静态区域mask (1=static, 0=dynamic)
            repaired_image: 用于初始化的修复图像
            detected_objects: 检测到的动态物体列表
        """
        # 获取动态物体mask和修复图像
        dynamic_mask, confidence, repaired_image = self.detect_and_segment(image, frame_idx)

        # 静态mask是动态mask的反向
        # dynamic_mask中 1=动态物体，0=静态背景
        # static_mask中 1=静态背景，0=动态物体
        static_mask = (1 - dynamic_mask).astype(np.uint8)

        # 获取检测到的物体信息
        detected_objects = []
        if hasattr(self, '_last_detected_objects'):
            detected_objects = self._last_detected_objects

        print(f"📊 Frame {frame_idx} mask summary:")
        print(f"  - Dynamic pixels (masked out): {np.sum(dynamic_mask > 0)} ({np.mean(dynamic_mask) * 100:.1f}%)")
        print(f"  - Static pixels (reconstructed): {np.sum(static_mask > 0)} ({np.mean(static_mask) * 100:.1f}%)")

        return static_mask, repaired_image, detected_objects

    def _refine_with_motion(self, current_frame, detection_mask):
        """使用光流运动信息优化mask"""
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
            print(f"❌ Motion detection failed: {e}")
            self.prev_frame = current_gray
            return detection_mask

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
        self.theta = 0

        # 初始化增强的动态物体遮罩器（使用Grounding DINO 2）
        self.enable_dynamic_filtering = config.get("dynamic_filtering", {}).get("enabled", True)
        self.filter_initialization = config.get("dynamic_filtering", {}).get("filter_initialization", True)
        self.save_masked_images = config.get("dynamic_filtering", {}).get("save_masked_images", True)
        self.use_ground_segmentation = config.get("dynamic_filtering", {}).get("use_ground_segmentation", True)

        if self.enable_dynamic_filtering and self.filter_initialization:
            # 设置保存目录
            mask_save_dir = config.get("dynamic_filtering", {}).get("save_dir", "./masked_images")
            scene_type = config.get("dynamic_filtering", {}).get("scene_type", "outdoor_street")
            grounding_dino_model = config.get("dynamic_filtering", {}).get("grounding_dino_model",
                                                                           "IDEA-Research/grounding-dino-1.5-pro")

            self.dynamic_masker = EnhancedDynamicObjectMasker(
                device=self.device,
                use_sam=config.get("dynamic_filtering", {}).get("use_sam", True),
                save_dir=mask_save_dir,
                save_images=self.save_masked_images,
                use_ground_segmentation=self.use_ground_segmentation,
                scene_type=scene_type,
                grounding_dino_model=grounding_dino_model
            )

            # 从配置设置场景
            self.dynamic_masker.set_scene_from_config(config)

            print(f"🎯 Enhanced Dynamic Filtering with Grounding DINO 2:")
            print(f"  - Enabled: {self.enable_dynamic_filtering}")
            print(f"  - Filter initialization: {self.filter_initialization}")
            print(f"  - SAM: {config.get('dynamic_filtering', {}).get('use_sam', True)}")
            print(f"  - Ground segmentation: {self.use_ground_segmentation}")
            print(f"  - Model: {grounding_dino_model}")
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
        """添加新关键帧，使用Grounding DINO 2动态物体掩码和地面修复"""
        rgb_boundary_threshold = self.config["Training"]["rgb_boundary_threshold"]
        if len(self.kf_indices) > 0:
            last_kf = self.kf_indices[-1]
            viewpoint_last = self.cameras[last_kf]
            R_last = viewpoint_last.R

        self.kf_indices.append(cur_frame_idx)
        viewpoint = self.cameras[cur_frame_idx]

        gt_img = viewpoint.original_image.cuda()
        valid_rgb = (gt_img.sum(dim=0) > rgb_boundary_threshold)[None]

        # ===== 核心：使用动态物体掩码 =====
        if self.enable_dynamic_filtering and (not init or self.filter_initialization):
            # 转换图像格式
            img_np = gt_img.permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)

            # 获取静态mask（1=静态要重建，0=动态要mask掉）
            static_mask_np, repaired_image_np, detected_objects = self.dynamic_masker.get_static_mask_for_gaussian_init(
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

            # 打印统计
            static_ratio = static_mask.float().mean().item()
            expanded_static_ratio = expanded_static_mask.float().mean().item()
            num_dynamic_objects = len(detected_objects)

            print(f"🔧 Frame {cur_frame_idx} keyframe processing:")
            print(f"  - Detected {num_dynamic_objects} dynamic objects")
            print(f"  - Original static ratio: {static_ratio:.1%}")
            print(f"  - After expansion: {expanded_static_ratio:.1%}")
            print(f"  - Dynamic objects MASKED OUT from reconstruction")

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

        return initial_depth[0].numpy() if 'initial_depth' in locals() else None
    def tracking(self, cur_frame_idx, viewpoint):
        """跟踪函数，包含Grounding DINO 2地面修复的动态物体过滤"""
        # 生成动态物体遮罩（主要用于统计和可视化）
        if self.enable_dynamic_filtering:
            gt_img = viewpoint.original_image
            img_np = gt_img.permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)

            # 使用带地面修复的Grounding DINO 2检测方法
            static_mask_np, repaired_image_np = self.dynamic_masker.get_static_mask_for_gaussian_init(
                img_np, frame_idx=cur_frame_idx
            )

            dynamic_mask = torch.from_numpy(1 - static_mask_np).to(self.device).bool()
            static_mask = torch.from_numpy(static_mask_np).to(self.device).bool()

            viewpoint.dynamic_mask = dynamic_mask
            viewpoint.static_mask = static_mask

            static_ratio = viewpoint.static_mask.float().mean().item()
            print(
                f"🎬 Tracking frame {cur_frame_idx}: Static ratio={static_ratio:.1%} (Grounding DINO 2 + ground repair)")

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
        """保存关键帧的掩码可视化"""
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
                for obj in viewpoint.detected_objects[:5]:  # 最多显示5个
                    label = f"{obj['label']}: {obj['score']:.2f}"
                    cv2.putText(vis_img, f"  - {label}",
                                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
                    y_offset += 25

            # 保存图像
            kf_path = os.path.join(kf_dir, f"keyframe_{cur_frame_idx:06d}_mask_vis.jpg")
            cv2.imwrite(kf_path, vis_img)

            print(f"💾 Saved keyframe mask visualization for frame {cur_frame_idx}")

        except Exception as e:
            print(f"Warning: Failed to save keyframe mask for frame {cur_frame_idx}: {e}")

    def is_keyframe(self, cur_frame_idx, last_keyframe_idx, cur_frame_visibility_filter, occ_aware_visibility):
        """考虑Grounding DINO 2动态物体检测的关键帧选择策略"""
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
                    f"🔄 Limited static region ({static_ratio:.1%}) after Grounding DINO 2 + ground repair, adjusted overlap: {adjusted_overlap:.3f}")
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

        # 在初始化阶段就应用Grounding DINO 2动态过滤和地面修复
        print(f"🔄 INITIALIZING with frame {cur_frame_idx}")
        if self.enable_dynamic_filtering and self.filter_initialization and self.use_ground_segmentation:
            print("  ✅ Grounding DINO 2 + Ground segmentation ENABLED for initialization")
            print("  🛠️  Ground shadows will be repaired automatically")
            print(f"  🎯 Scene type: {self.dynamic_masker.prompt_manager.current_scene}")
        elif self.enable_dynamic_filtering and self.filter_initialization:
            print("  ✅ Grounding DINO 2 filtering ENABLED (no ground repair)")
            print(f"  🎯 Scene type: {self.dynamic_masker.prompt_manager.current_scene}")
        elif self.enable_dynamic_filtering and not self.filter_initialization:
            print("  ⚠️  Grounding DINO 2 filtering enabled but SKIPPING initialization frame")
        else:
            print("  ❌ Dynamic filtering DISABLED - cars may appear as ghosts!")

        self.kf_indices = []
        depth_map = self.add_new_keyframe(cur_frame_idx, init=True)
        self.request_init(cur_frame_idx, viewpoint, depth_map)
        self.reset = False

    def run(self):
        # 主执行循环集成了Grounding DINO 2动态物体过滤和地面修复
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