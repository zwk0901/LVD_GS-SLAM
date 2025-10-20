import torch
import numpy as np
import cv2
from ultralytics import YOLO
from segment_anything import SamPredictor, sam_model_registry
import torch.nn.functional as F


class DynamicObjectFilter:
    """动态物体过滤器，用于在3D高斯重建中移除动态物体"""

    def __init__(self, device='cuda'):
        self.device = device

        # 加载YOLO分割模型
        self.yolo_model = YOLO('yolov8n-seg.pt')  # 或使用更大的模型如yolov8s-seg.pt

        # 定义动态物体类别ID (COCO数据集)
        self.dynamic_classes = {
            0: 'person',  # 人
            1: 'bicycle',  # 自行车
            2: 'car',  # 汽车
            3: 'motorcycle',  # 摩托车
            5: 'bus',  # 公交车
            7: 'truck',  # 卡车
            14: 'bird',  # 鸟
            15: 'cat',  # 猫
            16: 'dog',  # 狗
            17: 'horse',  # 马
            18: 'sheep',  # 羊
            19: 'cow',  # 牛
        }

        # 可选：加载SAM用于更精细的分割
        self.use_sam = False
        self.sam_predictor = None

    def enable_sam(self, sam_checkpoint_path="sam_vit_h_4b8939.pth"):
        """启用SAM进行更精细的分割"""
        try:
            sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint_path)
            sam.to(device=self.device)
            self.sam_predictor = SamPredictor(sam)
            self.use_sam = True
            print("SAM模型加载成功")
        except Exception as e:
            print(f"SAM模型加载失败: {e}")
            self.use_sam = False

    def detect_dynamic_objects(self, image):
        """
        检测图像中的动态物体

        Args:
            image: numpy array, shape (H, W, 3), RGB格式图像

        Returns:
            mask: numpy array, shape (H, W), 二值mask，1表示动态物体区域
        """
        # YOLO检测
        results = self.yolo_model(image)

        # 创建mask
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        for result in results:
            if result.masks is not None:
                # 获取分割结果
                masks = result.masks.data.cpu().numpy()
                boxes = result.boxes.data.cpu().numpy()

                for i, box in enumerate(boxes):
                    class_id = int(box[5])
                    confidence = box[4]

                    # 只处理动态物体且置信度高的检测结果
                    if class_id in self.dynamic_classes and confidence > 0.5:
                        # 调整mask尺寸
                        obj_mask = cv2.resize(masks[i], (w, h))
                        obj_mask = (obj_mask > 0.5).astype(np.uint8)

                        # 膨胀操作，确保完全覆盖物体边缘
                        kernel = np.ones((5, 5), np.uint8)
                        obj_mask = cv2.dilate(obj_mask, kernel, iterations=2)

                        mask = np.maximum(mask, obj_mask)

        return mask

    def refine_mask_with_sam(self, image, rough_mask):
        """使用SAM精细化mask"""
        if not self.use_sam or self.sam_predictor is None:
            return rough_mask

        self.sam_predictor.set_image(image)

        # 找到rough_mask的边界框
        contours, _ = cv2.findContours(rough_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        refined_mask = np.zeros_like(rough_mask)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w * h < 100:  # 忽略太小的区域
                continue

            # 使用SAM进行精细分割
            masks, _, _ = self.sam_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=np.array([x, y, x + w, y + h])[None, :],
                multimask_output=False,
            )

            refined_mask = np.maximum(refined_mask, masks[0].astype(np.uint8))

        return refined_mask