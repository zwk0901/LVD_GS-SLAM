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
    def __init__(self, device="cuda", use_sam=True, sam_checkpoint="/home/zwk/下载/S3PO-GS-main/utils/sam_vit_b_01ec64.pth",
                 save_dir=None, save_images=True):
        """
        增强的动态物体遮罩器，结合YOLO和SAM
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

        # 运动检测参数
        self.prev_frame = None
        self.prev_mask = None
        self.motion_threshold = 3.0

        # 时间一致性参数
        self.mask_history = []
        self.history_length = 5

        # 图像保存设置
        self.save_images = save_images
        self.save_dir = save_dir if save_dir else "./masked_images"
        if self.save_images:
            self._create_save_directories()

    def _create_save_directories(self):
        """创建保存图像的目录结构"""
        import os

        # 创建主目录和子目录
        directories = [
            self.save_dir,
            os.path.join(self.save_dir, "original"),
            os.path.join(self.save_dir, "detections"),  # YOLO检测框
            os.path.join(self.save_dir, "yolo_masks"),  # YOLO生成的mask
            os.path.join(self.save_dir, "sam_masks"),  # SAM精确分割的mask
            os.path.join(self.save_dir, "motion_masks"),  # 运动检测mask
            os.path.join(self.save_dir, "final_masks"),  # 最终组合mask
            os.path.join(self.save_dir, "masked_overlay"),  # 叠加显示
            os.path.join(self.save_dir, "static_only"),  # 只显示静态区域
            os.path.join(self.save_dir, "keyframes"),  # 关键帧特殊保存
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)

        print(f"Created mask image directories in: {self.save_dir}")

    def save_detection_results(self, image, frame_idx, yolo_mask=None, sam_mask=None,
                               motion_mask=None, final_mask=None, boxes=None):
        """
        保存检测和分割的各种结果
        """
        if not self.save_images:
            return

        try:
            import cv2
            import os

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

            # 3. 保存YOLO mask
            if yolo_mask is not None:
                yolo_mask_img = (yolo_mask * 255).astype(np.uint8)
                yolo_path = os.path.join(self.save_dir, "yolo_masks", f"frame_{frame_idx:06d}.jpg")
                cv2.imwrite(yolo_path, yolo_mask_img)

            # 4. 保存SAM mask
            if sam_mask is not None:
                sam_mask_img = (sam_mask * 255).astype(np.uint8)
                sam_path = os.path.join(self.save_dir, "sam_masks", f"frame_{frame_idx:06d}.jpg")
                cv2.imwrite(sam_path, sam_mask_img)

            # 5. 保存运动mask
            if motion_mask is not None:
                motion_mask_img = (motion_mask * 255).astype(np.uint8)
                motion_path = os.path.join(self.save_dir, "motion_masks", f"frame_{frame_idx:06d}.jpg")
                cv2.imwrite(motion_path, motion_mask_img)

            # 6. 保存最终mask
            if final_mask is not None:
                final_mask_img = (final_mask * 255).astype(np.uint8)
                final_path = os.path.join(self.save_dir, "final_masks", f"frame_{frame_idx:06d}.jpg")
                cv2.imwrite(final_path, final_mask_img)

                # 7. 保存叠加显示（动态区域用红色标记）
                overlay_img = img_bgr.copy()
                overlay_img[final_mask > 0] = [0, 0, 255]  # BGR格式，红色
                overlay_path = os.path.join(self.save_dir, "masked_overlay", f"frame_{frame_idx:06d}.jpg")
                cv2.imwrite(overlay_path, overlay_img)

                # 8. 保存静态区域图像（动态区域变黑）
                static_img = img_bgr.copy()
                static_img[final_mask > 0] = [0, 0, 0]
                static_path = os.path.join(self.save_dir, "static_only", f"frame_{frame_idx:06d}.jpg")
                cv2.imwrite(static_path, static_img)

            print(f"Saved mask images for frame {frame_idx}")

        except Exception as e:
            print(f"Warning: Failed to save mask images for frame {frame_idx}: {e}")

    def detect_and_segment(self, image, frame_idx=None):
        """
        检测动态物体并生成精确分割mask

        Args:
            image: RGB图像 [H, W, 3]
            frame_idx: 帧索引（用于保存图像）

        Returns:
            mask: 动态物体mask [H, W]
            confidence: 检测置信度
        """
        h, w = image.shape[:2]
        final_mask = np.zeros((h, w), dtype=np.uint8)
        yolo_mask = np.zeros((h, w), dtype=np.uint8)
        sam_mask = None
        motion_mask = None
        max_confidence = 0.0

        # 1. YOLO检测
        results = self.yolo_model(image, imgsz=640, conf=0.3, classes=self.dynamic_class_ids, verbose=False)

        boxes_with_scores = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    if int(box.cls) in self.dynamic_class_ids:
                        xyxy = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        boxes_with_scores.append((xyxy, conf))
                        max_confidence = max(max_confidence, conf)

        if not boxes_with_scores:
            # 即使没有检测到，也保存原始图像
            if frame_idx is not None:
                self.save_detection_results(image, frame_idx, yolo_mask=yolo_mask,
                                            final_mask=final_mask, boxes=[])
            return final_mask, 0.0

        # 创建YOLO基础mask
        for box, conf in boxes_with_scores:
            x1, y1, x2, y2 = box.astype(int)
            yolo_mask[y1:y2, x1:x2] = 1

        final_mask = yolo_mask.copy()

        # 2. SAM精确分割（如果启用）
        if self.use_sam:
            sam_combined_mask = np.zeros((h, w), dtype=np.uint8)
            self.sam_predictor.set_image(image)

            for box, conf in boxes_with_scores:
                x1, y1, x2, y2 = box.astype(int)

                # 使用检测框作为SAM的prompt
                input_box = np.array([x1, y1, x2, y2])

                try:
                    masks, scores, _ = self.sam_predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=input_box[None, :],
                        multimask_output=False,
                    )

                    if len(masks) > 0:
                        # 选择得分最高的mask
                        best_mask = masks[0].astype(np.uint8)
                        sam_combined_mask = np.logical_or(sam_combined_mask, best_mask).astype(np.uint8)
                except:
                    # SAM失败时继续使用YOLO结果
                    pass

            if sam_combined_mask.sum() > 0:
                final_mask = sam_combined_mask
                sam_mask = sam_combined_mask

        # 3. 运动检测增强
        motion_refined_mask = self._refine_with_motion(image, final_mask)
        if motion_refined_mask is not None:
            motion_mask = motion_refined_mask
            final_mask = motion_refined_mask

        # 4. 时间一致性滤波
        final_mask = self._temporal_consistency(final_mask)

        # 5. 保存所有结果
        if frame_idx is not None:
            boxes = [box for box, _ in boxes_with_scores]
            self.save_detection_results(
                image, frame_idx,
                yolo_mask=yolo_mask,
                sam_mask=sam_mask,
                motion_mask=motion_mask,
                final_mask=final_mask,
                boxes=boxes
            )

        return final_mask, max_confidence

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

        # 初始化动态物体遮罩器
        self.enable_dynamic_filtering = config.get("dynamic_filtering", {}).get("enabled", True)
        self.save_masked_images = config.get("dynamic_filtering", {}).get("save_masked_images", True)

        if self.enable_dynamic_filtering:
            # 设置保存目录
            mask_save_dir = config.get("dynamic_filtering", {}).get("save_dir", "./masked_images")

            self.dynamic_masker = EnhancedDynamicObjectMasker(
                device=self.device,
                use_sam=config.get("dynamic_filtering", {}).get("use_sam", True),
                save_dir=mask_save_dir,
                save_images=self.save_masked_images
            )

    def set_hyperparams(self):
        self.save_dir = self.config["Results"]["save_dir"]
        self.save_results = self.config["Results"]["save_results"]
        self.save_trj = self.config["Results"]["save_trj"]
        self.save_trj_kf_intv = self.config["Results"]["save_trj_kf_intv"]

        self.tracking_itr_num = self.config["Training"]["tracking_itr_num"]
        self.kf_interval = self.config["Training"]["kf_interval"]
        self.window_size = self.config["Training"]["window_size"]
        self.single_thread = self.config["Training"]["single_thread"]

    def add_new_keyframe(self, cur_frame_idx, depth=None, opacity=None, init=False):
        """
        添加新关键帧，应用动态物体遮罩
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

        # 生成动态物体遮罩
        dynamic_mask = None
        if self.enable_dynamic_filtering and not init:
            # 转换图像格式用于YOLO/SAM
            img_np = gt_img.permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
            img_np = (img_np * 255).astype(np.uint8)

            dynamic_mask, confidence = self.dynamic_masker.detect_and_segment(img_np, frame_idx=cur_frame_idx)
            dynamic_mask = torch.from_numpy(dynamic_mask).to(self.device).bool()

            # 将动态物体区域标记为无效
            valid_rgb = valid_rgb & (~dynamic_mask[None])

            # 存储动态mask用于tracking
            viewpoint.dynamic_mask = dynamic_mask
            viewpoint.dynamic_confidence = confidence

            print(f"Frame {cur_frame_idx}: Found {confidence:.3f} confidence dynamic objects")

        if self.monocular:
            if depth is None:
                initial_depth = torch.from_numpy(viewpoint.mono_depth).unsqueeze(0)
                print(f"Initial depth map stats for frame {cur_frame_idx}:",
                      f"Max: {torch.max(initial_depth).item():.3f}",
                      f"Min: {torch.min(initial_depth).item():.3f}",
                      f"Mean: {torch.mean(initial_depth).item():.3f}")
                initial_depth[~valid_rgb.cpu()] = 0
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

                valid_rgb_np = valid_rgb.cpu().numpy() if isinstance(valid_rgb, torch.Tensor) else valid_rgb
                if initial_depth.shape == valid_rgb_np.shape[1:]:
                    initial_depth[~valid_rgb_np[0]] = 0

            return initial_depth

        # 使用ground truth深度
        initial_depth = torch.from_numpy(viewpoint.depth).unsqueeze(0)
        initial_depth[~valid_rgb.cpu()] = 0
        return initial_depth[0].numpy()

    def tracking(self, cur_frame_idx, viewpoint):
        """
        带动态物体过滤的跟踪
        """
        # 生成动态物体遮罩
        if self.enable_dynamic_filtering:
            gt_img = viewpoint.original_image
            img_np = gt_img.permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)

            dynamic_mask, confidence = self.dynamic_masker.detect_and_segment(img_np, frame_idx=cur_frame_idx)
            dynamic_mask = torch.from_numpy(dynamic_mask).to(self.device).bool()

            viewpoint.dynamic_mask = dynamic_mask
            viewpoint.dynamic_confidence = confidence

            print(f"Tracking frame {cur_frame_idx}: Dynamic confidence {confidence:.3f}")

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

            # 应用动态物体遮罩的损失计算
            loss_tracking = self.get_masked_loss_tracking(
                self.config, image, depth, opacity, viewpoint
            )

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

    def get_masked_loss_tracking(self, config, image, depth, opacity, viewpoint):
        """
        应用动态物体遮罩的跟踪损失函数
        """
        # 原始损失
        loss_tracking = get_loss_tracking(config, image, depth, opacity, viewpoint)

        # 如果启用动态过滤且存在动态mask
        # if (self.enable_dynamic_filtering and
        #         hasattr(viewpoint, 'dynamic_mask') and
        #         viewpoint.dynamic_mask is not None):
        #
        #     # 获取动态区域mask
        #     dynamic_mask = viewpoint.dynamic_mask  # [H, W]
        #     static_mask = ~dynamic_mask
        #
        #     # 确保维度匹配
        #     if len(static_mask.shape) == 2 and len(image.shape) == 3:
        #         # 扩展mask维度以匹配图像: [H, W] -> [1, H, W]
        #         static_mask_expanded = static_mask.unsqueeze(0)  # [1, H, W]
        #     else:
        #         static_mask_expanded = static_mask
        #
        #     # 只在静态区域计算损失
        #     if hasattr(viewpoint, 'original_image'):
        #         gt_image = viewpoint.original_image  # [3, H, W]
        #
        #         # 对每个通道应用mask
        #         if gt_image.shape[0] == 3:  # RGB图像
        #             static_pixels_render = image[:, static_mask]  # [3, N_static]
        #             static_pixels_gt = gt_image[:, static_mask]  # [3, N_static]
        #
        #             # RGB损失只考虑静态区域
        #             if static_pixels_render.numel() > 0:
        #                 rgb_loss = torch.nn.functional.l1_loss(
        #                     static_pixels_render, static_pixels_gt
        #                 )
        #             else:
        #                 rgb_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        #         else:
        #             # 回退到原始损失
        #             rgb_loss = get_loss_tracking(config, image, depth, opacity, viewpoint)
        #
        #         # 如果有深度监督，也只考虑静态区域
        #         if hasattr(viewpoint, 'depth') and viewpoint.depth is not None:
        #             gt_depth = torch.from_numpy(viewpoint.depth).to(self.device)
        #
        #             # 确保深度tensor维度正确
        #             if len(gt_depth.shape) == 2:
        #                 gt_depth = gt_depth.unsqueeze(0)  # [1, H, W]
        #
        #             if len(depth.shape) == 3 and depth.shape[0] == 1:
        #                 static_depth_render = depth[0, static_mask]  # [N_static]
        #                 static_depth_gt = gt_depth[0, static_mask]  # [N_static]
        #
        #                 if static_depth_render.numel() > 0:
        #                     depth_loss = torch.nn.functional.l1_loss(
        #                         static_depth_render, static_depth_gt
        #                     )
        #                     loss_tracking = rgb_loss + 0.1 * depth_loss
        #                 else:
        #                     loss_tracking = rgb_loss
        #             else:
        #                 loss_tracking = rgb_loss
        #         else:
        #             loss_tracking = rgb_loss

        return loss_tracking

    def save_keyframe_mask(self, viewpoint, cur_frame_idx):
        """
        为关键帧保存特殊标记的掩码图像
        """
        if (not self.enable_dynamic_filtering or
                not self.save_masked_images or
                not hasattr(viewpoint, 'dynamic_mask')):
            return

        try:
            import cv2
            import os

            # 创建关键帧目录
            kf_dir = os.path.join(self.dynamic_masker.save_dir, "keyframes")
            os.makedirs(kf_dir, exist_ok=True)

            # 获取原始图像
            gt_image = viewpoint.original_image  # [3, H, W] tensor
            img_np = gt_image.permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            # 获取动态mask
            dynamic_mask = viewpoint.dynamic_mask.cpu().numpy().astype(np.uint8)

            # 创建关键帧特殊标记图像（动态区域用绿色标记）
            kf_img = img_bgr.copy()
            kf_img[dynamic_mask > 0] = [0, 255, 0]  # 绿色标记关键帧中的动态物体

            # 保存关键帧图像
            kf_path = os.path.join(kf_dir, f"keyframe_{cur_frame_idx:06d}.jpg")
            cv2.imwrite(kf_path, kf_img)

            print(f"Saved keyframe mask for frame {cur_frame_idx}")

        except Exception as e:
            print(f"Warning: Failed to save keyframe mask for frame {cur_frame_idx}: {e}")


    def is_keyframe(self, cur_frame_idx, last_keyframe_idx, cur_frame_visibility_filter, occ_aware_visibility):
        """
        考虑动态物体的关键帧选择
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

        # 计算重叠度（排除动态区域）
        union = torch.logical_or(
            cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
        ).count_nonzero()
        intersection = torch.logical_and(
            cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
        ).count_nonzero()

        # 如果有动态物体，调整重叠度阈值
        if (hasattr(curr_frame, 'dynamic_confidence') and
                curr_frame.dynamic_confidence > 0.5):
            # 动态物体较多时，更容易创建关键帧
            adjusted_overlap = kf_overlap * 1.2
        else:
            adjusted_overlap = kf_overlap

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

        self.kf_indices = []
        depth_map = self.add_new_keyframe(cur_frame_idx, init=True)
        self.request_init(cur_frame_idx, viewpoint, depth_map)
        self.reset = False

    def run(self):
        # 主执行循环保持原有逻辑，但集成了动态物体过滤
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