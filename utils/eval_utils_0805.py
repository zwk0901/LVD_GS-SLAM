import json
import os

os.environ['MPLBACKEND'] = 'Agg'

import matplotlib
import cv2
import evo
import numpy as np
import torch
from PIL import Image
from evo.core import metrics, trajectory
from evo.core.metrics import PoseRelation, Unit
from evo.core.trajectory import PosePath3D, PoseTrajectory3D
from evo.tools import plot
from evo.tools.plot import PlotMode
from evo.tools.settings import SETTINGS

matplotlib.use('Agg')
print("Current backend:", matplotlib.get_backend())
from matplotlib import pyplot as plt
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

import wandb
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.image_utils import psnr
from gaussian_splatting.utils.loss_utils import ssim
from gaussian_splatting.utils.system_utils import mkdir_p
from utils.logging_utils import Log


def evaluate_evo(poses_gt, poses_est, plot_dir, label, monocular=False):
    ## Plot
    traj_ref = PosePath3D(poses_se3=poses_gt)
    traj_est = PosePath3D(poses_se3=poses_est)
    traj_est_aligned = trajectory.align_trajectory(
        traj_est, traj_ref, correct_scale=monocular
    )
    ## RMSE
    pose_relation = metrics.PoseRelation.translation_part
    data = (traj_ref, traj_est_aligned)
    ape_metric = metrics.APE(pose_relation)
    ape_metric.process_data(data)
    ape_stat = ape_metric.get_statistic(metrics.StatisticsType.rmse)
    ape_stats = ape_metric.get_all_statistics()
    Log("RMSE ATE \[m]", ape_stat, tag="Eval")

    with open(
            os.path.join(plot_dir, "stats_{}.json".format(str(label))),
            "w",
            encoding="utf-8",
    ) as f:
        json.dump(ape_stats, f, indent=4)

    plot_mode = evo.tools.plot.PlotMode.xy
    fig = plt.figure()
    ax = evo.tools.plot.prepare_axis(fig, plot_mode)
    ax.set_title(f"ATE RMSE: {ape_stat}")
    evo.tools.plot.traj(ax, plot_mode, traj_ref, "--", "gray", "gt")
    evo.tools.plot.traj_colormap(
        ax,
        traj_est_aligned,
        ape_metric.error,
        plot_mode,
        min_map=ape_stats["min"],
        max_map=ape_stats["max"],
    )
    ax.legend()
    plt.savefig(os.path.join(plot_dir, "evo_2dplot_{}.png".format(str(label))))
    plt.close(fig)

    return ape_stat


def eval_ate(frames, kf_ids, save_dir, iterations, final=False, monocular=False, BA=False):
    trj_data = dict()
    latest_frame_idx = kf_ids[-1] + 2 if final else kf_ids[-1] + 1
    # latest_frame_idx = len(frames) if final else kf_ids[-1] + 1
    trj_id, trj_est, trj_gt = [], [], []
    trj_est_np, trj_gt_np = [], []

    def gen_pose_matrix(R, T):
        pose = np.eye(4)
        pose[0:3, 0:3] = R.cpu().numpy()
        pose[0:3, 3] = T.cpu().numpy()
        return pose

    for kf_id in kf_ids:
        # for kf_id in range(latest_frame_idx):
        # print(kf_id)
        kf = frames[kf_id]
        pose_est = np.linalg.inv(gen_pose_matrix(kf.R, kf.T))
        pose_gt = np.linalg.inv(gen_pose_matrix(kf.R_gt, kf.T_gt))

        trj_id.append(frames[kf_id].uid)
        trj_est.append(pose_est.tolist())
        trj_gt.append(pose_gt.tolist())

        trj_est_np.append(pose_est)
        trj_gt_np.append(pose_gt)

    trj_data["trj_id"] = trj_id
    trj_data["trj_est"] = trj_est
    trj_data["trj_gt"] = trj_gt

    plot_dir = os.path.join(save_dir, "plot")
    mkdir_p(plot_dir)

    if BA:
        label_evo = "after BA"
    elif final:
        label_evo = "final"
    else:
        label_evo = "{:04}".format(iterations)
    # label_evo = "final" if final else "{:04}".format(iterations)
    with open(
            os.path.join(plot_dir, f"trj_{label_evo}.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(trj_data, f, indent=4)

    ate = evaluate_evo(
        poses_gt=trj_gt_np,
        poses_est=trj_est_np,
        plot_dir=plot_dir,
        label=label_evo,
        monocular=monocular,
    )
    wandb.log({"frame_idx": latest_frame_idx, "ate": ate})
    return ate


def eval_rendering(
        frames,
        gaussians,
        dataset,
        save_dir,
        pipe,
        background,
        datatype,
        kf_indices,
        iteration="final",
):
    interval = 1
    img_pred, img_gt, saved_frame_idx, img_residual = [], [], [], []
    end_idx = len(frames) - 1 if iteration == "final" or "before_opt" else iteration
    psnr_array, ssim_array, lpips_array = [], [], []
    cal_lpips = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", normalize=True
    ).to("cuda")
    # Define directory to save images (uncomment if needed)
    viz_dir = os.path.join(save_dir, "viz")
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)
    render_dir = os.path.join(save_dir, "render_rgb")
    if not os.path.exists(render_dir):
        os.makedirs(render_dir)
    depth_dir = os.path.join(save_dir, "render_depth")
    if not os.path.exists(depth_dir):
        os.makedirs(depth_dir)
    depth_dir1 = os.path.join(save_dir, "render_depth_npy")
    if not os.path.exists(depth_dir1):
        os.makedirs(depth_dir1)

    for idx in range(0, end_idx, interval):
        if idx in kf_indices:
            continue
        saved_frame_idx.append(idx)
        frame = frames[idx]
        gt_image, _, _, _ = dataset[idx]

        render_pkg = render(frame, gaussians, pipe, background)
        rendering = render_pkg["render"]

        # Save depth map
        depth = render_pkg["depth"]
        depth = depth.squeeze()
        depth_np = depth.detach().cpu().numpy()
        depth_max = depth_np.max()
        depth_min = depth_np.min()
        depth_nor = (depth_np - depth_min) / (depth_max - depth_min)
        depth_nor = (depth_nor * 255).astype(np.uint8)

        img_depth = Image.fromarray(depth_nor)
        # save_path = os.path.join(depth_dir, f"{idx}_pred.png")
        # img_depth.save(save_path, dpi=(300, 300))

        # Save depth map as .npy file (unnormalized depth matrix)
        # save_path_npy = os.path.join(depth_dir1, f"{idx}_pred.npy")
        # np.save(save_path_npy, depth_np)

        image = torch.clamp(rendering, 0.0, 1.0)
        # Calculate metrics
        mask = gt_image > 0

        psnr_score = psnr((image[mask]).unsqueeze(0), (gt_image[mask]).unsqueeze(0))
        ssim_score = ssim((image).unsqueeze(0), (gt_image).unsqueeze(0))
        lpips_score = cal_lpips((image).unsqueeze(0), (gt_image).unsqueeze(0))

        psnr_array.append(psnr_score.item())
        ssim_array.append(ssim_score.item())
        lpips_array.append(lpips_score.item())

        gt = (gt_image.cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
        pred = (image.detach().cpu().numpy().transpose((1, 2, 0)) * 255).astype(
            np.uint8
        )
        residual = np.abs(pred.astype(np.float32) - gt.astype(np.float32))
        residual = np.clip(residual, 0, 255).astype(np.uint8)
        img_pred.append(pred)
        img_gt.append(gt)
        img_residual.append(residual)
        # Render comparison image
        plt.figure(figsize=(10, 10))

        plt.subplot(2, 2, 1)
        plt.imshow(gt)
        plt.title('Ground Truth')
        plt.axis('off')

        plt.subplot(2, 2, 2)
        plt.imshow(pred)
        plt.title('Rendered rgb')
        plt.axis('off')

        plt.subplot(2, 2, 3)
        plt.imshow(img_depth, cmap='gray')
        plt.title('Depth Map')
        plt.axis('off')

        plt.subplot(2, 2, 4)
        plt.imshow(residual)
        plt.title('Residual')
        plt.axis('off')

        plt.figtext(0.5, 0.01, f"PSNR: {psnr_score.item():.2f}", ha="center", fontsize=14)

        save_path = os.path.join(viz_dir, f"{idx}.png")
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

        # Save rendered image
        pred = (image.detach().cpu().numpy().transpose((1, 2, 0)) * 255).astype(
            np.uint8
        )
        pred_image = Image.fromarray(pred)
        save_path = os.path.join(render_dir, f"{idx}_pred.png")
        # pred_image.save(save_path, dpi=(300, 300))

    output = dict()
    output["mean_psnr"] = float(np.mean(psnr_array))
    output["mean_ssim"] = float(np.mean(ssim_array))
    output["mean_lpips"] = float(np.mean(lpips_array))

    Log(
        f'mean psnr: {output["mean_psnr"]}, ssim: {output["mean_ssim"]}, lpips: {output["mean_lpips"]}',
        tag="Eval",
    )

    psnr_save_dir = os.path.join(save_dir, "psnr", str(iteration))
    mkdir_p(psnr_save_dir)

    json.dump(
        output,
        open(os.path.join(psnr_save_dir, "final_result.json"), "w", encoding="utf-8"),
        indent=4,
    )
    return output


def save_gaussians(gaussians, name, iteration, final=False):
    if name is None:
        return
    if final:
        point_cloud_path = os.path.join(name, "point_cloud/final")
    else:
        point_cloud_path = os.path.join(
            name, "point_cloud/iteration_{}".format(str(iteration))
        )
    gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))