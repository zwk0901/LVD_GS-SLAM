from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs

import mast3r.utils.path_to_dust3r
from dust3r.inference import inference
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

import os
import numpy as np
import time
import cv2
import torch
from scipy.spatial.transform import Rotation as R
import PIL.Image
from PIL.ImageOps import exif_transpose
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

from gaussian_splatting.gaussian_renderer import render_with_custom_resolution

import torchvision.transforms as tvf
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def _resize_pil_image(img, long_edge_size):
    S = max(img.size)
    if S > long_edge_size:
        interp = PIL.Image.LANCZOS
    elif S <= long_edge_size:
        interp = PIL.Image.BICUBIC
    new_size = tuple(int(round(x*long_edge_size/S)) for x in img.size)
    return img.resize(new_size, interp)

def torch_images_to_dust3r_format(tensor_images, size, square_ok=False, verbose=False):
    """
    Convert a list of torch tensor images to the format required by the DUSt3R/MASt3R model.
    
    Args:
    - tensor_images (list of torch.Tensor): List of RGB images in torch tensor format.
    - size (int): Target size for the images.
    - square_ok (bool): Whether square images are acceptable.
    - verbose (bool): Whether to print verbose messages.

    Returns:
    - list of dict: Converted images in the required format.
    """
    imgs = []
    for idx, image in enumerate(tensor_images):
        image = image.permute(1, 2, 0).cpu().numpy() * 255  # Convert to HWC format and scale to [0, 255]
        image = image.astype(np.uint8)
        
        img = PIL.Image.fromarray(image, 'RGB')
        img = exif_transpose(img).convert('RGB')
        W1, H1 = img.size
        if size == 224:
            img = _resize_pil_image(img, round(size * max(W1/H1, H1/W1)))
        else:
            img = _resize_pil_image(img, size)
        W, H = img.size
        cx, cy = W // 2, H // 2
        if size == 224:
            half = min(cx, cy)
            img = img.crop((cx - half, cy - half, cx + half, cy + half))
        else:
            halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
            if not square_ok and W == H:
                halfh = 3 * halfw // 4
            img = img.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))

        W2, H2 = img.size
        imgs.append(dict(img=ImgNorm(img)[None], true_shape=np.int32([img.size[::-1]]), idx=idx, instance=str(idx)))

    assert imgs, 'no images found'
    return imgs

def depth_to_3d(depth_map, K, dist_coeffs):
    """
    Convert a depth map to 3D points, taking camera distortion into account.

    Args:
        - depth_map: Depth image
        - K: Camera intrinsic matrix
        - dist_coeffs: Distortion coefficients [k1, k2, p1, p2, k3]

    Returns:
        - points_3d: 3D point cloud
    """

    if len(depth_map.shape) == 3:
        # If shape is (C, H, W), remove the channel dimension
        depth_map = depth_map.squeeze(0)  
    
    h, w = depth_map.shape

    # Generate pixel coordinate grid
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    pixels = np.stack((u, v), axis=-1).reshape(-1, 2).astype(np.float32)  

    # Undistort pixel coordinates
    undistorted_pixels = cv2.undistortPoints(pixels, K, dist_coeffs, P=K).reshape(h, w, 2)
    u_undistorted = undistorted_pixels[..., 0]
    v_undistorted = undistorted_pixels[..., 1]
    
    # Project undistorted pixels to 3D space using depth and camera intrinsics
    Z = depth_map
    X = (u_undistorted - K[0, 2]) * Z / K[0, 0]
    Y = (v_undistorted - K[1, 2]) * Z / K[1, 1]
    points_3d = np.stack((X, Y, Z), axis=-1)

    return points_3d

def depth_to_3d1(depth_map, K):
    h, w = depth_map.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    Z = depth_map
    X = (u - K[0, 2]) * Z / K[0, 0]
    Y = (v - K[1, 2]) * Z / K[1, 1]
    points_3d = np.stack((X, Y, Z), axis=-1)
    return points_3d

# Estimate relative pose and return rendered depth
def get_pose(img1, img2, model, dist_coeffs, viewpoint, gaussians, pipeline_params, background):
    device = 'cuda'
    schedule = 'cosine'
    lr = 0.01
    niter = 300
    
    # Extract features from images and perform point matching
    images = torch_images_to_dust3r_format([img1, img2], size=512)
    output = inference([tuple(images)], model, device, batch_size=1, verbose=False)
    view1, pred1 = output['view1'], output['pred1']
    view2, pred2 = output['view2'], output['pred2']
    desc1, desc2 = pred1['desc'].squeeze(0).detach(), pred2['desc'].squeeze(0).detach()    
    
    # find 2D-2D matches between the two images
    matches_im1, matches_im2 = fast_reciprocal_NNs(desc1, desc2, subsample_or_initxy1=8,
                                                   device=device, dist='dot', block_size=2**13)
    
    H1 = view1['img'].shape[2]     
    W1 = view1['img'].shape[3]
    scale_H = H1 / viewpoint.image_height
    scale_W = W1 / viewpoint.image_width
    
    render_pkg = render_with_custom_resolution(viewpoint, gaussians, pipeline_params, background, target_width=W1, target_height=H1)
    render_depth = render_pkg["depth"]

    # Adjust camera intrinsic matrix
    fx_new = viewpoint.fx * scale_W
    fy_new = viewpoint.fy * scale_H
    cx_new = viewpoint.cx * scale_W
    cy_new = viewpoint.cy * scale_H
    
    K_new = np.array([
        [fx_new, 0, cx_new],
        [0, fy_new, cy_new],
        [0, 0, 1]
    ])

    pts3d = depth_to_3d(render_depth.detach().cpu().numpy(), K_new, dist_coeffs=dist_coeffs)

    # Extract 3D points from image 1 and corresponding 2D points from image 2 for PnP
    objectPoints = pts3d[matches_im1[:, 1].astype(int), matches_im1[:, 0].astype(int), :]
    objectPoints = objectPoints.astype(np.float32)
    imagePoints = matches_im2.astype(np.float32)

    # Skip PnP if there are not enough points
    if len(objectPoints) < 6 or len(imagePoints) < 6:
        print("Warning: Not enough points to perform PnP estimation.")
        print("Number of points:", len(objectPoints))
        success = False
    else:
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints, imagePoints, K_new, dist_coeffs, iterationsCount=100, reprojectionError=5, flags=cv2.SOLVEPNP_SQPNP
        )
    
    if success:
        R, _ = cv2.Rodrigues(rvec)
        pose_w2c = np.eye(4)
        pose_w2c[:3, :3] = R
        pose_w2c[:3, 3] = tvec[:, 0]
        return pose_w2c, render_depth.detach().cpu().numpy()  
    else:
        print("PnP估计失败")
        pose_w2c = np.eye(4)
        return pose_w2c, render_depth.detach().cpu().numpy()  

# Extract depth from MASt3R
def get_depth(img1, img2, model):
    device = 'cuda'
    schedule = 'cosine'
    lr = 0.01
    niter = 300

    H1 = img1.shape[1]
    W1 = img1.shape[2]
    
    images = torch_images_to_dust3r_format([img1, img2], size=512)
    output = inference([tuple(images)], model, device, batch_size=1, verbose=False)
    view1, pred1 = output['view1'], output['pred1']
    view2, pred2 = output['view2'], output['pred2']
    
    pts1 = pred1['pts3d'].squeeze(0)
    z1 = pts1[...,2]
    z1 = z1.detach().cpu().numpy()
    z1_resized = cv2.resize(z1, (W1,H1), interpolation=cv2.INTER_NEAREST)
   
    return z1_resized
    
# Visualize comparison of rendered depth
def save_depth_comparison(render_depth, mono_depth, rgb, cur_frame_idx, save_dir):
    '''
    Inputs:
        - render_depth: Rendered depth map, (H, W) or (C, H, W) numpy array
        - mono_depth: Monocular depth estimation, (H, W) numpy array
        - rgb: RGB image, (C, H, W) torch tensor
        - cur_frame_idx: Index of the current frame
        - save_dir: Path to save the result
    '''
    os.makedirs(save_dir, exist_ok=True)
    
    if render_depth.ndim == 3:
        render_depth = render_depth.squeeze(0)
    
    rgb_image = rgb.permute(1, 2, 0).cpu().numpy()  
    H, W = render_depth.shape
    if rgb_image.shape[:2] != (H, W):
        rgb_image = cv2.resize(rgb_image, (W, H), interpolation=cv2.INTER_LINEAR)
    
    if mono_depth.shape != (H, W):
        mono_depth = cv2.resize(mono_depth, (W, H), interpolation=cv2.INTER_NEAREST)
    
    # Normalize depth maps
    render_depth_norm = (render_depth - render_depth.min()) / (render_depth.max() - render_depth.min())
    mono_depth_norm = (mono_depth - mono_depth.min()) / (mono_depth.max() - mono_depth.min())
    
    # Compute depth error
    depth_error = np.abs(render_depth - mono_depth)
    depth_error_norm = (depth_error - depth_error.min()) / (depth_error.max() - depth_error.min())
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Frame {cur_frame_idx}", fontsize=20, y=0.93)
    
    render0 = axes[0,0].imshow(render_depth_norm, cmap="viridis", vmin=0, vmax=1)
    axes[0,0].set_title("Rendered Depth", fontsize=15)
    axes[0,0].axis("off")
    
    axes[0,1].imshow(mono_depth_norm, cmap="viridis", vmin=0, vmax=1)
    axes[0,1].set_title("MASt3R Mono Depth", fontsize=15)
    axes[0,1].axis("off")
    
    # Add a shared colorbar between the two depth maps
    cbar = fig.colorbar(render0, ax=axes[0, :], orientation="horizontal", fraction=0.05, pad=0.1)
    cbar.set_label("Normalized Depth Value", fontsize=12)
    
    # Plot depth error map
    error = axes[1,0].imshow(depth_error_norm, cmap="magma", vmin=0, vmax=1)
    axes[1,0].set_title("Depth Error", fontsize=15)
    axes[1,0].axis("off")
    
    # Add a separate colorbar for the error map
    cbar_error = fig.colorbar(error, ax=axes[1, 0], orientation="horizontal", fraction=0.05, pad=0.1)
    cbar_error.set_label("Normalized Depth Error", fontsize=12)
    
    axes[1,1].imshow(rgb_image)
    axes[1,1].set_title("RGB", fontsize=15)
    axes[1,1].axis("off")
    
    save_path = os.path.join(save_dir, f"{cur_frame_idx}.png")
    plt.savefig(save_path)
    plt.close(fig)
    
    return save_path