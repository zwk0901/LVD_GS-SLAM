import csv
import glob
import os

import cv2
import numpy as np
import torch
import trimesh
from PIL import Image
import json
from pathlib import Path

from gaussian_splatting.utils.graphics_utils import focal2fov
from scipy.spatial.transform import Rotation as R

try:
    import pyrealsense2 as rs
except Exception:
    pass

# We retain the input interfaces for ground-truth depth and other monocular depth estimations (e.g., from DepthAnything).
# In RGB-only scenarios, the first channel of the RGB image is used as a placeholder for depth input.

## ====================================data parser========================================
class dl3dvParser:
    def __init__(self, input_folder, config):
        self.input_folder = input_folder
        self.begin = config["Dataset"]["begin"]
        self.end = config["Dataset"]["end"]
        
        self.color_paths = sorted(glob.glob(f"{self.input_folder}/rgb/*.png"))[self.begin:self.end]
        self.depth_paths = sorted(glob.glob(f"{self.input_folder}/rgb/*.png"))[self.begin:self.end]
        self.mono_depth_paths = sorted(glob.glob(f"{self.input_folder}/rgb/*.png"))[self.begin:self.end]
        self.n_img = len(self.color_paths)
        
        self.load_poses(os.path.join(self.input_folder, "cameras.json"))

    def load_poses(self, pose_file):
        """ Read camera poses from camera.json and convert them to 4×4 matrices """
        self.poses = []
        self.frames = []

        with open(pose_file, "r") as f:
            all_poses = json.load(f)

        selected_poses = all_poses[self.begin:self.end]
        init_trans = np.array(selected_poses[0]["cam_trans"])

        for i, pose in enumerate(selected_poses):
            qx, qy, qz, qw = pose["cam_quat"]
            tx, ty, tz = pose["cam_trans"]

            rotation_matrix = R.from_quat([qx, qy, qz, qw]).as_matrix()
            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] = rotation_matrix
            transform_matrix[:3, 3] = [tx, ty, tz] - init_trans 
    
            inv_pose = np.linalg.inv(transform_matrix)
            self.poses.append(inv_pose)  
            frame = {
                "file_path": self.color_paths[i],
                "depth_path": self.color_paths[i],
                "mono_depth_path": self.color_paths[i],
                "transform_matrix": transform_matrix.tolist(),  
            }
            self.frames.append(frame)

class KITTIParser:
    def __init__(self, input_folder, config):
        self.input_folder = input_folder
        self.begin = config["Dataset"]["begin"]
        self.end = config["Dataset"]["end"] 
        self.color_paths = sorted(glob.glob(f"{self.input_folder}/image_2/*.jpg"))[self.begin:self.end]
        self.depth_paths = sorted(glob.glob(f"{self.input_folder}/image_2/*.jpg"))[self.begin:self.end]
        self.mono_depth_paths = sorted(glob.glob(f"{self.input_folder}/image_2/*.jpg"))[self.begin:self.end]
        self.n_img = len(self.color_paths)
        self.load_poses(f"{self.input_folder}gt/*.txt")

    def load_poses(self, path):
        self.poses = []
        self.frames = []
        pose_files = sorted(glob.glob(path))[self.begin:self.end]
        print(pose_files)
        arr = np.loadtxt(pose_files[0], delimiter=' ')
        if arr.size == 12:
            pose = arr.reshape(3, 4)
            pose_homo = np.eye(4)
            pose_homo[:3, :] = pose
            init_trans = pose_homo[:3, 3]

        for i in range(self.n_img):
            arr = np.loadtxt(pose_files[i], delimiter=' ')
            if arr.size != 12:
                raise ValueError(f"{pose_files[i]} 不是12个数，实际为{arr.size}个数")
            pose = arr.reshape(3, 4)
            pose_homo = np.eye(4)
            pose_homo[:3, :] = pose
            pose_homo[:3, 3] -= init_trans
            inv_pose = np.linalg.inv(pose_homo)
            self.poses.append(inv_pose)
            frame = {
                "file_path": self.color_paths[i],
                "depth_path": self.depth_paths[i],
                "mono_depth_path": self.mono_depth_paths[i],
                "transform_matrix": pose_homo.tolist(),
            }
            self.frames.append(frame)

class WaymoParser:
    def __init__(self, input_folder):
        self.input_folder = input_folder
        self.color_paths = sorted(glob.glob(f"{self.input_folder}/rgb/*.png"))
        self.depth_paths = sorted(glob.glob(f"{self.input_folder}/depth/*.png"))
        self.mono_depth_paths = sorted(glob.glob(f"{self.input_folder}/mono_depth/*.png"))
        self.n_img = len(self.color_paths)
        self.load_poses(f"{self.input_folder}/gt/*.txt")

    def load_poses(self, path):
        self.poses = []
        self.frames = []
        pose_files = sorted(glob.glob(path))

        for i in range(self.n_img):
            pose = np.loadtxt(pose_files[i], delimiter=' ').reshape(4, 4)
            inv_pose = np.linalg.inv(pose)  
            self.poses.append(inv_pose)     
            frame = {
                "file_path": self.color_paths[i],
                "depth_path": self.depth_paths[i],
                "mono_depth_path": self.mono_depth_paths[i],
                "transform_matrix": pose.tolist(),      
            }
            self.frames.append(frame)

class ReplicaParser:
    def __init__(self, input_folder):
        self.input_folder = input_folder
        self.color_paths = sorted(glob.glob(f"{self.input_folder}/results/frame*.png"))
        self.depth_paths = sorted(glob.glob(f"{self.input_folder}/results/depth*.png"))
        self.mono_depth_paths = sorted(glob.glob(f"{self.input_folder}/results/mono*.png"))
        self.n_img = len(self.color_paths)
        self.load_poses(f"{self.input_folder}traj.txt")

    def load_poses(self, path):
        self.poses = []
        with open(path, "r") as f:
            lines = f.readlines()

        frames = []
        for i in range(self.n_img):
            line = lines[i]
            pose = np.array(list(map(float, line.split()))).reshape(4, 4)
            pose = np.linalg.inv(pose)
            self.poses.append(pose)
            frame = {
                "file_path": self.color_paths[i],
                "depth_path": self.depth_paths[i],
                "mono_depth_path": self.mono_depth_paths[i],
                "transform_matrix": pose.tolist(),
            }

            frames.append(frame)
        self.frames = frames


class TUMParser:
    def __init__(self, input_folder):   
        self.input_folder = input_folder
        self.load_poses(self.input_folder, frame_rate=32)
        self.n_img = len(self.color_paths)

    def parse_list(self, filepath, skiprows=0):
        data = np.loadtxt(filepath, delimiter=" ", dtype=np.unicode_, skiprows=skiprows)
        return data

    def associate_frames(self, tstamp_image, tstamp_depth, tstamp_pose, max_dt=0.08):
        associations = []
        for i, t in enumerate(tstamp_image):       
            if tstamp_pose is None:
                j = np.argmin(np.abs(tstamp_depth - t))
                if np.abs(tstamp_depth[j] - t) < max_dt:
                    associations.append((i, j))

            else:
                j = np.argmin(np.abs(tstamp_depth - t))
                k = np.argmin(np.abs(tstamp_pose - t))

                if (np.abs(tstamp_depth[j] - t) < max_dt) and (
                    np.abs(tstamp_pose[k] - t) < max_dt
                ):
                    associations.append((i, j, k))

        return associations

    def load_poses(self, datapath, frame_rate=-1):
        if os.path.isfile(os.path.join(datapath, "groundtruth.txt")):
            pose_list = os.path.join(datapath, "groundtruth.txt")
        elif os.path.isfile(os.path.join(datapath, "pose.txt")):
            pose_list = os.path.join(datapath, "pose.txt")

        image_list = os.path.join(datapath, "rgb.txt")
        depth_list = os.path.join(datapath, "depth.txt")
        mono_depth_list = os.path.join(datapath, "mono_depth.txt")

        image_data = self.parse_list(image_list)
        depth_data = self.parse_list(depth_list)
        mono_depth_data = self.parse_list(mono_depth_list)
        pose_data = self.parse_list(pose_list, skiprows=1)
        pose_vecs = pose_data[:, 0:].astype(np.float64)

        tstamp_image = image_data[:, 0].astype(np.float64)
        tstamp_depth = depth_data[:, 0].astype(np.float64)
        tstamp_pose = pose_data[:, 0].astype(np.float64)
        associations = self.associate_frames(tstamp_image, tstamp_depth, tstamp_pose)
        print("标号:", tstamp_image[471])

        indicies = [0]
        for i in range(1, len(associations)):
            t0 = tstamp_image[associations[indicies[-1]][0]]
            t1 = tstamp_image[associations[i][0]]
            if t1 - t0 > 1.0 / frame_rate:
                indicies += [i]

        self.color_paths, self.poses, self.depth_paths, self.frames, self.mono_depth_paths = [], [], [], [], []

        for ix in indicies:
            (i, j, k) = associations[ix]
            self.color_paths += [os.path.join(datapath, image_data[i, 1])]
            self.depth_paths += [os.path.join(datapath, depth_data[j, 1])]
            self.mono_depth_paths += [os.path.join(datapath, mono_depth_data[i, 1])]

            quat = pose_vecs[k][4:]     
            trans = pose_vecs[k][1:4]   
            T = trimesh.transformations.quaternion_matrix(np.roll(quat, 1)) 
            T[:3, 3] = trans
            self.poses += [np.linalg.inv(T)]   

            frame = {
                "file_path": str(os.path.join(datapath, image_data[i, 1])),
                "depth_path": str(os.path.join(datapath, depth_data[j, 1])),
                "transform_matrix": (np.linalg.inv(T)).tolist(),
                "mono_depth_path": str(os.path.join(datapath, mono_depth_data[i, 1]))
            }

            self.frames.append(frame)

##=================================Define data base class==================================
class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, args, path, config):
        self.args = args
        self.path = path
        self.config = config
        self.device = "cuda:0"
        self.dtype = torch.float32
        self.num_imgs = 999999

    def __len__(self):
        return self.num_imgs

    def __getitem__(self, idx):
        pass

class MonocularDataset(BaseDataset):
    def __init__(self, args, path, config):     
        super().__init__(args, path, config)
        calibration = config["Dataset"]["Calibration"]
        # Camera prameters
        self.fx = calibration["fx"]
        self.fy = calibration["fy"]    
        self.cx = calibration["cx"]
        self.cy = calibration["cy"]   
        self.width = calibration["width"]
        self.height = calibration["height"]
        self.fovx = focal2fov(self.fx, self.width)
        self.fovy = focal2fov(self.fy, self.height)
        self.K = np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]
        )                               
        # distortion parameters
        self.disorted = calibration["distorted"] 
        self.dist_coeffs = np.array(
            [
                calibration["k1"],
                calibration["k2"],
                calibration["p1"],
                calibration["p2"],
                calibration["k3"],
            ]
        )
        self.map1x, self.map1y = cv2.initUndistortRectifyMap(  
            self.K,
            self.dist_coeffs,
            np.eye(3),
            self.K,
            (self.width, self.height),
            cv2.CV_32FC1,
        )
        # depth parameters
        self.has_depth = True if "depth_scale" in calibration.keys() else False
        self.depth_scale = calibration["depth_scale"] if self.has_depth else None

        # Default scene scale  
        nerf_normalization_radius = 5
        self.scene_info = {
            "nerf_normalization": {
                "radius": nerf_normalization_radius,
                "translation": np.zeros(3),
            },
        }

    def load_image(self, image_path):
        image = Image.open(image_path)
        image_array = np.array(image)

        # Check if the image is RGB (3 channels); if so, extract the first channel
        if len(image_array.shape) == 3:  
            return image_array[:, :, 0]  
        else:  
            return image_array  

    def __getitem__(self, idx):  
        color_path = self.color_paths[idx]
        pose = self.poses[idx]

        image = np.array(Image.open(color_path))
        depth = None

        if self.disorted:
            image = cv2.remap(image, self.map1x, self.map1y, cv2.INTER_LINEAR)  

        if self.has_depth:
            depth_path = self.depth_paths[idx]
            depth = self.load_image(depth_path) / self.depth_scale  
            mono_depth_path = self.mono_depth_paths[idx]
            mono_depth = self.load_image(mono_depth_path) / (self.depth_scale*5)

        image = (
            torch.from_numpy(image / 255.0)
            .clamp(0.0, 1.0)
            .permute(2, 0, 1)
            .to(device=self.device, dtype=self.dtype)
        )
        pose = torch.from_numpy(pose).to(device=self.device)
        return image, depth, pose, mono_depth

##=====================================# Define dataset class for specific dataset======================================
class dl3dvDataset(MonocularDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        dataset_path = config["Dataset"]["dataset_path"]
        
        parser = dl3dvParser(dataset_path, config)

        self.num_imgs = parser.n_img
        self.color_paths = parser.color_paths
        self.depth_paths = parser.color_paths  
        self.mono_depth_paths = parser.color_paths  
        self.poses = parser.poses  

class KITTIDataset(MonocularDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        dataset_path = config["Dataset"]["dataset_path"]
        parser = KITTIParser(dataset_path,config)
        self.num_imgs = parser.n_img
        self.color_paths = parser.color_paths
        self.depth_paths = parser.depth_paths
        self.mono_depth_paths = parser.mono_depth_paths
        self.poses = parser.poses      

class WaymoDataset(MonocularDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        dataset_path = config["Dataset"]["dataset_path"]
        parser = WaymoParser(dataset_path)
        self.num_imgs = parser.n_img
        self.color_paths = parser.color_paths
        self.depth_paths = parser.depth_paths
        self.mono_depth_paths = parser.mono_depth_paths
        self.poses = parser.poses       

class TUMDataset(MonocularDataset):  
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        dataset_path = config["Dataset"]["dataset_path"]
        parser = TUMParser(dataset_path)
        self.num_imgs = parser.n_img
        self.color_paths = parser.color_paths
        self.depth_paths = parser.depth_paths
        self.poses = parser.poses
        self.mono_depth_paths = parser.mono_depth_paths

class ReplicaDataset(MonocularDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        dataset_path = config["Dataset"]["dataset_path"]
        parser = ReplicaParser(dataset_path)
        self.num_imgs = parser.n_img
        self.color_paths = parser.color_paths
        self.depth_paths = parser.depth_paths
        self.mono_depth_paths = parser.mono_depth_paths
        self.poses = parser.poses

def load_dataset(args, path, config):
    if config["Dataset"]["type"] == "tum":
        return TUMDataset(args, path, config)
    elif config["Dataset"]["type"] == "replica":
        return ReplicaDataset(args, path, config)
    elif config["Dataset"]["type"] == "waymo":
        return WaymoDataset(args, path, config)
    elif config["Dataset"]["type"] == "KITTI":
        return KITTIDataset(args, path, config)
    elif config["Dataset"]["type"] == "dl3dv":
        return dl3dvDataset(args, path, config)
    else:
        raise ValueError("Unknown dataset type")
