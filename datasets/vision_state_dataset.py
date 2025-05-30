import torch
import os
from typing import Dict, List

from torch.utils.data import Dataset
import imageio
import numpy as np
import random
from path import Path
import pandas as pd
from PIL import ImageFile, Image
from datasets.utils import save_metric, load_metrics, RGBtoD

ImageFile.LOAD_TRUNCATED_IMAGES = True

def read_labels(label_file):
    """
    Read the txt file containing the robot state and reshape it to a meaningful vector to pair with the
    video frames.
    Parameters
    ----------
    root : str
        The root directory where the label files are
    name: str
        The name of the label file with the .txt extension
    Returns
    -------
    robot_state : ndarray
        A matrix with the 54 dimensional robot state vector for every frame in the video. The information is as follows:
        | 0 -> Time (t)
        | 1 to 6 -> Force sensor reading (fx, fy, fz, tx, ty, tz)
        | 7 to 19 -> Dvrk estimation task position, orientation, linear and angular velocities (px, py, pz, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz)
        | 20 to 26 -> Joint angles (q1, q2, q3, q4, q5, q6, q7)
        | 27 to 33 -> Joint velocities (vq1, vq2, vq3, vq4, vq5, vq6, vq7)
        | 44 to 40 -> Joint torque (tq1, tq2, tq3, tq4, tq5, tq6, tq7)
        | 41 to 47 -> Desired joint angle (q1d, q2d, q3d, q4d, q5d, q6d, q7d)
        | 48 to 54 -> Desired joint torque (tq1d, tq2d, tq3d, tq4d, tq5d, tq6d, tq7d)
        | 55 to 57 -> Estimated end effector force (psm_fx, psm_fy, psm_fz)
    """

    with open(label_file, "r") as file_object:
        lines = file_object.read().splitlines()
        robot_state = []
        for line in lines:
            row = []
            splitted = line.split(",")
            _ = [row.append(float(f)) for f in splitted]
            robot_state.append(row)

    robot_state = np.array(robot_state).astype(np.float32)
    state = robot_state[:, 1:55]
    force = robot_state[:, 55:58]

    return state, force

def dafoes_2_dvrk(labels):
    new_labels = np.zeros((labels.shape[0], 54))
    # Robot position and orientation
    new_labels[:, 6:13] = labels[:, :7]
    new_labels[:, 19:25] = labels[:, 7:13]
    new_labels[:, 40:46] = labels[:, 20:26]
    new_labels[:, 47:54] = labels[:, 13:20]

    return new_labels


def load_as_float(path: Path) -> np.ndarray:
    return  imageio.imread(path)[:,:, :3].astype(np.float32)

def load_depth(path: Path) -> np.ndarray:
    return np.array(Image.open(path)).astype(np.uint16).astype(np.float32)

def process_depth(rgb_depth: torch.Tensor) -> torch.Tensor:
    depth = torch.zeros((1, rgb_depth.shape[1], rgb_depth.shape[2]))
    for i in range(rgb_depth.shape[1]):
        for j in range(rgb_depth.shape[2]):
            pixel = RGBtoD(rgb_depth[0, i, j].item(), rgb_depth[1, i, j].item(), rgb_depth[2, i, j].item())
            depth[:, i, j].item = pixel
    
    return (depth.float() - depth.mean()) / depth.std()

class VisionStateDataset(Dataset):
    """A dataset to load data from different folders that are arranged this way:
        root/scene_1/000.png
        root/scene_1/001.png
        ...
        root/scene_1/labels.csv
        root/scene_2/000.png
        .
        transform functions takes in a list images and a numpy array representing the intrinsics of the camera and the robot state
    """

    def __init__(self, recurrency_size=5, load_depths=True, max_depth=25., mode="train", transform=None, seed=0, train_type="random",
                 occlude_param=None, dataset="dafoes"):
        
        assert dataset in ["dafoes", "dvrk", "mixed"], "The only available datasets are dafoes, dvrk or mixed"
        assert mode in ["train", "val", "test"], "There is only 3 modes for the dataset: train, validation or test"

        print(f"\nInitializing VisionStateDataset:")
        print(f"- Dataset type: {dataset}")
        print(f"- Mode: {mode}")
        print(f"- Recurrency size: {recurrency_size}")
        print(f"- Load depths: {load_depths}")
        print(f"- Train type: {train_type}")
        print(f"- Occlusion param: {occlude_param}")

        np.random.seed(seed)
        random.seed(seed)

        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        root = Path(root)
        self.dataset = dataset

        if dataset == "dafoes":
            data_root_dafoes = root/"visu_depth_haptic_data"
            self.data_root_dafoes = data_root_dafoes
            print(f"- DaFoEs data root: {data_root_dafoes}")
        elif dataset == "dvrk":
            data_root_dvrk = root/"experiment_data"
            self.data_root_dvrk = data_root_dvrk
            print(f"- DVRK data root: {data_root_dvrk}")
        else:
            data_root_dafoes = root/"visu_depth_haptic_data"
            data_root_dvrk = root/"experiment_data"
            self.data_root_dafoes = data_root_dafoes
            self.data_root_dvrk = data_root_dvrk
            print(f"- DaFoEs data root: {data_root_dafoes}")
            print(f"- DVRK data root: {data_root_dvrk}")

        self.occlusion = {"force_sensor": [0, 6],
                          "robot_p": [6, 9],
                          "robot_o": [9, 13],
                          "robot_v": [13, 16],
                          "robot_w": [16, 19],
                          "robot_q": [19, 26],
                          "robot_vq": [26, 33],
                          "robot_tq": [33, 40],
                          "robot_qd": [40, 47],
                          "robot_tqd": [47, 54]
                        }
        
        # Load scene lists and validate paths
        if dataset == "dafoes":
            scene_list_path = self.data_root_dafoes/"{}.txt".format(mode) if train_type=="random" else self.data_root_dafoes/"{}_{}.txt".format(mode, train_type)
            assert scene_list_path.exists(), f"Scene list not found: {scene_list_path}"
            self.scenes = [self.data_root_dafoes/folder[:-1] for folder in open(scene_list_path)][:-1]
            print(f"- Found {len(self.scenes)} DaFoEs scenes")
        elif dataset == "dvrk":
            scene_list_path = self.data_root_dvrk/"{}.txt".format(mode)
            assert scene_list_path.exists(), f"Scene list not found: {scene_list_path}"
            self.folder_index = [folder.split('_')[-1].rstrip('\n') for folder in open(scene_list_path)]
            print(f"- Found {len(self.folder_index)} DVRK folders")
        else:
            # Mixed dataset
            dafoes_list = self.data_root_dafoes/"{}.txt".format(mode) if train_type=="random" else self.data_root_dafoes/"{}_{}.txt".format(mode, train_type)
            dvrk_list = self.data_root_dvrk/"{}.txt".format(mode)
            assert dafoes_list.exists(), f"DaFoEs scene list not found: {dafoes_list}"
            assert dvrk_list.exists(), f"DVRK scene list not found: {dvrk_list}"
            self.scenes = [self.data_root_dafoes/folder[:-1] for folder in open(dafoes_list)][:-1]
            self.folder_index = [folder.split('_')[-1].rstrip('\n') for folder in open(dvrk_list)]
            print(f"- Found {len(self.scenes)} DaFoEs scenes and {len(self.folder_index)} DVRK folders")

        self.transform = transform
        self.mode = mode
        self.load_depths = load_depths
        self.max_depth = max_depth
        self.recurrency_size = recurrency_size
        self.occlude_param = occlude_param
        
        print("\nCrawling folders to build dataset...")
        self.crawl_folders()
        print(f"Dataset initialization complete. Total samples: {len(self)}\n")
        
    def crawl_folders(self):
        samples = []
        
        if self.dataset == "dafoes":
            samples = self.load_dafoes(samples)
            print(f"Loaded {len(samples)} DaFoEs samples")
        
        elif self.dataset == "dvrk":
            samples = self.load_dvrk(samples)
            print(f"Loaded {len(samples)} DVRK samples")
        
        else:
            samples = self.load_dafoes(samples)
            dafoes_count = len(samples)
            samples = self.load_dvrk(samples)
            print(f"Loaded {dafoes_count} DaFoEs samples and {len(samples) - dafoes_count} DVRK samples")

        if self.mode in ["train", "val"]:
            random.shuffle(samples)
            
        self.samples = samples
    
    def load_dafoes(self, samples):
        mean_labels, std_labels = [], []
        mean_forces, std_forces = [], []

        print("\nLoading DaFoEs data...")
        for scene in self.scenes:
            print(f"Processing scene: {scene}")
            labels = np.array(pd.read_csv(scene/'labels.csv')).astype(np.float32)
            scene_rgb = scene/"RGB_frames"
            if self.load_depths:
                scene_depth = scene/"Depth_frames"
                depth_maps = sorted(scene_depth.files("*.png"))
                print(f"- Found {len(depth_maps)} depth maps")

            # Validate and log data shapes
            print(f"- Labels shape: {labels.shape}")
            mean_labels.append(labels[:, :26].mean(axis=0))
            std_labels.append(labels[:, :26].std(axis=0))
            mean_forces.append((labels[:, 26:29]).mean(axis=0))
            std_forces.append((labels[:, 26:29]).std(axis=0))

            images = sorted(scene_rgb.files("*.png"))
            print(f"- Found {len(images)} RGB images")
            
            n_labels = len(labels) // len(images)
            step = 7

            for i in range(len(images)):
                if i < 20: continue
                if i + self.recurrency_size > len(images) - 20: break
                sample = {}
                sample['dataset'] = "dafoes"
                sample['img'] = [im for im in images[i:i+self.recurrency_size]]
                if self.load_depths:
                    sample['depth'] = [depth for depth in depth_maps[i:i+self.recurrency_size]]

                sample['label'] = [np.mean(labels[n_labels*i+a: (n_labels*i+a) + step, :26], axis=0) for a in range(self.recurrency_size)]
                sample['force'] = np.mean(labels[n_labels*i+(self.recurrency_size-1):(n_labels*i+(self.recurrency_size-1)) + step, 26:29], axis=0)
                samples.append(sample)

                # Log sample shapes periodically
                if len(samples) % 1000 == 0:
                    print(f"\nSample {len(samples)} shapes:")
                    print(f"- Image sequence length: {len(sample['img'])}")
                    print(f"- Label sequence shape: {np.array(sample['label']).shape}")
                    print(f"- Force target shape: {sample['force'].shape}")
                    if self.load_depths:
                        print(f"- Depth sequence length: {len(sample['depth'])}")
        
        if self.mode == "train":
            print("\nComputing and saving normalization metrics...")
            self.mean_labels = np.mean(mean_labels, axis=0)
            self.std_labels = np.mean(std_labels, axis=0)
            self.mean_forces = np.mean(mean_forces, axis=0)
            self.std_forces = np.mean(std_forces, axis=0)

            print(f"- Label means range: [{self.mean_labels.min():.3f}, {self.mean_labels.max():.3f}]")
            print(f"- Label stds range: [{self.std_labels.min():.3f}, {self.std_labels.max():.3f}]")
            print(f"- Force means: {self.mean_forces}")
            print(f"- Force stds: {self.std_forces}")

            save_metric('labels_mean.npy', self.mean_labels)
            save_metric('labels_std.npy', self.std_labels)
            save_metric('forces_mean.npy', self.mean_forces)
            save_metric('forces_std.npy', self.std_forces)
        else:
            print("\nLoading pre-computed normalization metrics...")
            self.mean_labels, self.std_labels, self.mean_forces, self.std_forces = load_metrics("dafoes")

        return samples
    
    def load_dvrk(self, samples):
        mean_labels, std_labels = [], []
        mean_forces, std_forces = [], []

        print("\nLoading DVRK data...")
        for index in self.folder_index:
            print(f"Processing folder {index}")
            labels, forces = read_labels(self.data_root_dvrk/'labels_{}.txt'.format(index))
            scene = self.data_root_dvrk/"imageset_{}".format(index)

            # Log data shapes
            print(f"- Labels shape: {labels.shape}")
            print(f"- Forces shape: {forces.shape}")

            mean_labels.append(labels.mean(axis=0))
            std_labels.append(labels.std(axis=0))
            mean_forces.append(forces.mean(axis=0))
            std_forces.append(forces.std(axis=0))

            images = sorted(scene.files("*.jpg"))
            print(f"- Found {len(images)} images")
            
            labels = labels.reshape(len(images), -1)
            forces = forces.reshape(len(images), -1)

            for i in range(len(images)):
                if i < 80: continue
                if i > len(images) - (25 + self.recurrency_size): break
                sample = {}
                sample['dataset'] = "dvrk"
                sample['img'] = [scene/'img_{}.jpg'.format(i+a) for a in range(self.recurrency_size)]
                sample['label'] = [labels[i+a] for a in range(self.recurrency_size)]
                sample['force'] = forces[i+self.recurrency_size-1]
                samples.append(sample)

                # Log sample shapes periodically
                if len(samples) % 1000 == 0:
                    print(f"\nSample {len(samples)} shapes:")
                    print(f"- Image sequence length: {len(sample['img'])}")
                    print(f"- Label sequence shape: {np.array(sample['label']).shape}")
                    print(f"- Force target shape: {sample['force'].shape}")

        if self.mode == "train":
            print("\nComputing normalization metrics...")
            print(f"- Label means range: [{np.mean(mean_labels).min():.3f}, {np.mean(mean_labels).max():.3f}]")
            print(f"- Label stds range: [{np.mean(std_labels).min():.3f}, {np.mean(std_labels).max():.3f}]")
            print(f"- Force means range: [{np.mean(mean_forces).min():.3f}, {np.mean(mean_forces).max():.3f}]")
            print(f"- Force stds range: [{np.mean(std_forces).min():.3f}, {np.mean(std_forces).max():.3f}]")

        return samples

    def __getitem__(self, index: int) -> Dict[str, List[torch.Tensor]]:
        """Get a sequence of frames with corresponding labels."""
        sample = self.samples[index]
        
        # Load and transform images
        imgs = []
        for img_path in sample['img']:
            img = imageio.imread(img_path)[:,:,:3].astype(np.float32)
            if self.transform is not None:
                img = self.transform(img)
            imgs.append(img)
        
        # Load depths if required
        depths = None
        if self.load_depths and 'depth' in sample:
            depths = []
            for depth_path in sample['depth']:
                depth = load_depth(depth_path)
                if self.transform is not None:
                    depth = self.transform(depth)
                depths.append(depth)
            depths = torch.stack(depths)
        
        # Convert labels and forces to tensors
        labels = torch.FloatTensor(sample['label'])
        force = torch.FloatTensor(sample['force'])
        
        # Log shapes periodically
        if index % 1000 == 0:
            print(f"\nItem {index} shapes:")
            print(f"- Images: {torch.stack(imgs).shape}")
            print(f"- Labels: {labels.shape}")
            print(f"- Force: {force.shape}")
            if depths is not None:
                print(f"- Depths: {depths.shape}")
        
        return {
            'images': torch.stack(imgs),
            'depths': depths,
            'labels': labels,
            'force': force,
            'dataset': sample['dataset']
        }

    def __len__(self):
        return len(self.samples)
