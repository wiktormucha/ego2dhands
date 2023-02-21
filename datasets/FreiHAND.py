import numpy as np
import os
from PIL import Image
import json
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from utils.utils import vector_to_heatmaps, project_points_3D_to_2D
from config import *


class FreiHAND(Dataset):
    """
    Class to load FreiHAND dataset. This dataset class is using only training part.
    It devides the dataste into three subsets in proportion 80/15/5.
    Parameters:
        config - config file (device, data_dir, )
        set_type - type of data split (train/val/test)
        img_transform - transform and augumentation for input img
        heatmap_transform - transform and augumentation for input heatmap with keypoints
    Link to dataset:
    https://lmb.informatik.uni-freiburg.de/resources/datasets/FreihandDataset.en.html
    """

    def __init__(self, config, set_type="train", img_transform=None, heatmap_transform=None):

        self.device = config["device"]
        self.image_dir = os.path.join(config["data_dir"], "training/rgb")
        self.image_names = np.sort(os.listdir(self.image_dir))

        fn_K_matrix = os.path.join(config["data_dir"], "training_K.json")
        with open(fn_K_matrix, "r") as f:
            K_matrix_temp = np.array(json.load(f))

        fn_anno = os.path.join(config["data_dir"], "training_xyz.json")
        with open(fn_anno, "r") as f:
            anno_temp = np.array(json.load(f))

        self.K_matrix = np.concatenate(
            (K_matrix_temp, K_matrix_temp, K_matrix_temp, K_matrix_temp), axis=0)
        self.anno = np.concatenate(
            (anno_temp, anno_temp, anno_temp, anno_temp), axis=0)

        assert len(self.K_matrix) == len(self.anno) == len(self.image_names)

        if set_type == "train":
            n_start = 0
            n_end = 104192
            # n_end = 26048
        elif set_type == "val":
            n_start = 104192
            n_end = 117216
        else:
            n_start = 117216
            n_end = len(self.anno)

        self.image_names = self.image_names[n_start:n_end]
        self.K_matrix = self.K_matrix[n_start:n_end]
        self.anno = self.anno[n_start:n_end]

        print(f'Number of {set_type} samples: {len(self.image_names)}')

        self.image_raw_transform = transforms.ToTensor()
        self.image_transform = img_transform
        self.heatmap_transform = heatmap_transform

    def __len__(self):
        return len(self.anno)

    def __getitem__(self, idx):

        image_name = self.image_names[idx]
        image_raw = Image.open(os.path.join(self.image_dir, image_name))

        # Apply transform to the image
        if self.image_transform != None:
            state = torch.get_rng_state()
            image = self.image_transform(image_raw)
        else:
            image = self.image_raw_transform(image_raw)

        # Convert 3D points ot 2D space
        keypoints = project_points_3D_to_2D(self.anno[idx], self.K_matrix[idx])

        # Apply transform to the heatmap
        if self.heatmap_transform != None:
            # When transform to heatmap is applied we need full heatmap dimension
            heatmaps = vector_to_heatmaps(
                keypoints, scale_factor=1, out_size=RAW_IMG_SIZE)
            heatmaps = torch.from_numpy(np.float32(heatmaps))
            torch.set_rng_state(state)
            heatmaps = self.heatmap_transform(heatmaps)
        else:
            # When transform to heatmap is NOT applied we need dimmension of model input image
            keypoints = keypoints / RAW_IMG_SIZE
            heatmaps = vector_to_heatmaps(
                keypoints, scale_factor=MODEL_IMG_SIZE, out_size=MODEL_IMG_SIZE)
            heatmaps = torch.from_numpy(np.float32(heatmaps))

        # Convert to tensors
        image_raw = self.image_raw_transform(image_raw)
        keypoints = torch.from_numpy(keypoints)

        return {
            "image": image,
            "keypoints": keypoints,
            "heatmaps": heatmaps,
            "image_name": image_name,
            "image_raw": image_raw,
        }


class FreiHAND_albu(Dataset):
    """
    Class to load FreiHAND dataset. This dataset class is using only training part.
    It devides the dataste into three subsets in proportion 80/15/5.
    Parameters:
        config - config file (device, data_dir, )
        set_type - type of data split (train/val/test)
        img_transform - transform and augumentation for input img
        heatmap_transform - transform and augumentation for input heatmap with keypoints
    Link to dataset:
    https://lmb.informatik.uni-freiburg.de/resources/datasets/FreihandDataset.en.html
    """

    def __init__(self, config, set_type="train", albumetations=None, img_transform=None, heatmap_transform=None):

        self.device = config["device"]
        self.image_dir = os.path.join(config["data_dir"], "training/rgb")
        self.image_names = np.sort(os.listdir(self.image_dir))

        fn_K_matrix = os.path.join(config["data_dir"], "training_K.json")
        with open(fn_K_matrix, "r") as f:
            K_matrix_temp = np.array(json.load(f))

        fn_anno = os.path.join(config["data_dir"], "training_xyz.json")
        with open(fn_anno, "r") as f:
            anno_temp = np.array(json.load(f))

        self.K_matrix = np.concatenate(
            (K_matrix_temp, K_matrix_temp, K_matrix_temp, K_matrix_temp), axis=0)
        self.anno = np.concatenate(
            (anno_temp, anno_temp, anno_temp, anno_temp), axis=0)

        assert len(self.K_matrix) == len(self.anno) == len(self.image_names)

        if set_type == "train":
            n_start = 0
            n_end = 104192
            # n_end = 26048
        elif set_type == "val":
            # n_start = 26048
            n_start = 104192
            n_end = 117216
            # n_end = 32560
        else:
            n_start = 117216
            n_end = len(self.anno)

        self.image_names = self.image_names[n_start:n_end]
        self.K_matrix = self.K_matrix[n_start:n_end]
        self.anno = self.anno[n_start:n_end]

        print(f'Number of {set_type} samples: {len(self.image_names)}')

        self.image_raw_transform = transforms.ToTensor()
        self.image_transform = img_transform
        self.heatmap_transform = heatmap_transform
        self.albumetations = albumetations

    def __len__(self):
        return len(self.anno)

    def __getitem__(self, idx):

        image_name = self.image_names[idx]
        image_raw = Image.open(os.path.join(self.image_dir, image_name))

        keypoints = project_points_3D_to_2D(self.anno[idx], self.K_matrix[idx])
        # keypoints = np.rint(keypoints)
        # print(keypoints)

        transformed = self.albumetations(
            image=np.asarray(image_raw), keypoints=keypoints)
        transformed_image = transformed['image']
        transformed_keypoints = np.asarray(transformed['keypoints'])

        heatmaps = vector_to_heatmaps(
            transformed_keypoints, scale_factor=1, out_size=MODEL_IMG_SIZE)

        # Apply transform to the image
        # if self.image_transform != None:
        #     state = torch.get_rng_state()
        #     image = self.image_transform(image_raw)
        # else:
        #     image = self.image_raw_transform(image_raw)

        # Convert 3D points ot 2D space
        # keypoints = project_points_3D_to_2D(self.anno[idx], self.K_matrix[idx])

        # Apply transform to the heatmap
        # if self.heatmap_transform != None:
        #     # When transform to heatmap is applied we need full heatmap dimension
        #     heatmaps = vector_to_heatmaps(keypoints, scale_factor = 1, out_size = RAW_IMG_SIZE)
        #     heatmaps = torch.from_numpy(np.float32(heatmaps))
        #     torch.set_rng_state(state)
        #     heatmaps = self.heatmap_transform(heatmaps)
        # else:
        #     # When transform to heatmap is NOT applied we need dimmension of model input image
        #     keypoints = keypoints / RAW_IMG_SIZE
        #     heatmaps = vector_to_heatmaps(keypoints, scale_factor = MODEL_IMG_SIZE, out_size = MODEL_IMG_SIZE)
        #     heatmaps = torch.from_numpy(np.float32(heatmaps))

        # Convert to tensors
        image_raw = self.image_raw_transform(image_raw)
        keypoints = torch.from_numpy(transformed_keypoints / MODEL_IMG_SIZE)

        return {
            "image": transformed_image,
            "keypoints": keypoints,
            "heatmaps": heatmaps,
            "image_name": image_name,
            "image_raw": image_raw,
        }


class FreiHAND_evaluation(Dataset):
    """
    Class to load FreiHAND dataset. This is the official evaluation dataset.
    Parameters:
        config - config file (device, data_dir, )
        img_transform - transform and augumentation for input img. 
                        For evaluation it should be just totensor,resize, normalization.
    Link to dataset:
    https://lmb.informatik.uni-freiburg.de/resources/datasets/FreihandDataset.en.html
    """

    def __init__(self, config, img_transform=None, albumetations=None):

        self.image_dir = os.path.join(config["data_dir"], "evaluation/rgb")

        fn_K_matrix = os.path.join(config["data_dir"], "evaluation_K.json")
        with open(fn_K_matrix, "r") as f:
            self.K_matrix = np.array(json.load(f))

        fn_anno = os.path.join(config["data_dir"], "evaluation_xyz.json")
        with open(fn_anno, "r") as f:
            self.anno = np.array(json.load(f))

        self.device = config["device"]
        self.image_names = np.sort(os.listdir(self.image_dir))

        self.image_raw_transform = transforms.ToTensor()
        self.image_transform = img_transform
        self.albumetations = albumetations

    def __len__(self):
        return len(self.anno)

    def __getitem__(self, idx):

        image_name = self.image_names[idx]
        image_raw = Image.open(os.path.join(self.image_dir, image_name))

        keypoints = project_points_3D_to_2D(self.anno[idx], self.K_matrix[idx])
        # keypoints = np.rint(keypoints)
        # print(keypoints)

        transformed = self.albumetations(
            image=np.asarray(image_raw), keypoints=keypoints)
        transformed_image = transformed['image']
        transformed_keypoints = np.asarray(transformed['keypoints'])

        heatmaps = vector_to_heatmaps(
            transformed_keypoints, scale_factor=1, out_size=MODEL_IMG_SIZE)

        # Apply transform to the image
        # if self.image_transform != None:
        #     state = torch.get_rng_state()
        #     image = self.image_transform(image_raw)
        # else:
        #     image = self.image_raw_transform(image_raw)

        # Convert 3D points ot 2D space
        # keypoints = project_points_3D_to_2D(self.anno[idx], self.K_matrix[idx])

        # Apply transform to the heatmap
        # if self.heatmap_transform != None:
        #     # When transform to heatmap is applied we need full heatmap dimension
        #     heatmaps = vector_to_heatmaps(keypoints, scale_factor = 1, out_size = RAW_IMG_SIZE)
        #     heatmaps = torch.from_numpy(np.float32(heatmaps))
        #     torch.set_rng_state(state)
        #     heatmaps = self.heatmap_transform(heatmaps)
        # else:
        #     # When transform to heatmap is NOT applied we need dimmension of model input image
        #     keypoints = keypoints / RAW_IMG_SIZE
        #     heatmaps = vector_to_heatmaps(keypoints, scale_factor = MODEL_IMG_SIZE, out_size = MODEL_IMG_SIZE)
        #     heatmaps = torch.from_numpy(np.float32(heatmaps))

        # Convert to tensors
        image_raw = self.image_raw_transform(image_raw)
        keypoints = torch.from_numpy(transformed_keypoints / MODEL_IMG_SIZE)

        return {
            "image": transformed_image,
            "keypoints": keypoints,
            "heatmaps": heatmaps,
            "image_name": image_name,
            "image_raw": image_raw,
        }
