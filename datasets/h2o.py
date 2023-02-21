# import torch
from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
# import sys
import matplotlib.pyplot as plt
# import cv2


def get_instr(arr):
    cam = np.zeros((3, 3))
    cam[0][0] = arr[0]
    cam[0][2] = arr[2]
    cam[1][1] = arr[1]
    cam[1][2] = arr[3]
    cam[2][2] = 1

    return cam


class H2O_Dataset(Dataset):
    """
    Class to load FreiHAND dataset. Only training part is used here.
    Augmented images are not used, only raw - first 32,560 images
    Link to dataset:
    https://lmb.informatik.uni-freiburg.de/resources/datasets/FreihandDataset.en.html
    """

    def __init__(self, config, set_type="train", img_transform=None, heatmap_transform=None):

        self.device = config["device"]

        # self.image_dir = os.path.join(
        #     config["data_dir"], "subject1", 'h1', '1', 'cam4', 'rgb')
        # /data/wmucha/datasets/h2o/h2o_CASA/subject1/h1/1/cam4/rgb/000193.png
        # self.image_dir = os.path.join(
        #     config["data_dir"], "subject3", 'k2', '1', 'cam4', 'rgb')
        # # /data/wmucha/datasets/h2o/h2o_CASA/subject3/k2/1/cam4/rgb/000056.png
        # self.image_dir = os.path.join(
        #     config["data_dir"], "subject6", 's2', '3', 'cam4', 'rgb')
        # /data/wmucha/datasets/h2o/h2o_CASA/subject6/s2/3/cam4/rgb/000091.png
        self.image_dir = os.path.join(
            config["data_dir"], "subject9", 's4', '6', 'cam4', 'rgb')
        # print(self.image_dir)
        self.image_names = np.sort(os.listdir(self.image_dir))
        # print(self.image_names)
        temp_rgbpaths = []
        for img_name in self.image_names:
            temp_rgbpaths.append(os.path.join(self.image_dir, img_name))
        # self.img_paths = os.path.join(self.image_dir, self.image_names)
        # print(temp_paths)

        # temp_campose_dir = os.path.join(config["data_dir"], "subject1",'h1','1','cam4','cam_pose')
        temp_cam_pose = []
        for cam_pose in temp_rgbpaths:
            temp_cam_pose.append(cam_pose.replace(
                'rgb', 'cam_pose').replace('.png', '.txt'))

        temp_hand_pose = []
        for hand_pose in temp_rgbpaths:
            temp_hand_pose.append(hand_pose.replace(
                'rgb', 'hand_pose').replace('.png', '.txt'))

        temp_hand_pose_mano = []
        for hand_pose_mano in temp_rgbpaths:
            temp_hand_pose_mano.append(hand_pose_mano.replace(
                'rgb', 'hand_pose_mano').replace('.png', '.txt'))

        temp_cam_instr = []
        for cam_instr in temp_rgbpaths:
            temp_cam_instr.append(os.path.join(
                config["data_dir"], "subject1", 'h1', '1', 'cam4', 'cam_intrinsics.txt'))

        self.img_paths = np.asarray(temp_rgbpaths)
        self.cam_pose = np.asarray(temp_cam_pose)
        self.hand_pose = np.asarray(temp_hand_pose)
        self.hand_pose_mano = np.asarray(temp_hand_pose_mano)
        self.cam_instr = np.asarray(temp_cam_instr)

        # print(self.cam_pose)
        # fn_K_matrix = os.path.join(config["data_dir"], "training_K.json")
        # with open(fn_K_matrix, "r") as f:
        #     K_matrix_temp = np.array(json.load(f))

        # fn_anno = os.path.join(config["data_dir"], "training_xyz.json")
        # with open(fn_anno, "r") as f:
        #     anno_temp = np.array(json.load(f))

        # self.K_matrix = np.concatenate((K_matrix_temp, K_matrix_temp, K_matrix_temp, K_matrix_temp) , axis =0)
        # self.anno = np.concatenate((anno_temp, anno_temp, anno_temp, anno_temp) , axis =0)

        # assert len(self.K_matrix) == len(self.anno) == len(self.image_names)

        # if set_type == "train":
        #     n_start = 0
        #     n_end = len(self.anno)
        # elif set_type == "val":
        #     n_start = 104192
        #     n_end = 123728
        # else:
        #     n_start = 123728
        #     n_end = len(self.anno)

        # self.image_names = self.image_names[n_start:n_end]
        # self.K_matrix = self.K_matrix[n_start:n_end]
        # self.anno = self.anno[n_start:n_end]

        # print(f'Number of {set_type} samples: {len(self.image_names)}')

        # bg_path="/data/wmucha/datasets/coco/train2017"
        # self.files=os.listdir(bg_path)

        # self.image_raw_transform = transforms.ToTensor()
        # if img_transform == None:
        #     self.image_transform = transforms.Compose(
        #         [
        #             transforms.Resize(MODEL_IMG_SIZE),
        #             # RandomBackground(files = self.files, bg_path= bg_path),
        #             transforms.ToTensor(),
        #             transforms.Normalize(mean=DATASET_MEANS, std=DATASET_STDS),
        #         ]
        #     )
        # else:
        #     self.image_transform = img_transform

        # self.heatmap_transform = heatmap_transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):

        # return item
        # image_name = self.image_names[idx]
        image_raw = Image.open(self.img_paths[idx])
        cam_pose = np.loadtxt(self.cam_pose[idx])
        hand_pose = np.loadtxt(self.hand_pose[idx])
        hand_pose_mano = np.loadtxt(self.hand_pose_mano[idx])
        cam_instr = get_instr(np.loadtxt(self.cam_instr[idx]))
        # state = torch.get_rng_state()
        # image = self.image_transform(image_raw)
        # image_raw = self.image_raw_transform(image_raw)

        # keypoints = projectPoints(self.anno[idx], self.K_matrix[idx])
        # keypoints = keypoints / RAW_IMG_SIZE
        # heatmaps = vector_to_heatmaps(keypoints)
        # keypoints = torch.from_numpy(keypoints)
        # heatmaps = torch.from_numpy(np.float32(heatmaps))

        # # if self.heatmap_transform != None:
        # #     torch.set_rng_state(state)
        # #     heatmaps = self.heatmap_transform(heatmaps)

        return {
            # "image": image,
            # "keypoints": keypoints,
            # "heatmaps": heatmaps,
            # "image_name": image_name,
            "image_raw": image_raw,
            "cam_pose": cam_pose,
            "hand_pose": hand_pose,
            "hand_pose_mano": hand_pose_mano,
            "cam_instr": cam_instr,
            "img_name": self.img_paths[idx]
        }
