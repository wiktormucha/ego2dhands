from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
import sys
sys.path.append("../")


def get_instr(arr: np.array) -> np.array:
    """
    Function to read camera intrinsic form a file given in H2O dataset.

    Args:
        arr (np.array): Array read for ma file.

    Returns:
        np.array: Correctly read intrinsics.
    """

    cam = np.zeros((3, 3))
    cam[0][0] = arr[0]
    cam[0][2] = arr[2]
    cam[1][1] = arr[1]
    cam[1][2] = arr[3]
    cam[2][2] = 1

    return cam


class H2O_Dataset(Dataset):
    """
    H2O Dataset for egocentric hand tests only
    """

    def __init__(self, config: dict):
        """
        Initialisation of the dataset

        Args:
            config (dict): Config dictionary with needed data for training.
        """

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

    def __len__(self) -> int:
        """
        Return length of the dataset

        Returns:
            int: Dataset length
        """
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> dict:
        """
        Returns item from dataset

        Args:
            idx (int): Given index in dataset

        Returns:
            dict: Dictionary with item containing:
                    - Raw image
                    - Camera position
                    - Hand position
                    - Hand position mano
                    - Camera intrinsic
                    - Image name
        """

        image_raw = Image.open(self.img_paths[idx])
        cam_pose = np.loadtxt(self.cam_pose[idx])
        hand_pose = np.loadtxt(self.hand_pose[idx])
        hand_pose_mano = np.loadtxt(self.hand_pose_mano[idx])
        cam_instr = get_instr(np.loadtxt(self.cam_instr[idx]))

        return {
            "image_raw": image_raw,
            "cam_pose": cam_pose,
            "hand_pose": hand_pose,
            "hand_pose_mano": hand_pose_mano,
            "cam_instr": cam_instr,
            "img_name": self.img_paths[idx]
        }
