from torch.utils.data import DataLoader
from config import *
import torch
from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
import sys
import matplotlib.pyplot as plt
import cv2
import mediapipe as mp
import copy as copy
from PIL import Image, ImageOps
# sys.path.append("../")

from utils.hand_detector import get_hands_img
from models.models import CustomHeatmapsModel, EfficientWaterfall
from torchvision import transforms
from utils.testing import batch_epe_calculation, batch_auc_calculation, batch_pck_calculation, show_batch_predictions
from utils.utils import project_points_3D_to_2D, heatmaps_to_coordinates
from datasets.h2o import H2O_Dataset
import pandas as pd
import tqdm
from datasets.h2o import get_instr
from utils.egocentric import run_model_on_hands

IMAGE_N = 10  # Index of image to see


config = {'device': 3,
          'data_dir': '/data/wmucha/datasets/h2o/h2o_CASA',
          'batch_size': 1,
          "model_path": "/caa/Homes01/wmucha/repos/applied_deep_learning/applied_dl/waterfall_fulldata_scratch9_121",
          }


# dataset = H2O_Dataset(config)

# dataloader = DataLoader(
#         dataset,
#         config["batch_size"],
#         shuffle=False,
#         drop_last=False,
#         num_workers=6,
#         pin_memory=True
#     )
# item = dataset[IMAGE_N]

# img = item['image_raw']
# cam_pose = item['cam_pose']
# hand_pose = item['hand_pose']
# hand_pose_mano = item['hand_pose_mano']
# cam_instr = item['cam_instr']

def get_testing_images(data_dir: str) -> dict:
    """
    Returns testing images from H2O dataset.

    Args:
        data_dir (str): Main folder path

    Returns:
        dict:   -'img_paths': img_paths,
                -'cam_pose': cam_pose,
                -'hand_pose': hand_pose,
                -'cam_instr': cam_instr
    """

    # image_dir = os.path.join(data_dir, "subject1", 'h1', '1', 'cam4', 'rgb')
    image_dir = os.path.join(data_dir, "subject6", 's2', '3', 'cam4', 'rgb')
    # image_dir = os.path.join(data_dir, "subject3", 'k2', '1', 'cam4', 'rgb')
    # N:\datasets\h2o_CASA\subject3\k2\1\cam4\rgb256
    # N:\datasets\h2o_CASA\subject2\k1\1\cam4\rgb
    # N:\datasets\h2o_CASA\subject6\s2\3\cam4\rgb256
    # print(image_dir)
    image_names = np.sort(os.listdir(image_dir))
    # print(image_names)
    temp_rgbpaths = []
    for img_name in image_names:
        temp_rgbpaths.append(os.path.join(image_dir, img_name))
        # img_paths = os.path.join(image_dir, image_names)
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

    img_paths = np.asarray(temp_rgbpaths)
    cam_pose = np.asarray(temp_cam_pose)
    hand_pose = np.asarray(temp_hand_pose)
    hand_pose_mano = np.asarray(temp_hand_pose_mano)
    cam_instr = np.asarray(temp_cam_instr)

    return {
        'img_paths': img_paths,
        'cam_pose': cam_pose,
        'hand_pose': hand_pose,
        'cam_instr': cam_instr
    }


# model = CustomHeatmapsModel(3, 21)
model = EfficientWaterfall(N_KEYPOINTS)
model.load_state_dict(
    torch.load(config["model_path"],
               map_location=torch.device(config["device"]))
)
model.eval()
print("Model loaded")

hand_model = mp.solutions.hands.Hands()

data_set = get_testing_images(config['data_dir'])

img_paths = data_set['img_paths']

cam_instr_paths = data_set['cam_instr']
hand_pose_paths = data_set['hand_pose']

if os.path.exists('egocentric_results.csv'):
    df = pd.read_csv('egocentric_results.csv')
else:
    df = pd.DataFrame(columns=['img', 'pck_acc', 'epe_lst', 'auc_lst'])


pck_acc = []
epe_lst = []
auc_lst = []

imgs_checked = []
# len(img_paths)
for idx in range(0, 700):
    # print(df)
    print(img_paths[idx])
    # print(df['img'])

    # Skip if exist in the file
    if not df.empty:
        if img_paths[idx] in df['img'].values:
            # print('istnieje')
            continue

    img = Image.open(img_paths[idx])
    # imgs_checked.append(img_paths[idx])
    # print(hand_pose)
    hand_pose = np.loadtxt(hand_pose_paths[idx])
    cam_instr = get_instr(np.loadtxt(cam_instr_paths[idx]))

    # print(hand_pose)
    # print(cam_instr)
    hands_dict = get_hands_img(
        img, hand_pose, cam_instr=cam_instr, hand_model=hand_model)

    imgs = hands_dict['hands_seg']
    gts = hands_dict['gt']
    hand_label = hands_dict['hand_type']

    pred = run_model_on_hands(model, imgs)

    gts0 = np.reshape(gts[0], (1, 21, 2))
    gts1 = np.reshape(gts[1], (1, 21, 2))
    gt_keyboards = np.concatenate((gts0, gts1))
    # print(gt_keyboards)
    # print(gt_keyboards.shape)
    # print(pred.shape)

    avg_acc = batch_pck_calculation(
        pred, gt_keyboards, treshold=0.2, mask=None, normalize=None)
    pck_acc.append(avg_acc)

    # Calculate EPE mean and median, mind that it depends on what scale of input keypoints
    epe = batch_epe_calculation(np.reshape(pred[0], (1, 21, 2)), gts0)
    epe2 = batch_epe_calculation(np.reshape(pred[1], (1, 21, 2)), gts1)
    # print(epe)
    # print(epe2)
    epe_lst.append(epe)
    # break
    # AUC calculation
    auc = batch_auc_calculation(pred, gt_keyboards, num_step=20, mask=None)
    auc_lst.append(auc)

    if epe > 12:

        plt.imshow(ImageOps.mirror(imgs[0]))
        plt.clf()
        plt.imshow((imgs[0]))
        # print(imgs[0].size[0])
        pts = pred[0] * imgs[0].size[0]
        # TODO draw on image
        for finger, params in COLORMAP.items():
            plt.plot(
                pts[params["ids"], 0],
                pts[params["ids"], 1],
                params["color"],
            )

        plt.savefig('left_'+str(idx)+'.png')

        plt.clf()

        plt.imshow((imgs[1]))
        # print(imgs[0].size[0])
        pts = pred[1] * imgs[1].size[0]
        # TODO draw on image
        for finger, params in COLORMAP.items():
            plt.plot(
                pts[params["ids"], 0],
                pts[params["ids"], 1],
                params["color"],
            )

        plt.savefig('right_'+str(idx)+'.png')

    # temp = pd.DataFrame()

    # temp['img'] = imgs_checked
    # temp['pck_acc'] = avg_acc
    # temp['epe_lst'] = epe
    # temp['auc_lst'] = auc

    df.loc[len(df.index)] = [img_paths[idx], avg_acc, epe, auc]

    # df = pd.concat([df,temp], join="inner")
    # print('Saving file...')
    df.to_csv('egocentric_results.csv', index=False)
