from config import *
from utils.utils import heatmaps_to_coordinates
from torchvision import transforms
import torch
import sys
import copy as copy
import numpy as np
from PIL import ImageOps
sys.path.append("../")


def run_model_on_hands(model: torch.nn.Module, imgs: list) -> np.array:
    """
    Function to run hand predicotr in egocentric data

    Args:
        model (torch.nn.Module): Prediction model for a single hand.
        imgs (list): Left and right hands segmented images.

    Returns:
        np.array: predicted keypoints
    """

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(MODEL_IMG_SIZE),
            transforms.Normalize(
                mean=TRAIN_DATASET_MEANS, std=TRAIN_DATASET_STDS)
        ]
    )

    img0 = imgs[0]
    img1 = imgs[1]

    img0 = transform(img0)
    img1 = transform(img1)
    inpt = torch.stack([img0, img1], dim=0)
    pred_heatmaps = model(inpt)
    left = pred_heatmaps[0]
    right = pred_heatmaps[1]
    heatmaps = torch.stack([left, right], dim=0)
    pred_keypoints = heatmaps_to_coordinates(heatmaps.detach().numpy())

    return pred_keypoints


def preds_to_full_image(predictions: np.array, hands_bb: list, scale: list) -> list:
    """
    Function taking predictions and moving coordinates to full size image with two hands

    Args:
        predictions (np.array): left and right hand prediction
        hands_bb (list): List of bb of hands
        scale (list): Size of imaes used for prediction

    Returns:
        list: Points transformed to input image
    """

    full_scale_preds = []

    pts = predictions[0] * scale[0]
    pts[:, 0] = pts[:, 0] + hands_bb[0]['x_min']
    pts[:, 1] = pts[:, 1] + hands_bb[0]['y_min']
    full_scale_preds.append(pts)

    pts = predictions[1] * scale[1]
    pts[:, 0] += hands_bb[1]['x_min']
    pts[:, 1] += hands_bb[1]['y_min']
    full_scale_preds.append(pts)

    return full_scale_preds
