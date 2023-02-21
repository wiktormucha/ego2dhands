from config import *
from utils.utils import heatmaps_to_coordinates
from torchvision import transforms
import torch
import sys
import copy as copy
from PIL import ImageOps
sys.path.append("../")


def run_model_on_hands(model, imgs):
    """Function to run hand predicotr in egocentric data

    Args:
        model (_type_): predicting model
        imgs (_type_): left and right hands cropped images

    Returns:
        _type_: predicted keypoints in range <0.1>
    """

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(MODEL_IMG_SIZE),
            transforms.Normalize(
                mean=TRAIN_DATASET_MEANS, std=TRAIN_DATASET_STDS)
        ]
    )

    # Flip left to right for dummy model
    img0 = ImageOps.mirror(imgs[0])
    img1 = imgs[1]
    img0 = transform(img0)
    img1 = transform(img1)
    flip = transforms.RandomHorizontalFlip(p=1)
    inpt = torch.stack([img0, img1], dim=0)

    # Predict
    pred_heatmaps = model(inpt)

    # Flip prediction back to real axis
    left = flip(pred_heatmaps[0])
    right = pred_heatmaps[1]
    heatmaps = torch.stack([left, right], dim=0)

    # Return coordinates
    pred_keypoints = heatmaps_to_coordinates(heatmaps.detach().numpy())

    return pred_keypoints


def preds_to_full_image(predictions, hands_bb, scale):
    """Function taking predictions and moving coordinates to full size image with two hands

    Args:
        predictions (_type_): left and right hand prediction
        hands_bb (_type_): list of bb of hands
        scale (_type_): size of imaes used for prediction

    Returns:
        _type_: predictions scaled to full image
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
