'''
File containing most basic functions 
'''
import numpy as np
import cv2 as cv2
from config import *
import torch
import torch.nn as nn
from PIL import Image

def pil_to_cv(pil_image):
    '''
    Converting PIL image format to opencv/numpy
    '''
    open_cv_image = np.array(pil_image) 
    # Convert RGB to BGR 
    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
    return open_cv_image

def cv_to_pil(cv_image):
    '''
    Converting opencv/numpy image format to PIL
    '''
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(cv_image)
    return pil_image

def count_parameters(model):
    '''
    Function to count parameters in a model that require learning
    '''
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def vector_to_heatmaps(keypoints, scale_factor = 1, out_size = MODEL_IMG_SIZE):
    '''
    Creates 2D heatmaps from keypoint locations for a single image.
    Parameters:
        keypoints - array of size N_KEYPOINTS x 2
        scale_factor - factor to scale keypoints (factor = 1 when keypoints are org size)
        out_size - size of output heatmap
    Output: array of size N_KEYPOINTS x MODEL_IMG_SIZE x MODEL_IMG_SIZE
    '''
    heatmaps = np.zeros([N_KEYPOINTS, out_size, out_size])
    for k, (x, y) in enumerate(keypoints):
        x, y = int(x * scale_factor), int(y * scale_factor)
        if (0 <= x < out_size) and (0 <= y < out_size):
            heatmaps[k, int(y), int(x)] = 1

    heatmaps = blur_heatmaps(heatmaps)
    return heatmaps

def blur_heatmaps(heatmaps):
    '''Blurs heatmaps using GaussinaBlur of defined size'''
    heatmaps_blurred = heatmaps.copy()
    for k in range(len(heatmaps)):
        if heatmaps_blurred[k].max() == 1:
            heatmaps_blurred[k] = cv2.GaussianBlur(heatmaps[k], (51, 51), 3)
            heatmaps_blurred[k] = heatmaps_blurred[k] / heatmaps_blurred[k].max()
    return heatmaps_blurred

def project_points_3D_to_2D(xyz, K):
    '''
    Projects 3D coordinates into 2D space.
    It is a part of FreiHAND dataset repository.
    Parameters:
        xyz - 3D keypoints
        K - camera intrinsic
    Out:
        2D coordinates of np.shape: (21,2)
    '''
    xyz = np.array(xyz)
    K = np.array(K)
    uv = np.matmul(K, xyz.T).T
    return uv[:, :2] / uv[:, -1:]

class IoULoss(nn.Module):
    '''
    Intersection over Union Loss.
    IoU loss is changed for heatmaps.
    '''

    def __init__(self,epsilon = 1e-6):
        super(IoULoss, self).__init__()
        self.epsilon = epsilon

    def _op_sum(self, x):
        return x.sum(-1).sum(-1)

    def forward(self, y_pred, y_true):
        inter = self._op_sum(y_true * y_pred)
        union = (
            self._op_sum(y_true ** 2)
            + self._op_sum(y_pred ** 2)
            - self._op_sum(y_true * y_pred)
        )
        iou = (inter + self.epsilon) / (union + self.epsilon)
        iou = torch.mean(iou)

        return 1 - iou

def heatmaps_to_coordinates(heatmaps):
    """
    Heatmaps is a numpy array
    Its size - (batch_size, n_keypoints, img_size, img_size)
    """
    batch_size = heatmaps.shape[0]
    sums = heatmaps.sum(axis=-1).sum(axis=-1)
    sums = np.expand_dims(sums, [2, 3])
    normalized = heatmaps / sums
    x_prob = normalized.sum(axis=2)
    y_prob = normalized.sum(axis=3)

    arr = np.tile(np.float32(np.arange(0, 128)), [batch_size, 21, 1])
    x = (arr * x_prob).sum(axis=2)
    y = (arr * y_prob).sum(axis=2)
    keypoints = np.stack([x, y], axis=-1)
    return keypoints / MODEL_IMG_SIZE