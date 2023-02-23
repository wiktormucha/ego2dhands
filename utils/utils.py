import numpy as np
import cv2 as cv2
from config import *
import torch
import torch.nn as nn


def count_parameters(model: nn.Module) -> int:
    """
    Counts parameters for training in a given model.

    Args:
        model (nn.Module): Input model

    Returns:
        int: No. of trainable parameters
    """

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def vector_to_heatmaps(keypoints: np.array, scale_factor: int = 1, out_size: int = MODEL_IMG_SIZE) -> np.array:
    """
    Creates 2D heatmaps from keypoint locations for a single image.

    Args:
        keypoints (np.array): array of size N_KEYPOINTS x 2
        scale_factor (int, optional): Factor to scale keypoints (factor = 1 when keypoints are org size). Defaults to 1.
        out_size (int, optional): Size of output heatmap. Defaults to MODEL_IMG_SIZE.

    Returns:
        np.array: Heatmap
    """
    heatmaps = np.zeros([N_KEYPOINTS, out_size, out_size])
    for k, (x, y) in enumerate(keypoints):
        x, y = int(x * scale_factor), int(y * scale_factor)
        if (0 <= x < out_size) and (0 <= y < out_size):
            heatmaps[k, int(y), int(x)] = 1

    heatmaps = blur_heatmaps(heatmaps)
    return heatmaps


def blur_heatmaps(heatmaps: np.array) -> np.array:
    """
    Blurs heatmaps using GaussinaBlur of defined size

    Args:
        heatmaps (np.array): Input heatmap

    Returns:
        np.array: Output heatmap
    """
    heatmaps_blurred = heatmaps.copy()
    for k in range(len(heatmaps)):
        if heatmaps_blurred[k].max() == 1:
            heatmaps_blurred[k] = cv2.GaussianBlur(heatmaps[k], (51, 51), 3)
            heatmaps_blurred[k] = heatmaps_blurred[k] / \
                heatmaps_blurred[k].max()
    return heatmaps_blurred


def project_points_3D_to_2D(xyz: list, K: list) -> np.array:
    """
    Projects 3D coordinates into 2D space. Taken from FreiHAND dataset repository.

    Args:
        xyz (list): 3D keypoints
        K (list): camera intrinsic

    Returns:
        np.array: 2D keypoints
    """
    xyz = np.array(xyz)
    K = np.array(K)
    uv = np.matmul(K, xyz.T).T
    return uv[:, :2] / uv[:, -1:]


class IoULoss(nn.Module):
    """
    Intersection over Union Loss. IoU loss is changed for heatmaps.
    """

    def __init__(self, epsilon=1e-6):
        super(IoULoss, self).__init__()
        self.epsilon = epsilon

    def _op_sum(self, x):
        return x.sum(-1).sum(-1)

    def forward(self, y_pred: np.array, y_true: np.array) -> float:
        """
        Forward pass

        Args:
            y_pred (np.array): Predicted heatmap
            y_true (np.array): GT heatmap

        Returns:
            float: Loss
        """
        inter = self._op_sum(y_true * y_pred)
        union = (
            self._op_sum(y_true ** 2)
            + self._op_sum(y_pred ** 2)
            - self._op_sum(y_true * y_pred)
        )
        iou = (inter + self.epsilon) / (union + self.epsilon)
        iou = torch.mean(iou)

        return 1 - iou


def heatmaps_to_coordinates(heatmaps: np.array) -> np.array:
    """
    Transforms heat,aps to 2d keypoints

    Args:
        heatmaps (np.array): Input heatmap

    Returns:
        np.array: Output points
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
