from torchvision import transforms
from datasets.augumentations import RandomBoxes, RandomNoise, RandomBackground
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

MODEL_NEURONS = 16
BB_FACTOR = 30

# Data parameters
N_KEYPOINTS = 21
N_IMG_CHANNELS = 3
RAW_IMG_SIZE = 224
MODEL_IMG_SIZE = 128
RANDOM_CROP_SIZE = 180
DATA_DIR = "/data/wmucha/datasets/FreiHAND"
TRAIN_DATASET_MEANS = [0.4532, 0.4522, 0.4034]
TRAIN_DATASET_STDS = [0.2218, 0.2186, 0.2413]
DATASET_MEANS = [0.3950, 0.4323, 0.2954]
DATASET_STDS = [0.1966, 0.1734, 0.1836]
ONLY_GREENSCREEN_IMAGES = False


# Training parameters
EXPERIMENT_NAME = "waterfall_fulldata_scratch17"
MAX_EPOCHS = 1000
BACTH_SIZE = 32
LEARNING_RATE = 0.0003125
DEVICE = 0
EARLY_STOPPING = 20
CONTINUE_FROM_CHECKPOINT = True
CHECKPOINT_DIR = "/caa/Homes01/wmucha/repos/applied_deep_learning/applied_dl/waterfall_fulldata_scratch16_47"
MOMENTUM = 0
GRAD_CLIPPING = 0
WEIGHT_DECAY = 1e-5

# Testing parameters
TESTING_DEVICE = 3
TESTING_BATCH_SIZE = 32
COLORMAP = {
    "thumb": {"ids": [0, 1, 2, 3, 4], "color": "g"},
    "index": {"ids": [0, 5, 6, 7, 8], "color": "c"},
    "middle": {"ids": [0, 9, 10, 11, 12], "color": "b"},
    "ring": {"ids": [0, 13, 14, 15, 16], "color": "m"},
    "little": {"ids": [0, 17, 18, 19, 20], "color": "r"},
}

# Image augumentations
TRAIN_IMG_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.RandomRotation(degrees=(0, 30)),
        transforms.RandomCrop(RANDOM_CROP_SIZE),
        transforms.Resize(MODEL_IMG_SIZE),
        # transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomVerticalFlip(p=0.5),
        # RandomNoise(p=0.3),
        # transforms.GaussianBlur(3),
        transforms.Normalize(
            mean=TRAIN_DATASET_MEANS, std=TRAIN_DATASET_STDS),
    ]
)

TRAIN_HEATMAP_TRANSFORM = transforms.Compose(
    [
        transforms.RandomRotation(degrees=(0, 30)),
        transforms.RandomCrop(RANDOM_CROP_SIZE),
        transforms.Resize(MODEL_IMG_SIZE),
        # transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomVerticalFlip(p=0.5),
    ]
)

VAL_IMG_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize(MODEL_IMG_SIZE),
        transforms.Normalize(
            mean=TRAIN_DATASET_MEANS, std=TRAIN_DATASET_STDS),
    ]
)


ALBUMENTATION_TRAIN = A.Compose(
    [
        A.SafeRotate(always_apply=False, p=0.5, limit=(-20, 20),
                     interpolation=0, border_mode=1, value=(0, 0, 0), mask_value=None),
        A.HorizontalFlip(always_apply=False, p=0.5),
        A.VerticalFlip(always_apply=False, p=0.5),
        A.RandomResizedCrop(always_apply=True, p=1, height=MODEL_IMG_SIZE, width=MODEL_IMG_SIZE, scale=(
            0.3, 1.0), ratio=(1, 1), interpolation=0),
        A.MotionBlur(always_apply=False, p=0.2,
                     blur_limit=(3, 7), allow_shifted=True),
        # A.RandomGridShuffle(always_apply=False, p=0.2, grid=(2, 2)),
        A.Downscale(always_apply=False, p=0.2,
                    scale_min=0.9, scale_max=0.99),
        A.Normalize(mean=TRAIN_DATASET_MEANS, std=TRAIN_DATASET_STDS),
        ToTensorV2()

    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)
)

ALBUMENTATION_VAL = A.Compose(
    [
        A.Resize(MODEL_IMG_SIZE, MODEL_IMG_SIZE),
        # A.HorizontalFlip(always_apply=False, p=0.5),
        # A.VerticalFlip(always_apply=False, p=0.5),
        # A.RandomResizedCrop(always_apply=True, p=0.5, height=MODEL_IMG_SIZE, width=MODEL_IMG_SIZE, scale=(0.6, 1.0), ratio=(1, 1), interpolation=0),
        # A.MotionBlur(always_apply=False, p=0.5, blur_limit=(3, 7), allow_shifted=True),
        # A.RandomGridShuffle(always_apply=False, p=0.2, grid=(2, 2)),
        # A.Downscale(always_apply=False, p=1.0, scale_min=0.75, scale_max=0.99),
        A.Normalize(mean=TRAIN_DATASET_MEANS, std=TRAIN_DATASET_STDS),
        ToTensorV2()

    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)
)

ALBUMENTATION_FINAL_TEST = A.Compose(
    [
        A.CenterCrop(always_apply=False, p=1.0, height=150, width=150),
        A.Resize(MODEL_IMG_SIZE, MODEL_IMG_SIZE),
        # A.RandomResizedCrop(always_apply=True, p=1, height=MODEL_IMG_SIZE, width=MODEL_IMG_SIZE, scale=(
        #     0.3, 1.0), ratio=(1, 1), interpolation=0),
        # A.HorizontalFlip(always_apply=False, p=0.5),
        # A.VerticalFlip(always_apply=False, p=0.5),
        # A.RandomResizedCrop(always_apply=True, p=0.5, height=MODEL_IMG_SIZE, width=MODEL_IMG_SIZE, scale=(0.6, 1.0), ratio=(1, 1), interpolation=0),
        # A.MotionBlur(always_apply=False, p=0.5, blur_limit=(3, 7), allow_shifted=True),
        # A.RandomGridShuffle(always_apply=False, p=0.2, grid=(2, 2)),
        # A.Downscale(always_apply=False, p=1.0, scale_min=0.75, scale_max=0.99),
        A.Normalize(mean=TRAIN_DATASET_MEANS, std=TRAIN_DATASET_STDS),
        ToTensorV2()

    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)
)


ALBUMENTATION_FINAL_TEST_RAW = A.Compose(
    [
        A.CenterCrop(always_apply=False, p=1.0, height=150, width=150),
        # A.Resize(MODEL_IMG_SIZE, MODEL_IMG_SIZE),
        # A.RandomResizedCrop(always_apply=True, p=1, height=MODEL_IMG_SIZE, width=MODEL_IMG_SIZE, scale=(
        #     0.3, 1.0), ratio=(1, 1), interpolation=0),
        # A.HorizontalFlip(always_apply=False, p=0.5),
        # A.VerticalFlip(always_apply=False, p=0.5),
        # A.RandomResizedCrop(always_apply=True, p=0.5, height=MODEL_IMG_SIZE, width=MODEL_IMG_SIZE, scale=(0.6, 1.0), ratio=(1, 1), interpolation=0),
        # A.MotionBlur(always_apply=False, p=0.5, blur_limit=(3, 7), allow_shifted=True),
        # A.RandomGridShuffle(always_apply=False, p=0.2, grid=(2, 2)),
        # A.Downscale(always_apply=False, p=1.0, scale_min=0.75, scale_max=0.99),
        # A.Normalize(mean=TRAIN_DATASET_MEANS, std=TRAIN_DATASET_STDS),
        ToTensorV2()

    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)
)
