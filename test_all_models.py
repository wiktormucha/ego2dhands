from utils.testing import batch_epe_calculation, batch_auc_calculation, batch_pck_calculation, show_batch_predictions, save_batch_predictions
from config import *
from models.models import CustomHeatmapsModel, EfficientWaterfall
from datasets.FreiHAND import FreiHAND, FreiHAND_evaluation, FreiHAND_albu
from utils.utils import heatmaps_to_coordinates
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torchvision import transforms
import sys
import torch
from torch.utils.data import DataLoader

sys.path.append("../")


def evaluate(model, dataloader, using_heatmaps=True, batch_size=0):
    accuracy_all = []
    image_id = []
    pred = []
    gt = []
    pck_acc = []
    epe_lst = []
    auc_lst = []

    for data in tqdm(dataloader):
        inputs = data["image"]
        pred_heatmaps = model(inputs)
        pred_heatmaps = pred_heatmaps.detach().numpy()
        true_keypoints = (data["keypoints"]).numpy()

        # heatmaps = data["heatmaps"]
        # print(heatmaps.shape)
        # heatmaps.resize(1,21,MODEL_IMG_SIZE,MODEL_IMG_SIZE)
        # print(heatmaps.shape)
        # true_keypoints = heatmaps_to_coordinates(np.array(heatmaps)) * MODEL_IMG_SIZE

        # keypoints = sample["keypoints"]
        # true_keypoints = keypoints[0] #* MODEL_IMG_SIZE

        if using_heatmaps == True:
            pred_keypoints = heatmaps_to_coordinates(pred_heatmaps)
        else:
            pred_keypoints = pred_heatmaps.reshape(batch_size, N_KEYPOINTS, 2)

        # print(true_keypoints)
        # print(pred_keypoints)
        accuracy_keypoint = ((true_keypoints - pred_keypoints)
                             ** 2).sum(axis=2) ** (1 / 2)
        accuracy_image = accuracy_keypoint.mean(axis=1)
        accuracy_all.extend(list(accuracy_image))

        # Calculate PCK@02
        avg_acc = batch_pck_calculation(
            pred_keypoints, true_keypoints, treshold=0.2, mask=None, normalize=None)
        pck_acc.append(avg_acc)

        # Calculate EPE mean and median, mind that it depends on what scale of input keypoints
        epe = batch_epe_calculation(pred_keypoints, true_keypoints)
        epe_lst.append(epe)

        # AUC calculation
        auc = batch_auc_calculation(
            pred_keypoints, true_keypoints, num_step=20, mask=None)
        auc_lst.append(auc)
        # break

        # if epe > 15:
        #     save_batch_predictions(data, model, epe)

    pck = sum(pck_acc) / len(pck_acc)
    epe_final = sum(epe_lst) / len(epe_lst)
    auc_final = sum(auc_lst) / len(auc_lst)

    # lines = [str(pck), str(epe_final), str(auc_final)]
    # with open('results.txt', 'w') as f:
    #     f.writelines(lines)

    print(f'PCK@2: {pck}, EPE: {epe_final}, AUC: {auc_final}')
    return accuracy_all, pck


model_pths = [
    "/caa/Homes01/wmucha/repos/applied_deep_learning/applied_dl/custom_heatmap_allaug4_30",
    "/caa/Homes01/wmucha/repos/applied_deep_learning/applied_dl/custom_heatmap_allaug4_43",
    "/caa/Homes01/wmucha/repos/applied_deep_learning/applied_dl/custom_heatmap_allaug5_0",
    "/caa/Homes01/wmucha/repos/applied_deep_learning/applied_dl/custom_heatmap_allaug5_11",
    "/caa/Homes01/wmucha/repos/applied_deep_learning/applied_dl/custom_heatmap_allaug5_13",
    "/caa/Homes01/wmucha/repos/applied_deep_learning/applied_dl/custom_heatmap_allaug5_21",
    "/caa/Homes01/wmucha/repos/applied_deep_learning/applied_dl/custom_heatmap_allaug5_26",
    "/caa/Homes01/wmucha/repos/applied_deep_learning/applied_dl/custom_heatmap_allaug5_27",
    "/caa/Homes01/wmucha/repos/applied_deep_learning/applied_dl/custom_heatmap_allaug5_61",
    # "/caa/Homes01/wmucha/repos/applied_deep_learning/applied_dl/waterfall_fulldata_scratch8_0",
    # "/caa/Homes01/wmucha/repos/applied_deep_learning/applied_dl/waterfall_fulldata_scratch8_1",
    # "/caa/Homes01/wmucha/repos/applied_deep_learning/applied_dl/waterfall_fulldata_scratch8_2",
    # "/caa/Homes01/wmucha/repos/applied_deep_learning/applied_dl/waterfall_fulldata_scratch9_0",
    # "/caa/Homes01/wmucha/repos/applied_deep_learning/applied_dl/waterfall_fulldata_scratch9_2",
    # "/caa/Homes01/wmucha/repos/applied_deep_learning/applied_dl/waterfall_fulldata_scratch9_8",
    # "/caa/Homes01/wmucha/repos/applied_deep_learning/applied_dl/waterfall_fulldata_scratch9_21",
    # "/caa/Homes01/wmucha/repos/applied_deep_learning/applied_dl/waterfall_fulldata_scratch9_25",
    # "/caa/Homes01/wmucha/repos/applied_deep_learning/applied_dl/waterfall_fulldata_scratch9_31",
    # "/caa/Homes01/wmucha/repos/applied_deep_learning/applied_dl/waterfall_fulldata_scratch9_32",
    # "/caa/Homes01/wmucha/repos/applied_deep_learning/applied_dl/waterfall_fulldata_scratch9_45",
    # "/caa/Homes01/wmucha/repos/applied_deep_learning/applied_dl/waterfall_fulldata_scratch9_56",
    # "/caa/Homes01/wmucha/repos/applied_deep_learning/applied_dl/waterfall_fulldata_scratch9_64",
    # "/caa/Homes01/wmucha/repos/applied_deep_learning/applied_dl/waterfall_fulldata_scratch9_69",
    # "/caa/Homes01/wmucha/repos/applied_deep_learning/applied_dl/waterfall_fulldata_scratch9_84",
    # "/caa/Homes01/wmucha/repos/applied_deep_learning/applied_dl/waterfall_fulldata_scratch9_86",
    # "/caa/Homes01/wmucha/repos/applied_deep_learning/applied_dl/waterfall_fulldata_scratch9_105",
    # "/caa/Homes01/wmucha/repos/applied_deep_learning/applied_dl/waterfall_fulldata_scratch9_121",
    # "/caa/Homes01/wmucha/repos/applied_deep_learning/applied_dl/waterfall_fulldata_scratch9_134",
    # "/caa/Homes01/wmucha/repos/applied_deep_learning/applied_dl/waterfall_fulldata_scratch10_0",
    # "/caa/Homes01/wmucha/repos/applied_deep_learning/applied_dl/waterfall_fulldata_scratch10_3",
    # "/caa/Homes01/wmucha/repos/applied_deep_learning/applied_dl/waterfall_fulldata_scratch10_6",
    # "/caa/Homes01/wmucha/repos/applied_deep_learning/applied_dl/waterfall_fulldata_scratch10_11",
    # "/caa/Homes01/wmucha/repos/applied_deep_learning/applied_dl/waterfall_fulldata_scratch10_13",
    # "/caa/Homes01/wmucha/repos/applied_deep_learning/applied_dl/waterfall_fulldata_scratch10_18",
    # "/caa/Homes01/wmucha/repos/applied_deep_learning/applied_dl/waterfall_fulldata_scratch10_33",
    # "/caa/Homes01/wmucha/repos/applied_deep_learning/applied_dl/waterfall_fulldata_scratch11_0",
    # "/caa/Homes01/wmucha/repos/applied_deep_learning/applied_dl/waterfall_fulldata_scratch11_6",
    # "/caa/Homes01/wmucha/repos/applied_deep_learning/applied_dl/waterfall_fulldata_scratch11_7",
    # "/caa/Homes01/wmucha/repos/applied_deep_learning/applied_dl/waterfall_fulldata_scratch15_38",
    # "/caa/Homes01/wmucha/repos/applied_deep_learning/applied_dl/waterfall_fulldata_scratch16_47",
    # "/caa/Homes01/wmucha/repos/applied_deep_learning/applied_dl/waterfall_fulldata_scratch16_5",
    # "/caa/Homes01/wmucha/repos/applied_deep_learning/applied_dl/waterfall_fulldata_scratch16_7",
    # "/caa/Homes01/wmucha/repos/applied_deep_learning/applied_dl/waterfall_fulldata_scratch16_41",
    # "/caa/Homes01/wmucha/repos/applied_deep_learning/applied_dl/waterfall_fulldata_scratch16_36",
    # "/caa/Homes01/wmucha/repos/applied_deep_learning/applied_dl/waterfall_fulldata_scratch16_47",

]

config = {
    "data_dir": "/data/wmucha/datasets/FreiHAND",
    "model_path": 'asdf',
    "test_batch_size": 1,
    "device": TESTING_DEVICE
}

# val_img_transform = val_image_transform = transforms.Compose(
#     [
#         transforms.Resize(MODEL_IMG_SIZE),
#         transforms.ToTensor(),
#         transforms.Normalize(
#             mean=TRAIN_DATASET_MEANS, std=TRAIN_DATASET_STDS)
#     ]
# )

final_evaluation = FreiHAND_evaluation(
    config, albumetations=ALBUMENTATION_VAL)


final_evaluation_dataloader = DataLoader(
    final_evaluation,
    config["test_batch_size"],
    shuffle=False,
    drop_last=False,
    num_workers=2,
)


test_dataset = FreiHAND_albu(
    config=config, set_type="test", albumetations=ALBUMENTATION_VAL)

test_dataloader = DataLoader(
    test_dataset,
    config["test_batch_size"],
    shuffle=False,
    drop_last=False,
    num_workers=2,
)


for model_pth in model_pths:

    model = CustomHeatmapsModel(3, 21)
    # model = EfficientWaterfall(N_KEYPOINTS)
    model.load_state_dict(
        torch.load(model_pth, map_location=torch.device(config["device"]))
    )
    model.eval()

    print(model_pth)
    accuracy_all, pck = evaluate(model, test_dataloader)
    accuracy_all, pck = evaluate(model, final_evaluation_dataloader)
# print('Hello Word')
