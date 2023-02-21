from config import *
from utils.egocentric import run_model_on_hands, preds_to_full_image
from datasets.h2o import H2O_Dataset
from models.models import CustomHeatmapsModel, EfficientWaterfall
from utils.hand_detector import get_hands_img
import torch
import sys
import matplotlib.pyplot as plt
import mediapipe as mp
import copy as copy
sys.path.append("../")


IMAGE_N = 10  # Index of image to see


def main(config):
    """Main function to run egocentric prediction

    Args:
        config (_type_): Dictionatry containing config information
    """

    dataset = H2O_Dataset(config)

    hand_model = mp.solutions.hands.Hands()

    model = EfficientWaterfall(N_KEYPOINTS)
    # model = CustomHeatmapsModel(3, 21)
    model.load_state_dict(
        torch.load(config["model_path"],
                   map_location=torch.device(config["device"]))
    )
    model.eval()
    print("Model loaded")

    for idx in range(530, 700, 5):
        print(idx)

        item = dataset[idx]

        img = item['image_raw']

        hand_pose = item['hand_pose']
        cam_instr = item['cam_instr']

        hands_dict = get_hands_img(
            img, hand_pose, cam_instr=cam_instr, hand_model=hand_model)

        imgs = hands_dict['hands_seg']

        pred = run_model_on_hands(model, imgs)

        scale = [imgs[0].size[0], imgs[1].size[0]]

        full_scale_preds = preds_to_full_image(
            predictions=pred, hands_bb=hands_dict['hands_bb'], scale=scale)

        pts = full_scale_preds[0]
        plt.scatter(pts[:, 0], pts[:, 1], c="k", alpha=0.5, s=8)
        for finger, params in COLORMAP.items():
            plt.plot(
                pts[params["ids"], 0],
                pts[params["ids"], 1],
                params["color"],
            )

        pts = full_scale_preds[1]
        plt.scatter(pts[:, 0], pts[:, 1], c="k", alpha=0.5, s=8)
        for finger, params in COLORMAP.items():
            plt.plot(
                pts[params["ids"], 0],
                pts[params["ids"], 1],
                params["color"],
            )

        plt.imshow(img)
        img_name = 'egocentric_prediction_'+str(idx)
        # plt.savefig(img_name)
        plt.axis('off')
        plt.savefig(img_name, transparent=True,
                    bbox_inches='tight', pad_inches=0)
        print('File saved..')
        plt.clf()


if __name__ == "__main__":

    config = {'device': 3,
              'data_dir': '/data/wmucha/datasets/h2o/h2o_CASA',
              "model_path": "/caa/Homes01/wmucha/repos/applied_deep_learning/applied_dl/waterfall_fulldata_scratch9_121",
              # "model_path": "/caa/Homes01/wmucha/repos/applied_deep_learning/applied_dl/albumentations_con12_final",
              }

    main(config=config)
