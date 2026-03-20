
import argparse
import torch
from albumentations.pytorch import ToTensorV2

from modeling.segmentation import SegmentationModel
from easydict import EasyDict as edict
from datamodules import SemanticSegmentationDataModule
from pytorch_lightning import Trainer
from tools import overwrite_config
import numpy as np
from PIL import Image
import albumentations as A
import torch.nn.functional as F

SEGMENTATION_DATASETS = ["cityscapes", "carla"]


def get_datamodule(args, hparams):

    if args.dataset == "cityscapes":

        datamodules = SemanticSegmentationDataModule(hparams)
        return datamodules

    elif args.dataset == "carla":
        datamodules = SemanticSegmentationDataModule(hparams)
        return datamodules
    else:
        raise ValueError(f"Undefined datamodule: {args.dataset}")

def seg(model):
    from torchmetrics import JaccardIndex

    iou_metric = JaccardIndex(
        task="multiclass",
        num_classes=30,
        ignore_index=255,
    )

    transform = A.Compose(
        [
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
    img_path = "/home/nicholas/Desktop/main_UE4/output_normal/HeavyRainFog/1/rgb/normal__292783.png"
    image = np.array(Image.open(img_path).convert('RGB'))
    label = np.array(Image.open(img_path.replace("rgb", "semantic/original")))
    label = label[:, :, 0]
    aut = transform(image=image, mask=label)
    image, label = aut['image'], aut['mask']
    image = image.unsqueeze(0)
    label = label.unsqueeze(0)
    print(image.shape)
    out = model.sliding_window_inference(image, label.shape, [308, 406], [140, 140])

    print(iou_metric(out, label))

    pred = out.squeeze()
    pred = torch.argmax(pred, dim=0)

    colors = [
        [0, 0, 0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [70, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
        [110, 190, 160],
        [170, 120, 50],
        [55, 90, 80],
        [45, 60, 150],
        [157, 234, 50],
        [81, 0, 81],
        [150, 100, 100],
        [230, 150, 140],
        [180, 165, 180],
        [180, 130, 70]
    ]

    color = get_seg_colormap(pred, colors)
    print(color.shape)

    Image.fromarray(np.uint8(color.cpu().numpy())).show()
    exit()

def main(args):

    # load config from ckpt
    ckpt = torch.load(args.ckpt)
    hparams = edict(ckpt["hyper_parameters"])
    state_dict = ckpt["state_dict"]
    model = SegmentationModel(hparams)
    model.load_state_dict(state_dict)

    hparams = overwrite_config(hparams, args.opts)

    # load datamodule

    datamodule = get_datamodule(args, hparams)

    seg(model)

    devices = 1
    if args.devices is not None:
        devices = [int(d) for d in args.devices.split(",")]
    output = Trainer(devices=devices).test(model, datamodule=datamodule)

    print(output)


def get_seg_colormap(preds, colors):
    """
    Assuming preds.shape = (H,W)
    """
    H, W = preds.shape
    color_map = torch.zeros((H, W, 3)).long()

    for i in range(len(colors)):
        mask = (preds == i)
        if mask.sum() == 0:
            continue
        color_map[mask, :] = torch.tensor(colors[i])

    return color_map

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a segmentation model")
    parser.add_argument(
        "--ckpt",
        type=str,
        help="Path to the segmentor model checkpoint"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="",
        help="Name of evaluation dataset"
    )
    parser.add_argument(
        "--devices",
        type=str,
        default=None,
        help="Devices to run the evaluation on"
    )

    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options using the command-line"
    )

    args = parser.parse_args()

    main(args)
