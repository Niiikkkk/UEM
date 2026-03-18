import os
import argparse

import numpy as np
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import average_precision_score, roc_curve, auc
from torch.utils.data import DataLoader
from tqdm import tqdm

from modeling.ood_segmentation import OoDSegmentationModel
from easydict import EasyDict as edict

from tools import overwrite_config
from pytorch_lightning import Trainer
import torch
import albumentations as A

def evaluate_single_ckpt(ckpt, args, opts):
    """
    opts is a list of even size containing the hyperparameters to overwrite
    the model config. even value indices are the keys, and odd value indices
    are the values.
    """
    print(f"Evaluating {ckpt}")

    segmentor_ckpt = None
    if args.segmentor_ckpt is not None:
        segmentor_ckpt = args.segmentor_ckpt

    model = OoDSegmentationModel.load_from_checkpoint(
        ckpt, segmentor_ckpt=segmentor_ckpt)

    print(opts)

    model.save_hyperparameters(overwrite_config(model.hparams, opts))

    from datamodules.datasets.carla_anomaly import CarlaAnomaly

    transforms = A.Compose(
        [
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    hparams = edict(
        dataset_root=os.path.join(
            "/home/nicholas/Desktop/main_UE4/output_anomaly/"
        )
    )
    dataset = CarlaAnomaly(
        hparams=hparams,
        transforms=transforms
    )
    torch.random.manual_seed(42)
    loader = DataLoader(dataset,shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    _ = model.eval()

    idd = []
    odd = []
    lrr = []
    gt = []
    j=0
    MAX_STEP = 3000
    for x,y in tqdm(loader):

        if j>MAX_STEP:
            break
        j+=1
        x=x.to(device)
        y=y.to(device)

        with torch.no_grad():
            output = model.sliding_window_inference(x,y.shape,30,[308,406],[140,140],True,True)

            ood_score = output.ood_score[:, 1]
            idd_score_under_ood = output.ood_score[:, 0]

            idd_score = output.sem_seg.max(dim=1)[0]
            lr_score = ood_score - idd_score - idd_score_under_ood

            idd.extend(-idd_score.cpu().numpy())
            odd.extend(ood_score.cpu().numpy())
            lrr.extend(lr_score.cpu().numpy())
            gt.extend(y.cpu().numpy())

    #print("Evaluating IDD...")
    #evaluate_ood(idd,gt)
    print("Evaluating ODD...")
    evaluate_ood(odd,gt)
    print("Evaluating LRR...")
    evaluate_ood(lrr,gt)

def evaluate_ood(anomaly_score, ood_gts, verbose=True):

    anomaly_score = np.array(anomaly_score)
    ood_gts = np.array(ood_gts)

    in_mask = ood_gts < 30

    result = {}

    for cls in ["all"]:
        if cls == "all":
            mask = ood_gts >= 30
        elif cls == "pothole":
            mask = (ood_gts == 30)
        elif cls == "tiny":
            mask = (ood_gts == 33)
        elif cls == "small":
            mask = (ood_gts == 34)
        elif cls == "medium":
            mask = (ood_gts == 35)
        elif cls == "large":
            mask = (ood_gts == 36)

        combined_mask = mask | in_mask

        ood_out = anomaly_score[mask]
        ind_out = anomaly_score[in_mask]

        ood_label = np.ones(len(ood_out))
        ind_label = np.zeros(len(ind_out))

        val_out = np.concatenate((ind_out, ood_out))
        val_label = np.concatenate((ind_label, ood_label))

        if verbose:
            print(f"Calculating Metrics for {len(val_out)} Points for class {cls}...")
            print(f"Number of OOD points: {len(ood_out)}, Number of ID points: {len(ind_out)}")

        auroc, aupr, fpr = calculate_ood_metrics(val_out, val_label)

        if verbose:
            print(f'Class {cls} - AUROC score: {auroc}')
            print(f'Class {cls} - AUPRC score: {aupr}')
            print(f'Class {cls} - FPR@TPR95: {fpr}')

        result[cls] = {
            "AUROC": auroc,
            "AUPR": aupr,
            "FPR95": fpr
        }

    return result

def calculate_ood_metrics(out, label):

    # fpr, tpr, _ = roc_curve(label, out)

    prc_auc = average_precision_score(label, out)
    roc_auc, fpr, _ = calculate_auroc(out, label)
    # roc_auc = auc(fpr, tpr)
    # fpr = fpr_at_95_tpr(out, label)

    return roc_auc, prc_auc, fpr

def calculate_auroc(conf, gt):
    fpr, tpr, threshold = roc_curve(gt, conf)
    roc_auc = auc(fpr, tpr)
    fpr_best = 0
    # print('Started FPR search.')
    for i, j, k in zip(tpr, fpr, threshold):
        if i > 0.95:
            fpr_best = j
            break
    # print(k)
    return roc_auc, fpr_best, k

def write_results(results, args):
    """
    Expected hierarchy of the results dictionary
    - model name
        - dataset name
            - metric name
    """

    if args.out_path is not None:
        # if args.out_path doesn't exist create it
        if not os.path.exists(args.out_path):
            os.makedirs(args.out_path, exist_ok=True)
    
        model_folder = '/'.join(args.ckpt.split("/")[:-1])
        model_name = args.ckpt.split("/")[-1].split(".")[0]
        out_path = os.path.join(model_folder, model_name + "_results")

        # in the first row of the output, leave the first column empty for the checkpoint name
        # and print the metrics in the order given in the argument
        with open(f"{out_path}.txt", "w") as f:
            f.write("\t")
            dataset_list = list(results[list(results.keys())[0]].keys())
            for dataset in dataset_list:
                for metric in args.metrics_order.split(","):
                    f.write(f"{dataset}_{metric}\t")
            f.write("\n")
            for ckpt, ckpt_results in results.items():
                f.write(f"{ckpt}\t")
                for dataset, dataset_results in ckpt_results.items():
                    for metric in args.metrics_order.split(","):
                        f.write(f"{100 * dataset_results[metric]:.4f}\t")
                f.write("\n")


def main(args):

    if args.multiple_datasets:
        datasets = args.dataset.split(",")
    else:
        datasets = [args.dataset]

    results = edict()
    ckpt_name = args.ckpt.split("/")[-1]
    results[ckpt_name] = edict()
    opts = ["DATA.DATASETS_FOLDER", args.datasets_folder]
    opts = []

    if args.segmentor_ckpt is not None:
        opts.extend(["MODEL.SEGMENTOR_CKPT", args.segmentor_ckpt])

    for dataset in datasets:
        results[ckpt_name][dataset] = evaluate_single_ckpt(
            args.ckpt,
            args,
            opts + ["DATA.EVAL_DATASET", dataset]
        )

    #write_results(results, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate OoD Metrics")

    parser.add_argument(
        "--ckpt",
        type=str,
        help="Path to the model checkpoint"
    )
    parser.add_argument(
        "--segmentor-ckpt",
        type=str,
        default=None,
        help="Path to the segmentation model checkpoint, if none then use the checkpoint stored in the model config"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="road_anomaly",
        help="Dataset to evaluate"
    )
    parser.add_argument(
        "--multiple-datasets",
        action="store_true",
        help="Evaluate multiple datasets, names given separated by commas in --dataset argument"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--devices",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--out-path",
        type=str,
        default=None,
        help="Path to save the results as a text file"
    )
    parser.add_argument(
        "--metrics-order",
        type=str,
        default="llr_AUPR,llr_AUROC,llr_FPR95",
    )
    parser.add_argument(
        "--datasets-folder",
        type=str,
        default=None,
        help="Path to the evaluation datasets folder"
    )
    parser.add_argument(
        "--store-anomaly-scores",
        action="store_true",
        help="Store anomaly scores in the output"
    )

    args = parser.parse_args()

    main(args)
