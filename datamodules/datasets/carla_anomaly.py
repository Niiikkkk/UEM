import os
import cv2
import glob

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import resize, InterpolationMode


def round_to_nearest_multiple(x, p):
    return int(((x - 1) // p + 1) * p)


class CarlaAnomaly(Dataset):

    def __init__(self, hparams, transforms=None):
        super().__init__()

        self.hparams = hparams
        self.transforms = transforms

        weather = os.listdir(hparams.dataset_root)

        # Weather condition: Foggy  HeavyFog  HeavyRain  HeavyRainFog  Overcast  Rainy  Sunny
        weather_filter = ["All"]
        if "All" not in weather_filter:
            weather = [w for w in weather if w in weather_filter]

        if "anomaly_sizes.txt"  in weather:
            weather.remove("anomaly_sizes.txt")

        print("OOD Testing on anomaly weather conditions: ", weather)

        self.images = []
        self.labels = []

        for w in weather:
            if w in ["anomaly_sizes.txt"]:
                continue
            dirs = os.listdir(os.path.join(hparams.dataset_root, w))
            for dir_ in dirs:
                self.images.extend(glob.glob(os.path.join(hparams.dataset_root, w,  dir_, "rgb", '*.png'))[50:])
                self.labels.extend(glob.glob(os.path.join(hparams.dataset_root, w,  dir_, "semantic/original", '*.png'))[50:])

        self.num_samples = len(self.images)

    def __getitem__(self, index):

        image = self.read_image(self.images[index])
        label = self.read_image(self.labels[index])

        label = label[..., 0]

        #label = np.where((label >= 0) & (label <= 29), 0, label)
        #label = np.where((label >= 30) & (label <= 36), 1, label)

        label = np.where((label == 32), 35, label)

        if self.transforms:
            aug = self.transforms(image=image, mask=label)
            image = aug['image']
            label = aug['mask']


        return image, label.type(torch.LongTensor)

    @staticmethod
    def read_image(path):

        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

        return img

    def __len__(self):
        return self.num_samples