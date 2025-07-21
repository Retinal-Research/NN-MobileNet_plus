from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from skimage.io import imread
import skimage.transform as skTrans
from torchvision.io import read_image
from os.path import normpath as fn  # Fixes window/linux path conventions
import warnings
from PIL import Image
import os

warnings.filterwarnings('ignore')

class UWF4DR(Dataset):
    def __init__(self, data_path, transform=None, resize=None):
        super().__init__()
        self.data_path = pd.read_csv(data_path)
        self.transform = transform
        self.resize = resize

    def __getitem__(self, idx):
        img_path = self.data_path.iloc[idx, 0]
        label = self.data_path.iloc[idx, 1]

        # img = read_image(img_path) / 255.
        # img = (img[0, ...] + img[1, ...] + img[2, ...]) / 3
        # img = img.view(1, 584, 565)
        img = Image.open(img_path)
        # img = Image.open(img_path)
        label_onehot = np.zeros(2)
        label_onehot[label] = 1
        if self.transform:
            img = self.transform(img)
        return img, label_onehot, label
        
    def __len__(self):
        return len(self.data_path)