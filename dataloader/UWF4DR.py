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
class MURED(Dataset):
    def __init__(self, data_path, transform=None, resize=None):
        super().__init__()
        self.data = pd.read_csv(data_path)
        self.transform = transform
        self.resize = resize

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 0] + '.png'              # first column: image path
        label = self.data.iloc[idx, 1:].to_numpy(dtype='float32')      # next two columns: DR, Glaucoma


        img = Image.open('/scratch/xinli38/data/mured/image/'+img_path)
        if self.transform:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.data)



# ===========================================================================
class MuCaRD(Dataset):
    def __init__(self, data_path, transform=None, resize=None):
        super().__init__()
        self.data = pd.read_csv(data_path)
        self.transform = transform
        self.resize = resize

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 0]              # first column: image path
        labels = self.data.iloc[idx, 1:3].values       # next two columns: DR, Glaucoma

        # Compute mask: 1 for valid, 0 for missing (-1)
        mask = (labels != -1).astype(float)

        # For BCEWithLogitsLoss, missing label (-1) is set to 0 (will be masked out anyway)
        label = np.where(labels == -1, 0, labels).astype(float)

        img = Image.open('./mucard_train/train_all/'+img_path)
        if self.transform:
            img = self.transform(img)

        return img, mask, label

    def __len__(self):
        return len(self.data)
# ===========================================================================

class UWF4DR(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        data = pd.read_csv(label_dir)
        self.image_all = data.iloc[:, 0].values
        self.label_all = data.iloc[:, 1].values

    def __getitem__(self, idx):
        #print(self.image_all[idx])
        image_name = str(self.image_all[idx]) #+ '.png'
        #print(self.data.iloc[idx, 0])
        label = self.label_all[idx]

        image_dir = os.path.join(self.image_dir, image_name)
        x = Image.open(image_dir)

        if self.transform:
            x = self.transform(x)

        label_onehot = np.zeros(2)
        label_onehot[label] = 1

        return x, label_onehot,label

    def get_labels(self):
        return self.label_all

    def __len__(self):
        return len(self.label_all)


# class UWF4DR(Dataset):
#     def __init__(self, image_dir, label_dir, transform=None):
#         self.image_dir = image_dir
#         self.transform = transform

#         # 读CSV，只保留第3列(索引2)有值的
#         data = pd.read_csv(label_dir, header=None)
#         data[2] = pd.to_numeric(data[2], errors='coerce')
#         data = data.dropna(subset=[2]).reset_index(drop=True)
#         data[2] = data[2].astype(int)

#         self.image_all = data.iloc[:, 0].values  
#         self.label_all = data.iloc[:, 2].values   

#     def __getitem__(self, idx):
#         image_name = str(self.image_all[idx])
#         label = int(self.label_all[idx])

#         image_path = os.path.join(self.image_dir, image_name)
#         x = Image.open(image_path).convert('RGB')

#         if self.transform:
#             x = self.transform(x)

#         label_onehot = np.zeros(2, dtype=np.float32)
#         label_onehot[label] = 1.0

#         return x, label_onehot, label

#     def get_labels(self):
#         return self.label_all

#     def __len__(self):
#         return len(self.label_all)


# if __name__ == "__main__":
#     image_dir = "/scratch/xinli38/data/UWF4DR/task2/train"  
#     label_dir = "/scratch/xinli38/data/UWF4DR/task2/train.csv"   

#     dataset = UWF4DR(image_dir, label_dir)
#     print(f"总共有 {len(dataset)} 张有效样本")