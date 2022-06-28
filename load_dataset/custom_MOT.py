import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage import io
from torchvision import transforms
import numpy as np

class custom_MOT(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        transform = transforms.Compose([transforms.ToTensor()])

        img_name = os.path.join(self.root_dir, self.data.iloc[idx, 0])
        image = io.imread(img_name)
        label = np.array(self.data.iloc[idx, 1].replace('#', ',').split(',')).astype(float)     # string from csv to 1D array
        label = np.reshape(label, (3, 3))                                                       # reshape array into 3x3 transformation matrix
        image = transform(image)
        sample = [image, label]

        if self.transform:
            sample = self.transform(sample)

        return sample
