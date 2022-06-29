import os
import pandas as pd
from torch.utils.data import Dataset
from skimage import io
from torchvision import transforms
import numpy as np
from numpy import genfromtxt
from homo_transform import transform

SET_NAME = 'ADL-Rundle-6'                   # name of dataset
BASE_PATH = 'data/MOT15/train/' + SET_NAME

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
        obj_ID = os.path.join(self.root_dir, self.data.iloc[idx, 1])

        gt = genfromtxt(BASE_PATH + '/gt/gt.txt', delimiter=',')                                # open/organise ground-truth tracking data
        gt = np.split(gt, np.where(np.diff(gt[:,0]))[0]+1)

        frame_num = int(img_name[0:-4]) - 1
        img_orig = io.imread(img_name)
        for i, obj in enumerate(gt[frame_num]):
            if obj[1] == obj_ID:
                line_num = i
        img_warp, label = transform(img_orig, gt[frame_num][line_num])

        image = transform(img_warp)
        sample = [image, label]

        if self.transform:
            sample = self.transform(sample)

        return sample
