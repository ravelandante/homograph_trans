import os
import pandas as pd
from torch.utils.data import Dataset
from skimage import io
from torchvision import transforms
import numpy as np
from numpy import genfromtxt
from homo_transform import img_transform


FRAMES = 70                                 # num of frames to generate (-1 to process all)

class custom_MOT(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.csv_file = csv_file
        self.root_dir = root_dir
        self.transform = transform

        s = self.root_dir.split('/')
        s.remove('img1')
        self.dataset_dir = '/'.join(s)

        custom_MOT.populate(self)                                           # populate csv file with filenames + object IDs
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def populate(self):
        """
        Populate corresponding CSV file (train, test, val) with image
        names and object IDs in form: Filename,ObjectID
        """
        gt = genfromtxt(self.dataset_dir + '/gt/gt.txt', delimiter=',')     # open, organise ground-truth tracking data
        gt = np.split(gt, np.where(np.diff(gt[:,0]))[0]+1)                  # split by frame number

        for i, _ in enumerate(os.listdir(self.root_dir)):                   # loop through frames
            for row in gt[i]:                                               # loop through detections in frame
                obj_ID = row[1]

                with open(self.csv_file, 'a') as f:                         # write filenames to csv file for dataloader use
                    if i == 0 and obj_ID == 1:
                        f.truncate(17)
                    f.write('\n{:06}.jpg,{}'.format(i + 1, int(obj_ID)))
            if i + 1 == FRAMES:
                break

    def __getitem__(self, idx):
        transform = transforms.Compose([transforms.ToTensor()])

        img_name = os.path.join(self.root_dir, str(self.data.iloc[idx, 0]))                 # get image path
        obj_ID = os.path.join(str(self.data.iloc[idx, 1]))                                  # get ID of object requested

        gt = genfromtxt(self.dataset_dir + '/gt/gt.txt', delimiter=',')                     # open/organise ground-truth tracking data
        gt = np.split(gt, np.where(np.diff(gt[:,0]))[0]+1)                                  # split by frame number

        frame_num = int(self.data.iloc[idx, 0][0:-4]) - 1                                   # get frame number from path
        img_orig = io.imread(img_name)
        
        height, width, _ = img_orig.shape
        line_num = 0
        for i, obj in enumerate(gt[frame_num]):                                             # find object ID in lines for corresponding frame
            if int(obj[1]) == int(obj_ID):
                line_num = i
                break
        img_warp, label = img_transform(img_orig, gt[frame_num][line_num], width, height)   # get warped image

        image = transform(img_warp)
        sample = [image, label]

        if self.transform:
            sample = self.transform(sample)

        return sample

