import os
import PIL
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torchvision

# Set seeds
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)

class TinyData(Dataset):
    def __init__(self, setname):
        self.setname = setname
        assert setname in ['train','val','test']
        
        # Define dataset
        overall_dataset_dir = os.path.join(os.path.join(os.getcwd(),'load_dataset'), 'MOT_data')
        self.selected_dataset_dir = os.path.join(overall_dataset_dir,setname)
        
        self.all_filenames = os.listdir(self.selected_dataset_dir)
        self.all_labels = pd.read_csv(os.path.join(overall_dataset_dir,'MOT_labels.csv'),header=0,index_col=0)
        self.label_meanings = self.all_labels.columns.values.tolist()
    
    def __len__(self):
        return len(self.all_filenames)
        
    def __getitem__(self, idx):
        selected_filename = self.all_filenames[idx]
        imagepil = PIL.Image.open(os.path.join(self.selected_dataset_dir,selected_filename)).convert('RGB')
        
        # convert image to Tensor
        image = torchvision.transforms.ToTensor(imagepil)
        
        # load label
        label = torch.Tensor(self.all_labels.loc[selected_filename,:].values)
        
        sample = {'data':image,
                  'label':label,
                  'img_idx':idx}
        return sample