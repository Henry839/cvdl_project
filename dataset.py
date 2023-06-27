# Annoncement:
# 1. create dataset for the data
import os
from PIL import Image
from os import listdir
from os.path import join, isdir, isfile
import torchvision.transforms as transforms
from torchvision.io import read_image
from torch.utils.data import Dataset
from tqdm import tqdm
import torch
import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler
class VideoDataset(Dataset):
    def __init__(self, label):

#        label_list = ['blueboy','car','dog','dressage','parkour_boy','rat','skating','skiing','skiing_slalom']
        label_list = []
        label_list.append(label)
        path = f"./dataset/data/"
        gif = []
        label = []
        to_tensor = transforms.ToTensor()
        resize = transforms.Resize([256,256])
        for t in tqdm(label_list):
            # one data one directory
            t_path = join(path,t)
            data = [d for d in listdir(t_path) if isdir(join(t_path,d))]
            for i in data:
                # read all the frames
                i_path = join(t_path,i)
                frames = [to_tensor(resize(Image.open(join(i_path,f)).convert('RGB'))) for f in listdir(i_path) if isfile(join(i_path,f))]
                # store the gif and the label
                gif.append(frames)
                label.append(t)
        self.gif = gif
        self.label = label
    def __len__(self):	
        length = len(self.label)
        if length != len(self.gif):
            print("Data number is not equal as the number of label number, please check your data")
            raise ValueError
        return length
    def __getitem__(self,idx):
        gif = self.gif[idx]
        label = self.label[idx]
        gif = torch.stack(gif, dim=1)
        return gif, label
def get_loader(dataset, batch_size):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    train_split = 0.8
    split = int(np.floor(train_split * dataset_size))

    # Be careful : Don't Shuffle
    train_indices, test_indices = indices[:split], indices[split:]
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                           sampler = train_sampler)

    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                          sampler = test_sampler)
    return train_loader, test_loader
		
		

			
			
	
			

