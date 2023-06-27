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
class CLVOS23_dataset(Dataset):
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
	
		
		

			
			
	
			


