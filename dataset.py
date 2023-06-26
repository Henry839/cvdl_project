# Annoncement:
# 1. create dataset for the data
import os
from os import listdir
from os.path import join, isdir
from torchvision.io import read_image
from torch.utils.data import Dataset
from tqdm import tqdm
class CLVOS23_dataset(Dataset):
	def __init__(self,transform=None):
		label_list = ['blueboy','car','dog','dressage','parkour_boy','rat','skating','skiing','skiing_slalom']
		path = f"./dataset/data/"
		gif = []
		label = []
		for t in tqdm(label_list):
			# one data one directory
			t_path = join(path,t)
			data = [d for d in listdir(t_path) if isdir(join(t_path,d))]

			for i in data:
				# read all the frames
				i_path = join(t_path,i)
				frames = [read_image(join(i_path,f)) for f in listdir(i_path) if isfile(join(i_path,f))]
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
		if self.transform:
			gif = [self.transform(frame) for frame in gif]
		return gif, label
	
		
		

			
			
	
			


