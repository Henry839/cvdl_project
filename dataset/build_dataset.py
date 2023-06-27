# Announcement:
# 1. 9 types
# 2. {type}:{frame_num}-- {dressage}:3589 {blueboy}:1416 {rat}:2606 {car}:1109 {dog}:891 {parkour}:1578 {skating}:778 {skiing}:692 {skiing-long}:903
# 3. 8 frames / data
import os
from os import listdir
from os.path import isfile, join

root_path = "./"
#type_name = os.listdir("./DAVIS/JPEGImages/480p")
type_name = ['worm']

#type_name = ['blueboy','car','dog','dressage','parkour_boy','rat','skating','skiing','skiing_slalom']
print("Begin")

for t in type_name:
	path = join(root_path,t)
	frames = [f for f in listdir(path) if isfile(join(path, f))]
	frames.sort()
	print(len(frames))
	i = 0 
	while( (8*i + 7) < len(frames)):
		os.makedirs(f"./data/{t}/{i}/")
		for j in range(8*i, 8*i + 8):
			os.rename(join(path,frames[j]),join(f"./data/{t}/{i}",frames[j]))
		i = i + 1
		


