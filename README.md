# cvdl_project
This is the repo for cvdl2023 project: Continuous learning of concept in videos
---
## Dataset Statistics:
* We collect data from CLVOS23, DAVIS16, SegTrack v2. 

 * Covering **13** types, **3** resource amount level.
 * 8 frames each data

| Label | Number| Resource Amount Level |
| --- | --- | --- |
| soccerball | 6 | low |
| bear | 10 | low |
| cows | 13 | low |
| camel | 11 | low |
| worm | 30 | low |
| skiing | 86 | middle |
| skating | 97 | middle |
| dog | 111 | middle |
| car | 138| middle |
| rat | 177 | middle |
| parkour_boy | 197 | middle |
| blueboy | 300 | hight |
| dressage | 448 |  high |


***Dataset loading method***
Please change {label} in to corresponding labels mentioned in the table above, such as "bear","cows"...
```python
from dataset import VideoDataset
dataset = VideoDataset({label})
```
---
## Model
* Covering **3** kinds of models: Basic Convolutional Network, Resnet3d, Mixed Convolutional 3d Network
```python
# Basic Convolutional Network
from model import ConvClassifier
model = ConvClassifier(13) # 13 classes for our dataset

# Resnet3d
from model import ResnetClassifier
model = ResnetClassifier(13)

# Mixed convolutional 3d network
model = MixedConvClassifier(13)
```


---
* dataset.py: load dataset
* model.py: load model

