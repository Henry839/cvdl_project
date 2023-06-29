# cvdl_project
This is the repo for cvdl2023 project: Continuous learning of concept in videos
---
## Dataset Statistics:
* We collect data from CLVOS23, DAVIS16, SegTrack v2. 

 * Covering **13** types, **3** resource amount level.
 * 8 frames each data
 * $Train : Test = 4 : 1$

| Label | Data Number| Resource Amount Level |
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

* Please change {label} in to corresponding labels mentioned in the table above, such as "bear","cows"...
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
## Experiment result
| Method | model | soccerball | bear | cows | camel | worm | skiing | skating | dog | car | rat | parkour_boy | blueboy | dressage| Avg | 
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |--- |
| Baseline | Conv | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 | 7.69 |
| Baseline | Resnet | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 | 7.69 |
| Baseline | Mixed | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 | 7.69 |
| Loss Function |  |     |     |     |     |     |     |     |     |     |     |     |       |
| Experience-Replay (0.1) | Conv | 100.0    | 100.0    | 100.0    | 0.0    | 100.0    | 77.8    | 95.0    | 43.5    | 100.0    |  88.9   |  100.0   |   78.3    | 100.0 |
| Experience-Replay (0.1) | Resnet | 0.0    | 0.0    | 0.0    | 0.0    | 0.0    | 88.9   | 100.0    | 0.0    | 96.4    |  38.9   |  90.0   |   36.7    | 100.0 |
| Experience-Replay (0.1) | Mixed | 0.0    | 0.0    | 100.0    | 0.0    | 100.0    | 94.4    | 100.0    | 21.7    | 100.0    |  21.7   |  100.0   |   86.1   | 97.5 | 76.7 | 100.0 |
| Dynamic Structure |    |     |     |     |     |     |     |     |     |     |     |     |       |
---
## Experience-Replay
TODO: Draw 3 figures (each model one) of performance vs memory size



---
* dataset.py: load dataset
* model.py: load model

