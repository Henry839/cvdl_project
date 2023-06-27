# cvdl_project
This is the repo for cvdl2023 project: Continuous learning of concept in videos
---
## Dataset Statistics:
We collect data from CLVOS23, DAVIS16, SegTrack v2.
Covering 13 domain

| Label | Number|
| --- | --- |
| soccerball | 6 | 
| bear | 10 |
| cows | 13 | 
| camel | 11 |
| worm | 30 | 
| skiing | 86 |
| skating | 97 |
| dog | 111 |
| car | 138|
| rat | 177 |
| parkour_boy | 197 | 
| blueboy | 300 |
| dressage | 448 | 


***Dataset loading method***
```python
from dataset import VideoDataset
dataset = VideoDataset({label})
```



---
* dataset.py: load dataset
* model.py: load model

