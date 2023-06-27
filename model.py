import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, mc3_18

class ConvClassifier(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.features = nn.Sequential(nn.Conv3d(3,64,kernel_size = 3, padding = 'same'), 
                                      nn.ReLU(inplace=True), 
                                      nn.MaxPool3d(kernel_size=(2,2,2)),
                                      nn.Conv3d(64, 128, kernel_size = 3, padding = 'same'),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool3d(kernel_size=(2,2,2)))
        self.fc_layers = nn.Sequential(nn.Linear(128 * 2 * 64 * 64,512),
                                       nn.ReLU(inplace=True),
                                       nn.Dropout(0.5),
                                       nn.Linear(512,num_classes))
    def forward(self,x):
        x = self.features(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch  
        x = self.fc_layers(x)    
        return x
class ResnetClassifier(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.resnet = r3d_18(weights = 'KINETICS400_V1')
        self.fc_layers = nn.Linear(self.resnet.fc.out_features,num_classes)
    def forward(self,x):
        x = self.resnet(x)
        x = self.fc_layers(x)
        return x
class MixedConvClassifier(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.mc3 = mc3_18(weights = "KINETICS400_V1")
        self.fc_layers = nn.Linear(self.mc3.fc.out_features, num_classes)
    def forward(self,x):
        x = self.mc3(x)
        x = self.fc_layers(x)
        return x

        
