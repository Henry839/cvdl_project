# Annoucenment:
# 1. This is the file for experience replay for video
from dataset import VideoDataset
from dataset import get_loader
import torch.nn as nn
import torch
label_list = ['soccerball','bear','cows','camel','worm','skiing','skating','dog','car','rat','parkour_boy','blueboy','dressage']
label2id = {label_list[i]:i for i in range(len(label_list))}
trainLoader_list = []
testLoader_list = []


dataset_list = [VideoDataset(label) for label in label_list]
batch_size = 4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
lr = 0.001
weight_decay = 0.01
training_epoch = 20


for dataset in dataset_list:
    train_loader, test_loader = get_loader(dataset, batch_size)
    trainLoader_list.append(train_loader)
    testLoader_list.append(test_loader)
# build new trainLoader_list:
import random
import math
memory_list = []
type_split = []
num = 0
for trainLoader in trainLoader_list:
    length = len(trainLoader)
    add_num = math.ceil(length / 10)
    i = 0
    for data in trainLoader:
        i = i + 1
        num = num + 1
        memory_list.append(data)
        if i == add_num:
            type_split.append(num - 1)
            break

# Basic Convolutional Network
from model import ConvClassifier, ResnetClassifier, MixedConvClassifier
base_model = ConvClassifier(13) # 13 classes for our dataset

# resnet
res_model = ResnetClassifier(13)

# Mixed convolutional 3d network
mix_model = MixedConvClassifier(13)
from random import randint
def train(trainLoader, model, type_num):
    optimizer = torch.optim.Adam(model.parameters(),lr = lr, weight_decay = weight_decay)
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    model.train()
    memory = None
    if type_num >= 1:
        type_split_num = type_split[type_num - 1]
        memory = memory_list[0:type_split_num + 1]
    for epoch in range(training_epoch):
        running_loss = 0.0
        for i, data in enumerate(trainLoader):
            inputs, labels = data
            # sample from memory
            if memory :
                sample = randint(0, len(memory) - 1)
                m = memory[sample]
                inputs = torch.cat((inputs, m[0]))
                labels = labels + m[1]
            inputs = inputs.to(device)
            labels = torch.tensor([label2id[label] for label in labels]).to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        if(epoch  % 5 == 0 or epoch == (training_epoch - 1)):
            print(f"Epoch {epoch} loss : {running_loss}")
def test(testLoader, model):
    model.eval()
    total = 0
    correct = 0
    label = ''
    with torch.no_grad():
        for data in testLoader:
            inputs, labels = data
            inputs = inputs.to(device)
            label = labels[0]
            labels = torch.tensor([label2id[label] for label in labels]).to(device)
            
            outputs = model(inputs)
            _, predict = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predict == labels).sum().item()
    print(f"{label} : Accuracy is {100 * correct / total}%")

# Base model
print("Begin Training: Base model !")
for i in range(len(trainLoader_list)):
    print(f"Label : {label_list[i]}")
    train(trainLoader_list[i], base_model,i)
print("Begin Testing: Base model")
for i in range(len(testLoader_list)):
    print(f"Label : {label_list[i]}")
    test(testLoader_list[i],base_model)

base_model.to('cpu')

# ResNet model
print("Begin Training: ResNet model !")
for i in range(len(trainLoader_list)):
    print(f"Label : {label_list[i]}")
    train(trainLoader_list[i], res_model,i)
print("Begin Testing: ResNet model")
for i in range(len(testLoader_list)):
    print(f"Label : {label_list[i]}")
    test(testLoader_list[i],res_model)
res_model.to('cpu')



# Mix model
print("Begin Training: Mix model !")
for i in range(len(trainLoader_list)):
    print(f"Label : {label_list[i]}")
    train(trainLoader_list[i], mix_model,i)
print("Begin Testing: Mix model")
for i in range(len(testLoader_list)):
    print(f"Label : {label_list[i]}")
    test(testLoader_list[i],mix_model)
mix_model.to('cpu')

