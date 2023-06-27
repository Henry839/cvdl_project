# continual learning baseline : without any continuous learning trick
# Train : Test = 2 : 1
from dataset import VideoDataset
from dataset import get_loader
import torch.nn as nn
import torch
label_list = ['soccerball','bear','cows','camel','worm','skiing','skating','dog','car','rat','parkour_boy','blueboy','dressage']
label2id = {label_list[i]:i for i in range(len(label_list))}

dataset_list = [VideoDataset(label) for label in label_list]
# split into train test split
trainLoader_list = []
testLoader_list = []

# hyperparameters
batch_size = 4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
lr = 0.001
weight_decay = 0.01
training_epoch = 20

for dataset in dataset_list:
    train_loader, test_loader = get_loader(dataset, batch_size)
    trainLoader_list.append(train_loader)
    testLoader_list.append(test_loader)
# Basic Convolutional Network
from model import ConvClassifier, ResnetClassifier, MixedConvClassifier
base_model = ConvClassifier(13) # 13 classes for our dataset

# resnet
res_model = ResnetClassifier(13)

# Mixed convolutional 3d network
mix_model = MixedConvClassifier(13)


def train(trainLoader, model):
    optimizer = torch.optim.Adam(model.parameters(),lr = lr, weight_decay = weight_decay)
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    model.train()
    for epoch in range(training_epoch):
        running_loss = 0.0
        for i, data in enumerate(trainLoader):
            inputs, labels = data
            labels = torch.tensor([label2id[label] for label in labels]).to(device)
            inputs = inputs.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        if(epoch % 5 == 0):
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
    train(trainLoader_list[i], base_model)
print("Begin Testing: Base model")
for i in range(len(testLoader_list)):
    print(f"Label : {label_list[i]}")
    test(testLoader_list[i],base_model)

base_model.to('cpu')

# ResNet model
print("Begin Training: ResNet model !")
for i in range(len(trainLoader_list)):
    print(f"Label : {label_list[i]}")
    train(trainLoader_list[i], res_model)
print("Begin Testing: ResNet model")
for i in range(len(testLoader_list)):
    print(f"Label : {label_list[i]}")
    test(testLoader_list[i],res_model)
res_model.to('cpu')



# Mix model
print("Begin Training: Mix model !")
for i in range(len(trainLoader_list)):
    print(f"Label : {label_list[i]}")
    train(trainLoader_list[i], mix_model)
print("Begin Testing: Mix model")
for i in range(len(testLoader_list)):
    print(f"Label : {label_list[i]}")
    test(testLoader_list[i],mix_model)
mix_model.to('cpu')

