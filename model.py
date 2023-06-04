import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch.optim import Adam
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import  DataLoader,SubsetRandomSampler,ConcatDataset,Subset
from sklearn.model_selection import KFold
import pandas as pd

batch_size = 10
classes = ('angry', 'fear', 'happy', 'neutral', 'sad', 'surprise','disgust')

train_transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])


train_dataset = datasets.ImageFolder('./color_dataset_2/train', transform=train_transform)

validation_dataset = datasets.ImageFolder('./fer_ckplus_dataset', transform=train_transform)
concat_data = ConcatDataset([train_dataset,validation_dataset])

k=10
splits=KFold(n_splits=k,shuffle=True,random_state=42)
foldperf={}

    
class ConvolutionNeuralNetwork1(nn.Module):
    #acuracia no teste=58.07%
    #acuracia no treino=61.2% || 68.27545615067687 %
    def __init__(self,num_classes=7):
        super(ConvolutionNeuralNetwork1, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=24, kernel_size=5, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(24)

        self.conv2 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=5, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(24)

        self.pool = nn.MaxPool2d(2,2)
        self.drop1=nn.Dropout(p=0.5)

        self.conv4 = nn.Conv2d(in_channels=24, out_channels=48, kernel_size=5, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(48)

        self.conv5 = nn.Conv2d(in_channels=48, out_channels=48, kernel_size=5, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(48)
        self.drop2=nn.Dropout(p=0.5)

        self.fc1 = nn.Linear(48*18*18, 192)
        self.drop3=nn.Dropout(p=0.5)

        self.fc2 = nn.Linear(192, 96)
        self.fc3 = nn.Linear(96, num_classes)


    def forward(self, input):
        output = F.relu(self.bn1(self.conv1(input)))      
        output = F.relu(self.bn2(self.conv2(output)))     
        output = self.pool(output)    
        output = self.drop1(output)   

        output = F.relu(self.bn4(self.conv4(output)))     
        output = F.relu(self.bn5(self.conv5(output)))   
        output = self.drop2(output)  
        output = output.view(-1, 48*18*18)

        output = F.relu(self.fc1(output))
        output = self.drop3(output)
        output = F.relu(self.fc2(output))
        output = self.fc3(output)

        return output

class ConvolutionNeuralNetwork(nn.Module):
    def __init__(self,num_classes=7):
        super(ConvolutionNeuralNetwork, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=24, kernel_size=5, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(24)
        self.conv2 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=5, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(24)

        self.pool1 = nn.MaxPool2d(2,2)

        self.conv3 = nn.Conv2d(in_channels=24, out_channels=48, kernel_size=5, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(48)
        self.conv4 = nn.Conv2d(in_channels=48, out_channels=48, kernel_size=5, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(48)

        self.pool2 = nn.MaxPool2d(2,2)
        self.drop1=nn.Dropout(p=0.2)

        self.conv5 = nn.Conv2d(in_channels=48, out_channels=96, kernel_size=5, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(96)
        self.conv6 = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=4, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(96)

        self.pool3 = nn.MaxPool2d(2,2)

        self.fc1 = nn.Linear(96*5*5, 192)
        self.drop2=nn.Dropout(p=0.2)

        self.fc2 = nn.Linear(192, 96)
        self.fc3 = nn.Linear(96, num_classes)


    def forward(self, input):
        output = F.relu(self.bn1(self.conv1(input)))      
        output = F.relu(self.bn2(self.conv2(output)))     
        output = self.pool1(output)  

        output = F.relu(self.bn3(self.conv3(output)))      
        output = F.relu(self.bn4(self.conv4(output)))     
        output = self.pool2(output)      
        output = self.drop1(output)  

        output = F.relu(self.bn5(self.conv5(output)))      
        output = F.relu(self.bn6(self.conv6(output)))  
        output = self.pool3(output)  

        output = output.reshape(output.size(0),-1)

        output = F.relu(self.fc1(output))
        output = self.drop2(output)
        output = F.relu(self.fc2(output))
        output = self.fc3(output)

        return output
    
learning_rate = 0.001

model = ConvolutionNeuralNetwork()

# Define your execution device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Runing on: "+ ("cuda" if torch.cuda.is_available() else "cpu"))

loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=learning_rate,  weight_decay = 0.001)
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.001, momentum = 0.9)

def saveModel():
    torch.save(model.state_dict(), "apurated_model_mycnn.pth")

def train_epoch(model,device,dataloader):
    train_loss,train_correct=0.0,0
    model.train()
    for images, labels in dataloader:

        images,labels = images.to(device),labels.to(device)

        optimizer.zero_grad()
        output = model(images)
        loss = loss_fn(output,labels)

        loss.backward()
        # accelerator.backward(loss)

        optimizer.step()
        train_loss += loss.item() * images.size(0)
        scores, predictions = torch.max(output.data, 1)
        train_correct += (predictions == labels).sum().item()

    return train_loss,train_correct
  
def valid_epoch(model,device,dataloader):
    valid_loss, val_correct = 0.0, 0
    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:

            images,labels = images.to(device),labels.to(device)
            output = model(images)
            loss=loss_fn(output,labels)
            
            valid_loss+=loss.item()*images.size(0)
            scores, predictions = torch.max(output.data,1)
            val_correct+=(predictions == labels).sum().item()

    return valid_loss,val_correct
def train(num_epochs):
    history = {'train_loss': [], 'test_loss': [],'train_acc':[],'test_acc':[]}
    best_accuracy = 0.0

    model.to(device)
    
    for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(concat_data)))):

        print('Fold {}'.format(fold + 1))

        test_sampler = SubsetRandomSampler(val_idx)
        train_samples =  SubsetRandomSampler(train_idx)

        test_loader = DataLoader(concat_data, batch_size=batch_size, sampler=test_sampler)
        train_loader = DataLoader(concat_data, batch_size=batch_size, sampler=train_samples)

        for epoch in range(num_epochs):
            train_loss, train_correct=train_epoch(model,device,train_loader)
            test_loss, test_correct=valid_epoch(model,device,test_loader)

            train_loss = train_loss / len(train_loader.sampler)
            train_acc = train_correct / len(train_loader.sampler) * 100

            test_loss = test_loss / len(test_loader.sampler)
            test_acc = test_correct / len(test_loader.sampler) * 100

            print("Epoch:{}/{} AVG Training Loss:{:.3f} AVG Test Loss:{:.3f} AVG Training Acc {:.2f} % AVG Test Acc {:.2f} %".format(epoch + 1,
                                                                                                                    num_epochs,
                                                                                                                    train_loss,
                                                                                                                    test_loss,
                                                                                                                    train_acc,
                                                                                                                    test_acc))
            if train_acc > best_accuracy:
                saveModel()
                best_accuracy = train_acc
                print("Best Accuracy:{} %".format(best_accuracy))

            history['train_loss'].append(train_loss)
            history['test_loss'].append(test_loss)
            history['train_acc'].append(train_acc)
            history['test_acc'].append(test_acc)   

    df_history = pd.DataFrame(data=history)
    df_history.to_csv("historic_mycnn.csv", encoding='utf-8', index=False)
            



if __name__ == '__main__':
    torch.cuda.empty_cache()
    print(torch.cuda.memory_summary(device=None, abbreviated=False))
    
    train(30)
    print('Finished Training')

   

    