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
number_of_labels = 7
classes = ('angry', 'fear', 'happy', 'neutral', 'sad', 'surprise','disgust')

train_transform = transforms.Compose([
    # transforms.Resize((227,227)), #change to alexnet use
    transforms.Resize((48,48)), #change to alexnet use
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

test_transform = transforms.Compose([
    # transforms.Resize((227,227)),
    transforms.Resize((48,48)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

train_dataset = datasets.ImageFolder('./color_dataset_2/train', transform=train_transform)
# train_dataset2 = datasets.ImageFolder('./fer_ckplus_dataset', transform=train_transform)
# train_dataset = ConcatDataset([train_dataset1,train_dataset2])

test_dataset = datasets.ImageFolder('./color_dataset_2/test', transform=test_transform)

train_loaded = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loaded = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


k=15
splits=KFold(n_splits=k,shuffle=True,random_state=42)
foldperf={}


def imageshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# imagest, labelst = next(iter(train_loaded))

# # show all images as one image grid
# imageshow(torchvision.utils.make_grid(imagest))

class AlexNet(nn.Module):
    #acuracia no teste=62.52% || 63.08%
    #acuracia no treino=81.46% || 86.80%
    def __init__(self):
        super(AlexNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=96, kernel_size=10, stride=4, padding=1)
        self.bn1 = nn.BatchNorm2d(96)
        self.pool1= nn.MaxPool2d(3,2)

        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d(3,2)

        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(384)

        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(384)

        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(3,2)

        self.fc1 = nn.Linear(9216, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, number_of_labels)


    def forward(self, input):
        output = F.relu(self.bn1(self.conv1(input)))   
        output = self.pool1(output)     
        output = F.relu(self.bn2(self.conv2(output)))     
        output = self.pool2(output)    

        output = F.relu(self.bn3(self.conv3(output)))     
        output = F.relu(self.bn4(self.conv4(output)))   
        output = F.relu(self.bn5(self.conv5(output))) 
        output = self.pool3(output)   

        output = output.reshape(output.size(0),-1)

        output = F.relu(self.fc1(output))
        output = F.relu(self.fc2(output))
        output = self.fc3(output)

        return output

class FacialExpressionAlexNet(nn.Module):
    def __init__(self, num_classes=7):
        super(FacialExpressionAlexNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=2)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(6 * 6 * 256, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 3, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 3, stride=2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.max_pool2d(x, 3, stride=2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    
class ConvolutionNeuralNetwork(nn.Module):
    #acuracia no teste=58.07%
    #acuracia no treino=61.2%
    def __init__(self):
        super(ConvolutionNeuralNetwork, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=24, kernel_size=5, stride=1, padding=1)
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
        self.fc3 = nn.Linear(96, number_of_labels)


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


learning_rate = 0.001

model = ConvolutionNeuralNetwork()

# Define your execution device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Runing on: "+ ("cuda" if torch.cuda.is_available() else "cpu"))

loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=learning_rate,  weight_decay = 0.0001)
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.001, momentum = 0.9)

def saveModel():
    torch.save(model.state_dict(), "apurated_model_fercnn.pth")


def train_epoch(model,device,dataloader):
    train_loss,train_correct=0.0,0
    model.train()
    for images, labels in dataloader:

        images,labels = images.to(device),labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = loss_fn(output,labels)
        loss.backward()
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
    
    for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(train_dataset)))):

        print('Fold {}'.format(fold + 1))

        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(val_idx)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
        test_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=test_sampler)
        
        model.to(device)

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
            if test_acc > best_accuracy and epoch>0:
                saveModel()
                best_accuracy = test_acc
                print("Best Accuracy:{} %".format(best_accuracy))

            history['train_loss'].append(train_loss)
            history['test_loss'].append(test_loss)
            history['train_acc'].append(train_acc)
            history['test_acc'].append(test_acc)   

    df_history = pd.DataFrame(data=history)
    df_history.to_csv("historic_fercnn.csv", encoding='utf-8', index=False)
            


def testBatch(model,device):
    model.to(device)
    model.eval()
    # get batch of images from the test DataLoader  
    images, labels = next(iter(test_loaded))
   
    # show all images as one image grid
    img = torchvision.utils.make_grid(images)     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
   
    # Show the real labels on the screen 
    print('Real labels: ', ' '.join('%5s' % classes[labels[j]] 
                               for j in range(1)))
  
    with torch.no_grad():
        images, labels = images.to(device),labels.to(device)
        output = model(images)
        scores, predictions = torch.max(output.data,1)
        
        # Let's show the predicted labels on the screen to compare with the real ones
        print('Predicted: ', ' '.join('%5s' % classes[predictions[j]] 
                                for j in range(1)))
    
    



if __name__ == '__main__':
    train(5)
    print('Finished Training')

    # model = ConvolutionNeuralNetwork()
    # path = "apurated_model_cnn.pth"
    # model.load_state_dict(torch.load(path))

    # testBatch(model,device)

    # test_loss,test_correct=valid_epoch(model,device,test_loaded)
    # test_loss = test_loss / len(test_loaded.sampler)
    # test_acc = test_correct / len(test_loaded.sampler) * 100
    # print(" AVG Test Loss:{:.3f} AVG Test Acc {:.2f} %".format( test_loss,test_acc))

    # imagest, labelst = next(iter(test_loaded))

    # # # show all images as one image grid
    # imageshow(torchvision.utils.make_grid(imagest[0]))