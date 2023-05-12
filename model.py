import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch.optim import Adam
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import  DataLoader,SubsetRandomSampler
from sklearn.model_selection import KFold

batch_size = 5
number_of_labels = 7
classes = ('angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised')

train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_dataset = datasets.ImageFolder('./dataset/train', transform=train_transform)
test_dataset = datasets.ImageFolder('./dataset/test', transform=test_transform)

train_loaded = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loaded = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


k=10
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


class ConvolutionNeuralNetwork(nn.Module):
    def __init__(self, features=96):
        super(ConvolutionNeuralNetwork, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=5, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(12)

        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=5, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(12)

        self.pool = nn.MaxPool2d(2,2)
        self.drop1=nn.Dropout(p=0.5)

        self.conv4 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(24)

        self.conv5 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=5, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(24)
        self.drop2=nn.Dropout(p=0.5)

        self.fc1 = nn.Linear(24*18*18, 96)
        self.drop3=nn.Dropout(p=0.5)

        self.fc2 = nn.Linear(features, 10)

    def forward(self, input):
        output = F.relu(self.bn1(self.conv1(input)))      
        output = F.relu(self.bn2(self.conv2(output)))     
        output = self.pool(output)    
        output = self.drop1(output)   

        output = F.relu(self.bn4(self.conv4(output)))     
        output = F.relu(self.bn5(self.conv5(output)))   
        output = self.drop2(output)  
        output = output.view(-1, 24*18*18)

        output = F.relu(self.fc1(output))
        output = self.drop3(output)
        output = self.fc2(output)

        return output

model = ConvolutionNeuralNetwork()

# Define your execution device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

def saveModel():
    torch.save(model.state_dict(), "apurated_model.pth")

def testAccuracy(test_dataset):
  model.eval()
  accuracy = 0.0
  total = 0.0
  
  with torch.no_grad():
      for data in test_dataset:
          images, labels = data

          images = Variable(images.to(device))
          labels = Variable(labels.to(device))
          
          outputs = model(images)
          # the label with the highest energy will be our prediction
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          accuracy += (predicted == labels).sum().item()
  
  # compute the accuracy over all test images
  accuracy = (100 * accuracy / total)
  return(accuracy)

def train(num_epochs):
    history = {'train_loss': [], 'test_loss': [],'train_acc':[],'test_acc':[]}
    best_accuracy = 0.0
    
    print("The model will be running on", device, "device")
    # Convert model parameters and buffers to CPU or Cuda
    model.to(device)

    for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(train_dataset)))):
        print('Fold {}'.format(fold + 1))
        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(val_idx)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
        test_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=test_sampler)


        for epoch in range(num_epochs):  # loop over the dataset multiple times
            running_loss = 0.0

            for i, (images, labels) in enumerate(train_loader, 0):
                
                # get the inputs
                images = Variable(images.to(device))
                labels = Variable(labels.to(device))
                # zero the parameter gradients
                optimizer.zero_grad()
                # predict classes using images from the training set
                outputs = model(images)
                # compute the loss based on model output and real labels
                loss = loss_fn(outputs, labels)
                # backpropagate the loss
                loss.backward()
                # adjust parameters based on the calculated gradients
                optimizer.step()
                # Let's print statistics for every 1,000 images
                running_loss += loss.item()     # extract the loss value
                if i % 1000 == 999:    
                    # print every 1000 (twice per epoch) 
                    print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / 1000))
                    history['train_loss'].append(running_loss / 1000)
                    # zero the loss
                    running_loss = 0.0

            # Compute and print the average accuracy fo this epoch when tested over all 10000 test images
            accuracy = testAccuracy(test_loader)
            print('For epoch', epoch+1,'the test accuracy over the whole test set is %d %%' % (accuracy))
            
            # we want to save the model if the accuracy is the best
            if accuracy > best_accuracy:
                saveModel()
                best_accuracy = accuracy
            
            
            history['test_acc'].append(accuracy)


def testBatch():
    # get batch of images from the test DataLoader  
    images, labels = next(iter(test_loaded))
    images = Variable(images.to(device))
    labels = Variable(labels.to(device))

    # show all images as one image grid
    imageshow(torchvision.utils.make_grid(images))
   
    # Show the real labels on the screen 
    print('Real labels: ', ' '.join('%5s' % classes[labels[j]] 
                               for j in range(batch_size)))
  
    # Let's see what if the model identifiers the  labels of those example
    outputs = model(images)
    
    # We got the probability for every 10 labels. The highest (max) probability should be correct label
    _, predicted = torch.max(outputs, 1)
    
    # Let's show the predicted labels on the screen to compare with the real ones
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] 
                              for j in range(batch_size)))


def teste_classes():
    model = ConvolutionNeuralNetwork()
    path = "apurated_model2.pth"
    model.load_state_dict(torch.load(path))

    model.eval()

    confusion_matrix = torch.zeros(7, 7)
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loaded):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            for t, p in zip(labels.view(-1), preds.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1

    print(confusion_matrix.diag()/confusion_matrix.sum(1))
    

if __name__ == '__main__':
    train(2)
    print('Finished Training')

    # model = ConvolutionNeuralNetwork()
    # path = "apurated_model2.pth"
    # model.load_state_dict(torch.load(path))
    # teste_classes()

    testBatch()