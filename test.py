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
from model import ConvolutionNeuralNetwork,loss_fn

batch_size = 10
classes = ('angry', 'fear', 'happy', 'neutral', 'sad', 'surprise','disgust')
test_transform = transforms.Compose([
    transforms.Resize((48,48)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])

test_dataset = datasets.ImageFolder('./color_dataset_2/test', transform=test_transform)
test_loaded = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


def valid(model,dataloader):
    valid_loss, val_correct = 0.0, 0
    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:

            output = model(images)
            loss=loss_fn(output,labels)
            valid_loss+=loss.item()*images.size(0)
            scores, predictions = torch.max(output.data,1)
            val_correct+=(predictions == labels).sum().item()

    return valid_loss,val_correct

def testBatch(model):
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
                               for j in range(batch_size)))
  
    with torch.no_grad():
        output = model(images)
        scores, predictions = torch.max(output.data,1)
        
        # Let's show the predicted labels on the screen to compare with the real ones
        print('Predicted: ', ' '.join('%5s' % classes[predictions[j]] 
                                for j in range(batch_size)))

if __name__ == '__main__':

    model = ConvolutionNeuralNetwork()
    path = "apurated_model_fer2.pth"
    model.load_state_dict(torch.load(path))

    testBatch(model)
# 
    # test_loss,test_correct=valid(model,test_loaded)
    # test_loss = test_loss / len(test_loaded.sampler)
    # test_acc = test_correct / len(test_loaded.sampler) * 100
    # print(" AVG Test Loss:{:.3f} AVG Test Acc {:.2f} %".format( test_loss,test_acc))