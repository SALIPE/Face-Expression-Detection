import numpy as np
import cv2 as cv
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision import  transforms
from PIL import Image

def get_facial_expression(imagefile):

    image = face_img_detector(imagefile)
    return classifier(image)


def face_img_detector(imagefile):
    face_classifier = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

    # image = cv.imread(imagefile)
    nparr = np.fromstring(imagefile, np.uint8)
    # decode image
    image_gray = cv.imdecode(nparr, cv.COLOR_BGR2GRAY)
    # image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    faces = face_classifier.detectMultiScale(image_gray, 1.3,5)

    cord1 = 0
    cord2 = 0
    x=0
    y=0
    padding = 40
    for (x,y,w,h) in faces:
        x = x-padding
        y = y-padding
        cord1 = y+h+padding +10
        cord2 = x+w+padding +10
        # cv.rectangle(image_gray,(x,y),(x+w,y+h), (255,0,0),1)

    cropped = image_gray[y:cord1,x:cord2]
    return Image.fromarray(cropped)
    # cv.imshow('imagem', cropped)
    # cv.waitKey(0)
    # cv.destroyAllWindows()


class AlexNet(nn.Module):
    #acuracia no teste=62.52% || 63.08%
    #acuracia no treino=81.46% || 86.80%
    def __init__(self,num_classes=7):
        super(AlexNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=10, stride=4, padding=1)
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
        self.fc3 = nn.Linear(4096, num_classes)


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

def classifier(img):
    classes = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise','disgust']

    test_transform = transforms.Compose([
        transforms.Resize((227,227)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ])

    tensor_image = test_transform(img).unsqueeze(0)

    model = AlexNet()
    path = "apurated_model_alex.pth"
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu'))) 

    model.eval()

    output = model(tensor_image)
    scores, predictions = torch.max(output.data,1)
    conf,y_pre = scores.item(), classes[predictions.item()-1]
    print(y_pre,conf)
    return '{0} at confidence score:{1:.2f}'.format(y_pre,conf)

    
