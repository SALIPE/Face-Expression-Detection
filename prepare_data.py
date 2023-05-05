import torch
from torchvision import datasets, transforms

def prepare_data():
    batch_size = 10
    transform = transforms.Compose([
        transforms.ToTensor()])

    train_dataset = datasets.ImageFolder('C:/Users/febue/Documents/complete_dataset/expression_faces_dataset/images/train', transform=transform)
    validation_dataset = datasets.ImageFolder('C:/Users/febue/Documents/complete_dataset/expression_faces_dataset/images/validation', transform=transform)

    train_loaded = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loaded = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)

    return train_loaded, validation_loaded

def get_parameters():
    batch_size = 10
    number_of_labels = 7
    classes = ('angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise')

    return batch_size,number_of_labels,classes

