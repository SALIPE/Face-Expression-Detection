import torch
from torchvision import datasets, transforms

transform = transforms.Compose([transforms.Resize(255),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor()])

train_dataset = datasets.ImageFolder('expression_faces_dataset/images/train', transform=transform)

dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

# Get one batch
images, labels = next(iter(dataloader))

print(images[0])