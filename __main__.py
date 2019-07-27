import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse


# Hyper parameters
num_epochs = 600
num_classes = 10
batch_size = 148
learning_rate = 0.001

device = torch.device('cuda:0' if torch.cuda.is_available() else exit())
#Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=148, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        # LAYER 1 
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels = 3,out_channels = 16,kernel_size=3,
                stride=1,padding=1,dilation=1,groups=1,bias = True,padding_mode ='zeros'),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        # LAYER 2
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels = 16,out_channels = 32,kernel_size=3,
                stride=1,padding=1,dilation=1,groups=1,bias = True,padding_mode ='zeros'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        # LAYER 3
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels = 32,out_channels = 64,kernel_size=3,
                stride=1,padding=1,dilation=1,groups=1,bias = True,padding_mode ='zeros'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        # LAYER 4
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels = 64,out_channels = 128,kernel_size=3,
                stride=1,padding=1,dilation=1,groups=1,bias = True,padding_mode ='zeros'),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        # LAYER 5
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels = 128,out_channels = 288,kernel_size=3,
                stride=1,padding=1,dilation=1,groups=1,bias = True,padding_mode ='zeros'),
            nn.BatchNorm2d(288),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        # LAYER 6
        self.layer6 = nn.Sequential(
            nn.Conv2d(in_channels = 288,out_channels = 128,kernel_size=3,
                stride=1,padding=1,dilation=1,groups=1,bias = True,padding_mode ='zeros'),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        # LAYER 7
        self.layer7 = nn.Sequential( 
            nn.Conv2d(in_channels =128,out_channels =32 ,kernel_size=3,
                stride=1,padding=1,dilation=1,groups=1,bias = True,padding_mode ='zeros'),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2,stride=2)
        )
        # LAYER 8
        self.layer8 = nn.Sequential(
            nn.Conv2d(in_channels = 32,out_channels =32 ,kernel_size=3,
                stride=1,padding=1,dilation=1,groups=1,bias = True,padding_mode ='zeros'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        # LAYER 9

        self.fcn9 = nn.Sequential( 
            nn.Dropout(),
            nn.Linear(32,4096)
        )
        # Layer 10
        self.fcn10 = nn.Linear(4096,10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = torch.flatten(out, 1)
        out = self.fcn9(out)

        # out = out.reshape(out.size(0), -1)
        out = self.fcn10(out)
        return out

def _save_policy_(model,epoch,optimizer):
    if epoch % 100 == 0:
      checkpoint = {
      'model': model,
      'state_dict': model.state_dict(),
      'optimizer_state':optimizer.state_dict(),
      'epoch':epoch,
      'optimizer': optimizer,
      'loss':loss
                   }
      torch.save(checkpoint,"ConvNet_{}.pth".format(epoch))

# torch.save(checkpoint, 'checkpoint.pth')
#           torch.save(model.state_dict(), "ConvNet_Cifar_{}.pth".format(epoch)) 
def _lr_policy(epoch):
    if epoch  == 150:
      learning_rate = 0.0009
    if epoch  == 280:
      learning_rate = 0.0007
    if epoch == 350:
      learning_rate = 0.0006
    if epoch == 400:
      learning_rate = 0.0005
    if epoch == 500:
      learning_rate = 0.0003


print("Creating Model...")
model = ConvNet().to(device)
print("Model created successfully!")

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# Train the model
def _train_():
  total_step = len(trainloader)
  for epoch in range(num_epochs):
      _save_policy_(model,epoch,optimizer)
      _lr_policy(epoch)
      for i, (images, labels) in enumerate(trainloader):
          images = images.to(device)
          labels = labels.to(device)

          # Forward pass
          outputs = model(images)
          loss = criterion(outputs, labels)

          # Backward and optimize
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

          if (i+1) % 100 == 0:
              print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                  .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

def _test_():
  # Test the model
  model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
  with torch.no_grad():
      correct = 0
      total = 0
      for images, labels in testloader:
          images = images.to(device)
          labels = labels.to(device)
          outputs = model(images)
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()

      print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))


_train_()  
_test_()
