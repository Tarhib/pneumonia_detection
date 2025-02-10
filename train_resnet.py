import os
import sys
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES']="0"
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import torchvision
import torchvision.transforms as transforms

import os
import argparse
import random
import numpy as np
#from data_process import get_CIFAR10_data, get_MUSHROOM_data
from scipy.spatial import distance
from models import Pneumonia_Detection
import torch
import numpy as np
import torch.nn as nn
from sklearn.metrics import confusion_matrix, classification_report

import utils.train_utils as train_utils
from torch.utils.data import Dataset, DataLoader


import torch
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision.datasets import ImageFolder
from utils.train_utils import progress_bar

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')

parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--epoch', default=50, type=int, help='Epoch')
parser.add_argument('--bs', '--batch_size', default=128, type=int,
                    help='mini-batch size (default: 64)')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--seed', default=48, type=int,
                    help='Random Seed')
parser.add_argument('--name', default='ResNet18_cifar', type=str,
                    help='name of experiment', required = False)
parser.add_argument('--model', default ='resnet', type = str, help='Model Type')



args = parser.parse_args()
torch.manual_seed(args.seed)
np.random.seed(args.seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch






# Define the image preprocessing transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet input size
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3-channel
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
])

root_dir_path= "./dataset/chest_xray"
train_data_path = root_dir_path+"/train"
val_data_path = root_dir_path+"/val"

# Load train and validation datasets
train_dataset = ImageFolder(root=train_data_path, transform=transform)
val_dataset = ImageFolder(root=val_data_path, transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False)





model = Pneumonia_Detection(fnet_type=args.model)  # Set m=1 since data is 1D sequences here

#model = Pneumonia_Detection(fnet_type='resnet')  # Set m=1 since data is 1D sequences here
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
# Define loss and optimizer
criterion = nn.CrossEntropyLoss().cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=1e-4)
if args.epoch == 100:
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,75,90], gamma = 0.1, verbose = True)
elif args.epoch == 200:
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,150], gamma = 0.1, verbose = True)
elif args.epoch == 50:
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 40], gamma=0.1, verbose=True)


# Move the model to the correct device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def validate(args, epoch):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    # Create empty lists to store predictions and ground-truth labels
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            # Accumulate labels and predictions (moving tensors to CPU and converting to NumPy)
            all_labels.extend(targets.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

            progress_bar(batch_idx, len(val_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    # Compute per-class accuracy using classification_report
    

    
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        
        
        directory = os.path.join(f"./checkpoints/{args.model}")
        if not os.path.exists(directory):
            os.makedirs(directory)
        filename = os.path.join(directory,'model_best.pth.tar')
        torch.save(state, filename)
        best_acc = acc
    # Define class names (adjust if necessary)
    class_names = ["Normal", "Pneumonia"]

    

print(f"Training for {args.epoch} epochs")
for epoch in range(start_epoch, start_epoch + args.epoch):
    print("===========Training for epoch: "+str(epoch)+"=========")
    train(epoch)
    print("===========Validation Dataset=========")
    validate(args, epoch)
    scheduler.step()
print(f"Best Accuracy :{best_acc}")
