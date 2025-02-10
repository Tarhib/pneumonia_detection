from pickle import FALSE
import torch
import os
import numpy as np
import torch.nn.functional as F
import time
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time
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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
parser = argparse.ArgumentParser(description='SNN testing with CIFAR benchmark')

parser.add_argument('--model', default ='resnet', type = str, help='Model Type')

parser.add_argument('--bs', default = 200, type = int, help='Batch size')

parser.add_argument('--layers', default=100, type=int, help='total number of layers (default: 100)')

parser.set_defaults(argument=True)

args = parser.parse_args()

torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


def model_loader(args):  
    model = Pneumonia_Detection(fnet_type=args.model)  # Set m=1 since data is 1D sequences here
    checkpoint = torch.load(f"./checkpoints/{args.model}/model_best.pth.tar")

    #model.load_state_dict(checkpoint['state_dict'], strict = True)
    model.load_state_dict(checkpoint['net'], strict=True)  # Use 'net' instead of 'state_dict'
    print("Validation Accuracy")
    print(checkpoint['acc'])

    model.cuda()
    model.eval()
    return model



# Define the image preprocessing transformations
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet input size
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3-channel
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
])

root_dir_path= "./dataset/chest_xray"
test_data_path = root_dir_path+"/test"

# Load train and validation datasets
test_dataset = ImageFolder(root=test_data_path, transform=transform_test)


# Create data loaders
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = model_loader(args)



def evaluate_test_set(model, test_loader, class_names=["Normal", "Pneumonia"]):
    """
    Evaluates the model on the test dataset and prints:
    - Overall Accuracy
    - Per-Class Accuracy (Precision, Recall, F1-Score)
    - Confusion Matrix (Optional)
    """
    model.eval()  # Set model to evaluation mode
    correct, total = 0, 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)  # Get class with highest probability

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute overall accuracy
    accuracy = 100.0 * correct / total
    print(f"\n Overall Test Accuracy: {accuracy:.2f}%\n")

    # Compute per-class accuracy using classification_report
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    print(report)
    

    return accuracy

# Example usage
evaluate_test_set(model, test_loader, class_names=["Normal", "Pneumonia"])