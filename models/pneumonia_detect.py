import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import torch
import torch.nn as nn
import torchvision.models as models

class Pneumonia_Detection(nn.Module):
    def __init__(self, fnet_type, num_classes=2):
        super(Pneumonia_Detection, self).__init__()
        
        if fnet_type == 'resnet':
            self.fnet = models.resnet18(pretrained=False) 
        elif fnet_type == 'resnet-pretrained':
            self.fnet = models.resnet18(pretrained=True)  
            for name, param in self.fnet.named_parameters():
                # Only leave parameters in the final layer (fc) trainable
                if "fc" not in name:
                    param.requires_grad = False
        
        num_ftrs = self.fnet.fc.in_features  # Extract input features of FC layer
        
        # Define the final classification layer
        self.fnet.fc = nn.Linear(num_ftrs, num_classes)  # Set correct output size

    def forward(self, x):
        final_output = self.fnet(x)  # Forward pass through ResNet
        return final_output

        
   