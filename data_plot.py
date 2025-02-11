import os
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import argparse
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from models import Pneumonia_Detection  # Ensure this is correctly implemented

# ----------------------------
# Set device and fixed random seed
# ----------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser(description='Pneumonia Detection Testing')
parser.add_argument('--model', default='resnet', type=str, help='Model Type')
parser.add_argument('--bs', default=200, type=int, help='Batch size')
parser.set_defaults(argument=True)
args = parser.parse_args()

torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)

# ----------------------------
# Model Loader Function
# ----------------------------
def model_loader(args):
    model = Pneumonia_Detection(fnet_type=args.model)
    checkpoint = torch.load(f"./checkpoints/resnet/model_best.pth.tar")
    model.load_state_dict(checkpoint['net'], strict=True)  # Use 'net' key
    print("Validation Accuracy:", checkpoint['acc'])
    model.to(device)
    model.eval()
    return model

# ----------------------------
# Define transforms and load the test dataset
# ----------------------------
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet input size
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3-channel
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
])

# Load test dataset
root_dir_path = "./dataset/chest_xray"
test_data_path = root_dir_path + "/test"
test_dataset = ImageFolder(root=test_data_path, transform=transform_test)

# Create DataLoader
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load the model
model = model_loader(args)

# ----------------------------
# Collect Misclassified Images
# ----------------------------
misclassified_images, misclassified_labels, misclassified_preds = [], [], []
num_failures_to_collect = 10

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        
        misclassified_idxs = (preds != labels).nonzero(as_tuple=True)[0]
        for idx in misclassified_idxs:
            misclassified_images.append(images[idx].cpu())
            misclassified_labels.append(labels[idx].cpu().item())
            misclassified_preds.append(preds[idx].cpu().item())
            if len(misclassified_images) >= num_failures_to_collect:
                break
        if len(misclassified_images) >= num_failures_to_collect:
            break

# ----------------------------
# Function to Save Misclassified Images
# ----------------------------
def save_misclassified_images(misclassified_images, misclassified_labels, misclassified_preds):
    """
    Saves misclassified images using OpenCV (cv2).
    """
    save_dir = "./misclassified_images"
    os.makedirs(save_dir, exist_ok=True)

    for i, (img, true_label, pred_label) in enumerate(zip(misclassified_images, misclassified_labels, misclassified_preds)):
        # Convert Tensor to NumPy Image
        img = img.numpy().transpose(1, 2, 0)  # Convert from (C,H,W) to (H,W,C)
        img = np.clip(img, 0, 1)  # Clip pixel values between 0 and 1
        img = np.uint8(255 * img)  # Convert to 0-255 scale

        # Save image
        save_path = os.path.join(save_dir, f"misclassified_{i}_true_{true_label}_pred_{pred_label}.png")
        cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))  # Save in BGR format
        print(f"Saved: {save_path}")

# ----------------------------
# Save Misclassified Images
# ----------------------------
save_misclassified_images(misclassified_images, misclassified_labels, misclassified_preds)

print("Misclassified images saved in: ./misclassified_images")
