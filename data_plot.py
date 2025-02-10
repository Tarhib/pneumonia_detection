import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2

from torchcam.methods import GradCAM  # Import GradCAM from torch-cam

# ----------------------------
# Set device and fixed random seed
# ----------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)
np.random.seed(42)

# ----------------------------
# Define transforms and load CIFAR-10 test set
# ----------------------------
# We use ImageNet normalization for the pretrained model.
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # resize to the input size of ResNet
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

# ----------------------------
# Load a pretrained model (ResNet18) and set to evaluation mode
# ----------------------------
model = torchvision.models.resnet18(pretrained=True)
model.eval()
model.to(device)

# ----------------------------
# Initialize TorchCAM GradCAM extractor
# ----------------------------
# For ResNet18, we use 'layer4' as the target layer.
cam_extractor = GradCAM(model, target_layer='layer4')

# ----------------------------
# Collect several misclassified examples from the test set
# ----------------------------
misclassified_images = []
misclassified_labels = []
misclassified_preds = []

# We'll collect 5 failure cases.
num_failures_to_collect = 5

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        # Find misclassified indices:
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
# Function to overlay CAM heatmap on the original image
# ----------------------------
def overlay_cam_on_image(image, cam, alpha=0.5):
    """
    Overlays a CAM heatmap on the original image.
    
    Args:
        image (Tensor): the normalized image tensor of shape (C,H,W).
        cam (ndarray): the CAM activation map (H',W').
        alpha (float): weight of the CAM overlay.
        
    Returns:
        overlayed (ndarray): the resulting image with CAM overlay.
    """
    # Unnormalize the image (assumes ImageNet stats)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_np = image.numpy().transpose(1, 2, 0)
    image_np = std * image_np + mean
    image_np = np.clip(image_np, 0, 1)
    image_np = np.uint8(255 * image_np)
    
    # Resize CAM to image size and normalize between 0 and 255.
    cam_resized = cv2.resize(cam, (image_np.shape[1], image_np.shape[0]))
    cam_resized = cam_resized - cam_resized.min()
    if cam_resized.max() > 0:
        cam_resized = cam_resized / cam_resized.max()
    cam_resized = np.uint8(255 * cam_resized)
    
    # Apply the JET color map to the CAM.
    heatmap = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Combine heatmap with the original image.
    overlayed = cv2.addWeighted(image_np, 1 - alpha, heatmap, alpha, 0)
    return overlayed

# ----------------------------
# Generate GradCAM visualizations for the failure cases
# ----------------------------
cam_results = []
for img, pred in zip(misclassified_images, misclassified_preds):
    # Create a batch dimension
    input_tensor = img.unsqueeze(0).to(device)
    # Forward pass through model
    output = model(input_tensor)
    # Extract the CAM for the predicted class. GradCAM expects:
    # cam_extractor(target_class, model_output)
    cam_dict = cam_extractor(pred, output)
    # The default method returns a dictionary; we assume the target layer is the only key.
    # Retrieve the CAM from the dictionary.
    cam = list(cam_dict.values())[0].squeeze(0).cpu().numpy()
    cam_results.append(cam)

# ----------------------------
# Display the misclassified images with CAM overlays
# ----------------------------
for i, (img, true_label, pred_label, cam) in enumerate(zip(misclassified_images,
                                                            misclassified_labels,
                                                            misclassified_preds,
                                                            cam_results)):
    overlayed_image = overlay_cam_on_image(img, cam, alpha=0.5)
    plt.figure(figsize=(4, 4))
    plt.imshow(overlayed_image)
    plt.title(f"True: {true_label}, Pred: {pred_label}")
    plt.axis('off')
    plt.show()
