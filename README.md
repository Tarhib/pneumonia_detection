# Pneumonia Classification using ResNet-18

## Overview
This project implements a deep learning-based pneumonia classification model using ResNet-18. The model is trained on chest X-ray images to classify whether a patient has pneumonia or not. The dataset used consists of labeled X-ray images of healthy individuals and pneumonia-affected patients.

## Dataset
The dataset consists of:
- Chest X-ray images categorized into two classes:
  - **Normal:** Healthy lung X-rays.
  - **Pneumonia:** X-rays indicating pneumonia.
- The dataset is preprocessed and split into training, validation, and test sets.

## Model Architecture
- **Backbone:** ResNet-18
- **Input Shape:** Preprocessed X-ray images (typically resized to 224x224 pixels)
- **Loss Function:** Binary Cross Entropy Loss
- **Optimizer:** Adam / SGD
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score

## Installation
To set up the project, follow these steps:

1. First Go to Folder:
   ```bash
   
   cd pneumonia_detection
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the dataset from https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

## Setting Up Dataset
1. First extract chest_xray dataset from downloded zip file.Then go to dataset folder
```bash
   
   cd dataset
   ```
2. Paste chest_xray folder into that "dataset" folder

## Training the Model ResNet-18 Scratch version
To train the model from scratch using dataset, run:

```bash
python train_resnet.py --model resnet --bs 32 --lr 0.01
```
Here ***bs*** represents batch_size and ***lr*** represents learning rate value.
For scratch version use model parameter value as ***"resnet"*** 
This will run for 50 epochs. One can change the epoch value using ***"-epoch"*** parameter

## Testing the model ResNet-18 Scratch version
To evaluate the model on the test set, run:
```bash
python test_resnet.py --model resnet
```
Again model param value needs to set as ***"resnet"*** 



## Training the Model ResNet-18 Pretrained version
To train the pretrained model  run:

```bash
python train_resnet.py --model resnet-pretrained --bs 32 --lr 0.1
```
Here ***bs*** represents batch_size and ***lr*** represents learning rate value.
For scratch version use model parameter value as ***"resnet"*** 

## Testing the model ResNet-18 Scratch version
To evaluate the model on the test set, run:
```bash
python test_resnet.py --model resnet-pretrained
```
Again model param value needs to set as ***"resnet-pretrained"*** 




## Results
The model achieves high accuracy in pneumonia classification, with detailed evaluation metrics provided in the final report.

## References
- Dataset source: [Kaggle NIH Chest X-ray Dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)


