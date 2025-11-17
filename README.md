# News Image Classifier (ResNet18 + PyTorch)

A real-world image classification project using PyTorch and Transfer Learning.  
The model predicts the category of a news image into one of five classes commonly used in media pipelines:

- Disaster
- Economy
- Health
- Politics
- Sports

This project was built end-to-end:  
data preparation → training → evaluation → inference.

## Project Structure

news-image-classifier/
│
├── data/
│   ├── train/
│   └── val/
│
├── model/
│   └── resnet18_model.pth
│
├── src/
│   ├── dataset.py
│   ├── train.py
│   └── predict.py
│
└── README.md

## Model Details

- Backbone: ResNet18 (ImageNet pretrained)  
- Fine-tuned on 5 custom categories  
- Input size: 224×224  
- Loss: CrossEntropyLoss  
- Optimizer: Adam (lr=1e-4)  
- Hardware: NVIDIA RTX 4080

## Validation Results

Accuracy: ~85%

precision, recall, f1-score:

disaster: 1.00 / 0.93 / 0.97  
economy: 0.75 / 1.00 / 0.86  
health: 0.92 / 0.73 / 0.81  
politics: 0.81 / 0.87 / 0.84  
sports: 0.85 / 0.73 / 0.79  

macro avg: 0.87 / 0.85 / 0.85  
weighted avg: 0.87 / 0.85 / 0.85

## Training

Run training:

cd src  
python3 train.py

This will:
- Load ResNet18 (ImageNet weights)
- Train on the custom dataset
- Evaluate model performance
- Save the model to: ../model/resnet18_model.pth

## Inference (Prediction)

Run prediction on any image:

cd src  
python3 predict.py /path/to/image.jpg

## Requirements

Install dependencies:

pip install -r requirements.txt

Minimal dependencies:
- torch
- torchvision
- scikit-learn
- Pillow

## Author

Qusai Ayyad  
AI Engineer / Computer Vision Learner  
GitHub: https://github.com/gtg0gtg
