# app/ml_model.py
import torch
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, random_split
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os
from PIL import Image

# Define image transformations for both training and validation sets
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load training dataset
train_dataset = datasets.ImageFolder(root='data/train', transform=train_transforms)
test_dataset = datasets.ImageFolder(root='data/test', transform=val_transforms)

# Calculate the length of the dataset split (80% train, 20% validation)
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size

# Split the dataset into train and validation sets
train_data, val_data = random_split(train_dataset, [train_size, val_size])

# Create DataLoaders for batching
train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

# Load the trained model
model = models.resnet18()
num_classes = len(train_dataset.classes)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load('ml_models/modified_resnet18.pth', map_location=torch.device('cpu')))
model.eval()

# Define image transformation for prediction
predict_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load nutritional data
nutrition_data = pd.read_csv('data/nutrition_data.csv')

def predict_image(image_path):
    image = Image.open(image_path)
    image_tensor = predict_transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)

    class_idx = predicted.item()
    class_name = train_dataset.classes[class_idx]

    # Get actual nutritional information
    nutrition = nutrition_data[nutrition_data['FoodCategory'] == class_name]

    # Dummy classification report and confusion matrix
    report = classification_report([0], [class_idx])
    matrix = confusion_matrix([0], [class_idx])

    return class_name, report, matrix, nutrition

def evaluate_model():
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    report = classification_report(all_labels, all_preds, target_names=train_dataset.classes)
    matrix = confusion_matrix(all_labels, all_preds)

    return all_labels, all_preds, report, matrix
