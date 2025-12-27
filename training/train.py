"""Optimized Training Script for Food Image Classification"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import numpy as np
import time
import os
import argparse
import json
from datetime import datetime

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Config:
    DATA_DIR = 'data'
    TRAIN_DIR = 'data/train'
    TEST_DIR = 'data/test'
    MODEL_TYPE = 'resnet18'
    NUM_CLASSES = 20
    PRETRAINED = True
    EPOCHS = 25
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    EARLY_STOP_PATIENCE = 7
    LR_PATIENCE = 3
    LR_FACTOR = 0.5
    MODEL_SAVE_PATH = 'ml_models'
    LOG_DIR = 'training/logs'
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_transforms():
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transforms, val_transforms


def get_data_loaders(config):
    train_transforms, val_transforms = get_transforms()
    
    train_dataset = datasets.ImageFolder(root=config.TRAIN_DIR, transform=train_transforms)
    test_dataset = datasets.ImageFolder(root=config.TEST_DIR, transform=val_transforms)
    
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    generator = torch.Generator().manual_seed(SEED)
    train_data, val_data = random_split(train_dataset, [train_size, val_size], generator=generator)
    
    class_counts = [0] * len(train_dataset.classes)
    for _, label in train_dataset.samples:
        class_counts[label] += 1
    
    total_samples = sum(class_counts)
    class_weights = torch.tensor([
        total_samples / (len(class_counts) * count) for count in class_counts
    ], dtype=torch.float32)
    
    pin_mem = torch.cuda.is_available()
    train_loader = DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=pin_mem)
    val_loader = DataLoader(val_data, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=pin_mem)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=pin_mem)
    
    print(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_dataset)} | Classes: {len(train_dataset.classes)}")
    
    return train_loader, val_loader, test_loader, train_dataset.classes, class_weights


def create_model(config):
    if config.PRETRAINED:
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    else:
        model = models.resnet18(weights=None)
    
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(256, config.NUM_CLASSES)
    )
    return model


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    num_batches = len(loader)
    
    for batch_idx, (inputs, labels) in enumerate(loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
            print(f"\r  Batch [{batch_idx+1:3d}/{num_batches}]", end="", flush=True)
    
    print()
    return running_loss / len(loader), correct / total


def validate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return running_loss / len(loader), correct / total


def train(config):
    print("\n" + "="*60)
    print("FOOD IMAGE CLASSIFICATION - TRAINING")
    print("="*60)
    
    device = config.DEVICE
    print(f"Device: {device}")
    
    train_loader, val_loader, test_loader, classes, class_weights = get_data_loaders(config)
    config.NUM_CLASSES = len(classes)
    class_weights = class_weights.to(device)
    
    model = create_model(config).to(device)
    print(f"Model: {config.MODEL_TYPE} | Params: {sum(p.numel() for p in model.parameters()):,}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=config.LR_FACTOR, patience=config.LR_PATIENCE)
    
    best_val_acc = 0.0
    patience_counter = 0
    start_time = time.time()
    
    print("\n" + "-"*60)
    
    for epoch in range(config.EPOCHS):
        epoch_start = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch {epoch+1:2d}/{config.EPOCHS} | Train: {train_acc*100:.1f}% | Val: {val_acc*100:.1f}% | Time: {epoch_time:.1f}s")
        
        scheduler.step(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(config.MODEL_SAVE_PATH, f'{config.MODEL_TYPE}_food.pth')
            torch.save({'model_state_dict': model.state_dict(), 'val_acc': val_acc}, save_path)
            patience_counter = 0
            print(f"  ✓ Best model saved!")
        else:
            patience_counter += 1
        
        if patience_counter >= config.EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break
    
    total_time = time.time() - start_time
    
    checkpoint = torch.load(os.path.join(config.MODEL_SAVE_PATH, f'{config.MODEL_TYPE}_food.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    
    print("\n" + "="*60)
    print(f"COMPLETE | Time: {total_time/60:.1f}min | Val: {best_val_acc*100:.1f}% | Test: {test_acc*100:.1f}%")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Train Food Classification Model')
    parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18', 'baseline_cnn'])
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    
    args = parser.parse_args()
    
    config = Config()
    config.MODEL_TYPE = args.model
    config.EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    config.LEARNING_RATE = args.lr
    
    os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
    train(config)


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
