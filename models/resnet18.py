"""ResNet-18 Transfer Learning for Food Image Classification"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import numpy as np
import time
import os

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_data_loaders(data_dir='data', batch_size=32):
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=train_transforms)
    test_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'test'), transform=val_transforms)
    
    # Calculate class weights for imbalanced data
    class_counts = [0] * len(train_dataset.classes)
    for _, label in train_dataset.samples:
        class_counts[label] += 1
    total = sum(class_counts)
    class_weights = torch.tensor([total / (len(class_counts) * c) for c in class_counts], dtype=torch.float32)
    
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_data, val_data = random_split(train_dataset, [train_size, val_size],
                                         generator=torch.Generator().manual_seed(SEED))
    
    pin_mem = torch.cuda.is_available()
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=pin_mem)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin_mem)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin_mem)
    
    return train_loader, val_loader, test_loader, train_dataset.classes, class_weights


def create_model(num_classes, pretrained=True):
    if pretrained:
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    else:
        model = models.resnet18(weights=None)
    
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes)
    )
    return model


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
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
            print(f"\r  Batch [{batch_idx+1:3d}/{num_batches}] - Loss: {loss.item():.4f}", end="", flush=True)
    
    print()
    return running_loss / len(loader), correct / total


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
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


def train_phase(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                num_epochs, device, save_path, phase_name="Training"):
    best_val_acc = 0.0
    patience_counter = 0
    early_stop_patience = 5
    
    print(f"\n{phase_name}")
    print("-" * 50)
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch [{epoch+1:2d}/{num_epochs}] | "
              f"Train: {train_acc*100:.1f}% | Val: {val_acc*100:.1f}% | "
              f"Time: {epoch_time:.1f}s")
        
        scheduler.step(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            patience_counter = 0
            print(f"  ✓ Best model saved!")
        else:
            patience_counter += 1
        
        if patience_counter >= early_stop_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    return best_val_acc


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    train_loader, val_loader, test_loader, classes, class_weights = get_data_loaders()
    class_weights = class_weights.to(device)
    print(f"Classes: {len(classes)} | Train batches: {len(train_loader)}")
    
    model = create_model(len(classes)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    save_path = 'ml_models/resnet18_food.pth'
    
    # Phase 1: Train classifier only (freeze backbone)
    print("\n" + "="*60)
    print("PHASE 1: Training Classifier (Backbone Frozen)")
    print("="*60)
    
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True
    
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    train_phase(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                num_epochs=5, device=device, save_path=save_path, phase_name="Phase 1")
    
    # Phase 2: Fine-tune entire model
    print("\n" + "="*60)
    print("PHASE 2: Fine-tuning Entire Model")
    print("="*60)
    
    for param in model.parameters():
        param.requires_grad = True
    
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    best_acc = train_phase(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                           num_epochs=15, device=device, save_path=save_path, phase_name="Phase 2")
    
    # Evaluate on test set
    model.load_state_dict(torch.load(save_path))
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    
    print("\n" + "="*60)
    print(f"TRAINING COMPLETE")
    print(f"Best Val Acc: {best_acc*100:.2f}% | Test Acc: {test_acc*100:.2f}%")
    print("="*60)


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
