# Data Challenges & Solutions

## Challenge 1: Class Imbalance

### Problem
The dataset has significant class imbalance:
- **Largest class**: chapati (289 images)
- **Smallest class**: paani_puri (90 images)
- **Imbalance ratio**: 3.21:1

### Solution: Weighted Loss Function

```python
# Calculate class weights (higher weight for rarer classes)
total_samples = sum(class_counts.values())
class_weights = torch.tensor([
    total_samples / (num_classes * class_counts[c]) for c in classes
], dtype=torch.float32)

# Use weighted cross-entropy
criterion = nn.CrossEntropyLoss(weight=class_weights)
```

### Result
- Paani_puri (smallest class): 89% F1-score
- Model performs well across all classes despite imbalance

---

## Challenge 2: Visual Similarity

### Problem
Several food categories look similar:
- Flatbreads: chapati, butter_naan, roti
- Curries: dal_makhani, kadai_paneer
- Fried items: pakode, samosa

### Solution: Data Augmentation

```python
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
])
```

### Result
- Model learns subtle texture differences
- Kaathi_rolls still challenging (73% F1) but acceptable

---

## Challenge 3: Overfitting

### Problem
- Small dataset (6,269 images for 20 classes)
- Risk of memorizing training data

### Solutions Applied

1. **Dropout Regularization**
   ```python
   nn.Dropout(0.3)  # In classifier
   nn.Dropout(0.2)  # Before final layer
   ```

2. **Weight Decay**
   ```python
   optimizer = optim.Adam(model.parameters(), weight_decay=1e-4)
   ```

3. **Early Stopping**
   ```python
   if patience_counter >= 5:
       break  # Stop training
   ```

4. **Transfer Learning**
   - Use pre-trained ImageNet weights
   - Two-phase training (freeze → fine-tune)

### Result
- Validation accuracy (89.80%) close to training (98%)
- Model generalizes well to test set (87.26%)

---

## Challenge 4: Training Efficiency

### Problem
- Training from scratch is slow and ineffective
- Baseline CNN only reached 31% accuracy

### Solution: Transfer Learning

**Phase 1**: Freeze backbone, train classifier (5 epochs)
```python
for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True
```

**Phase 2**: Fine-tune entire model with lower LR (15 epochs)
```python
for param in model.parameters():
    param.requires_grad = True
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # 10x lower
```

### Result
- Achieved 87.26% in ~26 minutes on Tesla T4
- 58% improvement over baseline

---

## Summary of Techniques

| Challenge | Solution | Impact |
|-----------|----------|--------|
| Class Imbalance | Weighted Loss | Balanced per-class performance |
| Visual Similarity | Data Augmentation | Better texture discrimination |
| Overfitting | Dropout + Weight Decay + Early Stopping | Good generalization |
| Training Efficiency | Transfer Learning | 58% accuracy boost |

All techniques were validated on the Kaggle notebook with real results.
