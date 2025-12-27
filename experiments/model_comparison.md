# Model Comparison: Baseline CNN vs ResNet-18

## Final Results (Kaggle, Tesla T4 GPU)

| Metric | Baseline CNN | ResNet-18 |
|--------|--------------|-----------|
| **Test Accuracy** | 29.26% | **87.26%** |
| **Best Val Accuracy** | 31.46% | **89.80%** |
| **Parameters** | 657,428 | 11,312,980 |
| **Training Time** | ~13 min | ~26 min |
| **Training Strategy** | From scratch | Transfer learning |

> **ResNet-18 improved accuracy by 58.0% over baseline!**

---

## Training Details

### Baseline CNN

**Architecture**: 4 convolutional blocks → Global Average Pooling → Fully Connected

```
Conv(3→32) → BatchNorm → ReLU → MaxPool
Conv(32→64) → BatchNorm → ReLU → MaxPool
Conv(64→128) → BatchNorm → ReLU → MaxPool
Conv(128→256) → BatchNorm → ReLU → MaxPool
AdaptiveAvgPool(1×1)
FC(256→512) → ReLU → Dropout(0.5)
FC(512→256) → ReLU → Dropout(0.3)
FC(256→20)
```

**Training Progress**:
| Epoch | Train Acc | Val Acc |
|-------|-----------|---------|
| 1 | 10.1% | 17.1% |
| 5 | 21.8% | 26.5% |
| 10 | 25.9% | 30.9% |
| **Best** | - | **31.46%** |

---

### ResNet-18

**Architecture**: Pre-trained ImageNet ResNet-18 + Custom Classifier

```
ResNet-18 Backbone (frozen Phase 1, fine-tuned Phase 2)
↓
Dropout(0.3) → FC(512→256) → ReLU → Dropout(0.2) → FC(256→20)
```

**Phase 1 (Frozen Backbone)** - 5 epochs:
| Epoch | Train Acc | Val Acc |
|-------|-----------|---------|
| 1 | 33.6% | 68.8% |
| 5 | 60.9% | **76.4%** |

**Phase 2 (Fine-tuning)** - 15 epochs:
| Epoch | Train Acc | Val Acc |
|-------|-----------|---------|
| 1 | 72.9% | 83.6% |
| 8 | 96.6% | 89.7% |
| 11 | 97.9% | **89.8%** |

---

## Per-Class Accuracy Comparison

| Class | Baseline CNN | ResNet-18 |
|-------|--------------|-----------|
| burger | 29% | **98%** |
| jalebi | 46% | **97%** |
| fried_rice | 39% | **94%** |
| dhokla | 0% | **92%** |
| chole_bhature | 33% | **91%** |
| idli | 49% | **90%** |
| chai | 49% | **89%** |
| paani_puri | 6% | **89%** |
| momos | 4% | **87%** |
| pizza | 44% | **90%** |

---

## Classification Reports

### Baseline CNN
```
               precision    recall  f1-score   support

       burger       0.18      0.74      0.29        47
  butter_naan       0.50      0.28      0.36        50
         chai       0.53      0.45      0.49        58
      chapati       0.37      0.27      0.31        62
       dhokla       0.00      0.00      0.00        44
       jalebi       0.30      0.91      0.46        45
        momos       0.17      0.02      0.04        48

     accuracy                           0.29       950
    macro avg       0.29      0.28      0.25       950
 weighted avg       0.30      0.29      0.26       950
```

### ResNet-18
```
               precision    recall  f1-score   support

       burger       0.98      0.98      0.98        47
  butter_naan       0.79      0.76      0.78        50
         chai       0.88      0.90      0.89        58
      chapati       0.79      0.84      0.81        62
       dhokla       0.93      0.91      0.92        44
       jalebi       0.98      0.96      0.97        45
        momos       0.83      0.92      0.87        48

     accuracy                           0.87       950
    macro avg       0.88      0.87      0.87       950
 weighted avg       0.88      0.87      0.87       950
```

---

## Why ResNet-18 Wins

1. **Pre-trained Features**: ImageNet weights provide robust low-level feature extraction
2. **Deeper Architecture**: 18 layers vs 4 conv blocks captures more complex patterns
3. **Skip Connections**: Residual connections enable training deeper networks
4. **Two-Phase Training**: Freeze→Fine-tune prevents catastrophic forgetting

---

## Conclusion

For production deployment, **ResNet-18 with transfer learning** is the clear choice:

- **3x better accuracy** (87% vs 29%)
- Robust across all food classes
- Only ~13 minutes extra training time

The baseline CNN serves as educational baseline to understand CNN fundamentals.
