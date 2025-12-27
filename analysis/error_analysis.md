# Error Analysis: ResNet-18 Food Classifier

## Overall Performance

| Metric | Value |
|--------|-------|
| Test Accuracy | 87.26% |
| Macro F1 | 0.87 |
| Weighted F1 | 0.87 |

---

## Top Performing Classes

| Class | Precision | Recall | F1-Score | Analysis |
|-------|-----------|--------|----------|----------|
| burger | 0.98 | 0.98 | 0.98 | Distinctive shape, easy to identify |
| jalebi | 0.98 | 0.96 | 0.97 | Unique orange spiral pattern |
| pakode | 0.97 | 0.86 | 0.91 | Consistent fried texture |
| paani_puri | 0.94 | 0.85 | 0.89 | Distinctive spherical shape |
| fried_rice | 0.93 | 0.94 | 0.94 | Clear grain texture patterns |

---

## Challenging Classes

| Class | Precision | Recall | F1-Score | Confusion With |
|-------|-----------|--------|----------|----------------|
| kaathi_rolls | 0.70 | 0.75 | 0.73 | chapati, butter_naan |
| butter_naan | 0.79 | 0.76 | 0.78 | chapati, paratha |
| chapati | 0.79 | 0.84 | 0.81 | butter_naan, roti |
| samosa | 0.94 | 0.80 | 0.86 | pakode (fried items) |

---

## Error Patterns

### 1. Visually Similar Foods
- **Flatbreads**: chapati, butter_naan, roti share similar circular/flat appearance
- **Fried Items**: pakode and samosa both have golden-brown fried exterior
- **Wrapped Foods**: kaathi_rolls confused with flatbreads when filling not visible

### 2. Color-Based Confusion
- Dal makhani and kadai paneer: both have orange/brown curry base
- Chole bhature: sometimes confused with puri-based dishes

### 3. Presentation Variations
- Same dish photographed from different angles affects recognition
- Garnishes and side dishes can distract the model

---

## Improvement Recommendations

### Short-term
1. **Data Collection**: Add more images for low-performing classes
2. **Augmentation**: Add more aggressive rotation/cropping for viewpoint invariance
3. **Ensemble**: Combine predictions from multiple models

### Long-term
1. **Attention Mechanisms**: Help model focus on discriminative regions
2. **Hierarchical Classification**: Group similar foods (flatbreads → specific type)
3. **Multi-scale Features**: Better handle varying food portion sizes

---

## Per-Class Test Results

```
               precision    recall  f1-score   support

       burger       0.98      0.98      0.98        47
  butter_naan       0.79      0.76      0.78        50
         chai       0.88      0.90      0.89        58
      chapati       0.79      0.84      0.81        62
chole_bhature       0.88      0.95      0.91        62
  dal_makhani       0.78      0.82      0.80        49
       dhokla       0.93      0.91      0.92        44
   fried_rice       0.93      0.94      0.94        54
         idli       0.88      0.91      0.90        47
       jalebi       0.98      0.96      0.97        45
 kaathi_rolls       0.70      0.75      0.73        44
 kadai_paneer       0.85      0.89      0.87        62
        kulfi       0.89      0.86      0.87        36
  masala_dosa       0.93      0.81      0.86        47
        momos       0.83      0.92      0.87        48
   paani_puri       0.94      0.85      0.89        20
       pakode       0.97      0.86      0.91        42
    pav_bhaji       0.84      0.81      0.83        53
        pizza       0.90      0.90      0.90        40
       samosa       0.94      0.80      0.86        40

     accuracy                           0.87       950
    macro avg       0.88      0.87      0.87       950
 weighted avg       0.88      0.87      0.87       950
```
