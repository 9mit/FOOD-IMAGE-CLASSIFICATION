# 🍛 Food Image Classification

A deep learning project for classifying **20 categories** of Indian and international food images using **Transfer Learning with ResNet-18**.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1aHg030xsHIfnKGO6UMQM0FicHG-LmDHA?usp=sharing)

## 📊 Results (Kaggle GPU - Tesla T4)

| Model | Test Accuracy | Best Val Accuracy | Parameters | Training Time |
|-------|---------------|-------------------|------------|---------------|
| Baseline CNN | 29.26% | 31.46% | 657,428 | ~13 min |
| **ResNet-18** | **87.26%** | **89.80%** | 11,312,980 | ~26 min |

> **+58% improvement** using transfer learning!

## 🍽️ Food Classes (20)

```
burger, butter_naan, chai, chapati, chole_bhature, dal_makhani, dhokla,
fried_rice, idli, jalebi, kaathi_rolls, kadai_paneer, kulfi, masala_dosa,
momos, paani_puri, pakode, pav_bhaji, pizza, samosa
```

## 📁 Project Structure

```
FoodImageClassification/
├── models/
│   ├── baseline_cnn.py      # Baseline CNN (657K params)
│   └── resnet18.py          # Transfer learning model
├── training/
│   └── train.py             # Training with early stopping, LR scheduling
├── inference/
│   └── predict.py           # Production inference API
├── evaluation/
│   └── metrics.py           # Accuracy, F1, confusion matrix
├── tests/
│   └── robustness_tests.py  # Noise, blur, brightness tests
├── analysis/
│   ├── error_analysis.md    # Model error patterns
│   └── data_challenges.md   # Class imbalance handling
├── experiments/
│   └── model_comparison.md  # Baseline vs ResNet comparison
├── data/
│   ├── train/               # 4,378 training images (20 class folders)
│   └── test/                # 950 test images (20 class folders)
├── ml_models/               # Saved model weights (.pth)
├── streamlit_app.py         # Web demo
├── foodimage-ipynb.ipynb    # Kaggle notebook with all results
└── requirements.txt
```

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/YOUR_USERNAME/FoodImageClassification.git
cd FoodImageClassification
pip install -r requirements.txt
```

### Run Inference
```bash
python inference/predict.py --image path/to/food.jpg
```

### Run Web Demo
```bash
streamlit run streamlit_app.py
```

## 🏋️ Training

Trained on **Kaggle with Tesla T4 GPU**.

### Training Strategy
1. **Baseline CNN**: 10 epochs from scratch
2. **ResNet-18 Phase 1**: 5 epochs (frozen backbone, train classifier only)
3. **ResNet-18 Phase 2**: 15 epochs (fine-tune entire model with lower LR)

### Key Hyperparameters
| Parameter | Value |
|-----------|-------|
| Batch Size | 32 |
| Learning Rate | 0.001 → 0.0001 |
| Optimizer | Adam with weight decay (1e-4) |
| LR Scheduler | ReduceLROnPlateau |
| Early Stopping | 5 epochs patience |

## 📈 Per-Class Performance (ResNet-18)

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| burger | 0.98 | 0.98 | 0.98 |
| jalebi | 0.98 | 0.96 | 0.97 |
| fried_rice | 0.93 | 0.94 | 0.94 |
| dhokla | 0.93 | 0.91 | 0.92 |
| chole_bhature | 0.88 | 0.95 | 0.91 |
| idli | 0.88 | 0.91 | 0.90 |
| chai | 0.88 | 0.90 | 0.89 |
| momos | 0.83 | 0.92 | 0.87 |
| **Weighted Avg** | **0.88** | **0.87** | **0.87** |

## 📊 Dataset

| Metric | Value |
|--------|-------|
| **Total Images** | 6,269 |
| **Training Set** | 4,378 (70%) |
| **Validation Set** | 941 (15%) |
| **Test Set** | 950 (15%) |
| **Classes** | 20 |
| **Imbalance Ratio** | 3.21:1 |

### Class Imbalance
- **Largest class**: chapati (289 images)
- **Smallest class**: paani_puri (90 images)
- **Solution**: Weighted cross-entropy loss

### Data Augmentation
```python
transforms.RandomHorizontalFlip(p=0.5)
transforms.RandomRotation(15)
transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

## 🔬 Key Technical Features

- **Reproducibility**: Fixed seed (42) across PyTorch, NumPy, CUDA
- **Class Imbalance**: Weighted cross-entropy loss
- **Data Augmentation**: RandomFlip, Rotation(15°), ColorJitter
- **Regularization**: Dropout (0.3, 0.2) + Weight Decay (1e-4)
- **Transfer Learning**: ImageNet pre-trained ResNet-18

## 🚀 Future Improvements

- [ ] Try EfficientNet or Vision Transformers
- [ ] Test-Time Augmentation for better accuracy
- [ ] Expand to 50+ food categories
- [ ] Mobile deployment (ONNX/TFLite)
- [ ] Add nutritional information lookup

## 📝 License

MIT License

## 🙏 Acknowledgments

- Dataset: [Food Image Dataset on Kaggle](https://www.kaggle.com/datasets)
- Pre-trained weights: PyTorch torchvision models
