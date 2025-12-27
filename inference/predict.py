"""Food Image Classification - Inference Module"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import os
import argparse
import json

FOOD_CLASSES = [
    'burger', 'butter_naan', 'chai', 'chapati', 'chole_bhature',
    'dal_makhani', 'dhokla', 'fried_rice', 'idli', 'jalebi',
    'kaathi_rolls', 'kadai_paneer', 'kulfi', 'masala_dosa', 'momos',
    'paani_puri', 'pakode', 'pav_bhaji', 'pizza', 'samosa'
]


class FoodClassifier:
    def __init__(self, model_path=None, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path or self._find_model()
        self.classes = FOOD_CLASSES
        self.num_classes = len(self.classes)
        self._model = None
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _find_model(self):
        candidates = ['ml_models/resnet18_food.pth', 'ml_models/modified_resnet18.pth']
        for path in candidates:
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f"No model found. Searched: {candidates}")
    
    def _load_model(self):
        model = models.resnet18()
        try:
            model.fc = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(model.fc.in_features, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(256, self.num_classes)
            )
            checkpoint = torch.load(self.model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        except:
            model = models.resnet18()
            model.fc = nn.Linear(model.fc.in_features, self.num_classes)
            model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        
        model = model.to(self.device)
        model.eval()
        return model
    
    @property
    def model(self):
        if self._model is None:
            self._model = self._load_model()
        return self._model
    
    def preprocess(self, image):
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('RGB')
        return self.transform(image)
    
    def predict(self, image, top_k=5):
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
        
        probs = probabilities.cpu().numpy()
        top_indices = np.argsort(probs)[::-1][:top_k]
        
        top_predictions = [
            {'class': self.classes[idx], 'confidence': float(probs[idx])}
            for idx in top_indices
        ]
        
        predicted_idx = top_indices[0]
        return {
            'predicted_class': self.classes[predicted_idx],
            'confidence': float(probs[predicted_idx]),
            'top_predictions': top_predictions
        }


def main():
    parser = argparse.ArgumentParser(description='Food Image Classification')
    parser.add_argument('--image', '-i', type=str, required=True)
    parser.add_argument('--top-k', '-k', type=int, default=5)
    parser.add_argument('--json', '-j', action='store_true')
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}")
        return 1
    
    classifier = FoodClassifier()
    result = classifier.predict(args.image, top_k=args.top_k)
    
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"\nPredicted: {result['predicted_class'].replace('_', ' ').title()}")
        print(f"Confidence: {result['confidence']:.1%}\n")
        for i, pred in enumerate(result['top_predictions'], 1):
            print(f"  {i}. {pred['class'].replace('_', ' ').title()}: {pred['confidence']:.1%}")


if __name__ == '__main__':
    main()
