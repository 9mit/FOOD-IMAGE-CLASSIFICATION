"""Robustness tests for Food Image Classification"""

import torch
from torchvision import transforms
from PIL import Image, ImageFilter
import numpy as np
import os


def apply_degradation(image, degradation_type, level):
    """Apply a degradation to an image."""
    
    if degradation_type == 'blur':
        return image.filter(ImageFilter.GaussianBlur(radius=level * 2))
    
    elif degradation_type == 'noise':
        img_array = np.array(image).astype(float)
        noise = np.random.normal(0, level * 50, img_array.shape)
        noisy = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy)
    
    elif degradation_type == 'resolution':
        w, h = image.size
        factor = max(0.1, 1 - level * 0.2)
        small = image.resize((int(w*factor), int(h*factor)), Image.BILINEAR)
        return small.resize((w, h), Image.BILINEAR)
    
    elif degradation_type == 'brightness':
        from PIL import ImageEnhance
        factor = 1 + (level - 2.5) * 0.2
        return ImageEnhance.Brightness(image).enhance(max(0.3, factor))
    
    elif degradation_type == 'contrast':
        from PIL import ImageEnhance
        factor = 1 + (level - 2.5) * 0.2
        return ImageEnhance.Contrast(image).enhance(max(0.3, factor))
    
    return image


def test_robustness(model, image_path, transform, device, class_names):
    """Test model robustness on a single image."""
    
    original = Image.open(image_path).convert('RGB')
    degradations = ['blur', 'noise', 'resolution', 'brightness', 'contrast']
    results = {}
    
    for deg in degradations:
        results[deg] = []
        for level in range(1, 6):
            degraded = apply_degradation(original, deg, level)
            tensor = transform(degraded).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(tensor)
                probs = torch.softmax(outputs, dim=1)[0]
                pred_idx = torch.argmax(probs).item()
                confidence = probs[pred_idx].item()
            
            results[deg].append({
                'level': level,
                'prediction': class_names[pred_idx],
                'confidence': confidence
            })
    
    return results


if __name__ == '__main__':
    print("Import this module to run robustness tests.")
