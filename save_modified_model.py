# save_modified_model.py
import torch
from torchvision import models

# Load the pre-trained model
model = models.resnet18(pretrained=True)

# Modify the final fully connected layer to match the number of classes in your dataset
num_classes = 20  # Update this based on your dataset
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

# Save the modified model
torch.save(model.state_dict(), 'ml_models/modified_resnet18.pth')
print("Modified model saved successfully.")
