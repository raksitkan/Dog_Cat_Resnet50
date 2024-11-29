import torch
import torchvision
from torch import nn

def create_resnet50_model(num_classes:int=3,
                          seed:int=42,
                          dropout_rate: float = 0.3):  # เพิ่ม dropout_rate
  # 1. Setup pretrained ResNet50 weights
  weights = torchvision.models.ResNet50_Weights.DEFAULT  # Use the default weights
  
  # 2. Get ResNet50 transforms
  transforms = weights.transforms()
  
  # 3. Setup pretrained model instance
  model = torchvision.models.resnet50(weights=weights)  # Use the pretrained ResNet50
  
  # 4. Freeze the base layers in the model (this will stop all layers from training)
  for param in model.parameters():
    param.requires_grad = False
  
  # 5. Change classifier head with random seed for reproducibility
  torch.manual_seed(seed)
  
  # เพิ่ม Dropout และกำหนด output เป็น num_classes
  model.fc = nn.Sequential(
    nn.Dropout(p=dropout_rate),  # เพิ่ม dropout
    nn.Linear(in_features=model.fc.in_features, out_features=num_classes, bias=True)
  )
  
  return model, transforms


