import torch
from torchvision import models

class EfficientNet:
    def __init__(self, num_classes):
        self.model = models.efficientnet_b0(pretrained=True)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = torch.nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)