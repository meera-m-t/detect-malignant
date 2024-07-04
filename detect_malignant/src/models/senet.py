import torch
import pretrainedmodels

class SENet(torch.nn.Module):
    def __init__(self, num_classes, pretrained):
        super(SENet, self).__init__()       
        self.model = pretrainedmodels.__dict__['senet154'](num_classes=1000, pretrained=pretrained)
        in_features = self.model.last_linear.in_features
        self.model.last_linear = torch.nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)