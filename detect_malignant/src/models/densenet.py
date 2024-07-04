import torch
from densenet_pytorch import DenseNet 

class DenseNet_(torch.nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(DenseNet_, self).__init__()
        self.model = DenseNet.from_pretrained('densenet121')
        in_features = self.model.classifier.in_features
        self.model.classifier = torch.nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        if isinstance(x, list):
            x = torch.stack(x)
        
        # Temporarily remove hooks
        hooks = []
        for module in self.model.modules():
            if hasattr(module, '_forward_hooks'):
                hooks.extend(module._forward_hooks.values())
                module._forward_hooks.clear()
        
        result = self.model(x)
        
        # Reattach hooks
        for hook in hooks:
            module.register_forward_hook(hook)
        
        return result
