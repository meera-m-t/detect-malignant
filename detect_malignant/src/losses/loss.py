import torch
import torch.nn as nn
import torch.nn.functional as F



class MalignantLoss(nn.Module):
    def __init__(self, config, num_classes, loss_dict):
        super(MalignantLoss, self).__init__()
        self.config = config
        self.num_classes = num_classes
        self.loss_dict = loss_dict #copy.deepcopy(config["loss_dict"])

        assert len(self.loss_dict) > 0, "Loss dictionary must not be empty."

        # Extract weights and magnitude scales using list comprehension
        self.lweights = torch.tensor([v.pop("weight") for v in self.loss_dict.values()])
        self.mag_scale = torch.tensor([v.pop("mag_scale") for v in self.loss_dict.values()])

        assert torch.isclose(self.lweights.sum(), torch.tensor(1.0)), "Weights must sum up to 1."
        self.loss_functions = [globals()[loss_name](**params) for loss_name, params in self.loss_dict.items()]

    def forward(self, inputs, targets, reduction=None):
        device = inputs.device
        self._move_to_device(device)
      
        targets = self._convert_targets(targets, device)

        loss = sum(loss_fn(inputs, targets) * scale for loss_fn, scale in zip(self.loss_functions, self.mag_scale))
        return (loss * self.lweights).sum()

    def _convert_targets(self, targets, device):
        temp = torch.zeros((targets.size(0), self.num_classes), device=device)
        return temp.scatter_(1, targets.view(-1, 1).long(), 1)

    def _move_to_device(self, device):
        """Move tensors to the specified device."""
        self.lweights = self.lweights.to(device)
        self.mag_scale = self.mag_scale.to(device)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        # Adding the logsigmoid module
        self.logsigmoid = nn.LogSigmoid()

    def forward(self, input, target):
        # Ensure input and target have the same shape
        if input.size() != target.size():
            raise ValueError(f"Target size {target.size()} must be the same as input size {input.size()}")

        # Compute the base loss
        max_val = (-input).clamp(min=0)
        base_loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()

        # Modify the base loss using the focal term
        invprobs = self.logsigmoid(-input * (target * 2.0 - 1.0))
        focal_loss = (invprobs * self.gamma).exp() * base_loss

        return focal_loss.sum(dim=1).mean()


class SmartCrossEntropyLoss(torch.nn.Module):
    def __init__(self,  config, num_classes, loss_dict, label_smoothing=0.1):
        super(SmartCrossEntropyLoss, self).__init__()
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        num_classes = logits.size(-1)
        with torch.no_grad():
            # Convert targets to one-hot encoding
            true_dist = torch.zeros_like(logits)
            true_dist.fill_(self.label_smoothing / (num_classes - 1))
            true_dist.scatter_(1, targets.data.unsqueeze(1), 1.0 - self.label_smoothing)

        log_probabilities = F.log_softmax(logits, dim=-1)
        loss = -torch.sum(true_dist * log_probabilities, dim=-1).mean()
        return loss



