import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=8):
        super().__init__()
        # frozen original weights
        self.linear = nn.Linear(in_features, out_features)
        self.linear.weight.requires_grad = False

        # Lora matrices, initalized with 0.01 noise
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.scaling = alpha / rank

    @property
    def weight(self):
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias

    def forward(self, x):
        # Frozen path + LoRA path
        return self.linear(x) + (x @ self.lora_A @ self.lora_B) * self.scaling