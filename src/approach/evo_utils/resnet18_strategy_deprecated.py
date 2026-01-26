import torch
import torch.nn as nn

from ..models.resnet18_for_evo import resnet18
from .mlp_strategy import MLPStrategy


class Resnet18Strategy(MLPStrategy):
    def __init__(self, input_size, S, F, alpha, device):
        self.S = S
        self.F = F
        self.alpha = alpha
        self.device = device
        self.std = 0.01
        self.resnet = resnet18(S, is_224=False)
        self.resnet.to(device)
        self.resnet.eval()
        self.matrix_1 = torch.zeros((1, 32, 256), device=device)
        self.bias_1 = torch.zeros((1, 1, 32), device=device)
        self.matrix_2 = torch.zeros((1, 16, 64), device=device)
        self.bias_2 = torch.zeros((1, 1, 16), device=device)
        self.activation = nn.GELU()
        self.head_matrix = None
        self.head_bias = None
        self.adapter_matrix_1 = None
        self.adapter_matrix_2 = None
        self.adapter_bias_1 = None
        self.adapter_bias_2 = None

    @torch.no_grad()
    def __call__(self, images):
        x, features = self.resnet(images)
        features = features.unsqueeze(0)
        x = x.flatten(2, 3)
        x = self.matrix_1.unsqueeze(1) @ x.unsqueeze(0)
        x += self.bias_1.unsqueeze(3)
        x = self.activation(x)
        x = self.matrix_2.unsqueeze(1) @ x.permute(0, 1, 3, 2)
        x += self.bias_2.unsqueeze(3)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x += features
        x = nn.functional.normalize(x, dim=2)
        return x  # Population x Batch x Features

    def load_weights_from_path(self, path):
        state_dict = torch.load(path)
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        self.resnet.load_state_dict(state_dict, strict=False)


