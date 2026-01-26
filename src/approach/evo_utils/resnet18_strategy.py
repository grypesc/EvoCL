import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.resnet18_for_evo import resnet18
from .mlp_strategy import MLPStrategy


class Resnet18Strategy(MLPStrategy):
    def __init__(self, input_size, S, F, alpha, num_channels, device):
        self.S = S
        self.F = F
        self.alpha = alpha
        self.device = device
        self.std = 0.01
        self.resnet = resnet18(S, is_224=False)
        self.resnet.to(device)
        self.resnet.eval()
        self.matrix_1 = torch.zeros((1, num_channels, 256, 3, 3), device=device)
        self.bias_1 = torch.zeros((1, 1, 32), device=device)
        self.matrix_2 = torch.zeros((1, 512, num_channels, 3, 3), device=device)
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
        bsz = images.shape[0]
        x, features = self.resnet(images)
        features = features.unsqueeze(0)
        x = F.conv2d(x, reshape_kernel(self.matrix_1), stride=2, padding=1)
        x = self.activation(x)
        x = F.conv2d(x, reshape_kernel(self.matrix_2), padding=1, groups=self.matrix_2.shape[0])
        x = x.mean([2, 3])
        x = x.reshape(bsz, -1, self.S)
        x = x.permute(1, 0, 2)
        x += nn.functional.normalize(features, dim=2)
        x = nn.functional.normalize(x, dim=2)
        return x  # Population x Batch x Features

    def load_weights_from_path(self, path):
        state_dict = torch.load(path)
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        self.resnet.load_state_dict(state_dict, strict=False)


def reshape_kernel(kernel):
    return kernel.reshape(-1, *kernel.shape[2:])
