import torch
from torch import nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    """ Following the VGGnet based on VGG16 but for smaller input (64x64)
        Check this blog for some info: https://learningai.io/projects/2017/06/29/tiny-imagenet.html
    """

    def __init__(self, S=32, num_classes=1000):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(3, S, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(S, S, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(S, S, 3, padding=1),
            nn.GELU()
            )
        self.fc = nn.Linear(S, out_features=num_classes)
        # and `head_var` with the name of the head, so it can be removed when doing incremental learning experiments
        self.head_var = 'fc'

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = self.backbone(x)
        x = x.mean([2, 3])
        x = self.fc(x)
        return x


def convnet(s=32, num_out=100, pretrained=False):
    if pretrained:
        raise NotImplementedError
    return ConvNet(s, num_out)
