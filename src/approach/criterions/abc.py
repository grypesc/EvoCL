import torch
from torch.nn import Parameter
import torch.nn.functional as F

from .proxy_nca import binarize_and_smooth_labels



class ABCLoss(torch.nn.Module):
    def __init__(self,
                 nb_classes,
                 sz_embedding,
                 device,
                 smoothing=0.1,
                 temperature=1,
                 ):
        super().__init__()

        self.proxies = Parameter(torch.randn(nb_classes, sz_embedding, device=device))
        self.smoothing = smoothing
        self.temperature = temperature
        self.margin = 1

    def forward(self, X, T, old_proxies=None):
        bsz = X.shape[0]
        D = torch.cdist(X, self.proxies)
        pos_D = torch.gather(D, 1, T.unsqueeze(1)).squeeze(1)
        T = binarize_and_smooth_labels(T, len(self.proxies), 0)
        neg_D = D[~T.bool()].reshape(bsz, -1)
        loss_pull = pos_D ** 2
        loss_push = ((neg_D - self.margin) ** 2).mean(1)
        loss_push_old = 0

        if old_proxies is not None:
            D = torch.cdist(X, old_proxies)
            loss_push_old = ((D - self.margin) ** 2).mean(1)

        loss = (loss_pull + loss_push + loss_push_old).mean()
        return loss, None
