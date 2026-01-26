import math
from functools import partial

from torch.nn.init import trunc_normal_

import torch
import torch.nn as nn
import torch.nn.functional as F


class ViTStrategy:
    def __init__(self, S, F, alpha, hidden_size, device):
        self.S = S
        self.hidden_size = hidden_size
        self.F = F
        self.alpha = alpha
        self.device = device
        self.vit = VisionTransformer(
            patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
        self.std = 0.01
        self.matrix_1 = torch.zeros((1, self.hidden_size, 384), device=device)
        self.bias_1 = torch.zeros((1, self.hidden_size, ), device=device)
        self.matrix_2 = torch.zeros((1, 384, self.hidden_size), device=device)
        self.bias_2 = torch.zeros((1, 384, ), device=device)
        self.matrix_3 = torch.zeros((1, S, 384), device=device)
        self.bias_3 = torch.zeros((1, S, ), device=device)
        self.activation = nn.GELU()
        self.head_matrix = None
        self.head_bias = None
        self.adapter_matrix_1 = None
        self.adapter_matrix_2 = None
        self.adapter_bias_1 = None
        self.adapter_bias_2 = None

    @torch.no_grad()
    def prepare_for_training(self, mu_, num_classes, t):
        self.matrix_1 = self.matrix_1.repeat((mu_, 1, 1))
        self.bias_1 = self.bias_1.repeat((mu_, 1))
        self.matrix_2 = self.matrix_2.repeat((mu_, 1, 1))
        self.bias_2 = self.bias_2.repeat((mu_, 1))
        self.matrix_3 = self.matrix_3.repeat((mu_, 1, 1))
        self.bias_3 = self.bias_3.repeat((mu_, 1))

        self._reset_head_and_adapter(num_classes, t)
        self.head_matrix = self.head_matrix.repeat((mu_, 1, 1))
        self.head_bias = self.head_bias.repeat((mu_, 1, 1))
        self.adapter_matrix_1 = self.adapter_matrix_1.repeat((mu_, 1, 1))
        self.adapter_matrix_2 = self.adapter_matrix_2.repeat((mu_, 1, 1))
        self.adapter_bias_1 = self.adapter_bias_1.repeat((mu_, 1, 1))
        self.adapter_bias_2 = self.adapter_bias_2.repeat((mu_, 1, 1))

    def _reset_head_and_adapter(self, num_classes, t):
        self.head_matrix = torch.randn((1, num_classes, self.S), device=self.device) * self.std
        self.head_bias = torch.randn((1, 1, num_classes), device=self.device) * self.std
        self.adapter_matrix_1 = torch.zeros((1, self.hidden_size, self.S), device=self.device)
        self.adapter_matrix_2 = torch.zeros((1, self.S, self.hidden_size), device=self.device)
        self.adapter_bias_1 = torch.zeros((1, 1, self.hidden_size), device=self.device)
        self.adapter_bias_2 = torch.zeros((1, 1, self.S), device=self.device)

    def __len__(self):
        return self.matrix_1.shape[0]

    @torch.no_grad()
    def __call__(self, images):
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)

        adapter = (self.matrix_1, self.bias_1, self.matrix_2, self.bias_2)
        x = self.vit(images, adapter)
        # apply bottleneck
        # x = self.matrix_3.unsqueeze(1) @ x.unsqueeze(3)
        # x = x.squeeze(3)
        # x += self.bias_3.unsqueeze(1)

        x = nn.functional.normalize(x, dim=2)
        return x  # Population x Batch x Features

    @torch.no_grad()
    def calculate_logits(self, features):
        x = self.head_matrix.unsqueeze(1) @ features.unsqueeze(3)
        logits = x + self.head_bias.unsqueeze(3)
        return logits.squeeze_(3)  # Population x Batch x Classes

    @torch.no_grad()
    def adapt(self, features):
        x = self.adapter_matrix_1.unsqueeze(1) @ features.unsqueeze(3)
        x += self.adapter_bias_1.unsqueeze(3)
        x = self.activation(x)
        x = self.adapter_matrix_2.unsqueeze(1) @ x
        x = x.squeeze(3) + self.adapter_bias_2
        return nn.functional.normalize(features + x, dim=2)  # Population x Batch x Features

    @torch.no_grad()
    def create_offspring(self, t, lambda_, lr, mutate_only_head=False, mutate_feature_extractor=True):
        # Create children by interpolation
        mu_ = len(self)
        mothers = torch.randint(0, mu_, (lambda_,), device=self.device)
        fathers = torch.randint(0, mu_, (lambda_,), device=self.device)
        interpolation_mothers = torch.rand((lambda_, 1, 1), device=self.device)
        interpolation_fathers = 1 - interpolation_mothers

        self.matrix_1 = interpolation_mothers * self.matrix_1[mothers] + interpolation_fathers * self.matrix_1[fathers]
        self.matrix_2 = interpolation_mothers * self.matrix_2[mothers] + interpolation_fathers * self.matrix_2[fathers]
        self.matrix_3 = interpolation_mothers * self.matrix_3[mothers] + interpolation_fathers * self.matrix_3[fathers]

        self.adapter_matrix_1 = interpolation_mothers * self.adapter_matrix_1[mothers] + interpolation_fathers * self.adapter_matrix_1[fathers]
        self.adapter_matrix_2 = interpolation_mothers * self.adapter_matrix_2[mothers] + interpolation_fathers * self.adapter_matrix_2[fathers]
        self.adapter_bias_1 = interpolation_mothers * self.adapter_bias_1[mothers] + interpolation_fathers * self.adapter_bias_1[fathers]
        self.adapter_bias_2 = interpolation_mothers * self.adapter_bias_2[mothers] + interpolation_fathers * self.adapter_bias_2[fathers]
        self.head_matrix = interpolation_mothers * self.head_matrix[mothers] + interpolation_fathers * self.head_matrix[fathers]
        self.head_bias = interpolation_mothers * self.head_bias[mothers] + interpolation_fathers * self.head_bias[fathers]

        interpolation_mothers.squeeze_(2)
        interpolation_fathers.squeeze_(2)
        self.bias_1 = interpolation_mothers * self.bias_1[mothers] + interpolation_fathers * self.bias_1[fathers]
        self.bias_2 = interpolation_mothers * self.bias_2[mothers] + interpolation_fathers * self.bias_2[fathers]
        self.bias_3 = interpolation_mothers * self.bias_3[mothers] + interpolation_fathers * self.bias_3[fathers]

        # Mutate children
        if mutate_only_head:
            self.head_matrix += torch.randn_like(self.head_matrix) * lr * 1000
            self.head_bias += torch.randn_like(self.head_bias) * lr * 1000
            if t == 0:
                self.matrix_3 += torch.randn_like(self.matrix_3) * lr
                self.bias_3 += torch.randn_like(self.bias_3) * lr
            return

        if mutate_feature_extractor:
            self.matrix_1 += torch.randn_like(self.matrix_1) * lr / self.F ** 2
            self.bias_1 += torch.randn_like(self.bias_1) * lr / self.F ** 2
            self.matrix_2 += torch.randn_like(self.matrix_2) * lr / self.F
            self.bias_2 += torch.randn_like(self.bias_2) * lr / self.F
            self.matrix_3 += torch.randn_like(self.matrix_3) * lr
            self.bias_3 += torch.randn_like(self.bias_3) * lr

        self.head_matrix += torch.randn_like(self.head_matrix) * lr * 100
        self.head_bias += torch.randn_like(self.head_bias) * lr * 100

        self.adapter_matrix_1 += torch.randn_like(self.adapter_matrix_1) * lr / self.F
        self.adapter_matrix_2 += torch.randn_like(self.adapter_matrix_2) * lr
        self.adapter_bias_1 += torch.randn_like(self.adapter_bias_1) * lr / self.F
        self.adapter_bias_2 += torch.randn_like(self.adapter_bias_2) * lr

    @torch.no_grad()
    def calculate_loss(self, features, targets, past_features, past_targets, old_features, wd, only_adapter_loss=False):
        bsz = targets.shape[0]
        p_size = len(self)
        wd_loss = self.calculate_weight_decay(wd)
        if past_features is None:
            logits = self.calculate_logits(features)
            logits = logits.reshape(bsz * p_size, -1)
            ce_loss = nn.functional.cross_entropy(logits, targets.repeat(p_size), reduction="none")
            ce_loss = ce_loss.reshape(p_size, -1).mean(1)
            return ce_loss + wd_loss, ce_loss, [-1.], [-1.], wd_loss

        # Adapter loss
        adapted_old_features = self.adapt(old_features)
        adapter_loss = nn.functional.mse_loss(adapted_old_features, features, reduction="none")
        adapter_loss = self.S * adapter_loss.mean([1, 2])
        # adapter_loss = - nn.functional.cosine_similarity(adapted_old_features, features, dim=2)
        # adapter_loss = adapter_loss.mean(1)
        # if only_adapter_loss:
        #     return self.alpha * adapter_loss + wd_loss, [-1.], [-1.], adapter_loss, wd_loss

        # Cross entropy loss for past features
        past_bsz = past_targets.shape[0]
        adapted_features = self.adapt(past_features.unsqueeze(0))
        past_logits = self.calculate_logits(adapted_features)
        past_logits = past_logits.reshape(past_bsz * p_size, -1)
        approx_ce_loss = nn.functional.cross_entropy(past_logits, past_targets.repeat(p_size), reduction="none")
        approx_ce_loss = approx_ce_loss.reshape(p_size, -1)

        # Cross entropy loss for features from current task
        logits = self.calculate_logits(features)
        logits = logits.reshape(bsz*p_size, -1)
        ce_loss = nn.functional.cross_entropy(logits, targets.repeat(p_size), reduction="none")
        ce_loss = ce_loss.reshape(p_size, -1)
        total_loss = torch.cat((ce_loss, approx_ce_loss), dim=1).mean(1) + self.alpha * adapter_loss + wd_loss
        return total_loss, ce_loss.mean(1), approx_ce_loss.mean(1), adapter_loss, wd_loss

    @torch.no_grad()
    def calculate_weight_decay(self, wd):
        l2 = 0
        for weights in [self.matrix_1, self.matrix_2, self.matrix_3]:
            l2 += weights.pow(2).sum([1, 2])
        for weights in [self.bias_1, self.bias_2, self.bias_3]:
            l2 += weights.pow(2).sum(1)
        return wd * l2

    @torch.no_grad()
    def survive(self, mu_, loss):
        indices = torch.argsort(loss)[:mu_]
        self.matrix_1 = self.matrix_1[indices]
        self.matrix_2 = self.matrix_2[indices]
        self.matrix_3 = self.matrix_3[indices]
        self.bias_1 = self.bias_1[indices]
        self.bias_2 = self.bias_2[indices]
        self.bias_3 = self.bias_3[indices]
        self.head_matrix = self.head_matrix[indices]
        self.head_bias = self.head_bias[indices]
        self.adapter_matrix_1 = self.adapter_matrix_1[indices]
        self.adapter_matrix_2 = self.adapter_matrix_2[indices]
        self.adapter_bias_1 = self.adapter_bias_1[indices]
        self.adapter_bias_2 = self.adapter_bias_2[indices]

    def load_weights_from_path(self, path):
        state_dict = torch.load("dino_deitsmall16_pretrain.pth")
        self.vit.load_state_dict(state_dict, strict=True)
        self.vit.eval()
        self.vit.to(self.device)

    @torch.no_grad()
    def finish_training(self):
        self.matrix_1 = self.matrix_1[:1]
        self.matrix_2 = self.matrix_2[:1]
        self.matrix_3 = self.matrix_3[:1]
        self.bias_1 = self.bias_1[:1]
        self.bias_2 = self.bias_2[:1]
        self.bias_3 = self.bias_3[:1]
        self.head_matrix = self.head_matrix[:1]
        self.head_bias = self.head_bias[:1]
        self.adapter_matrix_1 = self.adapter_matrix_1[:1]
        self.adapter_matrix_2 = self.adapter_matrix_2[:1]
        self.adapter_bias_1 = self.adapter_bias_1[:1]
        self.adapter_bias_2 = self.adapter_bias_2[:1]


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, adapter):
        if adapter is not None:
            matrix_1, bias_1, matrix_2, bias_2 = adapter
            z = matrix_1.unsqueeze(1).unsqueeze(1) @ x.unsqueeze(3).unsqueeze(0)
            z = z.squeeze(4)
            z += bias_1.unsqueeze(1).unsqueeze(1)
            z = nn.functional.relu(z)
            z = matrix_2.unsqueeze(1).unsqueeze(1) @ z.unsqueeze(4)
            z = z.squeeze(4)
            z += bias_2.unsqueeze(1).unsqueeze(1)
            x = x.unsqueeze(0) + z
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, adapter=None):
        y, attn = self.attn(self.norm1(x))
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x), adapter))

        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer """
    def __init__(self, img_size=[224], patch_size=16, in_chans=3, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, num_features=64, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_features = num_features

        self.patch_embed = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)

        return self.pos_drop(x)

    def forward(self, x, adapter):
        x = self.prepare_tokens(x)
        for blk in self.blocks[:-1]:
            x = blk(x)
        # x = self.norm(x)
        x = self.blocks[-1](x, adapter)
        return x[:, :, 0]




