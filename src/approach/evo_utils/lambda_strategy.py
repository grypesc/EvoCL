import torch
import torch.nn as nn
import torch.nn.functional as F


class LambdaStrategy:
    def __init__(self, S, F, alpha, adapter_size, device):
        self.S = S
        self.F = F
        self.alpha = alpha
        self.adapter_size = adapter_size
        self.device = device
        self.std = 0.01
        self.matrix_1 = torch.zeros((1, S, 3, 3, 3), device=device)
        self.bias_1 = torch.zeros((1, S, ), device=device)
        self.matrix_2 = torch.zeros((1, S, S, 3, 3), device=device)
        self.bias_2 = torch.zeros((1, S, ), device=device)
        self.matrix_3 = torch.zeros((1, S, S, 3, 3), device=device)
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
        self.matrix_1 = self.matrix_1.repeat((mu_, 1, 1, 1, 1))
        self.bias_1 = self.bias_1.repeat((mu_, 1))
        self.matrix_2 = self.matrix_2.repeat((mu_, 1, 1, 1, 1))
        self.bias_2 = self.bias_2.repeat((mu_, 1))
        self.matrix_3 = self.matrix_3.repeat((mu_, 1, 1, 1, 1))
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
        self.adapter_matrix_1 = torch.zeros((1, self.adapter_size, self.S), device=self.device)
        self.adapter_matrix_2 = torch.zeros((1, self.S, self.adapter_size), device=self.device)
        self.adapter_bias_1 = torch.zeros((1, 1, self.adapter_size), device=self.device)
        self.adapter_bias_2 = torch.zeros((1, 1, self.S), device=self.device)

    def __len__(self):
        return self.matrix_1.shape[0]

    @staticmethod
    def evo_conv2d(x, weights, bias, pop_size):
        weights = weights.reshape(-1, *weights.shape[2:])
        bias = bias.reshape(-1)
        x = F.conv2d(x, weights, bias, padding=1, groups=pop_size)
        return x

    @torch.no_grad()
    def __call__(self, images):
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)

        bsz = images.shape[0]
        pop_size = self.matrix_1.shape[0]
        x = self.evo_conv2d(images, self.matrix_1, self.bias_1, 1)
        x = self.activation(x)
        x = self.evo_conv2d(x, self.matrix_2, self.bias_2, pop_size)
        x = self.activation(x)
        x = self.evo_conv2d(x, self.matrix_3, self.bias_3, pop_size)
        x = self.activation(x)
        x = x.mean([2, 3])
        x = x.reshape(bsz, -1, self.S)
        x = x.permute(1, 0, 2)
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
    def create_offspring(self, t, lambda_, lr, mutate_only_head=False, mutate_feature_extractor=False):
        # Create children by interpolation
        mu_ = len(self)
        mothers = torch.randint(0, mu_, (lambda_,), device=self.device)
        fathers = torch.randint(0, mu_, (lambda_,), device=self.device)
        interpolation_mothers = torch.rand((lambda_, 1, 1, 1, 1), device=self.device)
        interpolation_fathers = 1 - interpolation_mothers

        self.matrix_1 = interpolation_mothers * self.matrix_1[mothers] + interpolation_fathers * self.matrix_1[fathers]
        self.matrix_2 = interpolation_mothers * self.matrix_2[mothers] + interpolation_fathers * self.matrix_2[fathers]
        self.matrix_3 = interpolation_mothers * self.matrix_3[mothers] + interpolation_fathers * self.matrix_3[fathers]

        interpolation_mothers.squeeze_(2).squeeze_(2)
        interpolation_fathers.squeeze_(2).squeeze_(2)
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
            l2 += weights.pow(2).sum([1, 2, 3, 4])
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
        state_dict = torch.load(path)
        self.matrix_1 = state_dict['backbone.0.weight'].unsqueeze(0)
        self.matrix_2 = state_dict['backbone.2.weight'].unsqueeze(0)
        self.matrix_3 = state_dict['backbone.4.weight'].unsqueeze(0)
        self.head_matrix = state_dict['head.weight'].unsqueeze(0)

        self.bias_1 = state_dict['backbone.0.bias'].unsqueeze(0)
        self.bias_2 = state_dict['backbone.2.bias'].unsqueeze(0)
        self.bias_3 = state_dict['backbone.4.bias'].unsqueeze(0)
        self.head_bias = state_dict['head.bias'].unsqueeze(0).unsqueeze(0)

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

