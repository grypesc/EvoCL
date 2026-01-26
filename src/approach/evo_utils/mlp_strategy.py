import torch
import torch.nn as nn
from .lambda_strategy import LambdaStrategy


class MLPStrategy(LambdaStrategy):
    def __init__(self, input_size, S, F, alpha, device):
        super().__init__(input_size, S, F, alpha, device)
        self.adapter_matrix_1 = None
        self.adapter_matrix_2 = None
        self.adapter_bias_1 = None
        self.adapter_bias_2 = None

    def adapt(self, features):
        x = self.adapter_matrix_1.unsqueeze(1) @ features.unsqueeze(3)
        x += self.adapter_bias_1.unsqueeze(3)
        x = nn.functional.gelu(x)
        x = self.adapter_matrix_2.unsqueeze(1) @ x
        x = x.squeeze(3) + self.adapter_bias_2
        x += nn.functional.normalize(features, dim=2)
        return nn.functional.normalize(x, dim=2)  # Population x Batch x Features

    def prepare_for_training(self, mu_, num_classes, t):
        self.matrix_1 = self.matrix_1.repeat((mu_, 1, 1, 1, 1))
        # self.bias_1 = self.bias_1.repeat((mu_, 1, 1))
        self.matrix_2 = self.matrix_2.repeat((mu_, 1, 1, 1, 1))
        # self.bias_2 = self.bias_2.repeat((mu_, 1, 1))

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
        self.adapter_matrix_1 = torch.zeros((1, 64, self.S), device=self.device)
        self.adapter_matrix_2 = torch.zeros((1, self.S, 64), device=self.device)
        self.adapter_bias_1 = torch.zeros((1, 1, 64), device=self.device)
        self.adapter_bias_2 = torch.zeros((1, 1, self.S), device=self.device)

    def create_offspring(self, lambda_, lr, mutate_only_head=False, train_only_adapter=False):
        # Create children by interpolation
        mu_ = len(self)
        mothers = torch.randint(0, mu_, (lambda_,), device=self.device)
        fathers = torch.randint(0, mu_, (lambda_,), device=self.device)
        interpolation_mothers = torch.rand((lambda_, 1, 1), device=self.device)
        interpolation_fathers = 1 - interpolation_mothers

        self.matrix_1 = interpolation_mothers.unsqueeze(1).unsqueeze(1) * self.matrix_1[mothers] + interpolation_fathers.unsqueeze(1).unsqueeze(1) * self.matrix_1[fathers]
        # self.bias_1 = interpolation_mothers * self.bias_1[mothers] + interpolation_fathers * self.bias_1[fathers]
        self.matrix_2 = interpolation_mothers.unsqueeze(1).unsqueeze(1) * self.matrix_2[mothers] + interpolation_fathers.unsqueeze(1).unsqueeze(1) * self.matrix_2[fathers]
        # self.bias_2 = interpolation_mothers * self.bias_2[mothers] + interpolation_fathers * self.bias_2[fathers]
        self.adapter_matrix_1 = interpolation_mothers * self.adapter_matrix_1[mothers] + interpolation_fathers * self.adapter_matrix_1[fathers]
        self.adapter_matrix_2 = interpolation_mothers * self.adapter_matrix_2[mothers] + interpolation_fathers * self.adapter_matrix_2[fathers]
        self.adapter_bias_1 = interpolation_mothers * self.adapter_bias_1[mothers] + interpolation_fathers * self.adapter_bias_1[fathers]
        self.adapter_bias_2 = interpolation_mothers * self.adapter_bias_2[mothers] + interpolation_fathers * self.adapter_bias_2[fathers]
        self.head_matrix = interpolation_mothers * self.head_matrix[mothers] + interpolation_fathers * self.head_matrix[fathers]
        self.head_bias = interpolation_mothers * self.head_bias[mothers] + interpolation_fathers * self.head_bias[fathers]

        # Mutate children
        if mutate_only_head:
            self.head_matrix += torch.randn_like(self.head_matrix) * lr * 100
            self.head_bias += torch.randn_like(self.head_bias) * lr * 100
        elif train_only_adapter:
            self.adapter_matrix_1 += torch.randn_like(self.adapter_matrix_1) * lr * self.F / 3
            self.adapter_matrix_2 += torch.randn_like(self.adapter_matrix_2) * lr * self.F
            self.adapter_bias_1 += torch.randn_like(self.adapter_bias_1) * lr * self.F / 3
            self.adapter_bias_2 += torch.randn_like(self.adapter_bias_2) * lr * self.F
        else:
            self.matrix_1 += torch.randn_like(self.matrix_1) * lr / 3
            # self.bias_1 += torch.randn_like(self.bias_1) * lr / 3
            self.matrix_2 += torch.randn_like(self.matrix_2) * lr
            # self.bias_2 += torch.randn_like(self.bias_2) * lr
            self.adapter_matrix_1 += torch.randn_like(self.adapter_matrix_1) * lr * self.F / 3
            self.adapter_matrix_2 += torch.randn_like(self.adapter_matrix_2) * lr * self.F
            self.adapter_bias_1 += torch.randn_like(self.adapter_bias_1) * lr * self.F / 3
            self.adapter_bias_2 += torch.randn_like(self.adapter_bias_2) * lr * self.F
            self.head_matrix += torch.randn_like(self.head_matrix) * lr * 3
            self.head_bias += torch.randn_like(self.head_bias) * lr * 3

    def survive(self, mu_, loss):
        indices = torch.argsort(loss)[:mu_]
        self.matrix_1 = self.matrix_1[indices]
        self.matrix_2 = self.matrix_2[indices]
        self.head_matrix = self.head_matrix[indices]
        self.adapter_matrix_1 = self.adapter_matrix_1[indices]
        self.adapter_matrix_2 = self.adapter_matrix_2[indices]
        self.adapter_bias_1 = self.adapter_bias_1[indices]
        self.adapter_bias_2 = self.adapter_bias_2[indices]
        # self.bias_1 = self.bias_1[indices]
        # self.bias_2 = self.bias_2[indices]
        self.head_bias = self.head_bias[indices]

    def finish_training(self):
        self.matrix_1 = self.matrix_1[:1]
        self.matrix_2 = self.matrix_2[:1]
        self.head_matrix = self.head_matrix[:1]
        self.adapter_matrix_1 = self.adapter_matrix_1[:1]
        self.adapter_matrix_2 = self.adapter_matrix_2[:1]
        self.adapter_bias_1 = self.adapter_bias_1[:1]
        self.adapter_bias_2 = self.adapter_bias_2[:1]
        # self.bias_1 = self.bias_1[:1]
        # self.bias_2 = self.bias_2[:1]
        self.head_bias = self.head_bias[:1]

    def calculate_weight_decay(self, wd):
        l2 = 0
        for weights in [self.head_matrix, self.head_bias, self.adapter_matrix_1, self.adapter_matrix_2, self.adapter_bias_1, self.adapter_bias_2]:
            l2 += weights.pow(2).sum([1, 2])
        l2 += self.matrix_1.pow(2).sum([1, 2, 3, 4])
        l2 += self.matrix_2.pow(2).sum([1, 2, 3, 4])
        return wd * l2

    def calculate_loss(self, features, targets, past_features, past_targets, old_features, wd, only_adapter_loss=False):
        bsz = targets.shape[0]
        p_size = len(self)
        wd_loss = self.calculate_weight_decay(wd)
        if past_features is None:
            raise NotImplementedError()

        # Adapter loss
        adapted_old_features = self.adapt(old_features)
        # adapter_loss = nn.functional.mse_loss(adapted_old_features, features, reduction="none")
        # adapter_loss = self.S * adapter_loss.mean([1, 2])
        adapter_loss = - nn.functional.cosine_similarity(adapted_old_features, features, dim=2)
        adapter_loss = adapter_loss.mean(1)
        if only_adapter_loss:
            return self.alpha * adapter_loss + wd_loss, [-1.], [-1.], adapter_loss, wd_loss
        # Cross entropy loss
        adapted_features = self.adapt(past_features.unsqueeze(0))
        logits = self.calculate_logits(torch.cat((features, adapted_features), dim=1))
        logits = logits.reshape(2*bsz*p_size, -1)
        ce_loss = nn.functional.cross_entropy(logits, torch.cat((targets, past_targets)).repeat(p_size), reduction="none")
        ce_loss = ce_loss.reshape(p_size, -1).mean(1)
        total_loss = ce_loss + self.alpha * adapter_loss + wd_loss
        return total_loss, ce_loss, [-1.], adapter_loss, wd_loss
