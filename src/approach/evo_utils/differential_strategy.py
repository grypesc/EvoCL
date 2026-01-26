import torch
import torch.nn as nn

from .lambda_strategy import LambdaStrategy


class DifferentialStrategy(LambdaStrategy):
    def create_offspring(self, lambda_, lr, mutate_only_head=False):
        mu_ = len(self)
        base = torch.randint(0, mu_, (lambda_,), device=self.device)
        mothers = torch.randint(0, mu_, (lambda_,), device=self.device)
        fathers = torch.randint(0, mu_, (lambda_,), device=self.device)

        F = self.F
        self.matrix_1 = self.matrix_1[base] + F * (self.matrix_1[mothers] - self.matrix_1[fathers])
        self.matrix_2 = self.matrix_2[base] + F * (self.matrix_2[mothers] - self.matrix_2[fathers])
        self.head_matrix = self.head_matrix[base] + F * (self.head_matrix[mothers] - self.head_matrix[fathers])
        self.adapter_matrix = self.adapter_matrix[base] + F * (self.adapter_matrix[mothers] - self.adapter_matrix[fathers])

        self.bias_1 = self.bias_1[base] + F * (self.bias_1[mothers] - self.bias_1[fathers])
        self.bias_2 = self.bias_2[base] + F * (self.bias_2[mothers] - self.bias_2[fathers])
        self.head_bias = self.head_bias[base] + F * (self.head_bias[mothers] - self.head_bias[fathers])
        self.adapter_bias = self.adapter_bias[base] + F * (self.adapter_bias[mothers] - self.adapter_bias[fathers])

        # Mutate children
        if mutate_only_head:
            self.head_matrix += torch.randn_like(self.head_matrix) * lr * 100
            self.head_bias += torch.randn_like(self.head_bias) * lr * 100
        else:
            self.matrix_1 += torch.randn_like(self.matrix_1) * lr / 3
            self.bias_1 += torch.randn_like(self.bias_1) * lr / 3
            self.matrix_2 += torch.randn_like(self.matrix_2) * lr
            self.bias_2 += torch.randn_like(self.bias_2) * lr
            self.adapter_matrix += torch.randn_like(self.adapter_matrix) * lr
            self.adapter_bias += torch.randn_like(self.adapter_bias) * lr
            self.head_matrix += torch.randn_like(self.head_matrix) * lr * 10
            self.head_bias += torch.randn_like(self.head_bias) * lr * 10

