import torch
import torch.nn as nn

from .lambda_strategy import LambdaStrategy


class MuPlusLambdaStrategy(LambdaStrategy):
    def create_offspring(self, lambda_, lr, mutate_only_head=False):
        # Create children by interpolation
        mu_ = len(self)
        mothers = torch.randint(0, mu_, (lambda_,), device=self.device)
        fathers = torch.randint(0, mu_, (lambda_,), device=self.device)
        interpolation_mothers = torch.rand((lambda_, 1, 1), device=self.device)
        interpolation_fathers = 1 - interpolation_mothers

        c_matrix_1 = interpolation_mothers * self.matrix_1[mothers] + interpolation_fathers * self.matrix_1[fathers]
        c_matrix_2 = interpolation_mothers * self.matrix_2[mothers] + interpolation_fathers * self.matrix_2[fathers]
        c_head_matrix = interpolation_mothers * self.head_matrix[mothers] + interpolation_fathers * self.head_matrix[fathers]
        c_adapter_matrix = interpolation_mothers * self.adapter_matrix[mothers] + interpolation_fathers * self.adapter_matrix[fathers]

        c_bias_1 = interpolation_mothers * self.bias_1[mothers] + interpolation_fathers * self.bias_1[fathers]
        c_bias_2 = interpolation_mothers * self.bias_2[mothers] + interpolation_fathers * self.bias_2[fathers]
        c_head_bias = interpolation_mothers * self.head_bias[mothers] + interpolation_fathers * self.head_bias[fathers]
        c_adapter_bias = interpolation_mothers * self.adapter_bias[mothers] + interpolation_fathers * self.adapter_bias[fathers]

        # Mutate children
        if mutate_only_head:
            c_head_matrix += torch.randn_like(c_head_matrix) * lr * 100
            c_head_bias += torch.randn_like(c_head_bias) * lr * 100
        else:
            c_matrix_1 += torch.randn_like(c_matrix_1) * lr / 3
            c_bias_1 += torch.randn_like(c_bias_1) * lr / 3
            c_matrix_2 += torch.randn_like(c_matrix_2) * lr
            c_bias_2 += torch.randn_like(c_bias_2) * lr
            c_adapter_matrix += torch.randn_like(c_adapter_matrix) * lr
            c_adapter_bias += torch.randn_like(c_adapter_bias) * lr
            c_head_matrix += torch.randn_like(c_head_matrix) * lr * 10
            c_head_bias += torch.randn_like(c_head_bias) * lr * 10

        # Concatenate parents with offsprings
        self.matrix_1 = torch.cat((self.matrix_1, c_matrix_1), dim=0)
        self.matrix_2 = torch.cat((self.matrix_2, c_matrix_2), dim=0)
        self.head_matrix = torch.cat((self.head_matrix, c_head_matrix), dim=0)
        self.adapter_matrix = torch.cat((self.adapter_matrix, c_adapter_matrix), dim=0)
        self.bias_1 = torch.cat((self.bias_1, c_bias_1), dim=0)
        self.bias_2 = torch.cat((self.bias_2, c_bias_2), dim=0)
        self.head_bias = torch.cat((self.head_bias, c_head_bias), dim=0)
        self.adapter_bias = torch.cat((self.adapter_bias, c_adapter_bias), dim=0)
