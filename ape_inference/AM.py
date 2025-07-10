import torch
import torch.nn as nn
import torch.nn.functional as F

# === Attention Mechanisms ===
def l2_norm(x):
    return torch.einsum("bcn, bn->bcn", x, 1 / torch.norm(x, p=2, dim=-2))

class PositionLinearAttention(nn.Module):
    def __init__(self, in_places, eps=1e-6):
        super(PositionLinearAttention, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.query_conv = nn.Conv2d(in_places, in_places // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_places, in_places // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_places, in_places, kernel_size=1)
        self.eps = eps

    def forward(self, x):
        batch_size, chnnels, width, height = x.size()
        Q = self.query_conv(x).view(batch_size, -1, width * height)
        K = self.key_conv(x).view(batch_size, -1, width * height)
        V = self.value_conv(x).view(batch_size, -1, width * height)

        Q = l2_norm(Q).permute(0, 2, 1)
        K = l2_norm(K)

        tailor_sum = 1 / (width * height + torch.einsum("bnc, bc->bn", Q, torch.sum(K, dim=-1) + self.eps))
        value_sum = torch.einsum("bcn->bc", V).unsqueeze(-1).expand(-1, chnnels, width * height)

        matrix = torch.einsum('bmn, bcn->bmc', K, V)
        matrix_sum = value_sum + torch.einsum("bnm, bmc->bcn", Q, matrix)

        weight_value = torch.einsum("bcn, bn->bcn", matrix_sum, tailor_sum)
        weight_value = weight_value.view(batch_size, chnnels, width, height)

        return (x + self.gamma * weight_value).contiguous()

class ChannelLinearAttention(nn.Module):
    def __init__(self, eps=1e-6):
        super(ChannelLinearAttention, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.eps = eps

    def forward(self, x):
        batch_size, chnnels, width, height = x.size()
        Q = x.view(batch_size, chnnels, -1)
        K = x.view(batch_size, chnnels, -1)
        V = x.view(batch_size, chnnels, -1)

        Q = l2_norm(Q)
        K = l2_norm(K).permute(0, 2, 1)

        tailor_sum = 1 / (width * height + torch.einsum("bnc, bn->bc", K, torch.sum(Q, dim=-2) + self.eps))
        value_sum = torch.einsum("bcn->bn", V).unsqueeze(-1).permute(0, 2, 1).expand(-1, chnnels, width * height)
        matrix = torch.einsum('bcn, bnm->bcm', V, K)
        matrix_sum = value_sum + torch.einsum("bcm, bmn->bcn", matrix, Q)

        weight_value = torch.einsum("bcn, bc->bcn", matrix_sum, tailor_sum)
        weight_value = weight_value.view(batch_size, chnnels, height, width)

        return (x + self.gamma * weight_value).contiguous()

class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super(AttentionBlock, self).__init__()
        self.position_attention = PositionLinearAttention(channels)
        self.channel_attention = ChannelLinearAttention()

    def forward(self, x):
        x = self.position_attention(x)
        x = self.channel_attention(x)
        return x