import torch
import torch.nn as nn
import torch.nn.functional as F

# === DUpsampling ===
class DUpsampling(nn.Module):
    def __init__(self, inplanes, scale, num_class, pad=0):
        super(DUpsampling, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, num_class * scale * scale, kernel_size=1, padding=pad, bias=False)
        self.scale = scale

    def forward(self, x):
        x = self.conv1(x)
        N, C, H, W = x.size()
        x_permuted = x.permute(0, 2, 3, 1)
        x_permuted = x_permuted.contiguous().view(N, H, W * self.scale, C // self.scale)
        x_permuted = x_permuted.permute(0, 2, 1, 3)
        x_permuted = x_permuted.contiguous().view(N, W * self.scale, H * self.scale, C // (self.scale * self.scale))
        x = x_permuted.permute(0, 3, 2, 1)
        return x
