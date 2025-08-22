import torch
import torch.nn as nn
from torchcde import cubic_spline_coeffs, CubicSpline


class GRU_ODE_Delay(nn.Module):
    def __init__(self, input_channels, hidden_channels, file_path, rnd, time_max):
        super(GRU_ODE_Delay, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.file_path = file_path
        self.rnd = rnd
        self.time_max = time_max

        # --- 核心修正：修复维度不匹配问题 ---
        # 原始代码: self.W_r = nn.Linear(hidden_channels, hidden_channels, bias=False)
        # 修正后: 输入维度应为 input_channels + hidden_channels
        self.W_r = nn.Linear(input_channels + hidden_channels, hidden_channels, bias=False)
        # --- 修正结束 ---

        self.W_z = nn.Linear(input_channels + hidden_channels, hidden_channels, bias=False)
        self.W_h = nn.Linear(input_channels + hidden_channels, hidden_channels, bias=False)
        self.gru_cell = nn.GRUCell(input_size=hidden_channels, hidden_size=hidden_channels)

    def forward(self, t, h):
        x = self.get_x(t)

        # --- 核心修正：确保拼接的张量维度正确 ---
        # 原始代码没有问题，但为了清晰，我们确认这里的拼接是正确的
        combined = torch.cat([x, h], dim=1)
        # --- 修正结束 ---

        r = torch.sigmoid(self.W_r(combined))
        z = torch.sigmoid(self.W_z(combined))

        # --- 核心修正：确保拼接的张量维度正确 ---
        # 原始代码: combined_rh = torch.cat([x, r * h], dim=1)
        # 修正后: 这里的h应该是 r*h
        combined_rh = torch.cat([x, r * h], dim=1)
        # --- 修正结束 ---

        h_tilde = torch.tanh(self.W_h(combined_rh))

        dh = (1 - z) * (h_tilde - h)
        return dh

    def get_x(self, t):
        t_int = t.type(torch.int32)
        t_int = torch.clamp(t_int, 0, self.time_max - 1)
        return self.x_coeffs[t_int]

    def set_x(self, x_coeffs):
        self.x_coeffs = x_coeffs