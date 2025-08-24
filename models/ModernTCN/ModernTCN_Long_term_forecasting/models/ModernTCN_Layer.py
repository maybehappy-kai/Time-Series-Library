#__all__ = ['Transpose', 'get_activation_fn', 'moving_avg', 'series_decomp', 'PositionalEncoding', 'SinCosPosEncoding', 'Coord2dPosEncoding', 'Coord1dPosEncoding', 'positional_encoding']
__all__ = ['moving_avg', 'series_decomp',  'Flatten_Head']
import torch
from torch import nn
import math
# decomposition

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


# forecast task head
class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        # print(f"\n[Flatten_Head INIT] Initializing Head...")
        # print(
        #     f"[Flatten_Head INIT] individual: {individual}, n_vars: {n_vars}, nf: {nf}, target_window: {target_window}\n")

        self.individual = individual
        self.n_vars = n_vars

        if self.individual:
            # 您当前的使用场景不会进入这个分支，但我们仍然保持其完整性
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            # 这是您实际使用的分支
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        # --- 内部法医级检查 ---
        # print("\n      [HEAD_FWD] --- 进入 Flatten_Head.forward ---")
        # print(f"      [HEAD_FWD] 接收到的 x 维度: {x.shape}")
        # print(f"      [HEAD_FWD] 接收到的 x 是否连续: {x.is_contiguous()}")

        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:, i, :, :])
                z = self.linears[i](z)
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)
        else:
            # 逐行执行并打印日志
            # print(f"      [HEAD_FWD] 即将执行 flatten 操作...")
            x = self.flatten(x)
            # print(f"      [HEAD_FWD] Flatten 操作后维度: {x.shape}")
            # print(f"      [HEAD_FWD] Flatten 操作后是否连续: {x.is_contiguous()}")

            # print(f"      [HEAD_FWD] 即将执行 linear 操作...")
            x = self.linear(x)
            # print(f"      [HEAD_FWD] Linear 操作后维度: {x.shape}")

            # print(f"      [HEAD_FWD] 即将执行 dropout 操作...")
            x = self.dropout(x)
            # print(f"      [HEAD_FWD] Dropout 操作后维度: {x.shape}")

        # print("      [HEAD_FWD] --- 成功退出 Flatten_Head.forward ---\n")
        return x