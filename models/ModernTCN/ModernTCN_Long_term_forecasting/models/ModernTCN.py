import torch
from torch import nn
import torch.nn.functional as F
import math
from ..layers.RevIN import RevIN
from ..models.ModernTCN_Layer import series_decomp, Flatten_Head


class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-6, data_format="channels_last"):
        super(LayerNorm, self).__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        B, M, D, N = x.shape
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(B * M, N, D)
        x = self.norm(x)
        x = x.reshape(B, M, N, D)
        x = x.permute(0, 1, 3, 2)
        return x

def get_conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias):
    # +++ 启用最终的监控探针 PART 4 +++
    # print("\n" + "!!!" * 15)
    # print("[CONV1D_DEBUG] Attempting to create nn.Conv1d with:")
    # print(f"[CONV1D_DEBUG] in_channels   = {in_channels} (Type: {type(in_channels)})")
    # print(f"[CONV1D_DEBUG] out_channels  = {out_channels} (Type: {type(out_channels)})")
    # print(f"[CONV1D_DEBUG] kernel_size   = {kernel_size} (Type: {type(kernel_size)})")
    # print(f"[CONV1D_DEBUG] stride        = {stride} (Type: {type(stride)})")
    # print(f"[CONV1D_DEBUG] padding       = {padding} (Type: {type(padding)})")
    # print(f"[CONV1D_DEBUG] dilation      = {dilation} (Type: {type(dilation)})")
    # print(f"[CONV1D_DEBUG] groups        = {groups} (Type: {type(groups)})")
    # print(f"[CONV1D_DEBUG] bias          = {bias} (Type: {type(bias)})")
    # print("!!!" * 15 + "\n")
    # +++ 监控结束 +++
    return nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=bias)


def get_bn(channels):
    return nn.BatchNorm1d(channels)

def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1,bias=False):
    if padding is None:
        padding = kernel_size // 2
    result = nn.Sequential()
    result.add_module('conv', get_conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias))
    result.add_module('bn', get_bn(out_channels))
    return result

def fuse_bn(conv, bn):

    kernel = conv.weight
    running_mean = bn.running_mean
    running_var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps
    std = (running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1)
    return kernel * t, beta - running_mean * gamma / std

class ReparamLargeKernelConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, groups,
                 small_kernel,
                 small_kernel_merged=False, nvars=7):
        super(ReparamLargeKernelConv, self).__init__()

        # +++ 启用并增强所有调试探针 +++
        # print("\n" + " R " * 20)
        # print("[REPARAM_INIT] Entering ReparamLargeKernelConv.__init__")
        # print(f"[REPARAM_INIT] in_channels: {in_channels} (type: {type(in_channels)})")
        # print(f"[REPARAM_INIT] out_channels: {out_channels} (type: {type(out_channels)})")
        # print(f"[REPARAM_INIT] kernel_size: {kernel_size} (type: {type(kernel_size)})")
        # print(f"[REPARAM_INIT] stride: {stride} (type: {type(stride)})")
        # # 保持您上一轮的修改，这在逻辑上仍然是更合理的
        # print(f"[REPARAM_INIT] groups: {groups} (type: {type(groups)})")
        # print(f"[REPARAM_INIT] small_kernel: {small_kernel} (type: {type(small_kernel)})")
        # print(f"[REPARAM_INIT] small_kernel_merged: {small_kernel_merged}")
        # print(" R " * 20 + "\n")
        # +++ 探针结束 +++

        self.kernel_size = kernel_size
        self.small_kernel = small_kernel
        # 确保 padding 的计算结果也是原生 int
        padding = int(kernel_size) // 2

        if small_kernel_merged:
            # +++ 最终的、决定性的修复 +++
            # 在调用前，将所有参数强制转换为 Python 原生 int 类型

            p_in_channels = int(in_channels)
            p_out_channels = int(out_channels)
            p_kernel_size = int(kernel_size)
            p_stride = int(stride)
            p_padding = int(padding)
            p_groups = int(groups)

            # +++ 新增的、最关键的调试探针：最终参数检查 +++
            # print("\n" + "!!!" * 20)
            # print("[CRITICAL CHECK] Preparing to call direct nn.Conv1d inside `if small_kernel_merged:`")
            # print(f"[CRITICAL CHECK] --> in_channels:  Value={p_in_channels}, Type={type(p_in_channels)}")
            # print(f"[CRITICAL CHECK] --> out_channels: Value={p_out_channels}, Type={type(p_out_channels)}")
            # print(f"[CRITICAL CHECK] --> kernel_size:  Value={p_kernel_size}, Type={type(p_kernel_size)}")
            # print(f"[CRITICAL CHECK] --> stride:       Value={p_stride}, Type={type(p_stride)}")
            # print(f"[CRITICAL CHECK] --> padding:      Value={p_padding}, Type={type(p_padding)}")
            # print(f"[CRITICAL CHECK] --> groups:       Value={p_groups}, Type={type(p_groups)}")
            # print("!!!" * 20 + "\n")
            # +++ 调试探针结束 +++

            # 使用强制转换后的纯净 int 变量来创建 Conv1d
            self.lkb_reparam = nn.Conv1d(in_channels=p_in_channels, out_channels=p_out_channels,
                                         kernel_size=p_kernel_size,
                                         stride=p_stride, padding=p_padding, dilation=1, groups=p_groups, bias=True)
        else:
            # 为了代码的健壮性，同样在这里进行强制转换
            # print("[REPARAM_INIT] Creating lkb_origin (large kernel)...")
            self.lkb_origin = conv_bn(in_channels=int(in_channels), out_channels=int(out_channels),
                                      kernel_size=int(kernel_size),
                                      stride=int(stride), padding=padding, dilation=1, groups=int(groups), bias=False)
            if small_kernel is not None:
                # print("[REPARAM_INIT] Creating small_conv (small kernel)...")
                assert small_kernel <= kernel_size, 'The kernel size for re-param cannot be larger than the large kernel!'
                self.small_conv = conv_bn(in_channels=int(in_channels), out_channels=int(out_channels),
                                          kernel_size=int(small_kernel),
                                          stride=int(stride), padding=int(small_kernel) // 2, groups=int(groups),
                                          dilation=1,
                                          bias=False)

    def forward(self, inputs):
        # +++ 新增的、最关键的 forward 探针 +++
        # print(f"\n      [REPARAM_FWD] --- Entering forward ---")
        # print(f"      [REPARAM_FWD] Input shape: {inputs.shape}")

        if hasattr(self, 'lkb_reparam'):
            # print("      [REPARAM_FWD] >> Executing self.lkb_reparam(inputs)...")
            out = self.lkb_reparam(inputs)
        else:
            # print("      [REPARAM_FWD] >> Executing self.lkb_origin(inputs)... (This is the large kernel)")
            out = self.lkb_origin(inputs)
            if hasattr(self, 'small_conv'):
                # print("      [REPARAM_FWD] >> Adding self.small_conv(inputs) output...")
                out += self.small_conv(inputs)

        # print(f"      [REPARAM_FWD] --- Exiting forward (SUCCESS) ---")
        # print(f"      [REPARAM_FWD] Output shape: {out.shape}")
        return out

    def PaddingTwoEdge1d(self,x,pad_length_left,pad_length_right,pad_values=0):

        D_out,D_in,ks=x.shape
        if pad_values ==0:
            pad_left = torch.zeros(D_out,D_in,pad_length_left)
            pad_right = torch.zeros(D_out,D_in,pad_length_right)
        else:
            pad_left = torch.ones(D_out, D_in, pad_length_left) * pad_values
            pad_right = torch.ones(D_out, D_in, pad_length_right) * pad_values
        x = torch.cat([pad_left,x],dims=-1)
        x = torch.cat([x,pad_right],dims=-1)
        return x

    def get_equivalent_kernel_bias(self):

        eq_k, eq_b = fuse_bn(self.lkb_origin.conv, self.lkb_origin.bn)

        if hasattr(self, 'small_conv'):
            small_k, small_b = fuse_bn(self.small_conv.conv, self.small_conv.bn)

            eq_b += small_b

            eq_k += self.PaddingTwoEdge1d(small_k, (self.kernel_size - self.small_kernel) // 2,
                                          (self.kernel_size - self.small_kernel) // 2, 0)
        return eq_k, eq_b

    def merge_kernel(self):
        eq_k, eq_b = self.get_equivalent_kernel_bias()
        self.lkb_reparam = nn.Conv1d(in_channels=self.lkb_origin.conv.in_channels,
                                     out_channels=self.lkb_origin.conv.out_channels,
                                     kernel_size=self.lkb_origin.conv.kernel_size, stride=self.lkb_origin.conv.stride,
                                     padding=self.lkb_origin.conv.padding, dilation=self.lkb_origin.conv.dilation,
                                     groups=self.lkb_origin.conv.groups, bias=True)
        self.lkb_reparam.weight.data = eq_k
        self.lkb_reparam.bias.data = eq_b
        self.__delattr__('lkb_origin')
        if hasattr(self, 'small_conv'):
            self.__delattr__('small_conv')


# 在 ModernTCN.py 文件中，用下面的代码替换整个 Block 类

class Block(nn.Module):
    def __init__(self, large_size, small_size, dmodel, dff, nvars, small_kernel_merged=False, drop=0.1):
        super(Block, self).__init__()

        # 使用正确的 groups 参数，这对于模型架构的逻辑一致性至关重要
        self.dw = ReparamLargeKernelConv(in_channels=int(nvars * dmodel), out_channels=int(nvars * dmodel),
                                         kernel_size=int(large_size), stride=1, groups=int(nvars),
                                         small_kernel=small_size, small_kernel_merged=small_kernel_merged, nvars=nvars)

        self.norm = nn.BatchNorm1d(int(dmodel))

        self.ffn_pw1 = nn.Conv1d(in_channels=int(nvars * dmodel), out_channels=int(nvars * dff), kernel_size=1,
                                 stride=1,
                                 padding=0, dilation=1, groups=int(nvars))
        self.ffn_act = nn.GELU()
        self.ffn_pw2 = nn.Conv1d(in_channels=int(nvars * dff), out_channels=int(nvars * dmodel), kernel_size=1,
                                 stride=1,
                                 padding=0, dilation=1, groups=int(nvars))
        self.drop1 = nn.Dropout(drop)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        input_tensor = x
        B, M, D, N_in = input_tensor.shape

        # 1. 深度可分离卷积
        x = x.reshape(B, M * D, N_in)
        x = self.dw(x)

        # 2. 从卷积输出动态获取新的序列长度 N_out
        _, _, N_out = x.shape
        x = x.reshape(B, M, D, N_out)

        # 3. 归一化 和 FFN
        x = x.reshape(B * M, D, N_out)
        x = self.norm(x)
        x = x.reshape(B, M, D, N_out)

        x = x.reshape(B, M * D, N_out)
        x = self.drop1(self.ffn_pw1(x))
        x = self.ffn_act(x)
        x = self.drop2(self.ffn_pw2(x))
        x = x.reshape(B, M, D, N_out)

        # 4. 健壮的残差连接
        if N_in != N_out:
            if N_out < N_in:
                input_tensor = input_tensor[..., :N_out]
            else:
                x = x[..., :N_in]

        x = input_tensor + x
        return x


class Stage(nn.Module):
    def __init__(self, ffn_ratio, num_blocks, large_size, small_size, dmodel, dw_model, nvars,
                 small_kernel_merged=False, drop=0.1):

        super(Stage, self).__init__()

        # +++ 最终的监控探针 PART 1 +++
        # print("\n" + "---" * 10)
        # print("[STAGE DEBUG] Entering Stage.__init__")
        # print(f"[STAGE DEBUG] Received num_blocks = {num_blocks} (Type: {type(num_blocks)})")
        # print(f"[STAGE DEBUG] Received large_size = {large_size} (Type: {type(large_size)})")
        # print(f"[STAGE DEBUG] Received small_size = {small_size} (Type: {type(small_size)})")
        # print(f"[STAGE DEBUG] Received dmodel = {dmodel} (Type: {type(dmodel)})")
        # print(f"[STAGE DEBUG] Received nvars = {nvars} (Type: {type(nvars)})")
        # print("---" * 10 + "\n")
        # +++ 监控结束 +++

        d_ffn = dmodel * ffn_ratio
        blks = []
        for i in range(num_blocks):
            blk = Block(large_size=large_size, small_size=small_size, dmodel=dmodel, dff=d_ffn, nvars=nvars,
                        small_kernel_merged=small_kernel_merged, drop=drop)
            blks.append(blk)

        self.blocks = nn.ModuleList(blks)

    def forward(self, x):

        for blk in self.blocks:
            x = blk(x)

        return x


class ModernTCN(nn.Module):
    def __init__(self,patch_size,patch_stride, stem_ratio, downsample_ratio, ffn_ratio, num_blocks, large_size, small_size, dims, dw_dims,
                 nvars, small_kernel_merged=False, backbone_dropout=0.1, head_dropout=0.1, use_multi_scale=True, revin=True, affine=True,
                 subtract_last=False, freq=None, seq_len=512, c_in=7, individual=False, target_window=96):

        super(ModernTCN, self).__init__()



        # RevIN
        self.revin = revin
        if self.revin:
            self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)

        # stem layer & down sampling layers(if needed)
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(

            nn.Conv1d(1, dims[0], kernel_size=patch_size, stride=patch_stride),
            nn.BatchNorm1d(dims[0])
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.BatchNorm1d(dims[i]),
                nn.Conv1d(dims[i], dims[i + 1], kernel_size=downsample_ratio, stride=downsample_ratio),
            )
            self.downsample_layers.append(downsample_layer)
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.downsample_ratio = downsample_ratio

        if freq == 'h':
            time_feature_num = 4
        elif freq == 't':
            time_feature_num = 5
        else:
            raise NotImplementedError("time_feature_num should be 4 or 5")

        self.te_patch = nn.Sequential(

            nn.Conv1d(time_feature_num, time_feature_num, kernel_size=patch_size, stride=patch_stride,groups=time_feature_num),
            nn.Conv1d(time_feature_num, dims[0], kernel_size=1, stride=1, groups=1),
            nn.BatchNorm1d(dims[0]))

        # backbone

        self.num_stage = len(num_blocks)
        self.stages = nn.ModuleList()
        for stage_idx in range(self.num_stage):
            layer = Stage(ffn_ratio, num_blocks[stage_idx], large_size[stage_idx], small_size[stage_idx], dmodel=dims[stage_idx],
                          dw_model=dw_dims[stage_idx], nvars=nvars, small_kernel_merged=small_kernel_merged, drop=backbone_dropout)
            self.stages.append(layer)

        # Multi scale fusing (if needed)
        self.use_multi_scale = use_multi_scale
        self.up_sample_ratio = downsample_ratio

        self.lat_layer = nn.ModuleList()
        self.smooth_layer = nn.ModuleList()
        self.up_sample_conv = nn.ModuleList()
        for i in range(self.num_stage):
            align_dim = dims[-1]
            lat = nn.Conv1d(dims[i], align_dim, kernel_size=1,
                            stride=1)
            self.lat_layer.append(lat)
            smooth = nn.Conv1d(align_dim, align_dim, kernel_size=3, stride=1, padding=1)
            self.smooth_layer.append(smooth)

            up_conv = nn.Sequential(
                nn.ConvTranspose1d(align_dim, align_dim, kernel_size=self.up_sample_ratio, stride=self.up_sample_ratio),
                nn.BatchNorm1d(align_dim))
            self.up_sample_conv.append(up_conv)

        # head
        patch_num = seq_len // patch_stride

        self.n_vars = c_in
        self.individual = individual
        d_model = dims[-1]
        if patch_num % pow(downsample_ratio, (self.num_stage - 1)) == 0:
            self.head_nf = d_model * patch_num // pow(downsample_ratio, (self.num_stage - 1))
        else:
            # 向上取整，确保维度正确
            self.head_nf = d_model * (patch_num // pow(downsample_ratio, (self.num_stage - 1)) + 1)

        self.head = Flatten_Head(self.individual, self.n_vars, self.head_nf, target_window,
                                 head_dropout=head_dropout)

    def up_sample(self, x, upsample_ratio):
        _, _, _, N = x.shape
        return F.upsample(x, size=N, scale_factor=upsample_ratio, mode='bilinear')

    def forward_feature(self, x, te=None):

        # print(f"\n[FF_DEBUG] --- 进入 forward_feature ---")
        # print(f"[FF_DEBUG] 初始输入 x 的维度: {x.shape}")

        # 原始代码中的 B, M, L 命名容易混淆
        # 我们在这里使用 B, C, N 来代表 (批次, 通道数, 序列长度)
        try:
            B, C, N_initial = x.shape
            # print(f"[FF_DEBUG] 成功解析维度: B={B}, C={C}, N_initial={N_initial}")
        except ValueError as e:
            # print(f"[FF_DEBUG] !!! 致命错误: 无法将维度 {x.shape} 解析为3个值. 错误: {e}")
            # 这里增加保护，防止程序直接在此处崩溃
            return x

        x = x.unsqueeze(-2)
        # print(f"[FF_DEBUG] 调用 unsqueeze(-2) 后维度: {x.shape}")

        for i in range(self.num_stage):
            # print(f"\n[FF_DEBUG] --- Stage {i} ---")

            try:
                B_stage, M_stage, D_stage, N_stage = x.shape
                # print(
                #     f"[FF_DEBUG] Stage {i} | 起始维度: {x.shape} (B={B_stage}, M={M_stage}, D={D_stage}, N={N_stage})")
            except ValueError as e:
                # print(f"[FF_DEBUG] !!! 致命错误: 在 Stage {i} 的开头，无法将维度 {x.shape} 解析为4个值. 错误: {e}")
                return x

            x = x.reshape(B_stage * M_stage, D_stage, N_stage)
            # print(f"[FF_DEBUG] Stage {i} | 为降采样进行 reshape 之后维度: {x.shape}")

            # 检查 padding 逻辑
            current_len = x.shape[2]
            if i == 0:
                if self.patch_size != self.patch_stride:
                    pad_len = self.patch_size - self.patch_stride
                    pad = x[:, :, -1:].repeat(1, 1, pad_len)
                    x = torch.cat([x, pad], dim=-1)
                    # print(f"[FF_DEBUG] Stage {i} | 应用了 stem padding. 当前维度: {x.shape}")
            else:
                if current_len % self.downsample_ratio != 0:
                    pad_len = self.downsample_ratio - (current_len % self.downsample_ratio)
                    x = torch.cat([x, x[:, :, -pad_len:]], dim=-1)
                    # print(f"[FF_DEBUG] Stage {i} | 应用了 downsample padding. 当前维度: {x.shape}")

            # 降采样层
            # print(f"[FF_DEBUG] Stage {i} | 即将调用 downsample_layers[{i}]...")
            x = self.downsample_layers[i](x)
            # print(f"[FF_DEBUG] Stage {i} | <<<<<< 调用 downsample_layers[{i}] 之后维度: {x.shape} >>>>>>")

            # 检查降采样后的维度是否还能被正确解析
            try:
                _, D_, N_ = x.shape
            except ValueError as e:
                # print(f"[FF_DEBUG] !!! 致命错误: 在 Stage {i} 的降采样之后，无法将维度 {x.shape} 解析为3个值. 错误: {e}")
                # print(f"[FF_DEBUG] 这几乎可以肯定是问题的根源。降采样层可能输出了一个非预期的维度。")
                return x

            x = x.reshape(B_stage, M_stage, D_, N_)
            # print(f"[FF_DEBUG] Stage {i} | 重塑回 4D 之后维度: {x.shape}")

            # print(f"[FF_DEBUG] Stage {i} | 即将调用 self.stages[{i}] (Block)...")
            x = self.stages[i](x)
            # print(f"[FF_DEBUG] Stage {i} | 调用 self.stages[{i}] 之后维度: {x.shape}")

        # print(f"[FF_DEBUG] --- 退出 forward_feature ---")
        return x

    def forward(self, x, te=None):
        # 1. RevIN
        if self.revin:
            x = self.revin_layer(x, 'norm')

        # 2. Permute for Backbone
        x = x.permute(0, 2, 1)

        # 3. Backbone
        x = self.forward_feature(x, te)

        # --- 最终的、决定性的调试防线 ---
        try:
            # print("\n" + " H " * 20)
            # print("[HEAD_PRE_CHECK] --- 准备进入 Head 模块 ---")
            # print(f"[HEAD_PRE_CHECK] 张量 x 的维度: {x.shape}")
            # print(f"[HEAD_PRE_CHECK] 张量 x 的数据类型: {x.dtype}")
            # print(f"[HEAD_PRE_CHECK] 张量 x 是否在内存中连续: {x.is_contiguous()}")

            # 强制使其内存连续，这是解决此类底层错误的关键步骤
            x_contiguous = x.contiguous()

            # print(f"[HEAD_PRE_CHECK] 调用 .contiguous() 后，张量是否在内存中连续: {x_contiguous.is_contiguous()}")
            # print("[HEAD_PRE_CHECK] --- 即将调用 self.head() ---")
            # print(" H " * 20 + "\n")

            # 使用内存连续的张量调用 head
            x = self.head(x_contiguous)

        except Exception as e:
            # print("\n" + "X" * 30)
            # print("!!!!!! 程序在调用 Head 模块时发生致命错误 !!!!!!")
            # print(f"捕获到的异常: {e}")
            # print(f"发生错误时，输入张量 x 的维度: {x.shape}")
            # print(f"发生错误时，输入张量 x 的数据类型: {x.dtype}")
            # print(f"发生错误时，输入张量 x 是否在内存中连续: {x.is_contiguous()}")
            # print("X" * 30 + "\n")
            # 重新引发异常，以便 benchmark 脚本能够捕获它
            raise e

        # 5. Permute for output
        x = x.permute(0, 2, 1)

        # 6. RevIN denorm
        if self.revin:
            x = self.revin_layer(x, 'denorm')

        return x


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        # hyper param

        # +++ 最终调试代码 +++
        # print("[DEBUG] Inside Model.__init__")
        # +++ 最终调试代码结束 +++

        self.stem_ratio = configs.stem_ratio
        self.downsample_ratio = configs.downsample_ratio
        self.ffn_ratio = configs.ffn_ratio
        self.num_blocks = configs.num_blocks
        self.large_size = configs.large_size
        self.small_size = configs.small_size
        self.dims = configs.dims
        self.dw_dims = configs.dw_dims

        self.nvars = configs.enc_in
        self.small_kernel_merged = configs.small_kernel_merged
        self.drop_backbone = configs.dropout
        self.drop_head = configs.head_dropout
        self.use_multi_scale = configs.use_multi_scale
        self.revin = configs.revin
        self.affine = configs.affine
        self.subtract_last = configs.subtract_last

        self.freq = configs.freq
        self.seq_len = configs.seq_len

        # *** 最终的、最关键的修复 ***
        self.c_in = int(self.nvars)
        # +++ 最终调试代码 +++
        # print(f"[DEBUG] self.c_in = {self.c_in} (Type: {type(self.c_in)})")
        # +++ 最终调试代码结束 +++

        self.individual = configs.individual
        self.target_window = configs.pred_len

        self.kernel_size = configs.kernel_size
        self.patch_size = configs.patch_size
        self.patch_stride = configs.patch_stride

        # decomp
        self.decomposition = configs.decomposition
        if self.decomposition:
            self.decomp_module = series_decomp(self.kernel_size)
            self.model_res = ModernTCN(patch_size=self.patch_size, patch_stride=self.patch_stride,
                                       stem_ratio=self.stem_ratio, downsample_ratio=self.downsample_ratio,
                                       ffn_ratio=self.ffn_ratio, num_blocks=self.num_blocks, large_size=self.large_size,
                                       small_size=self.small_size, dims=self.dims, dw_dims=self.dw_dims,
                                       nvars=self.nvars, small_kernel_merged=self.small_kernel_merged,
                                       backbone_dropout=self.drop_backbone, head_dropout=self.drop_head,
                                       use_multi_scale=self.use_multi_scale, revin=self.revin, affine=self.affine,
                                       subtract_last=self.subtract_last, freq=self.freq, seq_len=self.seq_len,
                                       c_in=self.c_in, individual=self.individual, target_window=self.target_window)
            self.model_trend = ModernTCN(patch_size=self.patch_size, patch_stride=self.patch_stride,
                                         stem_ratio=self.stem_ratio, downsample_ratio=self.downsample_ratio,
                                         ffn_ratio=self.ffn_ratio, num_blocks=self.num_blocks,
                                         large_size=self.large_size, small_size=self.small_size, dims=self.dims,
                                         dw_dims=self.dw_dims,
                                         nvars=self.nvars, small_kernel_merged=self.small_kernel_merged,
                                         backbone_dropout=self.drop_backbone, head_dropout=self.drop_head,
                                         use_multi_scale=self.use_multi_scale, revin=self.revin, affine=self.affine,
                                         subtract_last=self.subtract_last, freq=self.freq, seq_len=self.seq_len,
                                         c_in=self.c_in, individual=self.individual, target_window=self.target_window)
        else:
            self.model = ModernTCN(patch_size=self.patch_size, patch_stride=self.patch_stride,
                                   stem_ratio=self.stem_ratio, downsample_ratio=self.downsample_ratio,
                                   ffn_ratio=self.ffn_ratio, num_blocks=self.num_blocks, large_size=self.large_size,
                                   small_size=self.small_size, dims=self.dims, dw_dims=self.dw_dims,
                                   nvars=self.nvars, small_kernel_merged=self.small_kernel_merged,
                                   backbone_dropout=self.drop_backbone, head_dropout=self.drop_head,
                                   use_multi_scale=self.use_multi_scale, revin=self.revin, affine=self.affine,
                                   subtract_last=self.subtract_last, freq=self.freq, seq_len=self.seq_len,
                                   c_in=self.c_in, individual=self.individual, target_window=self.target_window)

    def forward(self, x, te=None):
        # x 的原始输入维度是 (B, L, C), 例如 (8, 96, 32)

        if self.decomposition:
            res_init, trend_init = self.decomp_module(x)
            res_init, trend_init = res_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
            if te is not None:
                te = te.permute(0, 2, 1)
            res = self.model_res(res_init, te)
            trend = self.model_trend(trend_init, te)
            x = res + trend
            # 分解分支的输出已经是 (B, C, L_pred)，最后需要转置回 (B, L_pred, C)
            x = x.permute(0, 2, 1)
        else:
            # --- 关键修正 ---
            # 1. 在调用核心模型前，将维度从 (B, L, C) 转换为 (B, C, L)
            x = x.permute(0, 2, 1)

            if te is not None:
                te = te.permute(0, 2, 1)

            # 2. 调用 ModernTCN 模型，它现在接收并处理 (B, C, L) 格式
            x = self.model(x, te)

            # 3. ModernTCN 模型输出为 (B, C, L_pred)，将其转置回标准的 (B, L_pred, C) 格式
            x = x.permute(0, 2, 1)
            # --- 修正结束 ---
        return x
