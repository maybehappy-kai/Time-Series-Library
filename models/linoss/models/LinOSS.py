from typing import List

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import nn
from jax.nn.initializers import normal
import math
from jax import random

# --- PyTorch / NumPy ---
import torch
import numpy as np


def simple_uniform_init(rng, shape, std=1.):
    weights = random.uniform(rng, shape) * 2. * std - std
    return weights


class GLU(eqx.Module):
    w1: eqx.nn.Linear
    w2: eqx.nn.Linear

    def __init__(self, input_dim, output_dim, key):
        w1_key, w2_key = jr.split(key, 2)
        self.w1 = eqx.nn.Linear(input_dim, output_dim, use_bias=True, key=w1_key)
        self.w2 = eqx.nn.Linear(input_dim, output_dim, use_bias=True, key=w2_key)

    def __call__(self, x):
        return self.w1(x) * jax.nn.sigmoid(self.w2(x))


def binary_operator(q_i, q_j):
    A_i, b_i = q_i
    A_j, b_j = q_j

    S4 = A_i.shape[-1]
    S = S4 // 4

    Ai = A_i.reshape(A_i.shape[:-1] + (4, S))
    Aj = A_j.reshape(A_j.shape[:-1] + (4, S))
    iA_, iB_, iC_, iD_ = Ai[..., 0, :], Ai[..., 1, :], Ai[..., 2, :], Ai[..., 3, :]
    jA_, jB_, jC_, jD_ = Aj[..., 0, :], Aj[..., 1, :], Aj[..., 2, :], Aj[..., 3, :]

    A_new = jA_ * iA_ + jB_ * iC_
    B_new = jA_ * iB_ + jB_ * iD_
    C_new = jC_ * iA_ + jD_ * iC_
    D_new = jC_ * iB_ + jD_ * iD_

    Anew = jnp.concatenate([A_new, B_new, C_new, D_new], axis=-1)

    bi = b_i.reshape(b_i.shape[:-1] + (2, S))
    bj = b_j.reshape(b_j.shape[:-1] + (2, S))
    b_i1, b_i2 = bi[..., 0, :], bi[..., 1, :]
    b_j1, b_j2 = bj[..., 0, :], bj[..., 1, :]

    new_b1 = jA_ * b_i1 + jB_ * b_i2
    new_b2 = jC_ * b_i1 + jD_ * b_i2
    new_b = jnp.concatenate([new_b1, new_b2], axis=-1)

    return Anew, new_b + b_j


def apply_linoss_im(A_diag, B, C_tilde, input_sequence, step):
    Bu_elements = input_sequence @ B.T

    schur_comp = 1. / (1. + step ** 2. * A_diag)
    M_IM_11 = 1. - step ** 2. * A_diag * schur_comp
    M_IM_12 = -1. * step * A_diag * schur_comp
    M_IM_21 = step * schur_comp
    M_IM_22 = schur_comp

    M_IM = jnp.concatenate([M_IM_11, M_IM_12, M_IM_21, M_IM_22])
    M_IM_elements = jnp.broadcast_to(M_IM, (input_sequence.shape[0], 4 * A_diag.shape[0]))

    F1 = M_IM_11 * Bu_elements * step
    F2 = M_IM_21 * Bu_elements * step
    F = jnp.hstack((F1, F2))

    _, xs = jax.lax.associative_scan(binary_operator, (M_IM_elements, F))
    ys = xs[..., A_diag.shape[0]:]

    return (ys @ C_tilde.T).real


def apply_linoss_imex(A_diag, B, C, input_sequence, step):
    Bu_elements = input_sequence @ B.T

    A_ = jnp.ones_like(A_diag)
    B_ = -1. * step * A_diag
    C_ = step
    D_ = 1. - (step ** 2.) * A_diag

    M_IMEX = jnp.concatenate([A_, B_, C_, D_])
    M_IMEX_elements = jnp.broadcast_to(M_IMEX, (input_sequence.shape[0], 4 * A_diag.shape[0]))

    F1 = Bu_elements * step
    F2 = Bu_elements * (step ** 2.)
    F = jnp.hstack((F1, F2))

    _, xs = jax.lax.associative_scan(binary_operator, (M_IMEX_elements, F))
    ys = xs[..., A_diag.shape[0]:]
    return (ys @ C.T).real


class LinOSSLayer(eqx.Module):
    A_diag: jax.Array
    B: jax.Array
    C: jax.Array
    D: jax.Array
    steps: jax.Array
    discretization: str = eqx.field(static=True)

    def __init__(self, ssm_size, H, discretization, *, key):
        B_key, C_key, D_key, A_key, step_key, key = jr.split(key, 6)
        self.A_diag = random.uniform(A_key, shape=(ssm_size,))
        self.B = simple_uniform_init(B_key, shape=(ssm_size, H, 2), std=1. / math.sqrt(H))
        self.C = simple_uniform_init(C_key, shape=(H, ssm_size, 2), std=1. / math.sqrt(ssm_size))
        self.D = normal(stddev=1.0)(D_key, (H,))
        self.steps = random.uniform(step_key, shape=(ssm_size,))
        self.discretization = discretization

    def __call__(self, input_sequence):
        A_diag = nn.relu(self.A_diag)

        B_complex = self.B[..., 0] + 1j * self.B[..., 1]
        C_complex = self.C[..., 0] + 1j * self.C[..., 1]

        steps = nn.sigmoid(self.steps)
        if self.discretization == 'IMEX':
            ys = apply_linoss_imex(A_diag, B_complex, C_complex, input_sequence, steps)
        elif self.discretization == 'IM':
            ys = apply_linoss_im(A_diag, B_complex, C_complex, input_sequence, steps)
        else:
            raise ValueError(f"Discretization '{self.discretization}' not implemented.")

        Du = jax.vmap(lambda u: self.D * u)(input_sequence)
        return ys + Du


class LinOSSBlock(eqx.Module):
    norm: eqx.nn.LayerNorm
    ssm: LinOSSLayer
    glu: GLU
    drop: eqx.nn.Dropout

    def __init__(self, ssm_size, H, discretization, drop_rate=0.05, *, key):
        ssmkey, glukey = jr.split(key, 2)
        self.norm = eqx.nn.LayerNorm(H)
        self.ssm = LinOSSLayer(ssm_size, H, discretization, key=ssmkey)
        self.glu = GLU(H, H, key=glukey)
        self.drop = eqx.nn.Dropout(p=drop_rate)

    def __call__(self, x, *, key):
        y = jax.vmap(self.norm)(x)
        y = self.ssm(y)
        y = jax.vmap(self.glu)(y)
        y = self.drop(y, key=key)
        return x + y


class _LinOSSImpl(eqx.Module):
    linear_encoder: eqx.nn.Linear
    blocks: List[LinOSSBlock]
    linear_layer: eqx.nn.Linear
    classification: bool = eqx.field(static=True)
    output_step: int = eqx.field(static=True)

    def __init__(self, num_blocks, N, ssm_size, H, output_dim, classification, output_step, discretization, *, key):
        linear_encoder_key, *block_keys, linear_layer_key = jr.split(key, num_blocks + 2)
        self.linear_encoder = eqx.nn.Linear(N, H, key=linear_encoder_key)
        self.blocks = [
            LinOSSBlock(ssm_size, H, discretization, key=key)
            for key in block_keys
        ]
        self.linear_layer = eqx.nn.Linear(H, output_dim, key=linear_layer_key)
        self.classification = classification
        self.output_step = output_step

    def __call__(self, x, *, key):
        x = jax.vmap(self.linear_encoder)(x)

        for block in self.blocks:
            key, subkey = jr.split(key)
            x = block(x, key=subkey)

        if self.classification:
            x = jnp.mean(x, axis=0)
            x = jax.nn.softmax(self.linear_layer(x), axis=0)
        else:
            x = x[self.output_step - 1:: self.output_step]
            x = jax.vmap(self.linear_layer)(x)
            x = jax.nn.tanh(x)

        return x


# --- PyTorch Wrapper ---
class LinOSS(torch.nn.Module):
    def __init__(self, num_blocks, N, ssm_size, H, output_dim, classification, output_step, discretization, *, key):
        super().__init__()
        self.key = key
        self.jax_model = _LinOSSImpl(
            num_blocks, N, ssm_size, H, output_dim,
            classification, output_step, discretization, key=key
        )
        self.jax_model_inf = eqx.tree_inference(self.jax_model, value=True)

        def _call_with_key(x, key):
            return self.jax_model_inf(x, key=key)

        self.batched_model_inf = jax.jit(
            jax.vmap(_call_with_key, in_axes=(0, None))
        )

        def _is_array(x):
            return isinstance(x, (jnp.ndarray, jax.Array))

        num_params = sum(x.size for x in jax.tree_util.tree_leaves(self.jax_model) if _is_array(x))
        self.dummy_param = torch.nn.Parameter(torch.empty(num_params))
        print(f"[Info] LinOSS JAX model initialized with {num_params} parameters.")

    def to(self, device):
        return self

    def forward(self, x_enc):
        if isinstance(x_enc, dict):
            x_enc = x_enc["X"]

        x_enc_to_jax = x_enc.contiguous()
        call_key, self.key = jr.split(self.key)

        if x_enc.is_cuda:
            dl = torch.utils.dlpack.to_dlpack(x_enc_to_jax)
            x_jax = jax.dlpack.from_dlpack(dl)
            output_jax = self.batched_model_inf(x_jax, call_key)
            output_jax.block_until_ready()
            # --- 关键修正：直接传递JAX数组 ---
            out_torch = torch.utils.dlpack.from_dlpack(output_jax)
        else:
            cpu_dev = jax.devices("cpu")[0]
            with jax.default_device(cpu_dev):
                x_jax = jnp.asarray(x_enc_to_jax.cpu().numpy())
                output_jax = self.batched_model_inf(x_jax, call_key)
                output_jax.block_until_ready()
                out_torch = torch.from_numpy(np.asarray(output_jax))

        return out_torch.to(x_enc.device)