"""
DeLaN 底层积木（来自 deep_lagrangian_networks 官方实现）。

本文件提供 L-Net 依赖的「可微分层」：
  - LowTri          : 向量 -> 下三角矩阵
  - LagrangianLayer : 普通全连接 + 同时算 ∂输出/∂q（不用 autograd 反复求导，更快）
  - build_l_pack_index : 重排 Cholesky 参数顺序的小工具
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LowTri:
    """
    把长度为 m = n(n+1)/2 的向量，填进 n×n 下三角矩阵 L。

    例：n_dof=2 时 m=3，向量 [a, b, c] 对应矩阵
        [[a, 0],
         [b, c]]
    """

    def __init__(self, n_dof: int) -> None:
        self.n_dof = n_dof
        # np.tril_indices 返回下三角位置的 (行索引, 列索引)
        self._idx = np.tril_indices(n_dof)

    def __call__(self, l: torch.Tensor) -> torch.Tensor:
        """让 LowTri 实例可以像函数一样调用：L = low_tri(l)。"""
        B = l.shape[0]  # batch 大小
        L = torch.zeros(B, self.n_dof, self.n_dof, device=l.device, dtype=l.dtype)
        # 用 NumPy 算好的下标，把 l 的每一列填进 L 的下三角
        L[:, self._idx[0], self._idx[1]] = l
        return L


class ReLUDer(nn.Module):
    """ReLU 的导数：x>0 为 1，x<=0 为 0（用于链式法则）。"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ceil(torch.clamp(x, 0.0, 1.0))


class LinearAct(nn.Module):
    """线性激活 f(x)=x（输出层常用，不压范围）。"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class LinearDer(nn.Module):
    """线性函数导数恒为 1。"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, 1.0, 1.0)


class SoftplusDer(nn.Module):
    """Softplus 导数 sigmoid(beta * x)，与官方 DeLaN 一致。"""

    def __init__(self, beta: float = 1.0) -> None:
        super().__init__()
        self._beta = beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cx = torch.clamp(x, -20.0, 20.0)
        exp_x = torch.exp(self._beta * cx)
        return exp_x / (exp_x + 1.0)


class LagrangianLayer(nn.Module):
    """
    DeLaN 专用层：一次前向同时得到
      - out : 层输出 y = activation(W @ x + b)
      - der : 输出对「最原始 q」的雅可比 ∂y/∂q，形状 (B, n_out, n_in_q)

    普通 nn.Linear 只返回 out；这里多传 der_prev（上一层雅可比），用链式法则更新 der。
    """

    def __init__(self, input_size: int, n_out: int, activation: str = "ReLu") -> None:
        super().__init__()
        self.n_out = n_out
        # nn.Parameter：告诉 PyTorch「这是要训练的权重」
        self.weight = nn.Parameter(torch.empty(n_out, input_size))
        self.bias = nn.Parameter(torch.empty(n_out))

        if activation in ("ReLu", "ReLU"):
            self.g = nn.ReLU()
            self.g_prime = ReLUDer()
        elif activation == "SoftPlus":
            self.g = nn.Softplus(beta=1.0)
            self.g_prime = SoftplusDer(beta=1.0)
        elif activation == "Linear":
            self.g = LinearAct()
            self.g_prime = LinearDer()
        else:
            raise ValueError(f"unsupported activation: {activation!r}")
    # 单层拉格朗日层的前向传播，计算输出和雅可比
    def forward(self, q: torch.Tensor, der_prev: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # 仿射变换 a = W q + b（F.linear 里 weight 形状是 out_features × in_features）
        a = F.linear(q, self.weight, self.bias)
        # 非线性激活
        out = self.g(a)

        # 链式法则：∂out/∂q = diag(g'(a)) @ W @ ∂(上一层)/∂q
        g_prime_a = self.g_prime(a).view(-1, self.n_out, 1) * self.weight
        der = torch.matmul(g_prime_a, der_prev)
        return out, der

# 初始化隐藏层权重时的缩放系数，用来配合激活函数，让训练一开始更稳，一般根号2
def init_hidden(layer: LagrangianLayer, b_init: float = 0.1, gain: float | None = None) -> None:
    """隐藏层权重初始化（Xavier + 常数偏置），训练更稳定。"""
    g = gain if gain is not None else torch.nn.init.calculate_gain("relu")
    nn.init.constant_(layer.bias, b_init)
    nn.init.xavier_normal_(layer.weight, g)

# 初始化输出层
def init_output(layer: LagrangianLayer, b_init: float = 0.1, gain: float = 0.125) -> None:
    """输出头（势能 V、l_lower）用稍小的增益初始化。"""
    nn.init.constant_(layer.bias, b_init)
    nn.init.xavier_normal_(layer.weight, gain)

# 重排 Cholesky 参数顺序的小工具
def build_l_pack_index(n_dof: int) -> np.ndarray:
    """
    net_ld 先输出对角、net_lo 再输出非对角；拼在一起后要重排成下三角顺序。

    返回的是「列索引」数组，用法：l_packed = torch.cat([l_diag, l_lower], dim=1)[:, pack_idx]
    """
    m = n_dof * (n_dof + 1) // 2
    # 下三角中「对角线元素」在长度 m 向量里的位置（官方 DeLaN 公式）
    idx_diag = np.arange(n_dof, dtype=np.int64) + 1
    idx_diag = idx_diag * (idx_diag + 1) // 2 - 1
    # 其余位置是非对角
    idx_tril = np.extract([x not in idx_diag for x in np.arange(m)], np.arange(m))
    cat_idx = np.hstack((idx_diag, idx_tril))
    order = np.argsort(cat_idx)
    return np.arange(cat_idx.size, dtype=np.int64)[order]
