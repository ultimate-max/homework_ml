"""
L-Net（刚体逆动力学）—— 对齐 deep_lagrangian_networks / DeLaN 论文实现。

---------------------------------------------------------------------------
读代码前：符号与张量形状
---------------------------------------------------------------------------
  q, qd, qdd  : 关节位置、速度、加速度，形状 (B, n_dof)
  B           : batch size，一次前向里有多少条样本（多少时刻/轨迹点）
  n_dof       : 自由度（关节数），2 就是 2 关节腿
  H           : 惯性矩阵 (B, n_dof, n_dof)
  tau         : 关节力矩 (B, n_dof)

---------------------------------------------------------------------------
物理公式（刚体部分，无摩擦）
---------------------------------------------------------------------------

    tau = H(q) @ qdd + c(q, qd) + g(q)

    H = L @ L^T + epsilon * I     # L 下三角，保证 H 对称正定
    g = dV/dq                       # 势能 V(q) 对 q 的梯度（重力项）
    c = dH/dt @ qd - 0.5 * d/dq (qd^T H qd)   # 科里奥利 + 离心

网络结构（与论文 Fig.1 相同）::

    q --[共享 layers]--> y --+-- net_ld --> l_diag (对角)
                             +-- net_lo --> l_lower (严格下三角)
                             +-- net_g  --> V(q)  --> g = dV/dq
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from .core import LagrangianLayer, LowTri, build_l_pack_index, init_hidden, init_output


# q ──► self.layers（多层 ReLU LagrangianLayer）──► y, der
#                            │
#            ┌───────────────┼───────────────┐
#            ▼               ▼               ▼
#       net_ld(y,der)   net_lo(y,der)   net_g(y,der)
#            │               │               │
#       l_diag (对角)    l_lower (严下)      V, ∂V/∂q
#            │               │
#            └──────► cat + pack ──► l ──► L ──► H

@dataclass
class RigidBodyDynamics:
    """
    把一次动力学计算得到的所有物理量打包成一个「数据类」。

    @dataclass 是 Python 语法糖：自动生成 __init__，下面字段可直接 dyn.tau、dyn.H 访问。
    """

    tau: torch.Tensor   # (B, n_dof) 总刚体力矩 = H@qdd + c + g
    H: torch.Tensor     # (B, n_dof, n_dof) 惯性矩阵
    c: torch.Tensor     # (B, n_dof) 科里奥利 + 离心
    g: torch.Tensor     # (B, n_dof) 重力（势能梯度）
    T: torch.Tensor     # (B,) 动能 T = 0.5 * qd^T H qd
    V: torch.Tensor     # (B,) 势能（网络输出）
    dTdt: torch.Tensor  # (B,) 动能对时间的变化率
    dVdt: torch.Tensor  # (B,) 势能变化率 = g^T qd
    L: torch.Tensor     # (B, n_dof, n_dof) Cholesky 下三角因子


class LNet(nn.Module):
    """
  Deep Lagrangian Network。

  继承 nn.Module 表示「可训练的网络」；PyTorch 会自动登记子模块和参数。

  常用接口：
    forward(q, qd, qdd)  -> (tau, H, g)     # 给 Mysteric-Net 用
    dynamics(q, qd, qdd)   -> RigidBodyDynamics  # 要调试物理量时用
    inv_dyn(q, qd, qdd)    -> tau             # 只要力矩
    """

    def __init__(
        self,
        dof: int,
        hidden_dim: int = 32,
        num_hidden_layers: int = 2,
        b_diagonal: float = 1.0e-2,
        *,
        numerical_H_ridge: float = 1.0e-2,
        b_init: float = 1.0e-3,
        activation: str = "SoftPlus",
    ) -> None:
        # super().__init__() 必须调用，才能正确注册下面的层和参数
        super().__init__()

        self.n_dof = dof
        self.b_diagonal = b_diagonal          # L 对角偏置初值（论文里的 b）
        self.diagonal_epsilon = numerical_H_ridge  # 加到 H 上的小量 εI，数值更稳

        # 下三角矩阵独立元素个数：n(n+1)/2，例如 2 关节 -> 3 个数
        self.m = dof * (dof + 1) // 2
        # 把网络输出的 [对角|非对角] 重排成下三角顺序用的列索引（NumPy 数组）
        self._l_pack_idx = build_l_pack_index(dof)
        # 工具：向量 l -> 矩阵 L
        self.low_tri = LowTri(dof)

        # ---------- 共享骨干：q -> y ----------
        # ModuleList 类似 list，但 PyTorch 能识别其中的层并训练其参数
        self.layers = nn.ModuleList()
        # 第 1 层：输入维度 = 关节数 dof，输出 = hidden_dim
        act = activation
        self.layers.append(LagrangianLayer(dof, hidden_dim, activation=act))
        init_hidden(self.layers[0], b_init=b_init)
        # 其余隐藏层：hidden_dim -> hidden_dim
        for _ in range(1, num_hidden_layers):
            layer = LagrangianLayer(hidden_dim, hidden_dim, activation=act)
            init_hidden(layer, b_init=b_init)
            self.layers.append(layer)

        # ---------- 三个输出头（都从同一个 y 分叉）----------
        l_lower_size = self.m - dof  # 严格下三角元素个数，2 关节时为 1

        # net_g（对齐 DeLaN Fig.1 的 Linear 头）：在共享特征 y=h(q) 上做仿射，输出标量势能 V(q)。
        # 论文式 (3)→(4) 代入 dV/dq = g(q)；式 (6) 逆模型中与之对应的学习项记为 g_hat。
        # 此处用 V 参数化：LagrangianLayer 前向得到 V 与 der_V=∂V/∂q（链式法则），_dyn_model 里 g = der_V.squeeze，
        # 即 g_hat 与力学符号 g=dV/dq 同一角色；Linear 只表示「V 对 y 仿射」，对 q 的非线性来自共享骨干。
        self.net_g = LagrangianLayer(hidden_dim, 1, activation="Linear")
        init_output(self.net_g, b_init=b_init)

        # net_lo：L 的严格下三角（线性输出，可正可负）
        # 若 dof==1 则没有下三角非对角，net_lo 设为 None
        self.net_lo: LagrangianLayer | None
        if l_lower_size > 0:
            self.net_lo = LagrangianLayer(hidden_dim, l_lower_size, activation="Linear")
            init_hidden(self.net_lo, b_init=b_init)
        else:
            self.net_lo = None

        # net_ld：L 的对角；ReLU 保证对角非负，再加 b_diagonal
        self.net_ld = LagrangianLayer(hidden_dim, dof, activation="ReLu")
        init_hidden(self.net_ld, b_init=b_init)
        nn.init.constant_(self.net_ld.bias, b_diagonal)

    @property
    def dof(self) -> int:
        """属性别名：外面写 lnet.dof 等价于 lnet.n_dof。"""
        return self.n_dof

    @property
    def mass_diag_eps(self) -> float:
        """旧代码里的名字，等价于 b_diagonal。"""
        return self.b_diagonal

    def H_hat_from_q(self, q: torch.Tensor) -> torch.Tensor:
        """只算惯性矩阵 H(q)；速度、加速度置零。"""
        return self.dynamics(q, torch.zeros_like(q), torch.zeros_like(q)).H

    def mass_matrix(self, q: torch.Tensor) -> torch.Tensor:
        """兼容旧接口名。"""
        return self.H_hat_from_q(q)

    def dynamics(self, q: torch.Tensor, qd: torch.Tensor, qdd: torch.Tensor) -> RigidBodyDynamics:
        """对外公开：返回包含全部物理量的 RigidBodyDynamics。"""
        return self._dyn_model(q, qd, qdd)

    def _dyn_model(self, q: torch.Tensor, qd: torch.Tensor, qdd: torch.Tensor) -> RigidBodyDynamics:
        """
        核心动力学计算（与官方 DeLaN 的 _dyn_model 一致）。

        输入 q, qd, qdd 形状均为 (B, n_dof)。
        """
        n_dof = self.n_dof
        B = q.shape[0]

        # view 改变张量形状但不改数据；便于做矩阵乘法
        # qd_3d: (B, n_dof, 1)  当作列向量
        qd_3d = qd.view(B, n_dof, 1)
        # qd_4d: (B, 1, n_dof, 1)  用于四阶张量运算（dH/dq 相关）
        qd_4d = qd.view(B, 1, n_dof, 1)

        # ---- Step 1: 共享骨干，同时传播 ∂y/∂q（雅可比，供后面求 dH/dq）----
        # der 初始为 ∂q/∂q = I，形状 (B, n_dof, n_dof)
        der = torch.eye(n_dof, device=q.device, dtype=q.dtype).unsqueeze(0).expand(B, -1, -1).clone()
        y, der = self.layers[0](q, der)
        for layer in self.layers[1:]:
            y, der = layer(y, der)
        # 此时 y: (B, hidden_dim)，der: (B, hidden_dim, n_dof)

        # ---- Step 2: 三个头，得到 l_diag, l_lower, V 及它们对 q 的导数 ----
        if self.net_lo is not None:
            l_lower, der_l_lower = self.net_lo(y, der)
        else:
            # 空张量占位（dof=1 时）
            l_lower = torch.zeros(B, 0, device=q.device, dtype=q.dtype)
            der_l_lower = torch.zeros(B, 0, n_dof, device=q.device, dtype=q.dtype)

        l_diag, der_l_diag = self.net_ld(y, der)
        V, der_V = self.net_g(y, der)
        V = V.squeeze(-1)           # (B, 1) -> (B,)
        g = der_V.squeeze(1)        # (B, 1, n_dof) -> (B, n_dof)，即 g = ∂V/∂q

        # 把 [对角元素 | 非对角元素] 按官方顺序重排，得到长度 m 的向量 l
        pack = torch.as_tensor(self._l_pack_idx, device=q.device, dtype=torch.long)
        l = torch.cat((l_diag, l_lower), dim=1)[:, pack]
        der_l = torch.cat((der_l_diag, der_l_lower), dim=1)[:, pack, :]

        # ---- Step 3: 组装 L，再算 H = L L^T + εI ----
        L = self.low_tri(l)                 # (B, n_dof, n_dof)
        LT = L.transpose(1, 2)              # L 的转置
        eps_I = self.diagonal_epsilon * torch.eye(n_dof, device=q.device, dtype=q.dtype)
        H = torch.matmul(L, LT) + eps_I

        # ---- Step 4: dH/dt 与 dH/dq（用于科里奥利项 c）----
        # dL/dt = (∂L/∂q) @ qd
        Ldt = self.low_tri(torch.matmul(der_l, qd_3d).view(B, self.m))
        Hdt = torch.matmul(L, Ldt.transpose(1, 2)) + torch.matmul(Ldt, LT)

        # dH/dq：四维张量 (B, n_dof, n_dof, n_dof)
        Ldq = self.low_tri(der_l.transpose(1, 2).reshape(-1, self.m)).reshape(B, n_dof, n_dof, n_dof)
        Hdq = torch.matmul(Ldq, LT.view(B, 1, n_dof, n_dof)) + torch.matmul(
            L.view(B, 1, n_dof, n_dof), Ldq.transpose(2, 3)
        )

        # c = (dH/dt) @ qd - 0.5 * qd^T (dH/dq) @ qd
        Hdt_qd = torch.matmul(Hdt, qd_3d).view(B, n_dof)
        quad_dq = torch.matmul(qd_4d.transpose(2, 3), torch.matmul(Hdq, qd_4d)).view(B, n_dof)
        c = Hdt_qd - 0.5 * quad_dq

        # ---- Step 5: 逆动力学力矩 tau = H@qdd + c + g ----
        H_qdd = torch.matmul(H, qdd.view(B, n_dof, 1)).view(B, n_dof)
        tau = H_qdd + c + g

        # ---- Step 6: 能量（用于损失函数或调试）----
        H_qd = torch.matmul(H, qd_3d).view(B, n_dof)
        T = 0.5 * torch.matmul(qd_4d.transpose(2, 3), H_qd.view(B, 1, n_dof, 1)).view(B)

        qd_H_qdd = torch.matmul(qd_4d.transpose(2, 3), H_qdd.view(B, 1, n_dof, 1)).view(B)
        qd_Hdt_qd = torch.matmul(qd_4d.transpose(2, 3), Hdt_qd.view(B, 1, n_dof, 1)).view(B)
        dTdt = qd_H_qdd + 0.5 * qd_Hdt_qd
        dVdt = torch.matmul(qd_4d.transpose(2, 3), g.view(B, 1, n_dof, 1)).view(B)

        return RigidBodyDynamics(tau=tau, H=H, c=c, g=g, T=T, V=V, dTdt=dTdt, dVdt=dVdt, L=L)

    def inv_dyn(self, q: torch.Tensor, qd: torch.Tensor, qdd: torch.Tensor) -> torch.Tensor:
        """逆动力学：已知运动 (q,qd,qdd)，求力矩 tau。"""
        return self._dyn_model(q, qd, qdd).tau

    def forward_delan(
        self, q: torch.Tensor, qd: torch.Tensor, qdd: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """与官方 ``DeepLagrangianNetwork.forward`` 一致：返回 (tau, dE/dt)。"""
        dyn = self._dyn_model(q, qd, qdd)
        return dyn.tau, dyn.dTdt + dyn.dVdt

    def forward(self, q: torch.Tensor, qd: torch.Tensor, qdd: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        PyTorch 规定：训练/推理时调用 model(q, qd, qdd) 会执行 forward。

        返回 (tau_rigid, H, g)，供 Mysteric-Net 再加上摩擦项。
        """
        out = self._dyn_model(q, qd, qdd)
        return out.tau, out.H, out.g
