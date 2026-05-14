"""
L-Net: Lagrangian-style rigid-body inverse dynamics (DeLaN-style, [16] in paper).

tau_rig_core = M(q) qdd + C(q, qd) qd + g(q)

Coriolis/离心项采用与 Christoffel 等价的 Slotine 形式（低自由度下数值一致），用 jvp 计算
  C(q,qd) qd = d/dt (M(q) qd_det) - 0.5 * nabla_q (qd_det^T M(q) qd_det)
其中 qd_det = stop_grad(qd)，对 q 求偏导时把 qd 当作常量，符合 ∂T/∂q 的定义。
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.autograd.functional as AF


class LNet(nn.Module):
    def __init__(
        self,
        dof: int,
        hidden_dim: int = 32,
        num_hidden_layers: int = 2,
        mass_diag_eps: float = 1.0e-2,
    ) -> None:
        super().__init__()
        self.dof = dof
        self.mass_diag_eps = mass_diag_eps

        def mlp(in_dim: int, out_dim: int) -> nn.Sequential:
            layers: list[nn.Module] = []
            d = in_dim
            for _ in range(num_hidden_layers):
                layers += [nn.Linear(d, hidden_dim), nn.Tanh()]
                d = hidden_dim
            layers.append(nn.Linear(d, out_dim))
            return nn.Sequential(*layers)

        tri_size = dof * (dof + 1) // 2
        self._mass_head = mlp(dof, tri_size)
        self._pot_head = mlp(dof, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def mass_matrix(self, q: torch.Tensor) -> torch.Tensor:
        """Return SPD M(q), shape (B, dof, dof)."""
        B, dof = q.shape
        raw = self._mass_head(q)
        L = torch.zeros(B, dof, dof, device=q.device, dtype=q.dtype)
        idx = 0
        for i in range(dof):
            for j in range(i + 1):
                if i == j:
                    L[:, i, j] = torch.nn.functional.softplus(raw[:, idx]) + self.mass_diag_eps
                else:
                    L[:, i, j] = raw[:, idx]
                idx += 1
        M = torch.bmm(L, L.transpose(1, 2))
        eye = torch.eye(dof, device=q.device, dtype=q.dtype).expand(B, dof, dof)
        return M + self.mass_diag_eps * eye

    def potential_energy(self, q: torch.Tensor) -> torch.Tensor:
        return self._pot_head(q).squeeze(-1)

    def _mdot_qd_jvp(self, q_req: torch.Tensor, qd: torch.Tensor) -> torch.Tensor:
        """(d/dt)(M(q) qd_det) 中与 q 有关的部分: J_{q}(M(q) v) @ qd, v=qd_det, shape (B, dof)."""
        B, dof = q_req.shape
        qd_det = qd.detach()
        out = torch.zeros(B, dof, device=q_req.device, dtype=q_req.dtype)
        for b in range(B):
            qb = q_req[b]

            def f(x: torch.Tensor) -> torch.Tensor:
                return torch.mv(self.mass_matrix(x.unsqueeze(0)).squeeze(0), qd_det[b])

            _, tang = AF.jvp(f, (qb,), (qd[b],), create_graph=True)
            out[b] = tang
        return out

    def forward(self, q: torch.Tensor, qd: torch.Tensor, qdd: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            tau_core: (B, dof) = M qdd + C(q,qd)qd + g
            M: (B, dof, dof)
            g: (B, dof)
        """
        # L-Net 需对 q 做 autograd.grad / jvp；推理时常见外层 torch.no_grad()，此处临时建图并在无 grad 时 detach 输出。
        want_grad = torch.is_grad_enabled()
        q_req = q.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            M = self.mass_matrix(q_req)

            V = self.potential_energy(q_req).sum()
            g = torch.autograd.grad(V, q_req, create_graph=True, retain_graph=True)[0]

            qd_det = qd.detach()
            S = torch.einsum("bi,bij,bj->b", qd_det, M, qd_det).sum()
            gradS = torch.autograd.grad(S, q_req, create_graph=True, retain_graph=True)[0]

            mdot_qd = self._mdot_qd_jvp(q_req, qd)
            cvec = mdot_qd - 0.5 * gradS

            tau_core = torch.bmm(M, qdd.unsqueeze(-1)).squeeze(-1) + cvec + g

        if not want_grad:
            tau_core = tau_core.detach()
            M = M.detach()
            g = g.detach()
        return tau_core, M, g
