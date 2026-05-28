"""PINN tau_blend：SCV 摩擦残差监督 + fo 单向对齐 SCV。"""

import torch

from RobotDynamics.DeLaN.losses import torque_loss
from RobotDynamics.FrictionModule import friction_pinn_tau_blend_loss
from RobotDynamics.FrictionModule.stribeck import HNetStribeckPINN
from RobotDynamics.MystericNet import MystericNet


def test_tau_blend_l_tau_skips_scv() -> None:
    hnet = HNetStribeckPINN(dof=2, seq_len=30)
    qs = torch.randn(4, 30, 2)
    qds = torch.randn(4, 30, 2)
    tau_pred, tau_scv = hnet(qs, qds)
    tau_core = torch.randn(4, 2)
    taub = torch.randn(4, 2)
    l_tau = torque_loss(tau_core + tau_pred, taub, "mse")
    hnet.zero_grad(set_to_none=True)
    l_tau.backward()
    assert hnet.mlp[0].weight.grad is not None
    assert hnet.scv.log_k_c.grad is None


def test_tau_blend_scv_sup_trains_scv_not_fo() -> None:
    hnet = HNetStribeckPINN(dof=2, seq_len=30)
    qs = torch.randn(4, 30, 2)
    qds = torch.randn(4, 30, 2)
    tau_pred, tau_scv = hnet(qs, qds)
    tau_core = torch.randn(4, 2, requires_grad=True)
    taub = torch.randn(4, 2)
    lf, l_sup, _ = friction_pinn_tau_blend_loss(
        tau_core, tau_pred, tau_scv, taub, lambda_physics=0.0, scv_supervision_loss="mse"
    )
    assert l_sup is not None
    hnet.zero_grad(set_to_none=True)
    tau_core.grad = None
    lf.backward()
    assert hnet.scv.log_k_c.grad is not None
    g = hnet.mlp[0].weight.grad
    assert g is None or g.abs().max() == 0
    assert tau_core.grad is None


def test_tau_blend_consist_trains_fo_not_scv() -> None:
    hnet = HNetStribeckPINN(dof=2, seq_len=30)
    qs = torch.randn(4, 30, 2)
    qds = torch.randn(4, 30, 2)
    tau_pred, tau_scv = hnet(qs, qds)
    tau_core = torch.randn(4, 2)
    taub = torch.randn(4, 2)
    _, _, l_consist = friction_pinn_tau_blend_loss(
        tau_core, tau_pred, tau_scv, taub, lambda_physics=0.0, scv_supervision_loss="mse"
    )
    hnet.zero_grad(set_to_none=True)
    l_consist.backward()
    assert hnet.mlp[0].weight.grad is not None
    assert hnet.scv.log_k_c.grad is None


def test_tau_blend_tau_hat_uses_pred() -> None:
    m = MystericNet(
        dof=2, seq_len=30, friction_backend="stribeck_pinn", pinn_friction_output="pred"
    )
    q = torch.randn(4, 2)
    qd = torch.randn(4, 2)
    qdd = torch.randn(4, 2)
    qs = torch.randn(4, 30, 2)
    qds = torch.randn(4, 30, 2)
    tau_hat, tau_core, tau_fri, _, _, tau_phys = m(q, qd, qdd, qs, qds)
    assert tau_phys is not None
    assert torch.allclose(tau_hat, tau_core + tau_fri)
