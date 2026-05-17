"""Stribeck SCV 与 PINN 摩擦模块冒烟测试。"""

import torch

from RobotDynamics.FrictionModule import HNetStribeck, HNetStribeckPINN, scv_torque


def test_scv_zero_velocity_smooth() -> None:
    qd = torch.zeros(4, 2)
    k_v = torch.ones(2) * 0.01
    k_c = torch.ones(2) * 0.1
    k_a = torch.ones(2) * 5.0
    k_s = torch.ones(2) * 0.2
    v_s = torch.ones(2) * 0.05
    alpha = torch.ones(2) * 1.5
    tau = scv_torque(qd, k_v, k_c, k_a, k_s, v_s, alpha)
    assert tau.shape == (4, 2)
    assert torch.allclose(tau, torch.zeros_like(tau), atol=1e-5)


def test_hnet_stribeck_pinn_shapes() -> None:
    B, L, dof = 8, 30, 6
    net = HNetStribeckPINN(dof, seq_len=L)
    qs = torch.randn(B, L, dof)
    qds = torch.randn(B, L, dof)
    pred, phys = net(qs, qds)
    assert pred.shape == (B, dof)
    assert phys.shape == (B, dof)


def test_hnet_stribeck_forward() -> None:
    net = HNetStribeck(3)
    qs = torch.randn(2, 10, 3)
    qds = torch.randn(2, 10, 3)
    tau, phys = net(qs, qds)
    assert tau.shape == (2, 3)
    assert phys.shape == (2, 3)
