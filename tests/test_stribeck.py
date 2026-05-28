"""Stribeck SCV 与 PINN 摩擦模块冒烟测试。"""

import torch

from RobotDynamics.FrictionModule import HNetStribeck, HNetStribeckPINN, StribeckSCVParams, scv_torque
from RobotDynamics.FrictionModule.stribeck import _init_log_positive, warmstart_scv_from_samples


def test_scv_default_scale() -> None:
    scv = StribeckSCVParams(6)
    c = scv.positive_coefficients()
    assert float(c["k_c"].median()) >= 1.5
    assert float(c["k_s"].median()) >= float(c["k_c"].median())


def test_warmstart_scv_from_samples() -> None:
    scv = StribeckSCVParams(2)
    torch.manual_seed(0)
    qd = torch.randn(16, 2) * 0.5
    tau = qd * 2.0 + 0.1 * torch.randn(16, 2)
    n = warmstart_scv_from_samples(scv, qd, tau)
    assert n == 2
    c = scv.positive_coefficients()
    assert torch.all(c["k_s"] >= c["k_c"] - 1e-9)
    assert float(c["k_c"].min()) > 0.05


def test_scv_k_s_ge_k_c() -> None:
    scv = StribeckSCVParams(4)
    c = scv.positive_coefficients()
    assert torch.all(c["k_s"] >= c["k_c"] - 1e-9)
    # 随机扰动参数后仍满足
    with torch.no_grad():
        for p in scv.parameters():
            p.add_(torch.randn_like(p) * 0.5)
    c2 = scv.positive_coefficients()
    assert torch.all(c2["k_s"] >= c2["k_c"] - 1e-9)


def test_scv_load_legacy_log_k_s() -> None:
    old = StribeckSCVParams(2)
    legacy = {
        "log_k_v": old.log_k_v.clone(),
        "log_k_c": old.log_k_c.clone(),
        "log_k_a": old.log_k_a.clone(),
        "log_k_s": torch.tensor([_init_log_positive(0.2), _init_log_positive(0.3)]),
        "log_v_s": old.log_v_s.clone(),
        "log_alpha": old.log_alpha.clone(),
    }
    new = StribeckSCVParams(2)
    new.load_state_dict(legacy)
    c = new.positive_coefficients()
    assert torch.all(c["k_s"] >= c["k_c"] - 1e-9)


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
