"""GMS 摩擦模块冒烟测试。"""

import torch

from RobotDynamics.FrictionModule import (
    GmsParams,
    HNetGMS,
    HNetGMSPINN,
    gms_integrate,
    gms_limit_surface,
)


def test_gms_limit_surface_shape() -> None:
    qd = torch.randn(4, 2)
    v_a = torch.ones(2) * 0.1
    k_str = torch.ones(2) * 0.05
    v_s = torch.ones(2) * 0.05
    delta = torch.ones(2) * 1.5
    s = gms_limit_surface(qd, v_a, k_str, v_s, delta)
    assert s.shape == (4, 2)


def test_gms_integrate_hysteresis_history() -> None:
    """同末速不同历史 → 摩擦不同（迟滞）。"""
    dof = 1
    n_elem = 2
    b, length = 1, 40
    dt = 0.01
    qd_hist = torch.zeros(b, length, dof)
    qd_hist[:, :20, 0] = 0.5
    qd_hist[:, 20:, 0] = -0.5
    qd_mono = torch.ones(b, length, dof) * -0.5

    k_i = torch.tensor([[10.0, 100.0]])
    c_i = torch.tensor([[5.0, 50.0]])
    v_a = torch.tensor([0.2])
    k_str = torch.tensor([0.05])
    v_s = torch.tensor([0.05])
    delta = torch.tensor([1.5])
    sigma_1 = torch.tensor([0.01])

    tau_hist = gms_integrate(
        qd_hist, k_i=k_i, c_i=c_i, v_a=v_a, k_str=k_str, v_s=v_s,
        delta=delta, sigma_1=sigma_1, dt=dt,
    )
    tau_mono = gms_integrate(
        qd_mono, k_i=k_i, c_i=c_i, v_a=v_a, k_str=k_str, v_s=v_s,
        delta=delta, sigma_1=sigma_1, dt=dt,
    )
    assert tau_hist.shape == (1, 1)
    assert not torch.allclose(tau_hist, tau_mono, atol=1e-3)


def test_hnet_gms_forward_and_grad() -> None:
    dof, length, b = 3, 20, 4
    net = HNetGMS(dof, seq_len=length, n_blocks=2, dt=0.01)
    qs = torch.randn(b, length, dof, requires_grad=True)
    qds = torch.randn(b, length, dof, requires_grad=True)
    tau, phys = net(qs, qds)
    assert tau.shape == (b, dof)
    assert phys.shape == (b, dof)
    loss = tau.sum()
    loss.backward()
    assert net.gms.log_v_a.grad is not None


def test_hnet_gms_pinn_shapes() -> None:
    b, length, dof = 8, 30, 6
    net = HNetGMSPINN(dof, seq_len=length, n_blocks=3)
    qs = torch.randn(b, length, dof)
    qds = torch.randn(b, length, dof)
    pred, phys = net(qs, qds)
    assert pred.shape == (b, dof)
    assert phys.shape == (b, dof)


def test_gms_params_forward() -> None:
    qd_seq = torch.randn(5, 10, 2)
    gms = GmsParams(2, n_blocks=3)
    tau = gms(qd_seq, dt=0.001)
    assert tau.shape == (5, 2)
