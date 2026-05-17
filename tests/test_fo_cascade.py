import torch
import torch.nn as nn

from RobotDynamics.FrictionModule import HNetFOCascade, HNetFOCascadePINN
from RobotDynamics.FrictionModule.fo_cascade import _build_stribeck_mlp


def test_fo_cascade_shapes_and_causality() -> None:
    dof, L, B = 2, 30, 3
    net = HNetFOCascade(dof=dof, seq_len=L)
    q_seq = torch.randn(B, L, dof)
    qd_seq = torch.randn(B, L, dof)
    tau = net(q_seq, qd_seq)
    assert tau.shape == (B, dof)

    tau2, v_last, s_last, s_raw_last, v_seq = net.forward_with_internals(q_seq, qd_seq)
    assert tau2.shape == (B, dof)
    assert v_last.shape == (B, dof)
    assert s_last.shape == (B, dof)
    assert s_raw_last.shape == (B, dof)
    assert v_seq.shape == (B, L, dof)


def test_integrator_1s_attensuates_high_frequency() -> None:
    from RobotDynamics.FrictionModule.fo_cascade import _CausalIntegrator1s

    B, L, dof = 1, 64, 1
    t = torch.arange(L, dtype=torch.float32).view(1, L, 1)
    x = torch.sin(2 * 3.14159 * 8 * t / L) + 0.2 * torch.sin(2 * 3.14159 * 20 * t / L)
    x = x.expand(B, L, dof)
    y = _CausalIntegrator1s(dof, init_alpha=0.15, init_leak=0.98)(x)
    hf_ratio = (torch.diff(y, dim=1).pow(2).mean() / torch.diff(x, dim=1).pow(2).mean()).item()
    assert hf_ratio < 0.5


def test_stribeck_mlp_has_multiple_hidden_layers() -> None:
    mlp = _build_stribeck_mlp(6, 24, num_hidden_layers=3)
    n_linear = sum(1 for m in mlp.modules() if isinstance(m, nn.Linear))
    n_tanh = sum(1 for m in mlp.modules() if isinstance(m, nn.Tanh))
    assert n_linear == 4
    assert n_tanh == 3
    y = mlp(torch.randn(5, 6))
    assert y.shape == (5, 6)


def test_fo_cascade_pinn_returns_physics() -> None:
    dof, L, B = 2, 30, 3
    net = HNetFOCascadePINN(dof=dof, seq_len=L)
    q_seq = torch.randn(B, L, dof)
    qd_seq = torch.randn(B, L, dof)
    tau_pred, tau_phys = net(q_seq, qd_seq)
    assert tau_pred.shape == (B, dof)
    assert tau_phys.shape == (B, dof)
