import torch

from RobotDynamics.FrictionModule import HNetFOCascade, HNetFOCascadePINN
from RobotDynamics.FrictionModule.fo_cascade import StribeckMLP, _build_stribeck_mlp


def test_fo_cascade_shapes_and_causality() -> None:
    dof, L, B = 2, 30, 3
    net = HNetFOCascade(dof=dof, seq_len=L)
    q_seq = torch.randn(B, L, dof)
    qd_seq = torch.randn(B, L, dof)
    tau = net(q_seq, qd_seq)
    assert tau.shape == (B, dof)
    qd_t = qd_seq[:, -1, :]
    mask = qd_t.abs() > 0.01
    assert torch.all((tau * qd_t)[mask] >= -1e-6)

    tau2, v_last, s_last, s_raw_last, v_seq = net.forward_with_internals(q_seq, qd_seq)
    assert tau2.shape == (B, dof)
    assert v_last.shape == (B, dof)
    assert s_last.shape == (B, dof)
    assert s_raw_last.shape == (B, dof)
    assert v_seq.shape == (B, L, dof)


def test_stribeck_mlp_two_layer_tanh() -> None:
    mlp = StribeckMLP(6, 24)
    assert mlp.hidden_dim == 24
    x = torch.randn(5, 6, requires_grad=True)
    y = mlp(x)
    assert y.shape == (5, 6)
    y.sum().backward()
    assert x.grad is not None


def test_build_stribeck_mlp_returns_two_layer() -> None:
    mlp = _build_stribeck_mlp(6, 24)
    assert isinstance(mlp, StribeckMLP)
    assert len(mlp.net) == 3


def test_fo_cascade_pinn_returns_physics() -> None:
    dof, L, B = 2, 30, 3
    net = HNetFOCascadePINN(dof=dof, seq_len=L)
    assert net.tcn_layers == 3
    q_seq = torch.randn(B, L, dof)
    qd_seq = torch.randn(B, L, dof)
    tau_pred, tau_phys = net(q_seq, qd_seq)
    assert tau_pred.shape == (B, dof)
    assert tau_phys.shape == (B, dof)
