import torch

from RobotDynamics.FrictionModule import HNetFOCascade


def test_fo_cascade_shapes_and_causality() -> None:
    dof, L, B = 2, 30, 3
    net = HNetFOCascade(dof=dof, seq_len=L)
    q_seq = torch.randn(B, L, dof)
    tau = net(q_seq)
    assert tau.shape == (B, dof)

    tau2, v_last, s_last, v_seq = net.forward_with_internals(q_seq)
    assert tau2.shape == (B, dof)
    assert v_last.shape == (B, dof)
    assert s_last.shape == (B, dof)
    assert v_seq.shape == (B, L, dof)
