import torch

from mysteric_net.model import MystericNet


def test_mysteric_forward_shapes():
    dof, L, B = 2, 30, 4
    m = MystericNet(dof=dof, seq_len=L)
    q = torch.randn(B, dof)
    qd = torch.randn(B, dof)
    qdd = torch.randn(B, dof)
    qs = torch.randn(B, L, dof)
    qds = torch.randn(B, L, dof)
    tau_hat, tau_core, tau_fri, M, g = m(q, qd, qdd, qs, qds)
    assert tau_hat.shape == (B, dof)
    assert tau_core.shape == (B, dof)
    assert tau_fri.shape == (B, dof)
    assert M.shape == (B, dof, dof)
    assert g.shape == (B, dof)
