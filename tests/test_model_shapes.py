import torch
import pytest

from RobotDynamics.MystericNet import MystericNet


@pytest.mark.parametrize(
    "backend",
    ["tcn", "fo_cascade", "fo_cascade_pinn", "stribeck", "stribeck_pinn", "gms", "gms_pinn"],
)
def test_mysteric_forward_shapes(backend: str) -> None:
    dof, L, B = 2, 30, 4
    pinn_out = "physics" if backend in ("stribeck_pinn", "fo_cascade_pinn", "gms_pinn") else "pred"
    m = MystericNet(
        dof=dof,
        seq_len=L,
        friction_backend=backend,
        pinn_friction_output=pinn_out,
    )
    q = torch.randn(B, dof)
    qd = torch.randn(B, dof)
    qdd = torch.randn(B, dof)
    qs = torch.randn(B, L, dof)
    qds = torch.randn(B, L, dof)
    tau_hat, tau_core, tau_fri, H_hat, g_hat, tau_phys = m(q, qd, qdd, qs, qds)
    assert tau_hat.shape == (B, dof)
    assert tau_core.shape == (B, dof)
    assert tau_fri.shape == (B, dof)
    assert H_hat.shape == (B, dof, dof)
    assert g_hat.shape == (B, dof)
    if backend in ("stribeck_pinn", "fo_cascade_pinn", "gms_pinn"):
        assert tau_phys is not None and tau_phys.shape == (B, dof)
    else:
        assert tau_phys is None
