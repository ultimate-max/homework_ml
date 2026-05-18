import torch

from RobotDynamics.FrictionModule import friction_pinn_loss, friction_supervised_loss


def test_friction_smape_bounded() -> None:
    pred = torch.tensor([[0.1, 0.0], [2.0, -0.5]])
    target = torch.tensor([[0.2, 0.1], [1.5, -0.3]])
    s = friction_supervised_loss(pred, target, "smape", smape_eps=1e-3)
    m = friction_supervised_loss(pred, target, "mse")
    assert 0.0 < float(s) < 2.0
    assert float(m) > 0.0


def test_friction_pinn_smape() -> None:
    pred = torch.randn(8, 6)
    target = torch.randn(8, 6)
    phys = torch.randn(8, 6)
    total, l_data, l_phys = friction_pinn_loss(
        pred, target, phys, lambda_physics=0.5, fri_loss="smape"
    )
    assert total.shape == ()
    assert l_data.shape == l_phys.shape == ()
