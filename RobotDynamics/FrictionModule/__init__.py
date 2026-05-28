"""
FrictionModule：关节摩擦建模（TCN / Stribeck SCV / PINN）。

  from RobotDynamics.FrictionModule import HNetTCN, HNetStribeckPINN, friction_pinn_loss, ...
"""

from .energy_loss import mysteric_losses
from .losses import (
    friction_pinn_loss,
    friction_pinn_tau_blend_loss,
    friction_supervised_loss,
)
from .sequence_data import (
    build_mysteric_tensors,
    load_pickle_trajectories,
    pickle_has_mcg_decomposition,
    stack_trajectories_to_flat,
)
from .gms import (
    GmsParams,
    HNetGMS,
    HNetGMSPINN,
    gms_integrate,
    gms_limit_surface,
    warmstart_gms_from_samples,
)
from .stribeck import (
    HNetStribeck,
    HNetStribeckPINN,
    StribeckSCVParams,
    cv_torque,
    scv_torque,
    warmstart_scv_from_samples,
)
from .synthetic_plant import build_windows, simulate_2dof_inverse_dynamics
from .fo_cascade import HNetFOCascade, HNetFOCascadePINN, StribeckMLP
from .tcn import HNetTCN

__all__ = [
    "HNetFOCascade",
    "HNetFOCascadePINN",
    "StribeckMLP",
    "HNetTCN",
    "HNetGMS",
    "HNetGMSPINN",
    "GmsParams",
    "gms_integrate",
    "gms_limit_surface",
    "warmstart_gms_from_samples",
    "HNetStribeck",
    "HNetStribeckPINN",
    "StribeckSCVParams",
    "cv_torque",
    "scv_torque",
    "warmstart_scv_from_samples",
    "friction_pinn_loss",
    "friction_pinn_tau_blend_loss",
    "friction_supervised_loss",
    "mysteric_losses",
    "build_windows",
    "simulate_2dof_inverse_dynamics",
    "build_mysteric_tensors",
    "load_pickle_trajectories",
    "stack_trajectories_to_flat",
    "pickle_has_mcg_decomposition",
]
