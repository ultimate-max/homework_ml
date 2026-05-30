"""多模型对比评估的共享逻辑（robot_compare_evaluate / robot_compare_metrics）。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from robot_evaluate import build_test_tensors, load_mysteric_checkpoint, predict


@dataclass
class ModelEval:
    name: str
    path: Path
    backend: str
    pred: dict[str, np.ndarray]
    rmse_tau: float
    rmse_tau_fri: float | None
    rmse_tau_j: np.ndarray
    rmse_tau_fri_j: np.ndarray | None


def parse_checkpoint_arg(s: str) -> tuple[Path, str]:
    """``path`` 或 ``path:label``。"""
    if ":" in s:
        path_s, _, label = s.partition(":")
        label = label.strip() or Path(path_s).stem
    else:
        path_s = s
        label = Path(s).stem
    return Path(path_s), label


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def rmse_per_joint(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((pred - target) ** 2, axis=0))


def eval_model(
    name: str,
    path: Path,
    data: dict[str, np.ndarray],
    device: torch.device,
) -> ModelEval:
    model, ckpt = load_mysteric_checkpoint(path, device)
    if model.dof != data["tau"].shape[1]:
        raise ValueError(
            f"模型 {name!r} n_dof={model.dof} 与数据 n_dof={data['tau'].shape[1]} 不一致"
        )
    pred = predict(model, data, device)
    tau = data["tau"]
    rmse_tau = rmse(pred["tau_hat"], tau)
    rmse_tau_j = rmse_per_joint(pred["tau_hat"], tau)

    tau_fri_true = data["tau_fri_true"]
    has_fri_ref = np.any(np.isfinite(tau_fri_true))
    if has_fri_ref:
        rmse_fri = rmse(pred["tau_fri"], tau_fri_true)
        rmse_fri_j = rmse_per_joint(pred["tau_fri"], tau_fri_true)
    else:
        rmse_fri = None
        rmse_fri_j = None

    backend = str(ckpt.get("friction_backend", "?"))
    return ModelEval(
        name=name,
        path=path,
        backend=backend,
        pred=pred,
        rmse_tau=rmse_tau,
        rmse_tau_fri=rmse_fri,
        rmse_tau_j=rmse_tau_j,
        rmse_tau_fri_j=rmse_fri_j,
    )


def resolve_seq_len(
    ckpt_paths: list[Path],
    seq_len_cli: int | None,
) -> int:
    seq_lens: list[int] = []
    for path in ckpt_paths:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict):
            seq_lens.append(int(ckpt.get("seq_len", 30)))
    return int(seq_len_cli or max(seq_lens, default=30))


def load_test_data(
    data_path: Path,
    test_labels: list[str],
    seq_len: int,
    device: torch.device,
):
    from RobotDynamics.FrictionModule import load_pickle_trajectories

    raw = load_pickle_trajectories(str(data_path))
    data, traj_labels, divider = build_test_tensors(
        raw, test_labels, seq_len, device
    )
    has_fri_ref = np.any(np.isfinite(data["tau_fri_true"]))
    return raw, data, traj_labels, divider, has_fri_ref


def eval_checkpoints(
    ckpt_specs: list[tuple[Path, str]],
    data: dict[str, np.ndarray],
    device: torch.device,
    *,
    verbose: bool = True,
) -> list[ModelEval]:
    results: list[ModelEval] = []
    for path, name in ckpt_specs:
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint 不存在: {path}")
        if verbose:
            print(f"加载 {name} ← {path}")
        results.append(eval_model(name, path, data, device))
    return results
