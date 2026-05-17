"""任意 n_dof：数据构建、加载与 LNet 前向冒烟测试。"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from RobotDynamics.DeLaN import (
    build_lnet,
    build_pickle_dict,
    import_npz,
    load_dataset,
    save_pickle,
    suggest_hyper,
    validate_pickle_raw,
)


def _synthetic_traj(T: int, n_dof: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    qp = rng.standard_normal((T, n_dof))
    qv = rng.standard_normal((T, n_dof)) * 0.1
    qa = rng.standard_normal((T, n_dof)) * 0.1
    tau = rng.standard_normal((T, n_dof))
    return {"t": np.arange(T) * 0.01, "qp": qp, "qv": qv, "qa": qa, "tau": tau}


@pytest.mark.parametrize("n_dof", [1, 2, 6, 7])
def test_lnet_forward_shapes(n_dof: int) -> None:
    hyper = suggest_hyper(n_dof, 500, base="delan_model")
    model = build_lnet(n_dof, hyper)
    B = 8
    q = torch.randn(B, n_dof)
    qd = torch.randn(B, n_dof)
    qdd = torch.randn(B, n_dof)
    dyn = model.dynamics(q, qd, qdd)
    assert dyn.tau.shape == (B, n_dof)
    assert dyn.H.shape == (B, n_dof, n_dof)


def test_pickle_without_mcg() -> None:
    data = build_pickle_dict(
        [_synthetic_traj(50, 4, 0), _synthetic_traj(40, 4, 1)],
        synthesize_decomposition=True,
    )
    n_dof, has_mcg = validate_pickle_raw(data)
    assert n_dof == 4
    assert has_mcg is False


def test_load_dataset_test_frac() -> None:
    data = build_pickle_dict(
        [_synthetic_traj(30, 3, i) for i in range(5)],
    )
    for i, lab in enumerate(["a", "b", "c", "d", "e"]):
        data["labels"][i] = lab

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "t.pickle"
        save_pickle(data, path)
        train_data, test_data, _div, _dt = load_dataset(
            filename=str(path),
            test_label=(),
            test_frac=0.4,
        )
        _tl, train_qp, *_ = train_data
        _testl, test_qp, *_ = test_data
        assert train_qp.shape[0] > 0
        assert test_qp.shape[0] > 0
        assert train_qp.shape[1] == 3


def test_import_npz_roundtrip() -> None:
    tr = _synthetic_traj(60, 5, 3)
    with tempfile.TemporaryDirectory() as td:
        npz_path = Path(td) / "one.npz"
        np.savez(npz_path, **tr)
        data = import_npz(npz_path)
        assert data["qp"][0].shape == (60, 5)
