#!/usr/bin/env python3
"""
数值验证：LNet 与官方 DeepLagrangianNetwork 是否同一套算法。

用法:
  python scripts/verify_delan_impl.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DLN = Path("/home/coral/project/deep_lagrangian_networks")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(DLN))

from deep_lagrangian_networks.DeLaN_model import DeepLagrangianNetwork  # noqa: E402
from mysteric_net.delan_data import load_dataset  # noqa: E402
from mysteric_net.lnet import LNet  # noqa: E402

DATA = DLN / "data" / "character_data.pickle.BAK"
CKPT = DLN / "data" / "delan_model.torch"


def _maxdiff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach() - b.detach()).abs().max().cpu())


def compare_same_weights() -> None:
    """相同随机权重下，逐步对比所有动力学量。"""
    torch.manual_seed(0)
    hyper = dict(
        n_width=64,
        n_depth=2,
        diagonal_epsilon=0.01,
        activation="SoftPlus",
        b_init=1e-4,
        b_diag_init=0.001,
    )
    off = DeepLagrangianNetwork(2, **hyper)
    lnet = LNet(
        2,
        hidden_dim=64,
        num_hidden_layers=2,
        b_diagonal=0.001,
        numerical_H_ridge=0.01,
        b_init=1e-4,
        activation="SoftPlus",
    )
    lnet.load_state_dict(off.state_dict())

    _, td, _, _ = load_dataset(filename=str(DATA))
    _, qp, qv, qa, _, _, tau, m, c, g = td
    q = torch.from_numpy(qp[:32]).float()
    qd = torch.from_numpy(qv[:32]).float()
    qdd = torch.from_numpy(qa[:32]).float()

    with torch.no_grad():
        dyn = lnet.dynamics(q, qd, qdd)
        t_off, H_off, c_off, g_off, T_off, V_off, dT_off, dV_off = off._dyn_model(q, qd, qdd)

    checks = [
        ("tau", dyn.tau, t_off),
        ("H", dyn.H, H_off),
        ("c", dyn.c, c_off),
        ("g", dyn.g, g_off),
        ("T", dyn.T, T_off),
        ("V", dyn.V, V_off.squeeze() if V_off.dim() > 1 else V_off),
        ("dTdt", dyn.dTdt, dT_off),
        ("dVdt", dyn.dVdt, dV_off),
    ]
    print("=== 1. 相同权重：LNet vs 官方 _dyn_model ===")
    for name, a, b in checks:
        print(f"  {name:6s} max|diff| = {_maxdiff(a, b):.3e}")
    print(f"  inv_dyn vs forward tau = {_maxdiff(dyn.tau, off(q, qd, qdd)[0]):.3e}")


def check_physics_identities() -> None:
    """内部物理恒等式（与实现无关，应在 LNet 上成立）。"""
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    h = ckpt["hyper"]
    model = LNet(
        2,
        hidden_dim=h["n_width"],
        num_hidden_layers=h["n_depth"],
        b_diagonal=h["b_diag_init"],
        numerical_H_ridge=h["diagonal_epsilon"],
        b_init=h["b_init"],
        activation=h["activation"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    _, td, _, _ = load_dataset(filename=str(DATA))
    _, qp, qv, qa, _, _, tau, m, c, g = td
    q = torch.from_numpy(qp).float()
    qd = torch.from_numpy(qv).float()
    qdd = torch.from_numpy(qa).float()

    with torch.no_grad():
        dyn = model.dynamics(q, qd, qdd)
        zq = torch.zeros_like(qd)
        zqd = torch.zeros_like(qdd)
        g_hat = model.inv_dyn(q, zq, zqd)
        c_hat = model.inv_dyn(q, qd, zqd) - g_hat
        m_hat = model.inv_dyn(q, zq, qdd) - g_hat
        tau_hat = dyn.tau

    print("\n=== 2. 物理分解恒等式（官方评估同款）===")
    print(f"  max|tau - (m+c+g)|_data     = {np.abs(tau - (m+c+g)).max():.3e}  (数据集真值)")
    print(f"  max|tau_hat - (m_hat+c_hat+g_hat)| = {(tau_hat.numpy() - (m_hat+c_hat+g_hat).numpy()).max():.3e}")
    print(f"  max|tau_hat - tau|_test      = {np.abs(tau_hat.numpy() - tau).max():.3e}")
    print(f"  MSE(tau_hat, tau)           = {np.mean((tau_hat.numpy() - tau) ** 2):.3e}")


def compare_pretrained_official() -> None:
    """官方预训练权重在 BAK 上的表现 — 若实现错，不可能达到该精度。"""
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    h = ckpt["hyper"]
    model = LNet(
        2,
        hidden_dim=h["n_width"],
        num_hidden_layers=h["n_depth"],
        b_diagonal=h["b_diag_init"],
        numerical_H_ridge=h["diagonal_epsilon"],
        b_init=h["b_init"],
        activation=h["activation"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    _, td, _, _ = load_dataset(filename=str(DATA))
    _, qp, qv, qa, _, _, tau, m, c, g = td
    q = torch.from_numpy(qp).float()
    qd = torch.from_numpy(qv).float()
    qdd = torch.from_numpy(qa).float()
    with torch.no_grad():
        pred = model.inv_dyn(q, qd, qdd).numpy()
        g_pred = model.inv_dyn(q, torch.zeros_like(qd), torch.zeros_like(qdd)).numpy()
    print("\n=== 3. 官方 delan_model.torch + 我们的 LNet 前向 ===")
    print(f"  Torque MSE (test) = {np.mean((pred - tau) ** 2):.3e}  (README ~4e-4)")
    print(f"  g MSE             = {np.mean((g_pred - g) ** 2):.3e}")


def main() -> None:
    if not DATA.is_file():
        raise SystemExit(f"缺少数据: {DATA}")
    compare_same_weights()
    check_physics_identities()
    if CKPT.is_file():
        compare_pretrained_official()
    else:
        print("\n(跳过 checkpoint 测试：未找到 delan_model.torch)")
    print("\n若第 1 节各项 diff≈0，则动力学算法与官方一致；第 3 节说明权重正确时精度可达 README 水平。")


if __name__ == "__main__":
    main()
