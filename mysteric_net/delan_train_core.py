"""
官方 example_DeLaN.py 训练核心：超参、建网、训练循环。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .lnet import LNet
from .replay_memory import PyTorchReplayMemory

# example_DeLaN.py 默认超参
HYPER_EXAMPLE: dict[str, Any] = {
    "n_width": 128,
    "n_depth": 8,
    "diagonal_epsilon": 0.01,
    "activation": "SoftPlus",
    "b_init": 1.0e-3,
    "b_diag_init": 0.01,
    "w_init": "xavier_normal",
    "gain_hidden": float(np.sqrt(2.0)),
    "gain_output": 0.1,
    "n_minibatch": 512,
    "learning_rate": 5.0e-3,
    "weight_decay": 1.0e-5,
    "max_epoch": 10000,
}

# data/delan_model.torch（BAK 2-DoF 上效果好）
HYPER_DELAN_MODEL: dict[str, Any] = {
    "n_width": 64,
    "n_depth": 2,
    "diagonal_epsilon": 0.01,
    "activation": "SoftPlus",
    "b_init": 1.0e-4,
    "b_diag_init": 0.001,
    "w_init": "xavier_normal",
    "gain_hidden": float(np.sqrt(2.0)),
    "gain_output": 0.1,
    "n_minibatch": 512,
    "learning_rate": 5.0e-4,
    "weight_decay": 1.0e-5,
    "max_epoch": 10000,
}


def build_lnet(n_dof: int, hyper: dict[str, Any]) -> LNet:
    return LNet(
        dof=n_dof,
        hidden_dim=int(hyper["n_width"]),
        num_hidden_layers=int(hyper["n_depth"]),
        b_diagonal=float(hyper["b_diag_init"]),
        numerical_H_ridge=float(hyper["diagonal_epsilon"]),
        b_init=float(hyper["b_init"]),
        activation=str(hyper["activation"]),
    )


def train_delan_loop(
    model: LNet,
    train_qp: np.ndarray,
    train_qv: np.ndarray,
    train_qa: np.ndarray,
    train_tau: np.ndarray,
    test_qp: np.ndarray,
    test_qv: np.ndarray,
    test_qa: np.ndarray,
    test_tau: np.ndarray,
    hyper: dict[str, Any],
    *,
    cuda: bool,
    use_energy_loss: bool = True,
) -> int:
    """
    推荐训练循环：每 batch 使用 l_tau + l_E（当前 batch 的能量项）。

    官方 example_DeLaN 使用 ``loss = l_tau + l_mem_mean_dEdt``（epoch 内累积的旧 batch
    能量损失），在 character_data.BAK 上易卡在较差解；本循环与改回前效果一致。
    """
    device = torch.device("cuda" if cuda else "cpu")
    model = model.to(device)

    q = torch.from_numpy(train_qp).to(device=device, dtype=torch.float32)
    qd = torch.from_numpy(train_qv).to(device=device, dtype=torch.float32)
    qdd = torch.from_numpy(train_qa).to(device=device, dtype=torch.float32)
    tau = torch.from_numpy(train_tau).to(device=device, dtype=torch.float32)
    n = q.shape[0]
    batch = int(hyper["n_minibatch"])

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hyper["learning_rate"],
        weight_decay=hyper["weight_decay"],
        amsgrad=True,
    )

    max_epoch = int(hyper["max_epoch"])
    for epoch in range(1, max_epoch + 1):
        perm = torch.randperm(n, device=device)
        loss_acc = tau_acc = e_acc = 0.0
        steps = 0
        for s in range(0, n, batch):
            idx = perm[s : s + batch]
            if idx.numel() < 2:
                continue
            qb, qdb, qddb, taub = q[idx], qd[idx], qdd[idx], tau[idx]
            dyn = model.dynamics(qb, qdb, qddb)
            l_tau = torch.mean((dyn.tau - taub) ** 2)
            if use_energy_loss:
                d_edt = torch.sum(taub * qdb, dim=1)
                l_e = torch.mean((dyn.dTdt + dyn.dVdt - d_edt) ** 2)
                loss = l_tau + l_e
            else:
                l_e = torch.zeros((), device=device, dtype=q.dtype)
                loss = l_tau
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_acc += float(loss.detach())
            tau_acc += float(l_tau.detach())
            e_acc += float(l_e.detach())
            steps += 1

        if epoch == 1 or epoch % 100 == 0 or epoch == max_epoch:
            with torch.no_grad():
                pt = model.inv_dyn(
                    torch.from_numpy(test_qp).float().to(device),
                    torch.from_numpy(test_qv).float().to(device),
                    torch.from_numpy(test_qa).float().to(device),
                )
                test_rmse = float(
                    torch.sqrt(torch.mean((pt.cpu() - torch.from_numpy(test_tau)) ** 2))
                )
            tag = "l_tau+l_E" if use_energy_loss else "l_tau"
            msg = (
                f"epoch {epoch:5d}  {tag}  l={loss_acc / max(steps, 1):.6f}  "
                f"l_tau={tau_acc / max(steps, 1):.6f}"
            )
            if use_energy_loss:
                msg += f"  l_E={e_acc / max(steps, 1):.6f}"
            msg += f"  RMSE_test={test_rmse:.5f}"
            print(msg)

    return max_epoch


def train_delan_official_loop(
    model: LNet,
    train_qp: np.ndarray,
    train_qv: np.ndarray,
    train_qa: np.ndarray,
    train_tau: np.ndarray,
    hyper: dict[str, Any],
    *,
    cuda: bool,
    load_model: bool = False,
) -> int:
    """
    与 example_DeLaN.py 相同的 while 训练循环（仅在与论文脚本逐行对照时使用）。
    """
    n_dof = train_qp.shape[-1]
    device = torch.device("cuda" if cuda else "cpu")
    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hyper["learning_rate"],
        weight_decay=hyper["weight_decay"],
        amsgrad=True,
    )

    mem_dim = ((n_dof,), (n_dof,), (n_dof,), (n_dof,))
    mem = PyTorchReplayMemory(train_qp.shape[0], hyper["n_minibatch"], mem_dim, cuda)
    mem.add_samples([train_qp, train_qv, train_qa, train_tau])

    t0_start = time.perf_counter()
    epoch_i = 0

    while epoch_i < hyper["max_epoch"] and not load_model:
        l_mem_mean_inv_dyn, l_mem_var_inv_dyn = 0.0, 0.0
        l_mem_mean_dEdt, l_mem_var_dEdt = 0.0, 0.0
        l_mem, n_batches = 0.0, 0.0

        for q, qd, qdd, tau in mem:
            optimizer.zero_grad()

            tau_hat, dEdt_hat = model.forward_delan(q, qd, qdd)

            err_inv = torch.sum((tau_hat - tau) ** 2, dim=1)
            l_mean_inv_dyn = torch.mean(err_inv)
            l_var_inv_dyn = torch.var(err_inv)

            dEdt = torch.matmul(
                qd.view(-1, n_dof, 1).transpose(dim0=1, dim1=2),
                tau.view(-1, n_dof, 1),
            ).view(-1)
            err_dEdt = (dEdt_hat - dEdt) ** 2
            l_mean_dEdt = torch.mean(err_dEdt)
            l_var_dEdt = torch.var(err_dEdt)

            loss = l_mean_inv_dyn + l_mem_mean_dEdt
            loss.backward()
            optimizer.step()

            n_batches += 1
            l_mem += loss.item()
            l_mem_mean_inv_dyn += l_mean_inv_dyn.item()
            l_mem_var_inv_dyn += l_var_inv_dyn.item()
            l_mem_mean_dEdt += l_mean_dEdt.item()
            l_mem_var_dEdt += l_var_dEdt.item()

        l_mem_mean_inv_dyn /= float(n_batches)
        l_mem_var_inv_dyn /= float(n_batches)
        l_mem_mean_dEdt /= float(n_batches)
        l_mem_var_dEdt /= float(n_batches)
        l_mem /= float(n_batches)
        epoch_i += 1

        if epoch_i == 1 or np.mod(epoch_i, 100) == 0:
            print("Epoch {0:05d}: ".format(epoch_i), end=" ")
            print("Time = {0:05.1f}s".format(time.perf_counter() - t0_start), end=", ")
            print("Loss = {0:.3e}".format(l_mem), end=", ")
            print(
                "Inv Dyn = {0:.3e} \u00b1 {1:.3e}".format(
                    l_mem_mean_inv_dyn, 1.96 * np.sqrt(l_mem_var_inv_dyn)
                ),
                end=", ",
            )
            print(
                "Power Con = {0:.3e} \u00b1 {1:.3e}".format(
                    l_mem_mean_dEdt, 1.96 * np.sqrt(l_mem_var_dEdt)
                )
            )

    return epoch_i


def save_delan_checkpoint(
    path: Path,
    model: LNet,
    hyper: dict[str, Any],
    epoch: int,
    *,
    extra: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "epoch": epoch,
        "hyper": hyper,
        "state_dict": model.state_dict(),
        "dof": model.n_dof,
        "hidden_dim": model.layers[0].weight.shape[0] if model.layers else hyper["n_width"],
        "num_hidden_layers": len(model.layers),
        "activation": hyper.get("activation", "SoftPlus"),
        "b_init": hyper.get("b_init"),
        "b_diagonal": hyper.get("b_diag_init"),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
