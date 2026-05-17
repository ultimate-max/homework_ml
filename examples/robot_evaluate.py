#!/usr/bin/env python3
"""
评估 robot_train.py 保存的 MystericNet，绘制摩擦力矩与总力矩曲线。

示例:
  python examples/robot_evaluate.py \\
    --checkpoint checkpoints/mysteric_robot.pt \\
    --data data/robot.pickle \\
    --test-labels e v q \\
    --figure-out figures/robot_friction.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from RobotDynamics.DeLaN import load_dataset
from RobotDynamics.FrictionModule import (
    build_windows,
    load_pickle_trajectories,
    pickle_has_mcg_decomposition,
)
from RobotDynamics.MystericNet import MystericNet


def _infer_lnet_hyper(state: dict) -> tuple[int, int]:
    w0 = state["lnet.layers.0.weight"]
    n_width = int(w0.shape[0])
    layer_ids = [
        int(k.split(".")[2])
        for k in state
        if k.startswith("lnet.layers.") and k.endswith(".weight")
    ]
    n_depth = max(layer_ids) + 1 if layer_ids else 2
    return n_width, n_depth


def _infer_pinn_hidden(state: dict, dof: int) -> tuple[int, ...]:
    """从 checkpoint 的 Linear 层推断 MLP 隐层宽度（不含输出层 dof）。"""
    widths: list[int] = []
    i = 0
    while f"hnet.mlp.{i}.weight" in state:
        w = state[f"hnet.mlp.{i}.weight"]
        if int(w.shape[0]) == dof:
            break
        widths.append(int(w.shape[0]))
        i += 2
    return tuple(widths) if widths else (128, 64)


def load_mysteric_checkpoint(path: Path, device: torch.device) -> tuple[MystericNet, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt["state_dict"]
    dof = int(ckpt["dof"])
    seq_len = int(ckpt.get("seq_len", 30))
    backend = ckpt.get("friction_backend", "stribeck_pinn")
    l_w = int(ckpt.get("lnet_hidden", _infer_lnet_hyper(state)[0]))
    l_d = int(ckpt.get("lnet_layers", _infer_lnet_hyper(state)[1]))
    kw: dict = dict(
        dof=dof,
        seq_len=seq_len,
        lnet_hidden=l_w,
        lnet_layers=l_d,
        friction_backend=backend,
    )
    if backend == "stribeck_pinn":
        kw["stribeck_hidden"] = _infer_pinn_hidden(state, dof)
    model = MystericNet(**kw).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, ckpt


def build_test_tensors(
    raw: dict,
    test_labels: list[str],
    seq_len: int,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], list[str], list[int]]:
    """按测试轨迹滑窗，拼接样本并记录轨迹边界。"""
    label_set = set(test_labels)
    has_mcg = pickle_has_mcg_decomposition(raw)

    chunks: dict[str, list[np.ndarray]] = {
        "qp": [],
        "qv": [],
        "qa": [],
        "tau": [],
        "tau_rigid": [],
        "tau_fri_true": [],
        "q_seq": [],
        "qd_seq": [],
    }
    used_labels: list[str] = []
    divider = [0]

    for i, lab in enumerate(raw["labels"]):
        if lab not in label_set:
            continue
        qp = np.asarray(raw["qp"][i], dtype=np.float64)
        qv = np.asarray(raw["qv"][i], dtype=np.float64)
        qa = np.asarray(raw["qa"][i], dtype=np.float64)
        tau = np.asarray(raw["tau"][i], dtype=np.float64)
        if qp.shape[0] < seq_len:
            continue

        q = torch.from_numpy(qp).float()
        qd = torch.from_numpy(qv).float()
        qdd = torch.from_numpy(qa).float()
        t = torch.from_numpy(tau).float()
        qi, qdi, qddi, taui, q_seq, qd_seq = build_windows(q, qd, qdd, t, seq_len)

        m = np.asarray(raw["m"][i], dtype=np.float64)
        c = np.asarray(raw["c"][i], dtype=np.float64)
        g = np.asarray(raw["g"][i], dtype=np.float64)
        tau_rigid = m + c + g
        tau_fri_true = tau - tau_rigid if has_mcg else np.full_like(tau, np.nan)

        n = int(qi.shape[0])
        chunks["qp"].append(qi.numpy())
        chunks["qv"].append(qdi.numpy())
        chunks["qa"].append(qddi.numpy())
        chunks["tau"].append(taui.numpy())
        chunks["tau_rigid"].append(
            tau_rigid[seq_len - 1 : seq_len - 1 + n]
        )
        chunks["tau_fri_true"].append(
            tau_fri_true[seq_len - 1 : seq_len - 1 + n]
        )
        chunks["q_seq"].append(q_seq.numpy())
        chunks["qd_seq"].append(qd_seq.numpy())
        used_labels.append(lab)
        divider.append(divider[-1] + n)

    if not used_labels:
        raise ValueError(f"测试标签 {test_labels} 在数据中未找到任何轨迹")

    out = {k: np.vstack(v) for k, v in chunks.items()}
    return out, used_labels, divider


@torch.no_grad()
def predict(
    model: MystericNet,
    data: dict[str, np.ndarray],
    device: torch.device,
) -> dict[str, np.ndarray]:
    qp = torch.from_numpy(data["qp"]).to(device=device, dtype=torch.float32)
    qv = torch.from_numpy(data["qv"]).to(device=device, dtype=torch.float32)
    qa = torch.from_numpy(data["qa"]).to(device=device, dtype=torch.float32)
    qs = torch.from_numpy(data["q_seq"]).to(device=device, dtype=torch.float32)
    qds = torch.from_numpy(data["qd_seq"]).to(device=device, dtype=torch.float32)

    tau_hat, tau_core, tau_fri, _H, _g, tau_phys = model(qp, qv, qa, qs, qds)
    out = {
        "tau_hat": tau_hat.cpu().numpy(),
        "tau_core": tau_core.cpu().numpy(),
        "tau_fri": tau_fri.cpu().numpy(),
    }
    if tau_phys is not None:
        out["tau_fri_phys"] = tau_phys.cpu().numpy()
    return out


def print_scv_params(model: MystericNet) -> None:
    hnet = model.hnet
    if not hasattr(hnet, "scv"):
        print("当前摩擦后端无 SCV 参数表（tcn / fo_cascade）。")
        return
    scv = hnet.scv
    names = ("k_v", "k_c", "k_a", "k_s", "v_s", "alpha")
    logs = (
        scv.log_k_v,
        scv.log_k_c,
        scv.log_k_a,
        scv.log_k_s,
        scv.log_v_s,
        scv.log_alpha,
    )
    print("\n学到的 SCV 参数（每关节）:")
    hdr = "joint  " + "  ".join(f"{n:>8}" for n in names)
    print(hdr)
    for j in range(scv.dof):
        vals = []
        for name, log_p in zip(names, logs):
            v = torch.nn.functional.softplus(log_p[j]).item()
            if name == "alpha":
                v = max(v, 0.5)
            vals.append(v)
        print(f"  J{j}   " + "  ".join(f"{v:8.4f}" for v in vals))


def plot_results(
    data: dict[str, np.ndarray],
    pred: dict[str, np.ndarray],
    labels: list[str],
    divider: list[int],
    *,
    figure_out: Path,
    show: bool,
) -> None:
    import matplotlib.pyplot as plt

    tau = data["tau"]
    tau_fri = pred["tau_fri"]
    tau_core = pred["tau_core"]
    tau_hat = pred["tau_hat"]
    tau_fri_true = data["tau_fri_true"]
    has_true_fri = np.any(np.isfinite(tau_fri_true))
    has_phys = "tau_fri_phys" in pred

    n_dof = tau.shape[1]
    n_rows = n_dof
    n_cols = 3 if has_phys else 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 2.2 * n_rows), squeeze=False)
    if has_phys:
        fig.suptitle("τ_fri pred | τ_fri SCV physics | τ total (core+fri vs meas)", fontsize=12)
    else:
        fig.suptitle("τ_fri pred | τ total (core+fri vs meas)", fontsize=12)

    ticks = [(divider[i] + divider[i + 1]) / 2 for i in range(len(labels))]
    x = np.arange(tau.shape[0])

    for j in range(n_dof):
        col = 0
        ax = axes[j, col]
        ax.plot(x, tau_fri[:, j], "r", alpha=0.85, label="τ_fri pred (MLP/SCV/TCN)")
        if has_true_fri:
            ax.plot(x, tau_fri_true[:, j], "k", alpha=0.7, label="τ_fri ref (τ-m-c-g)")
        ax.set_ylabel(f"J{j} [Nm]")
        if j == 0:
            ax.set_title("Friction estimate")
        if j == n_dof - 1:
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels)
            for d in divider:
                ax.axvline(d, color="gray", ls="--", lw=0.4)
        else:
            ax.set_xticks([])
        if j == 0:
            ax.legend(loc="upper right", fontsize=7)

        if has_phys:
            col += 1
            axp = axes[j, col]
            axp.plot(x, pred["tau_fri_phys"][:, j], "b", alpha=0.85, label="τ_fri SCV")
            axp.plot(x, tau_fri[:, j], "r", alpha=0.5, ls="--", label="τ_fri pred")
            if j == 0:
                axp.set_title("PINN physics branch")
            if j == n_dof - 1:
                axp.set_xticks(ticks)
                axp.set_xticklabels(labels)
                for d in divider:
                    axp.axvline(d, color="gray", ls="--", lw=0.4)
            else:
                axp.set_xticks([])
            if j == 0:
                axp.legend(loc="upper right", fontsize=7)

        col = n_cols - 1
        axt = axes[j, col]
        axt.plot(x, tau[:, j], "k", alpha=0.8, label="τ meas")
        axt.plot(x, tau_hat[:, j], "r", alpha=0.75, label="τ_hat")
        axt.plot(x, tau_core[:, j], "g", alpha=0.5, ls=":", label="τ_core")
        if j == 0:
            axt.set_title("Total torque")
        if j == n_dof - 1:
            axt.set_xticks(ticks)
            axt.set_xticklabels(labels)
            for d in divider:
                axt.axvline(d, color="gray", ls="--", lw=0.4)
        else:
            axt.set_xticks([])
        if j == 0:
            axt.legend(loc="upper right", fontsize=7)

    figure_out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(figure_out, dpi=120, bbox_inches="tight")
    print(f"Figure saved: {figure_out}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="MystericNet 摩擦与力矩评估绘图")
    p.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints" / "mysteric_robot.pt")
    p.add_argument("--data", type=Path, default=ROOT / "data" / "robot.pickle")
    p.add_argument("--test-labels", nargs="*", default=["e", "v", "q"])
    p.add_argument("--seq-len", type=int, default=None, help="默认从 checkpoint 读取")
    p.add_argument("--figure-out", type=Path, default=ROOT / "figures" / "robot_friction.png")
    p.add_argument("--show", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_mysteric_checkpoint(args.checkpoint, device)
    seq_len = int(args.seq_len or ckpt.get("seq_len", 30))

    raw = load_pickle_trajectories(str(args.data))
    data, traj_labels, divider = build_test_tensors(raw, list(args.test_labels), seq_len, device)
    pred = predict(model, data, device)

    tau = data["tau"]
    rmse_total = float(np.sqrt(np.mean((pred["tau_hat"] - tau) ** 2)))
    rmse_fri = float(np.sqrt(np.mean((pred["tau_fri"] - data["tau_fri_true"]) ** 2)))
    print(f"device={device}  backend={ckpt.get('friction_backend')}  n_dof={tau.shape[1]}")
    print(f"测试轨迹: {traj_labels}  样本数: {tau.shape[0]}")
    print(f"RMSE τ_hat: {rmse_total:.5f}")
    if np.any(np.isfinite(data["tau_fri_true"])):
        print(f"RMSE τ_fri (vs τ-m-c-g): {rmse_fri:.5f}")
    else:
        print("无 m/c/g 分解，未计算 τ_fri 参考 RMSE（图中仅显示预测摩擦）")

    print_scv_params(model)
    plot_results(data, pred, traj_labels, divider, figure_out=args.figure_out, show=args.show)


if __name__ == "__main__":
    main()
