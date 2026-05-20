#!/usr/bin/env python3
"""
单电机（n_dof=1）物理参数辨识：DeLaN L-Net 学等效惯量 H≈J，fo_cascade_pinn 学摩擦。

假设（水平轴或重力已补偿）：科氏 c≈0、重力 g≈0；总力矩 τ = J·q̈ + τ_fri。

数据：DeLaN ``.pickle`` 或单轨迹 ``.npz``（键 qp, qv, qa, tau；形状 T×1 或 T,）。

示例:
  python examples/motor_identify_train.py --data data/motor.npz -m 1
  python examples/motor_identify_train.py --data data/motor.pickle --known-J 0.0023 -m 1
  python examples/motor_identify_train.py --data data/motor.pickle --known-J 0.00243 \\
      --lnet-mass-eps 1e-4 -m 1
  python examples/motor_identify_train.py --inspect --data data/motor.pickle
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import dill as pickle
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from RobotDynamics.DeLaN import (
    import_npz,
    load_dataset,
    save_pickle,
    torque_loss,
    validate_pickle_raw,
)
from RobotDynamics.FrictionModule import (
    build_mysteric_tensors,
    friction_pinn_loss,
    stack_trajectories_to_flat,
)
from RobotDynamics.MystericNet import MystericNet

FRICTION_BACKEND = "fo_cascade_pinn"


@dataclass
class MotorIdentifyReport:
    j_median: float
    j_mean: float
    j_std: float
    c_rms: float
    g_rms: float
    tau_rmse: float
    tau_rigid_rmse: float
    tau_fri_rmse: float
    scv_k_v: float
    scv_k_c: float
    scv_k_s: float


def _load_raw(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".npz":
        return import_npz(path, n_dof=1)
    with open(path, "rb") as f:
        return pickle.load(f)


def _assert_single_dof(raw: dict[str, Any]) -> int:
    n_dof, _ = validate_pickle_raw(raw)
    if n_dof != 1:
        raise SystemExit(
            f"本脚本仅支持单电机 n_dof=1，当前数据 n_dof={n_dof}。"
            "多轴请用 examples/robot_train.py。"
        )
    return n_dof


@torch.no_grad()
def _motor_dynamics_report(
    model: MystericNet,
    qp: np.ndarray,
    qv: np.ndarray,
    qa: np.ndarray,
    tau: np.ndarray,
    seq_len: int,
    device: torch.device,
) -> MotorIdentifyReport:
    model.eval()
    n = qp.shape[0]
    q = torch.from_numpy(qp).float().to(device)
    qd = torch.from_numpy(qv).float().to(device)
    qdd = torch.from_numpy(qa).float().to(device)
    tt = torch.from_numpy(tau).float().to(device)
    qs = q.unsqueeze(1).expand(-1, seq_len, -1)
    qds = qd.unsqueeze(1).expand(-1, seq_len, -1)

    H = model.lnet.H_hat_from_q(q)
    j = H[:, 0, 0].cpu().numpy()

    z_qd = torch.zeros_like(qd)
    z_qdd = torch.zeros_like(qdd)
    g_vec = model.lnet.inv_dyn(q, z_qd, z_qdd)
    c_vec = model.lnet.inv_dyn(q, qd, z_qdd) - g_vec
    c_rms = float(torch.sqrt(torch.mean(c_vec**2)).cpu())
    g_rms = float(torch.sqrt(torch.mean(g_vec**2)).cpu())

    tau_hat, tau_core, tau_fri, _, _, _ = model(q, qd, qdd, qs, qds)
    tau_rmse = float(torch.sqrt(torch.mean((tau_hat - tt) ** 2)).cpu())
    tau_rigid_rmse = float(torch.sqrt(torch.mean((tau_core - tt) ** 2)).cpu())
    tau_fri_rmse = float(torch.sqrt(torch.mean(tau_fri**2)).cpu())

    scv = model.hnet.scv
    softplus = torch.nn.functional.softplus

    def _scv(name: str) -> float:
        p = getattr(scv, f"log_{name}")
        v = softplus(p[0]).item()
        return max(v, 0.5) if name == "alpha" else v

    return MotorIdentifyReport(
        j_median=float(np.median(j)),
        j_mean=float(np.mean(j)),
        j_std=float(np.std(j)),
        c_rms=c_rms,
        g_rms=g_rms,
        tau_rmse=tau_rmse,
        tau_rigid_rmse=tau_rigid_rmse,
        tau_fri_rmse=tau_fri_rmse,
        scv_k_v=_scv("k_v"),
        scv_k_c=_scv("k_c"),
        scv_k_s=_scv("k_s"),
    )


def _print_report(rep: MotorIdentifyReport, *, known_j: float | None) -> None:
    print("\n========== 单电机辨识结果 ==========")
    print(f"  等效惯量 J (H_00 中位数): {rep.j_median:.6e} kg·m²")
    print(f"  H_00  mean ± std:         {rep.j_mean:.6e} ± {rep.j_std:.6e}")
    if known_j is not None and known_j > 0:
        err = abs(rep.j_median - known_j) / known_j * 100.0
        print(f"  参考 J (--known-J):       {known_j:.6e}  相对误差 {err:.2f}%")
    print(f"  残余 c RMS:               {rep.c_rms:.6e} N·m  (理想≈0)")
    print(f"  残余 g RMS:               {rep.g_rms:.6e} N·m  (理想≈0)")
    print(f"  RMSE τ_total:             {rep.tau_rmse:.6e} N·m")
    print(f"  RMSE τ_rigid (仅刚体):    {rep.tau_rigid_rmse:.6e} N·m")
    print(f"  RMSE |τ_fri|:             {rep.tau_fri_rmse:.6e} N·m")
    print(
        f"  SCV (Hu): k_v={rep.scv_k_v:.4f}  k_c={rep.scv_k_c:.4f}  k_s={rep.scv_k_s:.4f}"
    )
    print("====================================\n")


def _checkpoint_payload(
    model: MystericNet,
    *,
    args: argparse.Namespace,
    l_w: int,
    l_d: int,
    epoch: int,
    report: MotorIdentifyReport | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "state_dict": model.state_dict(),
        "motor_identify": True,
        "epoch": epoch,
        "dof": 1,
        "seq_len": args.seq_len,
        "lnet_hidden": l_w,
        "lnet_layers": l_d,
        "friction_backend": FRICTION_BACKEND,
        "lambda_physics": args.lambda_physics,
        "friction_loss_weight": args.friction_loss_weight,
        "tau_loss": args.tau_loss,
        "fri_loss": args.fri_loss,
        "smape_eps": args.smape_eps,
        "data_path": str(args.data.resolve()),
        "fo_mlp_hidden_layers": args.fo_mlp_hidden_layers,
        "mass_diag_eps": args.lnet_mass_eps,
        "lnet_numerical_H_ridge": args.lnet_mass_eps,
    }
    if report is not None:
        payload["J_est"] = report.j_median
        payload["J_est_mean"] = report.j_mean
        payload["J_est_std"] = report.j_std
        payload["c_rms"] = report.c_rms
        payload["g_rms"] = report.g_rms
    if args.known_J is not None:
        payload["known_J"] = float(args.known_J)
    return payload


def _save_checkpoint(
    path: Path,
    model: MystericNet,
    *,
    args: argparse.Namespace,
    l_w: int,
    l_d: int,
    epoch: int,
    report: MotorIdentifyReport | None,
    interrupted: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _checkpoint_payload(
        model, args=args, l_w=l_w, l_d=l_d, epoch=epoch, report=report
    )
    payload["interrupted"] = interrupted
    torch.save(payload, path)
    print(f"已保存: {path.resolve()}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="单电机惯量+摩擦辨识（DeLaN + fo_cascade_pinn，n_dof=1）"
    )
    p.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data" / "motor.pickle",
        help=".pickle 或 .npz（qp,qv,qa,tau）",
    )
    p.add_argument(
        "--inspect",
        action="store_true",
        help="仅检查数据 n_dof=1 与轨迹数后退出",
    )
    p.add_argument("--test-labels", nargs="*", default=None)
    p.add_argument(
        "--test-frac",
        type=float,
        default=0.2,
        help="无匹配 test-label 时，最后该比例轨迹作测试（默认 0.2）",
    )
    p.add_argument(
        "--known-J",
        type=float,
        default=None,
        metavar="J",
        help="已知转子/反射惯量 (kg·m²)，用于打印相对误差",
    )
    p.add_argument("--seq-len", type=int, default=20)
    p.add_argument("--epochs", type=int, default=800)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--lnet-width",
        type=int,
        default=32,
        help="单轴常惯量可用较小 L-Net（默认 32）",
    )
    p.add_argument("--lnet-depth", type=int, default=2)
    p.add_argument(
        "--lnet-mass-eps",
        type=float,
        default=1e-2,
        metavar="EPS",
        help="L-Net 质量对角初值 b 与 H 数值脊 εI（默认 1e-2；"
        "J 卡在 0.01 时可试 1e-4~1e-3，需与真实 J 量级匹配）",
    )
    p.add_argument("--fo-mlp-hidden-layers", type=int, default=2)
    p.add_argument("--lambda-physics", type=float, default=0.5)
    p.add_argument("--friction-loss-weight", type=float, default=1.0)
    p.add_argument("--tau-loss", choices=("mse", "smape"), default="smape")
    p.add_argument("--fri-loss", choices=("mse", "smape"), default="smape")
    p.add_argument("--smape-eps", type=float, default=1e-3)
    p.add_argument("-m", nargs="?", const=0, default=0, type=int, help="保存 checkpoint")
    p.add_argument(
        "--save",
        type=Path,
        default=ROOT / "checkpoints" / "motor_identify.pt",
    )
    p.add_argument("-c", nargs="?", const=1, default=1, type=int)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.data.is_file():
        raise SystemExit(f"数据不存在: {args.data}")

    raw = _load_raw(args.data)
    _assert_single_dof(raw)

    if args.inspect:
        labels = raw["labels"]
        T0 = int(np.asarray(raw["qp"][0]).shape[0])
        print(f"文件: {args.data}")
        print(f"  轨迹数: {len(labels)}  标签: {labels}")
        print(f"  n_dof=1  首条长度 T={T0}")
        return

    cleanup_temp = False
    if args.data.suffix.lower() == ".pickle":
        dataset_file = args.data
    else:
        dataset_file = Path(tempfile.mkstemp(suffix=".motor_identify.pickle")[1])
        save_pickle(raw, dataset_file)
        cleanup_temp = True

    test_labels = tuple(args.test_labels) if args.test_labels else ("e", "v", "q")
    try:
        train_data, test_data, _, _ = load_dataset(
            filename=str(dataset_file),
            test_label=test_labels,
            test_frac=args.test_frac,
        )
    finally:
        if cleanup_temp:
            dataset_file.unlink(missing_ok=True)

    train_labels, *_ = train_data
    _test_labels_out, test_qp, test_qv, test_qa, *_rest = test_data
    test_tau = _rest[2]

    train_label_set = set(train_labels)
    qp, qv, qa, tau, _tau_rigid, tau_fri = stack_trajectories_to_flat(
        raw, train_labels=train_label_set
    )

    cuda = bool(args.c) and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    tensors = build_mysteric_tensors(
        qp, qv, qa, tau, tau_fri, args.seq_len, device=device
    )
    qi = tensors["qi"]
    qdi = tensors["qdi"]
    qddi = tensors["qddi"]
    taui = tensors["taui"]
    tau_fri_t = tensors["tau_fri"]
    q_seq = tensors["q_seq"]
    qd_seq = tensors["qd_seq"]

    l_w, l_d = args.lnet_width, args.lnet_depth
    mass_eps = float(args.lnet_mass_eps)
    if mass_eps <= 0:
        raise SystemExit(f"--lnet-mass-eps 须为正数，当前 {mass_eps}")
    model = MystericNet(
        dof=1,
        seq_len=args.seq_len,
        lnet_hidden=l_w,
        lnet_layers=l_d,
        friction_backend=FRICTION_BACKEND,
        fo_mlp_hidden_layers=args.fo_mlp_hidden_layers,
        mass_diag_eps=mass_eps,
        lnet_numerical_H_ridge=mass_eps,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5, amsgrad=True)
    N = qi.shape[0]
    B = args.batch
    w_fri = float(args.friction_loss_weight)

    print(
        "单电机辨识训练  model=DeLaN+fo_cascade_pinn  supervise_τ_fri=False\n"
        f"  device={device}  train_N={N}  test_N={test_qp.shape[0]}  "
        f"λ_phys={args.lambda_physics}  w_fri={w_fri}  lnet_mass_eps={mass_eps:.2e}\n"
        f"  提示: 激励需覆盖足够 |q̈| 与 |q̇|；J_med 若恒等于 mass_eps 可再减小 --lnet-mass-eps。"
    )

    last_epoch = 0
    last_report: MotorIdentifyReport | None = None

    def _interrupt_path() -> Path:
        if args.m:
            return args.save
        stem = args.save.stem
        if not stem.endswith("_interrupt"):
            stem = f"{stem}_interrupt"
        return args.save.with_name(stem + args.save.suffix)

    try:
        for epoch in range(1, args.epochs + 1):
            last_epoch = epoch
            t0 = time.perf_counter()
            perm = torch.randperm(N, device=device)
            loss_acc = ltau_acc = lf_acc = steps = 0

            for s in range(0, N, B):
                idx = perm[s : s + B]
                if idx.numel() < 4:
                    continue
                qb, qdb, qddb = qi[idx], qdi[idx], qddi[idx]
                taub, tfb = taui[idx], tau_fri_t[idx]
                qs, qds = q_seq[idx], qd_seq[idx]

                tau_hat, _core, tau_fri, _H, _g, tau_phys = model(
                    qb, qdb, qddb, qs, qds
                )
                assert tau_phys is not None
                lf, _, _ = friction_pinn_loss(
                    tau_fri,
                    tfb,
                    tau_phys,
                    lambda_physics=args.lambda_physics,
                    supervise_friction=False,
                    fri_loss=args.fri_loss,
                    smape_eps=args.smape_eps,
                )
                ltau = torque_loss(tau_hat, taub, args.tau_loss, smape_eps=args.smape_eps)
                loss = ltau + w_fri * lf

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                loss_acc += float(loss.detach())
                ltau_acc += float(ltau.detach())
                lf_acc += float(lf.detach())
                steps += 1

            if epoch == 1 or epoch % 50 == 0 or epoch == args.epochs:
                n_test = min(512, test_qp.shape[0])
                rep = _motor_dynamics_report(
                    model,
                    test_qp[:n_test],
                    test_qv[:n_test],
                    test_qa[:n_test],
                    test_tau[:n_test],
                    args.seq_len,
                    device,
                )
                last_report = rep
                n_s = max(steps, 1)
                print(
                    f"epoch {epoch:4d}  loss={loss_acc/n_s:.5f}  l_tau={ltau_acc/n_s:.4f}  "
                    f"l_fri={lf_acc/n_s:.4f}  RMSE_τ={rep.tau_rmse:.4f}  "
                    f"J_med={rep.j_median:.6e}  c_rms={rep.c_rms:.4f}  g_rms={rep.g_rms:.4f}  "
                    f"t={time.perf_counter()-t0:.2f}s"
                )

    except KeyboardInterrupt:
        print(f"\n训练被中断，保存 epoch={last_epoch} …", flush=True)
        if last_report is None:
            n_test = min(512, test_qp.shape[0])
            last_report = _motor_dynamics_report(
                model,
                test_qp[:n_test],
                test_qv[:n_test],
                test_qa[:n_test],
                test_tau[:n_test],
                args.seq_len,
                device,
            )
        _save_checkpoint(
            _interrupt_path(),
            model,
            args=args,
            l_w=l_w,
            l_d=l_d,
            epoch=last_epoch,
            report=last_report,
            interrupted=True,
        )
        _print_report(last_report, known_j=args.known_J)
        raise SystemExit(130) from None

    n_test = test_qp.shape[0]
    final_report = _motor_dynamics_report(
        model,
        test_qp,
        test_qv,
        test_qa,
        test_tau,
        args.seq_len,
        device,
    )
    _print_report(final_report, known_j=args.known_J)

    if args.m:
        _save_checkpoint(
            args.save,
            model,
            args=args,
            l_w=l_w,
            l_d=l_d,
            epoch=last_epoch,
            report=final_report,
            interrupted=False,
        )


if __name__ == "__main__":
    main()
