#!/usr/bin/env python3
"""
在 robot.pickle（或任意 DeLaN pickle）上训练 Mysteric-Net：L-Net + 摩擦网络。

摩擦后端（Hu 等 SCV / PINN）:
  --friction-backend stribeck       纯可学习 SCV 物理模型
  --friction-backend stribeck_pinn  MLP + SCV 物理损失（论文 Eq. (6)）
  --friction-backend tcn            原 TCN（Yeo 等）
  --friction-backend fo_cascade       TCN₁→两层 tanh MLP→TCN₂（Xun 图 4 简化）
  --friction-backend fo_cascade_pinn  fo_cascade + SCV PINN（Eq. 6）

示例:
  python examples/robot_train.py \\
    --data data/robot.pickle \\
    --friction-backend stribeck_pinn \\
    --lambda-physics 0.5 \\
    -m 1

  # 三阶段：联合 → 仅 L-Net → 仅摩擦
  python examples/robot_train.py \\
    --data data/robot_fric.pickle \\
    --friction-backend fo_cascade_pinn \\
    --stage1-epochs 2000 --stage2-epochs 2000 --stage3-epochs 1000 \\
    --stage1-lr 1e-3 --stage2-lr 5e-4 --stage3-lr 1e-3 \\
    -m 1
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Literal

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from RobotDynamics.DeLaN import load_dataset, suggest_hyper, torque_loss
from RobotDynamics.FrictionModule import (
    build_mysteric_tensors,
    friction_pinn_loss,
    friction_supervised_loss,
    load_pickle_trajectories,
    mysteric_losses,
    pickle_has_mcg_decomposition,
    stack_trajectories_to_flat,
    warmstart_scv_from_samples,
)
from RobotDynamics.MystericNet import MystericNet

TrainPhase = Literal["joint", "lnet", "friction"]

PHYSICS_ONLY_FRICTION = frozenset({"stribeck"})


def _checkpoint_payload(
    model: MystericNet,
    *,
    n_dof: int,
    args: argparse.Namespace,
    l_w: int,
    l_d: int,
    epoch: int,
) -> dict:
    return {
        "state_dict": model.state_dict(),
        "epoch": epoch,
        "dof": n_dof,
        "seq_len": args.seq_len,
        "lnet_hidden": l_w,
        "lnet_layers": l_d,
        "friction_backend": args.friction_backend,
        "lambda_physics": args.lambda_physics,
        "friction_loss_weight": args.friction_loss_weight,
        "energy_loss": args.energy_loss,
        "tau_loss": args.tau_loss,
        "fri_loss": args.fri_loss,
        "smape_eps": args.smape_eps,
        "data_path": str(args.data.resolve()),
        "fo_mlp_hidden_dim": args.fo_mlp_hidden,
        "stage1_epochs": int(getattr(args, "_stage1_epochs", args.epochs)),
        "stage2_epochs": int(args.stage2_epochs),
        "stage3_epochs": int(args.stage3_epochs),
        "stage1_lr": float(
            args.stage1_lr if args.stage1_lr is not None else args.lr
        ),
        "stage2_lr": float(
            args.stage2_lr if args.stage2_lr is not None else args.lr
        ),
        "stage3_lr": float(
            args.stage3_lr if args.stage3_lr is not None else args.lr
        ),
        "training_schedule": "joint_lnet_friction",
    }


def _save_checkpoint(
    path: Path,
    model: MystericNet,
    *,
    n_dof: int,
    args: argparse.Namespace,
    l_w: int,
    l_d: int,
    epoch: int,
    interrupted: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _checkpoint_payload(
        model, n_dof=n_dof, args=args, l_w=l_w, l_d=l_d, epoch=epoch
    )
    payload["interrupted"] = interrupted
    torch.save(payload, path)
    print(f"已保存: {path.resolve()}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mysteric-Net（L-Net + 摩擦）训练")
    p.add_argument("--data", type=Path, default=ROOT / "data" / "robot.pickle")
    p.add_argument("--test-labels", nargs="*", default=["e", "v", "q"])
    p.add_argument(
        "--friction-backend",
        choices=("tcn", "fo_cascade", "fo_cascade_pinn", "stribeck", "stribeck_pinn"),
        default="stribeck_pinn",
    )
    p.add_argument(
        "--fo-mlp-hidden",
        type=int,
        default=None,
        metavar="D",
        help="fo_cascade 两层 MLP 隐层宽度，默认 max(4*n_dof, 16)",
    )
    p.add_argument("--seq-len", type=int, default=30)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lnet-width", type=int, default=None, help="默认用 suggest_hyper")
    p.add_argument("--lnet-depth", type=int, default=None)
    p.add_argument(
        "--lambda-physics",
        type=float,
        default=0.5,
        help="PINN 摩擦物理项权重 λ（Eq. 6），用于 stribeck_pinn / fo_cascade_pinn",
    )
    p.add_argument(
        "--friction-loss-weight",
        type=float,
        default=1.0,
        help="总损失: loss = l_tau + w_fri * l_fri（默认 1.0）。"
        "l_tau 与 l_fri 均用 SMAPE 时量级相近；若 --fri-loss mse 仍很大可降到 0.01~0.1。",
    )
    p.add_argument("--energy-loss", action="store_true", help="阶段 1/2 加刚体能量率 l_E（见 --no-stage2-energy）")
    p.add_argument(
        "--no-stage2-energy",
        action="store_true",
        help="两/三阶段时，阶段 2 默认仍加 l_E；加此开关则阶段 2 仅 l_τ",
    )
    p.add_argument(
        "--tau-loss",
        choices=("mse", "smape"),
        default="smape",
        help="总力矩监督（多关节推荐 smape）",
    )
    p.add_argument(
        "--fri-loss",
        choices=("mse", "smape"),
        default="smape",
        help="摩擦监督与 PINN 物理项（多关节/小力矩关节推荐 smape，与 --tau-loss 独立可选）",
    )
    p.add_argument("--smape-eps", type=float, default=1e-3)
    p.add_argument("-m", nargs="?", const=0, default=0, type=int, help="保存 checkpoint")
    p.add_argument("--save", type=Path, default=ROOT / "checkpoints/mysteric_robot.pt")
    p.add_argument("-c", nargs="?", const=1, default=1, type=int)
    p.add_argument(
        "--friction-label",
        choices=("auto", "none", "decomposition"),
        default="auto",
        help="auto=有 m/c/g 分解才监督 τ_fri；none=仅总力矩+（可选）PINN 物理项",
    )
    p.add_argument(
        "--stage1-epochs",
        type=int,
        default=None,
        metavar="N",
        help="阶段 1（L-Net + hnet 联合）epoch；与 --stage2-epochs 联用",
    )
    p.add_argument(
        "--stage2-epochs",
        type=int,
        default=0,
        metavar="N",
        help="阶段 2（冻结 hnet，仅 L-Net）epoch；0=单阶段联合训练",
    )
    p.add_argument(
        "--stage3-epochs",
        type=int,
        default=0,
        metavar="N",
        help="阶段 3（冻结 L-Net，仅 hnet 摩擦）epoch；须 stage2>0",
    )
    p.add_argument(
        "--stage1-lr",
        type=float,
        default=None,
        help="阶段 1 学习率，默认与 --lr 相同",
    )
    p.add_argument(
        "--stage2-lr",
        type=float,
        default=None,
        help="阶段 2 学习率，默认与 --lr 相同",
    )
    p.add_argument(
        "--stage3-lr",
        type=float,
        default=None,
        help="阶段 3 学习率，默认与 --lr 相同",
    )
    return p.parse_args()


def _set_module_trainable(module: torch.nn.Module, trainable: bool) -> None:
    for p in module.parameters():
        p.requires_grad = trainable


def _freeze_lnet(model: MystericNet) -> int:
    """冻结 L-Net，仅训练摩擦分支。返回冻结参数量。"""
    _set_module_trainable(model.lnet, False)
    _set_module_trainable(model.hnet, True)
    model.lnet.eval()
    model.hnet.train()
    return sum(p.numel() for p in model.lnet.parameters())


def _freeze_friction_branch(model: MystericNet) -> int:
    """冻结 hnet，仅训练 L-Net。返回冻结参数量。"""
    _set_module_trainable(model.hnet, False)
    _set_module_trainable(model.lnet, True)
    model.hnet.eval()
    model.lnet.train()
    return sum(p.numel() for p in model.hnet.parameters())


def _build_optimizer(
    model: MystericNet,
    *,
    lr: float,
    phase: TrainPhase,
) -> torch.optim.Adam:
    if phase == "friction":
        params = model.hnet.parameters()
    elif phase == "lnet":
        params = model.lnet.parameters()
    else:
        params = model.parameters()
    return torch.optim.Adam(params, lr=lr, weight_decay=1e-5, amsgrad=True)


def _resolve_stage_epochs(args: argparse.Namespace) -> tuple[int, int, int, int]:
    """返回 (stage1_epochs, stage2_epochs, stage3_epochs, total_epochs)。"""
    stage2 = max(0, int(args.stage2_epochs))
    stage3 = max(0, int(args.stage3_epochs))
    if stage3 > 0 and stage2 <= 0:
        raise SystemExit("--stage3-epochs 须与 --stage2-epochs>0 联用")
    if stage2 > 0 or stage3 > 0:
        if args.stage1_epochs is not None:
            stage1 = max(1, int(args.stage1_epochs))
        else:
            stage1 = max(1, int(args.epochs) - stage2 - stage3)
        return stage1, stage2, stage3, stage1 + stage2 + stage3
    return max(1, int(args.epochs)), 0, 0, max(1, int(args.epochs))


def _phase_at_epoch(
    epoch: int, stage1_ep: int, stage2_ep: int, stage3_ep: int
) -> TrainPhase:
    if epoch <= stage1_ep:
        return "joint"
    if stage2_ep > 0 and epoch <= stage1_ep + stage2_ep:
        return "lnet"
    if stage3_ep > 0:
        return "friction"
    return "lnet"


def _phase_label(phase: TrainPhase) -> str:
    return {"joint": "S1", "lnet": "S2-L", "friction": "S3-F"}[phase]


def _format_schedule(
    stage1_ep: int,
    stage2_ep: int,
    stage3_ep: int,
    total_ep: int,
    *,
    stage1_lr: float,
    stage2_lr: float,
    stage3_lr: float,
) -> str:
    if stage2_ep <= 0:
        return f"单阶段联合训练（共 {total_ep} epoch）"
    s2_end = stage1_ep + stage2_ep
    if stage3_ep <= 0:
        return (
            f"两阶段: 1..{stage1_ep} 联合(lr={stage1_lr:g})"
            f" → {stage1_ep + 1}..{total_ep} 仅 L-Net(lr={stage2_lr:g})"
        )
    return (
        f"三阶段: 1..{stage1_ep} 联合(lr={stage1_lr:g})"
        f" → {stage1_ep + 1}..{s2_end} 仅 L-Net(lr={stage2_lr:g})"
        f" → {s2_end + 1}..{total_ep} 仅摩擦(lr={stage3_lr:g})"
    )


def _use_energy_loss(
    args: argparse.Namespace, phase: TrainPhase, stage2_ep: int
) -> bool:
    """是否在 joint / lnet 阶段加 l_E（摩擦阶段不加）。"""
    if phase == "friction":
        return False
    if args.energy_loss:
        return True
    # 阶段 2 冻结摩擦后默认 l_E：dE_rig/dt ≈ (τ−τ_fri)^T q̇，避免 L-Net 学摩擦
    return phase == "lnet" and stage2_ep > 0 and not args.no_stage2_energy


def _effective_fri_loss(args: argparse.Namespace) -> str:
    """纯 SCV 在 SMAPE 下 pred≪target 时 l_fri≈2 且不降，改用 MSE。"""
    if args.friction_backend in PHYSICS_ONLY_FRICTION and args.fri_loss == "smape":
        return "mse"
    return args.fri_loss


def _compute_friction_loss(
    *,
    args: argparse.Namespace,
    supervise_fri: bool,
    tau_fri: torch.Tensor,
    tfb: torch.Tensor,
    tau_phys: torch.Tensor | None,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    fri_kind = _effective_fri_loss(args)
    if args.friction_backend in ("stribeck_pinn", "fo_cascade_pinn"):
        assert tau_phys is not None
        lf, _, _ = friction_pinn_loss(
            tau_fri,
            tfb,
            tau_phys,
            lambda_physics=args.lambda_physics,
            supervise_friction=supervise_fri,
            fri_loss=fri_kind,
            smape_eps=args.smape_eps,
        )
        return lf
    if supervise_fri:
        return friction_supervised_loss(
            tau_fri,
            tfb,
            fri_kind,
            smape_eps=args.smape_eps,
        )
    return torch.zeros((), device=device, dtype=dtype)


def main() -> None:
    args = _parse_args()
    if not args.data.is_file():
        raise SystemExit(f"数据不存在: {args.data}")

    cuda = bool(args.c) and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    train_data, test_data, _, _ = load_dataset(
        filename=str(args.data),
        test_label=tuple(args.test_labels),
    )
    train_labels, *_ = train_data
    test_labels, test_qp, test_qv, test_qa, *_rest = test_data
    test_tau = _rest[2]
    n_dof = test_qp.shape[1]

    raw = load_pickle_trajectories(str(args.data))
    if args.friction_label == "auto":
        supervise_fri = pickle_has_mcg_decomposition(raw)
    else:
        supervise_fri = args.friction_label == "decomposition"
    if not supervise_fri:
        print(
            "摩擦监督: 无 τ_fri 真值 → 仅用总力矩 τ_hat 监督；"
            "PINN 时另加 SCV 物理项（不监督摩擦标签）。"
        )

    train_label_set = set(train_labels)
    qp, qv, qa, tau, _tau_rigid, tau_fri = stack_trajectories_to_flat(
        raw, train_labels=train_label_set
    )
    tensors = build_mysteric_tensors(
        qp, qv, qa, tau, tau_fri, args.seq_len, device=device
    )
    qi, qdi, qddi, taui = tensors["qi"], tensors["qdi"], tensors["qddi"], tensors["taui"]
    tau_fri_t = tensors["tau_fri"]
    q_seq, qd_seq = tensors["q_seq"], tensors["qd_seq"]

    hyper = suggest_hyper(n_dof, qi.shape[0], base="delan_model")
    l_w = args.lnet_width if args.lnet_width is not None else hyper["n_width"]
    l_d = args.lnet_depth if args.lnet_depth is not None else hyper["n_depth"]

    model = MystericNet(
        dof=n_dof,
        seq_len=args.seq_len,
        lnet_hidden=l_w,
        lnet_layers=l_d,
        friction_backend=args.friction_backend,
        fo_mlp_hidden_dim=args.fo_mlp_hidden,
    ).to(device)

    eff_fri = _effective_fri_loss(args)
    if args.friction_backend == "stribeck":
        if args.fri_loss == "smape" and eff_fri == "mse":
            print("  提示: stribeck 下 SMAPE 对 l_fri 易饱和≈2，已自动改用 fri_loss=mse。")
        if hasattr(model.hnet, "scv"):
            n_init = min(4096, qdi.shape[0])
            warmstart_scv_from_samples(
                model.hnet.scv, qdi[:n_init], tau_fri_t[:n_init]
            )
            print(f"  已 warm-start SCV（k_c/k_s，N={n_init}）。")

    stage1_ep, stage2_ep, stage3_ep, total_ep = _resolve_stage_epochs(args)
    args._stage1_epochs = stage1_ep
    stage1_lr = float(args.stage1_lr if args.stage1_lr is not None else args.lr)
    stage2_lr = float(args.stage2_lr if args.stage2_lr is not None else args.lr)
    stage3_lr = float(args.stage3_lr if args.stage3_lr is not None else args.lr)

    multi_stage = stage2_ep > 0
    phase: TrainPhase = "joint"
    train_lr = stage1_lr if multi_stage else args.lr
    opt = _build_optimizer(model, lr=train_lr, phase=phase)

    N = qi.shape[0]
    B = args.batch
    w_fri = float(args.friction_loss_weight)

    print(
        f"device={device}  n_dof={n_dof}  friction={args.friction_backend}  "
        f"λ_phys={args.lambda_physics}  w_fri={w_fri}  "
        f"tau_loss={args.tau_loss}  fri_loss={eff_fri}"
        + (f" (CLI={args.fri_loss})" if eff_fri != args.fri_loss else "")
        + f"  "
        f"train N={N}  test N={test_qp.shape[0]}\n"
        f"  {_format_schedule(stage1_ep, stage2_ep, stage3_ep, total_ep, stage1_lr=stage1_lr, stage2_lr=stage2_lr, stage3_lr=stage3_lr)}"
    )
    if stage3_ep > 0 and not supervise_fri and args.friction_backend not in (
        "stribeck_pinn",
        "fo_cascade_pinn",
    ):
        print(
            "  警告: 阶段 3 需要 l_fri，但无 τ_fri 监督且非 PINN 后端。",
            flush=True,
        )

    epoch_times: list[float] = []
    last_epoch = 0

    def _interrupt_save_path() -> Path:
        if args.m:
            return args.save
        stem = args.save.stem
        if not stem.endswith("_interrupt"):
            stem = f"{stem}_interrupt"
        return args.save.with_name(stem + args.save.suffix)

    try:
        for epoch in range(1, total_ep + 1):
            last_epoch = epoch
            new_phase = _phase_at_epoch(epoch, stage1_ep, stage2_ep, stage3_ep)
            if new_phase != phase:
                phase = new_phase
                if phase == "lnet":
                    n_fr = _freeze_friction_branch(model)
                    opt = _build_optimizer(model, lr=stage2_lr, phase=phase)
                    print(
                        f"\n>>> 阶段 2 开始：已冻结 hnet（{n_fr} 参数），"
                        f"仅优化 L-Net；损失=仅 l_τ"
                        f"{' + l_E（默认，动力学一致）' if _use_energy_loss(args, 'lnet', stage2_ep) else ''}，"
                        f"lr={stage2_lr:g}\n",
                        flush=True,
                    )
                elif phase == "friction":
                    n_ln = _freeze_lnet(model)
                    opt = _build_optimizer(model, lr=stage3_lr, phase=phase)
                    print(
                        f"\n>>> 阶段 3 开始：已冻结 L-Net（{n_ln} 参数），"
                        f"仅优化 hnet；损失=仅 w_fri·l_fri，"
                        f"lr={stage3_lr:g}\n",
                        flush=True,
                    )

            t_epoch_start = time.perf_counter()
            perm = torch.randperm(N, device=device)
            loss_acc = ltau_acc = lf_acc = steps = 0
            for s in range(0, N, B):
                idx = perm[s : s + B]
                if idx.numel() < 4:
                    continue
                qb, qdb, qddb = qi[idx], qdi[idx], qddi[idx]
                taub, tfb = taui[idx], tau_fri_t[idx]
                qs, qds = q_seq[idx], qd_seq[idx]

                tau_hat, _core, tau_fri, _H, g_hat, tau_phys = model(
                    qb, qdb, qddb, qs, qds
                )

                lf = _compute_friction_loss(
                    args=args,
                    supervise_fri=supervise_fri,
                    tau_fri=tau_fri,
                    tfb=tfb,
                    tau_phys=tau_phys,
                    device=device,
                    dtype=qb.dtype,
                )
                ltau = torque_loss(tau_hat, taub, args.tau_loss, smape_eps=args.smape_eps)

                if phase == "friction":
                    loss = w_fri * lf
                elif phase == "lnet":
                    loss = ltau
                else:
                    loss = ltau + w_fri * lf

                if _use_energy_loss(args, phase, stage2_ep):
                    _, _, lE = mysteric_losses(
                        model.lnet, tau_hat, taub, tau_fri, qb, qdb, qddb, g_hat
                    )
                    loss = loss + lE

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                loss_acc += float(loss.detach())
                ltau_acc += float(ltau.detach())
                lf_acc += float(lf.detach())
                steps += 1

            epoch_sec = time.perf_counter() - t_epoch_start
            epoch_times.append(epoch_sec)

            log_ep = epoch == 1 or epoch % 50 == 0 or epoch == total_ep
            if multi_stage and epoch in (stage1_ep, stage1_ep + stage2_ep):
                log_ep = True
            if log_ep:
                with torch.no_grad():
                    n_test = min(512, test_qp.shape[0])
                    qt = torch.from_numpy(test_qp[:n_test]).float().to(device)
                    qdt = torch.from_numpy(test_qv[:n_test]).float().to(device)
                    qddt = torch.from_numpy(test_qa[:n_test]).float().to(device)
                    tt = torch.from_numpy(test_tau[:n_test]).float().to(device)
                    qs_t = qt.unsqueeze(1).expand(-1, args.seq_len, -1)
                    qds_t = qdt.unsqueeze(1).expand(-1, args.seq_len, -1)
                    th, _, _, _, _, _ = model(qt, qdt, qddt, qs_t, qds_t)
                    rmse = float(torch.sqrt(torch.mean((th - tt) ** 2)).cpu())
                win = epoch_times[-50:]
                avg_50 = sum(win) / len(win)
                n_s = max(steps, 1)
                lt_m = ltau_acc / n_s
                lf_m = lf_acc / n_s
                if phase == "friction":
                    fri_contrib = w_fri * lf_m
                elif phase == "lnet":
                    fri_contrib = 0.0
                else:
                    fri_contrib = w_fri * lf_m
                fri_note = " (S2:冻结)" if phase == "lnet" else ""
                print(
                    f"epoch {epoch:4d} [{_phase_label(phase)}]  loss={loss_acc/n_s:.5f}  "
                    f"l_tau={lt_m:.4f}  l_fri={lf_m:.4f}{fri_note}  w_fri*l_fri={fri_contrib:.4f}  "
                    f"RMSE_test≈{rmse:.4f}  "
                    f"time/epoch={epoch_sec:.2f}s  avg50={avg_50:.2f}s/epoch"
                )
                if (
                    epoch == 1
                    and phase == "joint"
                    and w_fri > 0
                    and lf_m > 10 * max(lt_m, 1e-6)
                ):
                    print(
                        "  提示: l_fri 远大于 l_tau，可减小 --friction-loss-weight，"
                        "或确认 --fri-loss smape（与 --tau-loss 一致）。"
                    )
                elif epoch == 1 and phase == "joint" and w_fri < 0.05 and lf_m > 1e-6:
                    print(
                        "  提示: w_fri 过小，摩擦项对总损失贡献弱；SMAPE 摩擦下建议 --friction-loss-weight 1.0。"
                    )

    except KeyboardInterrupt:
        print(f"\n训练被中断 (Ctrl+C)，保存 epoch={last_epoch} 的权重 …", flush=True)
        _save_checkpoint(
            _interrupt_save_path(),
            model,
            n_dof=n_dof,
            args=args,
            l_w=l_w,
            l_d=l_d,
            epoch=last_epoch,
            interrupted=True,
        )
        raise SystemExit(130) from None

    if args.m:
        _save_checkpoint(
            args.save,
            model,
            n_dof=n_dof,
            args=args,
            l_w=l_w,
            l_d=l_d,
            epoch=last_epoch,
            interrupted=False,
        )


if __name__ == "__main__":
    main()
