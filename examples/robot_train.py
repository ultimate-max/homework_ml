#!/usr/bin/env python3
"""
在 robot.pickle（或任意 DeLaN pickle）上训练 Mysteric-Net：L-Net + 摩擦网络。

摩擦后端（Hu 等 SCV / PINN）:
  --friction-backend stribeck       纯可学习 SCV 物理模型
  --friction-backend stribeck_pinn  MLP + SCV 物理损失（论文 Eq. (6)）
  --friction-backend gms            纯可学习 GMS 物理模型（迟滞）
  --friction-backend gms_pinn       MLP + GMS 物理损失
  --friction-backend tcn            原 TCN（Yeo 等）
  --friction-backend fo_cascade       TCN₁→两层 tanh MLP→TCN₂（Xun 图 4 简化）
  --friction-backend fo_cascade_pinn  fo_cascade + SCV PINN（Eq. 6）

示例:
  python examples/robot_train.py \\
    --data data/robot_fric.pickle \\
    --friction-backend stribeck \\
    --stage1-epochs 2000 --stage2-epochs 1000 \\
    --stage2-lr 5e-4
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
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
    warmstart_gms_from_samples,
    warmstart_scv_from_samples,
)
from RobotDynamics.MystericNet import MystericNet

PHYSICS_ONLY_FRICTION = frozenset({"stribeck", "gms"})
SCV_FRICTION_BACKENDS = frozenset({"stribeck", "stribeck_pinn", "fo_cascade_pinn"})
TrainPhase = Literal["joint", "lnet", "friction"]


def _net_checkpoint_path(
    friction_backend: str,
    *,
    interrupt: bool = False,
    epoch: int | None = None,
    checkpoints_dir: Path | None = None,
) -> Path:
    stem = f"{friction_backend}_net"
    if interrupt:
        stem = f"{stem}_interrupt"
    elif epoch is not None:
        stem = f"{stem}_epoch{int(epoch):05d}"
    base = checkpoints_dir if checkpoints_dir is not None else ROOT / "checkpoints"
    return base / f"{stem}.pt"


def _loss_csv_path(friction_backend: str, *, checkpoints_dir: Path | None = None) -> Path:
    base = checkpoints_dir if checkpoints_dir is not None else ROOT / "checkpoints"
    return base / f"{friction_backend}_loss.csv"


def _write_loss_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ("epoch", "phase", "loss", "l_tau", "l_fri", "l_E")
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"已写入训练 loss CSV: {path.resolve()}  ({len(rows)} 行)")


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
        "energy_loss_weight": args.energy_loss_weight,
        "tau_loss": args.tau_loss,
        "fri_loss": args.fri_loss,
        "smape_eps": args.smape_eps,
        "data_path": str(args.data.resolve()),
        "fo_mlp_hidden_dim": args.fo_mlp_hidden,
        "fo_tcn_layers": args.fo_tcn_layers,
        "gms_n_blocks": args.gms_n_blocks,
        "gms_n_elements": args.gms_n_blocks,
        "gms_dt": getattr(args, "_gms_dt_resolved", args.gms_dt),
        "lr": float(args.lr),
        "stage1_epochs": int(getattr(args, "_stage1_epochs", args.epochs)),
        "stage2_epochs": int(args.stage2_epochs),
        "stage2_lr": float(
            args.stage2_lr if args.stage2_lr is not None else args.lr
        ),
        "stage3_epochs": int(args.stage3_epochs),
        "stage3_lr": float(
            args.stage3_lr if args.stage3_lr is not None else args.lr
        ),
        "training_schedule": (
            "joint_lnet_friction"
            if int(args.stage2_epochs) > 0 or int(args.stage3_epochs) > 0
            else "joint"
        ),
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


def _load_mysteric_checkpoint(path: Path, device: torch.device) -> tuple[MystericNet, dict]:
    """与 robot_evaluate.load_mysteric_checkpoint 相同逻辑（避免重复维护）。"""
    ev_path = Path(__file__).resolve().parent / "robot_evaluate.py"
    spec = importlib.util.spec_from_file_location("robot_evaluate", ev_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 {ev_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load_mysteric_checkpoint(path, device)


def _resume_start_epoch(ckpt: dict, path: Path) -> int:
    if "epoch" in ckpt:
        return int(ckpt["epoch"]) + 1
    from_name = path.stem
    if "_epoch" in from_name:
        try:
            return int(from_name.rsplit("_epoch", 1)[-1]) + 1
        except ValueError:
            pass
    return 1


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mysteric-Net（L-Net + 摩擦）联合训练")
    p.add_argument("--data", type=Path, default=ROOT / "data" / "robot.pickle")
    p.add_argument("--test-labels", nargs="*", default=["e", "v", "q"])
    p.add_argument(
        "--friction-backend",
        choices=(
            "tcn",
            "fo_cascade",
            "fo_cascade_pinn",
            "stribeck",
            "stribeck_pinn",
            "gms",
            "gms_pinn",
        ),
        default="stribeck_pinn",
    )
    p.add_argument(
        "--fo-mlp-hidden",
        type=int,
        default=None,
        metavar="D",
        help="fo_cascade 两层 MLP 隐层宽度，默认 max(4*n_dof, 16)",
    )
    p.add_argument(
        "--fo-tcn-layers",
        type=int,
        default=None,
        metavar="N",
        help="fo_cascade / fo_cascade_pinn 中 TCN₁ 与 TCN₂ 层数；"
        "默认 fo_cascade=2、fo_cascade_pinn=3",
    )
    p.add_argument(
        "--gms-blocks",
        "--gms-n-elements",
        dest="gms_n_blocks",
        type=int,
        default=3,
        metavar="N",
        help="每关节并联 GMS 块数 N（gms / gms_pinn；τ_fri = Σ F_i + σ₁v）",
    )
    p.add_argument(
        "--gms-dt",
        type=float,
        default=None,
        metavar="S",
        help="GMS 积分步长 (s)。gms/gms_pinn 下默认从 pickle 的 t 推断 mean(diff(t))",
    )
    p.add_argument("--seq-len", type=int, default=30)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="阶段 1 摩擦子网 hnet 学习率；阶段 2 为 L-Net 学习率（若未设 --stage2-lr）",
    )
    p.add_argument(
        "--lnet-lr",
        type=float,
        default=None,
        metavar="LR",
        help="阶段 1 联合训练时 L-Net 学习率（默认与 --lr 相同）。"
        "例: --lr 1e-3 --lnet-lr 1e-4 让摩擦(hnet)快、惯量(lnet)慢",
    )
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
    p.add_argument(
        "--energy-loss",
        action="store_true",
        help="加能量率 l_E：dE_rig/dt ≈ (τ−τ_fri)^T q̇，对 L-Net 与 hnet 联合反传",
    )
    p.add_argument(
        "--energy-loss-weight",
        type=float,
        default=1.0,
        help="l_E 权重：loss += w_E * l_E（默认 1.0）",
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
    p.add_argument(
        "-m",
        nargs="?",
        const=0,
        default=0,
        type=int,
        help="保留兼容；训练结束与中断均会保存 checkpoint",
    )
    p.add_argument(
        "--save",
        type=Path,
        default=None,
        help="checkpoint 路径，默认 checkpoints/{--friction-backend}_net.pt",
    )
    p.add_argument(
        "--resume",
        type=Path,
        default=None,
        metavar="CKPT",
        help="从已保存 checkpoint 继续训练（读取 state_dict 与 epoch）",
    )
    p.add_argument("-c", nargs="?", const=1, default=1, type=int)
    p.add_argument(
        "--loss-log-epoch-interval",
        type=int,
        default=50,
        metavar="N",
        help="每 N 个 epoch 将 loss/l_tau/l_fri/l_E 追加写入 CSV（默认 50）",
    )
    p.add_argument(
        "--loss-log-dir",
        type=Path,
        default=None,
        help="loss CSV 目录，默认 checkpoints/",
    )
    p.add_argument(
        "--no-loss-log",
        action="store_true",
        help="不写入训练 loss CSV",
    )
    p.add_argument(
        "--checkpoint-save-interval",
        type=int,
        default=500,
        metavar="N",
        help="每 N 个 epoch 另存一份 checkpoint（默认 500，0=关闭）",
    )
    p.add_argument(
        "--no-periodic-checkpoint",
        action="store_true",
        help="关闭按 epoch 间隔保存（仍会在结束/中断时保存）",
    )
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
        help="阶段 2（冻结 hnet，L-Net 拟合 τ−τ_fri）epoch；0=仅单阶段联合训练",
    )
    p.add_argument(
        "--stage2-lr",
        type=float,
        default=None,
        help="阶段 2 学习率，默认与 --lr 相同",
    )
    p.add_argument(
        "--stage3-epochs",
        type=int,
        default=0,
        metavar="N",
        help="阶段 3（冻结 L-Net，仅优化 hnet；但 τ_core 仍参与 loss）epoch；0=关闭",
    )
    p.add_argument(
        "--stage3-lr",
        type=float,
        default=None,
        help="阶段 3 学习率，默认与 --lr 相同",
    )
    return p.parse_args()


def _resolve_stage_epochs(args: argparse.Namespace) -> tuple[int, int, int, int]:
    """返回 (stage1_epochs, stage2_epochs, stage3_epochs, total_epochs)。"""
    stage2 = max(0, int(args.stage2_epochs))
    stage3 = max(0, int(args.stage3_epochs))
    if stage2 <= 0 and stage3 <= 0:
        total = max(1, int(args.epochs))
        return total, 0, 0, total
    if args.stage1_epochs is not None:
        stage1 = max(1, int(args.stage1_epochs))
    else:
        stage1 = max(1, int(args.epochs) - stage2 - stage3)
    return stage1, stage2, stage3, stage1 + stage2 + stage3


def _phase_at_epoch(epoch: int, stage1_ep: int, stage2_ep: int) -> TrainPhase:
    if epoch <= stage1_ep:
        return "joint"
    if epoch <= stage1_ep + stage2_ep:
        return "lnet"
    return "friction"


def _phase_label(phase: TrainPhase) -> str:
    return {"joint": "S1", "lnet": "S2-L", "friction": "S3-H"}[phase]


def _set_module_trainable(module: torch.nn.Module, trainable: bool) -> None:
    for p in module.parameters():
        p.requires_grad = trainable


def _freeze_friction_branch(model: MystericNet) -> int:
    """冻结 hnet，仅训练 L-Net。返回冻结参数量。"""
    _set_module_trainable(model.hnet, False)
    _set_module_trainable(model.lnet, True)
    model.hnet.eval()
    model.lnet.train()
    return sum(p.numel() for p in model.hnet.parameters())


def _freeze_lnet_branch(model: MystericNet) -> int:
    """冻结 L-Net，仅训练 hnet。返回冻结参数量。"""
    _set_module_trainable(model.lnet, False)
    _set_module_trainable(model.hnet, True)
    model.lnet.eval()
    model.hnet.train()
    return sum(p.numel() for p in model.lnet.parameters())


def _build_optimizer(
    model: MystericNet,
    *,
    lr: float,
    phase: TrainPhase,
    lnet_lr: float | None = None,
) -> torch.optim.Adam:
    """阶段 1 联合：lr 用于 hnet（摩擦），lnet_lr 用于 L-Net（可选更小）。"""
    wd = 1e-5
    if phase == "lnet":
        return torch.optim.Adam(
            (p for p in model.lnet.parameters() if p.requires_grad),
            lr=lr,
            weight_decay=wd,
            amsgrad=True,
        )
    if phase == "friction":
        return torch.optim.Adam(
            (p for p in model.hnet.parameters() if p.requires_grad),
            lr=lr,
            weight_decay=wd,
            amsgrad=True,
        )
    lnet_lr_eff = float(lr if lnet_lr is None else lnet_lr)
    hnet_params = [p for p in model.hnet.parameters() if p.requires_grad]
    lnet_params = [p for p in model.lnet.parameters() if p.requires_grad]
    if lnet_lr_eff == lr or not (hnet_params and lnet_params):
        return torch.optim.Adam(
            hnet_params + lnet_params,
            lr=lr,
            weight_decay=wd,
            amsgrad=True,
        )
    return torch.optim.Adam(
        [
            {"params": hnet_params, "lr": lr},
            {"params": lnet_params, "lr": lnet_lr_eff},
        ],
        weight_decay=wd,
        amsgrad=True,
    )


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
    if args.friction_backend in ("stribeck_pinn", "fo_cascade_pinn", "gms_pinn"):
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
    save_is_default = args.save is None
    if save_is_default:
        args.save = _net_checkpoint_path(args.friction_backend)
    loss_csv_path = _loss_csv_path(
        args.friction_backend,
        checkpoints_dir=args.loss_log_dir,
    )
    if not args.data.is_file():
        raise SystemExit(f"数据不存在: {args.data}")

    cuda = bool(args.c) and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    train_data, test_data, _, dt_mean = load_dataset(
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

    if args.friction_backend in ("gms", "gms_pinn") and args.gms_n_blocks < 1:
        raise SystemExit("--gms-blocks 须 ≥ 1")

    gms_dt = args.gms_dt
    if args.friction_backend in ("gms", "gms_pinn"):
        if gms_dt is None:
            gms_dt = float(dt_mean)
            print(f"  GMS: blocks={args.gms_n_blocks}  dt={gms_dt:g} s（由数据 t 推断）")
        else:
            print(f"  GMS: blocks={args.gms_n_blocks}  dt={gms_dt:g} s（CLI 指定）")
        args._gms_dt_resolved = gms_dt

    resume_ckpt: dict | None = None
    start_epoch = 1
    if args.resume is not None:
        if not args.resume.is_file():
            raise SystemExit(f"续训 checkpoint 不存在: {args.resume}")
        model, resume_ckpt = _load_mysteric_checkpoint(args.resume, device)
        if int(model.dof) != n_dof:
            raise SystemExit(
                f"checkpoint n_dof={model.dof} 与数据 n_dof={n_dof} 不一致: {args.resume}"
            )
        ckpt_backend = resume_ckpt.get("friction_backend")
        if ckpt_backend is not None and str(ckpt_backend) != args.friction_backend:
            print(
                f"  警告: CLI friction-backend={args.friction_backend!r} 与 "
                f"checkpoint={ckpt_backend!r} 不一致，以 CLI 为准但权重可能不匹配。"
            )
        start_epoch = _resume_start_epoch(resume_ckpt, args.resume)
        l_w = int(resume_ckpt.get("lnet_hidden", l_w))
        l_d = int(resume_ckpt.get("lnet_layers", l_d))
        if resume_ckpt.get("seq_len") is not None:
            ckpt_seq = int(resume_ckpt["seq_len"])
            if ckpt_seq != args.seq_len:
                print(
                    f"  警告: checkpoint seq_len={ckpt_seq}，CLI seq_len={args.seq_len}，"
                    "以 CLI 为准。"
                )
        print(
            f"续训: 已加载 {args.resume.resolve()}，"
            f"从 epoch {start_epoch} 训练到 {int(args.epochs)}"
        )
    else:
        model = MystericNet(
            dof=n_dof,
            seq_len=args.seq_len,
            lnet_hidden=l_w,
            lnet_layers=l_d,
            friction_backend=args.friction_backend,
            fo_mlp_hidden_dim=args.fo_mlp_hidden,
            fo_tcn_layers=args.fo_tcn_layers,
            gms_n_blocks=args.gms_n_blocks,
            gms_dt=gms_dt if args.friction_backend in ("gms", "gms_pinn") else 0.001,
        ).to(device)

        eff_fri = _effective_fri_loss(args)
        if args.friction_backend in ("gms", "gms_pinn"):
            if args.fri_loss == "smape" and eff_fri == "mse":
                print("  提示: gms 下 SMAPE 对 l_fri 易饱和≈2，已自动改用 fri_loss=mse。")
            if hasattr(model.hnet, "gms"):
                n_init = min(4096, qdi.shape[0])
                warmstart_gms_from_samples(
                    model.hnet.gms, qdi[:n_init], tau_fri_t[:n_init]
                )
                print(f"  已 warm-start GMS 极限面 v_a（N={n_init}）。")

        if args.friction_backend == "stribeck":
            if args.fri_loss == "smape" and eff_fri == "mse":
                print("  提示: stribeck 下 SMAPE 对 l_fri 易饱和≈2，已自动改用 fri_loss=mse。")

        if args.friction_backend in SCV_FRICTION_BACKENDS and hasattr(model.hnet, "scv"):
            n_init = min(4096, qdi.shape[0])
            n_joints = warmstart_scv_from_samples(
                model.hnet.scv, qdi[:n_init], tau_fri_t[:n_init]
            )
            c = model.hnet.scv.positive_coefficients()
            print(
                f"  已 warm-start SCV（{n_joints}/{n_dof} 关节，N={n_init}）："
                f"k_c≈{float(c['k_c'].median()):.3g}  "
                f"k_s≈{float(c['k_s'].median()):.3g}  "
                f"k_v≈{float(c['k_v'].median()):.3g}"
            )

    stage1_ep, stage2_ep, stage3_ep, total_ep = _resolve_stage_epochs(args)
    args._stage1_epochs = stage1_ep
    if start_epoch > total_ep:
        raise SystemExit(
            f"续训起点 epoch={start_epoch} 已超过 --epochs={total_ep}，"
            "请增大 --epochs 或换更早的 checkpoint。"
        )
    stage1_hnet_lr = float(args.lr)
    stage1_lnet_lr = (
        float(args.lnet_lr) if args.lnet_lr is not None else stage1_hnet_lr
    )
    stage2_lr = float(args.stage2_lr if args.stage2_lr is not None else args.lr)
    stage3_lr = float(args.stage3_lr if args.stage3_lr is not None else args.lr)
    if start_epoch > stage1_ep + stage2_ep and stage3_ep > 0:
        phase = "friction"
        _freeze_lnet_branch(model)
        opt = _build_optimizer(model, lr=stage3_lr, phase=phase)
        if args.resume is not None:
            print(
                f"  续训处于阶段 3 [S3-H]：lnet 已冻结，仅优化 hnet；"
                "τ_core 仍通过 τ_hat 参与总损失。"
            )
    elif start_epoch > stage1_ep and stage2_ep > 0:
        phase = "lnet"
        _freeze_friction_branch(model)
        opt = _build_optimizer(model, lr=stage2_lr, phase=phase)
        if args.resume is not None:
            print(
                f"  续训处于阶段 2 [S2-L]：hnet 已冻结，L-Net 拟合 τ_meas−τ_fri。"
            )
    else:
        phase = "joint"
        opt = _build_optimizer(
            model,
            lr=stage1_hnet_lr,
            phase=phase,
            lnet_lr=stage1_lnet_lr if args.lnet_lr is not None else None,
        )

    eff_fri = _effective_fri_loss(args)

    N = qi.shape[0]
    B = args.batch
    w_fri = float(args.friction_loss_weight)
    w_E = float(args.energy_loss_weight)
    use_energy = bool(args.energy_loss)
    loss_interval = max(1, int(args.loss_log_epoch_interval))
    ckpt_interval = max(0, int(args.checkpoint_save_interval))
    periodic_ckpt = ckpt_interval > 0 and not args.no_periodic_checkpoint
    checkpoints_dir = args.save.parent

    print(
        f"device={device}  n_dof={n_dof}  friction={args.friction_backend}  "
        + (
            f"gms_blocks={args.gms_n_blocks}  "
            if args.friction_backend in ("gms", "gms_pinn")
            else ""
        )
        + f"λ_phys={args.lambda_physics}  w_fri={w_fri}  "
        + f"w_E={w_E if use_energy else 0:g}  "
        + f"tau_loss={args.tau_loss}  fri_loss={eff_fri}"
        + (f" (CLI={args.fri_loss})" if eff_fri != args.fri_loss else "")
        + f"  train N={N}  test N={test_qp.shape[0]}\n"
        + (
            f"  分阶段: 1..{stage1_ep} 联合(hnet lr={stage1_hnet_lr:g}"
            + (
                f", lnet lr={stage1_lnet_lr:g})"
                if args.lnet_lr is not None
                else ")"
            )
            + (
                f" → {stage1_ep + 1}..{stage1_ep + stage2_ep} 仅 L-Net"
                f"(lr={stage2_lr:g}, τ_target=τ−τ_fri)"
                if stage2_ep > 0
                else ""
            )
            + (
                f" → {stage1_ep + stage2_ep + 1}..{total_ep} 仅 hnet"
                f"(lr={stage3_lr:g}, lnet 冻结但 τ_core 参与 loss)"
                if stage3_ep > 0
                else ""
            )
            + "\n"
            if stage2_ep > 0 or stage3_ep > 0
            else (
                f"  单阶段联合训练，共 {total_ep} epoch，hnet lr={stage1_hnet_lr:g}"
                + (
                    f", lnet lr={stage1_lnet_lr:g}\n"
                    if args.lnet_lr is not None
                    else f"\n"
                )
            )
        )
        + (
            f"  续训: epoch {start_epoch}..{total_ep}\n"
            if args.resume is not None
            else ""
        )
        + f"  checkpoint → {args.save.resolve()}\n"
        + (
            f"  周期保存 → {checkpoints_dir}/"
            f"{args.friction_backend}_net_epochNNNNN.pt（每 {ckpt_interval} epoch）\n"
            if periodic_ckpt
            else "  周期保存: 已关闭\n"
        )
        + (
            f"  loss CSV → {loss_csv_path.resolve()}（每 {loss_interval} epoch）\n"
            if not args.no_loss_log
            else "  loss CSV: 已关闭（--no-loss-log）\n"
        )
    )

    epoch_times: list[float] = []
    last_epoch = 0
    loss_rows: list[dict[str, float | int]] = []

    def _interrupt_save_path() -> Path:
        if save_is_default:
            return _net_checkpoint_path(args.friction_backend, interrupt=True)
        stem = args.save.stem
        if not stem.endswith("_interrupt"):
            stem = f"{stem}_interrupt"
        return args.save.with_name(stem + args.save.suffix)

    def _flush_loss_csv() -> None:
        if not args.no_loss_log:
            _write_loss_csv(loss_csv_path, loss_rows)

    try:
        for epoch in range(start_epoch, total_ep + 1):
            last_epoch = epoch
            new_phase = _phase_at_epoch(epoch, stage1_ep, stage2_ep)
            if new_phase != phase:
                phase = new_phase
                if phase == "lnet":
                    n_fr = _freeze_friction_branch(model)
                    opt = _build_optimizer(model, lr=stage2_lr, phase=phase)
                    print(
                        f"\n>>> 阶段 2 开始：已冻结 hnet（{n_fr} 参数）；"
                        f"L-Net 损失为 loss(τ_core, τ_meas−τ_fri)，不含摩擦项；"
                        f"lr={stage2_lr:g}\n",
                        flush=True,
                    )
                elif phase == "friction":
                    n_ln = _freeze_lnet_branch(model)
                    opt = _build_optimizer(model, lr=stage3_lr, phase=phase)
                    print(
                        f"\n>>> 阶段 3 开始：已冻结 lnet（{n_ln} 参数）；"
                        "仅优化 hnet，但 τ_core 仍通过 τ_hat 参与 loss；"
                        f"lr={stage3_lr:g}\n",
                        flush=True,
                    )

            t_epoch_start = time.perf_counter()
            perm = torch.randperm(N, device=device)
            loss_acc = ltau_acc = lf_acc = lE_acc = steps = 0
            for s in range(0, N, B):
                idx = perm[s : s + B]
                if idx.numel() < 4:
                    continue
                qb, qdb, qddb = qi[idx], qdi[idx], qddi[idx]
                taub, tfb = taui[idx], tau_fri_t[idx]
                qs, qds = q_seq[idx], qd_seq[idx]

                tau_hat, tau_core, tau_fri, _H, g_hat, tau_phys = model(
                    qb, qdb, qddb, qs, qds
                )

                if phase in ("joint", "friction"):
                    lf = _compute_friction_loss(
                        args=args,
                        supervise_fri=supervise_fri,
                        tau_fri=tau_fri,
                        tfb=tfb,
                        tau_phys=tau_phys,
                        device=device,
                        dtype=qb.dtype,
                    )
                    ltau = torque_loss(
                        tau_hat, taub, args.tau_loss, smape_eps=args.smape_eps
                    )
                    loss = ltau + w_fri * lf
                else:
                    # 阶段 2：hnet 冻结，L-Net 拟合去摩擦后的力矩目标
                    tau_fri_fixed = tau_fri.detach()
                    tau_rigid_target = taub - tau_fri_fixed
                    lf = torch.zeros((), device=device, dtype=qb.dtype)
                    ltau = torque_loss(
                        tau_core,
                        tau_rigid_target,
                        args.tau_loss,
                        smape_eps=args.smape_eps,
                    )
                    loss = ltau

                if use_energy and phase in ("joint", "friction"):
                    _, _, lE = mysteric_losses(
                        model.lnet, tau_hat, taub, tau_fri, qb, qdb, qddb, g_hat
                    )
                    loss = loss + w_E * lE
                elif use_energy and phase == "lnet":
                    _, _, lE = mysteric_losses(
                        model.lnet,
                        tau_hat,
                        taub,
                        tau_fri.detach(),
                        qb,
                        qdb,
                        qddb,
                        g_hat,
                    )
                    loss = loss + w_E * lE
                else:
                    with torch.no_grad():
                        _, _, lE = mysteric_losses(
                            model.lnet,
                            tau_hat,
                            taub,
                            tau_fri,
                            qb,
                            qdb,
                            qddb,
                            g_hat,
                        )

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                loss_acc += float(loss.detach())
                ltau_acc += float(ltau.detach())
                lf_acc += float(lf.detach())
                lE_acc += float(lE.detach())
                steps += 1

            epoch_sec = time.perf_counter() - t_epoch_start
            epoch_times.append(epoch_sec)

            log_ep = (
                epoch == 1
                or epoch % loss_interval == 0
                or epoch == total_ep
                or (stage2_ep > 0 and epoch == stage1_ep)
            )
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
                lE_m = lE_acc / n_s
                loss_m = loss_acc / n_s
                if not args.no_loss_log:
                    loss_rows.append(
                        {
                            "epoch": epoch,
                            "phase": _phase_label(phase),
                            "loss": loss_m,
                            "l_tau": lt_m,
                            "l_fri": lf_m,
                            "l_E": lE_m,
                        }
                    )
                if use_energy:
                    energy_str = f"  l_E={lE_m:.4f}  w_E*l_E={w_E * lE_m:.4f}"
                else:
                    energy_str = (
                        f"  l_E={lE_m:.4f} (监控,未计入loss; 训练加 --energy-loss)"
                    )
                if phase == "lnet":
                    fri_str = f"  l_fri={lf_m:.4f} (S2:冻结)"
                elif phase == "friction":
                    fri_str = f"  l_fri={lf_m:.4f}  w_fri*l_fri={w_fri * lf_m:.4f} (S3:训hnet)"
                else:
                    fri_str = f"  l_fri={lf_m:.4f}  w_fri*l_fri={w_fri * lf_m:.4f}"
                print(
                    f"epoch {epoch:4d} [{_phase_label(phase)}]  loss={loss_m:.5f}  "
                    f"l_tau={lt_m:.4f}{fri_str}{energy_str}  "
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

            if periodic_ckpt and epoch % ckpt_interval == 0:
                periodic_path = _net_checkpoint_path(
                    args.friction_backend,
                    epoch=epoch,
                    checkpoints_dir=checkpoints_dir,
                )
                _save_checkpoint(
                    periodic_path,
                    model,
                    n_dof=n_dof,
                    args=args,
                    l_w=l_w,
                    l_d=l_d,
                    epoch=epoch,
                    interrupted=False,
                )

    except KeyboardInterrupt:
        print(f"\n训练被中断 (Ctrl+C)，保存 epoch={last_epoch} 的权重 …", flush=True)
        _flush_loss_csv()
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

    _flush_loss_csv()

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
