#!/usr/bin/env python3
"""
单轴电机梯形速度激励 → DeLaN ``character_data.pickle``（用于 motor_identify_train / 合成调试）。

梯形速度（每个“半周”：加速 → 匀速平台 → 减速）::

    v
    ^     ___________
    |    /           \\
    |   /             \\
    +--+---------------+---> t
       |<-t_rise->|<-t_hold->|<-t_fall->|

默认已相对旧版（t_rise≈0.4s、t_hold≈0.2s）加大斜坡斜率、延长上底。

示例::

  python scripts/generate_motor_trapezoid.py -o data/motor_synth.pickle
  python scripts/generate_motor_trapezoid.py --t-rise 0.08 --t-hold 0.7 --v-max 2.0
  python scripts/generate_motor_trapezoid.py --accel 25 --t-hold 0.6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from RobotDynamics.DeLaN import build_pickle_dict, save_pickle


def trapezoid_pulse_velocity(
    t: np.ndarray,
    *,
    v_peak: float,
    t_rise: float,
    t_hold: float,
    t_fall: float,
    sign: float = 1.0,
    t0: float = 0.0,
) -> np.ndarray:
    """单次梯形脉冲（从 0 加速到 v_peak，平台，再减到 0）。"""
    v = np.zeros_like(t, dtype=np.float64)
    t1 = t0 + t_rise
    t2 = t1 + t_hold
    t3 = t2 + t_fall
    a = sign * abs(v_peak) / max(t_rise, 1e-9)

    mask_r = (t >= t0) & (t < t1)
    mask_h = (t >= t1) & (t < t2)
    mask_f = (t >= t2) & (t < t3)

    v[mask_r] = a * (t[mask_r] - t0)
    v[mask_h] = sign * abs(v_peak)
    v[mask_f] = sign * abs(v_peak) - a * (t[mask_f] - t2)
    return v


def trapezoid_velocity_profile(
    t: np.ndarray,
    *,
    v_max: float,
    t_rise: float,
    t_hold: float,
    t_fall: float | None = None,
    t_gap: float = 0.0,
) -> np.ndarray:
    """
    正脉冲 + 负脉冲（中间可选 t_gap 秒静止），用于摩擦/惯量辨识激励。
    """
    if t_fall is None:
        t_fall = t_rise
    v = np.zeros_like(t, dtype=np.float64)
    t0 = 0.0
    v += trapezoid_pulse_velocity(
        t, v_peak=v_max, t_rise=t_rise, t_hold=t_hold, t_fall=t_fall, sign=1.0, t0=t0
    )
    t0 += t_rise + t_hold + t_fall + t_gap
    v += trapezoid_pulse_velocity(
        t, v_peak=v_max, t_rise=t_rise, t_hold=t_hold, t_fall=t_fall, sign=-1.0, t0=t0
    )
    return v


def synthesize_trajectory(
    *,
    label: str,
    dt: float,
    v_max: float,
    t_rise: float,
    t_hold: float,
    t_fall: float,
    t_gap: float,
    J: float,
    B: float,
    C: float,
    t_pad: float = 0.0,
) -> dict:
    pulse_len = t_rise + t_hold + t_fall
    total = 2.0 * pulse_len + t_gap + 2.0 * t_pad
    n = max(int(np.ceil(total / dt)) + 1, 4)
    t = np.arange(n, dtype=np.float64) * dt
    qv = trapezoid_velocity_profile(
        t,
        v_max=v_max,
        t_rise=t_rise,
        t_hold=t_hold,
        t_fall=t_fall,
        t_gap=t_gap,
    )
    if t_pad > 0:
        qv[t < t_pad] = 0.0
        qv[t > total - t_pad] = 0.0

    qp = np.cumsum(qv) * dt
    qp = qp.reshape(-1, 1)
    qv = qv.reshape(-1, 1)
    qa = np.gradient(qv[:, 0], dt).reshape(-1, 1)
    tau = J * qa + B * qv + C * np.sign(qv)
    tau = np.where(np.abs(qv) < 1e-6, J * qa, tau)

    return {
        "label": label,
        "t": t,
        "qp": qp,
        "qv": qv,
        "qa": qa,
        "tau": tau,
        "m": np.zeros_like(qp),
        "c": np.zeros_like(qp),
        "g": np.zeros_like(qp),
        "p": np.zeros_like(qp),
        "pdot": np.zeros_like(qp),
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="生成单轴梯形速度激励 pickle")
    p.add_argument("-o", "--out", type=Path, default=ROOT / "data" / "motor_synth.pickle")
    p.add_argument("--dt", type=float, default=1e-3, help="采样周期 (s)")
    p.add_argument("--v-max", type=float, default=2.0, help="梯形平台速度幅值 |v| (rad/s)")
    p.add_argument(
        "--t-rise",
        type=float,
        default=0.10,
        help="加速段时间 (s)；腰的斜率 ≈ v_max/t_rise (rad/s²)",
    )
    p.add_argument(
        "--t-hold",
        type=float,
        default=0.60,
        help="匀速平台（上底）持续时间 (s)",
    )
    p.add_argument(
        "--t-fall",
        type=float,
        default=None,
        help="减速段时间 (s)，默认与 --t-rise 相同",
    )
    p.add_argument(
        "--accel",
        type=float,
        default=None,
        metavar="A",
        help="若指定，则 t_rise = v_max/accel，覆盖 --t-rise",
    )
    p.add_argument("--t-gap", type=float, default=0.0, help="正/负脉冲之间的静止间隔 (s)")
    p.add_argument("--t-pad", type=float, default=0.0, help="轨迹首尾置零速度时长 (s)")
    p.add_argument("--known-J", type=float, default=0.00243, help="合成力矩用惯量 (kg·m²)")
    p.add_argument("--viscous-B", type=float, default=0.00446, help="粘性摩擦 B (N·m·s/rad)")
    p.add_argument("--coulomb-C", type=float, default=0.0445, help="库仑摩擦 C (N·m)")
    p.add_argument(
        "--labels",
        nargs="*",
        default=["m1", "m2"],
        help="轨迹标签（默认两条相同参数，可改 v-max 做扫速）",
    )
    p.add_argument(
        "--v-max-list",
        nargs="*",
        type=float,
        default=None,
        help="与 --labels 等长的速度列表；未给则全部用 --v-max",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    t_rise = float(args.t_rise)
    if args.accel is not None and args.accel > 0:
        t_rise = float(args.v_max) / float(args.accel)
    t_fall = float(args.t_fall) if args.t_fall is not None else t_rise
    if t_rise <= 0 or args.t_hold <= 0 or t_fall <= 0:
        raise SystemExit("t_rise、t_hold、t_fall 须为正")

    labels = list(args.labels)
    if args.v_max_list is not None:
        if len(args.v_max_list) != len(labels):
            raise SystemExit("--v-max-list 长度须与 --labels 相同")
        v_list = [float(v) for v in args.v_max_list]
    else:
        v_list = [float(args.v_max)] * len(labels)

    trajectories = []
    for lab, vmax in zip(labels, v_list):
        trajectories.append(
            synthesize_trajectory(
                label=lab,
                dt=float(args.dt),
                v_max=vmax,
                t_rise=t_rise,
                t_hold=float(args.t_hold),
                t_fall=t_fall,
                t_gap=float(args.t_gap),
                J=float(args.known_J),
                B=float(args.viscous_B),
                C=float(args.coulomb_C),
                t_pad=float(args.t_pad),
            )
        )

    payload = build_pickle_dict(trajectories, n_dof=1, synthesize_decomposition=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_pickle(payload, args.out)

    pulse = t_rise + float(args.t_hold) + t_fall
    accel = float(args.v_max) / t_rise
    print(f"已保存: {args.out.resolve()}")
    print(
        f"  梯形: v_max={args.v_max:.4g} rad/s  t_rise=t_fall={t_rise:.4g}s  "
        f"t_hold={args.t_hold:.4g}s  斜坡加速度≈{accel:.2g} rad/s²"
    )
    print(f"  单脉冲时长={pulse:.4g}s  正+负周期≈{2*pulse + args.t_gap:.4g}s  dt={args.dt}")
    for lab, tr in zip(labels, trajectories):
        T = tr["qp"].shape[0]
        plat = int(np.sum(np.abs(tr["qv"]) > 0.95 * np.max(np.abs(tr["qv"]))))
        print(f"  轨迹 {lab}: T={T}  平台采样≈{plat} ({plat * args.dt:.3f}s)")


if __name__ == "__main__":
    main()
