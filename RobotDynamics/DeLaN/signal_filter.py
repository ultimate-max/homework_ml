"""
导入数据时的低通滤波（Butterworth + filtfilt，零相位）。
"""

from __future__ import annotations

import sys
from typing import Any, Sequence

import numpy as np

# 默认滤波的轨迹字段（运动与力矩；m/c/g 多为模型分解，默认不滤）
DEFAULT_FILTER_KEYS: tuple[str, ...] = ("qp", "qv", "qa", "tau", "p", "pdot")


def sampling_rate_from_t(t: np.ndarray, *, dt_hint: float | None = None) -> float:
    """由标量 dt 或时间向量得到采样率 fs = 1/dt (Hz)。"""
    if dt_hint is not None and dt_hint > 0:
        return 1.0 / float(dt_hint)
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    if t.size < 2:
        raise ValueError("无法推断采样率：时间向量长度 < 2 且未提供 dt")
    d = np.diff(t)
    var = float(np.var(d))
    if var >= 1e-12:
        print(
            f"警告: 时间间隔非常数 (var={var:.3e})，滤波使用 fs=1/mean(diff(t))。",
            file=sys.stderr,
        )
    dt = float(np.mean(d))
    if dt <= 0:
        raise ValueError(f"无效采样周期 dt={dt}")
    return 1.0 / dt


def lowpass_filter_axis0(
    y: np.ndarray,
    *,
    fs: float,
    cutoff_hz: float,
    order: int = 4,
) -> np.ndarray:
    """
    沿第 0 维（时间）零相位低通滤波。

    ``y``: (T,) 或 (T, n_dof)。
    """
    from scipy.signal import butter, filtfilt

    y = np.asarray(y, dtype=np.float64)
    if y.size == 0:
        return y.copy()
    nyq = 0.5 * fs
    if cutoff_hz <= 0:
        raise ValueError(f"截止频率须为正，得到 {cutoff_hz}")
    if cutoff_hz >= nyq:
        raise ValueError(
            f"截止频率 {cutoff_hz} Hz >= Nyquist {nyq:.4g} Hz "
            f"(fs={fs:.4g} Hz)；请增大 dt 或降低 --filter-cutoff。"
        )
    min_len = max(3 * order, 12)
    if y.shape[0] < min_len:
        print(
            f"警告: 序列长度 {y.shape[0]} < {min_len}，跳过滤波。",
            file=sys.stderr,
        )
        return y.copy()
    wn = cutoff_hz / nyq
    b, a = butter(order, wn, btype="low")
    if y.ndim == 1:
        return np.ascontiguousarray(filtfilt(b, a, y))
    out = np.empty_like(y)
    for j in range(y.shape[1]):
        out[:, j] = filtfilt(b, a, y[:, j])
    return out


def filter_character_data(
    data: dict[str, Any],
    *,
    cutoff_hz: float,
    order: int = 4,
    keys: Sequence[str] = DEFAULT_FILTER_KEYS,
    dt_hint: float | None = None,
) -> dict[str, Any]:
    """
    对 pickle 字典中每条轨迹的指定字段做低通滤波（就地修改并返回同一 dict）。
    """
    if keys is None:
        keys = DEFAULT_FILTER_KEYS
    n_traj = len(data["labels"])
    for i in range(n_traj):
        t_i = np.asarray(data["t"][i], dtype=np.float64).reshape(-1)
        fs = sampling_rate_from_t(t_i, dt_hint=dt_hint)
        label = data["labels"][i]
        print(
            f"滤波 轨迹 {label!r}: fs={fs:.4g} Hz, fc={cutoff_hz:g} Hz, order={order}",
            file=sys.stderr,
        )
        for key in keys:
            if key not in data:
                continue
            arr = np.asarray(data[key][i], dtype=np.float64)
            if arr.size == 0 or np.all(arr == 0):
                continue
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            data[key][i] = lowpass_filter_axis0(
                arr, fs=fs, cutoff_hz=cutoff_hz, order=order
            )
    return data


def read_mat_scalar_dt(path: str | Path, *, root: str | None = None) -> float | None:
    """从 .mat（含 character_data.dt）读取标量采样周期，秒。"""
    from scipy.io import loadmat

    from .mat_convert import _resolve_mat_payload

    mat = loadmat(
        str(path), squeeze_me=True, struct_as_record=False, chars_as_strings=True
    )
    payload = _resolve_mat_payload(mat, root)
    if "dt" not in payload:
        return None
    val = np.asarray(payload["dt"], dtype=np.float64).reshape(-1)
    if val.size == 0:
        return None
    dt = float(val[0])
    return dt if dt > 0 else None
