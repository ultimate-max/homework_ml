"""
将机械臂轨迹导入为 DeLaN ``character_data.pickle`` 格式（任意 n_dof）。

支持:
  - 内存中 ``build_pickle_dict`` / ``save_pickle``
  - 单条或多条 ``.npz``（见 ``import_npz``）
  - MATLAB ``.mat``（需 scipy，逻辑与官方 mat_to_character_pickle 一致）
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Sequence

import dill as pickle
import numpy as np

TRAIN_KEYS = ("qp", "qv", "qa", "tau")
OPTIONAL_DECOMP_KEYS = ("m", "c", "g")
OTHER_KEYS = ("p", "pdot")


def _as_label(x: Any, index: int) -> str:
    if isinstance(x, str):
        return x
    if isinstance(x, (bytes, np.bytes_)):
        return x.decode() if isinstance(x, bytes) else str(x)
    return f"traj_{index:04d}"


def _as_2d(a: np.ndarray, name: str, n_dof: int | None) -> np.ndarray:
    arr = np.asarray(a, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name} 须为 (T,) 或 (T, n_dof)，得到 shape={arr.shape}")
    if n_dof is not None and arr.shape[1] != n_dof:
        raise ValueError(f"{name} 列数 {arr.shape[1]} != n_dof={n_dof}")
    return arr


def build_pickle_dict(
    trajectories: Sequence[dict[str, Any]],
    *,
    n_dof: int | None = None,
    synthesize_decomposition: bool = True,
) -> dict[str, Any]:
    """
    由多条轨迹字典构建官方 pickle 结构。

    每条轨迹至少包含:
      t, qp, qv, qa, tau   shape (T,) 或 (T, n_dof)

    可选: m, c, g, p, pdot；若缺失且 synthesize_decomposition=True，
    则 m,c,g 置零（仅力矩训练/评估可用，分解指标无意义）。
    """
    if not trajectories:
        raise ValueError("trajectories 为空")

    if n_dof is None:
        n_dof = int(_as_2d(trajectories[0]["qp"], "qp", None).shape[1])

    out: dict[str, Any] = {
        "labels": [],
        "t": [],
        "qp": [],
        "qv": [],
        "qa": [],
        "tau": [],
        "m": [],
        "c": [],
        "g": [],
        "p": [],
        "pdot": [],
    }

    for i, tr in enumerate(trajectories):
        out["labels"].append(_as_label(tr.get("label", tr.get("name", i)), i))
        qp = _as_2d(tr["qp"], "qp", n_dof)
        T = qp.shape[0]
        t = np.asarray(tr.get("t", np.arange(T, dtype=np.float64) * 0.01), dtype=np.float64).reshape(-1)
        if t.shape[0] != T:
            raise ValueError(f"轨迹 {i}: t 长度 {t.shape[0]} != T={T}")

        qv = _as_2d(tr["qv"], "qv", n_dof)
        qa = _as_2d(tr["qa"], "qa", n_dof)
        tau = _as_2d(tr["tau"], "tau", n_dof)

        out["t"].append(t)
        out["qp"].append(qp)
        out["qv"].append(qv)
        out["qa"].append(qa)
        out["tau"].append(tau)

        for key in OTHER_KEYS:
            if key in tr and tr[key] is not None:
                out[key].append(_as_2d(tr[key], key, n_dof))
            else:
                out[key].append(np.zeros((T, n_dof), dtype=np.float64))

        for key in OPTIONAL_DECOMP_KEYS:
            if key in tr and tr[key] is not None:
                out[key].append(_as_2d(tr[key], key, n_dof))
            elif synthesize_decomposition:
                out[key].append(np.zeros((T, n_dof), dtype=np.float64))
            else:
                raise KeyError(f"轨迹 {i} 缺少 {key}，且 synthesize_decomposition=False")

    return out


def save_pickle(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)


def import_npz(
    path: str | Path,
    *,
    traj_key: str | None = None,
    n_dof: int | None = None,
) -> dict[str, Any]:
    """
  导入 ``.npz``。

  形式 1 — 单轨迹扁平键: qp, qv, qa, tau, t, (m,c,g 可选)
  形式 2 — 多轨迹: 键名 ``traj_0_qp`` 或传入 traj_key 前缀列表
    """
    path = Path(path)
    z = np.load(path, allow_pickle=True)
    keys = list(z.files)

    def _get(name: str) -> np.ndarray:
        if name not in z.files:
            raise KeyError(f"{path} 中无键 {name!r}，现有: {keys}")
        return z[name]

    if all(k in keys for k in ("qp", "qv", "qa", "tau")):
        tr = {k: _get(k) for k in ("qp", "qv", "qa", "tau")}
        if "t" in keys:
            tr["t"] = _get("t")
        for k in OPTIONAL_DECOMP_KEYS + OTHER_KEYS:
            if k in keys:
                tr[k] = _get(k)
        return build_pickle_dict([tr], n_dof=n_dof)

    prefixes: list[str] = []
    if traj_key is not None:
        prefixes = [traj_key.rstrip("_")]
    else:
        for k in keys:
            if k.endswith("_qp") and k != "qp":
                prefixes.append(k[: -len("_qp")])
        prefixes = sorted(set(prefixes))

    if not prefixes:
        raise ValueError(f"无法从 {path} 推断轨迹，需要 qp,qv,qa,tau 或 <prefix>_qp 形式")

    trajectories = []
    for p in prefixes:
        tr = {
            "label": p,
            "qp": _get(f"{p}_qp"),
            "qv": _get(f"{p}_qv"),
            "qa": _get(f"{p}_qa"),
            "tau": _get(f"{p}_tau"),
        }
        tk = f"{p}_t"
        if tk in keys:
            tr["t"] = _get(tk)
        for k in OPTIONAL_DECOMP_KEYS + OTHER_KEYS:
            kk = f"{p}_{k}"
            if kk in keys:
                tr[k] = _get(kk)
        trajectories.append(tr)

    return build_pickle_dict(trajectories, n_dof=n_dof)


def import_mat(
    path: str | Path,
    *,
    n_dof: int | None = None,
    transpose: bool = False,
    root: str | None = None,
) -> dict[str, Any]:
    """MATLAB .mat → pickle 字典（依赖 scipy）。"""
    from scipy.io import loadmat

    # 复用官方转换逻辑（拷贝精简版，避免跨仓库路径）
    from .mat_convert import mat_to_pickle_dict

    mat = loadmat(
        str(path), squeeze_me=True, struct_as_record=False, chars_as_strings=True
    )
    return mat_to_pickle_dict(mat, n_dof=n_dof, transpose=transpose, root=root)


def inspect_pickle_dict(data: dict[str, Any]) -> str:
    n_traj = len(data["labels"])
    n_dof = int(np.asarray(data["qp"][0]).shape[1])
    lengths = [int(np.asarray(data["qp"][i]).shape[0]) for i in range(n_traj)]
    has_mcg = all(
        np.any(np.asarray(data[k][i]) != 0)
        for k in OPTIONAL_DECOMP_KEYS
        for i in range(n_traj)
    )
    lines = [
        f"轨迹数: {n_traj}",
        f"n_dof: {n_dof}",
        f"总样本: {sum(lengths)}",
        f"每条 T: min={min(lengths)}, max={max(lengths)}",
        f"labels: {data['labels'][:8]}{'...' if n_traj > 8 else ''}",
        f"含非零 m/c/g 分解真值: {has_mcg}",
        f"Cholesky 参数个数 m={n_dof * (n_dof + 1) // 2}",
    ]
    return "\n".join(lines)
