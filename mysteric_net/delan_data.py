"""
官方 DeLaN 数据与训练环境工具（对齐 deep_lagrangian_networks.utils，无 JAX 依赖）。

  from mysteric_net.delan_data import load_dataset, init_env
"""

from __future__ import annotations

import sys
from typing import Any, Sequence, Tuple

import dill as pickle
import numpy as np
import torch


def init_env(args) -> tuple[int, bool, bool, bool, bool]:
    """
    与官方 ``deep_lagrangian_networks.utils.init_env`` 相同。

    期望 ``args`` 含属性 ``s, i, c, r, l, m``（各为长度 1 的列表，与 example_DeLaN 一致）。
    也接受 ``argparse.Namespace`` 上同名整型/布尔字段（本仓库 ``delan_train.py``）。
    """
    np.set_printoptions(
        suppress=True,
        precision=2,
        linewidth=500,
        formatter={"float_kind": lambda x: "{0:+08.2f}".format(x)},
    )

    def _one(x, default=0):
        if isinstance(x, (list, tuple)):
            return x[0]
        return x if x is not None else default

    seed = int(_one(getattr(args, "s", 42), 42))
    cuda_id = int(_one(getattr(args, "i", 0), 0))
    cuda_flag = bool(_one(getattr(args, "c", 1), 1))
    render = bool(_one(getattr(args, "r", 0), 0))
    load_model = bool(_one(getattr(args, "l", 0), 0))
    save_model = bool(_one(getattr(args, "m", 0), 0))

    cuda_flag = cuda_flag and torch.cuda.is_available()

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch.cuda.device_count() > 1 and cuda_flag:
        assert cuda_id < torch.cuda.device_count()
        torch.cuda.set_device(cuda_id)

    return seed, cuda_flag, render, load_model, save_model


def load_dataset(
    n_characters: int = 3,
    filename: str = "data/character_data.pickle",
    test_label: Sequence[str] = ("e", "q", "v"),
) -> Tuple[
    tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    list[int],
    float,
]:
    """
    加载并划分训练/测试集（与官方 ``load_dataset`` 一致）。

    返回:
        train_data: (labels, qp, qv, qa, p, pdot, tau)
        test_data:  (labels, qp, qv, qa, p, pdot, tau, m, c, g)
        divider, dt_mean
    """
    del n_characters  # 官方保留参数，按 test_label 划分

    with open(filename, "rb") as f:
        data: dict[str, Any] = pickle.load(f)

    n_dof = int(np.asarray(data["qp"][0]).shape[1])
    for i in range(len(data["labels"])):
        d_i = int(np.asarray(data["qp"][i]).shape[1])
        if d_i != n_dof:
            raise ValueError(f"数据集中各轨迹 n_dof 不一致: 轨迹0为{n_dof}，轨迹{i}为{d_i}")

    labels_all = data["labels"]
    test_idx: list[int] = []
    missing: list[str] = []
    for x in test_label:
        if x in labels_all:
            ix = labels_all.index(x)
            if ix not in test_idx:
                test_idx.append(ix)
        else:
            missing.append(x)
    if missing:
        print(
            "load_dataset 警告: 以下标签不在数据中，已从测试划分中忽略: "
            f"{missing}",
            file=sys.stderr,
        )

    use_full_overlap = False
    if not test_idx:
        print(
            "load_dataset 警告: 无任何请求的测试标签存在于数据中；"
            "将全部轨迹同时划入训练集与测试集。",
            file=sys.stderr,
        )
        test_idx = list(range(len(labels_all)))
        use_full_overlap = True

    dt = np.concatenate([data["t"][idx][1:] - data["t"][idx][:-1] for idx in test_idx])
    dt_mean, dt_var = float(np.mean(dt)), float(np.var(dt))
    assert dt_var < 1.0e-12, f"时间步长非常数 (var={dt_var:.3e})"

    train_labels: list[str] = []
    test_labels: list[str] = []
    train_qp = np.zeros((0, n_dof))
    train_qv = np.zeros((0, n_dof))
    train_qa = np.zeros((0, n_dof))
    train_tau = np.zeros((0, n_dof))
    train_p = np.zeros((0, n_dof))
    train_pd = np.zeros((0, n_dof))

    test_qp = np.zeros((0, n_dof))
    test_qv = np.zeros((0, n_dof))
    test_qa = np.zeros((0, n_dof))
    test_tau = np.zeros((0, n_dof))
    test_m = np.zeros((0, n_dof))
    test_c = np.zeros((0, n_dof))
    test_g = np.zeros((0, n_dof))
    test_p = np.zeros((0, n_dof))
    test_pd = np.zeros((0, n_dof))
    divider = [0]

    for i in range(len(labels_all)):
        in_test = i in test_idx
        in_train = use_full_overlap or (i not in test_idx)

        if in_test:
            test_labels.append(labels_all[i])
            test_qp = np.vstack((test_qp, data["qp"][i]))
            test_qv = np.vstack((test_qv, data["qv"][i]))
            test_qa = np.vstack((test_qa, data["qa"][i]))
            test_tau = np.vstack((test_tau, data["tau"][i]))
            test_m = np.vstack((test_m, data["m"][i]))
            test_c = np.vstack((test_c, data["c"][i]))
            test_g = np.vstack((test_g, data["g"][i]))
            test_p = np.vstack((test_p, data["p"][i]))
            test_pd = np.vstack((test_pd, data["pdot"][i]))
            divider.append(test_qp.shape[0])

        if in_train:
            train_labels.append(labels_all[i])
            train_qp = np.vstack((train_qp, data["qp"][i]))
            train_qv = np.vstack((train_qv, data["qv"][i]))
            train_qa = np.vstack((train_qa, data["qa"][i]))
            train_tau = np.vstack((train_tau, data["tau"][i]))
            train_p = np.vstack((train_p, data["p"][i]))
            train_pd = np.vstack((train_pd, data["pdot"][i]))

    if len(train_labels) == 0 and test_qp.shape[0] > 0:
        print(
            "load_dataset 警告: 训练集为空，已复制测试集用于训练。",
            file=sys.stderr,
        )
        train_labels = list(test_labels)
        train_qp = np.array(test_qp, copy=True)
        train_qv = np.array(test_qv, copy=True)
        train_qa = np.array(test_qa, copy=True)
        train_tau = np.array(test_tau, copy=True)
        train_p = np.array(test_p, copy=True)
        train_pd = np.array(test_pd, copy=True)

    train_data = (train_labels, train_qp, train_qv, train_qa, train_p, train_pd, train_tau)
    test_data = (
        test_labels,
        test_qp,
        test_qv,
        test_qa,
        test_p,
        test_pd,
        test_tau,
        test_m,
        test_c,
        test_g,
    )
    return train_data, test_data, divider, dt_mean

# 向后兼容旧名
load_character_dataset = load_dataset
