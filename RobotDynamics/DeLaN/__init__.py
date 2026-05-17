"""
DeLaN 模块：拉格朗日刚体动力学网络、数据加载、训练与评估。

  from RobotDynamics.DeLaN import LNet, load_dataset, train_delan_loop, ...
"""

from .core import LagrangianLayer, LowTri, build_l_pack_index
from .data import init_env, inspect_dataset, load_dataset, load_character_dataset, validate_pickle_raw
from .eval import DeLaNEvalResult, evaluate_delan_on_test, plot_delan_performance, print_eval_report
from .hyper import suggest_hyper
from .import_data import (
    TRAIN_KEYS,
    OPTIONAL_DECOMP_KEYS,
    build_pickle_dict,
    import_mat,
    import_npz,
    inspect_pickle_dict,
    save_pickle,
)
from .lnet import LNet
from .losses import torque_loss, torque_loss_mse, torque_loss_smape
from .replay_memory import PyTorchReplayMemory
from .train_core import (
    HYPER_DELAN_MODEL,
    HYPER_EXAMPLE,
    build_lnet,
    save_delan_checkpoint,
    train_delan_loop,
    train_delan_official_loop,
)

__all__ = [
    "LNet",
    "LagrangianLayer",
    "LowTri",
    "build_l_pack_index",
    "load_dataset",
    "load_character_dataset",
    "init_env",
    "inspect_dataset",
    "validate_pickle_raw",
    "build_pickle_dict",
    "import_mat",
    "import_npz",
    "save_pickle",
    "inspect_pickle_dict",
    "TRAIN_KEYS",
    "OPTIONAL_DECOMP_KEYS",
    "suggest_hyper",
    "torque_loss",
    "torque_loss_mse",
    "torque_loss_smape",
    "HYPER_EXAMPLE",
    "HYPER_DELAN_MODEL",
    "build_lnet",
    "train_delan_loop",
    "train_delan_official_loop",
    "save_delan_checkpoint",
    "PyTorchReplayMemory",
    "evaluate_delan_on_test",
    "plot_delan_performance",
    "print_eval_report",
    "DeLaNEvalResult",
]
