"""
RobotDynamics 顶层包（DeLaN_Stribeck 工程）。

子包:
  - ``DeLaN``           : 刚体拉格朗日动力学（L-Net、训练、数据）
  - ``FrictionModule``  : 摩擦子网络（TCN、Stribeck、PINN）
  - ``MystericNet``     : DeLaN + 摩擦联合模型
"""

from . import DeLaN, FrictionModule, MystericNet
from .MystericNet import FrictionBackend, MystericNet

__all__ = ["DeLaN", "FrictionModule", "MystericNet", "FrictionBackend"]
