"""向后兼容：请使用 ``RobotDynamics.DeLaN.load_dataset``。"""

from .data import load_character_dataset, load_dataset

__all__ = ["load_dataset", "load_character_dataset"]
