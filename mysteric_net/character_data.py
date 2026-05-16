"""向后兼容：请使用 ``mysteric_net.delan_data.load_dataset``。"""

from .delan_data import load_character_dataset, load_dataset

__all__ = ["load_dataset", "load_character_dataset"]
