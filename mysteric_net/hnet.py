"""
H-Net: MIMO temporal friction torque from stacked desired (or measured) joint sequences.

Paper Eq. (8):  tau_fri = TCN( q_d^{t-L:t}, qd_d^{t-L:t} )
Hyperparameters (Table I): hidden channels 8, kernel 3, sequence length L=30.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class HNetTCN(nn.Module):
    def __init__(
        self,
        dof: int,
        seq_len: int = 30,
        hidden_channels: int = 8,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.dof = dof
        self.seq_len = seq_len
        in_ch = 2 * dof
        pad = kernel_size // 2
        layers: list[nn.Module] = [
            nn.Conv1d(in_ch, hidden_channels, kernel_size, padding=pad),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=pad),
            nn.ReLU(inplace=True),
        ]
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_channels, dof)
        nn.init.xavier_normal_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, q_seq: torch.Tensor, qd_seq: torch.Tensor) -> torch.Tensor:
        """
        q_seq, qd_seq: (B, L, dof) with time increasing along dim=1, last index = current t.
        Returns tau_fri: (B, dof)
        """
        if q_seq.shape[1] != self.seq_len or qd_seq.shape[1] != self.seq_len:
            raise ValueError(f"Expected sequence length {self.seq_len}, got {q_seq.shape[1]}")
        x = torch.cat([q_seq, qd_seq], dim=-1).transpose(1, 2)
        h = self.tcn(x)
        last = h[:, :, -1]
        return self.head(last)
