"""
经验回放缓冲区（来自 deep_lagrangian_networks.replay_memory，训练 DeLaN 用）。
"""

from __future__ import annotations

import warnings

import numpy as np
import torch


def warning_on_one_line(message, category, filename, lineno, file=None, line=None):
    return "%s:%s: %s: %s\n" % (filename, lineno, category.__name__, message)


warnings.formatwarning = warning_on_one_line


class ReplayMemory:
    def __init__(self, maximum_number_of_samples, minibatch_size, dim):
        self._max_samples = maximum_number_of_samples
        self._minibatch_size = minibatch_size
        self._dim = dim
        self._data_idx = 0
        self._data_n = 0
        self._sampler_idx = 0
        self._order = None
        self._data = []
        for i in range(len(dim)):
            self._data.append(np.empty((self._max_samples,) + dim[i]))

    def __iter__(self):
        self._order = np.random.permutation(self._data_n)
        self._sampler_idx = 0
        return self

    def __next__(self):
        if self._order is None or self._sampler_idx >= self._order.size:
            raise StopIteration()

        tmp = self._sampler_idx
        self._sampler_idx += self._minibatch_size
        self._sampler_idx = min(self._sampler_idx, self._order.size)

        batch_idx = self._order[tmp : self._sampler_idx]
        if batch_idx.size < self._minibatch_size:
            raise StopIteration()

        return [x[batch_idx] for x in self._data]

    def add_samples(self, data):
        assert len(data) == len(self._data)
        add_idx = self._data_idx + np.arange(data[0].shape[0])
        add_idx = np.mod(add_idx, self._max_samples)

        for i in range(len(data)):
            self._data[i][add_idx] = data[i][:]

        self._data_idx = np.mod(add_idx[-1] + 1, self._max_samples)
        self._data_n = min(self._data_n + data[0].shape[0], self._max_samples)
        del data


class PyTorchReplayMemory(ReplayMemory):
    def __init__(self, max_samples, minibatch_size, dim, cuda):
        super().__init__(max_samples, minibatch_size, dim)
        self._cuda = cuda
        for i in range(len(dim)):
            self._data[i] = torch.empty((self._max_samples,) + dim[i])
            if self._cuda:
                self._data[i] = self._data[i].cuda()

    def add_samples(self, data):
        tmp_data = []
        for i, x in enumerate(data):
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x).float()
            tmp_data.append(x.type_as(self._data[i]))
        super().add_samples(tmp_data)
