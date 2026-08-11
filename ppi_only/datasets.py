"""Datasets + deterministic DataLoaders for the PPI-only pipeline."""
from __future__ import annotations

import random
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .graph import make_vector
from .image import image_tensor_from_vec


def seed_worker(worker_id: int) -> None:
    """Re-seed numpy/random per worker (fixes the legacy non-determinism gap)."""
    ws = (torch.initial_seed() + worker_id) % (2 ** 32)
    np.random.seed(ws)
    random.seed(ws)


def _build_cache(ids: List[str], rel: Dict, train_size: int, pooling: str,
                 width: int, rotulo: str) -> np.ndarray:
    """Pré-computa os vetores de todas as amostras, uma única vez.

    `make_vector` é função PURA de (pid, rel, train_size, pooling, out_size) — sem RNG. Sem cache
    ela é reexecutada a cada época: medido nesta base, 283 us/amostra em cc, ou 21 s por época,
    ~1,2 h em 200 épocas. Com num_workers=0 isso é serial e não se sobrepõe à GPU.

    Como a função é pura, cachear é bit a bit transparente (verificado com --no-vec-cache).
    NÃO se cacheia `pooling='raw'`: seriam len(ids) x train_size floats (27 TB em cc) — e é
    justamente o caso barato, porque o custo está no F.max_pool1d por amostra, não no vetor.
    """
    out = np.empty((len(ids), width), dtype=np.float32)
    t0 = time.perf_counter()
    for i, pid in enumerate(ids):
        out[i] = make_vector(pid, rel, train_size, pooling, width)
        if (i + 1) % 20000 == 0:
            print(f"      [cache {rotulo}] {i + 1:,}/{len(ids):,} "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)
    print(f"      [cache {rotulo}] {len(ids):,} vetores de {width}-d em "
          f"{time.perf_counter() - t0:.1f}s ({out.nbytes / 2 ** 20:.0f} MB)", flush=True)
    return out


class VectorDataset(Dataset):
    """MLP / CNN1D: returns the pooled train-score vector."""

    def __init__(self, ids: List[str], labels: np.ndarray, rel: Dict,
                 train_size: int, pooling: str, out_size: int, cache: bool = True):
        self.ids = list(ids)
        self.labels = labels.astype(np.float32)
        self.rel = rel
        self.train_size = int(train_size)
        self.pooling = pooling
        self.out_size = int(out_size)
        self._cache = None
        if cache and pooling != "raw":
            self._cache = _build_cache(self.ids, rel, self.train_size, pooling,
                                       self.out_size, "vetor")

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._cache is not None:
            # .copy() devolve um array próprio, como o make_vector fazia — o consumidor nunca
            # enxerga uma view do cache.
            v = self._cache[i].copy()
        else:
            v = make_vector(self.ids[i], self.rel, self.train_size, self.pooling, self.out_size)
        return torch.from_numpy(v), torch.from_numpy(self.labels[i])


class ImageDataset(Dataset):
    """ResNet50 / ConvNeXt-Tiny: outer-diff grayscale image (no dilate/enhance)."""

    def __init__(self, ids: List[str], labels: np.ndarray, rel: Dict,
                 train_size: int, pooling: str, image_size: int, cache: bool = True):
        self.ids = list(ids)
        self.labels = labels.astype(np.float32)
        self.rel = rel
        self.train_size = int(train_size)
        self.pooling = pooling
        self.image_size = int(image_size)
        self._cache = None
        if cache and pooling != "raw":
            # Cacheia o vetor POOLADO (image_size-d), não a imagem: as imagens 224x224 seriam
            # 4,7 GB por split, e a medição mostra que o custo está no pooling, não no outer-diff.
            self._cache = _build_cache(self.ids, rel, self.train_size, pooling,
                                       self.image_size, "imagem")

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._cache is not None:
            v = self._cache[i]
        else:
            # pool the vector to image_size so the outer-diff is (image_size x image_size)
            v = make_vector(self.ids[i], self.rel, self.train_size, self.pooling, self.image_size)
        t = image_tensor_from_vec(v)
        return t, torch.from_numpy(self.labels[i])


def build_loader(ds: Dataset, batch_size: int, shuffle: bool, num_workers: int,
                 seed: int) -> DataLoader:
    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    if shuffle:
        g = torch.Generator()
        g.manual_seed(seed)
        return DataLoader(ds, shuffle=True, worker_init_fn=seed_worker, generator=g, **kw)
    return DataLoader(ds, shuffle=False, worker_init_fn=seed_worker, **kw)
