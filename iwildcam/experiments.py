"""William's training options. Every one is off by default; with none set, `run_fold`
trains exactly as before.

* `train_weights: <csv>` -- a per-photo loss weight (`image_id,weight`); weight 0 drops
  the photo from training. Loss = sum(w * ce) / sum(w) per batch.
* `max_per_visit: K` -- each epoch, at most K photos of each visit (burst, species).
* `class_mix_of_visit_limit: K` -- the control: no visit limit, but each epoch has the
  per-species photo counts of a `max_per_visit: K` epoch.
* `fda_beta: b` (with `fda_p`) -- Fourier domain adaptation (Yang and Soatto, CVPR 2020):
  the low-frequency amplitude of a training photo from another camera, phase kept.
* `split: cv5` with `cv_fold: k` -- 5-fold cross-validation over the cameras of the
  `official_ood` train fold; no test. `save_val_outputs: true` keeps the val
  log-probabilities and penultimate features of the best epoch.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Sampler

from .data import ImageDataset, Task

OPTION_KEYS = ("train_weights", "max_per_visit", "class_mix_of_visit_limit", "fda_beta")


def any_option(cfg: dict) -> bool:
    """True when the config sets one of the options of this module."""
    return cfg.get("split") == "cv5" or any(cfg.get(k) is not None for k in OPTION_KEYS)


# --- option 1: per-photo loss weights ---------------------------------------------------

def apply_train_weights(task: Task, train_idx: np.ndarray, path: str | Path):
    """Read `image_id,weight` and weight the train photos; weight 0 removes a photo.

    Returns:
        The kept train indices and the weight of each one (photos not listed get 1.0).
    """
    tab = pd.read_csv(path)
    if (tab["weight"] < 0).any():
        raise ValueError(f"{path}: negative weight")
    weight = dict(zip(tab["image_id"], tab["weight"].astype(float)))
    w = task.df["image_id"].iloc[train_idx].map(weight).fillna(1.0).to_numpy(np.float32)
    keep = w > 0
    return train_idx[keep], w[keep]


# --- option 2: at most K photos per visit, and its control ------------------------------

def visit_groups(task: Task, idx: np.ndarray) -> np.ndarray:
    """The visit of each photo in `idx`: one integer per (burst_id, species)."""
    rows = task.df.iloc[idx]
    return pd.factorize(pd.MultiIndex.from_arrays([rows["burst_id"], rows["y"]]))[0]


def _members(groups: np.ndarray) -> list[np.ndarray]:
    order = np.argsort(groups, kind="stable")
    return np.split(order, np.flatnonzero(np.diff(groups[order])) + 1)


class VisitLimitSampler(Sampler):
    """Each epoch, at most `k` positions of each group, in random order.

    A group larger than `k` gives the next `k` photos of its own random order, so the
    draw changes each epoch and every photo comes before any photo repeats.
    """

    def __init__(self, groups: np.ndarray, k: int, seed: int):
        self.k, self.rng = int(k), np.random.default_rng(seed)
        self.members = _members(groups)
        self.queue = [self.rng.permutation(m) for m in self.members]
        self.pos = [0] * len(self.members)
        self.n = int(sum(min(len(m), self.k) for m in self.members))

    def __len__(self) -> int:
        return self.n

    def _take(self, g: int) -> np.ndarray:
        m = self.members[g]
        if len(m) <= self.k:
            return m
        take = list(self.queue[g][self.pos[g]:self.pos[g] + self.k])
        self.pos[g] += len(take)
        if len(take) < self.k:                      # order used up: draw a new one
            perm = self.rng.permutation(m)
            perm = np.concatenate([perm[~np.isin(perm, take)], perm[np.isin(perm, take)]])
            self.queue[g], self.pos[g] = perm, self.k - len(take)
            take += list(perm[:self.pos[g]])
        return np.asarray(take)

    def __iter__(self):
        idx = np.concatenate([self._take(g) for g in range(len(self.members))])
        return iter(self.rng.permutation(idx).tolist())


class ClassMixSampler(Sampler):
    """Each epoch, a fresh random draw with the per-species counts of a visit-limited
    epoch (`VisitLimitSampler` with the same `k`), but no limit per visit."""

    def __init__(self, groups: np.ndarray, y: np.ndarray, k: int, seed: int):
        self.rng = np.random.default_rng(seed)
        self.by_class = {c: np.flatnonzero(y == c) for c in np.unique(y)}
        self.count = {c: 0 for c in self.by_class}
        for m in _members(groups):
            self.count[y[m[0]]] += min(len(m), int(k))
        self.n = int(sum(self.count.values()))

    def __len__(self) -> int:
        return self.n

    def __iter__(self):
        idx = np.concatenate([self.rng.choice(pos, self.count[c], replace=False)
                              for c, pos in self.by_class.items()])
        return iter(self.rng.permutation(idx).tolist())


def train_sampler(cfg: dict, task: Task, train_idx: np.ndarray):
    """The sampler of option 2, or None when neither of its keys is set."""
    k, mix = cfg.get("max_per_visit"), cfg.get("class_mix_of_visit_limit")
    if k is None and mix is None:
        return None
    if k is not None and mix is not None:
        raise ValueError("set max_per_visit or class_mix_of_visit_limit, not both")
    groups = visit_groups(task, train_idx)
    if k is not None:
        return VisitLimitSampler(groups, k, cfg["seed"])
    return ClassMixSampler(groups, task.y[train_idx], mix, cfg["seed"])


# --- option 3: Fourier domain adaptation --------------------------------------------------

def fda_swap(src: np.ndarray, ref: np.ndarray, beta: float) -> np.ndarray:
    """`src` with the low-frequency amplitude of `ref`, per channel; phase of `src` kept.

    Args:
        src, ref: `(H, W, 3)` uint8 arrays of the same shape.
        beta: the half-width of the central square is `floor(beta * min(H, W))`.

    Returns:
        A `(H, W, 3)` uint8 array.
    """
    fa = np.fft.fftshift(np.fft.fft2(src.astype(np.float32), axes=(0, 1)), axes=(0, 1))
    fb = np.fft.fftshift(np.fft.fft2(ref.astype(np.float32), axes=(0, 1)), axes=(0, 1))
    amp, pha = np.abs(fa), np.angle(fa)
    h, w = src.shape[:2]
    b = int(np.floor(min(h, w) * beta))
    ch, cw = h // 2, w // 2
    win = (slice(ch - b, ch + b + 1), slice(cw - b, cw + b + 1))
    amp[win] = np.abs(fb)[win]
    out = np.fft.ifft2(np.fft.ifftshift(amp * np.exp(1j * pha), axes=(0, 1)), axes=(0, 1))
    return np.clip(np.rint(out.real), 0, 255).astype(np.uint8)


class ExperimentDataset(ImageDataset):
    """The train split with option 1 (weights) and/or option 3 (FDA).

    Items are `(image, label, domain)`, plus the loss weight when `weights` is given.
    With FDA, both photos are resized to `size x size` before the swap; the train
    transform then flips and normalises as usual.
    """

    def __init__(self, task: Task, idx: np.ndarray, transform, weights=None,
                 fda_beta: float | None = None, fda_p: float = 0.5, size: int = 224):
        super().__init__(task, idx, transform)
        self.weights = None if weights is None else np.asarray(weights, np.float32)
        self.fda_beta, self.fda_p, self.size = fda_beta, fda_p, size
        self.camera = task.camera[idx]

    def donor_index(self, i: int, tries: int = 1000) -> int:
        """A random position whose camera differs from the camera of position `i`."""
        for _ in range(tries):
            j = random.randrange(len(self.names))
            if self.camera[j] != self.camera[i]:
                return j
        raise RuntimeError("no training photo from another camera")

    def load(self, i: int) -> Image.Image:
        """The photo at position `i`, with FDA applied with probability `fda_p`."""
        im = Image.open(self.image_dir / self.names[i]).convert("RGB")
        if self.fda_beta is not None and random.random() < self.fda_p:
            ref = Image.open(self.image_dir / self.names[self.donor_index(i)]).convert("RGB")
            sz = (self.size, self.size)
            im = Image.fromarray(fda_swap(np.asarray(im.resize(sz, Image.BILINEAR)),
                                          np.asarray(ref.resize(sz, Image.BILINEAR)),
                                          self.fda_beta))
        return im

    def __getitem__(self, i):
        item = (self.transform(self.load(i)), int(self.y[i]), int(self.d[i]))
        return item if self.weights is None else (*item, float(self.weights[i]))


# --- option 4: 5-fold cross-validation over the official_ood train cameras ----------------

def cv5_fold(task: Task, k: int, n_groups: int = 5, seed: int = 0) -> dict[str, np.ndarray]:
    """Split the train cameras of `official_ood` into `n_groups` groups of about equal
    photo count (fixed `seed`, largest camera first into the smallest group); train on
    the others, validate on group `k`. No test split.
    """
    if not 0 <= k < n_groups:
        raise ValueError(f"cv_fold must be in 0..{n_groups - 1}, got {k}")
    train = np.flatnonzero(task.df["official_ood"].to_numpy() == "train")
    group = cv5_groups(task.camera[train], n_groups, seed)
    return {"train": train[group != k], "val": train[group == k],
            "test": np.array([], dtype=np.int64)}


def cv5_groups(camera: np.ndarray, n_groups: int = 5, seed: int = 0) -> np.ndarray:
    """The group of each photo, from the camera it came from."""
    cams, counts = np.unique(camera, return_counts=True)
    order = np.random.default_rng(seed).permutation(len(cams))
    order = order[np.argsort(-counts[order], kind="stable")]
    load, of_cam = np.zeros(n_groups), {}
    for i in order:
        g = int(np.argmin(load))
        of_cam[cams[i]] = g
        load[g] += counts[i]
    return np.array([of_cam[c] for c in camera])


@torch.no_grad()
def save_val_outputs(model, loader, task: Task, idx: np.ndarray, device: str, amp: bool,
                     path: Path, amp_dtype=torch.bfloat16) -> None:
    """Write the val log-probabilities, labels, ids, cameras and penultimate features."""
    model.eval()
    lps, feats = [], []
    for xb, _yb, _d in loader:
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp):
            f = model.features(xb.to(device, non_blocking=True))
            logits = model.logits(f)
        lps.append(torch.log_softmax(logits.float(), 1).cpu().numpy())
        feats.append(f.float().cpu().numpy())
    rows = task.df.iloc[idx]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, lp_val=np.concatenate(lps), feat_val=np.concatenate(feats),
                        y_val=task.y[idx], image_id_val=rows["image_id"].to_numpy(),
                        camera_val=task.camera[idx], burst_val=rows["burst_id"].to_numpy(),
                        seq_val=rows["seq_id"].to_numpy(), idx_val=idx,
                        classes=np.array(task.classes))
    print(f"val outputs: {path}", flush=True)
