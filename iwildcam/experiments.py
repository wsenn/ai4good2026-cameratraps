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
* `focus_lambda: l` (with `focus_conf`) -- the animal-focus loss: loss = (weighted) CE +
  l * the share of the true-class CAM outside the MegaDetector animal boxes, for photos
  with a box of confidence >= `focus_conf`. The model still sees the full photo.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Sampler
from torchvision import tv_tensors

from .data import ImageDataset, Task

OPTION_KEYS = ("train_weights", "max_per_visit", "class_mix_of_visit_limit", "fda_beta",
               "focus_lambda")


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
    """The train split with option 1 (weights), option 3 (FDA) and/or option 5 (focus).

    Items are `(image, label, domain)`, plus the loss weight when `weights` is given,
    plus the box mask when `boxes` is given (last). With FDA, both photos are resized to
    `size x size` before the swap; the train transform then flips and normalises as usual.
    """

    def __init__(self, task: Task, idx: np.ndarray, transform, weights=None,
                 fda_beta: float | None = None, fda_p: float = 0.5, size: int = 224,
                 boxes=None):
        super().__init__(task, idx, transform)
        self.weights = None if weights is None else np.asarray(weights, np.float32)
        self.fda_beta, self.fda_p, self.size = fda_beta, fda_p, size
        self.camera = task.camera[idx]
        self.boxes = boxes          # option 5: the `(n, 4)` boxes of each position

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
        if self.boxes is None:
            item = (self.transform(self.load(i)), int(self.y[i]), int(self.d[i]))
            return item if self.weights is None else (*item, float(self.weights[i]))
        x, mask = transform_with_mask(self.transform, self.load(i), self.boxes[i])
        item = (x, int(self.y[i]), int(self.d[i]))
        return (*item, *(() if self.weights is None else (float(self.weights[i]),)), mask)


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
    for xb, *_ in loader:             # (x, y, domain), plus the box mask with focus_lambda
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


# --- option 5: the animal-focus loss ------------------------------------------------------

MD_JSON = Path("/zfs/ai4good/groups/camera-traps/groupB/datasets/iWildCam2020/"
               "iwildcam2020_megadetector_results.json")


def focus_boxes(task: Task, conf: float, md_json: str | Path = MD_JSON) -> np.ndarray:
    """The MegaDetector v3 animal boxes with confidence >= `conf` of every row of `task.df`,
    read by the crop code's parser (`prepare_crops.load_boxes`).

    Returns:
        One `(n, 4)` float32 array of normalised `[x, y, w, h]` per row; `n = 0` without a box.
    """
    from prepare_crops import load_boxes        # repo root, like `python -m iwildcam.train`
    boxes = load_boxes(Path(md_json), conf)
    none = np.zeros((0, 4), np.float32)
    out = np.empty(len(task.df), dtype=object)
    for i, iid in enumerate(task.df["image_id"]):
        out[i] = np.array([b for _c, b in boxes[iid]], np.float32) if iid in boxes else none
    return out


def box_mask(width: int, height: int, boxes: np.ndarray) -> np.ndarray:
    """The union of `boxes` (normalised `[x, y, w, h]`) as a `(height, width)` 0/1 uint8 mask."""
    mask = np.zeros((height, width), np.uint8)
    for x, y, w, h in boxes:
        x0, y0 = round(x * width), round(y * height)
        x1, y1 = max(round((x + w) * width), x0 + 1), max(round((y + h) * height), y0 + 1)
        mask[max(y0, 0):min(y1, height), max(x0, 0):min(x1, width)] = 1
    return mask


def transform_with_mask(transform, im: Image.Image, boxes: np.ndarray):
    """`transform` applied to `im` and to its box mask, with the same random geometry.

    The mask is built at the photo's own resolution and goes through the same v2 call, so
    every geometric step (the resize to a square, the flip) moves it as it moves the pixels;
    v2 draws the random parameters once per call. A mask is nearest-resized.

    Returns:
        The transformed image and its `(size, size)` uint8 mask, 1 on an animal box.
    """
    mask = tv_tensors.Mask(torch.from_numpy(box_mask(im.width, im.height, boxes)))
    x, mask = transform(im, mask)
    return x, mask.as_subclass(torch.Tensor)


def class_activation_map(fmap: torch.Tensor, head_weight: torch.Tensor,
                         y: torch.Tensor) -> torch.Tensor:
    """The CAM of class `y` (Zhou et al., CVPR 2016), in float32: the last conv feature map
    weighted by the head's weights of `y`, ReLU, normalised to sum 1 per photo.

    Args:
        fmap: `(B, C, h, w)`, e.g. ResNet18's `layer4` output, `(B, 512, 7, 7)` at 224 px.
        head_weight: `(n_classes, C)`, the linear head on the pooled `fmap`.
        y: `(B,)`, the class of each photo.

    Returns:
        `(B, h, w)`; all zeros for a photo whose CAM is <= 0 everywhere.
    """
    cam = (fmap.float() * head_weight.float()[y][:, :, None, None]).sum(1).relu()
    return cam / cam.sum((1, 2), keepdim=True).clamp_min(1e-12)


def cam_shares(cam: torch.Tensor, mask: torch.Tensor):
    """The share of the CAM heat inside and outside the box mask.

    Args:
        cam: `(B, h, w)` from `class_activation_map`.
        mask: `(B, H, W)` 0/1 box mask at input resolution, area-averaged to `h x w`: a
            cell half covered by a box counts half.

    Returns:
        `(inside, outside, has_box)`, each `(B,)`; `has_box` is False for an empty mask.
    """
    m = F.adaptive_avg_pool2d(mask[:, None].float(), cam.shape[-2:])[:, 0]
    return (cam * m).sum((1, 2)), (cam * (1 - m)).sum((1, 2)), (mask.flatten(1) > 0).any(1)


class FocusLoss:
    """The term of option 5 on a torchvision ResNet: the CAM share outside the box.

    A forward hook keeps the output of the last conv stage (`backbone.layer4`), so the one
    `model(x)` of a step gives both the logits and the CAM. Sums the term and the share
    inside the box over the train photos with a box, until `epoch_stats`.
    """

    def __init__(self, model: torch.nn.Module, lam: float):
        layer = getattr(getattr(model, "backbone", None), "layer4", None)
        if layer is None:
            raise ValueError("focus_lambda needs a torchvision ResNet backbone (layer4)")
        self.lam, self.fmap = float(lam), None
        layer.register_forward_hook(self._keep)
        self._sums = torch.zeros(3)

    def _keep(self, _module, _inputs, out):
        self.fmap = out

    def shares(self, head_weight: torch.Tensor, y: torch.Tensor, mask: torch.Tensor):
        """`cam_shares` of the CAM of class `y` from the last forward pass."""
        with torch.autocast(self.fmap.device.type, enabled=False):
            return cam_shares(class_activation_map(self.fmap, head_weight, y), mask)

    def __call__(self, head_weight: torch.Tensor, y: torch.Tensor,
                 mask: torch.Tensor) -> torch.Tensor:
        """The term of one batch: the mean share outside the box over the photos with a box
        (0 when none has one)."""
        inside, outside, has = self.shares(head_weight, y, mask)
        n = has.sum().float()
        out_sum = (outside * has).sum()
        self._sums = self._sums.to(n.device) + torch.stack(
            [out_sum, (inside * has).sum(), n]).detach()
        return out_sum / n.clamp_min(1)

    def epoch_stats(self) -> dict:
        """Means over the train photos with a box since the last call: `focus_loss` and
        `focus_share` (the CAM share inside the box). Resets the sums."""
        f, s, n = self._sums.tolist()
        self._sums = torch.zeros(3)
        return {"focus_loss": f / max(n, 1), "focus_share": s / max(n, 1)}


def focus_val_summary(val: dict) -> dict:
    """The mean val CAM share inside the box over the val photos with a box: all of them,
    the right and the wrong predictions. Empty without `_focus_share` (from `evaluate`)."""
    share = val.get("_focus_share")
    if share is None:
        return {}
    box, right = ~np.isnan(share), val["_pred"] == val["_true"]
    return {f"val_focus_share{tag}": float(share[m].mean())
            for tag, m in (("", box), ("_right", box & right), ("_wrong", box & ~right))
            if m.any()}
