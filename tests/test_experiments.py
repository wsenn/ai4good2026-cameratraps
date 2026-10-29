"""Checks of the training options in `iwildcam/experiments.py`. CPU only, no images.

    python -m pytest tests/test_experiments.py -q
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from iwildcam.data import build_transform
from iwildcam.experiments import (ClassMixSampler, ExperimentDataset, FocusLoss,
                                  VisitLimitSampler, apply_train_weights, cv5_fold, fda_swap,
                                  train_sampler, visit_groups)
from iwildcam.models import ResNetClassifier
from iwildcam.train import train_loss
from iwildcam.train_utils import run_name
from tests.test_protocol import real_task, toy_task


def _epoch_counts(sampler, groups):
    idx = np.array(list(iter(sampler)))
    return idx, np.bincount(groups[idx], minlength=groups.max() + 1)


def test_weights_csv_is_read_and_weight_zero_removes_photos(tmp_path):
    task = toy_task()
    train = np.arange(len(task.df))
    ids = task.df["image_id"].to_numpy()
    csv = tmp_path / "w.csv"
    pd.DataFrame({"image_id": [ids[0], ids[1], ids[2]], "weight": [0.0, 0.2, 0.0]}).to_csv(
        csv, index=False)
    kept, w = apply_train_weights(task, train, csv)
    assert len(kept) == len(train) - 2
    assert ids[0] not in set(ids[kept]) and ids[2] not in set(ids[kept])
    assert w[list(kept).index(1)] == np.float32(0.2)
    assert (np.delete(w, list(kept).index(1)) == 1.0).all()


def test_visit_limit_never_exceeds_k_and_changes_each_epoch():
    task = toy_task(burst_len=25)
    train = np.arange(len(task.df))
    groups = visit_groups(task, train)
    sampler = train_sampler({"max_per_visit": 10, "seed": 0}, task, train)
    assert isinstance(sampler, VisitLimitSampler)
    seen, epochs = set(), []
    for _ in range(3):
        idx, counts = _epoch_counts(sampler, groups)
        assert counts.max() == 10 and len(idx) == len(sampler)
        assert len(set(idx.tolist())) == len(idx), "a photo twice in one epoch"
        epochs.append(set(idx.tolist()))
        seen |= epochs[-1]
    assert epochs[0] != epochs[1]
    assert seen == set(train.tolist()), "3 epochs of 10 must cover bursts of 25"


def test_control_matches_the_per_species_counts_of_the_visit_limit():
    task = toy_task(burst_len=25)
    train = np.arange(len(task.df))
    groups, y = visit_groups(task, train), task.y[train]
    limit = VisitLimitSampler(groups, 10, seed=0)
    control = train_sampler({"class_mix_of_visit_limit": 10, "seed": 0}, task, train)
    assert isinstance(control, ClassMixSampler)
    want = np.bincount(y[list(iter(limit))], minlength=task.n_classes)
    a, b = list(iter(control)), list(iter(control))
    assert (np.bincount(y[a], minlength=task.n_classes) == want).all()
    assert (np.bincount(y[b], minlength=task.n_classes) == want).all()
    assert set(a) != set(b)


def test_no_option_means_no_sampler_and_the_old_run_name():
    cfg = {"split": "official_ood", "model": "resnet18", "size": 224, "pretrained": True,
           "epochs": 10, "seed": 0}
    assert train_sampler(cfg, toy_task(), np.arange(3)) is None
    assert run_name(cfg).startswith("official_ood-resnet18-224-pre-e10-seed0-")


def test_fda_keeps_shape_and_dtype_and_takes_the_donor_mean():
    rng = np.random.default_rng(0)
    src = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
    out = fda_swap(src, src, 0.01)
    assert out.shape == (224, 224, 3) and out.dtype == np.uint8
    assert np.array_equal(out, src)
    flat = fda_swap(np.full((224, 224, 3), 50, np.uint8), np.full((224, 224, 3), 200, np.uint8),
                    0.01)
    assert (flat == 200).all()


def test_fda_donor_comes_from_another_camera():
    task = toy_task()
    train = np.arange(len(task.df))
    ds = ExperimentDataset(task, train, None, fda_beta=0.01)
    for i in range(0, len(train), 7):
        assert ds.camera[ds.donor_index(i)] != ds.camera[i]


def test_cv5_groups_are_disjoint_by_camera_and_cover_all_train_cameras():
    task = real_task()
    train = np.flatnonzero(task.df["official_ood"].to_numpy() == "train")
    val_cams, sizes = [], []
    for k in range(5):
        f = cv5_fold(task, k)
        assert len(f["test"]) == 0
        assert not set(task.camera[f["train"]]) & set(task.camera[f["val"]])
        assert set(np.concatenate([f["train"], f["val"]]).tolist()) == set(train.tolist())
        val_cams.append(set(task.camera[f["val"]]))
        sizes.append(len(f["val"]))
    assert sum(len(c) for c in val_cams) == len(set().union(*val_cams))
    assert set().union(*val_cams) == set(task.camera[train])
    assert sum(sizes) == len(train)
    assert max(sizes) - min(sizes) < 0.05 * len(train) / 5, f"unbalanced: {sizes}"


def test_focus_mask_follows_the_resize_and_the_flip_of_the_train_transform(tmp_path):
    # A 299 x 224 photo (the shape of an images_h224 frame), white on its one box only.
    W, H = 299, 224
    box = np.array([[1 / 7, 2 / 7, 2 / 7, 3 / 7]], np.float32)     # x, y, w, h
    photo = np.zeros((H, W, 3), np.uint8)
    photo[round(2 / 7 * H):round(5 / 7 * H), round(1 / 7 * W):round(3 / 7 * W)] = 255
    task = dataclasses.replace(toy_task(), image_dir=tmp_path)
    Image.fromarray(photo).save(tmp_path / task.df["file_name"][0], format="PNG")
    ds = ExperimentDataset(task, np.array([0]), build_transform(224, train=True), boxes=[box])
    unflipped = np.zeros((7, 7))
    unflipped[2:5, 1:3] = 1                  # rows 2-4, columns 1-2 of the 7 x 7 CAM grid
    torch.manual_seed(0)
    flips = set()
    for _ in range(20):
        x, _y, _d, mask = ds[0]
        assert mask.shape == (224, 224)
        flipped = bool(x[0, :, :112].mean() < x[0, :, 112:].mean())
        cells = F.adaptive_avg_pool2d(mask[None, None].float(), 7)[0, 0].numpy()
        expected = unflipped[:, ::-1] if flipped else unflipped
        assert np.abs(cells - expected).max() < 0.07
        assert ((x[0] > 0) == (mask > 0)).float().mean() > 0.98, "mask is off the box pixels"
        flips.add(flipped)
    assert flips == {False, True}


def test_focus_loss_is_0_inside_the_box_1_outside_and_has_a_gradient():
    focus = FocusLoss(ResNetClassifier(2, "resnet18", pretrained=False), lam=1.0)
    fmap = torch.zeros(4, 2, 7, 7)
    fmap[0, 0, 3, 3] = 1.0               # all heat inside the box
    fmap[1, 0, 0, 0] = 1.0               # all heat outside the box
    fmap[2, 0] = 1.0                     # heat everywhere: 1 of 49 cells inside
    fmap[3, 0] = 1.0                     # no box
    fmap.requires_grad_(True)
    focus.fmap = fmap                    # what the layer4 hook keeps
    head_weight = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    y = torch.zeros(4, dtype=torch.long)
    mask = torch.zeros(4, 224, 224, dtype=torch.uint8)
    mask[:3, 96:128, 96:128] = 1         # the box is exactly cell (3, 3)
    _inside, outside, has = focus.shares(head_weight, y, mask)
    assert torch.allclose(outside[:3], torch.tensor([0.0, 1.0, 48 / 49]))
    assert has.tolist() == [True, True, True, False]
    term = focus(head_weight, y, mask)
    assert torch.isclose(term, torch.tensor((0 + 1 + 48 / 49) / 3))
    term.backward()
    assert fmap.grad[2].abs().sum() > 0
    assert fmap.grad[3].abs().sum() == 0, "a photo without a box got a focus gradient"


def test_without_focus_lambda_the_train_loss_is_the_old_one():
    torch.manual_seed(0)
    model = ResNetClassifier(5, "resnet18", pretrained=False)
    xb, yb = torch.randn(4, 3, 64, 64), torch.tensor([0, 1, 2, 4])
    w = torch.tensor([1.0, 0.2, 1.0, 0.5], dtype=torch.float64)    # as the loader collates it
    crit, crit_each = nn.CrossEntropyLoss(), nn.CrossEntropyLoss(reduction="none")
    old = crit(model(xb), yb)                                       # the loop before option 5
    old_weighted = (crit_each(model(xb), yb) * w.float()).sum() / w.float().sum()
    loss, ce = train_loss(model, xb, yb, [], crit, crit_each, False)
    assert torch.equal(loss, old) and torch.equal(ce, old)
    loss, ce = train_loss(model, xb, yb, [w], crit, crit_each, True)
    assert torch.equal(loss, old_weighted) and torch.equal(ce, old_weighted)
