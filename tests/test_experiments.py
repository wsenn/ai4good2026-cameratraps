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
                                  VisitLimitSampler, apply_train_weights, burst_macro_f1,
                                  cv5_fold, fda_swap, train_sampler, visit_groups)
from iwildcam.models import ResNetClassifier
from iwildcam.train import evaluate, run_fold, train_loss
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


def test_burst_score_averages_the_probabilities_and_takes_the_most_common_label():
    # 8 photos of 3 bursts, in mixed order. Burst a: its photos disagree (argmax 1, 0, 0);
    # the mean probability picks 1, so a is wrong although most of its photos are right.
    # Burst b: two species (labels 2, 1, 1); its label is 1 and the mean picks 1.
    # Burst c: right. Burst labels (0, 1, 2), predictions (1, 1, 2): F1 0, 2/3, 1.
    burst = np.array(["b", "a", "c", "b", "a", "b", "a", "c"])
    y = np.array([2, 0, 2, 1, 0, 1, 0, 2])
    prob = np.array([[.1, .1, .8], [.05, .95, 0], [.2, .1, .7], [.1, .8, .1],
                     [.6, .4, 0], [.1, .6, .3], [.6, .4, 0], [.2, .1, .7]])
    assert np.isclose(burst_macro_f1(prob, y, burst), 5 / 9)


class _Recorder:
    """Stands in for the W&B run, like `train_utils._NoLogger`, and keeps what is logged."""

    def __init__(self):
        self.logged = []

    def log(self, data, step=None):
        self.logged.append(dict(data))

    def finish(self):
        pass


def _tiny_run(tmp_path, monkeypatch, wandb: bool, split: str = "official_ood"):
    """`run_fold` on CPU: 24 blank 16 px photos of 6 cameras, 2 epochs, W&B recorded.

    The val score of epoch 1 is made perfect (every photo right), so `run_fold` keeps
    epoch 1, not the last one, and the two epochs have different burst scores.
    """
    task = dataclasses.replace(toy_task(n_cameras=6, n_species=3, burst_len=2,
                                        bursts_per_cam=2), image_dir=tmp_path)
    cam = task.df["location"]
    task.df["official_ood"] = np.where(cam < 4, "train", np.where(cam == 4, "val", "test"))
    for i, name in enumerate(task.df["file_name"]):
        Image.new("RGB", (16, 16), (i * 10, 0, 0)).save(tmp_path / name)
    rec, calls = _Recorder(), []

    def epoch_1_perfect(*a, **k):
        out = evaluate(*a, **k)
        calls.append(out)
        if len(calls) == 1:                      # the val pass of epoch 1
            t = out["_true"]
            out.update(macro_f1_present=1.0, _pred=t, _prob=np.eye(out["_prob"].shape[1])[t])
        return out

    monkeypatch.setattr("iwildcam.train.evaluate", epoch_1_perfect)
    monkeypatch.setattr("iwildcam.train.load_task", lambda *a, **k: task)
    monkeypatch.setattr("iwildcam.train.wandb_run", lambda cfg: rec)
    cfg = {"split": split, "cv_fold": 0, "model": "resnet18", "pretrained": False, "seed": 0,
           "size": 32, "height": 224, "image_dir": str(tmp_path), "epochs": 2,
           "batch_size": 8, "num_workers": 0, "lr": 1e-3, "device": "cpu", "amp": False,
           "verbose": False, "wandb": wandb, "pred_dir": str(tmp_path / "preds")}
    return run_fold(cfg), rec.logged


def test_image_and_burst_keys_go_to_wandb_only_when_it_is_on(tmp_path, monkeypatch):
    res_on, on = _tiny_run(tmp_path, monkeypatch, wandb=True)
    res_off, off = _tiny_run(tmp_path, monkeypatch, wandb=False)
    *epochs, final = on
    for e in epochs:
        assert {"val/f1_image", "val/f1_burst"} <= set(e)
        assert e["val/f1_image"] == e["val_macro_f1_present"]
    assert {k for k in final if "/" in k} == {"val/f1_image_best", "val/f1_burst_best",
                                              "test/f1_image", "test/f1_burst"}
    assert final["val/f1_image_best"] == final["val_macro_f1_present_best"]
    kept = max(epochs, key=lambda e: e["val_macro_f1_present"])     # first of a tie, as run_fold
    assert kept is epochs[0] and kept["val/f1_burst"] != epochs[-1]["val/f1_burst"]
    assert final["val/f1_burst_best"] == kept["val/f1_burst"] == 1.0
    assert final["test/f1_image"] == final["test_macro_f1_present"]
    assert not any("/" in k for d in off for k in d)
    # what goes to runs.jsonl does not change
    assert set(res_on) == set(res_off)
    assert all(set(a) == set(b) for a, b in zip(res_on["history"], res_off["history"]))
    assert not any("/" in k for h in res_on["history"] for k in h)


def test_cv5_runs_get_no_test_keys(tmp_path, monkeypatch):
    _res, logged = _tiny_run(tmp_path, monkeypatch, wandb=True, split="cv5")
    final = logged[-1]
    assert {k for k in final if "/" in k} == {"val/f1_image_best", "val/f1_burst_best"}
