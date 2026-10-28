"""Checks of the training options in `iwildcam/experiments.py`. CPU only, no images.

    python -m pytest tests/test_experiments.py -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from iwildcam.experiments import (ClassMixSampler, ExperimentDataset, VisitLimitSampler,
                                  apply_train_weights, cv5_fold, fda_swap, train_sampler,
                                  visit_groups)
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
