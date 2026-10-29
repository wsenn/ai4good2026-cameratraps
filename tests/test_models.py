"""Focused tests for model configuration switches."""

import torch.nn as nn

from iwildcam.models import build_model


def _batch_norm_layers(model: nn.Module) -> list[nn.BatchNorm2d]:
    return [module for module in model.modules() if isinstance(module, nn.BatchNorm2d)]


def test_freeze_bn_preserves_stats_but_trains_affine_parameters():
    model = build_model("resnet18", n_classes=3, pretrained=False, freeze_bn=True)

    model.train()
    batch_norm = _batch_norm_layers(model)

    assert batch_norm
    assert all(not layer.training for layer in batch_norm)
    assert all(layer.weight.requires_grad and layer.bias.requires_grad for layer in batch_norm)
    assert model.head.training


def test_unfrozen_bn_uses_batch_statistics_during_training():
    model = build_model("resnet18", n_classes=3, pretrained=False, freeze_bn=False)

    model.train()
    batch_norm = _batch_norm_layers(model)

    assert batch_norm
    assert all(layer.training for layer in batch_norm)
