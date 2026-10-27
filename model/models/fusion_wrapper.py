from typing import Tuple

import torch
import torch.nn as nn

from model.models.fusion import (
    fuse_trigger_direction_logits,
    probabilities_to_logits,
)
from model.train_config import TrainTask


class DirectThreeClassInferenceWrapper(nn.Module):
    """Expose a stable (logits, probabilities) interface for direct classifiers."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.model(x)
        probabilities = torch.softmax(logits, dim=1)
        return logits, probabilities


class BinaryInferenceWrapper(nn.Module):
    """Map a single binary task model onto the shared 3-class inference contract."""

    def __init__(self, model: nn.Module, task_type: str):
        super().__init__()
        self.model = model
        self.task_type = task_type

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.model(x)
        probabilities = torch.softmax(logits, dim=1)
        p_negative = probabilities[:, 0]
        p_positive = probabilities[:, 1]
        zero = torch.zeros_like(p_positive)

        if self.task_type == TrainTask.DIRECTION:
            three_class_probs = torch.stack((p_negative, zero, p_positive), dim=1)
        elif self.task_type == TrainTask.LONG_OVR:
            three_class_probs = torch.stack((zero, p_negative, p_positive), dim=1)
        elif self.task_type == TrainTask.SHORT_OVR:
            three_class_probs = torch.stack((p_positive, p_negative, zero), dim=1)
        elif self.task_type == TrainTask.TRIGGER:
            raise ValueError(
                "A lone TRIGGER model predicts action/no-action but has no direction; "
                "use TRIGGER_DIRECTION fusion or a directional binary task for backtesting."
            )
        else:
            raise ValueError(f"Unsupported binary task_type: {self.task_type}")

        return probabilities_to_logits(three_class_probs), three_class_probs


class DualHeadInferenceWrapper(nn.Module):
    """Fuse a dual-head model's raw outputs for three-class inference."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        trigger_logits, direction_logits = self.model(x)
        probabilities = fuse_trigger_direction_logits(trigger_logits, direction_logits)
        return probabilities_to_logits(probabilities), probabilities


class FusionWrapper(nn.Module):
    """
    Inference-only wrapper.
    It does not save/load weights (handled by sub-models); it only chains models during forward.
    """
    def __init__(self, models_dict: dict[str, nn.Module], task_type: str):
        super().__init__()
        self.task_type = task_type
        # Use ModuleDict so sub-models switch together under eval(),
        # but we don't need to save this wrapper's state_dict
        self.models = nn.ModuleDict(models_dict)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits, probabilities, _ = self.forward_with_submodel_logits(x)
        return logits, probabilities

    def forward_with_submodel_logits(self, x: torch.Tensor):
        """Return fused output and reuse each model's logits for evaluation."""
        outputs = {name: model(x) for name, model in self.models.items()}
        if self.task_type == TrainTask.TRIGGER_DIRECTION:
            probabilities = fuse_trigger_direction_logits(outputs["trigger"], outputs["direction"])
        elif self.task_type == TrainTask.LONG_SHORT_OVR:
            probabilities = self._long_short_probabilities(outputs)
        else:
            raise ValueError(f"Unknown pipeline task_type: {self.task_type}")
        return probabilities_to_logits(probabilities), probabilities, outputs

    @staticmethod
    def _long_short_probabilities(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
        logits_long = outputs["long_ovr"]
        probs_long = torch.softmax(logits_long, dim=1)
        p_long = probs_long[:, 1]

        logits_short = outputs["short_ovr"]
        probs_short = torch.softmax(logits_short, dim=1)
        p_short = probs_short[:, 1]

        # One-sided evidence maps to its direction. No signal and conflicting
        # signals both map to neutral. These three terms already sum to one.
        p_short_only = p_short * (1.0 - p_long)
        p_long_only = p_long * (1.0 - p_short)
        p_neutral = (1.0 - p_short) * (1.0 - p_long) + p_short * p_long
        fused_probs = torch.stack((p_short_only, p_neutral, p_long_only), dim=1)
        return fused_probs
