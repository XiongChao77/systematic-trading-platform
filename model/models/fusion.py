import torch


def probabilities_to_logits(probabilities: torch.Tensor) -> torch.Tensor:
    """Convert normalized probabilities to numerically safe log-probabilities."""
    return torch.log(probabilities.clamp_min(1e-8))


def fuse_trigger_direction_logits(
    trigger_logits: torch.Tensor,
    direction_logits: torch.Tensor,
    *,
    hard_gate: bool = True,
) -> torch.Tensor:
    """Gate direction probabilities by the trigger's hard action decision.

    Return [Short, Neutral, Long]. A no-action decision (including a trigger
    tie) returns [0, 1, 0]; otherwise retain the conditional direction
    probabilities with zero neutral probability. Training losses can disable
    the hard gate to retain differentiable joint probabilities.
    """
    direction_probs = torch.softmax(direction_logits, dim=1)

    if not hard_gate:
        trigger_probs = torch.softmax(trigger_logits, dim=1)
        return torch.stack(
            (
                trigger_probs[:, 1] * direction_probs[:, 0],
                trigger_probs[:, 0],
                trigger_probs[:, 1] * direction_probs[:, 1],
            ),
            dim=1,
        )

    action = trigger_logits[:, 1] > trigger_logits[:, 0]
    p_short = torch.where(action, direction_probs[:, 0], 0.0)
    p_neutral = (~action).to(direction_probs.dtype)
    p_long = torch.where(action, direction_probs[:, 1], 0.0)
    return torch.stack((p_short, p_neutral, p_long), dim=1)
