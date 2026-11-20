"""Decision results and abstention policy."""

import math
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class AbstainPolicy:
    """When Arlo should refuse to decide and hand the case back to the caller.

    ``min_confidence``: abstain if the top probability is below this.
    ``min_margin``: abstain if top1 - top2 is below this (choices only).
    ``max_entropy``: abstain if normalised entropy (0 = certain, 1 = uniform) is above this.
    """

    min_confidence: float = 0.0
    min_margin: float = 0.0
    max_entropy: Optional[float] = None

    def reason(self, confidence: float, margin: float, entropy: float) -> Optional[str]:
        if confidence < self.min_confidence:
            return f"confidence {confidence:.3f} < {self.min_confidence}"
        if margin < self.min_margin:
            return f"margin {margin:.3f} < {self.min_margin}"
        if self.max_entropy is not None and entropy > self.max_entropy:
            return f"entropy {entropy:.3f} > {self.max_entropy}"
        return None


@dataclass
class Decision:
    """The outcome of one decision.

    ``index`` is the chosen option (choices) or 1 for yes / 0 for no (yes/no); ``None`` if Arlo abstained.
    ``probs`` are calibrated probabilities over the options (``[p_no, p_yes]`` for yes/no).
    """

    index: Optional[int]
    probs: torch.Tensor
    confidence: float
    margin: float
    entropy: float                    # normalised to [0, 1]
    abstained: bool = False
    reason: Optional[str] = None

    @property
    def answer(self) -> Optional[bool]:
        """For yes/no decisions: True / False, or None if abstained."""
        return None if self.index is None else bool(self.index)

    def to_dict(self) -> dict:
        return {"index": self.index, "probs": [float(p) for p in self.probs], "confidence": self.confidence,
                "margin": self.margin, "entropy": self.entropy, "abstained": self.abstained, "reason": self.reason}


def summarise(probs: torch.Tensor):
    """probs [N] -> (confidence, margin, normalised entropy)."""
    top = torch.topk(probs, min(2, probs.numel())).values
    confidence = float(top[0])
    margin = float(top[0] - top[1]) if top.numel() > 1 else confidence
    ent = -float((probs * torch.log(probs.clamp_min(1e-12))).sum())
    n = probs.numel()
    return confidence, margin, (ent / math.log(n) if n > 1 else 0.0)
