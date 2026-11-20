"""Evaluation of a decider on labelled decision examples (same format as ``train_arlo``)."""

import math
import warnings
from typing import Iterable

import torch

from .calibration import expected_calibration_error
from .decider import ArloDecider
from .decision import AbstainPolicy


def evaluate(
    decider: ArloDecider,
    examples: Iterable,
    task: str = "choice",
    threshold: float = 0.5,
    policy: AbstainPolicy | None = None,
) -> dict:
    """Quality of the decider's decisions.

    Returns ``items``, ``accuracy``, ``nll`` (mean negative log-likelihood of the gold answer), ``ece``
    (expected calibration error), and, if ``policy`` is given, ``coverage`` (share of items decided) and
    ``selective_accuracy`` (accuracy on the items it did decide, None if it decided none).
    """
    if task not in ("choice", "binary"):
        raise ValueError(f"unknown task {task!r}; expected 'choice' or 'binary'")
    n = correct = decided = decided_correct = 0
    nll, conf, ok = 0.0, [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for ex in examples:
            gold = int(ex["gold"])
            if task == "choice":
                probs = decider.choice(ex["context"], ex["options"], ex.get("question"))
                pred, p_gold = int(probs.argmax()), float(probs[gold])
                c = float(probs.max())
            else:
                p = float(decider.yes_no(ex["context"], ex.get("question")))
                pred, p_gold = int(p >= threshold), (p if gold == 1 else 1.0 - p)
                c = p if pred == 1 else 1.0 - p
                probs = torch.tensor([1.0 - p, p])
            n += 1
            correct += int(pred == gold)
            nll -= math.log(max(p_gold, 1e-12))
            conf.append(c)
            ok.append(int(pred == gold))
            if policy is not None:
                srt = torch.topk(probs, min(2, probs.numel())).values
                margin = float(srt[0] - srt[1]) if srt.numel() > 1 else c
                ent = -float((probs * torch.log(probs.clamp_min(1e-12))).sum()) / math.log(max(probs.numel(), 2))
                if policy.reason(c, margin, ent) is None:
                    decided += 1
                    decided_correct += int(pred == gold)
    out = {"items": n, "accuracy": correct / n if n else None, "nll": nll / n if n else None,
           "ece": expected_calibration_error(conf, ok) if n else None}
    if policy is not None:
        out["coverage"] = decided / n if n else None
        out["selective_accuracy"] = decided_correct / decided if decided else None
    return out
