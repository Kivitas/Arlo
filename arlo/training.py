"""Train the Arlo head on labelled decision examples with the encoder frozen, then calibrate it."""

import copy
from dataclasses import dataclass, field
from typing import Iterable, Optional

import torch

from .decider import ArloDecider


@dataclass
class TrainResult:
    losses: list = field(default_factory=list)       # mean training loss of each step
    val_losses: list = field(default_factory=list)   # validation loss after each epoch
    best_epoch: Optional[int] = None
    temperature: Optional[float] = None              # fitted choice temperature (task "choice")
    platt: Optional[tuple] = None                    # fitted (scale, bias) (task "binary")

    def __iter__(self):                              # so `losses = train_arlo(...)` style code still works
        return iter(self.losses)

    def __len__(self):
        return len(self.losses)

    def __getitem__(self, i):
        return self.losses[i]


def _check_gold(ex, task, n_options=None) -> int:
    gold = int(ex["gold"])
    if task == "choice":
        if not 0 <= gold < n_options:
            raise ValueError(f"choice gold {gold} is outside the {n_options} options")
    elif gold not in (0, 1):
        raise ValueError(f"binary gold must be 0 or 1, got {gold}")
    return gold


def _extract(decider: ArloDecider, examples, task: str) -> list:
    out = []
    for ex in examples:
        f = decider.features(ex["context"], task, ex.get("options"), ex.get("question"))
        n = None if f.options is None else f.options.shape[1]
        out.append((f, _check_gold(ex, task, n)))
    return out


def _loss(decider: ArloDecider, items: list, task: str) -> torch.Tensor:
    terms = []
    for f, gold in items:
        dev = f.query.device
        if task == "choice":
            g = torch.tensor([gold], device=dev)
            terms.append(decider.arlo.training_loss(f.query, f.options, f.states, g, "choice"))
        else:
            g = torch.tensor([float(gold)], device=dev)
            terms.append(decider.arlo.training_loss(f.query, None, f.states, g, "binary"))
    return torch.stack(terms).mean()


@torch.no_grad()
def _val_loss(decider, items, task) -> float:
    decider.arlo.eval()
    try:
        return float(_loss(decider, items, task))
    finally:
        decider.arlo.train()


@torch.no_grad()
def calibrate(decider: ArloDecider, examples: Iterable, task: str = "choice"):
    """Fit the calibration on held-out labelled examples. Returns the temperature (choice) or (scale, bias)."""
    items = _extract(decider, list(examples), task)
    if not items:
        raise ValueError("calibrate needs at least one example")
    decider.arlo.eval()
    if task == "choice":
        return decider.calibration.fit_choice([decider.scores(f) for f, _ in items], [g for _, g in items])
    logits = torch.stack([decider.logit(f) for f, _ in items])
    return decider.calibration.fit_binary(logits, torch.tensor([g for _, g in items]))


def train_arlo(
    decider: ArloDecider,
    batches: Iterable,
    task: str = "choice",
    *,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    epochs: int = 1,
    val: Iterable | None = None,
    patience: int | None = None,
    clip_grad: float | None = 1.0,
    calibrate_on: Iterable | None = None,
) -> TrainResult:
    """Fit the Arlo head. Returns a ``TrainResult`` (iterating it gives the step losses).

    ``batches`` is an iterable of batches; a batch is a list of example dicts:
      * task "choice": {"context": ids, "options": [ids, ...], "gold": index, "question": ids (optional)}
      * task "binary": {"context": ids, "gold": 0 or 1, "question": ids (optional)}

    The frozen encoder runs once per example (features are cached), under ``torch.no_grad``, in eval mode.
    ``val`` (a flat list of examples) is scored after every epoch; the weights with the best validation loss
    are restored at the end, and ``patience`` stops training after that many epochs without improvement.
    ``calibrate_on`` (a flat list of held-out examples, ideally distinct from ``val``) fits temperature
    (choice) or Platt (yes/no) calibration after training; without it calibration stays at identity.
    Only ``decider.arlo`` and ``decider.calibration`` change. Memory: cached features are about
    ``P * hidden_size`` floats per example.
    """
    if task not in ("choice", "binary"):
        raise ValueError(f"unknown task {task!r}; expected 'choice' or 'binary'")
    raw = [list(b) for b in batches]
    if not any(raw):
        raise ValueError("train_arlo needs at least one non-empty batch")

    enc = decider.encoder
    modules = [m for m in (getattr(enc, "model", None), getattr(enc, "fn", None), enc)
               if isinstance(m, torch.nn.Module)]
    prev = [(m, m.training) for m in modules]
    was_training = decider.arlo.training
    result = TrainResult()
    try:
        for m in modules:
            m.eval()
        cached = [_extract(decider, b, task) for b in raw if b]
        val_items = _extract(decider, list(val), task) if val is not None else None
        if val_items is not None and not val_items:
            raise ValueError("val is empty")
        optimiser = torch.optim.AdamW(decider.arlo.parameters(), lr=lr, weight_decay=weight_decay)
        decider.arlo.train()
        best, best_state, stale = float("inf"), None, 0
        for epoch in range(epochs):
            for items in cached:
                loss = _loss(decider, items, task)
                optimiser.zero_grad()
                loss.backward()
                if clip_grad:
                    torch.nn.utils.clip_grad_norm_(decider.arlo.parameters(), clip_grad)
                optimiser.step()
                decider.mark_trained()
                result.losses.append(float(loss.detach()))
            if val_items is not None:
                v = _val_loss(decider, val_items, task)
                result.val_losses.append(v)
                if v < best:
                    best, best_state, stale = v, copy.deepcopy(decider.arlo.state_dict()), 0
                    result.best_epoch = epoch
                else:
                    stale += 1
                    if patience is not None and stale >= patience:
                        break
        if best_state is not None:
            decider.arlo.load_state_dict(best_state)
        if calibrate_on is not None:
            fitted = calibrate(decider, calibrate_on, task)
            if task == "choice":
                result.temperature = fitted
            else:
                result.platt = fitted
    finally:
        for m, was in prev:
            m.train(was)
        decider.arlo.train(was_training)
    return result
