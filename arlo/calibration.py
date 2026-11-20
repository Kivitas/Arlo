"""Post-hoc calibration, so Arlo's probabilities can be trusted for abstention and thresholds.

Choice: temperature scaling, softmax(scores / T). Yes/No: Platt scaling, sigmoid(a * logit + b).
Both are fitted on held-out labelled decisions after the head is trained.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Calibration(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("log_temp", torch.zeros(()))     # choice temperature T = exp(log_temp)
        self.register_buffer("bin_scale", torch.ones(()))
        self.register_buffer("bin_bias", torch.zeros(()))
        self.register_buffer("fitted", torch.zeros((), dtype=torch.bool))

    def choice_probs(self, scores: torch.Tensor) -> torch.Tensor:
        return torch.softmax(scores / self.log_temp.exp(), dim=-1)

    def binary_prob(self, logit: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.bin_scale * logit + self.bin_bias)

    @torch.no_grad()
    def fit_choice(self, scores: list, gold: list) -> float:
        """scores: list of [N_i] raw logits; gold: indices. Fits T by NLL; returns the new T."""
        log_t = torch.zeros((), requires_grad=True)
        opt = torch.optim.LBFGS([log_t], lr=0.5, max_iter=100)

        def nll():
            opt.zero_grad()
            with torch.enable_grad():
                loss = torch.stack([F.cross_entropy((s / log_t.exp()).unsqueeze(0), torch.tensor([g]))
                                    for s, g in zip(scores, gold, strict=True)]).mean()
                loss.backward()
            return loss

        with torch.enable_grad():
            opt.step(nll)
        self.log_temp.copy_(log_t.detach().clamp(-3, 3))
        self.fitted.fill_(True)
        return float(self.log_temp.exp())

    @torch.no_grad()
    def fit_binary(self, logits: torch.Tensor, gold: torch.Tensor) -> tuple:
        """logits [M], gold [M] 0/1. Fits Platt scale and bias; returns (scale, bias)."""
        a = torch.ones((), requires_grad=True)
        b = torch.zeros((), requires_grad=True)
        opt = torch.optim.LBFGS([a, b], lr=0.5, max_iter=100)
        y = gold.float()

        def nll():
            opt.zero_grad()
            with torch.enable_grad():
                loss = F.binary_cross_entropy_with_logits(a * logits + b, y)
                loss.backward()
            return loss

        with torch.enable_grad():
            opt.step(nll)
        self.bin_scale.copy_(a.detach())
        self.bin_bias.copy_(b.detach())
        self.fitted.fill_(True)
        return float(a), float(b)


def expected_calibration_error(confidence, correct, bins: int = 10) -> float:
    """ECE: weighted mean |accuracy - confidence| over equal-width confidence bins."""
    conf = torch.as_tensor(confidence, dtype=torch.float32)
    ok = torch.as_tensor(correct, dtype=torch.float32)
    n = conf.numel()
    if n == 0:
        return 0.0
    edges = torch.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            ece += float(m.sum()) / n * abs(float(ok[m].mean()) - float(conf[m].mean()))
    return ece
