"""ArloDecider: attach an Arlo head to any small model and make calibrated, cost-aware decisions."""

import warnings
from typing import NamedTuple, Optional, Sequence

import torch

from .adapters import as_encoder
from .calibration import Calibration
from .config import ArloConfig
from .decision import AbstainPolicy, Decision, summarise
from .engine import Arlo


def _ids(x) -> torch.Tensor:
    """A list / 1-D / [1, L] integer tensor -> [1, L] long."""
    return torch.as_tensor(x, dtype=torch.long).reshape(1, -1)


def _option_ids(option, index: int) -> torch.Tensor:
    ids = _ids(option)
    if ids.shape[1] == 0:
        raise ValueError(f"option {index} is empty; every option needs at least one token")
    return ids


def _mean_pool_options(h: torch.Tensor, prefix_len: int, lengths) -> torch.Tensor:
    """Mean of each option's own token states. h [N, T, d]; row i = prefix then option i (right-padded)."""
    N, T, _ = h.shape
    pos = torch.arange(T, device=h.device).unsqueeze(0)
    lens = torch.as_tensor(list(lengths), device=h.device).unsqueeze(1)
    mask = ((pos >= prefix_len) & (pos < lens)).unsqueeze(-1).to(h.dtype)
    return (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)


class Features(NamedTuple):
    """Frozen-encoder features for one decision. query [1, d], states [1, P, d], options [1, N, d] or None."""

    query: torch.Tensor
    states: torch.Tensor
    options: Optional[torch.Tensor]


class ArloDecider(torch.nn.Module):
    """A decision maker for small models.

    ``encoder`` turns token ids into hidden states (an nn.Module, a function, or an adapter from
    ``arlo.adapters``). ``hidden_size`` is the width of those states. Arlo reads the states and decides:

      * ``decide_choice``  which of N options, with constraints, option weights and abstention
      * ``decide_yes_no``  yes or no, with a cost-aware threshold and abstention

    Probabilities are calibrated once ``train_arlo`` has fitted the calibration on validation data.
    The encoder is never modified. Only the Arlo head and its calibration are trained and saved.
    """

    def __init__(self, encoder, hidden_size: int | None = None, config: ArloConfig | None = None, **config_kwargs):
        super().__init__()
        if config is None:
            if hidden_size is None:
                raise ValueError("give hidden_size (the encoder's hidden width) or a config")
            config = ArloConfig(hidden_size=hidden_size, **config_kwargs)
        elif hidden_size is not None and hidden_size != config.hidden_size:
            raise ValueError(f"hidden_size {hidden_size} does not match config.hidden_size {config.hidden_size}")
        self.config = config
        self.arlo = Arlo(config)
        self.calibration = Calibration()
        self.trained = False
        self.set_encoder(encoder)

    # -- encoder plumbing -------------------------------------------------------------------
    def set_encoder(self, encoder) -> None:
        """Attach or swap the model Arlo reads. It is stored outside the module tree, so ``state_dict``
        and ``save`` contain Arlo and its calibration only."""
        object.__setattr__(self, "_encoder", as_encoder(encoder))

    @property
    def encoder(self):
        return self._encoder

    def _device(self) -> torch.device:
        return next(self.arlo.parameters()).device

    def _encode(self, ids: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self._encoder(ids.to(self._device()), mask)
        if h.ndim != 3 or h.shape[-1] != self.config.hidden_size:
            raise ValueError(
                f"encoder returned {tuple(h.shape)}; expected [B, T, {self.config.hidden_size}] "
                f"(config.hidden_size). Fix hidden_size or the encoder/adapter."
            )
        return h

    def _warn_if_untrained(self) -> None:
        if not self.trained:
            warnings.warn(
                "Arlo has not been trained: its answers are an untrained prior. Run arlo.train_arlo, or load "
                "a trained checkpoint with ArloDecider.load / mark_trained().",
                UserWarning,
                stacklevel=4,
            )

    def mark_trained(self) -> None:
        self.trained = True

    # -- features (shared by inference and training) ----------------------------------------
    @torch.no_grad()
    def features(self, context, task: str, options: Optional[Sequence] = None, question=None) -> Features:
        """Run the frozen encoder once for one decision."""
        context = _ids(context)
        prefix = context if question is None else torch.cat([context, _ids(question)], dim=1)
        P = prefix.shape[1]
        if task == "binary":
            h = self._encode(prefix)
            return Features(h[:, -1], h[:, :P], None)
        if task != "choice":
            raise ValueError(f"unknown task {task!r}; expected 'choice' or 'binary'")
        if options is None or len(options) == 0:
            raise ValueError("a choice needs at least one option")
        opts = [_option_ids(o, i) for i, o in enumerate(options)]
        seqs = [torch.cat([prefix, o], dim=1)[0] for o in opts]
        lengths = [s.shape[0] for s in seqs]
        batch = torch.zeros((len(seqs), max(lengths)), dtype=torch.long)
        mask = torch.zeros_like(batch, dtype=torch.bool)
        for i, s in enumerate(seqs):
            batch[i, :lengths[i]] = s
            mask[i, :lengths[i]] = True
        h = self._encode(batch, mask.to(self._device()))
        return Features(h[:1, P - 1], h[:1, :P], _mean_pool_options(h, P, lengths).unsqueeze(0))

    def scores(self, f: Features) -> torch.Tensor:
        """Raw option logits [N] from features (uncalibrated)."""
        return self.arlo.choice_scores(f.query, f.options, f.states)[0]

    def logit(self, f: Features) -> torch.Tensor:
        """Raw yes/no logit (scalar) from features (uncalibrated)."""
        return self.arlo.binary_logit(f.query, f.states)[0]

    # -- plain probabilities ----------------------------------------------------------------
    @torch.no_grad()
    def choice(self, context, options: Sequence, question=None) -> torch.Tensor:
        """Calibrated probability of each option, shape [N], sums to 1."""
        self._warn_if_untrained()
        f = self.features(context, "choice", options, question)
        return self.calibration.choice_probs(self.scores(f))

    @torch.no_grad()
    def yes_no(self, context, question=None) -> torch.Tensor:
        """Calibrated probability that the answer is yes, a scalar tensor."""
        self._warn_if_untrained()
        f = self.features(context, "binary", None, question)
        return self.calibration.binary_prob(self.logit(f))

    def pick(self, context, options: Sequence, question=None) -> int:
        """Index of the most probable option."""
        return int(self.choice(context, options, question).argmax())

    def rank(self, context, options: Sequence, question=None) -> list:
        """Option indices, best first."""
        return [int(i) for i in torch.argsort(self.choice(context, options, question), descending=True)]

    # -- full decisions ---------------------------------------------------------------------
    def decide_choice(
        self,
        context,
        options: Sequence,
        question=None,
        *,
        policy: AbstainPolicy | None = None,
        allowed: Sequence[bool] | None = None,
        weights: Sequence[float] | None = None,
    ) -> Decision:
        """Decide among options.

        ``allowed``: hard constraints, one bool per option. Disallowed options get probability 0 and the rest
        are renormalised. If none are allowed the decision abstains.
        ``weights``: non-negative per-option preference or cost weights multiplied into the probabilities
        before the pick (a cost-sensitive decision). The returned ``probs`` stay the calibrated probabilities
        (zeroed where disallowed); the pick maximises ``prob * weight``.
        ``policy``: abstain when confidence, margin or entropy cross the policy's limits.
        """
        probs = self.choice(context, options, question)
        n = probs.numel()
        if allowed is not None:
            if len(allowed) != n:
                raise ValueError(f"allowed has {len(allowed)} entries for {n} options")
            keep = torch.tensor([bool(a) for a in allowed])
            if not keep.any():
                return Decision(None, torch.zeros(n), 0.0, 0.0, 1.0, True, "no option is allowed")
            probs = probs * keep
            probs = probs / probs.sum()
        else:
            keep = torch.ones(n, dtype=torch.bool)
        value = probs
        if weights is not None:
            if len(weights) != n or any(w < 0 for w in weights):
                raise ValueError(f"weights must be {n} non-negative numbers")
            value = probs * torch.tensor([float(w) for w in weights])
        value = value.masked_fill(~keep, -1.0)
        idx = int(value.argmax())
        conf, margin, ent = summarise(probs[keep])
        reason = policy.reason(conf, margin, ent) if policy else None
        return Decision(None if reason else idx, probs, conf, margin, ent, reason is not None, reason)

    def decide_yes_no(
        self,
        context,
        question=None,
        *,
        policy: AbstainPolicy | None = None,
        cost_false_yes: float = 1.0,
        cost_false_no: float = 1.0,
    ) -> Decision:
        """Decide yes or no at the cost-optimal threshold.

        Answer yes when ``p_yes >= cost_false_yes / (cost_false_yes + cost_false_no)``: the bayes decision for
        those two error costs (equal costs give 0.5). ``Decision.probs`` is ``[p_no, p_yes]``.
        """
        if cost_false_yes < 0 or cost_false_no < 0 or cost_false_yes + cost_false_no == 0:
            raise ValueError("costs must be non-negative and not both zero")
        p = float(self.yes_no(context, question))
        threshold = cost_false_yes / (cost_false_yes + cost_false_no)
        probs = torch.tensor([1.0 - p, p])
        conf, margin, ent = summarise(probs)
        reason = policy.reason(conf, margin, ent) if policy else None
        return Decision(None if reason else int(p >= threshold), probs, conf, margin, ent, reason is not None, reason)

    def decide(self, context, question=None, threshold: float = 0.5) -> bool:
        """Shortcut: True if P(yes) >= threshold."""
        return bool(self.yes_no(context, question) >= threshold)

    # -- checkpoints ------------------------------------------------------------------------
    def save(self, path) -> None:
        """Save the Arlo head and its calibration (not the encoder)."""
        torch.save({"config": self.config.to_dict(), "state_dict": self.state_dict(), "trained": self.trained}, path)

    @classmethod
    def load(cls, path, encoder, map_location="cpu") -> "ArloDecider":
        """Load a head saved with ``save`` and attach it to ``encoder``."""
        blob = torch.load(path, map_location=map_location, weights_only=True)
        dec = cls(encoder, config=ArloConfig.from_dict(blob["config"]))
        dec.load_state_dict(blob["state_dict"])
        dec.trained = bool(blob.get("trained", True))
        return dec
