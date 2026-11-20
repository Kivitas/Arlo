# Arlo

A tiny, calibrated decision maker for small models. Arlo reads a model's hidden states and answers two kinds of
question, without generating any text:

- **Choice**: which of N options is best? Returns a calibrated probability for each option.
- **Yes/No**: does this hold? Returns P(yes).

It is about 0.4 M parameters at width 384 (8 slots, width 128). It never changes the model it reads.
You train only Arlo, on labelled decisions, with the model frozen.

Arlo was extracted from the Inky project, where it was ported from the
Choice and Noul primitives of Rhen. The maths is unchanged: with the same weights it gives the same scores
as the original (checked numerically). This package adds an encoder-agnostic interface, padding masks,
calibration, abstention, constraints, cost-aware decisions, validation, checkpoints and tests.

**Status:** Arlo starts untrained. Its answers mean nothing until you train it on your own labelled
decisions. It warns once per call while untrained. No pretrained weights ship with this repository.

## Install

```bash
pip install git+https://github.com/kivitas/arlo-decision.git      # or: pip install -e . from a clone
pip install "arlo-decision[hf]"                                    # optional, for Hugging Face models
```

Requires Python 3.10+ and PyTorch.

## Use with your model

Arlo needs one thing from your model: **token ids in, hidden states `[B, T, hidden_size]` out.** The model
should be causal (decoder or recurrent). Any `nn.Module` or function with that shape works as is.

```python
from arlo import ArloDecider, AbstainPolicy, train_arlo, evaluate

decider = ArloDecider(my_model, hidden_size=384)       # my_model(ids) -> [B, T, 384]

# 1. Train the head (model frozen). Examples are token-id lists. Hold out validation and calibration data.
data = [{"context": ctx_ids, "options": [a_ids, b_ids, c_ids], "gold": 1}, ...]
train_arlo(decider, batches, "choice", epochs=20, val=val_data, patience=3, calibrate_on=cal_data)

# 2. Decide.
policy = AbstainPolicy(min_confidence=0.7, min_margin=0.15)
d = decider.decide_choice(ctx_ids, [a_ids, b_ids, c_ids], policy=policy)
d.index        # chosen option, or None if Arlo abstained
d.probs        # calibrated probabilities
d.abstained, d.reason, d.confidence, d.margin, d.entropy

# 3. Check it, including how accurate it is on the cases it chooses to answer.
evaluate(decider, test_data, "choice", policy=policy)   # accuracy, nll, ece, coverage, selective_accuracy
```

### What makes it a decision maker

- **Calibrated probabilities.** `calibrate_on=` fits temperature scaling (choice) or Platt scaling (yes/no)
  after training. Without it calibration is the identity and the probabilities are only as honest as the head.
- **Abstention.** `AbstainPolicy(min_confidence, min_margin, max_entropy)` makes Arlo refuse unclear cases
  so the caller can fall back to a rule, a bigger model or a person. `evaluate` reports coverage and
  selective accuracy so you can pick the thresholds.
- **Hard constraints.** `allowed=[True, False, True]` removes options that are not permitted. Their
  probability is 0 and the rest are renormalised. If nothing is allowed it abstains.
- **Cost-sensitive choices.** `weights=[...]` multiplies per-option preference or cost weights into the pick.
- **Cost-aware yes/no.** `decide_yes_no(ctx, cost_false_yes=5, cost_false_no=1)` answers yes only when
  `p_yes >= 5 / 6`, the cost-optimal threshold. `Decision.answer` is True, False or None.
- **Validation and early stopping.** `val=` and `patience=` stop training when validation loss stops improving
  and restore the best weights. Features from the frozen model are computed once and cached.
- **Plain probabilities** are still there: `choice`, `yes_no`, `pick`, `rank`.

```python
decider.decide_yes_no(ctx_ids, cost_false_yes=5, cost_false_no=1, policy=policy).answer
decider.decide_choice(ctx_ids, opts, allowed=[True, False, True], weights=[1, 1, 2])
decider.save("arlo.pt")                                  # Arlo weights + calibration only
decider = ArloDecider.load("arlo.pt", my_model)
```

`question` is optional extra ids appended after the context. In a choice, each option is scored after the
context and question, and is represented by the mean of its own token states. Yes/no examples are
`{"context": ids, "gold": 0 or 1}`.

### Adapters

If your model does not return a bare tensor, wrap it:

```python
from arlo import HFEncoder, AttrEncoder, FnEncoder

ArloDecider(HFEncoder(hf_model), hidden_size=hf_model.config.hidden_size)     # last hidden layer
ArloDecider(AttrEncoder(model, "hidden", return_hidden=True), hidden_size=384) # model(ids, return_hidden=True).hidden
ArloDecider(FnEncoder(lambda ids: ...), hidden_size=384)
```

A custom encoder may take `(input_ids, attention_mask)`; name the second parameter `attention_mask` and Arlo
passes a bool mask (True = real token), which matters for non-causal encoders.

### Hidden states you already have

The engine takes tensors, so you can skip the decider and the encoder:

```python
from arlo import Arlo, ArloConfig
arlo = Arlo(ArloConfig(hidden_size=384))
scores = arlo.choice_scores(query, options, states)   # [B, d], [B, N, d], [B, S, d] -> [B, N] logits
logit  = arlo.binary_logit(query, states)             # -> [B]
loss   = arlo.training_loss(query, options, states, gold, task="choice")
```

## Configuration

`ArloConfig(hidden_size, slots=8, dim=128, heads=4, pointer_rank=64, refine_steps=2)`. `hidden_size` must
match your model. Pass the others as keyword arguments to `ArloDecider` to make Arlo smaller or larger.

## How it works

1. K latent decision slots cross-attend to the hidden states and refine each other through a gated update.
   They are reset for every decision.
2. The query (the last state of the context or question) is shifted by the slots that match it best.
3. Choice: a multi-head bilinear pointer scores the query against each option. Option scores depend on the
   other options and do not depend on option order. Yes/No: a linear head gives one logit.

Not included: Rhen's Score, Range, Extract and Cascade primitives, its calibration layers, abstention and
trajectory memory.

## Example and tests

```bash
python examples/toy_model.py     # trains, calibrates and evaluates Arlo on a toy GRU
python -m pytest -q
```

## Licence

MIT, copyright (c) 2026 Kivitas. See `LICENSE`.
