"""End to end: a toy small model + Arlo. Train a decision head, then decide.

Run:  python examples/toy_model.py
"""

import torch
import torch.nn as nn

from arlo import AbstainPolicy, ArloDecider, evaluate, train_arlo

VOCAB, D = 32, 48


class ToyLM(nn.Module):
    """Stand-in for your small model: ids [B, T] -> hidden states [B, T, D]."""

    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, D)
        self.rnn = nn.GRU(D, D, batch_first=True)   # causal

    def forward(self, ids):
        return self.rnn(self.emb(ids))[0]


torch.manual_seed(0)
lm = ToyLM()

# Task: given a context, pick the option that repeats the context's first token.
def make(n):
    out = []
    for _ in range(n):
        ctx = torch.randint(2, VOCAB, (6,)).tolist()
        right = [ctx[0]]
        wrong = [t for t in torch.randint(2, VOCAB, (3,)).tolist() if t != ctx[0]][:2] or [2]
        opts = [[w] for w in wrong] + [right]
        out.append({"context": ctx, "options": opts, "gold": len(opts) - 1})
    return out


data = make(240)
arlo = ArloDecider(lm, hidden_size=D, slots=4, dim=32, heads=2, pointer_rank=16)
train, val, cal, test = data[:128], data[128:160], data[160:200], data[200:]
res = train_arlo(arlo, [train[i:i + 16] for i in range(0, 128, 16)], "choice", lr=3e-3, epochs=20,
                 val=val, patience=4, calibrate_on=cal)
print(f"loss {res.losses[0]:.3f} -> {res.losses[-1]:.3f}, best epoch {res.best_epoch}, T={res.temperature:.2f}")
print("held-out:", evaluate(arlo, test, "choice", policy=AbstainPolicy(min_confidence=0.7)))

ex = test[0]
d = arlo.decide_choice(ex["context"], ex["options"], policy=AbstainPolicy(min_confidence=0.7))
print("decision:", d.to_dict())
print("with the last option forbidden:",
      arlo.decide_choice(ex["context"], ex["options"], allowed=[True] * (len(ex["options"]) - 1) + [False]).index)
