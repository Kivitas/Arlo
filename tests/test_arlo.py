import warnings

import pytest
import torch
import torch.nn as nn

from arlo import Arlo, ArloConfig, ArloDecider, FnEncoder, evaluate, train_arlo

V, D = 20, 24


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.rnn = nn.GRU(D, D, batch_first=True)

    def forward(self, ids):
        return self.rnn(self.emb(ids))[0]


def make(**kw):
    torch.manual_seed(0)
    return ArloDecider(Toy(), hidden_size=D, slots=3, dim=16, heads=2, pointer_rank=8, **kw)


def test_config_validation():
    with pytest.raises(ValueError):
        ArloConfig(hidden_size=0)
    with pytest.raises(ValueError):
        ArloDecider(Toy())


def test_engine_shapes_and_option_order_invariance():
    cfg = ArloConfig(hidden_size=D, slots=3, dim=16, heads=2, pointer_rank=8)
    a = Arlo(cfg).eval()
    q, o, s = torch.randn(2, D), torch.randn(2, 4, D), torch.randn(2, 5, D)
    sc = a.choice_scores(q, o, s)
    assert sc.shape == (2, 4)
    perm = torch.tensor([2, 0, 3, 1])
    assert torch.allclose(a.choice_scores(q, o[:, perm], s), sc[:, perm], atol=1e-5)
    assert a.binary_logit(q, s).shape == (2,)


def test_mask_ignores_masked_states():
    a = Arlo(ArloConfig(hidden_size=D, slots=3, dim=16, heads=2, pointer_rank=8)).eval()
    q, s = torch.randn(1, D), torch.randn(1, 6, D)
    mask = torch.tensor([[True] * 4 + [False] * 2])
    s2 = s.clone()
    s2[:, 4:] = 99.0
    assert torch.allclose(a.binary_logit(q, s, mask), a.binary_logit(q, s2, mask), atol=1e-5)


def test_choice_and_yes_no_outputs():
    dec = make()
    with pytest.warns(UserWarning):
        p = dec.choice([1, 2, 3], [[4], [5, 6], [7]])
    assert p.shape == (3,) and abs(float(p.sum()) - 1) < 1e-5
    with pytest.warns(UserWarning):
        y = dec.yes_no([1, 2, 3], question=[4])
    assert 0 <= float(y) <= 1


def test_empty_option_and_wrong_width_rejected():
    dec = make()
    dec.mark_trained()
    with pytest.raises(ValueError):
        dec.choice([1, 2], [[3], []])
    with pytest.raises(ValueError):
        dec.choice([1, 2], [])
    bad = ArloDecider(lambda ids: torch.zeros(*ids.shape, D + 1), hidden_size=D)
    bad.mark_trained()
    with pytest.raises(ValueError, match="hidden_size"):
        bad.yes_no([1])


def test_padding_does_not_change_scores():
    dec = make()
    dec.mark_trained()
    dec.eval()
    a = dec.choice([1, 2, 3], [[4], [5, 6, 7]])
    b = dec.choice([1, 2, 3], [[4]] + [[5, 6, 7]])
    assert torch.allclose(a, b)
    solo = dec.choice([1, 2, 3], [[4], [9, 9, 9, 9, 9]])  # longer sibling => more padding on option 0
    assert torch.allclose(dec.features([1, 2, 3], "choice", [[4], [5]])[2][0, 0],
                          dec.features([1, 2, 3], "choice", [[4], [9, 9, 9, 9]])[2][0, 0], atol=1e-5)
    assert solo.shape == (2,)


def test_training_updates_only_arlo_and_restores_modes():
    dec = make()
    lm = dec.encoder.fn
    lm.train()
    before = {k: v.clone() for k, v in lm.state_dict().items()}
    data = [{"context": [1, 2, 3, i % 5 + 2], "options": [[2], [3]], "gold": i % 2} for i in range(8)]
    losses = train_arlo(dec, [data[:4], data[4:]], "choice", epochs=2)
    assert len(losses) == 4 and dec.trained
    assert lm.training
    assert all(torch.equal(before[k], v) for k, v in lm.state_dict().items())
    assert all(p.grad is None for p in lm.parameters())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        dec.choice([1, 2], [[3], [4]])        # trained: no warning


def test_learns_a_simple_binary_task():
    torch.manual_seed(1)
    dec = ArloDecider(Toy(), hidden_size=D, slots=4, dim=16, heads=2, pointer_rank=8)
    data = [{"context": [1, 5, 5, 5] if i % 2 else [1, 9, 9, 9], "gold": i % 2} for i in range(32)]
    train_arlo(dec, [data[i:i + 8] for i in range(0, 32, 8)], "binary", lr=1e-2, epochs=40)
    assert evaluate(dec, data, "binary")["accuracy"] == 1.0


def test_bad_gold_rejected():
    dec = make()
    with pytest.raises(ValueError):
        train_arlo(dec, [[{"context": [1], "options": [[2], [3]], "gold": 5}]], "choice")
    with pytest.raises(ValueError):
        train_arlo(dec, [[{"context": [1], "gold": 2}]], "binary")
    with pytest.raises(ValueError):
        train_arlo(dec, [[]], "choice")


def test_save_load_roundtrip(tmp_path):
    dec = make()
    dec.mark_trained()
    path = tmp_path / "arlo.pt"
    dec.save(path)
    new = ArloDecider.load(path, dec.encoder)
    assert new.trained and new.config == dec.config
    opts = [[4], [5, 6]]
    assert torch.allclose(dec.choice([1, 2, 3], opts), new.choice([1, 2, 3], opts))
    assert not any(k.startswith("emb") or k.startswith("rnn") for k in dec.state_dict())


def test_encoder_adapters():
    lm = Toy()

    class Out:
        def __init__(self, h):
            self.hidden = h

    class Wrapped(nn.Module):
        def forward(self, ids, return_hidden=False):
            return Out(lm(ids))

    from arlo import AttrEncoder
    dec = ArloDecider(AttrEncoder(Wrapped(), return_hidden=True), hidden_size=D)
    dec.mark_trained()
    assert dec.yes_no([1, 2]).ndim == 0
    dec.set_encoder(FnEncoder(lm))
    assert dec.pick([1, 2], [[3], [4]]) in (0, 1)


# ---- v0.2: decisions, abstention, constraints, costs, calibration, validation ----
from arlo import AbstainPolicy, calibrate, expected_calibration_error  # noqa: E402


def trained(task="choice", seed=2):
    torch.manual_seed(seed)
    dec = ArloDecider(Toy(), hidden_size=D, slots=4, dim=16, heads=2, pointer_rank=8)
    if task == "binary":
        data = [{"context": [1, 5, 5, 5] if i % 2 else [1, 9, 9, 9], "gold": i % 2} for i in range(32)]
    else:
        data = [{"context": [1, 2, 3, i % 5 + 2], "options": [[2], [3], [4]], "gold": i % 3} for i in range(30)]
    dec.mark_trained()
    return dec, data


def test_decide_choice_constraints_weights_and_abstention():
    dec, _ = trained()
    ctx, opts = [1, 2, 3], [[2], [3], [4]]
    d = dec.decide_choice(ctx, opts)
    assert d.index in (0, 1, 2) and not d.abstained and 0 <= d.entropy <= 1
    assert abs(float(d.probs.sum()) - 1) < 1e-5
    # constraints
    d = dec.decide_choice(ctx, opts, allowed=[False, True, False])
    assert d.index == 1 and float(d.probs[0]) == 0.0
    none = dec.decide_choice(ctx, opts, allowed=[False] * 3)
    assert none.abstained and none.index is None
    with pytest.raises(ValueError):
        dec.decide_choice(ctx, opts, allowed=[True])
    # weights can flip the pick
    d = dec.decide_choice(ctx, opts, weights=[1e6, 0, 0])
    assert d.index == 0
    with pytest.raises(ValueError):
        dec.decide_choice(ctx, opts, weights=[-1, 1, 1])
    # abstention
    ab = dec.decide_choice(ctx, opts, policy=AbstainPolicy(min_confidence=1.01))
    assert ab.abstained and ab.index is None and "confidence" in ab.reason
    assert dec.decide_choice(ctx, opts, policy=AbstainPolicy(max_entropy=0.0)).abstained
    assert dec.rank(ctx, opts)[0] == dec.pick(ctx, opts)


def test_decide_yes_no_cost_threshold():
    dec, _ = trained("binary")
    p = float(dec.yes_no([1, 5, 5, 5]))
    d = dec.decide_yes_no([1, 5, 5, 5])
    assert d.answer == (p >= 0.5) and abs(float(d.probs[1]) - p) < 1e-6
    # very costly false yes => never yes unless p is ~1
    assert dec.decide_yes_no([1, 5, 5, 5], cost_false_yes=1e9, cost_false_no=1).answer is False
    assert dec.decide_yes_no([1, 5, 5, 5], cost_false_yes=1, cost_false_no=1e9).answer is True
    assert dec.decide_yes_no([1, 5], policy=AbstainPolicy(min_confidence=1.01)).abstained
    with pytest.raises(ValueError):
        dec.decide_yes_no([1], cost_false_yes=0, cost_false_no=0)


def test_calibration_reduces_overconfidence_choice():
    torch.manual_seed(3)
    dec = ArloDecider(Toy(), hidden_size=D, slots=4, dim=16, heads=2, pointer_rank=8)
    # noisy labels: an overfit head is overconfident; temperature scaling must raise T above 1
    data = [{"context": [1, 2, 3, i % 5 + 2], "options": [[2], [3], [4]], "gold": int(torch.randint(0, 3, ()))}
            for i in range(60)]
    train_arlo(dec, [data[:20]], "choice", lr=1e-2, epochs=60)
    t = calibrate(dec, data[20:], "choice")
    assert t > 1.0
    assert bool(dec.calibration.fitted)
    p = dec.choice([1, 2, 3, 2], [[2], [3], [4]])
    assert abs(float(p.sum()) - 1) < 1e-5


def test_platt_fit_and_ece():
    dec, data = trained("binary")
    scale, bias = calibrate(dec, data, "binary")
    assert scale == scale and bias == bias
    assert expected_calibration_error([1.0, 1.0], [1, 1]) == 0.0
    assert abs(expected_calibration_error([0.9, 0.9], [0, 0]) - 0.9) < 1e-6


def test_validation_early_stopping_restores_best_and_reports():
    torch.manual_seed(4)
    dec = ArloDecider(Toy(), hidden_size=D, slots=4, dim=16, heads=2, pointer_rank=8)
    data = [{"context": [1, 2, 3, i % 5 + 2], "options": [[2], [3]], "gold": i % 2} for i in range(24)]
    res = train_arlo(dec, [data[:8], data[8:16]], "choice", epochs=30, lr=1e-2, val=data[16:], patience=2,
                     calibrate_on=data[16:])
    assert len(res.val_losses) <= 30 and res.best_epoch is not None
    assert res.val_losses[res.best_epoch] == min(res.val_losses)
    assert res.temperature is not None
    assert len(list(res)) == len(res.losses)


def test_evaluate_metrics_and_selective_accuracy():
    dec, data = trained("binary")
    m = evaluate(dec, data, "binary", policy=AbstainPolicy(min_confidence=0.0))
    assert m["coverage"] == 1.0 and m["nll"] > 0 and 0 <= m["ece"] <= 1
    m2 = evaluate(dec, data, "binary", policy=AbstainPolicy(min_confidence=1.01))
    assert m2["coverage"] == 0.0 and m2["selective_accuracy"] is None


def test_checkpoint_keeps_calibration(tmp_path):
    dec, data = trained()
    dec.calibration.log_temp.fill_(0.7)
    dec.calibration.fitted.fill_(True)
    dec.save(tmp_path / "a.pt")
    new = ArloDecider.load(tmp_path / "a.pt", dec.encoder)
    assert abs(float(new.calibration.log_temp) - 0.7) < 1e-6 and bool(new.calibration.fitted)
