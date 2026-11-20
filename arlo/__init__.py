"""Arlo: a tiny, calibrated decision maker for small models.

Attach it to any model that gives hidden states; it decides "which option?" and "yes or no?", with
abstention, hard constraints, cost-sensitive thresholds and calibrated probabilities.
"""

from .adapters import AttrEncoder, FnEncoder, HFEncoder, as_encoder
from .calibration import Calibration, expected_calibration_error
from .config import ArloConfig
from .decider import ArloDecider, Features
from .decision import AbstainPolicy, Decision
from .engine import Arlo, MultiHeadPointer, binary_loss, choice_loss
from .metrics import evaluate
from .training import TrainResult, calibrate, train_arlo

__version__ = "0.2.0"

__all__ = [
    "Arlo", "ArloConfig", "ArloDecider", "Features", "MultiHeadPointer", "choice_loss", "binary_loss",
    "AbstainPolicy", "Decision", "Calibration", "expected_calibration_error",
    "train_arlo", "calibrate", "TrainResult", "evaluate",
    "FnEncoder", "HFEncoder", "AttrEncoder", "as_encoder",
]
