"""jev-eval: measure and calibrate TypeSafe AI's Jev on your own labeled data."""

from .analysis import composite, report, score_records
from .cache import load_cache, run
from .contract import thresholds_document
from .dataset import LabeledRow, load_config, load_rows, parse_inline
from .metrics import bin_by_confidence, ece, mce, risk_coverage, selective_accuracy
from .provider import Provider, TypeSafeProvider, provider_from_env
from .score import Scored, score_answer
from .thresholds import DEFAULT_GUARD, recommend_threshold

__all__ = [
    "DEFAULT_GUARD",
    "LabeledRow",
    "Provider",
    "Scored",
    "TypeSafeProvider",
    "bin_by_confidence",
    "composite",
    "ece",
    "load_cache",
    "load_config",
    "load_rows",
    "mce",
    "parse_inline",
    "provider_from_env",
    "recommend_threshold",
    "report",
    "risk_coverage",
    "run",
    "score_answer",
    "score_records",
    "selective_accuracy",
    "thresholds_document",
]
