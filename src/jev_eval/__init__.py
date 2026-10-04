"""jev-eval: measure and calibrate TypeSafe AI's Jev on your own labeled data."""

from .analysis import composite, report, score_records
from .cache import load_cache, run
from .calibrate import calibration_document, isotonic, recalibration
from .compare import compare
from .contract import GuardOptions, recommend_all, thresholds_document
from .dataset import LabeledRow, load_config, load_rows, parse_inline
from .metrics import bin_by_confidence, ece, mce, risk_coverage, selective_accuracy
from .provider import Provider, TypeSafeProvider, provider_from_env
from .score import Scored, score_answer
from .slices import slice_report
from .thresholds import DEFAULT_GUARD, recommend_threshold

__version__ = "1.0.0"

__all__ = [
    "DEFAULT_GUARD",
    "GuardOptions",
    "LabeledRow",
    "Provider",
    "Scored",
    "TypeSafeProvider",
    "__version__",
    "bin_by_confidence",
    "calibration_document",
    "compare",
    "composite",
    "ece",
    "isotonic",
    "load_cache",
    "load_config",
    "load_rows",
    "mce",
    "parse_inline",
    "provider_from_env",
    "recalibration",
    "recommend_all",
    "recommend_threshold",
    "report",
    "risk_coverage",
    "run",
    "score_answer",
    "score_records",
    "selective_accuracy",
    "slice_report",
    "thresholds_document",
]
