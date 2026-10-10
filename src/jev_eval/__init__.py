"""jev-eval: measure and calibrate TypeSafe AI's Jev on your own labeled data."""

from .analysis import composite, report, score_records
from .budget import Usage, usage
from .cache import load_cache, run
from .calibrate import calibration_document, isotonic, recalibration
from .check import check
from .compare import compare
from .contract import GuardOptions, recommend_all, thresholds_document
from .dataset import LabeledRow, load_config, load_rows, parse_inline
from .metrics import bin_by_confidence, ece, mce, risk_coverage, selective_accuracy
from .provider import Provider, RequestRejected, TypeSafeProvider, provider_from_env
from .score import Scored, score_answer
from .slices import slice_report
from .thresholds import DEFAULT_GUARD, recommend_threshold
from .yes import yes_cuts

__version__ = "1.2.0"

__all__ = [
    "DEFAULT_GUARD",
    "GuardOptions",
    "LabeledRow",
    "Provider",
    "RequestRejected",
    "Scored",
    "TypeSafeProvider",
    "Usage",
    "__version__",
    "bin_by_confidence",
    "calibration_document",
    "check",
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
    "usage",
    "yes_cuts",
]
