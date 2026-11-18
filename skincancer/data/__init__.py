"""HAM10000 data handling: metadata, validation, lesion-grouped splits, datasets."""

from skincancer.data.metadata import DX_FULL_NAMES, MetadataError, class_distribution, load_metadata
from skincancer.data.splits import (
    LeakageError,
    load_or_create_splits,
    make_lesion_splits,
    summarize_splits,
    verify_no_leakage,
)
from skincancer.data.validation import ValidationReport, validate_images

__all__ = [
    "DX_FULL_NAMES",
    "LeakageError",
    "MetadataError",
    "ValidationReport",
    "class_distribution",
    "load_metadata",
    "load_or_create_splits",
    "make_lesion_splits",
    "summarize_splits",
    "validate_images",
    "verify_no_leakage",
]
