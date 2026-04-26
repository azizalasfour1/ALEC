"""

Refactor: this file has been completely rewritten, and its factoring no longer reflects its intent at all.
TODO:
    - rename the file
    - refactor signatures and function names

Automatically construct and apply filter based on the configuration selected by the user (or the default configuration).
"""

import numpy as np
from typing import Dict, Optional, Tuple

from config import TRANSFORM, WINDOW, POLYORDER, RADIUS, NUMERATOR

from .transforms import get_transform


def _construct_description(tr: str) -> str:
    parts = []
    if tr in {"MA", "EXP", "GF", "SG"}:
        parts.append(f"w={WINDOW}")
    if tr == "SG":
        parts.append(f"p={POLYORDER}")
    elif tr == "DFT":
        parts.append(f"r={RADIUS}")
    elif tr == "SIV_prime":
        parts.append(f"n={NUMERATOR}")
    return f"{tr}({', '.join(parts)})"
TRANSFORM_DESC = _construct_description(TRANSFORM)


def get_adaptive_transform(
    X_train: np.ndarray,
    y_train,
) -> Tuple[bool, int, int, str]:
    """
    Get a recommendation to apply + explanation

    In the production code, we always return True, but in our experiments we often override this function to get an always-transform and never-transform results. So do not remove this function!

    Also sometimes we override this in order to get data-specific heuristic; so do not change signature!

    Args:
        X_train: training split of this dataset
        y_train: labels

    Returns:
        Tuple of (recommend_apply, window_length, polyorder, explanation)
    """
    recommend_apply = True
    explanation = f"{TRANSFORM_DESC}"
    fake_window_length, fake_polyorder = (69, 69) if recommend_apply else (0, 0) # downstream logic still relies on this - TODO refactor
    return recommend_apply, fake_window_length, fake_polyorder, explanation


def apply_pretransform(
    X: np.ndarray,
) -> np.ndarray:
    """
    Apply the configured filter to time series data.

    Args:
        X: Time series data (n_samples, n_timepoints)

    Returns:
        Filtered time series data
    """
    transf = get_transform(
        transform_id=TRANSFORM,
        window=WINDOW,
        polyorder=POLYORDER,
        radius=RADIUS,
        numerator=NUMERATOR,
    )
    return transf(X)
