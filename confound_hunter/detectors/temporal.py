"""
Temporal Confounder Detector (Placeholder).

This detector will eventually identify features that correlate with the
target due to shared temporal trends rather than causal relationships.

For Phase 2, it returns zero suspicion scores.
"""

from typing import Dict
import pandas as pd


def temporal_confounder(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Dict[str, float]:
    """
    Placeholder implementation for temporal confounder detection.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series
        Training target vector.

    Returns
    -------
    Dict[str, float]
        Feature-to-score mapping (all zeros).
    """

    return {feature: 0.0 for feature in X_train.columns}