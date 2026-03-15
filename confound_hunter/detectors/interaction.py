"""
Interaction-based Confounder Detector (Placeholder).

This detector will eventually identify spurious signals that appear
only through interactions between two or more features.

For Phase 2, it returns zero suspicion scores.
"""

from typing import Dict
import pandas as pd


def interaction_confounder(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Dict[str, float]:
    """
    Placeholder implementation for interaction-based confounder detection.

    Parameters
    ----------
    model : object
        Trained machine learning model.
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