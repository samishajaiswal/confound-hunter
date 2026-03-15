"""
Residual Correlation Detector (Placeholder).

This detector will eventually identify features that correlate with
model residuals in the training set but not in the test set, indicating
potential noise absorption or spurious relationships.

For Phase 2, it returns zero suspicion scores.
"""

from typing import Dict
import pandas as pd


def residual_correlation(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, float]:
    """
    Placeholder implementation of the residual correlation detector.

    Parameters
    ----------
    model : object
        Trained machine learning model.
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series
        Training target vector.
    X_test : pd.DataFrame
        Test feature matrix.
    y_test : pd.Series
        Test target vector.

    Returns
    -------
    Dict[str, float]
        Feature-to-score mapping (all zeros).
    """

    return {feature: 0.0 for feature in X_train.columns}