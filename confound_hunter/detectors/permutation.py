"""
Permutation Stability Detector (Placeholder).

This module provides a placeholder implementation for the permutation
stability detector used in Confound Hunter.

The final implementation will measure the difference between permutation
feature importance on the training and test sets.

For Phase 2, this detector simply returns zero suspicion scores for all
features so that the audit engine can run without errors.
"""

from typing import Dict
import pandas as pd


def permutation_stability(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, float]:
    """
    Placeholder implementation of the permutation stability detector.

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
        Mapping of feature names to suspicion scores (all zeros).
    """

    return {feature: 0.0 for feature in X_train.columns}