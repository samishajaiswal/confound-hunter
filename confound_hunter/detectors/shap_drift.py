"""
SHAP Train/Test Consistency Detector (Placeholder).

This module will eventually detect features whose SHAP importance
differs significantly between training and test data.

For Phase 2, it returns zero scores for all features.
"""

from typing import Dict
import pandas as pd


def shap_train_test_consistency(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Dict[str, float]:
    """
    Placeholder implementation for SHAP consistency detection.

    Parameters
    ----------
    model : object
        Trained machine learning model.
    X_train : pd.DataFrame
        Training feature matrix.
    X_test : pd.DataFrame
        Test feature matrix.

    Returns
    -------
    Dict[str, float]
        Feature-to-score mapping (all zeros).
    """

    return {feature: 0.0 for feature in X_train.columns}