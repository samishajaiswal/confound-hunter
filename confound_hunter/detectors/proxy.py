"""
Proxy Confounder Detector (Placeholder).

This detector will eventually use mutual information to identify
features that act as proxies for other variables rather than providing
independent predictive signal.

For Phase 2, it returns zero scores for all features.
"""

from typing import Dict
import pandas as pd


def proxy_confounder(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Dict[str, float]:
    """
    Placeholder implementation for proxy confounder detection.

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