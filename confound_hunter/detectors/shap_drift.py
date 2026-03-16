"""
SHAP Train/Test Consistency Detector.

This detector identifies features whose SHAP contributions differ
between training and test datasets. Large divergence indicates that
the model relies on the feature differently on unseen data, which may
signal spurious correlations or overfitting.
"""

from typing import Dict

import numpy as np
import pandas as pd
import shap


class ShapDriftDetector:
    """
    Detects train/test inconsistency in SHAP feature contributions.
    """

    def __init__(
        self,
        model,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        sample_size: int = 500,
        random_state: int = 42,
    ):
        """
        Initialize the detector.

        Parameters
        ----------
        model : trained ML model
            The trained machine learning model.

        X_train : pd.DataFrame
            Training feature matrix.

        X_test : pd.DataFrame
            Test feature matrix.

        sample_size : int
            Number of rows sampled from training data for SHAP computation.

        random_state : int
            Random seed for reproducibility.
        """

        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.sample_size = min(sample_size, len(X_train))
        self.random_state = random_state

    def run(self) -> Dict[str, float]:
        """
        Run the SHAP drift detector.

        Returns
        -------
        Dict[str, float]
            Dictionary mapping feature names to suspicion scores (0–1).
        """

        # Sample training rows for SHAP efficiency
        train_sample = self.X_train.sample(
            self.sample_size,
            random_state=self.random_state,
        )

        # Create SHAP explainer
        explainer = shap.Explainer(self.model, train_sample)

        # Compute SHAP values
        shap_train = explainer(train_sample)
        shap_test = explainer(self.X_test)

        # Mean absolute SHAP importance
        train_importance = np.abs(shap_train.values).mean(axis=0)
        test_importance = np.abs(shap_test.values).mean(axis=0)

        # Prevent division by zero
        epsilon = 1e-6
        ratios = (train_importance + epsilon) / (test_importance + epsilon)

        scores = self._normalize_scores(ratios)

        return dict(zip(self.X_train.columns, scores))

    @staticmethod
    def _normalize_scores(values: np.ndarray) -> np.ndarray:
        """
        Normalize values to range [0, 1].
        """

        min_val = np.min(values)
        max_val = np.max(values)

        if max_val - min_val == 0:
            return np.zeros_like(values)

        return (values - min_val) / (max_val - min_val)