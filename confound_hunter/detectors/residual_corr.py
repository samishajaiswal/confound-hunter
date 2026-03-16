"""
Residual Correlation Detector.

This detector identifies features that correlate strongly with
model residuals in training data but not in test data.

Such features are likely absorbing training noise rather than
capturing true signal.
"""

from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone


class ResidualCorrelationDetector:
    """
    Detect features correlated with model residuals.
    """

    def __init__(
        self,
        model,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ):
        """
        Initialize the detector.

        Parameters
        ----------
        model : trained ML model
            The base model used for residual computation.

        X_train : pd.DataFrame
            Training feature matrix.

        y_train : pd.Series
            Training targets.

        X_test : pd.DataFrame
            Test feature matrix.

        y_test : pd.Series
            Test targets.
        """

        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def run(self) -> Dict[str, float]:
        """
        Execute residual correlation detection.

        Returns
        -------
        Dict[str, float]
            Feature suspicion scores in range [0, 1].
        """

        scores = {}

        for feature in self.X_train.columns:

            # Remove feature from dataset
            X_train_reduced = self.X_train.drop(columns=[feature])
            X_test_reduced = self.X_test.drop(columns=[feature])

            # Clone model to avoid modifying original
            model_clone = clone(self.model)

            # Train model without this feature
            model_clone.fit(X_train_reduced, self.y_train)

            # Compute predictions
            train_pred = model_clone.predict(X_train_reduced)
            test_pred = model_clone.predict(X_test_reduced)

            # Residuals
            train_residuals = self.y_train - train_pred
            test_residuals = self.y_test - test_pred

            # Correlation between feature and residuals
            train_corr, _ = spearmanr(self.X_train[feature], train_residuals)
            test_corr, _ = spearmanr(self.X_test[feature], test_residuals)

            train_corr = abs(train_corr)
            test_corr = abs(test_corr)

            # Avoid divide-by-zero
            epsilon = 1e-6
            ratio = (train_corr + epsilon) / (test_corr + epsilon)

            scores[feature] = ratio

        return self._normalize_scores(scores)

    def _normalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize suspicion scores to range [0,1].
        """

        values = np.array(list(scores.values()))
        min_val = values.min()
        max_val = values.max()

        normalized = {}

        for feature, value in scores.items():

            if max_val - min_val == 0:
                normalized[feature] = 0.0
            else:
                normalized[feature] = (value - min_val) / (max_val - min_val)

        return normalized