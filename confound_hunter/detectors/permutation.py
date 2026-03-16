"""
Permutation Stability Detector.

This module implements Detector 1 from the Confound Hunter pipeline:
Permutation Stability.

The goal is to identify features whose importance is significantly higher
on the training data compared to the test data. Such features may indicate
overfitting or spurious correlations.

Detection Logic
---------------
1. Compute permutation importance on training data.
2. Compute permutation importance on test data.
3. Calculate stability ratio = train_importance / test_importance.
4. Convert ratios into normalized suspicion scores in the range [0, 1].

A high ratio suggests the feature relies on patterns present in training
data but not generalizable to unseen data.
"""

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


class PermutationStabilityDetector:
    """
    Detect features with unstable permutation importance between train and test sets.
    """

    def __init__(
        self,
        model,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        n_repeats: int = 10,
        random_state: int = 42,
    ):
        """
        Initialize the detector.

        Parameters
        ----------
        model : sklearn-compatible estimator
            Trained machine learning model.

        X_train : pd.DataFrame
            Training feature matrix.

        y_train : pd.Series
            Training target vector.

        X_test : pd.DataFrame
            Test feature matrix.

        y_test : pd.Series
            Test target vector.

        n_repeats : int
            Number of permutations per feature.

        random_state : int
            Random seed for reproducibility.
        """
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.n_repeats = n_repeats
        self.random_state = random_state

    def run(self) -> Dict[str, float]:
        """
        Execute permutation stability detection.

        Returns
        -------
        Dict[str, float]
            Mapping of feature name to suspicion score (0–1).
        """

        # Compute permutation importance for train data
        train_importance = permutation_importance(
            self.model,
            self.X_train,
            self.y_train,
            n_repeats=self.n_repeats,
            random_state=self.random_state,
            n_jobs=-1,
        )

        # Compute permutation importance for test data
        test_importance = permutation_importance(
            self.model,
            self.X_test,
            self.y_test,
            n_repeats=self.n_repeats,
            random_state=self.random_state,
            n_jobs=-1,
        )

        train_scores = train_importance.importances_mean
        test_scores = test_importance.importances_mean

        # Avoid division by zero
        epsilon = 1e-6
        ratios = (train_scores + epsilon) / (np.abs(test_scores) + epsilon)

        # Normalize ratios into [0, 1] suspicion scores
        suspicion_scores = self._normalize_scores(ratios)

        return dict(zip(self.X_train.columns, suspicion_scores))

    @staticmethod
    def _normalize_scores(values: np.ndarray) -> np.ndarray:
        """
        Normalize an array of values to the range [0, 1].

        Parameters
        ----------
        values : np.ndarray
            Raw ratio values.

        Returns
        -------
        np.ndarray
            Normalized suspicion scores.
        """
        min_val = np.min(values)
        max_val = np.max(values)

        if max_val - min_val == 0:
            return np.zeros_like(values)

        return (values - min_val) / (max_val - min_val)