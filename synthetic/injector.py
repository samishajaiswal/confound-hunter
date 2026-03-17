"""
confound_hunter.synthetic.injector
----------------------------------

Synthetic confounder injection engine used to benchmark the
Confound Hunter auditing system.

The ConfoundInjector class inserts known confounders into a
clean dataset so that the detection pipeline can be evaluated
using precision and recall.

Six injection types are supported:

A. Spurious Correlation
B. Leaky Feature
C. Proxy Confounder
D. Clean Signal (control)
E. Temporal Confounder
F. Interaction Confounder

Each injected feature is tracked in a ground-truth registry
so benchmarks can measure whether the detectors correctly
flag the injected confounders.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


class ConfoundInjector:
    """
    Inject synthetic confounders into tabular datasets.

    This class is used by the benchmark engine to create datasets
    with known spurious features. These injected features serve as
    ground truth for evaluating the performance of Confound Hunter.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target variable.
    random_state : int, optional
        Random seed for reproducibility.

    Attributes
    ----------
    X : pd.DataFrame
        Feature matrix (modified with injected features).
    y : pd.Series
        Target variable.
    rng : np.random.Generator
        Random number generator.
    ground_truth : Dict[str, str]
        Mapping of injected feature names -> confounder type.
    """

    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        random_state: int | None = None,
    ) -> None:
        """
        Initialise the confound injector.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Target variable.
        random_state : int, optional
            Seed for deterministic injections.
        """

        self.X = X.copy()
        self.y = y.copy()

        self.rng = np.random.default_rng(random_state)

        # Ground truth registry
        # {feature_name: injection_type}
        self.ground_truth: Dict[str, str] = {}

    # ---------------------------------------------------------
    # Utility methods
    # ---------------------------------------------------------

    def _register_feature(self, feature_name: str, injection_type: str) -> None:
        """
        Register a newly injected feature.

        Parameters
        ----------
        feature_name : str
            Name of the injected feature.
        injection_type : str
            Type of confounder injected.
        """

        self.ground_truth[feature_name] = injection_type

    def get_dataset(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Return the dataset with injected confounders.

        Returns
        -------
        Tuple[pd.DataFrame, pd.Series]
            Modified feature matrix and original target.
        """

        return self.X, self.y

    def get_ground_truth(self) -> Dict[str, str]:
        """
        Return ground truth mapping for injected features.

        Returns
        -------
        Dict[str, str]
            Mapping of feature -> confounder type.
        """

        return self.ground_truth.copy()

    # ---------------------------------------------------------
    # Injection methods (to be implemented in later steps)
    # ---------------------------------------------------------

    def inject_spurious_correlation(
        self,
        noise_std: float = 0.2,
        feature_name: str = "spurious_corr",
    ) -> str:
        """
        Inject a spurious correlation feature.

        The feature is constructed as the target variable plus
        Gaussian noise. It will appear highly predictive but
        represents a non-causal relationship.

        Parameters
        ----------
        noise_std : float, default=0.2
            Standard deviation of Gaussian noise.
        feature_name : str, default="spurious_corr"
            Name of the injected feature.

        Returns
        -------
        str
            Name of the injected feature.
        """

        if feature_name in self.X.columns:
            raise ValueError(f"Feature '{feature_name}' already exists.")

        # Convert target to numeric array
        y_values = self.y.to_numpy()

        # Generate Gaussian noise
        noise = self.rng.normal(
            loc=0.0,
            scale=noise_std,
            size=len(self.X),
        )

        # Create spurious feature
        spurious_feature = y_values + noise

        # Insert into feature matrix
        self.X[feature_name] = spurious_feature

        # Register ground truth
        self._register_feature(feature_name, "spurious_correlation")

        return feature_name

    def inject_leaky_feature(
        self,
        leak_fraction: float = 0.9,
        noise_std: float = 0.05,
        feature_name: str = "leaky_feature",
    ) -> str:
        """
        Inject a target leakage feature.

        The feature is partially derived from the target variable,
        simulating common data leakage scenarios such as post-event
        features or improperly constructed aggregates.

        Parameters
        ----------
        leak_fraction : float, default=0.9
            Fraction of target information leaked into the feature.
        noise_std : float, default=0.05
            Standard deviation of Gaussian noise.
        feature_name : str, default="leaky_feature"
            Name of injected feature.

        Returns
        -------
        str
            Name of the injected feature.
        """

        if feature_name in self.X.columns:
            raise ValueError(f"Feature '{feature_name}' already exists.")

        y_values = self.y.to_numpy()

        noise = self.rng.normal(
            loc=0.0,
            scale=noise_std,
            size=len(self.X),
        )

        # Create leakage feature
        leak_feature = (y_values * leak_fraction) + noise

        # Add to dataset
        self.X[feature_name] = leak_feature

        # Register ground truth
        self._register_feature(feature_name, "leaky_feature")

        return feature_name

    def inject_proxy_feature(
        self,
        noise_std: float = 0.05,
        feature_name: str = "proxy_feature",
    ) -> str:
        """
        Inject a proxy confounder feature.

        A proxy feature is created by cloning an existing feature
        and adding small Gaussian noise. This simulates variables
        that appear predictive only because they mirror another
        variable in the dataset.

        Parameters
        ----------
        noise_std : float, default=0.05
            Standard deviation of Gaussian noise to add.
        feature_name : str, default="proxy_feature"
            Name of the injected proxy feature.

        Returns
        -------
        str
            Name of the injected feature.
        """
        if feature_name in self.X.columns:
            raise ValueError(f"Feature '{feature_name}' already exists.")

        # Randomly choose an existing feature to clone
        source_feature = self.rng.choice(self.X.columns)
        source_values = self.X[source_feature].to_numpy()

        noise = self.rng.normal(
            loc=0.0,
            scale=noise_std,
            size=len(self.X),
        )

        proxy_feature = source_values + noise

        # Insert into dataset
        self.X[feature_name] = proxy_feature

        # Register ground truth
        self._register_feature(feature_name, "proxy_confounder")
        return feature_name

    def inject_clean_signal(
        self,
        signal_strength: float = 0.5,
        noise_std: float = 0.2,
        feature_name: str = "clean_signal",
    ) -> str:
        """
        Inject a genuine predictive feature (clean signal).

        This feature is partially correlated with the target variable,
        representing a legitimate signal that SHOULD NOT be flagged
        as a confounder.

        Parameters
        ----------
        signal_strength : float, default=0.5
            Strength of correlation with target.
        noise_std : float, default=0.2
            Standard deviation of Gaussian noise.
        feature_name : str, default="clean_signal"
            Name of injected feature.

        Returns
        -------
        str
            Name of injected feature.
        """
        if feature_name in self.X.columns:
            raise ValueError(f"Feature '{feature_name}' already exists.")

        y_values = self.y.to_numpy()

        noise = self.rng.normal(
            loc=0.0,
            scale=noise_std,
            size=len(self.X),
        )

        clean_feature = (signal_strength * y_values) + noise

        self.X[feature_name] = clean_feature

        # IMPORTANT: This is NOT a confounder
        self._register_feature(feature_name, "clean_signal")

        return feature_name

    def inject_temporal_confounder(
        self,
        noise_std: float = 0.1,
        feature_name: str = "temporal_confounder",
    ) -> str:
        """
        Inject a temporal confounder feature.

        This feature simulates a time-dependent trend that correlates
        with the target due to shared temporal structure rather than
        a causal relationship.

        Parameters
        ----------
        noise_std : float, default=0.1
            Standard deviation of Gaussian noise.
        feature_name : str, default="temporal_confounder"
            Name of injected feature.

        Returns
        -------
        str
            Name of injected feature.
        """
        if feature_name in self.X.columns:
            raise ValueError(f"Feature '{feature_name}' already exists.")

        n = len(self.X)

        # Create a synthetic time index (normalized)
        time_index = np.arange(n) / n

        noise = self.rng.normal(
            loc=0.0,
            scale=noise_std,
            size=n,
        )

        temporal_feature = time_index + noise

        self.X[feature_name] = temporal_feature

        self._register_feature(feature_name, "temporal_confounder")

        return feature_name
    def inject_interaction_confounder(self) -> Tuple[str, str]:
        """
        Inject an interaction-based confounder.

        Returns
        -------
        Tuple[str, str]
            Names of the injected interaction features.
        """
        raise NotImplementedError
