"""
Proxy Confounder Detector
-------------------------

Identifies proxy confounders by measuring correlation between
candidate variables, treatment, and outcome.

A proxy confounder typically correlates with both the treatment
and the outcome, suggesting it may represent an unobserved factor.

Author: Confound Hunter Project
"""

from typing import Dict, List

import pandas as pd
from scipy.stats import pearsonr


class ProxyConfounderDetector:
    """
    Detect proxy confounders based on correlation structure.

    A variable is flagged if it significantly correlates with both
    treatment and outcome.
    """

    def __init__(
        self,
        corr_threshold: float = 0.3,
        p_threshold: float = 0.05
    ):
        """
        Initialize detector.

        Parameters
        ----------
        corr_threshold : float
            Minimum correlation magnitude required.
        p_threshold : float
            Statistical significance threshold.
        """

        self.corr_threshold = corr_threshold
        self.p_threshold = p_threshold

    def detect(
        self,
        df: pd.DataFrame,
        treatment: str,
        outcome: str,
        candidate_vars: List[str]
    ) -> Dict[str, Dict]:
        """
        Detect proxy confounders.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataset.
        treatment : str
            Treatment variable.
        outcome : str
            Outcome variable.
        candidate_vars : List[str]
            Variables to evaluate.

        Returns
        -------
        Dict[str, Dict]
            Detection results.
        """

        results = {}

        for var in candidate_vars:

            # Correlation with treatment
            corr_t, p_t = pearsonr(df[var], df[treatment])

            # Correlation with outcome
            corr_y, p_y = pearsonr(df[var], df[outcome])

            is_proxy = (
                abs(corr_t) > self.corr_threshold and
                abs(corr_y) > self.corr_threshold and
                p_t < self.p_threshold and
                p_y < self.p_threshold
            )

            results[var] = {
                "corr_with_treatment": corr_t,
                "p_treatment": p_t,
                "corr_with_outcome": corr_y,
                "p_outcome": p_y,
                "is_proxy_confounder": is_proxy
            }

        return results