"""
Interaction Confounder Detector
-------------------------------

Detects potential confounders that influence the outcome through
interaction effects with the treatment.

The detector evaluates whether interaction terms between candidate
variables and the treatment significantly affect the outcome.

Author: Confound Hunter Project
"""

from typing import Dict, List

import pandas as pd
import statsmodels.api as sm


class InteractionConfounderDetector:
    """
    Detect confounders using interaction effects.
    """

    def __init__(self, p_threshold: float = 0.05):
        """
        Initialize detector.

        Parameters
        ----------
        p_threshold : float
            Statistical significance threshold.
        """

        self.p_threshold = p_threshold

    def detect(
        self,
        df: pd.DataFrame,
        treatment: str,
        outcome: str,
        candidate_vars: List[str]
    ) -> Dict[str, Dict]:
        """
        Detect interaction-based confounders.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataset.
        treatment : str
            Treatment variable.
        outcome : str
            Outcome variable.
        candidate_vars : List[str]
            Variables to test for interaction effects.

        Returns
        -------
        Dict[str, Dict]
            Detection results for each variable.
        """

        results = {}

        y = df[outcome]

        for var in candidate_vars:

            # Create interaction feature
            interaction_term = df[treatment] * df[var]

            X = pd.DataFrame({
                "treatment": df[treatment],
                "variable": df[var],
                "interaction": interaction_term
            })

            X = sm.add_constant(X)

            # Fit regression model
            model = sm.OLS(y, X).fit()

            p_value = model.pvalues["interaction"]
            coef = model.params["interaction"]

            is_interaction_confounder = p_value < self.p_threshold

            results[var] = {
                "interaction_coefficient": coef,
                "interaction_p_value": p_value,
                "is_interaction_confounder": is_interaction_confounder
            }

        return results