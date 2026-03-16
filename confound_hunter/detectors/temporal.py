"""
Temporal Confounder Detector
----------------------------

Identifies potential confounders based on temporal ordering.

A valid confounder must occur before both the treatment and the
outcome in time. Variables that occur after treatment cannot
causally influence it and therefore cannot be confounders.

Author: Confound Hunter Project
"""

from typing import Dict, List

import pandas as pd


class TemporalConfounderDetector:
    """
    Detect confounders using temporal ordering of variables.
    """

    def __init__(self):
        """Initialize the temporal detector."""
        pass

    def detect(
        self,
        df: pd.DataFrame,
        treatment: str,
        outcome: str,
        candidate_vars: List[str],
        time_columns: Dict[str, str]
    ) -> Dict[str, Dict]:
        """
        Detect temporal confounders.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataset.
        treatment : str
            Treatment variable name.
        outcome : str
            Outcome variable name.
        candidate_vars : List[str]
            Variables to evaluate as potential confounders.
        time_columns : Dict[str, str]
            Mapping of variables to their timestamp columns.

            Example:
            {
                "age": "age_recorded_time",
                "income": "income_time"
            }

        Returns
        -------
        Dict[str, Dict]
            Detection results for each candidate variable.
        """

        results = {}

        treatment_time = df[time_columns[treatment]]
        outcome_time = df[time_columns[outcome]]

        for var in candidate_vars:

            if var not in time_columns:
                continue

            var_time = df[time_columns[var]]

            # Check if variable occurs before treatment
            before_treatment = (var_time < treatment_time).all()

            # Check if variable occurs before outcome
            before_outcome = (var_time < outcome_time).all()

            is_temporal_confounder = before_treatment and before_outcome

            results[var] = {
                "before_treatment": before_treatment,
                "before_outcome": before_outcome,
                "is_temporal_confounder": is_temporal_confounder
            }

        return results