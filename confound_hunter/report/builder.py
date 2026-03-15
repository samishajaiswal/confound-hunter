"""
Audit report builder for Confound Hunter.

This module defines the AuditReport class which stores
the results of a confounder audit and provides utilities
to export the results.
"""

from typing import Dict, List, Any
import pandas as pd
import json


class AuditReport:
    """
    Container for results produced by ConfounderAudit.

    Attributes
    ----------
    suspicion_scores : Dict[str, float]
        Final suspicion score per feature.
    flagged_features : List[Dict[str, Any]]
        Features exceeding the suspicion threshold.
    evidence_trail : Dict[str, Dict[str, float]]
        Per-detector scores for each feature.
    """

    def __init__(
        self,
        suspicion_scores: Dict[str, float],
        flagged_features: List[Dict[str, Any]],
        evidence_trail: Dict[str, Dict[str, float]],
    ) -> None:
        self.suspicion_scores = suspicion_scores
        self.flagged_features = flagged_features
        self.evidence_trail = evidence_trail

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert audit results into a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame containing feature scores and detector evidence.
        """

        rows = []

        for feature, score in self.suspicion_scores.items():

            row = {
                "feature": feature,
                "score": score,
            }

            row.update(self.evidence_trail.get(feature, {}))

            rows.append(row)

        df = pd.DataFrame(rows)

        return df.sort_values("score", ascending=False)

    def to_json(self, path: str) -> None:
        """
        Save audit results as a JSON file.

        Parameters
        ----------
        path : str
            Destination path for the JSON file.
        """

        with open(path, "w") as f:

            json.dump(
                {
                    "scores": self.suspicion_scores,
                    "flagged_features": self.flagged_features,
                    "evidence_trail": self.evidence_trail,
                },
                f,
                indent=4,
            )