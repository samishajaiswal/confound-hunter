"""
Audit report object used to store and export audit results.
"""

from typing import Dict, List, Any
import pandas as pd
import json


class AuditReport:
    """
    Structured output of ConfounderAudit.
    """

    def __init__(
        self,
        suspicion_scores: Dict[str, float],
        flagged_features: List[Dict[str, Any]],
        evidence_trail: Dict[str, Dict[str, float]],
    ):
        self.suspicion_scores = suspicion_scores
        self.flagged_features = flagged_features
        self.evidence_trail = evidence_trail

    def to_dataframe(self) -> pd.DataFrame:
        """Return results as a pandas DataFrame."""

        df = pd.DataFrame(
            [
                {
                    "feature": f,
                    "score": score,
                    **self.evidence_trail[f],
                }
                for f, score in self.suspicion_scores.items()
            ]
        )

        return df.sort_values("score", ascending=False)

    def to_json(self, path: str) -> None:
        """Save report as JSON."""

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