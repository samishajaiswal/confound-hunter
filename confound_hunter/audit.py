"""
Core audit engine for Confound Hunter.

This module implements the ConfounderAudit class, which orchestrates
all detectors, aggregates suspicion scores, and generates structured
audit reports.

Author: Confound Hunter
"""

from __future__ import annotations

from typing import Dict, List, Any
import numpy as np
import pandas as pd

from confound_hunter.report.builder import AuditReport


class ConfounderAudit:
    """
    Main interface for running confounder detection audits.

    Parameters
    ----------
    model : object
        Trained machine learning model (sklearn-compatible).
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series
        Training target vector.
    X_test : pd.DataFrame
        Test feature matrix.
    y_test : pd.Series
        Test target vector.
    """

    # detector weights (from design spec)
    DETECTOR_WEIGHTS = {
        "permutation": 0.20,
        "shap": 0.20,
        "residual": 0.20,
        "proxy": 0.15,
        "temporal": 0.15,
        "interaction": 0.10,
    }

    def __init__(
        self,
        model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> None:
        self.model = model

        self.X_train = X_train
        self.y_train = y_train

        self.X_test = X_test
        self.y_test = y_test

        self.features = list(X_train.columns)

        # stores detector results
        self.detector_scores: Dict[str, Dict[str, float]] = {}

        # final suspicion scores
        self.suspicion_scores: Dict[str, float] = {}

        # detector evidence per feature
        self.evidence_trail: Dict[str, Dict[str, float]] = {}

    # ---------------------------------------------------------
    # PUBLIC API
    # ---------------------------------------------------------

    def run(self, threshold: float = 0.5) -> AuditReport:
        """
        Run all detectors and generate an audit report.

        Parameters
        ----------
        threshold : float
            Suspicion score threshold for flagging features.

        Returns
        -------
        AuditReport
            Structured report object containing results.
        """

        self._run_detectors()

        self._aggregate_scores()

        flagged = self._flag_features(threshold)

        report = AuditReport(
            suspicion_scores=self.suspicion_scores,
            flagged_features=flagged,
            evidence_trail=self.evidence_trail,
        )

        return report

    # ---------------------------------------------------------
    # DETECTOR ORCHESTRATION
    # ---------------------------------------------------------

    def _run_detectors(self) -> None:
        """
        Execute all registered detectors.

        Each detector returns:
        {
            feature_name: score_between_0_and_1
        }
        """

        from confound_hunter.detectors.permutation import permutation_stability
        from confound_hunter.detectors.shap_drift import shap_train_test_consistency
        from confound_hunter.detectors.residual_corr import residual_correlation
        from confound_hunter.detectors.proxy import proxy_confounder
        from confound_hunter.detectors.temporal import temporal_confounder
        from confound_hunter.detectors.interaction import interaction_confounder

        self.detector_scores["permutation"] = permutation_stability(
            self.model,
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
        )

        self.detector_scores["shap"] = shap_train_test_consistency(
            self.model,
            self.X_train,
            self.X_test,
        )

        self.detector_scores["residual"] = residual_correlation(
            self.model,
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
        )

        self.detector_scores["proxy"] = proxy_confounder(
            self.X_train,
            self.y_train,
        )

        self.detector_scores["temporal"] = temporal_confounder(
            self.X_train,
            self.y_train,
        )

        self.detector_scores["interaction"] = interaction_confounder(
            self.model,
            self.X_train,
            self.y_train,
        )

    # ---------------------------------------------------------
    # AGGREGATION
    # ---------------------------------------------------------

    def _aggregate_scores(self) -> None:
        """
        Combine detector outputs into a single suspicion score.
        """

        for feature in self.features:

            weighted_sum = 0.0
            evidence = {}

            for detector, weight in self.DETECTOR_WEIGHTS.items():

                score = self.detector_scores.get(detector, {}).get(feature, 0.0)

                score = max(0.0, min(1.0, score))
                weighted_sum += score * weight

                evidence[detector] = score

            self.suspicion_scores[feature] = round(weighted_sum, 4)

            self.evidence_trail[feature] = evidence

    # ---------------------------------------------------------
    # FLAGGING LOGIC
    # ---------------------------------------------------------

    def _flag_features(self, threshold: float) -> List[Dict[str, Any]]:
        """
        Identify features exceeding the suspicion threshold.

        Returns
        -------
        list
            Sorted list of flagged feature dictionaries.
        """

        flagged = []

        for feature, score in self.suspicion_scores.items():

            if score >= threshold:

                flagged.append(
                    {
                        "feature": feature,
                        "score": score,
                        "type": self._classify_feature(feature),
                    }
                )

        flagged.sort(key=lambda x: x["score"], reverse=True)

        return flagged

    # ---------------------------------------------------------
    # CONFONDER TYPE CLASSIFICATION
    # ---------------------------------------------------------

    def _classify_feature(self, feature: str) -> str:
        """
        Assign human-readable confounder type based on detector pattern.
        """

        evidence = self.evidence_trail.get(feature, {})

        perm = evidence.get("permutation", 0)
        shap = evidence.get("shap", 0)
        resid = evidence.get("residual", 0)
        proxy = evidence.get("proxy", 0)
        temporal = evidence.get("temporal", 0)
        interaction = evidence.get("interaction", 0)

        if perm > 0.6 and shap > 0.6:
            return "Train-Overfit Spurious"

        if resid > 0.6 and proxy < 0.3:
            return "Noise Absorber"

        if proxy > 0.6:
            return "Proxy Confounder"

        if temporal > 0.6:
            return "Temporal Confounder"

        if interaction > 0.6:
            return "Interaction Confounder"

        if sum([perm, shap, resid, proxy]) > 2.0:
            return "Systemic Leakage"

        return "Weak Signal / Monitor"