"""
Classification Metrics for Legal NLP Research Framework.
"""

import logging
from typing import Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)

logger = logging.getLogger(__name__)


class ClassificationMetrics:
    """Computes standard multi-class and multi-label classification metrics."""

    def __init__(self, config: dict):
        self.config = config
        metrics_cfg = config.get("evaluation", {}).get("metrics", [])
        self.requested_metrics = metrics_cfg or [
            "accuracy", "f1_macro", "f1_micro", "precision_macro", "recall_macro"
        ]

    def compute(
        self,
        y_true: list,
        y_pred: list,
        y_scores: Optional[np.ndarray] = None,
        label_names: Optional[list] = None,
    ) -> dict:
        """
        Compute all requested metrics.

        Args:
            y_true: ground truth labels (int or list for multi-label)
            y_pred: predicted labels
            y_scores: optional predicted probabilities (N, C)
            label_names: optional list of class name strings

        Returns:
            dict of metric_name → value
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        results = {}

        if "accuracy" in self.requested_metrics:
            results["accuracy"] = float(accuracy_score(y_true, y_pred))

        if "f1_macro" in self.requested_metrics:
            results["f1_macro"] = float(
                f1_score(y_true, y_pred, average="macro", zero_division=0)
            )
        if "f1_micro" in self.requested_metrics:
            results["f1_micro"] = float(
                f1_score(y_true, y_pred, average="micro", zero_division=0)
            )
        if "f1_weighted" in self.requested_metrics:
            results["f1_weighted"] = float(
                f1_score(y_true, y_pred, average="weighted", zero_division=0)
            )
        if "precision_macro" in self.requested_metrics:
            results["precision_macro"] = float(
                precision_score(y_true, y_pred, average="macro", zero_division=0)
            )
        if "recall_macro" in self.requested_metrics:
            results["recall_macro"] = float(
                recall_score(y_true, y_pred, average="macro", zero_division=0)
            )

        if "roc_auc" in self.requested_metrics and y_scores is not None:
            try:
                from sklearn.metrics import roc_auc_score
                if y_scores.ndim == 2 and y_scores.shape[1] > 2:
                    results["roc_auc"] = float(
                        roc_auc_score(y_true, y_scores, multi_class="ovr", average="macro")
                    )
                elif y_scores.ndim == 2:
                    results["roc_auc"] = float(
                        roc_auc_score(y_true, y_scores[:, 1])
                    )
            except Exception as e:
                results["roc_auc"] = None
                logger.debug(f"ROC-AUC skipped: {e}")

        # Per-class breakdown
        try:
            results["classification_report"] = classification_report(
                y_true, y_pred, target_names=label_names, zero_division=0, output_dict=True
            )
        except Exception:
            pass

        # Confusion matrix (compact)
        try:
            cm = confusion_matrix(y_true, y_pred)
            results["confusion_matrix"] = cm.tolist()
        except Exception:
            pass

        return results

    def compare_results(self, results_dict: dict) -> dict:
        """
        Compare multiple model results side-by-side.

        Args:
            results_dict: {model_name: metrics_dict}

        Returns:
            comparison dict with rankings
        """
        summary = {}
        for key_metric in ["accuracy", "f1_macro", "f1_micro"]:
            scores = {
                name: res.get(key_metric, 0.0)
                for name, res in results_dict.items()
                if isinstance(res.get(key_metric), (int, float))
            }
            ranked = sorted(scores.items(), key=lambda x: -x[1])
            summary[key_metric] = {
                "scores": scores,
                "ranking": [name for name, _ in ranked],
                "best": ranked[0][0] if ranked else None,
            }
        return summary
