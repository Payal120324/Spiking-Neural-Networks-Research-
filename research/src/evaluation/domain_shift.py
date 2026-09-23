"""
Domain Shift Evaluation.
Measures how models trained on a source legal domain generalize
to target legal domains.
Addresses Research Question 3 and Hypothesis H4.
"""

import logging

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

logger = logging.getLogger(__name__)


class DomainShiftEvaluator:
    """
    Evaluates model robustness under domain shift by testing models
    trained on a source dataset on different target datasets.
    """

    def __init__(self, config: dict):
        self.config = config
        ds_cfg = config.get("evaluation", {}).get("domain_shift", {})
        self.source_dataset = ds_cfg.get("source_dataset", "case_hold")
        self.target_datasets = ds_cfg.get("target_datasets", ["ecthr_a", "ecthr_b"])

    def evaluate_transfer(
        self,
        model,
        source_name: str,
        target_name: str,
        source_embeddings: np.ndarray,
        source_labels: np.ndarray,
        target_embeddings: np.ndarray,
        target_labels: np.ndarray,
        model_type: str = "transformer",
    ) -> dict:
        """
        Evaluate zero-shot transfer from source to target domain.
        Fits on a source TRAIN split and evaluates "source_accuracy" on a
        held-out source split (NOT the data the classifier was fit on),
        then evaluates the same fitted classifier on the target domain.

        [FIX, 2026-09]: the previous implementation evaluated source_accuracy
        on the same rows used to fit `clf` (train/test leakage), which caused
        source_accuracy to saturate at ~1.0 for every model type regardless
        of encoding, making accuracy_drop == 1 - target_accuracy for all
        models. This held-out split makes source_accuracy an honest estimate
        of in-domain generalisation.

        Returns:
            dict with source performance, target performance, and drop metrics
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        from sklearn.model_selection import train_test_split

        logger.info(f"Domain shift: {source_name} → {target_name} [{model_type}]")

        le = LabelEncoder()
        y_src_all = le.fit_transform([str(l) for l in source_labels])

        # Hold out a source-side validation split BEFORE fitting, so that
        # source_accuracy measures generalisation, not memorisation.
        n_total = len(y_src_all)
        test_frac = 0.3
        try:
            X_fit_idx, X_eval_idx = train_test_split(
                np.arange(n_total),
                test_size=test_frac,
                random_state=42,
                stratify=y_src_all if len(set(y_src_all)) > 1 else None,
            )
        except ValueError:
            # Fallback if stratification is infeasible (e.g. singleton classes)
            X_fit_idx, X_eval_idx = train_test_split(
                np.arange(n_total), test_size=test_frac, random_state=42
            )

        source_embeddings = np.asarray(source_embeddings)
        scaler = StandardScaler()
        X_fit = scaler.fit_transform(source_embeddings[X_fit_idx])
        X_src_eval = scaler.transform(source_embeddings[X_eval_idx])
        y_fit = y_src_all[X_fit_idx]
        y_src_eval = y_src_all[X_eval_idx]

        clf = LogisticRegression(max_iter=500, C=1.0)
        clf.fit(X_fit, y_fit)

        # Source eval: held-out source split, NOT the fitting data
        y_src_pred = clf.predict(X_src_eval)
        src_acc = float(accuracy_score(y_src_eval, y_src_pred))
        src_f1 = float(f1_score(y_src_eval, y_src_pred, average="macro", zero_division=0))

        # Target eval: align label space
        X_tgt = scaler.transform(target_embeddings)
        y_tgt_raw = [str(l) for l in target_labels]

        # Handle unseen labels
        known = set(le.classes_)
        y_tgt_mapped = [y if y in known else le.classes_[0] for y in y_tgt_raw]
        y_tgt = le.transform(y_tgt_mapped)

        y_tgt_pred = clf.predict(X_tgt)
        tgt_acc = float(accuracy_score(y_tgt, y_tgt_pred))
        tgt_f1 = float(f1_score(y_tgt, y_tgt_pred, average="macro", zero_division=0))

        # H-score (Harmonic mean of source and target performance)
        h_score = (
            2 * src_acc * tgt_acc / (src_acc + tgt_acc)
            if (src_acc + tgt_acc) > 0
            else 0.0
        )

        result = {
            "source": source_name,
            "target": target_name,
            "model_type": model_type,
            "source_accuracy": src_acc,
            "source_f1_macro": src_f1,
            "target_accuracy": tgt_acc,
            "target_f1_macro": tgt_f1,
            "accuracy_drop": float(src_acc - tgt_acc),
            "f1_drop": float(src_f1 - tgt_f1),
            "relative_drop_pct": float((src_acc - tgt_acc) / max(src_acc, 1e-9) * 100),
            "h_score": h_score,
        }
        logger.info(
            f"  Source acc={src_acc:.4f}, Target acc={tgt_acc:.4f}, "
            f"Drop={result['accuracy_drop']:.4f}"
        )
        return result

    def compare_models_on_domain_shift(
        self,
        results: list[dict],
    ) -> dict:
        """
        Aggregate domain shift results across models and compute rankings.

        Args:
            results: list of result dicts from evaluate_transfer()

        Returns:
            summary dict
        """
        by_model = {}
        for r in results:
            key = f"{r['model_type']}"
            if key not in by_model:
                by_model[key] = []
            by_model[key].append(r)

        summary = {}
        for model_key, model_results in by_model.items():
            avg_drop = np.mean([r["accuracy_drop"] for r in model_results])
            avg_target = np.mean([r["target_accuracy"] for r in model_results])
            avg_h = np.mean([r["h_score"] for r in model_results])
            summary[model_key] = {
                "avg_accuracy_drop": float(avg_drop),
                "avg_target_accuracy": float(avg_target),
                "avg_h_score": float(avg_h),
                "n_transfers": len(model_results),
                "per_transfer": model_results,
            }

        # Rank by robustness (lowest accuracy drop)
        ranked = sorted(summary.items(), key=lambda x: x[1]["avg_accuracy_drop"])
        summary["_ranking_by_robustness"] = [k for k, _ in ranked]

        return summary

    def analyze_embedding_shift(
        self,
        source_embeddings: np.ndarray,
        target_embeddings: np.ndarray,
        source_name: str,
        target_name: str,
    ) -> dict:
        """
        Measure distributional shift between source and target embeddings
        using Maximum Mean Discrepancy (MMD) and centroid distance.
        """
        from sklearn.metrics.pairwise import rbf_kernel

        n = min(200, len(source_embeddings), len(target_embeddings))
        src = source_embeddings[:n]
        tgt = target_embeddings[:n]

        # Centroid shift
        src_centroid = src.mean(axis=0)
        tgt_centroid = tgt.mean(axis=0)
        centroid_dist = float(np.linalg.norm(src_centroid - tgt_centroid))

        # Cosine distance between centroids
        cos_sim = float(
            np.dot(src_centroid, tgt_centroid)
            / (np.linalg.norm(src_centroid) * np.linalg.norm(tgt_centroid) + 1e-9)
        )

        # Simplified MMD (Gaussian kernel)
        try:
            gamma = 1.0 / src.shape[1]
            K_ss = rbf_kernel(src, src, gamma=gamma).mean()
            K_tt = rbf_kernel(tgt, tgt, gamma=gamma).mean()
            K_st = rbf_kernel(src, tgt, gamma=gamma).mean()
            mmd = float(K_ss + K_tt - 2 * K_st)
        except Exception:
            mmd = None

        # Variance ratio
        src_var = float(src.var(axis=0).mean())
        tgt_var = float(tgt.var(axis=0).mean())

        return {
            "source": source_name,
            "target": target_name,
            "centroid_distance": centroid_dist,
            "centroid_cosine_similarity": cos_sim,
            "mmd": mmd,
            "source_variance": src_var,
            "target_variance": tgt_var,
            "variance_ratio": float(tgt_var / max(src_var, 1e-9)),
        }