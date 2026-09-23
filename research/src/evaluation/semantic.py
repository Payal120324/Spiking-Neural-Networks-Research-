"""
Semantic Preservation Analysis.
Tests whether spike-encoded representations preserve the semantic
structure of the original transformer embeddings.
Addresses Research Question 2 and Hypothesis H2.
"""

import logging

import numpy as np
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class SemanticPreservation:
    """
    Measures how well spike encoding preserves the semantic geometry
    of transformer embeddings.

    Metrics:
      - Mean cosine similarity (original vs decoded)
      - Spearman rank correlation of pairwise distances
      - Kendall-τ rank correlation
      - Top-k nearest neighbor overlap
    """

    def __init__(self, config: dict):
        eval_cfg = config.get("evaluation", {})
        sem_cfg = eval_cfg.get("semantic", {})
        self.top_k_values = sem_cfg.get("top_k", [5, 10, 20])

    def analyze(
        self,
        original_embeddings: np.ndarray,
        decoded_reconstruction: np.ndarray,
        encoding_name: str = "unknown",
        n_samples: int = 200,
    ) -> dict:
        """
        Compute semantic preservation metrics.

        Args:
            original_embeddings:    (N, D) transformer embeddings
            decoded_reconstruction: (N, D) estimate produced by the
                                     encoding's own `decode()` method
                                     (already collapsed over time and,
                                     for expanding encodings such as
                                     population coding, already mapped
                                     back to D dimensions)
            encoding_name:           name of the encoding method
            n_samples:               max samples to use (subsample if larger)

        Returns:
            dict of preservation metrics
        """
        N = min(n_samples, len(original_embeddings), len(decoded_reconstruction))
        idx = np.random.choice(len(original_embeddings), N, replace=False)
        emb_raw = original_embeddings[idx].astype(np.float32)
        dec = decoded_reconstruction[idx].astype(np.float32)

        # Compare on the SAME scale the encoder actually operated on:
        # every encoder in this codebase min-max normalizes each sample
        # to [0, 1] before spiking, so comparing decoded output against
        # the raw (roughly zero-mean, unbounded) embedding was an
        # apples-to-oranges scale/shift mismatch. Normalize here too.
        emb = self._normalize_minmax(emb_raw)

        # Safety net: trim to matching dims if a decoder ever returns
        # a different width than the embedding (should not normally
        # happen now that decode() maps back to D dimensions).
        if dec.shape[1] != emb.shape[1]:
            min_dim = min(dec.shape[1], emb.shape[1])
            emb = emb[:, :min_dim]
            dec = dec[:, :min_dim]

        results = {
            "encoding": encoding_name,
            "n_samples": N,
        }

        # 1. Mean cosine similarity between paired samples
        cos_sims = self._pairwise_cosine(emb, dec)
        results["mean_cosine_similarity"] = float(cos_sims.mean())
        results["std_cosine_similarity"] = float(cos_sims.std())

        # 2. Pairwise distance correlation (Spearman)
        spearman_rho, spearman_p = self._distance_rank_correlation(emb, dec, method="spearman")
        results["spearman_rho"] = float(spearman_rho)
        results["spearman_p"] = float(spearman_p)

        # 3. Kendall-τ correlation
        kendall_tau, kendall_p = self._distance_rank_correlation(emb, dec, method="kendall")
        results["kendall_tau"] = float(kendall_tau)
        results["kendall_p"] = float(kendall_p)

        # 4. Top-k nearest-neighbor overlap
        for k in self.top_k_values:
            overlap = self._topk_nn_overlap(emb, dec, k=k)
            results[f"topk_nn_overlap_k{k}"] = float(overlap)

        # 5. Reconstruction quality proxy (normalized MSE)
        emb_n = self._unit_normalize(emb)
        dec_n = self._unit_normalize(dec)
        mse = float(np.mean((emb_n - dec_n) ** 2))
        results["normalized_mse"] = mse

        logger.info(
            f"[{encoding_name}] cosine={results['mean_cosine_similarity']:.4f} "
            f"spearman={spearman_rho:.4f} "
            f"kendall={kendall_tau:.4f}"
        )
        return results

    def compare_encodings(
        self,
        original_embeddings: np.ndarray,
        decoded_dict: dict,
        n_samples: int = 200,
    ) -> dict:
        """
        Compare multiple spike encodings on semantic preservation.

        Args:
            decoded_dict: {encoding_name: decoded_reconstruction_array},
                           where each array is that encoding's own
                           decode() output (see analyze())

        Returns:
            {encoding_name: metrics_dict}
        """
        results = {}
        for enc_name, dec in decoded_dict.items():
            results[enc_name] = self.analyze(
                original_embeddings, dec, enc_name, n_samples
            )
        return results

    # ─────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _normalize_minmax(x: np.ndarray) -> np.ndarray:
        """Per-sample min-max normalize to [0, 1] (mirrors BaseSpikeEncoder._normalize)."""
        mins = x.min(axis=-1, keepdims=True)
        maxs = x.max(axis=-1, keepdims=True)
        denom = maxs - mins
        denom = np.where(denom == 0, 1.0, denom)
        return (x - mins) / denom

    @staticmethod
    def _pairwise_cosine(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Element-wise cosine similarity between matched rows."""
        a_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
        b_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
        return (a_norm * b_norm).sum(axis=1)

    @staticmethod
    def _pairwise_distances(X: np.ndarray) -> np.ndarray:
        """Compute upper-triangle of pairwise Euclidean distance matrix."""
        n = X.shape[0]
        idx = np.triu_indices(n, k=1)
        diffs = X[idx[0]] - X[idx[1]]
        return np.sqrt((diffs ** 2).sum(axis=1))

    def _distance_rank_correlation(
        self, emb: np.ndarray, spk: np.ndarray, method: str = "spearman"
    ) -> tuple:
        """Rank correlation between pairwise distance distributions."""
        # Subsample to at most 100 samples for tractability
        n = min(100, len(emb))
        idx = np.random.choice(len(emb), n, replace=False)
        d_emb = self._pairwise_distances(emb[idx])
        d_spk = self._pairwise_distances(spk[idx])
        if method == "spearman":
            r, p = spearmanr(d_emb, d_spk)
        else:
            r, p = kendalltau(d_emb, d_spk)
        return float(r), float(p)

    @staticmethod
    def _topk_nn_overlap(emb: np.ndarray, spk: np.ndarray, k: int = 10) -> float:
        """
        Fraction of true top-k neighbors that appear in spike-space top-k.
        Averaged over all samples.
        """
        n = min(200, len(emb))
        idx = np.random.choice(len(emb), n, replace=False)
        emb_s = emb[idx]
        spk_s = spk[idx]

        cos_emb = cosine_similarity(emb_s)
        cos_spk = cosine_similarity(spk_s)
        np.fill_diagonal(cos_emb, -np.inf)
        np.fill_diagonal(cos_spk, -np.inf)

        overlaps = []
        for i in range(n):
            true_nn = set(np.argsort(-cos_emb[i])[:k])
            pred_nn = set(np.argsort(-cos_spk[i])[:k])
            overlaps.append(len(true_nn & pred_nn) / k)
        return float(np.mean(overlaps))

    @staticmethod
    def _unit_normalize(X: np.ndarray) -> np.ndarray:
        mins = X.min(axis=0, keepdims=True)
        maxs = X.max(axis=0, keepdims=True)
        denom = maxs - mins
        denom[denom == 0] = 1.0
        return (X - mins) / denom
