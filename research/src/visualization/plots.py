"""
Publication-Quality Visualization for Legal NLP Research Framework.
Generates all figures required for the research paper.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

logger = logging.getLogger(__name__)


def _setup_style(style: str = "publication"):
    """Apply matplotlib style for publication-quality figures."""
    if style == "publication":
        plt.rcParams.update({
            "font.family": "serif",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.dpi": 150,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
        })
    sns.set_palette("colorblind")


class ResearchPlotter:
    """Generates all publication-quality figures for the research paper."""

    def __init__(self, config: dict):
        self.config = config
        vis_cfg = config.get("visualization", {})
        self.dpi = vis_cfg.get("dpi", 300)
        self.fmt = vis_cfg.get("format", "pdf")
        self.style = vis_cfg.get("style", "publication")
        self.fig_w = vis_cfg.get("figure_width", 8)
        self.fig_h = vis_cfg.get("figure_height", 6)
        self.out_dir = Path(config.get("storage", {}).get("figures", "storage/results/figures"))
        self.out_dir.mkdir(parents=True, exist_ok=True)
        _setup_style(self.style)

    # ─────────────────────────────────────────────────────────────────
    # Classification Performance
    # ─────────────────────────────────────────────────────────────────

    def plot_classification_comparison(
        self,
        results: dict,
        metric: str = "f1_macro",
        title: str = "Classification Performance",
        filename: str = "classification_comparison",
    ) -> str:
        """
        Bar chart comparing classification performance across models and encodings.

        Args:
            results: {model_name: {metric: value}}
        """
        fig, ax = plt.subplots(figsize=(self.fig_w, self.fig_h))
        names = list(results.keys())
        
        # Extract numeric values, handling nested dict structures
        values = []
        for n in names:
            val = results[n].get(metric, 0.0)
            # If the value is a dict, try to get the metric from it
            if isinstance(val, dict):
                val = val.get(metric, 0.0)
            # Ensure we have a numeric value
            try:
                values.append(float(val))
            except (TypeError, ValueError):
                values.append(0.0)

        colors = sns.color_palette("colorblind", len(names))
        bars = ax.bar(range(len(names)), values, color=colors, edgecolor="black", linewidth=0.5)

        # Value labels on bars
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=9,
            )

        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=30, ha="right")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(title)
        ymax = max(values) if values else 0
        ax.set_ylim(0, min(1.0, ymax * 1.2) if ymax > 0 else 1.0)

        plt.tight_layout()
        path = self._save(fig, filename)
        plt.close(fig)
        return path

    def plot_grouped_classification(
        self,
        results_by_model: dict,
        datasets: list,
        metric: str = "f1_macro",
        filename: str = "grouped_classification",
    ) -> str:
        """
        Grouped bar chart: models × datasets.

        Args:
            results_by_model: {model_name: {dataset: {metric: value}}}
        """
        models = list(results_by_model.keys())
        n_models = len(models)
        n_datasets = len(datasets)
        width = 0.8 / n_models

        fig, ax = plt.subplots(figsize=(self.fig_w + 2, self.fig_h))
        colors = sns.color_palette("colorblind", n_models)

        for i, model in enumerate(models):
            x_positions = np.arange(n_datasets) + i * width
            values = [
                results_by_model[model].get(ds, {}).get(metric, 0.0)
                for ds in datasets
            ]
            ax.bar(x_positions, values, width=width * 0.9, color=colors[i],
                   label=model, edgecolor="black", linewidth=0.4)

        ax.set_xticks(np.arange(n_datasets) + (n_models - 1) * width / 2)
        ax.set_xticklabels(datasets, rotation=20, ha="right")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"Performance Comparison — {metric.replace('_', ' ').title()}")
        ax.legend(loc="upper right", framealpha=0.9)
        ax.set_ylim(0, 1.0)

        plt.tight_layout()
        path = self._save(fig, filename)
        plt.close(fig)
        return path

    # ─────────────────────────────────────────────────────────────────
    # Spike Encoding Visualizations
    # ─────────────────────────────────────────────────────────────────

    def plot_spike_raster(
        self,
        spike_trains: np.ndarray,
        encoding_name: str,
        sample_idx: int = 0,
        n_neurons: int = 50,
        filename: Optional[str] = None,
    ) -> str:
        """
        Raster plot of spike train for a single sample.

        Args:
            spike_trains: (N, T, F)
        """
        spk = spike_trains[sample_idx, :, :n_neurons]  # (T, n_neurons)
        T, n = spk.shape

        fig, ax = plt.subplots(figsize=(self.fig_w, self.fig_h * 0.8))

        for neuron in range(n):
            spike_times = np.where(spk[:, neuron])[0]
            ax.scatter(spike_times, np.full_like(spike_times, neuron),
                      s=4, color="black", marker="|", linewidths=0.5)

        ax.set_xlabel("Time Step")
        ax.set_ylabel("Neuron (Feature Index)")
        ax.set_title(f"Spike Raster — {encoding_name.replace('_', ' ').title()}")
        ax.set_xlim(-0.5, T - 0.5)
        ax.set_ylim(-0.5, n - 0.5)

        sparsity = 1.0 - spk.mean()
        ax.text(0.98, 0.02, f"Sparsity: {sparsity:.2%}",
               transform=ax.transAxes, ha="right", va="bottom",
               fontsize=9, color="gray")

        plt.tight_layout()
        fname = filename or f"raster_{encoding_name}"
        path = self._save(fig, fname)
        plt.close(fig)
        return path

    def plot_encoding_comparison_rasters(
        self, spike_trains_dict: dict, sample_idx: int = 0, n_neurons: int = 50
    ) -> str:
        """
        Multi-panel raster comparison of all encodings for one sample.
        """
        encodings = list(spike_trains_dict.keys())
        n_enc = len(encodings)
        fig, axes = plt.subplots(n_enc, 1, figsize=(self.fig_w, 2.5 * n_enc), sharex=True)
        if n_enc == 1:
            axes = [axes]

        for ax, enc_name in zip(axes, encodings):
            spk = spike_trains_dict[enc_name][sample_idx, :, :n_neurons]  # (T, n)
            T, n = spk.shape
            for neuron in range(n):
                times = np.where(spk[:, neuron])[0]
                ax.scatter(times, np.full_like(times, neuron), s=3,
                          color="steelblue", marker="|", linewidths=0.5)
            sparsity = 1.0 - spk.mean()
            ax.set_ylabel(enc_name.replace("_", "\n").title(), fontsize=9)
            ax.set_ylim(-0.5, n - 0.5)
            ax.text(0.99, 0.85, f"sp={sparsity:.2%}",
                   transform=ax.transAxes, ha="right", va="top",
                   fontsize=8, color="gray")

        axes[-1].set_xlabel("Time Step")
        fig.suptitle("Spike Train Comparison — All Encodings", y=1.01, fontsize=13)
        plt.tight_layout()
        path = self._save(fig, "encoding_comparison_rasters")
        plt.close(fig)
        return path

    def plot_firing_rates(self, spike_trains_dict: dict, filename: str = "firing_rates") -> str:
        """Violin plot of firing rates across encodings."""
        fig, ax = plt.subplots(figsize=(self.fig_w, self.fig_h))
        data = []
        labels = []
        for enc_name, spk in spike_trains_dict.items():
            rates = spk.mean(axis=(1, 2))  # avg firing rate per sample
            data.append(rates)
            labels.append(enc_name.replace("_", "\n"))

        parts = ax.violinplot(data, showmedians=True, showextrema=True)
        for pc in parts["bodies"]:
            pc.set_alpha(0.7)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel("Average Firing Rate")
        ax.set_title("Firing Rate Distribution by Encoding Method")
        plt.tight_layout()
        path = self._save(fig, filename)
        plt.close(fig)
        return path

    # ─────────────────────────────────────────────────────────────────
    # Semantic Preservation
    # ─────────────────────────────────────────────────────────────────

    def plot_semantic_preservation(
        self,
        semantic_results: dict,
        filename: str = "semantic_preservation",
    ) -> str:
        """
        Radar chart + bar chart for semantic preservation metrics.
        """
        encodings = list(semantic_results.keys())
        metrics = [
            ("mean_cosine_similarity", "Cosine Sim."),
            ("spearman_rho", "Spearman ρ"),
            ("kendall_tau", "Kendall τ"),
        ]
        # Add top-k metrics
        for k in [5, 10]:
            key = f"topk_nn_overlap_k{k}"
            if key in list(semantic_results.values())[0]:
                metrics.append((key, f"Top-{k} NN"))

        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(3 * n_metrics, self.fig_h))
        if n_metrics == 1:
            axes = [axes]

        colors = sns.color_palette("colorblind", len(encodings))

        for ax, (metric_key, metric_label) in zip(axes, metrics):
            values = [semantic_results[enc].get(metric_key, 0.0) for enc in encodings]
            bars = ax.barh(encodings, values, color=colors, edgecolor="black", linewidth=0.4)
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=8,
                )
            ax.set_title(metric_label, fontsize=10)
            xmax = max(values) if values else 0
            ax.set_xlim(0, max(xmax * 1.2, 0.1) if xmax > 0 else 0.1)
            ax.set_xlabel("Score")

        fig.suptitle("Semantic Preservation by Encoding Method", fontsize=13)
        plt.tight_layout()
        path = self._save(fig, filename)
        plt.close(fig)
        return path

    def plot_embedding_scatter(
        self,
        original_embeddings: np.ndarray,
        spike_embeddings: dict,
        labels: np.ndarray,
        filename: str = "embedding_scatter",
        n_samples: int = 300,
    ) -> str:
        """
        UMAP/PCA scatter of original vs spike-encoded embeddings.
        """
        try:
            from sklearn.decomposition import PCA
            n = min(n_samples, len(original_embeddings))
            idx = np.random.choice(len(original_embeddings), n, replace=False)

            all_encodings = {"Original": original_embeddings[idx]}
            for enc_name, spk in spike_embeddings.items():
                all_encodings[enc_name] = spk[idx].mean(axis=1)

            n_plots = len(all_encodings)
            fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4))
            if n_plots == 1:
                axes = [axes]

            unique_labels = np.unique(labels[idx])
            colors = sns.color_palette("tab10", len(unique_labels))

            for ax, (enc_name, emb) in zip(axes, all_encodings.items()):
                pca = PCA(n_components=2, random_state=42)
                proj = pca.fit_transform(emb)
                for i, lbl in enumerate(unique_labels):
                    mask = labels[idx] == lbl
                    ax.scatter(proj[mask, 0], proj[mask, 1],
                              c=[colors[i]], s=15, alpha=0.6, label=str(lbl))
                ax.set_title(enc_name, fontsize=10)
                ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
                ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")

            fig.suptitle("PCA of Embeddings vs Spike-Encoded Representations", fontsize=12)
            plt.tight_layout()
        except Exception as e:
            logger.warning(f"Embedding scatter plot failed: {e}")
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.text(0.5, 0.5, f"Plot unavailable:\n{e}", ha="center", va="center")

        path = self._save(fig, filename)
        plt.close(fig)
        return path

    # ─────────────────────────────────────────────────────────────────
    # Energy Analysis
    # ─────────────────────────────────────────────────────────────────

    def plot_energy_comparison(
        self, energy_results: dict, filename: str = "energy_comparison"
    ) -> str:
        """
        Stacked bar chart: compute vs memory energy for transformer and SNN variants.
        """
        encodings = list(energy_results.keys())
        compute_energies = []
        memory_energies = []
        labels = []

        # Add transformer bar first
        first = energy_results[encodings[0]]
        t_compute = first["transformer_energy"]["compute_energy_pj"] / 1e6  # → µJ
        t_memory = first["transformer_energy"]["memory_energy_pj"] / 1e6

        labels.append("Transformer")
        compute_energies.append(t_compute)
        memory_energies.append(t_memory)

        for enc in encodings:
            snn_e = energy_results[enc]["snn_energy"]
            labels.append(f"SNN\n({enc.replace('_', ' ')})")
            compute_energies.append(snn_e["compute_energy_pj"] / 1e6)
            memory_energies.append(snn_e["memory_energy_pj"] / 1e6)

        fig, ax = plt.subplots(figsize=(self.fig_w + 1, self.fig_h))
        x = range(len(labels))
        bars1 = ax.bar(x, compute_energies, label="Compute", color="#4878CF", edgecolor="black", linewidth=0.4)
        bars2 = ax.bar(x, memory_energies, bottom=compute_energies,
                      label="Memory Access", color="#6ACC65", edgecolor="black", linewidth=0.4)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("Energy (µJ per sample)")
        ax.set_title("Energy Consumption: Transformer vs SNN Encodings")
        ax.legend(framealpha=0.9)

        # Savings annotation
        for i, enc in enumerate(encodings, 1):
            ratio = energy_results[enc]["comparison"]["energy_savings_pct"]
            total = compute_energies[i] + memory_energies[i]
            ax.text(i, total + max(compute_energies) * 0.02,
                   f"−{ratio:.0f}%", ha="center", va="bottom", fontsize=8, color="darkred")

        plt.tight_layout()
        path = self._save(fig, filename)
        plt.close(fig)
        return path

    def plot_energy_sparsity_tradeoff(
        self, energy_results: dict, semantic_results: dict, filename: str = "energy_semantic_tradeoff"
    ) -> str:
        """Scatter plot: energy savings vs semantic preservation."""
        fig, ax = plt.subplots(figsize=(self.fig_w * 0.9, self.fig_h))
        colors = sns.color_palette("colorblind", len(energy_results))

        for i, (enc, en_res) in enumerate(energy_results.items()):
            savings = en_res["comparison"]["energy_savings_pct"]
            sem = semantic_results.get(enc, {}).get("mean_cosine_similarity", 0.0)
            sparsity = en_res["snn_energy"]["sparsity"]
            ax.scatter(savings, sem, s=100 + sparsity * 200, color=colors[i],
                      label=enc.replace("_", " "), edgecolors="black", linewidths=0.5, zorder=5)
            ax.annotate(enc.replace("_", " "), (savings, sem),
                       textcoords="offset points", xytext=(5, 5), fontsize=8)

        ax.set_xlabel("Energy Savings vs Transformer (%)")
        ax.set_ylabel("Semantic Preservation (Cosine Similarity)")
        ax.set_title("Energy–Semantic Tradeoff by Encoding Method")
        ax.legend(fontsize=8, loc="lower right")
        plt.tight_layout()
        path = self._save(fig, filename)
        plt.close(fig)
        return path

    # ─────────────────────────────────────────────────────────────────
    # Domain Shift
    # ─────────────────────────────────────────────────────────────────

    def plot_domain_shift_heatmap(
        self, shift_results: dict, filename: str = "domain_shift_heatmap"
    ) -> str:
        """
        Heatmap of accuracy drop under domain shift for each model.
        """
        # Build matrix: rows = models, cols = target datasets
        model_keys = [k for k in shift_results if not k.startswith("_")]
        targets = list({
            r["target"]
            for m in model_keys
            for r in shift_results[m].get("per_transfer", [])
        })

        if not targets:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No domain shift data", ha="center", va="center")
            path = self._save(fig, filename)
            plt.close(fig)
            return path

        matrix = np.zeros((len(model_keys), len(targets)))
        for i, model in enumerate(model_keys):
            for transfer in shift_results[model].get("per_transfer", []):
                if transfer["target"] in targets:
                    j = targets.index(transfer["target"])
                    matrix[i, j] = transfer["accuracy_drop"]

        fig, ax = plt.subplots(figsize=(max(6, len(targets) * 2), max(4, len(model_keys) * 1.2)))
        im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=0.3)
        plt.colorbar(im, ax=ax, label="Accuracy Drop")

        ax.set_xticks(range(len(targets)))
        ax.set_xticklabels(targets, rotation=30, ha="right")
        ax.set_yticks(range(len(model_keys)))
        ax.set_yticklabels(model_keys)
        ax.set_title("Domain Shift — Accuracy Drop (Source → Target)")

        for i in range(len(model_keys)):
            for j in range(len(targets)):
                ax.text(j, i, f"{matrix[i,j]:.3f}", ha="center", va="center", fontsize=8)

        plt.tight_layout()
        path = self._save(fig, filename)
        plt.close(fig)
        return path

    # ─────────────────────────────────────────────────────────────────
    # Dataset Statistics
    # ─────────────────────────────────────────────────────────────────

    def plot_label_distribution(
        self, stats: dict, dataset_name: str, filename: Optional[str] = None
    ) -> str:
        """Bar chart of label distribution in training split."""
        train_stats = stats.get("splits", {}).get("train", {})
        dist = train_stats.get("label_distribution", {})
        if not dist:
            logger.warning("No label distribution data available")
            return ""

        labels = list(dist.keys())[:30]
        counts = [dist[l] for l in labels]

        fig, ax = plt.subplots(figsize=(self.fig_w, self.fig_h))
        ax.bar(range(len(labels)), counts, color=sns.color_palette("colorblind", len(labels)))
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_xlabel("Class Label")
        ax.set_ylabel("Count")
        ax.set_title(f"Label Distribution — {dataset_name} (Train)")
        plt.tight_layout()
        fname = filename or f"label_dist_{dataset_name}"
        path = self._save(fig, fname)
        plt.close(fig)
        return path

    def plot_document_length_distribution(
        self, stats: dict, dataset_name: str, filename: Optional[str] = None
    ) -> str:
        """Histogram of word/token lengths per split."""
        splits = stats.get("splits", {})
        fig, ax = plt.subplots(figsize=(self.fig_w, self.fig_h))
        colors = sns.color_palette("colorblind", len(splits))

        for (split_name, sp), color in zip(splits.items(), colors):
            mean = sp["word_stats"]["mean"]
            std = sp["word_stats"]["std"]
            # Simulate distribution from stats (Gaussian approximation)
            if mean > 0 and std > 0:
                x = np.linspace(max(0, mean - 4 * std), mean + 4 * std, 200)
                from scipy.stats import norm
                y = norm.pdf(x, mean, std)
                ax.plot(x, y, label=split_name, color=color)
                ax.axvline(mean, color=color, linestyle="--", alpha=0.5, linewidth=1)

        ax.axvline(512, color="red", linestyle=":", linewidth=1.5, label="512 token limit (×1.3)")
        ax.set_xlabel("Document Length (words)")
        ax.set_ylabel("Density")
        ax.set_title(f"Document Length Distribution — {dataset_name}")
        ax.legend()
        plt.tight_layout()
        fname = filename or f"doc_length_{dataset_name}"
        path = self._save(fig, fname)
        plt.close(fig)
        return path

    # ─────────────────────────────────────────────────────────────────
    # Training Curves
    # ─────────────────────────────────────────────────────────────────

    def plot_training_curves(
        self, history: list[dict], title: str = "Training Curves",
        filename: str = "training_curves"
    ) -> str:
        """Plot loss and accuracy over training epochs."""
        epochs = [h["epoch"] for h in history]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(self.fig_w, self.fig_h * 0.8))

        if "train_loss" in history[0]:
            ax1.plot(epochs, [h["train_loss"] for h in history], "b-o", label="Train")
        if "val_loss" in history[0]:
            ax1.plot(epochs, [h.get("val_loss", 0) for h in history], "r-s", label="Val")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training Loss")
        ax1.legend()

        for metric, style, color in [("train_acc", "b-o", "blue"), ("val_accuracy", "r-s", "red")]:
            if metric in history[0]:
                ax2.plot(epochs, [h.get(metric, 0) for h in history],
                        style, color=color, label=metric.replace("_", " "))
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Accuracy")
        ax2.legend()

        fig.suptitle(title, fontsize=13)
        plt.tight_layout()
        path = self._save(fig, filename)
        plt.close(fig)
        return path

    # ─────────────────────────────────────────────────────────────────
    # Summary Figure
    # ─────────────────────────────────────────────────────────────────

    def plot_summary_dashboard(
        self,
        classification_results: dict,
        semantic_results: dict,
        energy_results: dict,
        filename: str = "summary_dashboard",
    ) -> str:
        """
        4-panel summary dashboard combining key results.
        """
        fig = plt.figure(figsize=(14, 10))
        gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

        # Panel 1: Classification comparison
        ax1 = fig.add_subplot(gs[0, 0])
        names = list(classification_results.keys())
        f1_scores = [classification_results[n].get("f1_macro", 0) for n in names]
        colors = sns.color_palette("colorblind", len(names))
        ax1.bar(range(len(names)), f1_scores, color=colors)
        ax1.set_xticks(range(len(names)))
        ax1.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        ax1.set_ylabel("F1 Macro")
        ax1.set_title("Classification Performance")
        ax1.set_ylim(0, 1.0)

        # Panel 2: Semantic preservation
        ax2 = fig.add_subplot(gs[0, 1])
        enc_names = list(semantic_results.keys())
        cosine_sims = [semantic_results[e].get("mean_cosine_similarity", 0) for e in enc_names]
        ax2.barh(enc_names, cosine_sims, color=sns.color_palette("colorblind", len(enc_names)))
        ax2.set_xlabel("Mean Cosine Similarity")
        ax2.set_title("Semantic Preservation")
        ax2.set_xlim(0, 1.0)

        # Panel 3: Energy savings
        ax3 = fig.add_subplot(gs[1, 0])
        enc_names_e = list(energy_results.keys())
        savings = [energy_results[e]["comparison"]["energy_savings_pct"] for e in enc_names_e]
        ax3.bar(enc_names_e, savings, color=sns.color_palette("husl", len(enc_names_e)))
        ax3.set_xticks(range(len(enc_names_e)))
        ax3.set_xticklabels(enc_names_e, rotation=30, ha="right", fontsize=8)
        ax3.set_ylabel("Energy Savings (%)")
        ax3.set_title("Energy Efficiency vs Transformer")
        ax3.axhline(0, color="black", linewidth=0.5)

        # Panel 4: Sparsity
        ax4 = fig.add_subplot(gs[1, 1])
        sparsities = [energy_results[e]["snn_energy"]["sparsity"] for e in enc_names_e]
        ax4.bar(enc_names_e, sparsities, color=sns.color_palette("muted", len(enc_names_e)))
        ax4.set_xticks(range(len(enc_names_e)))
        ax4.set_xticklabels(enc_names_e, rotation=30, ha="right", fontsize=8)
        ax4.set_ylabel("Spike Sparsity")
        ax4.set_title("Encoding Sparsity")
        ax4.set_ylim(0, 1.0)

        fig.suptitle(
            "Research Summary: Spike Encoding for Legal Text Classification",
            fontsize=13, y=1.01,
        )
        path = self._save(fig, filename)
        plt.close(fig)
        return path

    # ─────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────

    def _save(self, fig, filename: str) -> str:
        path = self.out_dir / f"{filename}.{self.fmt}"
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        logger.info(f"Saved figure -> {path}")
        return str(path)
