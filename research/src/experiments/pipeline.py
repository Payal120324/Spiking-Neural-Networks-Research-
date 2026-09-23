"""
Experiment Pipeline for Legal NLP Research Framework.
Orchestrates the full end-to-end experimental workflow:
  Dataset → Embeddings → Spike Encoding → SNN Training →
  Evaluation → Visualization → Report
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class ExperimentPipeline:
    """
    Full experiment pipeline for spike encoding vs transformer comparison.
    Each stage is independently runnable and results are cached.
    """

    def __init__(self, config: dict):
        self.config = config
        self.results_dir = Path(config.get("storage", {}).get("results", "storage/results"))
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._results: dict = {}

    # ─────────────────────────────────────────────────────────────────
    # Full Pipeline
    # ─────────────────────────────────────────────────────────────────

    def run(
        self,
        dataset_key: str = "case_hold",
        transformer_key: str = "legal_bert",
        encodings: Optional[list[str]] = None,
        skip_stages: Optional[list[str]] = None,
        target_dataset_key: Optional[str] = None,
        num_seeds: int = 1,
    ) -> dict:
        """
        Run the complete experiment pipeline.

        Args:
            dataset_key:         which dataset to use as the source domain
            transformer_key:     which transformer model to use
            encodings:           list of encoding names (None = all enabled in config)
            skip_stages:         list of stage names to skip
            target_dataset_key:  optional second dataset to evaluate zero-shot
                                  domain-shift transfer against (RQ3 / H4).
                                  If None, the domain-shift stage is skipped.
            num_seeds:           number of random seeds for multi-seed averaging (default: 1)

        Returns:
            dict with all experiment results
        """
        skip = set(skip_stages or [])
        # Handle encodings: None means use all enabled, empty list means use all enabled
        if encodings is None or len(encodings) == 0:
            enc_cfg = self.config.get("encoding", {}).get("methods", {})
            encodings = [k for k, v in enc_cfg.items() if v.get("enabled", True)]
        else:
            # Validate that requested encodings exist
            from .. import encoding as enc_module
            valid_encodings = []
            for enc_name in encodings:
                if enc_name in enc_module.ENCODERS:
                    valid_encodings.append(enc_name)
                else:
                    logger.warning(f"Unknown encoder '{enc_name}', skipping")
            encodings = valid_encodings
            if not encodings:
                raise ValueError(f"No valid encodings found. Requested: {list(encodings)}. Available: {list(enc_module.ENCODERS.keys())}")

        logger.info("=" * 60)
        logger.info("SPIKE-LEGAL-NLP EXPERIMENT PIPELINE")
        logger.info("=" * 60)
        logger.info(f"Dataset:     {dataset_key}")
        logger.info(f"Transformer: {transformer_key}")
        logger.info(f"Encodings:   {encodings}")
        logger.info(f"Seeds:       {num_seeds}")
        logger.info("=" * 60)

        t0 = time.time()

        # Multi-seed averaging
        if num_seeds > 1:
            logger.info(f"Running {num_seeds} seeds for statistical robustness...")
            all_seed_results = []
            base_seed = self.config.get("snn", {}).get("training", {}).get("seed", 42)
            
            for seed_idx in range(num_seeds):
                seed = base_seed + seed_idx
                logger.info(f"\n{'='*60}")
                logger.info(f"SEED {seed_idx + 1}/{num_seeds} (seed={seed})")
                logger.info(f"{'='*60}")
                
                # Set seed for this run
                self.config["snn"]["training"]["seed"] = seed
                self.config["models"]["training"]["seed"] = seed
                
                # Run single seed
                seed_results = self._run_single_seed(
                    dataset_key, transformer_key, encodings, skip,
                    target_dataset_key
                )
                all_seed_results.append(seed_results)
            
            # Aggregate results across seeds
            self._results = self._aggregate_seed_results(all_seed_results)
            
            # Save per-seed raw results
            seeds_dir = self.results_dir / "seeds"
            seeds_dir.mkdir(parents=True, exist_ok=True)
            for seed_idx, seed_result in enumerate(all_seed_results):
                seed_path = seeds_dir / f"seed_{seed_idx + 1}_{dataset_key}_{transformer_key}.json"
                with open(seed_path, "w") as f:
                    json.dump(seed_result, f, indent=2, default=str)
                logger.info(f"Saved seed {seed_idx + 1} results -> {seed_path}")
        else:
            # Single seed run (original behavior)
            self._results = self._run_single_seed(
                dataset_key, transformer_key, encodings, skip,
                target_dataset_key
            )

        # ── Final: Visualization + Report (only once after aggregation) ──
        # Extract data needed for visualization from aggregated results
        cached_result = self._load_cached_data(dataset_key)
        if len(cached_result) != 2:
            logger.error(f"Expected 2 values from _load_cached_data, got {len(cached_result)}")
            logger.error(f"Content: {cached_result}")
        data, ds_info = cached_result
        embeddings = self._load_cached_embeddings(transformer_key, dataset_key)
        
        # Reconstruct spike trains for visualization (use first seed's data)
        if "dataset" not in skip and "encoding" not in skip:
            spike_trains_all, encoders = self._stage_spike_encoding(embeddings, encodings)
        else:
            spike_trains_all = {}
            encoders = {}
        
        # Get aggregated results
        clf_results = self._results.get("classification", {})
        semantic_results = self._results.get("semantic", {})
        energy_results = self._results.get("energy", {})
        shift_results = self._results.get("domain_shift", {})
        
        logger.info("\n[Stage 9/9] Generating visualizations and report…")
        figure_paths = []
        if spike_trains_all:
            figure_paths.extend(
                self._stage_visualizations(
                    data, embeddings, spike_trains_all,
                    clf_results, semantic_results, energy_results,
                )
            )

        report_path = self._stage_report(
            dataset_key, transformer_key,
            clf_results, semantic_results, energy_results,
            shift_results, figure_paths,
        )
        self._results["report"] = report_path

        elapsed = time.time() - t0
        logger.info(f"\n{'='*60}")
        logger.info(f"Pipeline complete in {elapsed:.1f}s")
        logger.info(f"Report: {report_path}")
        logger.info(f"{'='*60}")

        # Save full results JSON
        out_path = self.results_dir / f"results_{dataset_key}_{transformer_key}.json"
        with open(out_path, "w") as f:
            json.dump(self._results, f, indent=2, default=str)
        logger.info(f"Full results saved → {out_path}")

        return self._results

    def _run_single_seed(
        self, dataset_key: str, transformer_key: str, encodings: list[str],
        skip: set, target_dataset_key: Optional[str]
    ) -> dict:
        """Run a single seed of the experiment pipeline."""
        results = {}
        
        # ── Stage 1: Load dataset ─────────────────────────────────────
        if "dataset" not in skip:
            logger.info("\n[Stage 1/9] Loading dataset…")
            data, ds_info = self._stage_load_dataset(dataset_key)
            results["dataset_key"] = dataset_key
            results["dataset_info"] = ds_info
        else:
            logger.info("[Stage 1/9] Skipped (dataset)")
            data, ds_info = self._load_cached_data(dataset_key)

        # ── Stage 2: Compute embeddings ───────────────────────────────
        if "embeddings" not in skip:
            logger.info("\n[Stage 2/9] Extracting transformer embeddings…")
            embeddings = self._stage_embeddings(data, transformer_key, dataset_key)
            results["embeddings_shape"] = {
                k: v.shape for k, v in embeddings.items()
            }
        else:
            logger.info("[Stage 2/9] Skipped (embeddings)")
            embeddings = self._load_cached_embeddings(transformer_key, dataset_key)

        # ── Stage 3: Spike encoding ───────────────────────────────────
        if "encoding" not in skip:
            logger.info("\n[Stage 3/9] Generating spike trains…")
            spike_trains_all, encoders = self._stage_spike_encoding(embeddings, encodings)
        else:
            logger.info("[Stage 3/9] Skipped (encoding)")
            spike_trains_all = {}
            encoders = {}

        # ── Stage 4: Transformer baseline ────────────────────────────
        if "transformer_eval" not in skip:
            logger.info("\n[Stage 4/9] Evaluating transformer baseline…")
            transformer_result = self._stage_transformer_eval(
                data, embeddings, transformer_key, dataset_key
            )
            results["transformer"] = transformer_result
        else:
            logger.info("[Stage 4/9] Skipped (transformer_eval)")
            transformer_result = {}

        # ── Stage 5: SNN training ─────────────────────────────────────
        if "snn" not in skip:
            logger.info("\n[Stage 5/9] Training SNN classifiers…")
            snn_results = self._stage_snn_training(data, spike_trains_all, embeddings)
            results["snn"] = snn_results
        else:
            logger.info("[Stage 5/9] Skipped (snn)")
            snn_results = {}

        # Merge classification results
        clf_results = {}
        if transformer_result:
            clf_results[f"transformer_{transformer_key}"] = transformer_result
        for enc_name, res in snn_results.items():
            clf_results[f"snn_{enc_name}"] = res.get("final_val", res)
        results["classification"] = clf_results

        # ── Stage 6: Semantic preservation ────────────────────────────
        if "semantic" not in skip and spike_trains_all and "train" in embeddings:
            logger.info("\n[Stage 6/9] Semantic preservation analysis…")
            semantic_results = self._stage_semantic(embeddings, spike_trains_all, encoders)
            results["semantic"] = semantic_results
        else:
            logger.info("[Stage 6/9] Skipped (semantic)")
            semantic_results = {}

        # ── Stage 7: Energy analysis ──────────────────────────────────
        if "energy" not in skip and spike_trains_all and len(snn_results) > 0:
            logger.info("\n[Stage 7/9] Energy analysis…")
            energy_results = self._stage_energy(
                data, embeddings, spike_trains_all, transformer_key, snn_results
            )
            results["energy"] = energy_results
        else:
            skip_reason = "energy" if "energy" in skip else "snn" if len(snn_results) == 0 else "encoding"
            logger.info(f"[Stage 7/9] Skipped ({skip_reason})")
            energy_results = {}

        # ── Stage 8: Domain shift (RQ3 / H4) ───────────────────────────
        shift_results = {}
        if "domain_shift" not in skip and target_dataset_key:
            logger.info(f"\n[Stage 8/9] Domain shift evaluation ({dataset_key} → {target_dataset_key})…")
            shift_results = self._stage_domain_shift(
                dataset_key, target_dataset_key, transformer_key,
                data, embeddings, spike_trains_all, encoders,
            )
            results["domain_shift"] = shift_results
        else:
            logger.info("[Stage 8/9] Skipped (domain_shift)" if not target_dataset_key
                        else "[Stage 8/9] Skipped (domain_shift, --skip)")
        
        return results

    def _aggregate_seed_results(self, all_seed_results: list[dict]) -> dict:
        """Aggregate results across multiple seeds with mean±std statistics."""
        import numpy as np
        
        aggregated = {}
        
        # Copy non-aggregated fields from first seed
        first_seed = all_seed_results[0]
        for key in ["dataset_key", "dataset_info", "embeddings_shape", "transformer", "semantic", "domain_shift"]:
            if key in first_seed:
                aggregated[key] = first_seed[key]
        
        # Aggregate classification results
        aggregated["classification"] = {}
        for model_key in first_seed.get("classification", {}):
            values = []
            for seed_result in all_seed_results:
                if model_key in seed_result.get("classification", {}):
                    result = seed_result["classification"][model_key]
                    final_val = result.get("final_val", result)
                    values.append({
                        "accuracy": final_val.get("accuracy", 0),
                        "f1_macro": final_val.get("f1_macro", 0),
                        "f1_micro": final_val.get("f1_micro", 0),
                    })
            
            if values:
                accs = [v["accuracy"] for v in values]
                f1_macros = [v["f1_macro"] for v in values]
                f1_micros = [v["f1_micro"] for v in values]
                
                aggregated["classification"][model_key] = {
                    "accuracy": {"mean": np.mean(accs), "std": np.std(accs)},
                    "f1_macro": {"mean": np.mean(f1_macros), "std": np.std(f1_macros)},
                    "f1_micro": {"mean": np.mean(f1_micros), "std": np.std(f1_micros)},
                }
        
        # Aggregate energy results
        aggregated["energy"] = {}
        for enc_name in first_seed.get("energy", {}):
            energy_ratios = []
            energy_savings = []
            sparsity_pcts = []
            accuracies = []
            f1_macros = []
            
            for seed_result in all_seed_results:
                if enc_name in seed_result.get("energy", {}):
                    enc_energy = seed_result["energy"][enc_name]
                    comp = enc_energy.get("comparison", {})
                    energy_ratios.append(comp.get("energy_ratio", 0))
                    energy_savings.append(comp.get("energy_savings_pct", 0))
                    sparsity_pcts.append(comp.get("spike_sparsity_pct", 0))
                    
                    # Classification metrics
                    clf = enc_energy.get("classification", {})
                    accuracies.append(clf.get("accuracy", 0))
                    f1_macros.append(clf.get("f1_macro", 0))
            
            if energy_ratios:
                aggregated["energy"][enc_name] = {
                    "snn_energy": first_seed["energy"][enc_name]["snn_energy"],
                    "transformer_energy": first_seed["energy"][enc_name]["transformer_energy"],
                    "comparison": {
                        "energy_ratio": {"mean": np.mean(energy_ratios), "std": np.std(energy_ratios)},
                        "energy_savings_pct": {"mean": np.mean(energy_savings), "std": np.std(energy_savings)},
                        "spike_sparsity_pct": {"mean": np.mean(sparsity_pcts), "std": np.std(sparsity_pcts)},
                    },
                    "classification": {
                        "accuracy": {"mean": np.mean(accuracies), "std": np.std(accuracies)},
                        "f1_macro": {"mean": np.mean(f1_macros), "std": np.std(f1_macros)},
                    },
                }
        
        return aggregated

    # ─────────────────────────────────────────────────────────────────
    # Stage Implementations
    # ─────────────────────────────────────────────────────────────────

    def _stage_load_dataset(self, dataset_key: str) -> tuple:
        from ..datasets import DatasetManager, DatasetStatistics
        from ..datasets.manager import DATASET_REGISTRY

        dm = DatasetManager(self.config)
        data = dm.load(dataset_key)
        info = DATASET_REGISTRY.get(dataset_key, {})
        stats = DatasetStatistics(data, info)
        stats_dict = stats.compute_all()
        stats.print_summary()

        # Save stats
        stats_path = self.results_dir / f"dataset_stats_{dataset_key}.json"
        with open(stats_path, "w") as f:
            json.dump(stats_dict, f, indent=2, default=str)

        return data, stats_dict

    def _stage_embeddings(self, data: dict, transformer_key: str, dataset_key: str) -> dict:
        from ..models import TransformerBaseline

        model = TransformerBaseline(self.config, transformer_key)
        
        # Check if fine-tuning is enabled
        use_finetuning = self.config.get("models", {}).get("transformers", {}).get(transformer_key, {}).get("finetune", False)
        
        if use_finetuning:
            logger.info("  Fine-tuning transformer on training data...")
            num_labels = len(set([row["label"] for row in data.get("train", [])]))
            model.load(num_labels=num_labels, mode="finetune")
            model.finetune(data["train"], data.get("validation", []))
            logger.info("  Fine-tuning complete, extracting embeddings...")
        
        embeddings = {}
        for split_name, rows in data.items():
            if not rows:
                continue
            logger.info(f"  Extracting {split_name} embeddings ({len(rows)} samples)…")
            emb = model.get_embeddings(
                rows,
                batch_size=32,
                cache_key=f"{dataset_key}_{split_name}_finetuned" if use_finetuning else f"{dataset_key}_{split_name}",
            )
            embeddings[split_name] = emb
        return embeddings

    def _stage_spike_encoding(self, embeddings: dict, encodings: list[str]) -> tuple:
        from .. import encoding as enc_module

        time_steps = self.config.get("encoding", {}).get("time_steps", 50)
        enc_cfg = self.config.get("encoding", {}).get("methods", {})
        spike_trains_all = {}
        encoders = {}

        max_memory_gb = self.config.get("encoding", {}).get("max_memory_gb", 8.0)

        for enc_name in encodings:
            if enc_name not in enc_module.ENCODERS:
                logger.warning(f"Unknown encoder '{enc_name}', skipping")
                continue
            enc_params = enc_cfg.get(enc_name, {})
            encoder = enc_module.ENCODERS[enc_name](time_steps=time_steps, **enc_params)
            encoders[enc_name] = encoder
            # [FIX, 2026-09]: fit per-feature normalization stats on the
            # training split ONCE, before encoding any split, so train/val/
            # test are all normalized consistently and without test-set
            # leakage. See BaseSpikeEncoder.fit_normalize for why this
            # matters (per-sample min-max was dominated by 1-2 outlier
            # embedding dimensions present in ~75% of samples).
            if "train" in embeddings:
                encoder.fit_normalize(embeddings["train"])
            else:
                logger.warning(
                    f"  [{enc_name}] No 'train' split available to fit "
                    f"normalization stats; falling back to per-sample "
                    f"normalization (pre-fix behaviour)."
                )
            spike_trains_all[enc_name] = {}
            for split_name, emb in embeddings.items():
                self._check_spike_memory(
                    enc_name, split_name, emb, time_steps,
                    enc_params.get("n_neurons", 1), max_memory_gb,
                )
                logger.info(f"  [{enc_name}] Encoding {split_name} split ({len(emb)} samples)…")
                spk = encoder.encode(emb)
                spike_trains_all[enc_name][split_name] = spk
                logger.info(
                    f"    -> shape={spk.shape}, sparsity={encoder.sparsity(spk):.2%}"
                )
        return spike_trains_all, encoders

    @staticmethod
    def _check_spike_memory(
        enc_name: str, split_name: str, embeddings, time_steps: int,
        n_neurons: int, max_memory_gb: float,
    ) -> None:
        """Raise before allocating a spike tensor larger than the configured budget."""
        n_samples, n_features = embeddings.shape
        gb = n_samples * time_steps * n_features * max(1, n_neurons) * 4 / 1024 ** 3
        logger.info(
            f"  [{enc_name}] {split_name}: spike tensor "
            f"({n_samples}, {time_steps}, {n_features * max(1, n_neurons)}) float32 ≈ {gb:.2f} GB"
        )
        if gb > max_memory_gb:
            raise MemoryError(
                f"Spike encoding '{enc_name}' on the '{split_name}' split needs ~{gb:.1f} GB "
                f"of RAM (limit: {max_memory_gb} GB). Reduce the number of samples "
                f"(--max-train-samples / --max-val-samples / --max-test-samples or --quick), "
                f"lower encoding.time_steps, or raise encoding.max_memory_gb in config.yaml."
            )

    def _stage_transformer_eval(
        self, data: dict, embeddings: dict, transformer_key: str, dataset_key: str
    ) -> dict:
        from ..models import TransformerBaseline
        from ..evaluation import ClassificationMetrics

        model = TransformerBaseline(self.config, transformer_key)
        train_rows = data.get("train", [])
        val_rows = data.get("validation", data.get("test", []))

        if not train_rows or not val_rows:
            logger.warning("Insufficient data for transformer eval")
            return {}

        val_split = "validation" if data.get("validation") else "test"
        result = model.train_linear_probe(
            train_rows, val_rows, cache_key=f"{dataset_key}", val_split=val_split
        )
        return result

    def _stage_snn_training(
        self, data: dict, spike_trains_all: dict, embeddings: dict
    ) -> dict:
        from ..models import SNNClassifier

        snn_results = {}
        train_rows = data.get("train", [])
        val_rows = data.get("validation", data.get("test", []))

        # Encode labels as integers
        all_labels = [r["label"] for r in train_rows + val_rows]
        if isinstance(all_labels[0], list):
            # Multi-label: convert to int via argmax of label list
            all_labels_flat = [tuple(sorted(l)) for l in all_labels]
            unique = sorted(set(all_labels_flat))
            label_map = {l: i for i, l in enumerate(unique)}
            y_train = np.array([label_map[tuple(sorted(r["label"]))] for r in train_rows])
            y_val = np.array([label_map.get(tuple(sorted(r["label"])), 0) for r in val_rows])
        else:
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y_all = le.fit_transform([str(r["label"]) for r in train_rows + val_rows])
            y_train = y_all[: len(train_rows)]
            y_val = y_all[len(train_rows) :]

        for enc_name, splits in spike_trains_all.items():
            spk_train = splits.get("train")
            spk_val = splits.get("validation", splits.get("test"))

            if spk_train is None or spk_val is None:
                continue

            # Match label count with available samples
            n_train = min(len(y_train), len(spk_train))
            n_val = min(len(y_val), len(spk_val))

            logger.info(f"  Training SNN [{enc_name}] n_train={n_train}…")
            clf = SNNClassifier(self.config, encoding_name=enc_name)
            result = clf.train(
                spk_train[:n_train],
                y_train[:n_train],
                spk_val[:n_val],
                y_val[:n_val],
            )
            snn_results[enc_name] = result

        return snn_results

    def _stage_semantic(self, embeddings: dict, spike_trains_all: dict, encoders: dict) -> dict:
        from ..evaluation import SemanticPreservation

        sem = SemanticPreservation(self.config)
        original_emb = embeddings.get("train") if "train" in embeddings else embeddings.get("test")
        if original_emb is None:
            return {}

        decoded_by_enc = {}
        for enc_name, splits in spike_trains_all.items():
            spk = splits.get("train", splits.get("test"))
            if spk is None:
                continue
            encoder = encoders.get(enc_name)
            if encoder is None:
                continue
            decoded_by_enc[enc_name] = encoder.decode(spk)

        return sem.compare_encodings(original_emb, decoded_by_enc)

    def _stage_energy(
        self, data, embeddings, spike_trains_all, transformer_key, snn_results
    ) -> dict:
        from ..models import TransformerBaseline, SNNClassifier
        from ..evaluation import EnergyAnalyzer

        analyzer = EnergyAnalyzer(self.config)
        model = TransformerBaseline(self.config, transformer_key)
        model.load(mode="embedding")

        max_len = self.config.get("preprocessing", {}).get("max_length", 512)
        n_mac = model.count_mac_operations(seq_length=max_len)
        n_params = sum(p.numel() for p in model.model.parameters())
        transformer_energy = analyzer.estimate_transformer_energy(n_mac, n_params, max_len)

        # SNN params estimate
        snn_hidden = self.config.get("snn", {}).get("architecture", {}).get("hidden_size", 256)
        first_emb = list(embeddings.values())[0] if embeddings else None
        input_dim = first_emb.shape[1] if first_emb is not None else 768
        n_layers = self.config.get("snn", {}).get("architecture", {}).get("num_hidden_layers", 2)
        snn_params = input_dim * snn_hidden + (n_layers - 1) * snn_hidden ** 2

        # Collect SOPs
        sop_counts = {}
        spk_by_enc = {}
        for enc_name, splits in spike_trains_all.items():
            spk = splits.get("train") if "train" in splits else splits.get("test")
            if spk is None:
                continue
            spk_by_enc[enc_name] = spk
            clf = SNNClassifier(self.config, enc_name)
            train_labels = [r["label"] for r in data.get("train", [])]
            # Handle multi-label lists
            if train_labels and isinstance(train_labels[0], list):
                flat_labels = [item for sublist in train_labels for item in sublist]
                max_label = max(flat_labels) if flat_labels else 1
            else:
                max_label = max(train_labels) if train_labels else 1
            clf.num_classes = max(2, int(max_label) + 1)
            sop_info = clf.count_synaptic_operations(spk[:200])
            sop_counts[enc_name] = sop_info["avg_sops_per_sample"]

        energy_results = analyzer.analyze_all_encodings(
            spk_by_enc, sop_counts, transformer_energy, snn_params, include_memory=True
        )
        
        # Pair energy with accuracy - fail loudly if snn_results is missing
        for enc_name in energy_results.keys():
            if enc_name not in snn_results:
                raise ValueError(
                    f"Energy analysis computed for encoding '{enc_name}' but SNN results are missing. "
                    f"Cannot report energy efficiency without classification accuracy. "
                    f"Ensure SNN training stage was not skipped for this encoding."
                )
            # Add accuracy/F1 metrics to energy results
            enc_snn_result = snn_results[enc_name]
            final_val = enc_snn_result.get("final_val", enc_snn_result)
            energy_results[enc_name]["classification"] = {
                "accuracy": final_val.get("accuracy", None),
                "f1_macro": final_val.get("f1_macro", None),
                "f1_micro": final_val.get("f1_micro", None),
            }
        
        return energy_results

    def _stage_domain_shift(
        self,
        source_dataset_key: str,
        target_dataset_key: str,
        transformer_key: str,
        source_data: dict,
        source_embeddings: dict,
        spike_trains_all: dict,
        encoders: dict,
    ) -> dict:
        """
        Evaluate zero-shot domain-shift robustness (RQ3 / H4): train a
        linear probe on the SOURCE dataset's representation and evaluate
        it, without retraining, on the TARGET dataset — once for the raw
        transformer embedding, and once per spike encoding using that
        encoding's own decode() reconstruction. This makes the transfer
        test directly comparable across representations.
        """
        from ..evaluation import DomainShiftEvaluator

        ds_cfg = self.config.get("evaluation", {}).get("domain_shift", {})
        max_samples = ds_cfg.get("max_samples", 300)

        evaluator = DomainShiftEvaluator(self.config)

        logger.info(f"  Loading target dataset '{target_dataset_key}'…")
        target_data, _ = self._stage_load_dataset(target_dataset_key)
        logger.info(f"  Extracting target embeddings [{transformer_key}]…")
        target_embeddings = self._stage_embeddings(target_data, transformer_key, target_dataset_key)

        src_rows = source_data.get("train", [])
        tgt_rows = target_data.get("train", target_data.get("test", []))
        src_emb = source_embeddings.get("train")
        tgt_emb = target_embeddings.get("train", target_embeddings.get("test"))

        if src_emb is None or tgt_emb is None or not src_rows or not tgt_rows:
            logger.warning("  Insufficient data for domain shift evaluation; skipping.")
            return {}

        src_labels = [r["label"] for r in src_rows]
        tgt_labels = [r["label"] for r in tgt_rows]

        n_src = min(len(src_emb), len(src_labels), max_samples)
        n_tgt = min(len(tgt_emb), len(tgt_labels), max_samples)
        if n_src < 2 or n_tgt < 2:
            logger.warning("  Not enough samples for domain shift evaluation; skipping.")
            return {}

        transfer_results = []

        # Transformer-embedding transfer
        transfer_results.append(
            evaluator.evaluate_transfer(
                model=None,
                source_name=source_dataset_key,
                target_name=target_dataset_key,
                source_embeddings=src_emb[:n_src],
                source_labels=src_labels[:n_src],
                target_embeddings=tgt_emb[:n_tgt],
                target_labels=tgt_labels[:n_tgt],
                model_type="transformer",
            )
        )

        # Per-encoding transfer, using each encoding's own decode()
        for enc_name, encoder in encoders.items():
            src_splits = spike_trains_all.get(enc_name, {})
            src_spk = src_splits.get("train")
            if src_spk is None:
                continue

            logger.info(f"  [snn_{enc_name}] Encoding target split for transfer test…")
            tgt_spk = encoder.encode(tgt_emb[:n_tgt])

            src_dec = encoder.decode(src_spk[:n_src])
            tgt_dec = encoder.decode(tgt_spk)

            transfer_results.append(
                evaluator.evaluate_transfer(
                    model=None,
                    source_name=source_dataset_key,
                    target_name=target_dataset_key,
                    source_embeddings=src_dec,
                    source_labels=src_labels[:n_src],
                    target_embeddings=tgt_dec,
                    target_labels=tgt_labels[:n_tgt],
                    model_type=f"snn_{enc_name}",
                )
            )

        summary = evaluator.compare_models_on_domain_shift(transfer_results)
        summary["_embedding_shift"] = evaluator.analyze_embedding_shift(
            src_emb[:n_src], tgt_emb[:n_tgt], source_dataset_key, target_dataset_key
        )
        return summary

    def _stage_visualizations(
        self, data, embeddings, spike_trains_all, clf_results, semantic_results, energy_results
    ) -> list[str]:
        from ..visualization import ResearchPlotter

        plotter = ResearchPlotter(self.config)
        paths = []

        # Classification comparison
        if clf_results:
            p = plotter.plot_classification_comparison(clf_results)
            paths.append(p)

        # Spike raster plots
        spk_for_raster = {
            enc: splits.get("train", splits.get("test"))
            for enc, splits in spike_trains_all.items()
            if splits.get("train") is not None or splits.get("test") is not None
        }
        if spk_for_raster:
            p = plotter.plot_encoding_comparison_rasters(spk_for_raster)
            paths.append(p)
            p = plotter.plot_firing_rates(spk_for_raster)
            paths.append(p)

        # Semantic preservation
        if semantic_results:
            p = plotter.plot_semantic_preservation(semantic_results)
            paths.append(p)

        # Embedding scatter
        if embeddings and spk_for_raster:
            emb = embeddings.get("train") if "train" in embeddings else list(embeddings.values())[0]
            train_rows = data.get("train", [])
            if train_rows and len(train_rows) > 0:
                raw_labels = [r["label"] for r in train_rows]
                if isinstance(raw_labels[0], list):
                    labels = np.array([len(l) for l in raw_labels])
                else:
                    labels = np.array(raw_labels)
                labels = labels[:len(emb)]
                p = plotter.plot_embedding_scatter(emb, spk_for_raster, labels)
                paths.append(p)

        # Energy analysis
        if energy_results:
            p = plotter.plot_energy_comparison(energy_results)
            paths.append(p)
            if semantic_results:
                p = plotter.plot_energy_sparsity_tradeoff(energy_results, semantic_results)
                paths.append(p)

        # Summary dashboard
        if clf_results and semantic_results and energy_results:
            p = plotter.plot_summary_dashboard(clf_results, semantic_results, energy_results)
            paths.append(p)

        return [p for p in paths if p]

    def _stage_report(
        self, dataset_key, transformer_key, clf_results, semantic_results,
        energy_results, shift_results, figure_paths
    ) -> str:
        from ..reporting import ReportGenerator

        gen = ReportGenerator(self.config)
        title = (
            "A Semantic- and Energy-Aware Study of Spike Encoding "
            "for Legal Text Classification Under Domain Shift"
        )
        return gen.generate(
            title=title,
            experiment_config=self.config,
            classification_results=clf_results,
            semantic_results=semantic_results,
            energy_results=energy_results,
            domain_shift_results=shift_results if shift_results else None,
            figure_paths=figure_paths,
            filename=f"report_{dataset_key}_{transformer_key}",
        )

    # ─────────────────────────────────────────────────────────────────
    # Cache helpers
    # ─────────────────────────────────────────────────────────────────

    def _load_cached_data(self, dataset_key: str) -> tuple:
        from ..datasets import DatasetManager, DatasetStatistics
        from ..datasets.manager import DATASET_REGISTRY
        dm = DatasetManager(self.config)
        data = dm.load(dataset_key)
        info = DATASET_REGISTRY.get(dataset_key, {})
        stats = DatasetStatistics(data, info)
        ds_info = stats.compute_all()
        result = (data, ds_info)
        return result

    def _load_cached_embeddings(self, transformer_key: str, dataset_key: str) -> dict:
        import pickle
        emb_dir = Path(self.config.get("storage", {}).get("embeddings_cache", "storage/embeddings"))
        result = {}
        for split in ["train", "validation", "test"]:
            p = emb_dir / f"{transformer_key}_{dataset_key}_{split}.pkl"
            if p.exists():
                with open(p, "rb") as f:
                    result[split] = pickle.load(f)
        return result