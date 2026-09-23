"""
Energy Analysis for Spike-based vs Transformer-based classifiers.
Addresses Research Question 4 and Hypothesis H3.

Energy model based on:
  - MAC (Multiply-Accumulate) energy on GPU: ~4.6 pJ (Horowitz 2014)
  - SOP (Synaptic Operation) energy on neuromorphic: ~0.9 pJ (Davies 2018)
  - DRAM access energy: ~100 pJ (Horowitz 2014)
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class EnergyAnalyzer:
    """
    Estimates and compares energy consumption of transformer vs SNN models.

    The analysis follows Horowitz (2014) energy numbers and is standard
    in the neuromorphic computing literature.
    """

    def __init__(self, config: dict):
        energy_cfg = config.get("evaluation", {}).get("energy", {})
        self.mac_energy_pj = energy_cfg.get("mac_energy_pj", 4.6)    # pJ per MAC
        self.sop_energy_pj = energy_cfg.get("sop_energy_pj", 0.9)    # pJ per SOP
        self.dram_energy_pj = energy_cfg.get("memory_access_energy_pj", 100.0)

    def estimate_transformer_energy(
        self,
        n_mac_operations: int,
        model_params: int,
        seq_length: int = 512,
        dtype_bytes: int = 4,
    ) -> dict:
        """
        Estimate energy for one transformer forward pass.

        Args:
            n_mac_operations: number of MAC ops (from TransformerBaseline.count_mac_operations)
            model_params:      number of model parameters
            seq_length:        input sequence length
            dtype_bytes:       bytes per parameter (4 for float32, 2 for fp16)

        Returns:
            dict with energy breakdowns in pJ and mJ
        """
        # Compute energy
        compute_energy_pj = n_mac_operations * self.mac_energy_pj

        # Memory: parameters are loaded from DRAM once per forward pass
        # Plus activations: seq_length × hidden_size × n_layers × dtype_bytes
        param_memory_pj = model_params * dtype_bytes * self.dram_energy_pj / 8
        # (divide by 8 bytes per 64-bit access per Horowitz 2014)

        total_pj = compute_energy_pj + param_memory_pj
        return {
            "model": "transformer",
            "n_mac_ops": n_mac_operations,
            "compute_energy_pj": compute_energy_pj,
            "memory_energy_pj": param_memory_pj,
            "total_energy_pj": total_pj,
            "total_energy_mj": total_pj / 1e9,
            "total_energy_uj": total_pj / 1e6,
        }

    def estimate_snn_energy(
        self,
        n_sop_per_sample: float,
        spike_trains: np.ndarray,
        snn_params: int,
        include_memory: bool = True,
    ) -> dict:
        """
        Estimate energy for one SNN forward pass.

        Args:
            n_sop_per_sample:  average synaptic operations per sample
            spike_trains:      (N, T, F) spike trains for sparsity calculation
            snn_params:        number of SNN parameters
            include_memory:    whether to include DRAM memory access energy

        Returns:
            dict with energy breakdowns
        """
        compute_energy_pj = n_sop_per_sample * self.sop_energy_pj

        sparsity = float(1.0 - spike_trains.mean())
        active_fraction = 1.0 - sparsity

        if include_memory:
            # Memory access reduced by sparsity (event-driven processing)
            param_memory_pj = (
                snn_params * 4 * self.dram_energy_pj / 8 * active_fraction
            )
        else:
            param_memory_pj = 0.0

        total_pj = compute_energy_pj + param_memory_pj
        return {
            "model": "snn",
            "n_sop_per_sample": n_sop_per_sample,
            "sparsity": sparsity,
            "active_fraction": active_fraction,
            "compute_energy_pj": compute_energy_pj,
            "memory_energy_pj": param_memory_pj,
            "total_energy_pj": total_pj,
            "total_energy_uj": total_pj / 1e6,
            "total_energy_mj": total_pj / 1e9,
        }

    def compute_efficiency_ratio(
        self, transformer_energy: dict, snn_energy: dict
    ) -> dict:
        """
        Compare transformer vs SNN energy. Reports speedup and savings.
        """
        t_total = transformer_energy["total_energy_pj"]
        s_total = snn_energy["total_energy_pj"]

        ratio = t_total / max(s_total, 1e-9)
        savings_pct = (1.0 - s_total / max(t_total, 1e-9)) * 100

        return {
            "transformer_total_pj": t_total,
            "snn_total_pj": s_total,
            "energy_ratio": float(ratio),      # transformer / snn
            "energy_savings_pct": float(savings_pct),
            "snn_is_cheaper": s_total < t_total,
            "compute_only_ratio": float(
                transformer_energy["compute_energy_pj"]
                / max(snn_energy["compute_energy_pj"], 1e-9)
            ),
            "memory_penalty": float(
                snn_energy.get("memory_energy_pj", 0)
                / max(t_total, 1e-9)
                * 100
            ),
        }

    def analyze_all_encodings(
        self,
        spike_trains_dict: dict,
        sop_counts_dict: dict,
        transformer_energy: dict,
        snn_params: int,
        include_memory: bool = True,
    ) -> dict:
        """
        Perform energy analysis for all encoding methods.

        Args:
            spike_trains_dict:  {encoding_name: spike_trains (N, T, F)}
            sop_counts_dict:    {encoding_name: avg_sops_per_sample}
            transformer_energy: from estimate_transformer_energy()
            snn_params:         SNN parameter count
            include_memory:     include DRAM in SNN energy estimate

        Returns:
            {encoding_name: {snn_energy, comparison}}
        """
        results = {}
        for enc_name, spk in spike_trains_dict.items():
            sops = sop_counts_dict.get(enc_name, 1000.0)
            snn_e = self.estimate_snn_energy(sops, spk, snn_params, include_memory)
            comparison = self.compute_efficiency_ratio(transformer_energy, snn_e)
            
            # Add spike_sparsity_pct for transparency
            spike_sparsity_pct = snn_e["sparsity"] * 100
            comparison["spike_sparsity_pct"] = spike_sparsity_pct
            
            results[enc_name] = {
                "snn_energy": snn_e,
                "transformer_energy": transformer_energy,
                "comparison": comparison,
            }
            
            # Warning for extreme ratios driven by sparsity
            if comparison["energy_ratio"] > 50000 and spike_sparsity_pct > 95:
                logger.warning(
                    f"[{enc_name}] Energy ratio ({comparison['energy_ratio']:.1f}x) is extremely high "
                    f"and spike sparsity is {spike_sparsity_pct:.1f}%. This ratio may be dominated by "
                    f"near-empty spike trains rather than genuine task-relevant sparsity. "
                    f"Review before reporting."
                )
            
            logger.info(
                f"[{enc_name}] energy ratio: {comparison['energy_ratio']:.2f}x "
                f"savings: {comparison['energy_savings_pct']:.1f}% "
                f"(sparsity: {spike_sparsity_pct:.1f}%)"
            )
        return results
