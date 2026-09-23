"""Tests for results_summary.csv generation in report.py."""

import csv
import tempfile
from pathlib import Path
import pytest
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.reporting.report import ReportGenerator


def test_write_results_summary_csv_single_seed():
    """Test CSV generation with single-seed results."""
    config = {
        "reporting": {
            "format": "html",
            "include_figures": True,
            "include_tables": True,
        },
        "storage": {
            "reports": "storage/results/reports"
        }
    }
    
    # Mock energy results (single seed)
    energy_results = {
        "poisson_rate": {
            "snn_energy": {
                "sparsity": 0.5,
                "compute_energy_pj": 1000,
                "memory_energy_pj": 500,
                "total_energy_pj": 1500,
            },
            "transformer_energy": {
                "compute_energy_pj": 100000,
                "memory_energy_pj": 50000,
                "total_energy_pj": 150000,
            },
            "comparison": {
                "energy_ratio": 100.0,
                "energy_savings_pct": 99.0,
                "spike_sparsity_pct": 50.0,
            },
            "classification": {
                "accuracy": 0.85,
                "f1_macro": 0.82,
                "f1_micro": 0.84,
            }
        },
        "latency": {
            "snn_energy": {
                "sparsity": 0.98,
                "compute_energy_pj": 100,
                "memory_energy_pj": 50,
                "total_energy_pj": 150,
            },
            "transformer_energy": {
                "compute_energy_pj": 100000,
                "memory_energy_pj": 50000,
                "total_energy_pj": 150000,
            },
            "comparison": {
                "energy_ratio": 1000.0,
                "energy_savings_pct": 99.9,
                "spike_sparsity_pct": 98.0,
            },
            "classification": {
                "accuracy": 0.75,
                "f1_macro": 0.72,
                "f1_micro": 0.74,
            }
        }
    }
    
    # Mock semantic results
    semantic_results = {
        "poisson_rate": {
            "mean_cosine_similarity": 0.85,
        },
        "latency": {
            "mean_cosine_similarity": 0.92,
        }
    }
    
    gen = ReportGenerator(config)
    
    # Use temp directory for test
    with tempfile.TemporaryDirectory() as tmpdir:
        gen.out_dir = Path(tmpdir)
        
        csv_path = gen._write_results_summary_csv(energy_results, semantic_results, "test")
        
        # Verify CSV was created
        assert Path(csv_path).exists()
        
        # Read and verify CSV content
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
            
            # Check header
            assert rows[0] == [
                "encoding", "accuracy_mean", "accuracy_std",
                "f1_macro_mean", "f1_macro_std",
                "spike_sparsity_pct_mean", "spike_sparsity_pct_std",
                "energy_ratio_mean", "energy_ratio_std",
                "energy_savings_pct_mean", "energy_savings_pct_std",
                "semantic_preservation_score"
            ]
            
            # Check poisson_rate row (single seed - no std)
            poisson_row = [r for r in rows if r[0] == "poisson_rate"][0]
            assert poisson_row[1] == "0.85"  # accuracy_mean
            assert poisson_row[2] == ""  # accuracy_std (empty for single seed)
            assert poisson_row[3] == "0.82"  # f1_macro_mean
            assert poisson_row[4] == ""  # f1_macro_std
            assert poisson_row[5] == "50.0"  # sparsity_mean
            assert poisson_row[6] == ""  # sparsity_std
            assert poisson_row[7] == "100.0"  # energy_ratio_mean
            assert poisson_row[8] == ""  # energy_ratio_std
            assert poisson_row[9] == "99.0"  # savings_mean
            assert poisson_row[10] == ""  # savings_std
            assert poisson_row[11] == "0.85"  # semantic score
            
            # Check latency row
            latency_row = [r for r in rows if r[0] == "latency"][0]
            assert latency_row[1] == "0.75"
            assert latency_row[11] == "0.92"


def test_write_results_summary_csv_multi_seed():
    """Test CSV generation with multi-seed results (mean±std)."""
    config = {
        "reporting": {
            "format": "html",
            "include_figures": True,
            "include_tables": True,
        },
        "storage": {
            "reports": "storage/results/reports"
        }
    }
    
    # Mock energy results (multi-seed with mean±std)
    energy_results = {
        "poisson_rate": {
            "snn_energy": {
                "sparsity": 0.5,
                "compute_energy_pj": 1000,
                "memory_energy_pj": 500,
                "total_energy_pj": 1500,
            },
            "transformer_energy": {
                "compute_energy_pj": 100000,
                "memory_energy_pj": 50000,
                "total_energy_pj": 150000,
            },
            "comparison": {
                "energy_ratio": {"mean": 9950.0, "std": 25.5},
                "energy_savings_pct": {"mean": 99.0, "std": 0.1},
                "spike_sparsity_pct": {"mean": 50.0, "std": 0.5},
            },
            "classification": {
                "accuracy": {"mean": 0.85, "std": 0.02},
                "f1_macro": {"mean": 0.82, "std": 0.03},
                "f1_micro": {"mean": 0.84, "std": 0.02},
            }
        }
    }
    
    semantic_results = {
        "poisson_rate": {
            "mean_cosine_similarity": 0.85,
        }
    }
    
    gen = ReportGenerator(config)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        gen.out_dir = Path(tmpdir)
        
        csv_path = gen._write_results_summary_csv(energy_results, semantic_results, "test_multi")
        
        assert Path(csv_path).exists()
        
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
            
            # Check poisson_rate row (multi-seed - with std)
            poisson_row = [r for r in rows if r[0] == "poisson_rate"][0]
            assert poisson_row[1] == "0.85"  # accuracy_mean
            assert poisson_row[2] == "0.02"  # accuracy_std
            assert poisson_row[3] == "0.82"  # f1_macro_mean
            assert poisson_row[4] == "0.03"  # f1_macro_std
            assert poisson_row[5] == "50.0"  # sparsity_mean
            assert poisson_row[6] == "0.5"  # sparsity_std
            assert poisson_row[7] == "9950.0"  # energy_ratio_mean
            assert poisson_row[8] == "25.5"  # energy_ratio_std
            assert poisson_row[9] == "99.0"  # savings_mean
            assert poisson_row[10] == "0.1"  # savings_std


def test_write_results_summary_csv_empty_energy():
    """Test CSV generation with empty energy results."""
    config = {
        "reporting": {
            "format": "html",
            "include_figures": True,
            "include_tables": True,
        },
        "storage": {
            "reports": "storage/results/reports"
        }
    }
    
    gen = ReportGenerator(config)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        gen.out_dir = Path(tmpdir)
        
        # Should not crash with empty results
        csv_path = gen._write_results_summary_csv({}, {}, "test_empty")
        
        # CSV should still be created with just header
        assert Path(csv_path).exists()
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
            assert len(rows) == 1  # Only header
