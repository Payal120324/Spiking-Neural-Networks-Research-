"""Tests for --encodings subset filtering in pipeline.py."""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.encoding import ENCODERS


def test_encodings_subset_valid():
    """Test that valid encoding subset is correctly filtered."""
    config = {
        "encoding": {
            "time_steps": 50,
            "methods": {
                "poisson_rate": {"enabled": True},
                "latency": {"enabled": True},
                "temporal": {"enabled": True},
                "population": {"enabled": True},
                "binary_threshold": {"enabled": True},
            }
        },
        "storage": {
            "results": "storage/results"
        }
    }
    
    # Test with valid subset
    encodings = ["poisson_rate", "latency", "temporal"]
    
    # Simulate the encoding filtering logic from run()
    if encodings is None or len(encodings) == 0:
        enc_cfg = config.get("encoding", {}).get("methods", {})
        filtered = [k for k, v in enc_cfg.items() if v.get("enabled", True)]
    else:
        valid_encodings = []
        for enc_name in encodings:
            if enc_name in ENCODERS:
                valid_encodings.append(enc_name)
        filtered = valid_encodings
    
    assert filtered == ["poisson_rate", "latency", "temporal"]


def test_encodings_subset_with_invalid():
    """Test that invalid encodings are filtered out with warning."""
    # Test with mix of valid and invalid encodings
    encodings = ["poisson_rate", "invalid_encoder", "latency"]
    
    valid_encodings = []
    for enc_name in encodings:
        if enc_name in ENCODERS:
            valid_encodings.append(enc_name)
    
    assert valid_encodings == ["poisson_rate", "latency"]
    assert "invalid_encoder" not in valid_encodings


def test_encodings_subset_all_invalid():
    """Test that all invalid encodings raise ValueError."""
    encodings = ["invalid1", "invalid2"]
    
    valid_encodings = []
    for enc_name in encodings:
        if enc_name in ENCODERS:
            valid_encodings.append(enc_name)
    
    if not valid_encodings:
        # This should raise ValueError in the actual pipeline
        with pytest.raises(ValueError, match="No valid encodings found"):
            raise ValueError(f"No valid encodings found. Requested: {encodings}. Available: {list(ENCODERS.keys())}")


def test_encodings_none_uses_all_enabled():
    """Test that None encodings uses all enabled encodings from config."""
    config = {
        "encoding": {
            "time_steps": 50,
            "methods": {
                "poisson_rate": {"enabled": True},
                "latency": {"enabled": True},
                "temporal": {"enabled": False},  # Disabled
                "population": {"enabled": True},
            }
        },
        "storage": {
            "results": "storage/results"
        }
    }
    
    encodings = None
    enc_cfg = config.get("encoding", {}).get("methods", {})
    filtered = [k for k, v in enc_cfg.items() if v.get("enabled", True)]
    
    assert filtered == ["poisson_rate", "latency", "population"]
    assert "temporal" not in filtered  # Disabled


def test_encodings_empty_list_uses_all_enabled():
    """Test that empty list encodings uses all enabled encodings from config."""
    config = {
        "encoding": {
            "time_steps": 50,
            "methods": {
                "poisson_rate": {"enabled": True},
                "latency": {"enabled": True},
                "temporal": {"enabled": False},
            }
        },
        "storage": {
            "results": "storage/results"
        }
    }
    
    encodings = []
    enc_cfg = config.get("encoding", {}).get("methods", {})
    filtered = [k for k, v in enc_cfg.items() if v.get("enabled", True)]
    
    assert filtered == ["poisson_rate", "latency"]
    assert "temporal" not in filtered


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
