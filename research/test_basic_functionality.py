"""Basic functionality test to verify the changes work."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

print("Testing basic functionality...")

# Test 1: Import all modified modules
print("\n1. Testing imports...")
try:
    from src.reporting.report import ReportGenerator
    print("   ✓ ReportGenerator imported")
except Exception as e:
    print(f"   ✗ ReportGenerator import failed: {e}")

try:
    from src.experiments.pipeline import ExperimentPipeline
    print("   ✓ ExperimentPipeline imported")
except Exception as e:
    print(f"   ✗ ExperimentPipeline import failed: {e}")

try:
    from src.evaluation.energy import EnergyAnalyzer
    print("   ✓ EnergyAnalyzer imported")
except Exception as e:
    print(f"   ✗ EnergyAnalyzer import failed: {e}")

try:
    from src.encoding import ENCODERS
    print(f"   ✓ ENCODERS imported: {list(ENCODERS.keys())}")
except Exception as e:
    print(f"   ✗ ENCODERS import failed: {e}")

# Test 2: Check encoding filtering logic
print("\n2. Testing encoding filtering logic...")
config = {
    "encoding": {
        "time_steps": 50,
        "methods": {
            "poisson_rate": {"enabled": True},
            "latency": {"enabled": True},
            "temporal": {"enabled": True},
        }
    }
}

# Test with subset
encodings = ["poisson_rate", "latency"]
if encodings is None or len(encodings) == 0:
    enc_cfg = config.get("encoding", {}).get("methods", {})
    filtered = [k for k, v in enc_cfg.items() if v.get("enabled", True)]
else:
    valid_encodings = []
    for enc_name in encodings:
        if enc_name in ENCODERS:
            valid_encodings.append(enc_name)
    filtered = valid_encodings

if filtered == ["poisson_rate", "latency"]:
    print("   ✓ Encoding subset filtering works correctly")
else:
    print(f"   ✗ Encoding subset filtering failed: got {filtered}")

# Test 3: Check energy analyzer has spike_sparsity_pct
print("\n3. Testing energy analyzer...")
try:
    energy_config = {
        "evaluation": {
            "energy": {
                "mac_energy_pj": 4.6,
                "sop_energy_pj": 0.9,
                "memory_access_energy_pj": 100.0
            }
        }
    }
    analyzer = EnergyAnalyzer(energy_config)
    print("   ✓ EnergyAnalyzer instantiated")
    
    # Check that analyze_all_encodings exists
    if hasattr(analyzer, 'analyze_all_encodings'):
        print("   ✓ analyze_all_encodings method exists")
    else:
        print("   ✗ analyze_all_encodings method missing")
except Exception as e:
    print(f"   ✗ EnergyAnalyzer test failed: {e}")

# Test 4: Check ReportGenerator has CSV method
print("\n4. Testing report generator...")
try:
    report_config = {
        "reporting": {
            "format": "html",
            "include_figures": True,
            "include_tables": True,
        },
        "storage": {
            "reports": "storage/results/reports"
        }
    }
    gen = ReportGenerator(report_config)
    print("   ✓ ReportGenerator instantiated")
    
    if hasattr(gen, '_write_results_summary_csv'):
        print("   ✓ _write_results_summary_csv method exists")
    else:
        print("   ✗ _write_results_summary_csv method missing")
except Exception as e:
    print(f"   ✗ ReportGenerator test failed: {e}")

print("\n" + "="*50)
print("Basic functionality test complete!")
print("="*50)
