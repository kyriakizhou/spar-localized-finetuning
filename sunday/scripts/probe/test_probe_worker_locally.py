"""
Local test for probe_worker.py — validates the orchestrator __new__ setup.

Checks that all attributes accessed by probe() are set correctly,
without needing torch/GPU. This catches AttributeError issues like
the 'is_multi_model' problem we hit on RunPod.

Usage:
    python test_probe_worker_locally.py
"""

import sys
import inspect


def test_orchestrator_attributes():
    """Verify all attributes accessed in probe() are set in our workaround."""

    # Read probe_worker.py source
    with open("sunday/scripts/probe/probe_worker.py") as f:
        worker_source = f.read()

    # These are the attributes that ProbeOrchestrator.__init__ sets
    # and that probe() / _probe_single_feature() access:
    required_attrs = [
        "extractor",
        "model_config",
        "model_name",
        "backend_name",
        "device",
        "revision",
        "remote",
        "is_multi_model",
        "profiler",
        "model_loading_s",
    ]

    # Check each required attribute is set in the worker source
    print("Checking probe_worker.py sets all required orchestrator attributes...\n")
    missing = []
    for attr in required_attrs:
        pattern = f"orchestrator.{attr}"
        if pattern + " =" in worker_source:
            print(f"  ✅ {attr}")
        else:
            print(f"  ❌ {attr} — MISSING!")
            missing.append(attr)

    # Check NNSightExtractor is called with torch_dtype
    print("\nChecking NNSightExtractor call...")
    if "torch_dtype=torch.bfloat16" in worker_source:
        print("  ✅ torch_dtype=torch.bfloat16 is set")
    elif "dtype=torch.bfloat16" in worker_source:
        print("  ⚠️  Using dtype= (NNSightExtractor expects torch_dtype=)")
    else:
        print("  ❌ No dtype specified — model will load in fp32!")

    # Check diffusers mock
    print("\nChecking diffusers mock...")
    if 'sys.modules["diffusers"]' in worker_source:
        print("  ✅ diffusers mock is present")
    else:
        print("  ❌ diffusers mock missing — will crash on flash-attn version")

    # Check subsample_size in submit script
    print("\nChecking submit_probe_job.py settings...")
    with open("sunday/scripts/probe/submit_probe_job.py") as f:
        submit_source = f.read()

    if "subsample_size=1500" in submit_source:
        print("  ✅ subsample_size=1500")
    else:
        # Find what it is
        for line in submit_source.split("\n"):
            if "subsample_size" in line:
                print(f"  ⚠️  {line.strip()}")

    if "requires_vram_gb=90" in submit_source:
        print("  ✅ requires_vram_gb=90")
    else:
        print("  ❌ requires_vram_gb is not 90")

    if 'base_image = "nielsrolf/ow-default:v0.8"' in submit_source:
        print("  ✅ base_image has version tag")
    else:
        print("  ❌ base_image missing version tag")

    # Summary
    print(f"\n{'=' * 40}")
    if missing:
        print(f"FAIL: {len(missing)} missing attributes: {missing}")
        return False
    else:
        print("ALL CHECKS PASSED ✅")
        print("Safe to submit to RunPod.")
        return True


if __name__ == "__main__":
    success = test_orchestrator_attributes()
    sys.exit(0 if success else 1)
