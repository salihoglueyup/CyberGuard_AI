"""Test import"""

import sys

sys.path.insert(0, ".")

try:
    print("Testing direct import...")
    from src.network_detection.models.ssa_lstmids import SSA_LSTMIDS

    print("✅ SSA_LSTMIDS import OK")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback

    traceback.print_exc()
