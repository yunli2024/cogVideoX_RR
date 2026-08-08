#!/usr/bin/env python3
"""Run-scoped wrapper for the frozen CogKit RR RORD-280 artifact audit."""

from __future__ import annotations

import importlib.util
from pathlib import Path


BASE_AUDIT = Path(__file__).with_name("audit_cogkit_rr_rord280_full_20260807.py")
EXPECTED_RUN_ID = "run-20260807-235059-17ff7603"


def main() -> None:
    spec = importlib.util.spec_from_file_location("cogkit_rr_rord280_base_audit", BASE_AUDIT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load base audit: {BASE_AUDIT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.RUN_ID = EXPECTED_RUN_ID
    module.main()


if __name__ == "__main__":
    main()
