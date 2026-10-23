#!/usr/bin/env python3
"""Read-only validation of the JSON contract consumed by production flux.py."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .prepare_handoff import validate_rows
except ImportError:
    from prepare_handoff import validate_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    rows = json.loads(args.manifest.read_text(encoding="utf-8"))
    print(json.dumps({"valid": True, "manifest": str(args.manifest.resolve()),
                      **validate_rows(rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
