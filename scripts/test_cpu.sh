#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
"${PYTHON:-python}" -m pytest -q tests
cd human
"${HUMAN_TEST_PYTHON:-${PYTHON:-python}}" -m unittest discover -s tests -v
