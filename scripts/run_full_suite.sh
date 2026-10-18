#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

echo "[1/4] Compiling C++17 Stochastic & Monte Carlo Engine (-O3 -march=native)..."
mkdir -p build data figures
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build build -j

echo "[2/4] Executing C++17 GMM EM & Barrier Monte Carlo Benchmarks..."
./build/main_benchmark --export-data data/aapl_ipo_returns.csv --export-json build/cpp_metrics.json

echo "[3/4] Running Mathematical Invariant & Anti-Leakage Unit Tests..."
MPLCONFIGDIR=/tmp python3 -m pytest tests/ -v

echo "[4/4] Generating High-DPI Figures & 2-Page Technical Note PDF..."
MPLCONFIGDIR=/tmp python3 scripts/reproduce_all_figures.py
MPLCONFIGDIR=/tmp python3 scripts/build_technical_note.py

echo "All benchmarks, tests, figures, and Technical_Note.pdf completed successfully."

