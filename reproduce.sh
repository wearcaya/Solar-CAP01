#!/usr/bin/env bash
set -euo pipefail

# Run from the root of the extracted package.
python src/run_experiments.py
python src/extended_analysis.py

# Optional: compile the IEEE paper if a LaTeX distribution is installed.
# The manuscript uses a self-contained thebibliography block, so BibTeX is not required.
#if command -v pdflatex >/dev/null 2>&1; then
#  (cd paper && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex)
#fi
