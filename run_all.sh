#!/usr/bin/env bash
# Reproduce every table and figure. Takes roughly an hour on a laptop.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if [ ! -d data/raw/labels ]; then
    bash scripts/fetch_data.sh
fi
python experiments/01_build_dataset.py
python experiments/02_relaxation_audit.py
python experiments/03_proxy_accuracy.py
python experiments/04_conformal_screening.py
python experiments/05_secondary_analyses.py
python experiments/08_alloy_subset_check.py
python experiments/09_external_vacancies.py
python experiments/10_robustness.py
python experiments/06_figures.py
python experiments/11_toc_graphic.py
python experiments/07_tables.py
