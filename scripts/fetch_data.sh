#!/usr/bin/env bash
# Download the public zinc blende defect dataset of Rahman et al. (2024) and
# unpack the neutral/charged trajectory label files used in this study.
#
# Source: https://github.com/msehabibur/defect_GNN_gen_1
# Paper:  M. H. Rahman et al., APL Machine Learning 2, 016122 (2024)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW="${ROOT}/data/raw"
UPSTREAM="https://github.com/msehabibur/defect_GNN_gen_1.git"
COMMIT="d4853e1625ee2bcee2c26cb9fd4551a6562925c5"

mkdir -p "${RAW}"
if [ ! -d "${RAW}/defect_GNN_gen_1/.git" ]; then
    git clone "${UPSTREAM}" "${RAW}/defect_GNN_gen_1"
fi
git -C "${RAW}/defect_GNN_gen_1" checkout -q "${COMMIT}"

SRC="${RAW}/defect_GNN_gen_1"
OUT="${RAW}/labels"
rm -rf "${OUT}"
mkdir -p "${OUT}/native" "${OUT}/impurity/A_rich"

# Dataset 1: native defects (vacancies, antisites, self-interstitials)
unzip -q -o "${SRC}/Dataset_1/Dataset_1.zip" "*.csv" -x "__MACOSX/*" -d "${OUT}/native"

# Dataset 2: extrinsic impurities (substitutional and interstitial)
for q in Neutral Charged+1 Charged-1 Charged+2 Charged-2; do
    unzip -q -o "${SRC}/Dataset_2/A_rich/${q}.zip" "*.csv" -x "__MACOSX/*" -d "${OUT}/impurity/A_rich"
done

find "${OUT}" -name "*.csv" | sort
