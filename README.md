# relaxfree-defect-audit

**When can defect relaxation be skipped?** A cross-host audit of relaxation-free
point-defect prescreening in zinc blende semiconductors.

High-throughput defect workflows spend most of their budget relaxing supercells.
Prescreening models try to avoid that step, but they are usually judged by a
single global error. This repository asks a decision-level question instead:
if a candidate is discarded *before* relaxation, how often was it actually a
low-energy defect, and how does that depend on the host family and the defect
type?

The answer is measured on 1549 neutral point defects in 34 group IV, III-V and
II-VI hosts, for which the public DFT relaxation trajectories of Rahman et al.
give both the unrelaxed (ideal-site) and the relaxed formation energy.

![Miss rate of the unrelaxed screen by host family and defect class](figures/fig2_naive_miss_matrix.png)

## Main results

Screening task: keep every defect whose relaxed formation energy is below
3 eV (634 of 1549 defects).

1. **Relaxation is not a small correction.** The median relaxation energy is
   0.46 eV, a quarter of the defects relax by more than 1 eV, and the 99th
   percentile is 6.8 eV. Vacancies in III-V and II-VI hosts relax by about
   0.2 eV; interstitials in group IV and III-V hosts by 0.8-1.1 eV.
2. **Screening on the unrelaxed energy loses a quarter of the hits** (161 of
   634). The loss is 0 % for III-V and II-VI vacancies and 46-67 % for
   self-interstitials in III-V and group IV hosts.
3. **A calibrated margin fixes the average, not the cells.** A single
   split-conformal margin brings the miss rate to the 10 % target and still
   avoids 57 % of the recorded relaxation steps. Under that same margin,
   III-V interstitial impurities are missed at 30 % and group IV antisites at
   24 %.
4. **Cell-wise calibration is honest but data-hungry.** Calibrating each
   family x class cell separately keeps every calibrated cell at or below the
   target, yet 27 % of candidates sit in cells with too few known hits to
   calibrate at all (62 % when the host is new). Those cells must be relaxed.
5. **What transfers.** Margins calibrated on other hosts carry over to an
   unseen host for all three proxies (8-9 % missed at a 10 % target). When a
   whole family is new, the proxies anchored on a single-point calculation
   still hold (8 %), while the identity-only model misses 22 % overall and 34 %
   in group IV.

| proxy (tau = 3 eV, alpha = 0.1, pooled margin) | seen hosts | unseen host | unseen family |
|---|---|---|---|
| unrelaxed energy: missed / steps avoided | 10.0 % / 57 % | 8.2 % / 48 % | 7.7 % / 48 % |
| identity-only model | 10.2 % / 58 % | 8.9 % / 53 % | **22.0 %** / 54 % |
| unrelaxed + learned relaxation energy | 9.8 % / 61 % | 8.0 % / 53 % | 8.3 % / 53 % |

## Proxies

| name | needs | what it is |
|---|---|---|
| `unrelaxed` | one SCF per candidate | formation energy of the defect on its ideal site |
| `identity` | nothing | gradient-boosted regression on host, site and species descriptors |
| `hybrid` | one SCF per candidate | unrelaxed energy minus a learned relaxation energy |

A candidate is kept when `proxy - margin <= tau`. The margin is the split-conformal
quantile of the proxy values of calibration hits, which bounds the expected
miss rate by `alpha` for hits exchangeable with the calibration hits. Margins
are computed once for all defects (pooled), per defect class, or per
family x class cell; a group with fewer than `ceil(1/alpha) - 1` calibration
hits gets no margin and all of its candidates are kept (abstention).

Three regimes are evaluated: random splits (hosts seen), leave-one-host-out,
and leave-one-family-out. In the last two, calibration hosts are disjoint from
both the training and the test hosts.

## Reproducing

```bash
pip install -e ".[dev]"
bash scripts/fetch_data.sh     # optional: the processed tables are already in data/processed
bash run_all.sh                # about 30 minutes
pytest
```

`run_all.sh` rebuilds `data/processed/` only if `data/raw/labels` exists, then
writes every table to `results/` and every figure to `figures/`.

| script | output |
|---|---|
| `experiments/01_build_dataset.py` | processed tables and QC counts |
| `experiments/02_relaxation_audit.py` | relaxation energies and unrelaxed-screen miss rates per cell |
| `experiments/03_proxy_accuracy.py` | MAE of each proxy and model under the three regimes |
| `experiments/04_conformal_screening.py` | calibrated screening: miss rate, discard fraction, steps avoided |
| `experiments/05_secondary_analyses.py` | drivers of the relaxation energy; charged states |
| `experiments/06_figures.py` | figures |

## Using the calibration on your own proxy

```python
import numpy as np
from relaxfree.conformal import recall_margin, keep_mask

# proxy_cal, y_cal: proxy values and relaxed energies of defects you have already relaxed
margin = recall_margin(proxy_cal, y_cal, tau=3.0, alpha=0.1)
keep = keep_mask(proxy_new, margin, tau=3.0)   # relax only these
```

`grouped_recall_margin` does the same per group and returns an infinite margin
for groups that cannot be calibrated.

## Scope and caveats

- One dataset, one structure type, semi-local DFT in 64-atom supercells, as
  published upstream. The numbers describe that setting.
- "Relaxed" means the last recorded ionic step of the upstream trajectory. The
  search for symmetry-broken ground states (e.g. ShakeNBreak) was not part of
  those calculations, so true relaxation energies can only be larger.
- Trajectories without an ideal-site frame (79, mostly above 15 eV unrelaxed)
  and single-frame trajectories (86) are excluded; see `data/README.md`.
- Charged-state trajectories start from the neutral relaxed geometry, so they
  answer a different question and are analysed separately.
- "Relaxation steps avoided" counts recorded ionic steps, not CPU hours.

## Data and licence

Code is released under the MIT licence. The formation energies are derived from
the dataset of Rahman et al. (APL Mach. Learn. 2, 016122, 2024;
<https://github.com/msehabibur/defect_GNN_gen_1>) and Mannodi-Kanakkithodi et
al. (Patterns 3, 100450, 2022); cite them if you use the tables. Elemental
properties are from the Magpie tables distributed with matminer.
