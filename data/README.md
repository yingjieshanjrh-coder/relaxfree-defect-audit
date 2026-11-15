# Data

## Source

All formation energies come from the public dataset released with

> M. H. Rahman, P. Gollapalli, P. Manganaris, S. K. Yadav, G. Pilania, B. DeCost,
> K. Choudhary and A. Mannodi-Kanakkithodi, *Accelerating defect predictions in
> semiconductors using graph neural networks*, APL Machine Learning 2, 016122 (2024).
> Repository: <https://github.com/msehabibur/defect_GNN_gen_1> (commit `d4853e1`).

which extends the zinc blende defect dataset of Mannodi-Kanakkithodi et al.,
Patterns 3, 100450 (2022). Please cite these works when using the tables below.
`scripts/fetch_data.sh` downloads the upstream repository at the pinned commit
and extracts the label files into `data/raw/` (not tracked).

The upstream files list one row per ionic step of each DFT relaxation
(`<id>-<host>-<defect>-POSCAR<k>.cif, formation energy`). Step 0 is the defect
on its ideal site; the last step is the relaxed defect. We keep only these two
numbers per trajectory plus the number of recorded steps. No structures are
redistributed here.

## Processed tables

`processed/defects_neutral.csv` (1549 rows) is the table used in the main analysis:
neutral defects, A-rich chemical potentials.

| column | meaning |
|---|---|
| `defect_id` | `host:site:species:charge` |
| `host`, `family` | zinc blende host and its family (IV, III-V, II-VI) |
| `host_A`, `host_B` | species on the two sublattices (equal for elemental hosts) |
| `defect_class` | vacancy, antisite, sub_impurity, self_interstitial, int_impurity |
| `site` | A, B (lattice sites) or i_A, i_B, i_neut (tetrahedral interstitial sites, upstream naming) |
| `element`, `replaced` | species placed / removed (empty when not applicable) |
| `n_frames` | recorded ionic steps of the relaxation |
| `e_unrelaxed` | formation energy at step 0 (eV) |
| `e_relaxed` | formation energy at the last step (eV) |
| `e_min` | lowest formation energy along the trajectory (eV) |
| `e_relax` | `e_unrelaxed - e_relaxed` (eV) |

`processed/defects_charged.csv` has the same columns for charge states ±1 and ±2.
These relaxations start from the neutral relaxed geometry rather than from the
ideal site, so `e_unrelaxed` there means "single point at the neutral geometry".

`processed/defects_censored.csv` lists neutral trajectories that were excluded
because the ideal-site frame is not available (frames above 15 eV were removed
upstream for the impurity set; the same cap is applied to native defects).

`processed/defects_single_frame.csv` lists neutral trajectories with a single
recorded frame, used only in a sensitivity check.

## Filtering

| rule | neutral impurity | neutral native |
|---|---|---|
| trajectories in upstream files | 1430 | 310 |
| no ideal-site frame (first step > 0) | 59 | 0 |
| ideal-site energy above the 15 eV cap | 0 | 20 |
| corrupt energies (below -10 eV, or energy rising by > 0.1 eV) | 3 | 0 |
| single recorded frame | 50 | 36 |
| kept | 1318 | 254 |

Defects that appear in both subsets (a host species listed as an "impurity")
are counted once, which leaves 1549 rows.

## Elemental properties

`external/elements.csv` is a subset of the Magpie elemental property tables
(L. Ward et al., npj Comput. Mater. 2, 16028 (2016)) as distributed with
matminer (BSD-style licence, copyright The Regents of the University of
California). Units follow the Magpie tables: `rcov` in pm, `ie1` in eV, `ea` in
kJ/mol, `polariz` in Å^3, `vol_pa` in Å^3 per atom.
