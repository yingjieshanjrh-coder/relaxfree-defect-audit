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

`processed/defects_neutral.csv` (1549 rows) holds all kept neutral defects under
A-rich chemical potentials. The main analysis uses the 813 rows with
`ideal_start == True`; `relaxfree.dataset.load_neutral` applies this filter by
default.

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
| `start_n_displaced` | atoms of the first frame more than 0.02 Å from an ideal zinc blende site |
| `start_max_disp` | largest such displacement (Å) |
| `ideal_start` | first frame is the ideal-site defect (`start_max_disp <= 0.02`) |

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

The upstream workflow started many relaxations from randomly displaced
geometries (`atomsk -disturb` with amplitudes up to 0.3 Å). The first frame of
each neutral trajectory is therefore compared with the ideal sites of the
64-atom zinc blende supercell, whose fractional coordinates (including the
tetrahedral interstitial sites) are multiples of 1/16. 813 trajectories start
within 0.02 Å of those sites and form the main analysis; the other 736 start
with on average 35 atoms displaced by up to 0.7 Å. 722 of these are in II-VI
hosts, including every defect in CdS and CdSe.

## External chalcogenide vacancies

`processed/external_chalcogenide_vacancies.csv` (142 rows) is derived from
`defect_relaxations.tar.gz` of Zenodo record 10.5281/zenodo.10579527, the
HSE06 relaxations of cation vacancies in about 50 chalcogenide and mixed-anion
hosts from I. Mosquera-Lois, S. R. Kavanagh, A. M. Ganose and A. Walsh, npj
Comput. Mater. 10, 121 (2024). Each vacancy was relaxed from its ideal,
high-symmetry geometry ("Unperturbed") and from 14-15 bond-distorted or rattled
starts. `scripts/extract_external_vacancies.py` streams the archive and writes
the energy of every ionic step of every run; `relaxfree.external` joins the
restarted segments of a run in time order and reduces them to one row per
vacancy.

| column | meaning |
|---|---|
| `subset`, `host`, `defect`, `element`, `charge` | as labelled in the archive (`v_<element>_s<site>_<charge>`) |
| `unp_e_first`, `unp_e_last` | first and last total energy of the Unperturbed run (eV) |
| `e_relax` | `unp_e_first - unp_e_last`, relaxation energy from the ideal site (eV) |
| `e_best_final` | lowest final energy over all starts with a converged final step (eV) |
| `e_reconstruction` | `unp_e_last - e_best_final`, energy missed by relaxing from the ideal site (eV) |
| `unp_max_join` | largest energy jump between consecutive restart segments (eV) |
| `unp_n_unconverged` | ionic steps that hit the electronic step limit (NELM) |
| `usable` | final segment complete, first and last steps converged, joins below 0.1 eV |
| `clean` | usable, no unconverged step and no energy rise above 0.05 eV along the run |

The main external comparison uses the 65 neutral vacancies flagged `clean`;
the 83 `usable` ones give a sensitivity check. The first frame of every
Unperturbed run differs from the first frame of the rattled run of the same
vacancy by about 0.3 Å for 98 % of the atoms, consistent with the Unperturbed
start being the unrattled ideal geometry. The archive passed a full gzip
integrity check. Two archived segments stored without compression (BiI3,
two distorted starts) were not read by the first extraction pass; neither is an
Unperturbed run, and the published script handles them.

## Elemental properties

`external/elements.csv` is a subset of the Magpie elemental property tables
(L. Ward et al., npj Comput. Mater. 2, 16028 (2016)) as distributed with
matminer (BSD-style licence, copyright The Regents of the University of
California). Units follow the Magpie tables: `rcov` in pm, `ie1` in eV, `ea` in
kJ/mol, `polariz` in Å^3, `vol_pa` in Å^3 per atom.
