# Baseline implementations

The baseline results in `results/komparasi_RC3_12dataset.csv` were produced outside this repository, except for AMPHM and K-Prototypes. The table must be completed with the exact source and version of every implementation before the repository is cited in the article.

| Method | Implementation source | Version / commit | Key settings | Run in this repository |
|---|---|---|---|---|
| K-Prototype (classic) | to be completed | to be completed | k = number of classes | `run_figure6_kproto.py` (kmodes, Huang init, n_init = 10) for Figure 6 |
| K-Prototype + KDSum, + Gower | to be completed | to be completed | to be completed | no |
| MFCM | to be completed | to be completed | to be completed | no |
| HARR-V | to be completed | to be completed | to be completed | no |
| KAMILA | to be completed | to be completed | to be completed | no |
| PDQ | to be completed | to be completed | to be completed | no |
| AMDPC (estimates k) | to be completed | to be completed | to be completed | no |
| EW-LMD (A) | to be completed | to be completed | k = number of classes | no |
| AMPHM | `amphm_baseline.py` (re-implementation) | this repository | k = number of classes, merge_factor = 2.0 | `run_full.py` |

AMPHM: the original source code is not public. `amphm_baseline.py` is a re-implementation from Algorithms 1 and 2 and Eqs. (5) to (12) of Zhang et al., Complex Intell. Syst. 11:84 (2025), doi:10.1007/s40747-024-01695-7. It has not been validated against the published results and departs from the original in several respects (attribute weights of Eqs. 8-9 not implemented; conditional distributions use categorical attributes only; neighbor rank fixed to sqrt(n); the number of representatives kept per merging round is a reconstruction; ordinal attributes treated as numeric). Its output (`results/amphm_baseline_15_datasets.csv`) is provided for transparency only.
