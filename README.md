# Bombing-LMD

Code, schemas, and result files for the article "Bombing-LMD: k-Autonomous Centroid-Based Clustering of Mixed-Type Data with a Learnable Distance and Post Hoc Merging" (Nuraeni, Syukur, Marjuni, Rijati, Kurniadi), submitted to the International Journal of Intelligent Engineering and Systems (IJIES).

Bombing-LMD clusters datasets with numeric, nominal, and ordinal attributes. It builds a joint nearest neighbor graph (jNNG) with a learnable mixed-type distance (LMD), detects the number of clusters by density bombing, merges over-segmented clusters by the Degree of Adjacency (Post Hoc Merger), and refines the partition with a learnable distance and Silhouette Recovery. The number of clusters is estimated from the data; the topology parameters (tau, T2, dc) are fixed across datasets and are not tuned per dataset.

## Repository layout

| Path | Content |
|---|---|
| `subsampling_bombinglmd_lib.py` | Bombing-LMD (`LMDBombingHybrid`), the subsampling wrapper (`SubsampledBombingLMDHybrid`), distances, evaluation metrics, and the one-step k-prototypes imputation |
| `vectorized_patches.py` | NumPy-vectorized replacements that give the same results as the loop-based code (speed only) |
| `amphm_baseline.py` | Unvalidated re-implementation of AMPHM (see `BASELINES.md`) |
| `run_full.py`, `run_sensitivity.py` | Comparison on the 12 real datasets and the one-factor-at-a-time sensitivity analysis (Sections 4.2 and 4.5) |
| `run_figure6_kproto.py` | K-Prototypes under three k protocols, compared with Bombing-LMD (Figure 6) |
| `ablation_study_pipeline2.ipynb` | Ablation studies (Sections 4.3 and 4.4) |
| `diagnostik_bombinglmd.py`, `subsampling_bombinglmd_clustering.ipynb` | Diagnostics and exploratory runs |
| `new_53_dataset_schemas.json` | Attribute types, ordinal category orders, and label column of every dataset |
| `bombing_lmd_hybrid_15_datasets_REVISI.csv` | Bombing-LMD results; also the input of `run_figure6_kproto.py` |
| `results/` | Sensitivity, ablation, baseline comparison, and AMPHM output files |
| `docs/` | Flowcharts of the experiment scripts (Mermaid) |
| `DATASETS.md`, `BASELINES.md` | Dataset sources and baseline implementation sources |

## Installation

Python 3.9 or later.

```
pip install -r requirements.txt
```

## Data

The datasets are not included. Follow `DATASETS.md` to place them in `53_mix_datasets/`.

## Reproducing the results

Run every script from the repository root. All runs are resumable: finished datasets or settings are skipped when a script is started again.

1. Comparison on the 12 real datasets: `python run_full.py`
2. Sensitivity analysis: `python run_sensitivity.py`
3. Figure 6: `python run_figure6_kproto.py`
4. Ablation studies: open `ablation_study_pipeline2.ipynb`

Results depend on library-level thread scheduling and hash ordering, so the scripts fix `PYTHONHASHSEED=0`, one BLAS/OpenMP thread, and global NumPy seed 42. The scripts restart themselves with these settings on Linux and macOS. On Windows, set them before starting Python, for example in PowerShell (PYTHONUTF8 avoids an encoding error from the progress messages):

```
$env:PYTHONUTF8="1"; $env:PYTHONHASHSEED="0"; $env:OPENBLAS_NUM_THREADS="1"; $env:MKL_NUM_THREADS="1"; $env:OMP_NUM_THREADS="1"; $env:NUMEXPR_NUM_THREADS="1"
python run_full.py
```

Telco Customer Churn is not processed by Bombing-LMD because of its size, so the comparison covers 12 datasets for which every compared method produced results.

## Limitations

- The AMPHM output is from an unvalidated re-implementation and is not used in the rankings or statistical tests of the article.
- The estimated number of clusters is unstable on several datasets (for example Obesity and Stroke); see Table 8a of the article.
- Memory consumption was not measured; the pairwise distance matrix needs O(n^2) memory.

## License

MIT, see `LICENSE`. The datasets keep the licenses of their original providers.

## Citation

Please cite the article once published. A `CITATION.cff` file will be added with the DOI.
