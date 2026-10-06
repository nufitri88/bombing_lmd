"""
run_sensitivity.py — Re-run PENUH Uji Sensitivitas Parameter tau, T2, dc
(Bagian 4.2 draft, Gambar 1)
=======================================================================
Desain OFAT (One-Factor-At-a-Time), 3 faktor, mengikuti persis desain pada
bombing_lmd_hybrid_REVISI.ipynb / bombing_lmd_hybrid_2_REVISI.ipynb:
  - Variasi tau {0.4, 0.5, 0.6, 0.7, 0.8}  dengan T2=0.2, dc=0.15 tetap
  - Variasi T2  {0.1, 0.15, 0.2, 0.25, 0.3} dengan tau=0.6, dc=0.15 tetap
  - Variasi dc  {0.05, 0.1, 0.15, 0.2, 0.25} dengan tau=0.6, T2=0.2 tetap
  - Titik pusat (tau=0.6, T2=0.2, dc=0.15) hanya dijalankan sekali (dedup)
  -> 13 skenario x 5 dataset = 65 kombinasi total.

Memakai kode PRODUKSI VERSI REVISI (DE dinonaktifkan) DAN patch numpy
tervektorisasi yang sudah diverifikasi identik dengan versi loop-Python
murni (lihat test_equivalence.py dan docstring run_full.py). Ini penting
di sini karena beberapa kombinasi (khususnya Heart-Disease/NPHA pada tau
tinggi) pernah memakan hingga ~11 jam per kombinasi dengan kode loop-Python
murni -- dengan patch ini, waktu tersebut turun signifikan (terbukti 21.7x
pada Automobile; rasio sesungguhnya bisa bervariasi menurut ukuran dataset).

CARA PAKAI
----------
1. Letakkan file ini SATU FOLDER dengan:
     - subsampling_bombinglmd_lib.py   (kelas & fungsi, versi revisi/DE-fix)
     - vectorized_patches.py           (patch numpy tervektorisasi)
     - new_53_dataset_schemas.json
     - folder 53_mix_datasets/         (berisi 5 dataset di SENSITIVITY_DATASET_LIST)
2. pip install pandas numpy scikit-learn scipy scikit_posthocs seaborn matplotlib
3. Jalankan:  python run_sensitivity.py
   (resume-safe: hasil disimpan ke CSV setelah SETIAP kombinasi selesai,
    kombinasi yang sudah ada otomatis dilewati saat dijalankan ulang --
    aman dihentikan kapan saja, misalnya kalau butuh mematikan komputer)

OUTPUT
------
sensitivity_tau_T2_dc_results_REVISI.csv -- satu baris per kombinasi, kolom:
Dataset, K_true, varied_factor, tau, T2, dc, K_detected, SIL, Dunn, ARI,
runtime_sec, status. Ringkasan stabilitas (rentang & variasi relatif
terhadap baseline) dicetak otomatis di akhir, mengikuti logika
summarize_stability() pada notebook asli.
"""
import os
import sys

# [PENTING - Reproducibility] Lihat penjelasan lengkap di docstring/komentar
# run_full.py. Ringkas: thread BLAS multi-core dan PYTHONHASHSEED yang
# diacak Python per-proses adalah sumber nondeterminisme UTAMA (lebih besar
# dari random_state), diverifikasi lewat dua run independen pada
# Heart-Disease yang baru identik (K=13=K=13) setelah keduanya dikunci.
_REQUIRED_ENV = {"PYTHONHASHSEED": "0", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}
if any(os.environ.get(k) != v for k, v in _REQUIRED_ENV.items()):
    env = os.environ.copy()
    env.update(_REQUIRED_ENV)
    os.execvpe(sys.executable, [sys.executable] + sys.argv, env)

import time
import json
import traceback
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import adjusted_rand_score

import subsampling_bombinglmd_lib as LIB
import vectorized_patches as VP

# [PENTING - Reproducibility] Lihat catatan yang sama pada run_full.py:
# beberapa titik kode ASLI memakai np.random global yang tidak di-seed oleh
# random_state. Seed berikut mengunci itu agar 65 kombinasi ini deterministik
# dan bisa direproduksi ulang -- tanpa mengubah logika algoritma.
np.random.seed(42)

VP.apply_patches(LIB)

json_path = "new_53_dataset_schemas.json"
dataset_folder = "53_mix_datasets"
CANDIDATE_LABELS = ["class", "Class", "label", "Label", "target", "Target"]

with open(json_path, "r") as f:
    schema_all = {entry["filename"]: entry for entry in json.load(f)}

SENSITIVITY_DATASET_LIST = [
    "new_01_automobile.csv",
    "new_03_heart_disease.csv",
    "new_10_npha-doctor-visits.csv",
    "new_13_obesity.csv",
    "new_04_australian_credit.csv",
    "new_15_stroke-data.csv",
]

TAU_DEFAULT, T2_DEFAULT, DC_DEFAULT = 0.6, 0.2, 0.15
TAU_RANGE = [0.4, 0.5, 0.6, 0.7, 0.8]
T2_RANGE = [0.1, 0.15, 0.2, 0.25, 0.3]
DC_RANGE = [0.05, 0.1, 0.15, 0.2, 0.25]

_scenarios_raw = (
    [("tau", t, T2_DEFAULT, DC_DEFAULT) for t in TAU_RANGE]
    + [("T2", TAU_DEFAULT, t2, DC_DEFAULT) for t2 in T2_RANGE]
    + [("dc", TAU_DEFAULT, T2_DEFAULT, dc) for dc in DC_RANGE]
)
_seen, SCENARIOS = set(), []
for factor, tau_val, t2_val, dc_val in _scenarios_raw:
    key = (round(tau_val, 3), round(t2_val, 3), round(dc_val, 3))
    if key in _seen:
        continue
    _seen.add(key)
    SCENARIOS.append((factor, tau_val, t2_val, dc_val))

print(f"Total skenario per dataset: {len(SCENARIOS)}  |  Total kombinasi: {len(SCENARIOS) * len(SENSITIVITY_DATASET_LIST)}")

output_csv = "sensitivity_tau_T2_dc_results_REVISI.csv"

if os.path.exists(output_csv):
    existing_results_df = pd.read_csv(output_csv)
    if "dc" not in existing_results_df.columns:
        existing_results_df["dc"] = DC_DEFAULT
    processed_keys = set(
        zip(
            existing_results_df["Dataset"],
            existing_results_df["varied_factor"],
            existing_results_df["tau"].round(3),
            existing_results_df["T2"].round(3),
            existing_results_df["dc"].round(3),
        )
    )
    all_results = existing_results_df.to_dict("records")
    print(f"🔄 Resuming. {len(processed_keys)} kombinasi sudah diproses.")
else:
    processed_keys = set()
    all_results = []
    print("🆕 Memulai run baru (CSV belum ada).")

t_start_all = time.time()

for filename in SENSITIVITY_DATASET_LIST:
    if filename not in schema_all:
        print(f"⚠️ {filename} tidak ditemukan di schema. Dilewati.")
        continue
    meta = schema_all[filename]
    print(f"\n{'='*70}\n📂 Processing dataset: {filename}\n{'='*70}")

    try:
        path = os.path.join(dataset_folder, filename)
        if not os.path.exists(path):
            print(f"⚠️ {filename} not found di folder dataset.")
            continue

        df = pd.read_csv(path).replace([" ?", "?"], np.nan)

        label_col = meta.get("label_column") or next((c for c in CANDIDATE_LABELS if c in df.columns), None)
        if not label_col:
            print("⚠️ No label column.")
            continue

        le = LabelEncoder()
        y_true = le.fit_transform(df[label_col].astype(str))
        k_true = len(np.unique(y_true))
        X = df.drop(columns=[label_col] + [c for c in df.columns if c.lower() == "id"])

        attr_types, attr_orders = {}, {}
        for col in meta.get("nominal_attributes", []) + meta.get("binary_attributes", []):
            if col in X.columns:
                attr_types[col] = "Nominal"
        for col in meta.get("numeric_attributes", []):
            if col in X.columns:
                attr_types[col] = "Numeric"
        for col in meta.get("ordinal_attributes", []):
            if col in X.columns:
                attr_types[col] = "Ordinal"
                attr_orders[col] = meta["ordinal_orders"].get(col, sorted(X[col].dropna().unique()))

        numeric_cols = [c for c, t in attr_types.items() if t == "Numeric"]
        cat_cols = [c for c, t in attr_types.items() if t != "Numeric"]
        if X.isnull().sum().any():
            X_imputed, _, _ = LIB.one_step_kprototypes(df=X.copy(), k=k_true, numeric_cols=numeric_cols, cat_cols=cat_cols)
        else:
            X_imputed = X.copy()

        for factor, tau_val, t2_val, dc_val in SCENARIOS:
            key = (filename, factor, round(tau_val, 3), round(t2_val, 3), round(dc_val, 3))
            if key in processed_keys:
                print(f"⏭️ Skip {filename} | factor={factor} tau={tau_val} T2={t2_val} dc={dc_val} (sudah ada)")
                continue

            print(f"\n▶️ {filename} | varied={factor} | tau={tau_val} | T2={t2_val} | dc={dc_val}")

            try:
                model = LIB.LMDBombingHybrid(
                    attr_types=attr_types,
                    ordinal_orders=attr_orders,
                    max_iter_refine=20,
                    random_state=42,
                    verbose=False,
                )
                model.tau = tau_val
                model.T2 = t2_val
                model.dc = dc_val

                # Re-seed TEPAT sebelum fit() setiap kombinasi -- lihat
                # catatan reproducibility di atas file ini.
                np.random.seed(42)
                t0 = time.time()
                model.fit(X_imputed)
                runtime = time.time() - t0

                final_labels = model.labels_
                detected_K = model.K_auto_

                mask_valid = y_true != -1
                y_pred_valid = final_labels[mask_valid]
                ari = adjusted_rand_score(y_true[mask_valid], y_pred_valid)

                D_lmd = LIB.compute_lmd_distance_matrix(X_imputed, model)
                D_valid = D_lmd[np.ix_(mask_valid, mask_valid)]
                try:
                    sil = LIB.silhouette_score_ewlmd(D_valid, y_pred_valid)
                    dunn = LIB.dunn_index_ewlmd(D_valid, y_pred_valid)
                except Exception as e:
                    print(f"⚠️ Metrik internal gagal: {e}")
                    sil = dunn = np.nan

                status = "ok"
            except Exception as e:
                print(f"❌ ERROR pada {filename} (tau={tau_val}, T2={t2_val}, dc={dc_val}): {e}")
                traceback.print_exc()
                detected_K = sil = dunn = ari = np.nan
                runtime = np.nan
                status = f"error: {e}"

            record = {
                "Dataset": filename, "K_true": k_true, "varied_factor": factor,
                "tau": tau_val, "T2": t2_val, "dc": dc_val,
                "K_detected": detected_K, "SIL": sil, "Dunn": dunn, "ARI": ari,
                "runtime_sec": round(runtime, 3) if runtime == runtime else runtime,
                "status": status,
            }
            all_results.append(record)
            processed_keys.add(key)
            print(f"   -> K_detected={detected_K}, SIL={sil}, Dunn={dunn}, ARI={ari}, runtime={runtime}")

            pd.DataFrame(all_results).to_csv(output_csv, index=False)

    except Exception as e:
        print(f"\n❌ ERROR FATAL pada {filename}: {e}")
        traceback.print_exc()
        pd.DataFrame(all_results).to_csv(output_csv, index=False)
        continue

print(f"\n🎉 Uji sensitivitas selesai untuk seluruh dataset & skenario. Total waktu: {time.time() - t_start_all:.1f}s")
print(f"💾 Hasil tersimpan di {output_csv}")


# =============================================================================
# RINGKASAN STABILITAS (identik dengan logika notebook asli)
# =============================================================================
def summarize_stability(df, rel_variation_threshold=0.10):
    rows = []
    for (dataset, factor), group in df.groupby(["Dataset", "varied_factor"]):
        baseline_row = group[
            (group["tau"].round(3) == TAU_DEFAULT)
            & (group["T2"].round(3) == T2_DEFAULT)
            & (group["dc"].round(3) == DC_DEFAULT)
        ]
        for metric in ["K_detected", "SIL", "Dunn", "ARI"]:
            vals = pd.to_numeric(group[metric], errors="coerce").dropna()
            if len(vals) == 0:
                continue
            baseline = (
                pd.to_numeric(baseline_row[metric], errors="coerce").values[0]
                if len(baseline_row)
                else vals.mean()
            )
            rng = vals.max() - vals.min()
            rel_var = abs(rng / baseline) if baseline else np.nan
            rows.append({
                "Dataset": dataset, "varied_factor": factor, "metric": metric,
                "baseline": baseline, "min": vals.min(), "max": vals.max(),
                "range": rng, "rel_variation": rel_var,
                "stabil (<10%)": (rel_var < rel_variation_threshold) if rel_var == rel_var else None,
            })
    return pd.DataFrame(rows)


final_df = pd.DataFrame(all_results)
if len(final_df) and (final_df["status"] == "ok").any():
    stability_df = summarize_stability(final_df[final_df["status"] == "ok"])
    stability_df.to_csv("sensitivity_stability_summary_REVISI.csv", index=False)
    print("\n=== RINGKASAN STABILITAS (disimpan ke sensitivity_stability_summary_REVISI.csv) ===")
    print(stability_df.to_string(index=False))
