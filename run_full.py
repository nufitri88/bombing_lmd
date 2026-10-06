"""
run_full.py — Re-run PENUH Tabel 6-8 (Studi Komparasi, Bagian 4.5-4.7)
  + Baseline AMPHM (Langkah 3 -- menjawab komentar reviewer soal
    "principal state-of-the-art comparison is incomplete")
=======================================================================
Memakai kode PRODUKSI VERSI REVISI (DE dinonaktifkan sesuai hasil Studi
Ablasi -- lihat "Catatan Resmi Langkah 1") DAN patch numpy tervektorisasi
yang sudah diverifikasi menghasilkan angka IDENTIK dengan versi loop-Python
murni (lihat test_equivalence.py: ARI=1.000000, pergeseran centroid=0,
speedup 21.7x pada dataset Automobile).

Baseline AMPHM (amphm_baseline.py) dijalankan dengan k = k_ground_truth
(BUKAN otonom -- lihat "panduan-revisi-langkah3-implementasi-amphm.md",
Bagian 2: publikasi asli AMPHM tetap memerlukan k sebagai input, sehingga ia
masuk kelompok "ground_truth" pada kolom k_mode, sejajar dengan EW-LMD,
HARR-V, K-Prototype, MFCM -- bukan sejajar dengan Bombing-LMD/AMDPC).

CARA PAKAI
----------
1. Letakkan file ini SATU FOLDER dengan:
     - subsampling_bombinglmd_lib.py   (kelas & fungsi, versi revisi/DE-fix)
     - vectorized_patches.py           (patch numpy tervektorisasi)
     - amphm_baseline.py               (baseline AMPHM, Langkah 3)
     - new_53_dataset_schemas.json
     - folder 53_mix_datasets/         (berisi 14 file dataset di DATASET_LIST)
2. pip install pandas numpy scikit-learn scipy scikit_posthocs seaborn matplotlib
3. Jalankan:  python run_full.py
   (aman dihentikan & dilanjutkan kapan saja -- hasil per dataset langsung
    disimpan ke CSV, dataset yang sudah selesai otomatis dilewati saat
    dijalankan ulang. Bombing-LMD dan AMPHM masing-masing punya CSV dan
    status resume SENDIRI-SENDIRI, jadi kegagalan pada salah satu baseline
    tidak menghentikan yang lain.)

OUTPUT
------
bombing_lmd_hybrid_15_datasets_REVISI.csv  -- satu baris per dataset, kolom
sama persis dengan Tabel 6-8 draft: Dataset, Label, Cluster, NMI, ARI, FMI,
PUR, SIL, Dunn, CU, ER, Time.

amphm_baseline_15_datasets.csv  -- satu baris per dataset untuk baseline
AMPHM, kolom sama dengan di atas PLUS kolom "k_mode" (selalu "ground_truth")
dan "used_subsampling" (True jika n > 5000, mengikuti anggaran komputasi
yang sama seperti SubsampledBombingLMDHybrid).
"""
import os
import sys

# =============================================================================
# [PENTING - Reproducibility] Kunci sumber nondeterminisme di LUAR algoritma
# sebelum numpy/scipy diimpor:
#   1. Thread BLAS (OpenBLAS/MKL) -- operasi matriks multi-thread bisa
#      mengubah URUTAN penjumlahan floating-point antar-run, yang lewat
#      ambang batas diskret di Bombing/Post-Hoc-Merger bisa mengubah K
#      terdeteksi. Dipaksa 1 thread agar deterministik.
#   2. PYTHONHASHSEED -- Python mengacak urutan iterasi set() setiap kali
#      proses baru dimulai (kecuali variabel ini di-set SEBELUM interpreter
#      jalan, makanya proses di-restart otomatis di sini via os.execvpe).
#      Ini relevan juga untuk AMPHM, yang memakai set() pada representative_local.
# Diverifikasi: dua run independen dengan pengaturan ini pada dataset
# Heart-Disease menghasilkan K=13 yang identik pada keduanya (sebelumnya
# bervariasi 2/7/9/13 antar-run tanpa pengaturan ini).
_REQUIRED_ENV = {"PYTHONHASHSEED": "0", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}
if any(os.environ.get(k) != v for k, v in _REQUIRED_ENV.items()):
    env = os.environ.copy()
    env.update(_REQUIRED_ENV)
    os.execvpe(sys.executable, [sys.executable] + sys.argv, env)

import time
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, fowlkes_mallows_score

import subsampling_bombinglmd_lib as LIB
import amphm_baseline as AMPHM_LIB
import vectorized_patches as VP

# [PENTING - Reproducibility] Beberapa titik pada kode ASLI (PostHocMerger,
# deteksi trap Silhouette Recovery, perturbasi label, imputasi k-prototypes)
# memanggil np.random.choice/np.random.rand GLOBAL, bukan self._rng yang
# di-seed dari random_state -- sehingga hasil bisa berbeda antar-run tanpa
# ini. Seed global berikut MENGUNCI sumber acak tsb (tidak mengubah logika
# algoritma sama sekali) agar seluruh re-run ini deterministik & dapat
# direproduksi ulang kapan saja.
np.random.seed(42)

# Tempel patch tervektorisasi ke kelas LMDBombingHybrid milik LIB.
# Hasil numerik dijamin identik dengan versi loop-python murni (lihat
# docstring modul ini) -- satu-satunya perbedaan adalah kecepatan (~20x).
VP.apply_patches(LIB)

# -------------------------------------------------------------------------
# Konfigurasi (identik dengan blok "4. MAIN EXECUTION" pada
# subsampling_bombinglmd.py versi revisi, hanya dipisah agar bisa diimpor
# sebagai modul tanpa efek samping)
# -------------------------------------------------------------------------
json_path = "new_53_dataset_schemas.json"
dataset_folder = "53_mix_datasets"
CANDIDATE_LABELS = ["class", "Class", "label", "Label", "target", "Target"]

with open(json_path, "r") as f:
    schema_all = {entry["filename"]: entry for entry in json.load(f)}

DATASET_LIST = [
    "new_01_automobile.csv",
    "new_02_german_credit.csv",
    "new_03_heart_disease.csv",
    "new_04_australian_credit.csv",
    "new_05_student_performace_mat.csv",
    "new_07_cylinder-band.csv",
    "new_08_exasens.csv",
    "new_09_abalone.csv",
    "new_10_npha-doctor-visits.csv",
    "new_11_cirrhosis.csv",
    "new_12_caesar.csv",
    "new_13_obesity.csv",
    "new_14_heart_failure.csv",
    "new_15_stroke-data.csv",
]

output_csv = "bombing_lmd_hybrid_15_datasets_REVISI.csv"
output_csv_amphm = "amphm_baseline_15_datasets.csv"

# -------------------------------------------------------------------------
# Resume capability -- Bombing-LMD dan AMPHM masing-masing punya status
# resume SENDIRI (dua CSV terpisah), supaya kegagalan/penghentian pada satu
# baseline tidak memblokir yang lain saat skrip dijalankan ulang.
# -------------------------------------------------------------------------
if os.path.exists(output_csv):
    existing_results_df = pd.read_csv(output_csv)
    processed_datasets = existing_results_df["Dataset"].tolist()
    all_results = existing_results_df.to_dict("records")
    print(f"🔄 [Bombing-LMD] Resuming dari run sebelumnya. {len(processed_datasets)} dataset sudah diproses.")
else:
    processed_datasets = []
    all_results = []
    print("🆕 [Bombing-LMD] Memulai run baru (CSV belum ada).")

if os.path.exists(output_csv_amphm):
    existing_results_amphm_df = pd.read_csv(output_csv_amphm)
    processed_datasets_amphm = existing_results_amphm_df["Dataset"].tolist()
    all_results_amphm = existing_results_amphm_df.to_dict("records")
    print(f"🔄 [AMPHM] Resuming dari run sebelumnya. {len(processed_datasets_amphm)} dataset sudah diproses.")
else:
    processed_datasets_amphm = []
    all_results_amphm = []
    print("🆕 [AMPHM] Memulai run baru (CSV belum ada).")

t_start_all = time.time()

for filename, meta in schema_all.items():
    if filename not in DATASET_LIST:
        continue
    if filename in processed_datasets and filename in processed_datasets_amphm:
        print(f"\n⏭️ Skipping {filename} (sudah ada di kedua CSV: Bombing-LMD & AMPHM).")
        continue

    print(f"\n{'='*70}\n📂 Processing: {filename}\n{'='*70}")

    # -----------------------------------------------------------------
    # Pra-pemrosesan dataset -- DIBAGIKAN antara Bombing-LMD dan AMPHM,
    # supaya keduanya dievaluasi pada X_imputed, attr_types, dan k
    # ground-truth yang PERSIS SAMA (protokol identik, Langkah 2).
    # -----------------------------------------------------------------
    try:
        path = os.path.join(dataset_folder, filename)
        if not os.path.exists(path):
            print(f"⚠️ {filename} not found.")
            continue

        df = pd.read_csv(path).replace([" ?", "?"], np.nan)

        label_col = meta.get("label_column") or next((c for c in CANDIDATE_LABELS if c in df.columns), None)
        if not label_col:
            print("⚠️ No label column.")
            continue

        le = LabelEncoder()
        y_true = le.fit_transform(df[label_col].astype(str))
        k = len(np.unique(y_true))
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
            X_imputed, _, _ = LIB.one_step_kprototypes(df=X.copy(), k=k, numeric_cols=numeric_cols, cat_cols=cat_cols)
        else:
            X_imputed = X.copy()

        mask_valid = y_true != -1

    except Exception as e:
        import traceback
        print(f"\n❌ ERROR pra-pemrosesan pada {filename}: {e}")
        traceback.print_exc()
        continue

    # -----------------------------------------------------------------
    # BASELINE 1/2: Bombing-LMD (subsampled, K otonom)
    # -----------------------------------------------------------------
    if filename in processed_datasets:
        print(f"⏭️ [Bombing-LMD] Skipping {filename} (sudah ada di CSV).")
    else:
        try:
            subsampled_model = LIB.SubsampledBombingLMDHybrid(
                base_model_class=LIB.LMDBombingHybrid,
                attr_types=attr_types,
                ordinal_orders=attr_orders,
                n_sample=5000,
                threshold=5000,
                random_state=42,
            )

            # Re-seed TEPAT sebelum fit() setiap dataset -- np.random GLOBAL
            # dikonsumsi secara berurutan lintas dataset (lihat catatan di atas
            # file ini), sehingga seed sekali di awal skrip TIDAK cukup untuk
            # menjamin dataset ke-2, ke-3, dst. deterministik terlepas urutan
            # proses sebelumnya.
            np.random.seed(42)
            t0 = time.time()
            subsampled_model.fit(X_imputed)
            runtime = time.time() - t0

            final_labels = subsampled_model.labels_
            detected_K = subsampled_model.K_auto_
            print(f"[Bombing-LMD] Detected K: {detected_K}  |  runtime: {runtime:.1f}s")

            y_pred = final_labels
            D_lmd = LIB.compute_lmd_distance_matrix(X_imputed, subsampled_model)
            y_pred_valid = final_labels[mask_valid]
            D_valid = D_lmd[np.ix_(mask_valid, mask_valid)]

            ari = adjusted_rand_score(y_true[mask_valid], y_pred_valid)
            nmi = normalized_mutual_info_score(y_true[mask_valid], y_pred_valid)
            pur = LIB.purity_score(y_true[mask_valid], y_pred_valid)
            fmi = fowlkes_mallows_score(y_true[mask_valid], y_pred_valid)

            try:
                sil = LIB.silhouette_score_ewlmd(D_valid, y_pred_valid)
                dunn = LIB.dunn_index_ewlmd(D_valid, y_pred_valid)
            except Exception as e:
                print(f"⚠️ Warning: Internal metrics failed: {e}")
                sil = dunn = 0.0

            try:
                cut = LIB.category_utility(X_imputed, y_pred)
                enr = LIB.entropy_reduction(X_imputed, y_pred, attr_types)
            except Exception as e:
                print(f"⚠️ Warning: CU/ER metrics failed: {e}")
                cut = enr = 0.0

            print(f"[Bombing-LMD] ARI={ari:.4f}  NMI={nmi:.4f}  PUR={pur:.4f}  FMI={fmi:.4f}  SIL={sil:.4f}  Dunn={dunn:.4f}  CU={cut:.4f}  ER={enr:.4f}")

            dataset_entry = {
                "Dataset": filename, "Label": k, "Cluster": detected_K,
                "NMI": nmi, "ARI": ari, "FMI": fmi, "PUR": pur,
                "SIL": sil, "Dunn": dunn, "CU": cut, "ER": enr, "Time": runtime,
                "k_mode": "autonomous",
            }
            all_results.append(dataset_entry)
            pd.DataFrame(all_results).to_csv(output_csv, index=False)
            print(f"💾 [Bombing-LMD] Progres disimpan ke {output_csv}")

        except Exception as e:
            import traceback
            print(f"\n❌ ERROR [Bombing-LMD] pada {filename}: {e}")
            traceback.print_exc()
            pd.DataFrame(all_results).to_csv(output_csv, index=False)

    # -----------------------------------------------------------------
    # BASELINE 2/2: AMPHM (Langkah 3) -- k = k_ground_truth
    # -----------------------------------------------------------------
    if filename in processed_datasets_amphm:
        print(f"⏭️ [AMPHM] Skipping {filename} (sudah ada di CSV).")
        continue

    try:
        amphm_model = AMPHM_LIB.AMPHMScalable(
            attr_types=attr_types,
            ordinal_orders=attr_orders,
            k=k,                    # k ground-truth -- lihat catatan di kepala file ini
            merge_factor=2.0,
            n_sample=5000,
            threshold=5000,
            random_state=42,
            verbose=False,
        )

        # Re-seed TEPAT sebelum fit() setiap dataset, konsisten dengan
        # perlakuan Bombing-LMD di atas (AMPHMScalable memakai np.random
        # via default_rng ber-seed sendiri, tetapi disamakan di sini demi
        # konsistensi prosedural antar-baseline).
        np.random.seed(42)
        t0_amphm = time.time()
        amphm_model.fit(X_imputed)
        runtime_amphm = time.time() - t0_amphm

        y_pred_amphm = amphm_model.labels_
        detected_K_amphm = len(np.unique(y_pred_amphm))
        print(f"[AMPHM] K (ground-truth dipaksa) = {detected_K_amphm}  |  "
              f"subsampling={amphm_model.used_subsampling_}  |  runtime: {runtime_amphm:.1f}s")

        # Matriks jarak HADM PENUH (profil dibangun dari X_imputed penuh),
        # dipakai utk SIL/Dunn -- sama seperti compute_lmd_distance_matrix()
        # untuk Bombing-LMD di atas, evaluasi akhir selalu pada dataset penuh
        # terlepas dari subsampling yang dipakai saat fit.
        D_amphm = AMPHM_LIB.compute_amphm_distance_matrix(X_imputed, attr_types, attr_orders)
        y_pred_amphm_valid = y_pred_amphm[mask_valid]
        D_amphm_valid = D_amphm[np.ix_(mask_valid, mask_valid)]

        ari_a = adjusted_rand_score(y_true[mask_valid], y_pred_amphm_valid)
        nmi_a = normalized_mutual_info_score(y_true[mask_valid], y_pred_amphm_valid)
        pur_a = LIB.purity_score(y_true[mask_valid], y_pred_amphm_valid)
        fmi_a = fowlkes_mallows_score(y_true[mask_valid], y_pred_amphm_valid)

        try:
            sil_a = LIB.silhouette_score_ewlmd(D_amphm_valid, y_pred_amphm_valid)
            dunn_a = LIB.dunn_index_ewlmd(D_amphm_valid, y_pred_amphm_valid)
        except Exception as e:
            print(f"⚠️ [AMPHM] Warning: Internal metrics failed: {e}")
            sil_a = dunn_a = 0.0

        try:
            cut_a = LIB.category_utility(X_imputed, y_pred_amphm)
            enr_a = LIB.entropy_reduction(X_imputed, y_pred_amphm, attr_types)
        except Exception as e:
            print(f"⚠️ [AMPHM] Warning: CU/ER metrics failed: {e}")
            cut_a = enr_a = 0.0

        print(f"[AMPHM] ARI={ari_a:.4f}  NMI={nmi_a:.4f}  PUR={pur_a:.4f}  FMI={fmi_a:.4f}  "
              f"SIL={sil_a:.4f}  Dunn={dunn_a:.4f}  CU={cut_a:.4f}  ER={enr_a:.4f}")

        dataset_entry_amphm = {
            "Dataset": filename, "Label": k, "Cluster": detected_K_amphm,
            "NMI": nmi_a, "ARI": ari_a, "FMI": fmi_a, "PUR": pur_a,
            "SIL": sil_a, "Dunn": dunn_a, "CU": cut_a, "ER": enr_a, "Time": runtime_amphm,
            "k_mode": "ground_truth", "used_subsampling": amphm_model.used_subsampling_,
        }
        all_results_amphm.append(dataset_entry_amphm)
        pd.DataFrame(all_results_amphm).to_csv(output_csv_amphm, index=False)
        print(f"💾 [AMPHM] Progres disimpan ke {output_csv_amphm}")

    except Exception as e:
        import traceback
        print(f"\n❌ ERROR [AMPHM] pada {filename}: {e}")
        traceback.print_exc()
        pd.DataFrame(all_results_amphm).to_csv(output_csv_amphm, index=False)
        continue

print(f"\n🎉 Selesai. Total waktu: {time.time() - t_start_all:.1f}s")
print("\n=== Bombing-LMD ===")
print(pd.DataFrame(all_results).to_string(index=False))
print("\n=== AMPHM ===")
print(pd.DataFrame(all_results_amphm).to_string(index=False))
