"""
diagnostik_bombinglmd.py -- Uji diagnostik kecil untuk empat butir audit (4 Oktober 2026)
==========================================================================================
Butir No. 2  : Eq. 6a / _validate_merge  -- apakah kriteria DA/radius benar-benar menyaring merger?
Butir No. 4  : jarak nominal/ordinal negatif (dis_const + phi) dan dampaknya pada matriks D
Butir No. 8  : sanity-check baseline AMPHM -- partisi degenerat (satu klaster raksasa)?
Butir No. 18 : K_max (jumlah komponen terhubung jNNG) vs K hasil Bombing vs K akhir -- fragmentasi graf?

PRINSIP: skrip ini TIDAK mengubah algoritma. Ia hanya membungkus (monkeypatch) fungsi
produksi agar mencatat nilai antara, lalu menjalankan fit() persis seperti run_full.py
(seed, pengunci thread, PYTHONHASHSEED sama; np.random state dipulihkan setelah setiap
pengukuran tambahan sehingga urutan bilangan acak tidak bergeser).

CARA PAKAI (letakkan SATU folder dengan run_full.py, subsampling_bombinglmd_lib.py,
vectorized_patches.py, amphm_baseline.py, new_53_dataset_schemas.json, 53_mix_datasets/):
    python diagnostik_bombinglmd.py                       # 6 dataset kecil (default)
    python diagnostik_bombinglmd.py --include-slow        # + Australian Credit (~6 menit)
    python diagnostik_bombinglmd.py --datasets new_01_automobile.csv new_03_heart_disease.csv
    python diagnostik_bombinglmd.py --selftest            # data sintetis, tanpa folder dataset
    python diagnostik_bombinglmd.py --skip-amphm          # hanya butir 2, 4, 18
    python diagnostik_bombinglmd.py --skip-bombing        # hanya butir 8

KELUARAN (folder --out, default diagnostik_out/):
    diag_bombing_ringkasan.csv     satu baris per dataset (butir 2, 4, 18)
    diag_merge_validasi.csv        satu baris per percobaan validasi merger (butir 2)
    diag_amphm_ringkasan.csv       satu baris per dataset x varian pemilihan representatif (butir 8)
"""
import os
import sys

_REQUIRED_ENV = {"PYTHONHASHSEED": "0", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                 "OMP_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}
if any(os.environ.get(k) != v for k, v in _REQUIRED_ENV.items()):
    _env = os.environ.copy()
    _env.update(_REQUIRED_ENV)
    os.execvpe(sys.executable, [sys.executable] + sys.argv, _env)

import io
import json
import time
import argparse
import contextlib
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

import subsampling_bombinglmd_lib as LIB
import amphm_baseline as AMPHM_LIB
import vectorized_patches as VP

VP.apply_patches(LIB)          # sama seperti run_full.py

DEFAULT_DATASETS = [
    "new_01_automobile.csv", "new_02_german_credit.csv", "new_03_heart_disease.csv",
    "new_11_cirrhosis.csv", "new_12_caesar.csv", "new_14_heart_failure.csv",
]
SLOW_DATASETS = ["new_04_australian_credit.csv"]
CANDIDATE_LABELS = ["class", "Class", "label", "Label", "target", "Target"]


# =============================================================================
# 1. PROBE: pembungkus (monkeypatch) yang hanya MENCATAT, tidak mengubah hasil
# =============================================================================
class _Probe:
    def __init__(self):
        self.reset()

    def reset(self):
        self.validations = []      # butir 2
        self.gate = None           # butir 18
        self.jnng_adj = None       # butir 18
        self.min_dist_trace = []   # butir 4: (min, n_neg, n_total) setelah tiap pembaruan phi


PROBE = _Probe()


def _cat_dist_stats(model):
    vals = []
    for d in model.nominal_dist.values():
        vals.extend(d.values())
    for d in model.ordinal_adj_dist.values():
        vals.extend(d.values())
    if not vals:
        return (np.nan, 0, 0)
    v = np.asarray(vals, dtype=float)
    return (float(v.min()), int((v < 0).sum()), int(v.size))


def _install_probes():
    # ---- butir 2: catat setiap panggilan _validate_merge --------------------
    orig_validate = LIB.PostHocMerger._validate_merge

    def validate_logged(self, c1, c2, dist, labels):
        st0 = np.random.get_state()
        result = orig_validate(self, c1, c2, dist, labels)      # persis perilaku produksi
        st1 = np.random.get_state()
        np.random.set_state(st0)                                # ulang sampel acak yang SAMA
        r1 = self._compute_cluster_radius(c1, labels)
        r2 = self._compute_cluster_radius(c2, labels)
        np.random.set_state(st1)                                # pulihkan aliran acak produksi
        eps = 1e-12
        d_cent = self._compute_centroid_distance_ewlmd(c1, c2)
        r_max = max(r1, r2, eps)
        ratio_code = dist / r_max                               # yang dihitung kode: DA / radius
        ratio_alt = d_cent / r_max                              # kandidat konsisten-dimensi
        PROBE.validations.append({
            "c1": int(c1), "c2": int(c2), "DA": float(dist), "r1": float(r1), "r2": float(r2),
            "d_centroid": float(d_cent), "ratio_code": float(ratio_code),
            "pass_code": bool(result), "ratio_alt": float(ratio_alt),
            "pass_alt": bool(ratio_alt <= 1.5),
        })
        return result

    LIB.PostHocMerger._validate_merge = validate_logged

    # ---- butir 18: catat gerbang aktivasi PM (Eq. 6) -----------------------
    orig_gate = LIB.should_merge_parameter_free

    def gate_logged(metrics, K_max, T2=0.2):
        out = orig_gate(metrics, K_max, T2=T2)
        sizes = np.asarray(metrics["cluster_sizes"])
        size_ratio = float(sizes.max() / sizes.min()) if sizes.min() > 0 else np.inf
        PROBE.gate = {
            "K_bombing": int(metrics["K"]), "K_max": int(K_max), "size_ratio": size_ratio,
            "max_DA": float(metrics["max_da"]), "CV_intra": float(metrics["intra_density_cv"]),
            "cond_K_gt_Kmax": bool(metrics["K"] > K_max), "cond_size_ratio_gt10": bool(size_ratio > 10),
            "cond_maxDA_gt_T2": bool(metrics["max_da"] > T2), "cond_CV_gt_0.5": bool(metrics["intra_density_cv"] > 0.5),
            "trigger": bool(out),
        }
        return out

    LIB.should_merge_parameter_free = gate_logged    # fit() mencari nama ini di namespace modul

    # ---- butir 18: simpan adjacency jNNG ------------------------------------
    orig_build = LIB.LMDBombingHybrid._build_jnng

    def build_logged(self, D_scaled):
        out = orig_build(self, D_scaled)
        PROBE.jnng_adj = out[2]
        return out

    LIB.LMDBombingHybrid._build_jnng = build_logged

    # ---- butir 4: nilai minimum jarak kategorikal setelah tiap pembaruan phi -
    orig_phi = LIB.LMDBombingHybrid._update_nominal_ordinal_phi

    def phi_logged(self, X, labels):
        r = orig_phi(self, X, labels)
        PROBE.min_dist_trace.append(_cat_dist_stats(self))
        return r

    LIB.LMDBombingHybrid._update_nominal_ordinal_phi = phi_logged


# =============================================================================
# 2. PEMUAT DATASET (salinan logika run_full.py) + generator sintetis
# =============================================================================
def load_dataset(filename, schema_all, dataset_folder):
    meta = schema_all[filename]
    df = pd.read_csv(os.path.join(dataset_folder, filename)).replace([" ?", "?"], np.nan)
    label_col = meta.get("label_column") or next((c for c in CANDIDATE_LABELS if c in df.columns), None)
    y_true = LabelEncoder().fit_transform(df[label_col].astype(str))
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
        X, _, _ = LIB.one_step_kprototypes(df=X.copy(), k=k, numeric_cols=numeric_cols, cat_cols=cat_cols)
    return X, attr_types, attr_orders, y_true, k


def synthetic_mixed(n=150, k=3, seed=0, nominal_purity=0.8):
    """Data campuran sintetis berstruktur jelas (untuk selftest / sanity check implementasi)."""
    from sklearn.datasets import make_blobs
    rng = np.random.default_rng(seed)
    Xn, y = make_blobs(n_samples=n, centers=k, n_features=3, cluster_std=1.0, center_box=(-8, 8), random_state=seed)
    df = pd.DataFrame(Xn, columns=["num_0", "num_1", "num_2"])
    letters = np.array(list("abcdefgh"))[:k]
    for c in ("nom_0", "nom_1"):
        noisy = rng.random(n) > nominal_purity
        vals = np.where(noisy, rng.integers(0, k, n), y)
        df[c] = letters[vals]
    levels = np.array(["low", "mid", "high"])
    ordv = np.where(rng.random(n) > nominal_purity, rng.integers(0, 3, n), y % 3)
    df["ord_0"] = levels[ordv]
    attr_types = {"num_0": "Numeric", "num_1": "Numeric", "num_2": "Numeric",
                  "nom_0": "Nominal", "nom_1": "Nominal", "ord_0": "Ordinal"}
    return df, attr_types, {"ord_0": ["low", "mid", "high"]}, y, k


# =============================================================================
# 3. DIAGNOSTIK BOMBING-LMD  (butir 2, 4, 18)
# =============================================================================
def _prep_numeric(X, attr_types):
    Xc = X.drop(columns=["Label"], errors="ignore").copy()
    for col in Xc.columns:
        if attr_types[col] == "Numeric":
            Xc[col] = pd.to_numeric(Xc[col], errors="coerce").fillna(Xc[col].mean())
    return Xc


def diag_bombing(name, X, attr_types, orders, y_true, verbose=False, max_n_D=3000):
    PROBE.reset()
    np.random.seed(42)
    model = LIB.LMDBombingHybrid(attr_types=attr_types, ordinal_orders=orders, random_state=42, verbose=False)
    ctx = contextlib.nullcontext() if verbose else contextlib.redirect_stdout(io.StringIO())
    t0 = time.time()
    with ctx:
        model.fit(X)
    runtime = time.time() - t0
    Xc = _prep_numeric(X, attr_types)
    n = len(Xc)
    row = {"dataset": name, "n": n, "k_true": int(len(np.unique(y_true))), "runtime_s": round(runtime, 1),
           "dis_const": model.dis_const, "eta": model.eta_cat}

    # ---------------- butir 18: K_max vs K -------------------------------
    g = PROBE.gate or {}
    row.update({"K_max_komponen": g.get("K_max", model.K_max_), "K_bombing": g.get("K_bombing"),
                "K_akhir": int(len(np.unique(model.labels_))), "gate_trigger": g.get("trigger"),
                "cond_K_gt_Kmax": g.get("cond_K_gt_Kmax"), "cond_size_ratio_gt10": g.get("cond_size_ratio_gt10"),
                "cond_maxDA_gt_T2": g.get("cond_maxDA_gt_T2"), "cond_CV_gt_0.5": g.get("cond_CV_gt_0.5"),
                "max_DA": g.get("max_DA"), "CV_intra": g.get("CV_intra"), "size_ratio": g.get("size_ratio")})
    if PROBE.jnng_adj is not None:
        n_cc, comp = connected_components(PROBE.jnng_adj, directed=False)
        sizes = np.bincount(comp)
        row.update({"n_komponen": int(n_cc), "komponen_terbesar_%": round(100 * sizes.max() / n, 1),
                    "komponen_singleton": int((sizes == 1).sum()), "komponen_ukuran_lt5": int((sizes < 5).sum())})
    lab_sizes = np.bincount(pd.factorize(model.labels_)[0])
    row.update({"klaster_akhir_ukuran_lt5": int((lab_sizes < 5).sum()),
                "klaster_akhir_terbesar_%": round(100 * lab_sizes.max() / n, 1)})
    if row.get("K_bombing") is not None and row.get("K_max_komponen"):
        row["K_bombing/K_max"] = round(row["K_bombing"] / row["K_max_komponen"], 2)

    # ---------------- butir 4: jarak negatif -----------------------------
    mn, nneg, ntot = _cat_dist_stats(model)
    trace = PROBE.min_dist_trace
    first_neg = next((i + 1 for i, t in enumerate(trace) if t[0] < 0), None)
    row.update({"kat_jarak_min_akhir": mn, "kat_jarak_negatif_n": nneg, "kat_jarak_total_n": ntot,
                "kat_jarak_negatif_%": round(100 * nneg / ntot, 1) if ntot else 0.0,
                "iterasi_phi_pertama_negatif": first_neg, "n_pembaruan_phi": len(trace),
                "batas_bawah_teoretis(dis_const-1)": round(model.dis_const - 1.0, 3),
                "K_dipakai_phi": model.K_auto_})
    if n <= max_n_D:
        D = model._compute_distance_matrix(Xc)
        off = D[~np.eye(n, dtype=bool)]
        row.update({"D_min_offdiag": float(off.min()), "D_negatif_%": round(100 * float((off < 0).mean()), 3)})
        rng = np.random.default_rng(0)
        T = rng.integers(0, n, size=(20000, 3))
        T = T[(T[:, 0] != T[:, 1]) & (T[:, 1] != T[:, 2]) & (T[:, 0] != T[:, 2])]
        viol = D[T[:, 0], T[:, 2]] > D[T[:, 0], T[:, 1]] + D[T[:, 1], T[:, 2]] + 1e-9
        row["segitiga_dilanggar_%"] = round(100 * float(viol.mean()), 2)
        P = rng.integers(0, n, size=(300, 2))
        P = P[P[:, 0] != P[:, 1]]
        diff = [abs(model.distance(Xc.iloc[i], Xc.iloc[j]) - D[i, j]) for i, j in P]
        row["skalar_vs_matriks_beda_%"] = round(100 * float((np.asarray(diff) > 1e-9).mean()), 1)

    # ---------------- butir 2: statistik validasi merger -----------------
    val = pd.DataFrame(PROBE.validations)
    row["merge_n_validasi"] = len(val)
    if len(val):
        val.insert(0, "dataset", name)
        row.update({"merge_pass_kode_%": round(100 * val["pass_code"].mean(), 1),
                    "merge_pass_alt_%": round(100 * val["pass_alt"].mean(), 1),
                    "merge_keputusan_beda_%": round(100 * float((val["pass_code"] != val["pass_alt"]).mean()), 1),
                    "merge_median_DA": round(float(val["DA"].median()), 4),
                    "merge_median_radius": round(float(np.maximum(val["r1"], val["r2"]).median()), 3),
                    "merge_median_ratio_kode": round(float(val["ratio_code"].median()), 4),
                    "merge_median_ratio_alt": round(float(val["ratio_alt"].median()), 3)})
    return row, val


# =============================================================================
# 4. DIAGNOSTIK AMPHM  (butir 8)
# =============================================================================
def amphm_labels_variant(X, attr_types, orders, k, selection="xi", merge_factor=2.0, max_rounds=200):
    """Salinan alur AMPHM.fit() dengan satu-satunya perbedaan: skor pemilihan representatif.
       selection='xi'    -> persis produksi (xi tertinggi saja)
       selection='gamma' -> xi / rho  (jauh DAN padat; gaya decision-graph density-peak)"""
    X = X.drop(columns=["Label"], errors="ignore").reset_index(drop=True)
    n = len(X)
    helper = AMPHM_LIB.AMPHM(attr_types, k, orders, merge_factor=merge_factor)
    Z_full, num_cols, cat_cols = AMPHM_LIB._minmax_ordinal_prepare(X, attr_types, orders)
    profiles = AMPHM_LIB.build_categorical_profiles(X, cat_cols)
    g = max(2, int(np.sqrt(n)))
    ownership = {i: {i} for i in range(n)}
    active = np.arange(n)
    rounds = 0
    while len(active) > k and rounds < max_rounds:
        rounds += 1
        Xr = X.iloc[active].reset_index(drop=True)
        Zr = Z_full.iloc[active].reset_index(drop=True)
        D = AMPHM_LIB.hadm_distance_matrix(Zr, Xr, num_cols, cat_cols, profiles)
        rho, xi, nd = helper._micro_partition_round(D, min(g, len(active) - 1))
        score = xi if selection == "xi" else xi / (rho + 1e-12)
        m = max(k, int(np.ceil(len(active) / merge_factor)))
        rep = set(np.argsort(-score)[:m].tolist())
        local_to_repr = {}
        for li in range(len(active)):
            cur, seen = li, set()
            while cur not in rep:
                if cur in seen or nd[cur] == -1:
                    rep.add(cur)
                    break
                seen.add(cur)
                cur = nd[cur]
            local_to_repr[li] = cur
        new_own = {}
        for r in sorted(rep):
            merged = set()
            for li, rl in local_to_repr.items():
                if rl == r:
                    merged |= ownership[active[li]]
            new_own[active[r]] = merged
        active = np.array(sorted(new_own))
        ownership = new_own
    labels = np.full(n, -1, dtype=int)
    for cid, r in enumerate(sorted(active)):
        for i in ownership[r]:
            labels[i] = cid
    return labels


def _purity(y, c):
    return float(pd.DataFrame({"y": y, "c": c}).groupby("c")["y"].agg(lambda s: s.value_counts().iloc[0]).sum() / len(y))


def profile_labels(labels, y_true):
    sizes = np.sort(np.bincount(pd.factorize(labels)[0]))[::-1]
    n = len(labels)
    maj = float(np.bincount(y_true).max() / n)
    pur = _purity(y_true, labels)
    return {"K": int(len(sizes)), "ukuran_top3": ",".join(str(int(s)) for s in sizes[:3]),
            "klaster_terbesar_%": round(100 * sizes[0] / n, 1), "singleton": int((sizes == 1).sum()),
            "ukuran_lt1%": int((sizes < 0.01 * n).sum()), "PUR": round(pur, 4), "proporsi_kelas_mayoritas": round(maj, 4),
            "PUR-mayoritas": round(pur - maj, 4), "ARI": round(adjusted_rand_score(y_true, labels), 4),
            "NMI": round(normalized_mutual_info_score(y_true, labels), 4),
            "degenerat(>=80%)": bool(sizes[0] >= 0.8 * n)}


def reference_linkage_ari(D, y_true, k):
    """ARI klasterisasi hierarkis klasik pada matriks jarak HADM yang SAMA: batas 'jarak memadai'.
       Bila ARI ini tinggi tetapi AMPHM rendah, kelemahan ada pada mekanisme partisi/penggabungan, bukan jaraknya."""
    Dm = (D + D.T) / 2.0
    np.fill_diagonal(Dm, 0.0)
    cond = squareform(Dm, checks=False)
    out = {}
    for meth in ("average", "complete"):
        lab = fcluster(linkage(cond, meth), k, "maxclust")
        out["ARI_ref_" + meth] = round(adjusted_rand_score(y_true, lab), 4)
    return out


def diag_amphm(name, X, attr_types, orders, y_true, k):
    rows = []
    pm = AMPHM_LIB.AMPHM(attr_types, k, orders, merge_factor=2.0).fit(X)
    prod = pm.labels_
    ref = reference_linkage_ari(pm.D_full_, y_true, k)
    for sel in ("xi", "gamma"):
        t0 = time.time()
        lab = amphm_labels_variant(X, attr_types, orders, k, selection=sel)
        r = {"dataset": name, "n": len(X), "k_true": k, "varian": "produksi(xi)" if sel == "xi" else "gamma(xi/rho)"}
        r.update(profile_labels(lab, y_true))
        r.update(ref)
        if sel == "xi":
            r["salinan_sama_dgn_produksi(ARI)"] = round(adjusted_rand_score(prod, lab), 6)
        r["runtime_s"] = round(time.time() - t0, 1)
        rows.append(r)
    return rows


# =============================================================================
# 5. RINGKASAN & KESIMPULAN OTOMATIS
# =============================================================================
def _fmt(df, cols):
    cols = [c for c in cols if c in df.columns]
    with pd.option_context("display.width", 250, "display.max_columns", 50, "display.max_colwidth", 40):
        print(df[cols].to_string(index=False))


def summarize_bombing(B):
    print("\n" + "=" * 100 + "\nBUTIR No. 2 -- Eq. 6a (_validate_merge): apakah kriteria menyaring?\n" + "=" * 100)
    _fmt(B, ["dataset", "merge_n_validasi", "merge_pass_kode_%", "merge_pass_alt_%", "merge_keputusan_beda_%",
             "merge_median_DA", "merge_median_radius", "merge_median_ratio_kode", "merge_median_ratio_alt"])
    v = B[B["merge_n_validasi"] > 0]
    if len(v) == 0:
        print("-> Tidak ada percobaan merger pada dataset ini (gerbang PM tidak aktif atau DA=0): Eq. 6a tidak teruji.")
    else:
        if (v["merge_pass_kode_%"] >= 95).all():
            print("-> Kriteria kode lolos >=95% di semua dataset: Eq. 6a praktis TIDAK menyaring (DA tak berdimensi dibagi radius berdimensi jarak).")
        if (v["merge_keputusan_beda_%"] > 10).any():
            print("-> Pada sebagian dataset >10% keputusan BERBEDA bila rasio dibuat konsisten dimensi (d_centroid/radius): perbaikan Eq. 6a akan mengubah hasil.")
        else:
            print("-> Keputusan merger sama pada kedua definisi: perbaikan dimensi Eq. 6a berdampak kecil; cukup perbaiki rumus di naskah.")

    print("\n" + "=" * 100 + "\nBUTIR No. 4 -- jarak kategorikal negatif\n" + "=" * 100)
    _fmt(B, ["dataset", "n", "dis_const", "eta", "K_dipakai_phi", "kat_jarak_min_akhir", "kat_jarak_negatif_%",
             "iterasi_phi_pertama_negatif", "D_min_offdiag", "D_negatif_%", "segitiga_dilanggar_%", "skalar_vs_matriks_beda_%"])
    if (B["kat_jarak_min_akhir"] < 0).any():
        print("-> Jarak negatif TERBUKTI pada: " + ", ".join(B.loc[B["kat_jarak_min_akhir"] < 0, "dataset"]) +
              ". Perlu batas bawah delta >= delta_min > 0 sebelum re-run dataset besar.")
    else:
        print("-> Tidak ada jarak kategorikal negatif pada dataset yang diuji (belum bukti bahwa tidak terjadi pada K besar: Obesity/Stroke/Abalone).")
    if "skalar_vs_matriks_beda_%" in B and (B["skalar_vs_matriks_beda_%"] > 0).any():
        print("-> Ada ketidakcocokan jarak skalar vs matriks tervektorisasi: patch 'identik' tidak berlaku pada rezim jarak negatif.")

    print("\n" + "=" * 100 + "\nBUTIR No. 18 -- K_max (komponen jNNG) vs K Bombing vs K akhir\n" + "=" * 100)
    _fmt(B, ["dataset", "k_true", "n_komponen", "komponen_terbesar_%", "komponen_singleton", "K_bombing", "K_bombing/K_max",
             "K_akhir", "gate_trigger", "cond_K_gt_Kmax", "cond_size_ratio_gt10", "cond_maxDA_gt_T2", "cond_CV_gt_0.5",
             "klaster_akhir_ukuran_lt5"])
    if B["cond_K_gt_Kmax"].fillna(False).all():
        print("-> Syarat K > K_max benar di SEMUA dataset: gerbang PM praktis selalu aktif (bukan keputusan data-driven).")
    if (B["K_bombing/K_max"] >= 0.8).any():
        print("-> K Bombing ~ jumlah komponen terhubung pada: " + ", ".join(B.loc[B["K_bombing/K_max"] >= 0.8, "dataset"]) +
              " -> K lebih mencerminkan fragmentasi graf daripada struktur densitas.")


def summarize_amphm(A):
    print("\n" + "=" * 100 + "\nBUTIR No. 8 -- sanity AMPHM (k = k_true)\n" + "=" * 100)
    _fmt(A, ["dataset", "n", "k_true", "varian", "K", "ukuran_top3", "klaster_terbesar_%", "singleton", "PUR",
             "proporsi_kelas_mayoritas", "PUR-mayoritas", "ARI", "NMI", "ARI_ref_average", "ARI_ref_complete",
             "degenerat(>=80%)", "salinan_sama_dgn_produksi(ARI)"])
    prod = A[A["varian"].str.startswith("produksi")]
    gam = A[A["varian"].str.startswith("gamma")]
    if "salinan_sama_dgn_produksi(ARI)" in prod and (prod["salinan_sama_dgn_produksi(ARI)"] < 0.999999).any():
        print("-> PERINGATAN: salinan 'xi' tidak identik dengan AMPHM produksi; periksa sebelum menyimpulkan.")
    deg = prod[prod["degenerat(>=80%)"]]
    if len(deg):
        print("-> Produksi DEGENERAT (klaster terbesar >=80%) pada: " + ", ".join(deg["dataset"]))
        fixed = [d for d in deg["dataset"] if not gam.set_index("dataset").loc[d, "degenerat(>=80%)"]]
        if fixed:
            print("   Varian gamma tidak degenerat pada: " + ", ".join(fixed) + " -> dugaan 'representatif=pencilan' didukung.")
    else:
        print("-> Tidak ada partisi degenerat pada dataset yang diuji.")
    gap = prod[(prod[["ARI_ref_average", "ARI_ref_complete"]].max(axis=1) - prod["ARI"]) > 0.2]
    if len(gap):
        print("-> ARI AMPHM jauh di bawah linkage klasik pada jarak HADM yang sama (selisih >0,2) pada: " + ", ".join(gap["dataset"]) +
              "\n   Artinya jarak HADM memadai; kelemahan ada pada mekanisme partisi/penggabungan hasil rekonstruksi (Algorithm 1-2).")


# =============================================================================
# 6. MAIN
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema", default="new_53_dataset_schemas.json")
    ap.add_argument("--data-dir", default="53_mix_datasets")
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--include-slow", action="store_true")
    ap.add_argument("--max-n", type=int, default=1500, help="lewati dataset dengan n lebih besar (butir 2/4/18 O(n^2) lambat)")
    ap.add_argument("--out", default="diagnostik_out")
    ap.add_argument("--skip-bombing", action="store_true")
    ap.add_argument("--skip-amphm", action="store_true")
    ap.add_argument("--selftest", action="store_true", help="pakai data sintetis kecil, tanpa folder dataset")
    ap.add_argument("--selftest-n", type=int, default=120)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    _install_probes()

    jobs = []   # (name, X, attr_types, orders, y_true, k)
    if args.selftest:
        df, at, od, y, k = synthetic_mixed(n=args.selftest_n)
        jobs.append(("sintetis_n%d" % args.selftest_n, df, at, od, y, k))
    else:
        with open(args.schema) as f:
            schema_all = {e["filename"]: e for e in json.load(f)}
        names = args.datasets or (DEFAULT_DATASETS + (SLOW_DATASETS if args.include_slow else []))
        for nm in names:
            try:
                X, at, od, y, k = load_dataset(nm, schema_all, args.data_dir)
            except Exception as e:
                print(f"[LEWAT] {nm}: gagal dimuat ({e})")
                continue
            if len(X) > args.max_n:
                print(f"[LEWAT] {nm}: n={len(X)} > --max-n {args.max_n}")
                continue
            jobs.append((nm, X, at, od, y, k))

    B_rows, V_all, A_rows = [], [], []
    for nm, X, at, od, y, k in jobs:
        print(f"\n### {nm}  (n={len(X)}, k_true={k})")
        if not args.skip_bombing:
            try:
                row, val = diag_bombing(nm, X, at, od, y, verbose=args.verbose)
                B_rows.append(row)
                V_all.append(val)
                print(f"   Bombing: K_bombing={row.get('K_bombing')} K_max={row.get('K_max_komponen')} K_akhir={row['K_akhir']} ({row['runtime_s']}s)")
            except Exception as e:
                import traceback
                print(f"   [GAGAL Bombing] {e}")
                traceback.print_exc()
        if not args.skip_amphm:
            try:
                A_rows.extend(diag_amphm(nm, X, at, od, y, k))
                print("   AMPHM: selesai")
            except Exception as e:
                import traceback
                print(f"   [GAGAL AMPHM] {e}")
                traceback.print_exc()

    if args.selftest and not args.skip_amphm:
        # sanity-check yang dijanjikan di panduan Langkah 3: data sintetis berstruktur jelas -> ARI tinggi
        df, at, od, y, k = synthetic_mixed(n=200, k=3, seed=1)
        lab = amphm_labels_variant(df, at, od, k, "xi")
        ref = reference_linkage_ari(AMPHM_LIB.AMPHM(at, k, od).fit(df).D_full_, y, k)
        print(f"\n[Sanity AMPHM, n=200, k=3] ARI(produksi xi) = {adjusted_rand_score(y, lab):.3f} "
              f"| ARI(gamma) = {adjusted_rand_score(y, amphm_labels_variant(df, at, od, k, 'gamma')):.3f} "
              f"| ref linkage {ref}  (harapan implementasi benar: >0,8)")

    if B_rows:
        B = pd.DataFrame(B_rows)
        B.to_csv(os.path.join(args.out, "diag_bombing_ringkasan.csv"), index=False)
        V = pd.concat([v for v in V_all if len(v)], ignore_index=True) if any(len(v) for v in V_all) else pd.DataFrame()
        V.to_csv(os.path.join(args.out, "diag_merge_validasi.csv"), index=False)
        summarize_bombing(B)
    if A_rows:
        A = pd.DataFrame(A_rows)
        A.to_csv(os.path.join(args.out, "diag_amphm_ringkasan.csv"), index=False)
        summarize_amphm(A)
    print(f"\nSelesai. Berkas keluaran ada di folder '{args.out}'.")


if __name__ == "__main__":
    main()
