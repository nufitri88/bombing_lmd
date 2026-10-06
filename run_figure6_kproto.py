"""
run_figure6_kproto.py -- Figure 6 (versi 12 dataset Tabel 6-8): HANYA K-Prototypes
==================================================================================
Bombing-LMD TIDAK dijalankan ulang. Hasil Bombing-LMD revisi dibaca dari
bombing_lmd_hybrid_15_datasets_REVISI.csv (sumber yang sama dengan Tabel 6-8).
Hanya K-Prototypes (tiga lengan) yang dihitung, pada 12 dataset yang sama dengan Tabel 6-8.

Lengan:  A1 k=2 tetap | A2 k=k_true | A3 k dipilih Silhouette pada jarak Gower (rentang 2..K_MAX)
Metrik : CU, ER, ARI, NMI, Purity (hanya bergantung pada label, sebanding dengan Bombing-LMD).
         SIL/Dunn Bombing-LMD dihitung pada matriks jarak LMD yang tidak tersimpan, sehingga
         tidak dibandingkan di sini. Jarak Gower hanya dipakai untuk memilih k pada lengan A3.

Satu folder dengan: subsampling_bombinglmd_lib.py, vectorized_patches.py (hanya untuk imputasi
dan fungsi metrik), new_53_dataset_schemas.json, 53_mix_datasets/, bombing_lmd_hybrid_15_datasets_REVISI.csv
pip install kmodes scikit-learn scipy pandas numpy

Keluaran: fig6k_raw.csv (per seed), fig6k_per_dataset.csv, fig6k_summary.csv,
          fig6k_wincount.csv, fig6k_wilcoxon.csv.  Aman dihentikan dan dilanjutkan.
"""
import os, sys
_REQ = {"PYTHONHASHSEED": "0", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}
_SCRIPT = sys.argv[0].lower().endswith(".py") and "ipykernel" not in sys.modules and "spyder_kernels" not in sys.modules
if any(os.environ.get(k) != v for k, v in _REQ.items()):
    env = os.environ.copy(); env.update(_REQ)
    if not _SCRIPT:
        # Jupyter/Spyder/IPython: JANGAN memulai ulang proses. Kunci yang masih mungkin (thread BLAS) saja;
        # PYTHONHASHSEED tidak bisa diubah setelah interpreter berjalan. Disarankan: jalankan dari terminal.
        os.environ.update({k: v for k, v in _REQ.items() if k != "PYTHONHASHSEED"})
        print("PERINGATAN: dijalankan dari lingkungan interaktif; PYTHONHASHSEED tidak terkunci. "
              "Untuk hasil yang dapat direproduksi, jalankan:  python run_figure6_kproto.py")
    elif os.name == "nt":   # Windows (terminal): execvpe tidak mengganti proses; jalankan anak proses dan TUNGGU
        import subprocess
        sys.exit(subprocess.call([sys.executable] + sys.argv, env=env))
    else:
        os.execvpe(sys.executable, [sys.executable] + sys.argv, env)

import json, time, traceback
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace'); sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
import numpy as np, pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from scipy.stats import wilcoxon
from kmodes.kprototypes import KPrototypes

import subsampling_bombinglmd_lib as LIB
import vectorized_patches as VP
VP.apply_patches(LIB)

FOLDER, JSON_PATH = "53_mix_datasets", "new_53_dataset_schemas.json"
BOMBING_CSV = "bombing_lmd_hybrid_15_datasets_REVISI.csv"
DATASETS = ["new_01_automobile.csv", "new_02_german_credit.csv", "new_03_heart_disease.csv",
            "new_04_australian_credit.csv", "new_05_student_performace_mat.csv", "new_08_exasens.csv",
            "new_10_npha-doctor-visits.csv", "new_11_cirrhosis.csv", "new_12_caesar.csv",
            "new_13_obesity.csv", "new_14_heart_failure.csv", "new_15_stroke-data.csv"]
SEEDS, K_MAX, KP_INIT, KP_NINIT = [0, 1, 2, 3, 4], 10, "Huang", 10
OUT_RAW = "fig6k_raw.csv"
METRICS = ["CU", "ER", "ARI", "NMI", "PUR"]
CAND = ["class", "Class", "label", "Label", "target", "Target"]

schema = {e["filename"]: e for e in json.load(open(JSON_PATH))}
if not os.path.exists(BOMBING_CSV):
    sys.exit(f"Berkas {BOMBING_CSV} tidak ada di folder ini ({os.getcwd()}). Salin dari folder Revisi IJIES.")
B = pd.read_csv(BOMBING_CSV).set_index("Dataset")
B = B.loc[DATASETS]                                   # gagal keras bila ada dataset tanpa hasil Bombing-LMD
assert B[["ARI", "NMI", "PUR", "CU", "ER"]].notna().all().all()


def load(fn):
    meta = schema[fn]
    df = pd.read_csv(os.path.join(FOLDER, fn)).replace([" ?", "?"], np.nan)
    lab = meta.get("label_column") or next(c for c in CAND if c in df.columns)
    y = LabelEncoder().fit_transform(df[lab].astype(str)); k = len(np.unique(y))
    X = df.drop(columns=[lab] + [c for c in df.columns if c.lower() == "id"])
    attr, orders = {}, {}
    for c in meta.get("nominal_attributes", []) + meta.get("binary_attributes", []):
        if c in X: attr[c] = "Nominal"
    for c in meta.get("numeric_attributes", []):
        if c in X: attr[c] = "Numeric"
    for c in meta.get("ordinal_attributes", []):
        if c in X:
            attr[c] = "Ordinal"; orders[c] = meta["ordinal_orders"].get(c, sorted(X[c].dropna().unique()))
    num = [c for c, t in attr.items() if t == "Numeric"]; cat = [c for c, t in attr.items() if t != "Numeric"]
    if X.isnull().sum().any():                         # imputasi sama seperti run_full.py
        np.random.seed(42)
        X, _, _ = LIB.one_step_kprototypes(df=X.copy(), k=k, numeric_cols=num, cat_cols=cat)
    return X, y, k, attr, orders


def gower(X, attr, orders):
    n = len(X); D = np.zeros((n, n), dtype=np.float32)
    for c, t in attr.items():
        if t == "Numeric":
            v = X[c].astype(float).values; r = np.ptp(v) or 1.0
        elif t == "Ordinal":
            m = {v: i for i, v in enumerate(orders[c])}
            v = X[c].map(m).astype(float).values; r = max(len(orders[c]) - 1, 1)
        else:
            v = pd.factorize(X[c].astype(str))[0].astype(float); r = None
        d = (np.abs(v[:, None] - v[None, :]) / r) if r is not None else (v[:, None] != v[None, :]).astype(np.float32)
        D += d.astype(np.float32)
    return D / max(len(attr), 1)


def kproto(Xk, cat_idx, k, seed):
    try:
        return KPrototypes(n_clusters=k, init=KP_INIT, n_init=KP_NINIT, random_state=seed).fit_predict(Xk, categorical=cat_idx)
    except Exception:
        # inisialisasi Huang gagal (mis. k melebihi jumlah pola unik): cadangan Cao (deterministik)
        return KPrototypes(n_clusters=k, init="Cao", n_init=1, random_state=seed).fit_predict(Xk, categorical=cat_idx)


def metrics(X, y, lab, attr):
    return dict(CU=LIB.category_utility(X, lab), ER=LIB.entropy_reduction(X, lab, attr),
                ARI=adjusted_rand_score(y, lab), NMI=normalized_mutual_info_score(y, lab),
                PUR=LIB.purity_score(y, lab))


rows = pd.read_csv(OUT_RAW).to_dict("records") if os.path.exists(OUT_RAW) and os.path.getsize(OUT_RAW) > 0 else []
done = {(r["Dataset"], r["Arm"], r["Seed"]) for r in rows}
t_all = time.time()
for fn in DATASETS:
    need = [(fn, a, s) for a in ("A1", "A2", "A3") for s in SEEDS]
    if all(n in done for n in need): print("skip", fn); continue
    try:
        X, y, k_true, attr, orders = load(fn)
        assert k_true == int(B.loc[fn, "Label"]), (fn, k_true, B.loc[fn, "Label"])
        cat_cols = [c for c, t in attr.items() if t != "Numeric"]
        Xk = X.copy()
        for c in cat_cols: Xk[c] = Xk[c].astype(str)
        cat_idx = [list(Xk.columns).index(c) for c in cat_cols]
        print(f"\n=== {fn} n={len(X)} p={X.shape[1]} k_true={k_true}", flush=True)
        D = gower(X, attr, orders)                     # hanya untuk memilih k pada A3
        cache = {}
        for a in ("A1", "A2", "A3"):
            for s in SEEDS:
                if (fn, a, s) in done: continue
                try:
                    if a == "A1": k_a = 2; lab = cache.get((2, s)) if (2, s) in cache else kproto(Xk, cat_idx, 2, s)
                    elif a == "A2": k_a = k_true; lab = cache.get((k_true, s)) if (k_true, s) in cache else kproto(Xk, cat_idx, k_true, s)
                    else:
                        best = (-9, None, None)
                        for kk in range(2, min(K_MAX, len(X) - 1) + 1):
                            l = cache.get((kk, s)) if (kk, s) in cache else kproto(Xk, cat_idx, kk, s)
                            cache[(kk, s)] = l
                            if len(np.unique(l)) < 2: continue
                            sc = silhouette_score(D, l, metric="precomputed")
                            if sc > best[0]: best = (sc, kk, l)
                        _, k_a, lab = best
                    cache[(k_a, s)] = lab
                    rows.append(dict(Dataset=fn, Arm=a, Seed=s, K=int(k_a), k_true=k_true, status="ok",
                                     **metrics(X, y, lab, attr)))
                except Exception as e:
                    rows.append(dict(Dataset=fn, Arm=a, Seed=s, status=f"FAIL: {e}")); traceback.print_exc()
                done.add((fn, a, s))
        pd.DataFrame(rows).to_csv(OUT_RAW, index=False); print("saved", len(rows), flush=True)
    except Exception:
        traceback.print_exc(); pd.DataFrame(rows).to_csv(OUT_RAW, index=False)

raw = pd.read_csv(OUT_RAW); bad = raw[raw.status != "ok"]
if len(bad): print("PERINGATAN baris gagal:\n", bad)
ok = raw[raw.status == "ok"]
KP = ok.groupby(["Arm", "Dataset"])[METRICS + ["K"]].mean(numeric_only=True); KP.to_csv("fig6k_per_dataset.csv")
summary, wins, tests = [], [], []
for a in sorted(KP.index.get_level_values(0).unique()):
    A = KP.loc[a]; common = B.index.intersection(A.index)
    for mt in METRICS:
        bm = B.loc[common, mt]; km = A.loc[common, mt]; b4, k4 = bm.round(4), km.round(4)
        wb = wk = 0.0; ties = 0
        for d in common:
            if b4[d] > k4[d]: wb += 1
            elif b4[d] < k4[d]: wk += 1
            else: wb += .5; wk += .5; ties += 1
        diff = bm - km
        try: p = wilcoxon(diff, method="exact", zero_method="wilcox").pvalue
        except Exception: p = np.nan
        summary.append(dict(Arm=a, Metric=mt, N=len(common), Bombing_mean=bm.mean(), KProto_mean=km.mean(), MedianDiff=diff.median()))
        wins.append(dict(Arm=a, Metric=mt, N=len(common), Bombing_wins=wb, KProto_wins=wk, ties=ties))
        tests.append(dict(Arm=a, Metric=mt, p_raw=p))
tdf = pd.DataFrame(tests); tdf["p_holm"] = np.nan
for mt, g in tdf.groupby("Metric"):
    ps = g.p_raw.values; o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for r, i in enumerate(o): run = max(run, (m - r) * ps[i]); adj[i] = min(1, run)
    tdf.loc[g.index, "p_holm"] = adj
pd.DataFrame(summary).to_csv("fig6k_summary.csv", index=False)
pd.DataFrame(wins).to_csv("fig6k_wincount.csv", index=False); tdf.to_csv("fig6k_wilcoxon.csv", index=False)
print("\nSELESAI", f"{time.time() - t_all:.0f}s")
print(pd.DataFrame(summary).round(3).to_string(index=False)); print(pd.DataFrame(wins).to_string(index=False)); print(tdf.round(4).to_string(index=False))
