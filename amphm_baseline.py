"""
amphm_baseline.py
==================
Implementasi ulang AMPHM (Adaptive Micro Partition and Hierarchical Merging)
sebagai baseline pembanding untuk Bombing-LMD, mengikuti:

Zhang, Y., Zou, R., Zhang, Y., Zhang, Y., Cheung, Y.-M., & Li, K. (2024).
Adaptive micro partition and hierarchical merging for accurate mixed data
clustering. Complex & Intelligent Systems, 11, Article 01695.
https://doi.org/10.1007/s40747-024-01695-7

CATATAN METODOLOGIS (wajib disalin ke tabel konfigurasi baseline manuskrip --
lihat "panduan-revisi-langkah3-implementasi-amphm.md"):
- Kode sumber resmi AMPHM tidak tersedia publik pada saat implementasi ini
  dibuat -> diimplementasikan ulang dari Algorithm 1-2 dan Persamaan 5-12
  pada paper di atas.
- AMPHM MEMERLUKAN k sebagai input. Pada protokol Langkah 2, AMPHM dijalankan
  dengan k = k_ground_truth, SEKELOMPOK dengan EW-LMD, HARR-V, K-Prototype,
  MFCM -- BUKAN sekelompok dengan Bombing-LMD/AMDPC yang mengestimasi k
  secara otonom. Lihat kolom "k_mode" pada output run_full.py.
- Kriteria pemilihan jumlah representative per ronde (parameter
  `merge_factor`) adalah REKONSTRUKSI, karena paper asli hanya memberi
  deskripsi tekstual untuk bagian ini, bukan rumus eksplisit.
- Atribut Ordinal dipetakan ke skala numerik (rank/len(order)) sebelum
  dihitung, karena AMPHM hanya mendefinisikan dua tipe atribut (numerik vs
  kategorikal).
- AMPHMScalable membungkus AMPHM dengan strategi subsampling (stratified,
  n_sample/threshold) yang meniru SubsampledBombingLMDHybrid milik
  Bombing-LMD, supaya anggaran komputasi kedua metode setara pada dataset
  besar (n > threshold) -- menjawab kritik reviewer soal
  "identical ... computational budgets".
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional


# =============================================================================
# 1. HELPER: pra-pemrosesan atribut & metrik jarak HADM (Eq. 5, 7-9)
# =============================================================================
def _minmax_ordinal_prepare(X: pd.DataFrame, attr_types: Dict[str, str],
                             ordinal_orders: Dict[str, List]):
    """Numeric -> min-max [0,1]; Ordinal -> rank/len(order) pada [0,1]
    (diperlakukan numerik, lihat catatan modul); Nominal -> tetap kategorikal."""
    Z = X.copy()
    num_cols, cat_cols = [], []
    for col in X.columns:
        t = attr_types[col]
        if t == "Numeric":
            v = pd.to_numeric(X[col], errors="coerce")
            v = v.fillna(v.mean())
            vmin, vmax = v.min(), v.max()
            Z[col] = (v - vmin) / (vmax - vmin) if vmax > vmin else 0.0
            num_cols.append(col)
        elif t == "Ordinal":
            order = list(ordinal_orders[col])
            mapping = {val: i / max(1, len(order) - 1) for i, val in enumerate(order)}
            Z[col] = X[col].map(mapping).fillna(0.5)
            num_cols.append(col)
        else:  # Nominal
            cat_cols.append(col)
    return Z, num_cols, cat_cols


def build_categorical_profiles(X: pd.DataFrame, cat_cols: List[str]):
    """Profil distribusi kondisional tiap nilai kategori terhadap atribut
    kategorikal lain (dasar untuk jarak bertipe EMD/total-variation, Eq. 7-9)."""
    profiles = {}
    for col in cat_cols:
        other_cols = [c for c in cat_cols if c != col]
        vals = pd.unique(X[col])
        prof = {}
        for v in vals:
            mask = X[col] == v
            sub = X.loc[mask, other_cols] if other_cols else None
            vec = {}
            if other_cols:
                for oc in other_cols:
                    freq = sub[oc].value_counts(normalize=True)
                    for cat_val, p in freq.items():
                        vec[(oc, cat_val)] = p
            prof[v] = vec
        profiles[col] = prof
    return profiles


def _cat_distance(col: str, a, b, profiles) -> float:
    """Total-variation distance antar profil dua nilai kategori (setara EMD
    untuk ruang kategori nominal/tak berurut)."""
    if a == b:
        return 0.0
    va, vb = profiles[col].get(a, {}), profiles[col].get(b, {})
    keys = set(va) | set(vb)
    if not keys:
        return 1.0
    return 0.5 * sum(abs(va.get(kk, 0.0) - vb.get(kk, 0.0)) for kk in keys)


def hadm_distance_matrix(Z: pd.DataFrame, X_orig: pd.DataFrame, num_cols: List[str],
                          cat_cols: List[str], profiles) -> np.ndarray:
    """Matriks jarak HADM pasangan penuh (Eq. 5) untuk seluruh baris pada Z/X_orig."""
    n = len(Z)
    D2 = np.zeros((n, n))
    for col in num_cols:
        v = Z[col].to_numpy(dtype=float)
        D2 += (v[:, None] - v[None, :]) ** 2
    for col in cat_cols:
        vals = X_orig[col].to_numpy()
        uniq = pd.unique(vals)
        idx = {v: i for i, v in enumerate(uniq)}
        lut = np.zeros((len(uniq), len(uniq)))
        for a in uniq:
            for b in uniq:
                lut[idx[a], idx[b]] = _cat_distance(col, a, b, profiles)
        code = np.array([idx[v] for v in vals])
        D2 += lut[code[:, None], code[None, :]] ** 2
    return np.sqrt(D2)


def compute_amphm_distance_matrix(X: pd.DataFrame, attr_types: Dict[str, str],
                                   ordinal_orders: Optional[Dict[str, List]] = None) -> np.ndarray:
    """
    Membangun matriks jarak HADM PENUH (n x n) pada dataset X, dengan profil
    kategorikal dihitung dari SELURUH X (bukan hanya subsampel yang dipakai
    saat fit). Dipakai untuk menghitung SIL/Dunn pada evaluasi akhir,
    mengikuti pola yang sama seperti compute_lmd_distance_matrix() milik
    Bombing-LMD: evaluasi akhir selalu dihitung pada dataset penuh, terlepas
    dari subsampling yang dipakai saat fit.
    """
    X = X.drop(columns=["Label"], errors="ignore").reset_index(drop=True)
    Z, num_cols, cat_cols = _minmax_ordinal_prepare(X, attr_types, ordinal_orders or {})
    profiles = build_categorical_profiles(X, cat_cols)
    return hadm_distance_matrix(Z, X, num_cols, cat_cols, profiles)


# =============================================================================
# 2. AMPHM -- Algorithm 1 (micro-partitioning) + Algorithm 2 (hierarchical merging)
# =============================================================================
class AMPHM:
    """
    Parameters
    ----------
    attr_types : dict[str, str]        "Numeric", "Nominal", atau "Ordinal" per kolom.
    k : int                            Jumlah klaster target (ground-truth k).
    ordinal_orders : dict[str, list]   Urutan kategori untuk kolom Ordinal.
    g : int | None                     Rank tetangga utk estimasi densitas (Eq. 11).
                                        Default sqrt(n) (heuristik density-peak umum;
                                        paper AMPHM tidak menyebut nilai default eksplisit).
    merge_factor : float               Rasio pengurangan jumlah representative per ronde
                                        (REKONSTRUKSI -- lihat catatan modul). Default 2.0.
    random_state : int
    """

    def __init__(self, attr_types: Dict[str, str], k: int,
                 ordinal_orders: Optional[Dict[str, List]] = None,
                 g: Optional[int] = None, merge_factor: float = 2.0,
                 random_state: int = 42, verbose: bool = False):
        self.attr_types = attr_types.copy()
        self.ordinal_orders = (ordinal_orders or {}).copy()
        self.k = k
        self.g = g
        self.merge_factor = merge_factor
        self.random_state = random_state
        self.verbose = verbose

        self.labels_ = None
        self.n_rounds_ = 0
        self.K_final_ = None
        self._profiles = None
        self._Z_full = None
        self.D_full_ = None
        self.num_cols_, self.cat_cols_ = [], []

    def _micro_partition_round(self, D: np.ndarray, g: int):
        n = len(D)
        D_sorted = np.sort(D, axis=1)
        g_eff = min(g, n - 1)
        rho = D_sorted[:, g_eff] / g_eff                       # Eq. 11

        xi = np.zeros(n)
        nearest_denser = np.full(n, -1, dtype=int)
        order_by_rho = np.argsort(rho)                         # menaik: paling padat dulu
        for rank, i in enumerate(order_by_rho):
            denser = order_by_rho[:rank]
            if len(denser) == 0:
                xi[i] = D[i, :].max()
                nearest_denser[i] = -1
            else:
                d_to_denser = D[i, denser]
                j_local = np.argmin(d_to_denser)
                xi[i] = d_to_denser[j_local]                   # Def. 1
                nearest_denser[i] = denser[j_local]
        return rho, xi, nearest_denser

    def fit(self, X: pd.DataFrame) -> "AMPHM":
        X = X.drop(columns=["Label"], errors="ignore").reset_index(drop=True)
        n = len(X)
        Z_full, self.num_cols_, self.cat_cols_ = _minmax_ordinal_prepare(
            X, self.attr_types, self.ordinal_orders
        )
        profiles = build_categorical_profiles(X, self.cat_cols_)
        g = self.g or max(2, int(np.sqrt(n)))

        ownership = {i: {i} for i in range(n)}
        active_idx = np.arange(n)
        round_no = 0
        D_first_round = None

        while len(active_idx) > self.k:
            round_no += 1
            X_round = X.iloc[active_idx].reset_index(drop=True)
            Z_round = Z_full.iloc[active_idx].reset_index(drop=True)
            D = hadm_distance_matrix(Z_round, X_round, self.num_cols_, self.cat_cols_, profiles)
            if D_first_round is None:
                D_first_round = D  # simpan utk reuse evaluasi, hindari hitung ulang O(n^2)

            g_round = min(g, len(active_idx) - 1)
            rho, xi, nearest_denser = self._micro_partition_round(D, g_round)

            m = max(self.k, int(np.ceil(len(active_idx) / self.merge_factor)))
            order_by_xi = np.argsort(-xi)                      # menurun
            representative_local = set(order_by_xi[:m].tolist())

            local_to_repr = {}
            for local_i in range(len(active_idx)):
                cur = local_i
                visited = set()
                while cur not in representative_local:
                    if cur in visited or nearest_denser[cur] == -1:
                        representative_local.add(cur)
                        break
                    visited.add(cur)
                    cur = nearest_denser[cur]
                local_to_repr[local_i] = cur

            representative_local = sorted(representative_local)
            new_ownership = {}
            for r_local in representative_local:
                r_global = active_idx[r_local]
                members_local = [li for li, rl in local_to_repr.items() if rl == r_local]
                merged = set()
                for ml in members_local:
                    merged |= ownership[active_idx[ml]]
                new_ownership[r_global] = merged

            active_idx = np.array(sorted(new_ownership.keys()))
            ownership = new_ownership

            if self.verbose:
                print(f"   [AMPHM] Ronde {round_no}: |NR| = {len(active_idx)} (target k = {self.k})")

        labels = np.full(n, -1, dtype=int)
        for cluster_id, r_global in enumerate(sorted(active_idx)):
            for i in ownership[r_global]:
                labels[i] = cluster_id

        self.labels_ = labels
        self.n_rounds_ = round_no
        self.K_final_ = len(np.unique(labels))
        self._profiles = profiles
        self._Z_full = Z_full
        self.D_full_ = D_first_round if D_first_round is not None else hadm_distance_matrix(
            Z_full, X, self.num_cols_, self.cat_cols_, profiles
        )
        return self


# =============================================================================
# 3. AMPHMScalable -- pembungkus subsampling (paritas anggaran komputasi
#    dengan SubsampledBombingLMDHybrid milik Bombing-LMD)
# =============================================================================
class AMPHMScalable:
    def __init__(self, attr_types: Dict[str, str], k: int,
                 ordinal_orders: Optional[Dict[str, List]] = None,
                 g: Optional[int] = None, merge_factor: float = 2.0,
                 n_sample: int = 5000, threshold: int = 5000,
                 random_state: int = 42, verbose: bool = False):
        self.attr_types = attr_types
        self.k = k
        self.ordinal_orders = ordinal_orders or {}
        self.g = g
        self.merge_factor = merge_factor
        self.n_sample = n_sample
        self.threshold = threshold
        self.random_state = random_state
        self.verbose = verbose
        self._rng = np.random.default_rng(random_state)

        self.labels_ = None
        self.model_ = None
        self.used_subsampling_ = False

    def _stratified_subsample(self, X: pd.DataFrame):
        n = len(X)
        strata_col = None
        for col, t in self.attr_types.items():
            if t in ("Nominal", "Ordinal") and col in X.columns:
                strata_col = col
                break
        if strata_col is None or X[strata_col].nunique() == 1:
            idx_sub = self._rng.choice(n, size=self.n_sample, replace=False)
        else:
            strata_vals = X[strata_col].unique()
            n_per = int(np.ceil(self.n_sample / len(strata_vals)))
            idx_sub = []
            for val in strata_vals:
                idx_s = np.where(X[strata_col].values == val)[0]
                n_s = min(n_per, len(idx_s))
                if n_s > 0:
                    idx_sub.extend(self._rng.choice(idx_s, size=n_s, replace=False).tolist())
            idx_sub = np.array(sorted(set(idx_sub)))
            if len(idx_sub) < self.n_sample:
                remaining = np.setdiff1d(np.arange(n), idx_sub)
                extra_n = min(self.n_sample - len(idx_sub), len(remaining))
                if extra_n > 0:
                    extra = self._rng.choice(remaining, size=extra_n, replace=False)
                    idx_sub = np.concatenate([idx_sub, extra])
            idx_sub = idx_sub[: self.n_sample]
        idx_sub = np.sort(idx_sub)
        idx_rest = np.setdiff1d(np.arange(n), idx_sub)
        return idx_sub, idx_rest

    def fit(self, X: pd.DataFrame) -> "AMPHMScalable":
        X = X.drop(columns=["Label"], errors="ignore").reset_index(drop=True)
        n = len(X)

        if n <= self.threshold:
            self.used_subsampling_ = False
            self.model_ = AMPHM(
                self.attr_types, self.k, self.ordinal_orders, self.g,
                self.merge_factor, self.random_state, self.verbose,
            ).fit(X)
            self.labels_ = self.model_.labels_
            return self

        self.used_subsampling_ = True
        idx_sub, idx_rest = self._stratified_subsample(X)
        X_sub = X.iloc[idx_sub].reset_index(drop=True)

        self.model_ = AMPHM(
            self.attr_types, self.k, self.ordinal_orders, self.g,
            self.merge_factor, self.random_state, self.verbose,
        ).fit(X_sub)

        # Assign titik sisa ke label representative TERDEKAT (1-NN pada ruang
        # HADM yang dipelajari dari X_sub) -- meniru semangat assign_remaining()
        # pada SubsampledBombingLMDHybrid milik Bombing-LMD.
        X_rest = X.iloc[idx_rest].reset_index(drop=True)
        Z_rest, _, _ = _minmax_ordinal_prepare(X_rest, self.attr_types, self.ordinal_orders)
        Z_sub = self.model_._Z_full
        profiles = self.model_._profiles

        labels_rest = np.empty(len(idx_rest), dtype=int)
        for i_local in range(len(X_rest)):
            row = X_rest.iloc[i_local]
            row_Z = Z_rest.iloc[i_local]
            d2 = np.zeros(len(X_sub))
            for col in self.model_.num_cols_:
                d2 += (row_Z[col] - Z_sub[col].to_numpy(dtype=float)) ** 2
            for col in self.model_.cat_cols_:
                d_col = np.array([_cat_distance(col, row[col], b, profiles) for b in X_sub[col]])
                d2 += d_col ** 2
            nearest_j = int(np.argmin(d2))
            labels_rest[i_local] = self.model_.labels_[nearest_j]

        labels_full = np.empty(n, dtype=int)
        labels_full[idx_sub] = self.model_.labels_
        labels_full[idx_rest] = labels_rest
        self.labels_ = labels_full
        return self
