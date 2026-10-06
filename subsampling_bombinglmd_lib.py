import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
from itertools import combinations
from typing import Dict, List, Tuple, Optional, Literal, Any, Callable
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.stats import friedmanchisquare, wilcoxon
from scipy.spatial.distance import pdist, squareform
import scikit_posthocs as sp
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.sparse.csgraph import connected_components
from sklearn.datasets import make_blobs
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import dijkstra
from collections import deque
import traceback 
import os
import json
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, fowlkes_mallows_score
from sklearn.manifold import MDS
from scipy.stats import gaussian_kde
from matplotlib.colors import Normalize
from matplotlib import cm

import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. Helper Functions & Hybrid Selection Logic
# =============================================================================
def calculate_shannon_entropy(series: pd.Series) -> float:
    p = series.value_counts(normalize=True)
    return -np.sum(p * np.log2(p + 1e-9))

def rule_based_selection(structure: dict) -> dict:
    dist_cons, eta, n_bins = 0.1, 5, 5
    
    if structure['n_nominal'] >= 8 and structure['max_nominal_card'] > 10: dist_cons = 0.2
    elif structure['max_nominal_card'] > 25: dist_cons = 0.2
        
    if structure['is_numeric_heavy'] and structure['n_numeric'] >= 7: eta = 25
    elif structure['is_wide'] and structure['max_nominal_card'] > 15: eta = 25
    elif structure['is_wide'] and structure['n_ordinal'] >= 5: eta = 15
    elif structure['n_ordinal'] >= 3 and structure['n_nominal'] >= 5: eta = 20
        
    if structure['n_numeric'] >= 8: n_bins = 15
    elif structure['n_numeric'] >= 5: n_bins = 10
    elif structure['n_rows'] < 500 and structure['n_numeric'] > 3: n_bins = 10
        
    return {"dist_cons": dist_cons, "eta": float(eta), "n_bins_numeric": int(n_bins)}

def detect_domain(structure: dict) -> str:
    if structure['n_nominal'] >= 8 and structure['max_nominal_card'] > 10: return "medical_categorical"
    elif structure['n_numeric'] >= 10 and structure['n_numeric'] / structure['n_cols'] > 0.5: return "numeric_heavy"
    elif structure['n_ordinal'] >= 5: return "ordinal_rich"
    elif structure['n_rows'] > 5000: return "large_scale"
    return "general"

def hybrid_selection_conservative(df: pd.DataFrame, attr_types: Dict[str, str], structure: dict) -> dict:
    base = rule_based_selection(structure)
    domain = detect_domain(structure)
    
    ents = [calculate_shannon_entropy(df[col]) for col in df.columns]
    mean_ent, std_ent = np.mean(ents), np.std(ents)
    
    thresholds = {
        "medical_categorical": {"ent_high": 1.8, "std_high": 0.9},
        "numeric_heavy": {"ent_high": 2.0, "std_high": 1.0},
        "ordinal_rich": {"ent_high": 2.2, "std_high": 1.1},
        "large_scale": {"ent_high": 1.5, "std_high": 0.8},
        "general": {"ent_high": 2.5, "std_high": 1.2}
    }
    th = thresholds.get(domain, thresholds["general"])
    
    # INTERVENSI HANYA JIKA baseline di zona ekstrem DAN entropi konsisten
    if base['eta'] <= 5 and mean_ent > th['ent_high']:
        base['eta'] = min(25, 5 + (mean_ent - th['ent_high']) * 8)
        
    num_ents = [calculate_shannon_entropy(df[c]) for c, t in attr_types.items() if t == "Numeric"]
    if num_ents and np.std(num_ents) > th['std_high'] and base['n_bins_numeric'] <= 5:
        base['n_bins_numeric'] = min(15, 5 + int(np.std(num_ents) * 4))
        
    max_nom_card = max([df[c].nunique() for c, t in attr_types.items() if t == "Nominal"] or [0])
    if max_nom_card > 30 and base['dist_cons'] == 0.1:
        base['dist_cons'] = 0.2
        
    return base
	

# ==============================================================================
# POST-HOC MERGER (OPTIMIZED FOR OVER-SEGMENTATION)
# ==============================================================================
class PostHocMerger:
    def __init__(self, model, X: pd.DataFrame, D_ewlmd: np.ndarray, jnng_adj):
        self.model = model
        self.X = X
        self.D_ewlmd = D_ewlmd
        self.jnng_adj = jnng_adj  # ✅ SIMPAN jnng_adj SEBAGAI ATRIBUT
        self.labels_original = model.labels_.copy()
        self.centers_original = model.centers_.copy()
        self.K_original = len(np.unique(self.labels_original))
        
    def _compute_centroid_distance_ewlmd(self, c1: int, c2: int) -> float:
        center1 = self.centers_original.iloc[c1]
        center2 = self.centers_original.iloc[c2]
        total_dist = 0.0
        for col in self.X.columns:
            # w = self.model.attr_weights.get(col, 1.0)
            w = 1
            d_attr = self.model._dist_attr(col, center1[col], center2[col])
            total_dist += w * d_attr
        return total_dist

    def _compute_cluster_radius(self, label: int, labels: np.ndarray) -> float:
        mask = (labels == label)
        idx_cluster = np.where(mask)[0]
        if len(idx_cluster) < 2: return 0.0
        
        center = self.centers_original.loc[label]
        max_dist = 0.0
        indices = idx_cluster if len(idx_cluster) <= 50 else np.random.choice(idx_cluster, 50, replace=False)
        
        for i in indices:
            row = self.X.iloc[i]
            w = 1
            # d = sum(self.model.attr_weights.get(col, 1.0) * self.model._dist_attr(col, row[col], center[col]) for col in self.X.columns)
            d = sum(w * self.model._dist_attr(col, row[col], center[col]) for col in self.X.columns)
            if d > max_dist: max_dist = d
        return max_dist

    def _validate_merge(self, c1: int, c2: int, dist: float, labels: np.ndarray) -> bool:
        r1 = self._compute_cluster_radius(c1, labels)
        r2 = self._compute_cluster_radius(c2, labels)
        
        # PERBAIKAN: Relaksasi separation_ratio dari 1.2 menjadi 1.5
        separation_ratio = dist / max(r1, r2, 1e-12)
        if separation_ratio > 1.5: 
            return False 
        return True

    def _compute_da_between(self, c1: int, c2: int, labels: np.ndarray) -> float:
        """
        Menghitung Degree of Adjacency (DA) antara dua cluster menggunakan self.jnng_adj.
        """
        members_c1 = np.where(labels == c1)[0]
        members_c2 = np.where(labels == c2)[0]
        
        # Konversi sparse matrix ke list of lists untuk iterasi cepat
        adj_list = self.jnng_adj.tolil().rows 
        
        nap = 0  # Number of Adjacent Pairs
        for u in members_c1:
            for v in adj_list[u]:
                if v in members_c2:
                    nap += 1
                    
        min_size = min(len(members_c1), len(members_c2))
        da = nap / min_size if min_size > 0 else 0.0
        return da

    def smart_merge_stable(self, min_K: int = 2, max_merge_ratio: float = 1.0) -> Dict:
        from itertools import combinations
        
        labels = self.labels_original.copy()
        merge_history = []
        K_start = len(np.unique(labels))
        merged_count = 0
        
        max_allowed_merges = int((K_start - min_K) * max_merge_ratio)
        if max_allowed_merges < 1 and K_start > min_K: 
            max_allowed_merges = K_start - min_K 
            
        iteration = 0
        while iteration < 50:
            unique_labels = np.unique(labels)
            K_current = len(unique_labels)
            
            if K_current <= min_K:
                if self.model.verbose: print(f" -> [Guard] Reached min_K ({min_K}). Stopping.")
                break
            if merged_count >= max_allowed_merges:
                if self.model.verbose: print(f" -> [Guard] Reached max_merge limit. Stopping.")
                break

            # ✅ PERBAIKAN: Cari pasangan dengan DA tertinggi (Topology-driven)
            best_da, best_pair = -1, None
            for (c1, c2) in combinations(unique_labels, 2):
                da = self._compute_da_between(c1, c2, labels) # Gunakan self.jnng_adj di dalamnya
                if da > best_da:
                    best_da, best_pair = da, (c1, c2)
                    
            if best_pair and best_da > 0:
                c1, c2 = best_pair
                if self._validate_merge(c1, c2, best_da, labels):
                    labels[labels == c2] = c1
                    merge_history.append((c1, c2))
                    merged_count += 1
                else:
                    # Jika validasi gagal, break untuk mencegah infinite loop
                    break 
            else:
                # Tidak ada pasangan yang terhubung (DA = 0)
                break
                
            iteration += 1
            
        self.labels_merged = labels
        self.K_final = len(np.unique(labels))
        self.merge_history = merge_history
        
        # Update centers
        self.model.centers_ = self.model._update_centers_refinement(self.X, self.labels_merged)
        self.model.labels_ = self.labels_merged
        self.model.K_auto_ = self.K_final
        
        return {
            'labels_merged': self.labels_merged,
            'K_before': K_start,
            'K_final': self.K_final,
            'merge_history': merge_history,
            'metrics': {'merges_applied': merged_count}
        }

    def _compute_all_centroid_distances(self, labels: np.ndarray) -> Dict[Tuple[int, int], float]:
        unique_labels = np.unique(labels)
        dists = {}
        for i, c1 in enumerate(unique_labels):
            for c2 in unique_labels[i+1:]:
                d = self._compute_centroid_distance_ewlmd(c1, c2)
                dists[(c1, c2)] = d
        return dists

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def _mode(series: pd.Series) -> Any:
    """Deterministic mode with lexicographic tie-break."""
    vc = series.value_counts(dropna=False)
    if vc.empty: return np.nan
    m = vc.max()
    cands = sorted([v for v, c in vc.items() if c == m], key=lambda x: str(x))
    return cands[0]

def compute_merge_metrics(labels: np.ndarray, jnng_adj: csr_matrix, mixture_density_P: np.ndarray) -> Dict[str, float]:
    """Menghitung statistik struktural untuk trigger Post-Hoc Merger."""
    unique_labels = np.unique(labels)
    K = len(unique_labels)
    N = len(labels)
    cluster_sizes = np.array([np.sum(labels == c) for c in unique_labels])
    
    max_da = 0.0
    adj_list = jnng_adj.tolil().rows
    for i_idx, c1 in enumerate(unique_labels):
        mask_c1 = (labels == c1)
        idx_c1 = np.where(mask_c1)[0]
        for c2 in unique_labels[i_idx+1:]:
            mask_c2 = (labels == c2)
            idx_c2 = np.where(mask_c2)[0]
            if len(idx_c1) == 0 or len(idx_c2) == 0: continue
            
            nap = sum(1 for u in idx_c1 for v in adj_list[u] if mask_c2[v])
            da = nap / min(len(idx_c1), len(idx_c2)) if min(len(idx_c1), len(idx_c2)) > 0 else 0
            if da > max_da: max_da = da
            
    intra_cvs = []
    for c in unique_labels:
        mask = (labels == c)
        if np.sum(mask) < 2: continue
        density_in_cluster = mixture_density_P[mask]
        mean_d = np.mean(density_in_cluster)
        std_d = np.std(density_in_cluster)
        intra_cvs.append(std_d / (mean_d + 1e-12))
    avg_cv = np.mean(intra_cvs) if intra_cvs else 0.0

    return {'K': K, 'N': N, 'cluster_sizes': cluster_sizes, 'max_da': max_da, 'intra_density_cv': avg_cv}

# K_max harus dihitung dari CC(A) saat jNNG dibangun
# dan dioper ke fungsi ini sebagai parameter
def should_merge_parameter_free(metrics, K_max, T2=0.2):
    K, N = metrics['K'], metrics['N']
    sizes, max_da, cv = metrics['cluster_sizes'], metrics['max_da'], metrics['intra_density_cv']

    if K > K_max: return True  # ← data-driven
    if np.min(sizes) > 0 and np.max(sizes)/np.min(sizes) > 10: return True
    if max_da > T2: return True
    if cv > 0.5: return True
    return False

# ==============================================================================
# MAIN HYBRID CLASS (PURE BOMBING ARCHITECTURE)
# ==============================================================================
# ARSITEKTUR FINAL (per Studi Ablasi I & II, Bagian 4.3-4.4 draft artikel):
#   - Density Bombing (Bagian 3.3)               : AKTIF (adopsi [10])
#   - Post Hoc Merger / PM (Bagian 3.4)           : AKTIF (signifikan, dipertahankan)
#   - Density-Excluded Initialization / DE        : TIDAK AKTIF (tidak signifikan,
#                                                     tidak disertakan pada desain
#                                                     final -> lihat _compute_initial_centroids)
#   - Iterative Refinement + Silhouette Recovery  : AKTIF (Bagian 3.6)
# Kelas ini setara dengan `LMDBombingHybrid_V3` ("wo-DE") pada
# ablation_study_pipeline2.ipynb, yaitu model yang dipilih sebagai model akhir.
# ==============================================================================
class LMDBombingHybrid:
    def __init__(self, attr_types: dict, ordinal_orders: dict = None,
                 max_iter_refine: int = 20, random_state: int = 42, verbose: bool = True):
        
        self.attr_types = attr_types.copy()
        self.ordinal_orders = (ordinal_orders or {}).copy()
        self.dis_const = 0.3
        self.eta_cat = 20.0
        self.eta_num = 20.0 
        self.alpha_reg = 0.8
        self.tau_weight = 1e-3
        self.n_bins_numeric = 10
        self.max_pairs_per_cluster = 200
        self.max_iter_refine = max_iter_refine
        self.conv_threshold = 0.98
        self._rng = np.random.default_rng(random_state)
        self.verbose = verbose
        
        # Bombing Parameters
        self.k1, self.k2 = 2, 10
        self.dc, self.tau, self.T1, self.T2 = 0.15, 0.6, 0.9, 0.2
        
        # self.attr_weights = {}
        self.nominal_dist, self.ordinal_adj_dist = {}, {}
        self.num_min, self.num_max = {}, {}
        self.num_bin_psi = {}
        self.nominal_phi, self.ordinal_phi, self.numeric_phi = {}, {}, {}
        
        self.labels_ = None
        self.centers_ = None
        self.K_auto_ = None
        self.K_max_ = None
        self._reachable_indices = {} 

    def _compute_structure(self, X: pd.DataFrame) -> dict:
        counts = {"Numeric": 0, "Ordinal": 0, "Nominal": 0}
        for t in self.attr_types.values():
            counts["Nominal" if t not in ["Numeric", "Ordinal"] else t] += 1
        max_nom = max([X[c].nunique() for c, t in self.attr_types.items() if t == "Nominal"] or [0])
        return {
            'n_rows': len(X), 'n_cols': len(X.columns),
            'n_numeric': counts['Numeric'], 'n_ordinal': counts['Ordinal'], 'n_nominal': counts['Nominal'],
            'max_nominal_card': max_nom,
            'is_wide': len(X.columns) > 20,
            'is_numeric_heavy': counts['Numeric'] > counts['Nominal'],
            'is_nominal_heavy': counts['Nominal'] > counts['Numeric']
        }
    # --- Distance Helpers ---
    def _init_distances(self, X: pd.DataFrame):
        for col in X.columns:
            t = self.attr_types[col]
            if t == "Nominal":
                vals = list(pd.unique(X[col]))
                self.nominal_dist[col] = {(a,b): self.dis_const for a in vals for b in vals if a!=b}
                self.nominal_phi[col] = {(a,b): 0.0 for a in vals for b in vals if a!=b}
            elif t == "Ordinal":
                order = list(self.ordinal_orders[col])
                self.ordinal_adj_dist[col] = {order[i]: self.dis_const for i in range(len(order)-1)}
                self.ordinal_phi[col] = {order[i]: 0.0 for i in range(len(order)-1)}
            elif t == "Numeric":
                v = pd.to_numeric(X[col], errors='coerce').fillna(X[col].mean())
                self.num_min[col], self.num_max[col] = float(v.min()), float(v.max())
                self.num_bin_psi[col] = np.zeros(self.n_bins_numeric, dtype=float)
                self.numeric_phi[col] = np.zeros(self.n_bins_numeric, dtype=float)

    def _dist_attr(self, col: str, a: Any, b: Any) -> float:
        t = self.attr_types[col]
        if t == "Nominal":
            if a == b: return 0.0
            mp = self.nominal_dist[col]
            return mp.get((a,b), mp.get((b,a), self.dis_const))
        elif t == "Ordinal":
            if a == b: return 0.0
            order = list(self.ordinal_orders[col])
            try: ia, ib = order.index(a), order.index(b)
            except ValueError: return 0.0
            if ia > ib: ia, ib = ib, ia
            return sum(self.ordinal_adj_dist[col].get(order[i], self.dis_const) for i in range(ia, ib))
        elif t == "Numeric":
            vmin, vmax = self.num_min[col], self.num_max[col]
            if vmax <= vmin: return 0.0
            try: va, vb = float(a), float(b)
            except Exception: va, vb = (vmin+vmax)/2.0, (vmin+vmax)/2.0
            
            z_a = np.clip((va - vmin) / (vmax - vmin), 0.0, 1.0)
            z_b = np.clip((vb - vmin) / (vmax - vmin), 0.0, 1.0)
            
            # PERBAIKAN MURNI BOMBING: Absolute Difference (BUKAN Squared)
            # Ini menjaga konektivitas manifold jNNG agar tidak terputus (sparse)
            delta = abs(z_a - z_b)
            
            psi_vec = self.num_bin_psi[col]
            if psi_vec is None or len(psi_vec) == 0: return delta
            bidx = int(np.clip(delta * self.n_bins_numeric, 0, self.n_bins_numeric - 1))
            scale = max(0.1, 1.0 + psi_vec[bidx])
            return delta * scale
        return 0.0

    def distance(self, x: pd.Series, y: pd.Series) -> float:
        d = 0.0
        for col in y.index:
            d += self._dist_attr(col, x[col], y[col])
        return float(d)

    def _compute_distance_matrix(self, X: pd.DataFrame) -> np.ndarray:
        n = len(X)
        D = np.zeros((n, n))
        cols = list(X.columns)
        for i in range(n):
            for j in range(i+1, n):
                d = sum(self._dist_attr(col, X.iloc[i][col], X.iloc[j][col]) for col in cols)
                D[i, j] = D[j, i] = d
        return D

    # --- Bombing Core Mechanics ---
    def _build_jnng(self, D_scaled: np.ndarray):
        n = len(D_scaled)
        
        k1_idx = np.argpartition(D_scaled, self.k1 + 1, axis=1)[:, 1:self.k1 + 1] # +1 untuk skip diri sendiri
        k1_adj = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in k1_idx[i]:
                k1_adj[i, j] = k1_adj[j, i] = True
                
        # --- k2 (MUTUAL k-NN FIX) ---
        # Ambil k2+1, lalu slice [:, 1:] untuk membuang jarak ke diri sendiri (yang bernilai 0)
        k2_idx = np.argpartition(D_scaled, self.k2 + 1, axis=1)[:, 1:self.k2 + 1] 
        k2_adj = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in k2_idx[i]:
                # Cek mutualitas yang sebenarnya (j != i sudah dijamin oleh slice [1:])
                if i in k2_idx[j]: 
                    k2_adj[i, j] = k2_adj[j, i] = True
                
        jnng_adj = k1_adj | k2_adj
        jnng_sparse = csr_matrix(jnng_adj.astype(float) * D_scaled)
        
        # Geodesic distance
        geo_dist = shortest_path(jnng_sparse, method='D', directed=False)
        geo_dist[np.isinf(geo_dist)] = geo_dist[np.isfinite(geo_dist)].max() * 1.5

        n_cc, _ = connected_components(csr_matrix(jnng_adj), directed=False)
        self.K_max_ = int(n_cc)   # ← K_max data-driven, disimpan sebagai atribut
        
        return jnng_sparse, geo_dist, csr_matrix(jnng_adj.astype(float))

    def _gaussian_kernel_density_old(self, geo_dist: np.ndarray) -> np.ndarray:
        """
        PERBAIKAN MURNI BOMBING: Local Scaling (Zelnik-Perona style).
        Mencegah density flattening dengan membuat bandwidth adaptif terhadap kepadatan lokal.
        """
        n = geo_dist.shape[0]
        k_local = min(7, n - 1)
        
        sigmas = np.partition(geo_dist, k_local, axis=1)[:, k_local]
        sigmas[sigmas == 0] = 1e-12
        
        sigma_matrix = np.outer(sigmas, sigmas)
        rho = np.sum(np.exp(-geo_dist**2 / (2 * sigma_matrix + 1e-12)), axis=1)
        return rho

    def _gaussian_kernel_density(self, geo_dist: np.ndarray) -> np.ndarray:
        """
        Mengikuti Eq. 9 dari Paper Fei et al. (2025).
        Menggunakan varians tetap (dc) untuk menjaga validitas Two-Round Sigmoid.
        """
        # self.dc = 0.15 (atau sesuai parameter default paper)
        rho = np.sum(np.exp(-(geo_dist ** 2) / (2 * (self.dc ** 2))), axis=1)
        return rho

    def _two_round_sigmoid_correction(self, rho_gjnn: np.ndarray) -> np.ndarray:
        rho_max = rho_gjnn.max()
        slope, delta = 10/rho_max, rho_max/2
        rho_tilde = 1 / (1 + np.exp(-slope * (rho_gjnn - delta)))
        mask = rho_tilde < self.T1
        if np.any(mask):
            rho_max_below = rho_gjnn[mask].max()
            s2, d2 = 10/rho_max_below, rho_max_below/2
            rho_tilde[mask] = 1 / (1 + np.exp(-s2 * (rho_tilde[mask] - d2)))
        return rho_tilde

    def _local_neighbor_density(self, D_scaled: np.ndarray, r: float, valid_mask: np.ndarray) -> np.ndarray:
        """Menghitung rho_r (Eq. 16) hanya berdasarkan titik-titik yang valid (belum di-cluster)."""
        n = len(D_scaled)
        rho_r = np.zeros(n)
        for i in range(n):
            if not valid_mask[i]: continue
            # Hitung tetangga dalam radius r dari titik i
            neighbors_in_r = np.sum((D_scaled[i] <= r) & valid_mask)
            rho_r[i] = neighbors_in_r / self.k2
        return rho_r

    def _bfs_generalized_cluster(self, adj_mat: csr_matrix, core_idx: int, P: np.ndarray, visited_mask: np.ndarray) -> Tuple[List[int], List[int]]:
        R_t, B_t = [], []
        queue = [core_idx]
        visited_mask[core_idx] = True
        R_t.append(core_idx)
        adj_list = adj_mat.tolil().rows
        first_boundary = set()
        
        while queue:
            u = queue.pop(0)
            for v in adj_list[u]:
                if visited_mask[v]: continue
                if P[v] >= self.tau:
                    visited_mask[v] = True
                    R_t.append(v)
                    queue.append(v)
                else:
                    first_boundary.add(v)
                    
        second_boundary = set()
        for v in first_boundary:
            for w in adj_list[v]:
                if not visited_mask[w] and w not in first_boundary and w not in R_t:
                    second_boundary.add(w)
        B_t = list(first_boundary | second_boundary)
        return R_t, B_t

    def _degree_of_adjacency(self, adj_mat: csr_matrix, R_t: List[int], B_t: List[int], existing_clusters: dict) -> Tuple[float, int]:
        if not existing_clusters: return 0.0, -1
        C_star = len(R_t) + len(B_t)
        if C_star == 0: return 0.0, -1
        adj_list = adj_mat.tolil().rows
        max_nap, best_s = 0, -1
        for s, (_, B_s) in existing_clusters.items():
            nap = sum(1 for u in R_t for v in adj_list[u] if v in B_s)
            da = nap / C_star
            if da > max_nap: max_nap, best_s = da, s
        return max_nap, best_s

    def _run_bombing_phase(self, X: pd.DataFrame, D_scaled: np.ndarray, geo_dist: np.ndarray, rho_tilde: np.ndarray, jnng_adj: csr_matrix):
        n = len(X)
        labels = np.full(n, -1, dtype=int)
        visited = np.zeros(n, dtype=bool)
        existing_clusters = {}
        t = 0
        self._reachable_indices = {}
        
        # Copy rho_tilde agar bisa dimanipulasi per iterasi
        current_rho_tilde = rho_tilde.copy()
        
        while not np.all(visited):
            cand = ~visited
            if not np.any(cand): break
            
            # 1. Tentukan Core Point dari titik yang BELUM dikunjungi
            core_idx = np.argmax(current_rho_tilde * cand)
            if current_rho_tilde[core_idx] < 1e-6: break

            # 2. Hitung Ulang Local Neighbor Density (rho_r) berdasarkan sisa titik
            # Paper: r adalah jarak ke k-th nearest neighbor dari core point
            unvisited_indices = np.where(cand)[0]
            if len(unvisited_indices) <= self.k2:
                r = np.max(D_scaled[core_idx, unvisited_indices])
            else:
                r = np.partition(D_scaled[core_idx, unvisited_indices], self.k2)[self.k2]
                
            rho_r = self._local_neighbor_density(D_scaled, r, cand)

            # 3. Mixture Density (Eq. 17)
            P = current_rho_tilde + rho_r
            P_min, P_max = P.min(), P.max()
            if P_max > P_min: 
                P = (P - P_min) / (P_max - P_min)

            # 4. BFS dengan TAU TETAP (0.6) - Ini adalah klaim utama paper!
            R_t, B_t = self._bfs_generalized_cluster(jnng_adj, core_idx, P, visited.copy())
            
            # Tandai titik yang di-cluster dan boundary sebagai visited (dihapus dari X)
            idxs = R_t + B_t
            for idx in idxs: 
                visited[idx] = True

            # 5. Hitung Degree of Adjacency (DA)
            da, best_s = self._degree_of_adjacency(jnng_adj, R_t, B_t, existing_clusters)

            # 6. Aturan Penggabungan (Generalized Cluster)
            if da >= self.T2 and best_s != -1:
                labels[idxs] = best_s
            else:
                labels[idxs] = t
                existing_clusters[t] = (R_t, B_t)
                self._reachable_indices[t] = R_t
                t += 1

            # 7. UPDATE DINAMIS: Nol-kan densitas titik yang sudah di-cluster
            # Agar iterasi berikutnya hanya mencari core point dari sisa dataset
            current_rho_tilde[visited] = 0.0
            
            # Opsional: Recalculate Sigmoid Round 1 & 2 untuk sisa titik jika diperlukan
            # (Paper melakukan recalculate rho_max dari sisa titik)
            rho_max_rem = current_rho_tilde.max()
            if rho_max_rem > 1e-6:
                slope, delta = 10 / rho_max_rem, rho_max_rem / 2
                # Terapkan ulang sigmoid ke titik yang belum visited
                current_rho_tilde[~visited] = 1 / (1 + np.exp(-slope * (current_rho_tilde[~visited] - delta)))

        # Final Clean-up (Assign unvisited/boundary ke nearest visited point)
        unvisited = np.where(labels == -1)[0]
        if len(unvisited) > 0:
            visited_pts = np.where(labels != -1)[0]
            if len(visited_pts) > 0:
                for u in unvisited:
                    labels[u] = labels[visited_pts[np.argmin(D_scaled[u, visited_pts])]]
            else:
                labels[:n//2] = 0
                labels[n//2:] = 1

        self.K_auto_ = len(np.unique(labels[labels != -1]))
        return self.K_auto_, labels

    def _compute_initial_centroids(self, X: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
        """
        [FINAL ARCHITECTURE — hasil Studi Ablasi I & II, Bagian 4.3-4.4]
        Density-Excluded Initialization (DE) TIDAK disertakan pada desain final
        Bombing-LMD. Centroid awal dihitung dari SELURUH anggota klaster hasil
        Bombing + Post Hoc Merger (Persamaan 8), tanpa membedakan wilayah
        reachable (R_k) dan boundary (B_k).

        Implementasi ini meniru persis `LMDBombingHybrid_V3` (varian wo-DE) pada
        ablation_study_pipeline2.ipynb, yang menjadi model akhir terpilih:
        `self._reachable_indices` sengaja TIDAK dipakai di sini. Atribut tersebut
        tetap dipertahankan di tempat lain (lihat _run_bombing_phase) semata-mata
        untuk keperluan visualisasi (visualize_bombing_clusters), bukan untuk
        komputasi centroid.
        """
        centers = []
        unique_labels = np.unique(labels)

        for lbl in unique_labels:
            cl = X[labels == lbl]

            c = {}
            for col in X.columns:
                if self.attr_types[col] in ('Nominal', 'Ordinal'):
                    c[col] = _mode(cl[col])
                else:
                    c[col] = cl[col].mean()
            centers.append(c)
        return pd.DataFrame(centers, columns=X.columns)

    # --- EW-LMD Refinement Helpers ---
    def _update_nominal_ordinal_phi(self, X: pd.DataFrame, labels: np.ndarray):
        eps = 1e-12
        k = self.K_auto_
        for col in X.columns:
            if self.attr_types[col] not in ('Nominal', 'Ordinal'): continue
            vals = list(pd.unique(X[col]))
            if len(vals) < 2: continue
            P = {}
            for a in vals:
                for b in vals:
                    if a == b: continue
                    val_sum = sum(((X.iloc[np.where(labels==c)[0]][col]==a).sum()/(len(X[labels==c])+eps)) * 
                                  ((X.iloc[np.where(labels==c)[0]][col]==b).sum()/(len(X[labels==c])+eps)) 
                                  for c in range(k) if len(X[labels==c]) > 0)
                    P[(a,b)] = val_sum
            if not P: continue
            P_vals = list(P.values())
            P_max, P_min = max(P_vals), min(P_vals)
            if abs(P_max-P_min) < eps: continue
            
            phi_raw = {key: (-P_max/self.eta_cat if v==P_max else (1-P_min)/self.eta_cat if v==P_min else 0.0) for key, v in P.items()}
            phi_old = self.nominal_phi[col] if self.attr_types[col]=='Nominal' else self.ordinal_phi[col]
            phi_new = {key: np.clip(self.alpha_reg*phi_old.get(key,0.0) + (1-self.alpha_reg)*phi_raw[key], -1, 1) for key in phi_raw}
            
            if self.attr_types[col]=='Nominal':
                self.nominal_phi[col] = phi_new
                self.nominal_dist[col] = {k: self.dis_const + v for k, v in phi_new.items()}
            else:
                self.ordinal_phi[col] = phi_new
                self.ordinal_adj_dist[col] = {k: self.dis_const + v for k, v in phi_new.items()}

    def _update_numeric_phi(self, X: pd.DataFrame, labels: np.ndarray):
        eps = 1e-12
        B = self.n_bins_numeric
        k = self.K_auto_
        for col in X.columns:
            if self.attr_types[col] != 'Numeric': continue
            if self.num_min[col] >= self.num_max[col]: continue
            z_full = np.clip((pd.to_numeric(X[col], errors='coerce').fillna(X[col].mean()).to_numpy() - self.num_min[col]) / (self.num_max[col] - self.num_min[col]), 0.0, 1.0)
            P_sum = np.zeros(B)
            
            for c in range(k):
                idx = np.where(labels==c)[0]
                m = len(idx)
                if m < 2: continue
                if m > self.max_pairs_per_cluster:
                    idx_sample = self._rng.choice(idx, size=self.max_pairs_per_cluster, replace=False)
                else:
                    idx_sample = idx
                m_s = len(idx_sample)
                if m_s < 2: continue
                
                bin_counts = np.zeros(B, dtype=float)
                for i in range(m_s):
                    for j in range(i + 1, m_s):
                        delta = abs(z_full[idx_sample[i]] - z_full[idx_sample[j]])
                        bidx = int(np.clip(delta * B, 0, B - 1))
                        bin_counts[bidx] += 1.0
                total_pairs = bin_counts.sum()
                if total_pairs <= 0: continue
                probs = bin_counts / (total_pairs + eps)
                P_sum += probs * probs

            if np.all(P_sum <= eps): continue
            P_max, P_min = P_sum.max(), P_sum.min()
            if abs(P_max - P_min) < eps: continue

            phi_raw_vec = np.zeros(B, dtype=float)
            for bidx in range(B):
                val = P_sum[bidx]
                if val == P_max: phi = -P_max / self.eta_num
                elif val == P_min: phi = (1.0 - P_min) / self.eta_num
                else: phi = 0.0
                phi_raw_vec[bidx] = phi

            phi_old_vec = self.numeric_phi[col]
            if phi_old_vec.shape[0] != B: phi_old_vec = np.zeros(B, dtype=float)
            phi_new_vec = np.clip(self.alpha_reg * phi_old_vec + (1.0 - self.alpha_reg) * phi_raw_vec, -1.0, 1.0)
            self.numeric_phi[col] = phi_new_vec
            self.num_bin_psi[col] = phi_new_vec.copy()

    def _assign_refinement(self, X: pd.DataFrame, jnng_adj: csr_matrix) -> np.ndarray:
        """
        REFINEMENT DENGAN KENDALA KONEKTIVITAS jNNG (FIX CELAH #1)
        Titik hanya boleh pindah klaster jika memiliki edge jNNG dengan anggota klaster target.
        """
        n = len(X)
        labels = np.empty(n, dtype=int)
        
        # 1. Pre-compute anggota klaster saat ini untuk lookup O(1)
        cluster_members = {c: set(np.where(self.labels_ == c)[0]) for c in range(self.K_auto_)}
        
        # 2. Konversi adjacency matrix ke list of sets untuk akses tetangga yang cepat
        adj_list = jnng_adj.tolil().rows

        for i in range(n):
            # Hitung jarak EW-LMD ke semua sentroid
            dists = [
                sum(self._dist_attr(col, X.iloc[i][col], self.centers_.iloc[c][col]) 
                    for col in X.columns) 
                for c in range(self.K_auto_)
            ]
            best_c = int(np.argmin(dists)) # Klaster terdekat secara geometris
            
            # Cek konektivitas topologi jNNG
            neighbors_i = set(adj_list[i])
            
            # Cari klaster mana saja yang memiliki anggota terhubung dengan titik i
            connected_clusters = [
                c for c, members in cluster_members.items() 
                if neighbors_i.intersection(members)
            ]
            
            # 3. Keputusan Alokasi
            if connected_clusters:
                # Jika ada koneksi, pilih klaster yang TERHUBUNG yang jaraknya paling minimum
                labels[i] = min(connected_clusters, key=lambda c: dists[c])
            else:
                # Fallback: Titik terisolasi (tidak ada edge ke klaster manapun)
                # Gunakan jarak geometris murni agar tidak error
                labels[i] = best_c

        return labels

    def _update_centers_refinement(self, X: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
        centers = []
        for c in range(self.K_auto_):
            cl = X[labels==c]
            if cl.empty: 
                centers.append(self.centers_.iloc[c].to_dict())
                continue
            c_dict = {col: _mode(cl[col]) if self.attr_types[col] in ('Nominal','Ordinal') else cl[col].mean() for col in X.columns}
            centers.append(c_dict)
        return pd.DataFrame(centers)

    def fit(self, df: pd.DataFrame):
        X = df.drop(columns=["Label"], errors="ignore").copy()
        for col in X.columns:
            if self.attr_types[col] == "Numeric":
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(X[col].mean())

        # 1. HYBRID PARAMETER TUNING
        structure = self._compute_structure(X)
        params = hybrid_selection_conservative(X, self.attr_types, structure)
        self.dis_const = params['dist_cons']
        self.eta_cat = self.eta_num = params['eta']
        self.n_bins_numeric = params['n_bins_numeric']  # ← PERBAIKAN
        self.auto_params_log = params.copy()
        if self.verbose: print(f"[AUTONOMOUS] Hybrid-tuned: {self.auto_params_log}")

        print("[Phase 0] Computing initializing distances...")
        # self._compute_entropy_weights(X)
        self._init_distances(X)
        
        print("[Phase 1] Building jNNG with LMD distance...")
        D = self._compute_distance_matrix(X)
        n = len(X)
        median_dist = np.median(D[np.triu_indices(n, k=1)])
        D_scaled = D / (median_dist + 1e-12)
        
        jnng_sparse, geo_dist, jnng_adj = self._build_jnng(D_scaled)
        
        # Menggunakan Local Scaling di sini
        rho_gjnn = self._gaussian_kernel_density(geo_dist)
        rho_tilde = self._two_round_sigmoid_correction(rho_gjnn)
        
        print("[Phase 2] Running Bombing process (Automatic K detection)...")
        K_before, labels_bombing = self._run_bombing_phase(X, D_scaled, geo_dist, rho_tilde, jnng_adj)
        print(f"   -> Detected K = {K_before}")
        
        self.labels_ = labels_bombing
        self.centers_ = self._compute_initial_centroids(X, labels_bombing)
        
        print("[Phase 2.5] Evaluating structural coherence for conditional merging...")
        metrics = compute_merge_metrics(labels_bombing, jnng_adj, rho_tilde)
        trigger = should_merge_parameter_free(metrics, self.K_max_, T2=self.T2)
        
        if trigger:
            print("   -> Trigger activated. Initializing Post-Hoc Merger...")
            merger = PostHocMerger(self, X, D_scaled, jnng_adj)
            merge_results = merger.smart_merge_stable(min_K=2, max_merge_ratio=1.0)
            labels_refined = merge_results['labels_merged']
            K_after = merge_results['K_final']
            print(f"   -> Merge Result: K {merge_results['K_before']} → {K_after}")
        else:
            print("   -> Structure coherent. Skipping merging phase.")
            labels_refined = labels_bombing
            K_after = K_before
            
        self.labels_ = labels_refined
        self.K_auto_ = K_after

        print("[Phase 3] Computing centroids from full cluster membership (DE excluded per ablation result)...")
        self.centers_ = self._compute_initial_centroids(X, labels_refined)
        
        print("[Phase 4-6] LMD Iterative Refinement Loop...")
        prev_labels = self.labels_.copy()
        best_silhouette = -1.0
        best_labels = self.labels_.copy()
        restart_count = 0
        
        for step in range(self.max_iter_refine):
            current_alpha = 0.5 if step < 2 else self.alpha_reg
            orig_alpha = self.alpha_reg
            self.alpha_reg = current_alpha
            
            self._update_nominal_ordinal_phi(X, self.labels_)
            self._update_numeric_phi(X, self.labels_)
            self.alpha_reg = orig_alpha
            
            self.labels_ = self._assign_refinement(X, jnng_adj)
            self.centers_ = self._update_centers_refinement(X, self.labels_)
            
            if step >= 2:
                ari_internal = adjusted_rand_score(prev_labels, self.labels_)
                
                # === PERBAIKAN: TRAP DETECTION via Silhouette ===
                sample_idx = np.random.choice(len(X), min(1000, len(X)), replace=False) if len(X) > 1000 else np.arange(len(X))
                D_sample = np.zeros((len(sample_idx), len(sample_idx)))
                for i, idx_i in enumerate(sample_idx):
                    for j, idx_j in enumerate(sample_idx):
                        # Menggunakan fungsi distance yang sudah ada
                        D_sample[i,j] = self.distance(X.iloc[idx_i], X.iloc[idx_j])
                
                try:
                    current_sil = silhouette_score(D_sample, self.labels_[sample_idx], metric='precomputed')
                except ValueError:
                    current_sil = 0.0
                    
                if current_sil > best_silhouette:
                    best_silhouette = current_sil
                    best_labels = self.labels_.copy()
                    
                print(f"   Iter {step+1}: Internal ARI = {ari_internal:.4f} | Silhouette = {current_sil:.4f}")
                
                if ari_internal > self.conv_threshold:
                    if current_sil < 0.1 and restart_count < 2:
                        print(f"   -> [Trap Detected] Silhouette < 0.1. Perturbing labels to escape local optimum...")
                        noise_mask = np.random.rand(len(X)) < 0.2
                        self.labels_[noise_mask] = self._rng.integers(0, self.K_auto_, size=np.sum(noise_mask))
                        self.centers_ = self._update_centers_refinement(X, self.labels_)
                        restart_count += 1
                        prev_labels = self.labels_.copy()
                        continue
                    else:
                        print("   -> Convergence reached.")
                        break
                        
            prev_labels = self.labels_.copy()

        if best_silhouette > current_sil:
            if self.verbose: print(f"   -> [Recovery] Restoring best labels from Iter with Silhouette = {best_silhouette:.4f}")
            self.labels_ = best_labels
            self.centers_ = self._update_centers_refinement(X, self.labels_)
            
        return self

import numpy as np
import pandas as pd
from copy import deepcopy
from typing import Tuple, Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

# Asumsi: Class LMDBombingHybrid dan fungsi pendukungnya sudah di-load di environment Anda
# from ewlmd_bombing_hybrid import EWLMDBombingHybrid, compute_ewlmd_distance_matrix

class SubsampledBombingLMDHybrid:
    """
    Algoritma Bombing-LMD Hybrid Clustering dengan Sub-sampling Strategis.
    Dirancang untuk menangani dataset mixed-type berskala besar (Large-Scale Mixed-Type Clustering).
    """
    
    def __init__(self, 
                 base_model_class, 
                 attr_types: Dict[str, str], 
                 ordinal_orders: Dict[str, List[str]] = None,
                 n_sample: int = 2000, 
                 threshold: int = 5000,
                 max_iter_refine: int = 15,
                 conv_threshold: float = 0.98,
                 random_state: int = 42,
                 **base_model_kwargs):
        """
        Parameters:
        - base_model_class: Class EWLMDBombingHybrid (atau variannya).
        - attr_types: Dictionary tipe atribut.
        - n_sample: Ukuran subsampel untuk fase Bombing.
        - threshold: Batas ukuran dataset untuk mengaktifkan sub-sampling.
        - base_model_kwargs: Parameter tambahan untuk base_model_class (eta, alpha_reg, dll).
        """
        self.base_model_class = base_model_class
        self.attr_types = attr_types
        self.ordinal_orders = ordinal_orders or {}
        self.n_sample = n_sample
        self.threshold = threshold
        self.max_iter_refine = max_iter_refine
        self.conv_threshold = conv_threshold
        self.random_state = random_state
        self.base_kwargs = base_model_kwargs
        
        self._rng = np.random.default_rng(random_state)
        
        # Atribut output
        self.labels_ = None
        self.centers_ = None
        self.K_auto_ = None
        self.model_sub_ = None  # Menyimpan model dari fase subsampel untuk referensi

    # =========================================================================
    # PHASE 0 & 1: SUBSAMPLING STRATEGY
    # =========================================================================
    def _stratified_subsample(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Melakukan random stratified subsampling berdasarkan atribut kategorikal 
        untuk menjaga representasi topologi global.
        """
        n = len(X)
        all_indices = np.arange(n)
        
        # Cari atribut nominal/ordinal pertama untuk stratifikasi
        strata_col = None
        for col, t in self.attr_types.items():
            if t in ('Nominal', 'Ordinal'):
                strata_col = col
                break
        
        if strata_col is None or X[strata_col].nunique() == 1:
            # Fallback: Pure random sampling jika tidak ada fitur kategorikal yang valid
            print("⚠️ [Subsampling] No valid stratification column found. Using pure random sampling.")
            idx_sub = self._rng.choice(n, size=self.n_sample, replace=False)
        else:
            # Stratified sampling
            strata_vals = X[strata_col].unique()
            n_per_stratum = int(np.ceil(self.n_sample / len(strata_vals)))
            idx_sub = []
            
            for val in strata_vals:
                idx_s = np.where(X[strata_col] == val)[0]
                n_s = min(n_per_stratum, len(idx_s))
                if n_s > 0:
                    sampled = self._rng.choice(idx_s, size=n_s, replace=False)
                    idx_sub.extend(sampled)
            
            idx_sub = np.array(list(set(idx_sub))) # Hapus duplikat jika ada
            
            # Tambah titik acak jika kurang dari n_sample
            if len(idx_sub) < self.n_sample:
                idx_remaining = np.setdiff1d(all_indices, idx_sub)
                n_extra = min(self.n_sample - len(idx_sub), len(idx_remaining))
                if n_extra > 0:
                    extra = self._rng.choice(idx_remaining, size=n_extra, replace=False)
                    idx_sub = np.concatenate([idx_sub, extra])
                    
            idx_sub = idx_sub[:self.n_sample] # Pastikan persis n_sample
            
        idx_sub = np.sort(idx_sub)
        idx_rest = np.setdiff1d(all_indices, idx_sub)
        
        return idx_sub, idx_rest

    # =========================================================================
    # PHASE 2: BOMBING PADA SUBSAMPEL
    # =========================================================================
    def _fit_subsample(self, X_sub: pd.DataFrame) -> Any:
        """
        Menjalankan pipeline Bombing-LMD lengkap pada subsampel.
        Mengembalikan model yang sudah terlatih (termasuk K, Centers, dan learned parameters).
        """
        print(f"[Phase 2] Running Bombing-LMD on subsample (n={len(X_sub)})...")
        
        model_sub = self.base_model_class(
            attr_types=self.attr_types,
            ordinal_orders=self.ordinal_orders,
            random_state=self.random_state,
            **self.base_kwargs
        )
        
        # Fit model pada subsampel
        # Catatan: Pastikan X_sub tidak memiliki kolom 'Label' jika base_model_class mendropnya secara otomatis
        model_sub.fit(X_sub)
        
        return model_sub

    # =========================================================================
    # PHASE 3: ASSIGN TITIK SISA
    # =========================================================================
    def assign_remaining(self, X: pd.DataFrame, idx_rest: np.ndarray, model_sub: Any) -> np.ndarray:
        """
        Mengalokasikan titik-titik di idx_rest ke klaster terdekat berdasarkan 
        metrik LMD yang sudah dipelajari (learned distance) dari model_sub.
        """
        print(f"[Phase 3] Assigning {len(idx_rest)} remaining points using learned LMD...")
        
        labels_rest = np.zeros(len(idx_rest), dtype=int)
        centers = model_sub.centers_
        
        # Hitung jarak dari setiap titik sisa ke semua centroid
        for i, idx in enumerate(idx_rest):
            x_i = X.iloc[idx]
            dists = []
            for c_idx in range(len(centers)):
                center = centers.iloc[c_idx]
                # Gunakan distance method dari model_sub
                d = 0.0
                for col in X.columns:
                    d += model_sub._dist_attr(col, x_i[col], center[col])
                dists.append(d)
            labels_rest[i] = np.argmin(dists)
            
        return labels_rest

    # =========================================================================
    # PHASE 4: REFINEMENT PADA DATA PENUH
    # =========================================================================
    def _run_refinement_full(self, X: pd.DataFrame, initial_labels: np.ndarray, model_sub: Any):
        """
        Menjalankan loop LMD Iterative Refinement pada dataset penuh.
        """
        print(f"[Phase 4] Running LMD Iterative Refinement on full dataset (n={len(X)})...")
        
        # Inisialisasi model penuh
        model_full = self.base_model_class(
            attr_types=self.attr_types,
            ordinal_orders=self.ordinal_orders,
            random_state=self.random_state,
            **self.base_kwargs
        )
        
        # Transfer learned parameters dari subsampel ke model penuh
        model_full.dis_const = model_sub.dis_const
        model_full.eta_cat = model_sub.eta_cat
        model_full.eta_num = model_sub.eta_num
        model_full.alpha_reg = model_sub.alpha_reg
        model_full.n_bins_numeric = model_sub.n_bins_numeric
        
        model_full.nominal_phi = deepcopy(model_sub.nominal_phi)
        model_full.ordinal_phi = deepcopy(model_sub.ordinal_phi)
        model_full.numeric_phi = deepcopy(model_sub.numeric_phi)
        model_full.nominal_dist = deepcopy(model_sub.nominal_dist)
        model_full.ordinal_adj_dist = deepcopy(model_sub.ordinal_adj_dist)
        model_full.num_min = deepcopy(model_sub.num_min)
        model_full.num_max = deepcopy(model_sub.num_max)
        model_full.num_bin_psi = deepcopy(model_sub.num_bin_psi)
        
        # Set state awal
        model_full.labels_ = initial_labels.copy()
        
        # ✅ PERBAIKAN KRITIS: Sinkronisasi K_auto_ dan centers_ dengan labels aktual
        unique_labels = np.unique(model_full.labels_)
        model_full.K_auto_ = len(unique_labels)
        
        # Remap labels agar kontinu 0..K-1 (mencegah IndexError jika ada gap/cluster kosong)
        label_map = {old_lbl: new_lbl for new_lbl, old_lbl in enumerate(unique_labels)}
        model_full.labels_ = np.array([label_map[l] for l in initial_labels])
        
        # Hitung ulang centers dari dataset penuh X agar shape-nya persis K_auto_
        # Ini menjamin self.centers_ memiliki tepat K_auto_ baris
        model_full.centers_ = model_full._compute_initial_centroids(X, model_full.labels_)
        
        # Re-build jNNG untuk dataset penuh
        print("  [Re-building jNNG for full dataset...]")
        D_full = model_full._compute_distance_matrix(X)
        median_dist = np.median(D_full[np.triu_indices(len(X), k=1)])
        D_scaled = D_full / (median_dist + 1e-12)
        
        jnng_sparse, geo_dist, jnng_adj = model_full._build_jnng(D_scaled)
        
        # Jalankan loop refinement
        prev_labels = model_full.labels_.copy()
        
        for step in range(self.max_iter_refine):
            # 1. Update phi
            model_full._update_nominal_ordinal_phi(X, model_full.labels_)
            model_full._update_numeric_phi(X, model_full.labels_)
            
            # 2. Re-assign labels
            model_full.labels_ = model_full._assign_refinement(X, jnng_adj)
            
            # 3. Update centers
            model_full.centers_ = model_full._update_centers_refinement(X, model_full.labels_)
            
            # 4. Cek konvergensi
            if step >= 3:
                from sklearn.metrics import adjusted_rand_score
                ari = adjusted_rand_score(prev_labels, model_full.labels_)
                if step % 2 == 0:
                    print(f"  [Iter {step}] ARI = {ari:.4f}")
                if ari > self.conv_threshold:
                    print(f"  -> Convergence reached at Iter {step}.")
                    break
                    
            prev_labels = model_full.labels_.copy()
            
        return model_full

    # =========================================================================
    # MAIN FIT METHOD
    # =========================================================================
    def fit(self, X: pd.DataFrame) -> 'SubsampledBombingLMDHybrid':
        """
        Pipeline utama: Subsampling -> Bombing -> Assign Rest -> Refinement.
        """
        n = len(X)
        print("="*60)
        print("SUBSAMPLED BOMBING-LMD HYBRID CLUSTERING")
        print("="*60)
        
        # PHASE 0: CEK SUBSAMPLING
        if n <= self.threshold:
            print(f"[Phase 0] Dataset size ({n}) <= threshold ({self.threshold}). Skipping subsampling.")
            idx_sub = np.arange(n)
            idx_rest = np.array([])
            X_sub = X
        else:
            print(f"[Phase 0] Dataset size ({n}) > threshold ({self.threshold}). Activating subsampling.")
            idx_sub, idx_rest = self._stratified_subsample(X)
            X_sub = X.iloc[idx_sub]
            
        # PHASE 2: BOMBING PADA SUBSAMPEL (Atau Full Dataset jika skip)
        self.model_sub_ = self._fit_subsample(X_sub)
        
        # PHASE 3: ASSIGN TITIK SISA
        if len(idx_rest) > 0:
            labels_rest = self.assign_remaining(X, idx_rest, self.model_sub_)
            
            # Gabungkan label
            labels_full = np.empty(n, dtype=int)
            labels_full[idx_sub] = self.model_sub_.labels_
            labels_full[idx_rest] = labels_rest
            
            # PHASE 4: REFINEMENT PADA DATA PENUH (Hanya jika ada titik sisa)
            self.model_full_ = self._run_refinement_full(X, labels_full, self.model_sub_)
        else:
            # PERBAIKAN: Jika tidak ada subsampling, model_sub_ sudah melakukan refinement penuh.
            # Langsung gunakan hasilnya tanpa redundansi.
            print("[Phase 4] Subsampling skipped. Using results from full-dataset Bombing-LMD directly.")
            self.model_full_ = self.model_sub_
            labels_full = self.model_sub_.labels_
        
        # Simpan hasil akhir
        self.labels_ = self.model_full_.labels_
        self.centers_ = self.model_full_.centers_
        self.K_auto_ = self.model_full_.K_auto_
        
        print(f"\n✅ Clustering Complete. Final K = {self.K_auto_}")
        return self

    def predict(self, X_new: pd.DataFrame) -> np.ndarray:
        """
        Memprediksi label untuk data baru menggunakan parameter yang sudah dipelajari.
        """
        if self.model_full_ is None:
            raise RuntimeError("Model belum di-fit. Panggil method fit() terlebih dahulu.")
            
        labels_new = np.zeros(len(X_new), dtype=int)
        for i in range(len(X_new)):
            x_i = X_new.iloc[i]
            dists = []
            for c_idx in range(len(self.centers_)):
                center = self.centers_.iloc[c_idx]
                d = self.model_full_.distance(x_i, center)
                dists.append(d)
            labels_new[i] = np.argmin(dists)
        return labels_new

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. FUNGSI DASAR: MATRIKS JARAK & MEDOID
# =============================================================================
def compute_ewlmd_distance_matrix(X, model):
    """Menghitung matriks jarak pairwise EW-LMD (Precomputed Distance Matrix)."""
    n_samples = X.shape[0]
    D = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            dist = model.base_model_class.distance(X.iloc[i], X.iloc[j])
            D[i, j] = dist
            D[j, i] = dist
    return D

def compute_lmd_distance_matrix(X, model):
    """
    Menghitung matriks jarak pairwise LMD (Precomputed Distance Matrix).
    Mendukung input baik dari Base Model instance maupun Subsampled Wrapper instance.
    """
    n_samples = X.shape[0]
    D = np.zeros((n_samples, n_samples))
    
    # 1. Identifikasi instance model yang sudah di-fit
    if hasattr(model, 'model_full_') and model.model_full_ is not None:
        fitted_model = model.model_full_
    elif hasattr(model, 'model_sub_') and model.model_sub_ is not None:
        fitted_model = model.model_sub_
    else:
        # Fallback jika yang dipassing langsung adalah instance base model (LMDBombingHybrid)
        fitted_model = model 
        
    # 2. Hitung jarak pairwise
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            # Panggil method distance dari INSTANCE (fitted_model)
            dist = fitted_model.distance(X.iloc[i], X.iloc[j])
            D[i, j] = dist
            D[j, i] = dist
            
    return D

def find_cluster_medoids(D, labels):
    """
    Mencari medoid (pusat representatif) untuk setiap klaster berdasarkan matriks jarak D.
    Medoid adalah titik di dalam klaster yang memiliki total jarak ke titik lain paling minimum.
    Ini menggantikan konsep 'mean centroid' yang tidak valid untuk data campuran.
    """
    unique_labels = np.unique(labels)
    medoids = {}
    
    for k in unique_labels:
        idx_k = np.where(labels == k)[0]
        if len(idx_k) == 1:
            medoids[k] = idx_k[0]
            continue
            
        # Sub-matriks jarak untuk klaster k
        D_k = D[np.ix_(idx_k, idx_k)]
        # Jumlahkan jarak setiap titik ke semua titik lain di klaster yang sama
        sum_distances = np.sum(D_k, axis=1)
        # Medoid adalah indeks lokal dengan jarak total minimum
        local_medoid_idx = np.argmin(sum_distances)
        # Konversi ke indeks global
        medoids[k] = idx_k[local_medoid_idx]
        
    return medoids

# =============================================================================
# 2. IMPLEMENTASI METRIK EVALUASI BERBASIS MATRIKS JARAK EW-LMD
# =============================================================================

def silhouette_score_ewlmd(D, labels):
     # ✅ PERBAIKAN: Konversi D ke numpy array untuk menghindari KeyError pada DataFrame
    D = np.asarray(D)
    n = len(labels)
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2: return 0.0
    s_i = np.zeros(n)
    for i in range(n):
        cluster_i = labels[i]
        mask_intra = (labels == cluster_i)
        mask_intra[i] = False
        n_intra = np.sum(mask_intra)
        a_i = np.mean(D[i, mask_intra]) if n_intra > 0 else 0.0
        b_i = np.inf
        for c in unique_labels:
            if c == cluster_i: continue
            mask_inter = (labels == c)
            if np.sum(mask_inter) == 0: continue
            avg_dist = np.mean(D[i, mask_inter])
            if avg_dist < b_i: b_i = avg_dist
        denom = max(a_i, b_i)
        s_i[i] = (b_i - a_i) / denom if denom > 1e-12 else 0.0
    return float(np.mean(s_i))

def dunn_index_ewlmd(D, labels):
     # ✅ PERBAIKAN: Konversi D ke numpy array untuk menghindari KeyError pada DataFrame
    D = np.asarray(D)
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2: return 0.0
    diameters = []
    for c in unique_labels:
        mask = (labels == c)
        idx = np.where(mask)[0]
        if len(idx) < 2: diameters.append(0.0)
        else:
            sub_D = D[np.ix_(idx, idx)]
            diameters.append(np.max(sub_D[np.triu_indices(len(idx), k=1)]))
    max_diam = max(diameters) if diameters else 1e-12
    if max_diam < 1e-12: max_diam = 1e-12
    min_inter_dist = np.inf
    for i_idx, c1 in enumerate(unique_labels):
        for c2 in unique_labels[i_idx+1:]:
            mask1, mask2 = (labels == c1), (labels == c2)
            inter_D = D[mask1][:, mask2]
            min_dist = np.min(inter_D)
            if min_dist < min_inter_dist: min_inter_dist = min_dist
    return float(min_inter_dist / max_diam)

def davies_bouldin_index_ewlmd(D, labels):
    # =========================================================================
    # LANGKAH 1: PAKSA KONVERSI KE NUMPY ARRAY (ULTRA-ROBUST)
    # =========================================================================
    # Cek jika D adalah pandas DataFrame, gunakan .values untuk konversi paling aman
    if hasattr(D, 'values'):
        D = D.values
    # Jika bukan numpy array, konversi paksa
    if not isinstance(D, np.ndarray):
        D = np.asarray(D)
        
    # Debugging: Pastikan shape dan tipe data benar
    # print(f"[DEBUG] D type: {type(D)}, shape: {D.shape}")

    unique_labels = np.unique(labels)
    k = len(unique_labels)
    if k < 2: 
        return np.inf
    
    scatter = {}
    centroids = {}
    
    # =========================================================================
    # LANGKAH 2: HITUNG SCATTER & CENTROID (MEDOID)
    # =========================================================================
    for c in unique_labels:
        mask = (labels == c)
        idx = np.where(mask)[0]
        
        if len(idx) == 1: 
            scatter[c] = 0.0
            centroids[c] = idx[0] # Ambil satu-satunya titik sebagai centroid
        else:
            # Ambil sub-matriks jarak untuk cluster ini
            sub_D = D[np.ix_(idx, idx)]
            avg_dist = np.mean
    R = []
    for i in unique_labels:
        R_i = 0
        for j in unique_labels:
            if i == j: continue
            d_ij = D[centroids[i], centroids[j]]
            if d_ij < 1e-12: continue
            R_ij = (scatter[i] + scatter[j]) / d_ij
            if R_ij > R_i: R_i = R_ij
        R.append(R_i)
    return np.mean(R)

def calinski_harabasz_ewlmd(D, labels):
     # ✅ PERBAIKAN: Konversi D ke numpy array untuk menghindari KeyError pada DataFrame
    D = np.asarray(D)
    n = len(labels)
    unique_labels = np.unique(labels)
    k = len(unique_labels)
    if k < 2 or k == n: return 0.0
    SS_W = 0
    centroids = {}
    n_c = {}
    for c in unique_labels:
        mask = (labels == c)
        idx = np.where(mask)[0]
        n_c[c] = len(idx)
        if len(idx) < 2: continue
        sub_D = D[np.ix_(idx, idx)]
        SS_W += np.sum(np.triu(sub_D, k=1))
        avg_dist = np.mean(sub_D, axis=1)
        centroids[c] = idx[np.argmin(avg_dist)]
    SS_B = 0
    for i in unique_labels:
        for j in unique_labels:
            if i >= j: continue
            d_ij = D[centroids[i], centroids[j]]
            SS_B += n_c[i] * n_c[j] / (n_c[i] + n_c[j]) * (d_ij ** 2)
    return (SS_B / (k - 1)) / (SS_W / (n - k)) if (n - k) > 0 else 0.0

def purity_score(y_true, y_pred):
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    return np.sum([grp["y_true"].value_counts().max() for _, grp in df.groupby("y_pred")]) / len(df)

def category_utility(df, labels):
    cu = 0
    k = len(np.unique(labels))
    for cl in np.unique(labels):
        subset = df[labels==cl]
        p = len(subset)/len(df)
        for c in df.columns:
            freq_global = df[c].value_counts(normalize=True)
            freq_local = subset[c].value_counts(normalize=True)
            cu += p * (freq_local.pow(2).sum() - freq_global.pow(2).sum())
    return cu / k

def entropy_reduction(df, labels, attr_types):
    # Hanya hitung untuk Nominal dan Ordinal
    valid_cols = [c for c in df.columns if attr_types.get(c) in ('Nominal', 'Ordinal')]
    if not valid_cols: return 0.0
    
    df_cat = df[valid_cols]
    
    # Shannon Entropy yang benar: -sum(p * log(p))
    def shannon_entropy(subset):
        H = 0.0
        for c in subset.columns:
            probs = subset[c].value_counts(normalize=True)
            H += -np.sum(probs * np.log(probs + 1e-12))
        return H

    H_before = shannon_entropy(df_cat)
    H_after = 0
    for cl in np.unique(labels):
        subset = df_cat[labels == cl]
        if len(subset) == 0: continue
        p = len(subset) / len(df_cat)
        H_after += p * shannon_entropy(subset)
        
    return H_before - H_after # Entropy Reduction seharusnya H_before - H_after (penurunan entropi)

def visualisasi_density_cluster(D_valid, y_pred_valid):    
    # ==========================================
    # PERSIAPAN DATA (SAMA SEPERTI SEBELUMNYA)
    # ==========================================
    # Pastikan D_valid dan y_pred_valid sudah tersedia dari kode sebelumnya
    # D_valid = D_ewlmd[np.ix_(mask_valid, mask_valid)]
    # y_pred_valid = model.labels_[mask_valid]
    
    # Pastikan matriks jarak simetris
    D_valid_sym = (D_valid + D_valid.T) / 2.0
    np.fill_diagonal(D_valid_sym, 0)
    
    # Reduksi dimensi menggunakan MDS
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, n_init=4)
    coords_2d = mds.fit_transform(D_valid_sym)
    
    # Buat DataFrame untuk plotting
    df_plot = pd.DataFrame(coords_2d, columns=['Dim1', 'Dim2'])
    df_plot['Cluster'] = y_pred_valid
    
    n_clusters = len(np.unique(y_pred_valid))
    palette_name = 'tab20' if n_clusters <= 20 else 'husl'
    colors = sns.color_palette(palette_name, n_clusters)
    
    # ==========================================
    # VISUALISASI DENSITY CLUSTERING
    # ==========================================
    fig = plt.figure(figsize=(20, 15))
    
    # ----------------------------------------
    # PLOT 1: Scatter dengan Alpha (Visual Density Sederhana)
    # ----------------------------------------
    ax1 = plt.subplot(2, 2, 1)
    sns.scatterplot(
        data=df_plot, x='Dim1', y='Dim2', hue='Cluster',
        palette=palette_name, s=40, alpha=0.3, edgecolor='none', ax=ax1
    )
    ax1.set_title('Density Visualization (Alpha Blending)\nArea gelap = kepadatan tinggi', 
                  fontsize=13, fontweight='bold')
    ax1.set_xlabel('MDS Dimension 1')
    ax1.set_ylabel('MDS Dimension 2')
    ax1.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, linestyle=':', alpha=0.5)
    
    # ----------------------------------------
    # PLOT 2: KDE Plot (Kernel Density Estimation) - Semua Cluster
    # ----------------------------------------
    ax2 = plt.subplot(2, 2, 2)
    for i, cluster_id in enumerate(np.unique(y_pred_valid)):
        subset = df_plot[df_plot['Cluster'] == cluster_id]
        sns.kdeplot(
            data=subset, x='Dim1', y='Dim2', ax=ax2,
            color=colors[i], fill=True, alpha=0.3,
            levels=5, thresh=0.1
        )
    
    # Tambahkan scatter di atas KDE
    sns.scatterplot(
        data=df_plot, x='Dim1', y='Dim2', hue='Cluster',
        palette=palette_name, s=15, alpha=0.5, edgecolor='none', 
        ax=ax2, legend=False
    )
    ax2.set_title('KDE (Kernel Density Estimation)\nArea berwarna = distribusi density', 
                  fontsize=13, fontweight='bold')
    ax2.set_xlabel('MDS Dimension 1')
    ax2.set_ylabel('MDS Dimension 2')
    ax2.grid(True, linestyle=':', alpha=0.5)
    
    # ----------------------------------------
    # PLOT 3: Contour Plot (Garis Kontur Density)
    # ----------------------------------------
    ax3 = plt.subplot(2, 2, 3)
    x_min, x_max = df_plot['Dim1'].min() - 0.5, df_plot['Dim1'].max() + 0.5
    y_min, y_max = df_plot['Dim2'].min() - 0.5, df_plot['Dim2'].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    for i, cluster_id in enumerate(np.unique(y_pred_valid)):
        subset = df_plot[df_plot['Cluster'] == cluster_id]
        if len(subset) < 3:
            continue
        
        # Hitung KDE
        kde = gaussian_kde(subset[['Dim1', 'Dim2']].T)
        positions = np.vstack([xx.ravel(), yy.ravel()])
        density = kde(positions).reshape(xx.shape)
        
        # Plot contour lines
        contour = ax3.contour(xx, yy, density, levels=8, 
                             colors=[colors[i]], alpha=0.6, linewidths=1.5)
        ax3.clabel(contour, inline=True, fontsize=8, fmt='%.2f')
        
        # Plot titik centroid
        centroid_x = subset['Dim1'].mean()
        centroid_y = subset['Dim2'].mean()
        ax3.plot(centroid_x, centroid_y, '*', color=colors[i], 
                markersize=15, markeredgecolor='black', markeredgewidth=1.5)
        ax3.text(centroid_x, centroid_y, f'C{cluster_id}', 
                fontsize=9, ha='center', va='bottom', fontweight='bold')
    
    # Tambahkan scatter point
    sns.scatterplot(
        data=df_plot, x='Dim1', y='Dim2', hue='Cluster',
        palette=palette_name, s=20, alpha=0.4, edgecolor='none', 
        ax=ax3, legend=False
    )
    ax3.set_title('Contour Plot (Garis Kepadatan)\nGaris rapat = density tinggi', 
                  fontsize=13, fontweight='bold')
    ax3.set_xlabel('MDS Dimension 1')
    ax3.set_ylabel('MDS Dimension 2')
    ax3.grid(True, linestyle=':', alpha=0.5)
    
    # ----------------------------------------
    # PLOT 4: Heatmap Density dengan Filled Contour
    # ----------------------------------------
    ax4 = plt.subplot(2, 2, 4)
    
    # Hitung density total semua data point
    kde_total = gaussian_kde(df_plot[['Dim1', 'Dim2']].T)
    density_total = kde_total(positions).reshape(xx.shape)
    
    # Plot heatmap density
    im = ax4.contourf(xx, yy, density_total, levels=15, cmap='viridis', alpha=0.7)
    plt.colorbar(im, ax=ax4, label='Density Score')
    
    # Overlay contour lines per cluster
    for i, cluster_id in enumerate(np.unique(y_pred_valid)):
        subset = df_plot[df_plot['Cluster'] == cluster_id]
        if len(subset) < 3:
            continue
        kde = gaussian_kde(subset[['Dim1', 'Dim2']].T)
        density = kde(positions).reshape(xx.shape)
        ax4.contour(xx, yy, density, levels=3, colors=[colors[i]], 
                   linewidths=2, alpha=0.8)
    
    # Tambahkan scatter point
    sns.scatterplot(
        data=df_plot, x='Dim1', y='Dim2', hue='Cluster',
        palette=palette_name, s=20, alpha=0.6, edgecolor='white', 
        ax=ax4, legend=False
    )
    ax4.set_title('Heatmap Density (Filled Contour)\nWarna terang = area paling padat', 
                  fontsize=13, fontweight='bold')
    ax4.set_xlabel('MDS Dimension 1')
    ax4.set_ylabel('MDS Dimension 2')
    
    plt.tight_layout()
    plt.savefig('clustering_density_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # ==========================================
    # STATISTIK DENSITY PER CLUSTER
    # ==========================================
    print("\n" + "="*60)
    print("STATISTIK DENSITY PER CLUSTER")
    print("="*60)
    
    for cluster_id in np.unique(y_pred_valid):
        subset = df_plot[df_plot['Cluster'] == cluster_id]
        
        # Hitung density di centroid cluster
        centroid = subset[['Dim1', 'Dim2']].mean().values
        kde = gaussian_kde(df_plot[['Dim1', 'Dim2']].T, bw_method='scott')
        density_at_centroid = kde(centroid)[0]
        
        # Hitung average density untuk semua point di cluster ini
        densities = kde.evaluate(subset[['Dim1', 'Dim2']].T)
        avg_density = np.mean(densities)
        
        # Hitung spread (std dev) sebagai indikator kekompakan
        spread_x = subset['Dim1'].std()
        spread_y = subset['Dim2'].std()
        
        print(f"\nCluster {cluster_id}:")
        # Tambahkan kolom density ke DataFrame
        subset['Density'] = densities
        
        print(f"✅ Density berhasil dihitung untuk {len(subset)} titik data.")
        print(f"   Range density: [{densities.min():.6f}, {densities.max():.6f}]")
        
        # ==========================================
        # TAMPILKAN DENSITY PER TITIK (TABEL)
        # ==========================================
        print("\n" + "="*80)
        print("CONTOH NILAI DENSITY PER TITIK (10 titik pertama)")
        print("="*80)
        print(subset[['Dim1', 'Dim2', 'Cluster', 'Density']].head(10).to_string(index=True))
        print(f"\n  - Jumlah Data Point: {len(subset)}")
        print(f"  - Density di Centroid: {density_at_centroid:.4f}")
        print(f"  - Average Density: {avg_density:.4f}")
        print(f"  - Spread (Dim1): {spread_x:.3f}")
        print(f"  - Spread (Dim2): {spread_y:.3f}")
        print(f"  - Compactness Score: {1/(spread_x + spread_y):.4f}")
    
    print("="*60)

def visualize_bombing_clusters(model, X: pd.DataFrame, D_matrix: np.ndarray, true_labels: np.ndarray = None, save_path: str = None):
    """
    Memvisualisasikan hasil clustering EW-LMD Bombing Hybrid.
    
    Parameters:
    - model: Instance dari EWLMDBombingHybrid yang sudah di-fit.
    - X: DataFrame asli (tanpa kolom 'Label').
    - D_matrix: Matriks jarak pairwise (D_scaled atau matriks jarak EW-LMD akhir).
    - true_labels: Array label ground truth (opsional, untuk judul).
    - save_path: Path untuk menyimpan gambar (opsional).
    """
    print("[Visualization] Projecting mixed-type data to 2D space using MDS (precomputed distance)...")
    
    # 1. PROYEKSI 2D MENGGUNAKAN MDS (PENTING: Menggunakan jarak precomputed)
    # MDS mempertahankan struktur jarak global lebih baik daripada t-SNE untuk analisis densitas
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, normalized_stress='auto')
    try:
        X_2d = mds.fit_transform(D_matrix)
    except Exception as e:
        print(f"[Warning] MDS failed ({e}). Falling back to classical MDS...")
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
        X_2d = mds.fit_transform(D_matrix)

    # 2. IDENTIFIKASI PERAN TITIK (Berdasarkan filosofi Bombing)
    labels = model.labels_
    unique_clusters = np.unique(labels[labels != -1])
    
    # Mask untuk Reachable (Core) dan Boundary
    reachable_mask = np.zeros(len(X), dtype=bool)
    if hasattr(model, '_reachable_indices') and model._reachable_indices:
        for r_idx in model._reachable_indices.values():
            reachable_mask[r_idx] = True
            
    # Boundary adalah titik yang ter-cluster tapi BUKAN reachable
    boundary_mask = ~reachable_mask & (labels != -1)
    
    # 3. ESTIMASI DENSITY LOKAL (Untuk ukuran titik)
    # Menggunakan jarak ke tetangga ke-5 sebagai proksi densitas lokal
    k = 5
    sorted_D = np.sort(D_matrix, axis=1)
    # Hindari pembagian dengan nol, ambil rata-rata jarak ke k tetangga terdekat (indeks 1 sampai k)
    local_density = 1.0 / (np.mean(sorted_D[:, 1:k+1], axis=1) + 1e-6)
    
    # Normalisasi densitas ke rentang [0.2, 1.0] untuk ukuran marker yang bagus
    density_norm = (local_density - local_density.min()) / (local_density.max() - local_density.min() + 1e-6)
    marker_sizes = 15 + (density_norm * 80) 

    # 4. MENENTUKAN POSISI CENTER (Centroid) DI RUANG 2D
    # Karena centroid adalah konsep abstrak di ruang campuran, kita cari titik data 
    # yang PALING DEKAT dengan centroid di setiap cluster untuk diplot sebagai "Center"
    center_2d_coords = []
    for c in unique_clusters:
        cluster_indices = np.where(labels == c)[0]
        # Cari titik di cluster ini yang memiliki jarak rata-rata terkecil ke semua titik lain di cluster yang sama
        # Ini adalah pendekatan "medoid" yang paling stabil untuk divisualisasikan
        sub_D = D_matrix[np.ix_(cluster_indices, cluster_indices)]
        mean_dist_to_others = np.mean(sub_D, axis=1)
        medoid_local_idx = np.argmin(mean_dist_to_others)
        medoid_global_idx = cluster_indices[medoid_local_idx]
        center_2d_coords.append(X_2d[medoid_global_idx])
        
    center_2d_coords = np.array(center_2d_coords)

    # 5. PLOTTING
    fig, ax = plt.subplots(figsize=(12, 9))
    palette = sns.color_palette("husl", len(unique_clusters))
    
    # Layer 1: Plot Boundary Points (Transparan, marker 'o' dengan edge)
    if np.any(boundary_mask):
        ax.scatter(X_2d[boundary_mask, 0], X_2d[boundary_mask, 1], 
                   c='gray', s=20, alpha=0.4, marker='o', edgecolors='black', linewidths=0.5, label='Boundary Region')
        
    # Layer 2: Plot Reachable (Core) Points (Ukuran bervariasi berdasarkan density, solid)
    for i, c in enumerate(unique_clusters):
        cluster_mask = (labels == c) & reachable_mask
        if np.any(cluster_mask):
            ax.scatter(X_2d[cluster_mask, 0], X_2d[cluster_mask, 1], 
                       c=[palette[i]], s=marker_sizes[cluster_mask], alpha=0.7, 
                       marker='o', edgecolors='white', linewidths=0.5, label=f'Cluster {c} (Reachable)')
            
    # Layer 3: Plot Centers (Bintang besar, warna kontras)
    if len(center_2d_coords) > 0:
        ax.scatter(center_2d_coords[:, 0], center_2d_coords[:, 1], 
                   c='gold', s=250, marker='*', edgecolors='black', linewidths=1.5, 
                   zorder=10, label='Cluster Centers (Medoid Proxy)')

    # 6. KONFIGURASI TAMPILAN
    title = f"EW-LMD Bombing Hybrid Clustering (K = {len(unique_clusters)})"
    if true_labels is not None:
        # Hitung ARI sederhana untuk judul jika true_labels disediakan
        from sklearn.metrics import adjusted_rand_score
        ari = adjusted_rand_score(true_labels, labels)
        title += f" | ARI vs True: {ari:.3f}"
        
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel("MDS Dimension 1", fontsize=12)
    ax.set_ylabel("MDS Dimension 2", fontsize=12)
    
    # Custom Legend
    handles, labels_legend = ax.get_legend_handles_labels()
    # Urutkan legend agar rapi
    order = [len(unique_clusters)] # Center dulu
    order.extend(range(len(unique_clusters))) # Lalu reachable
    if np.any(boundary_mask):
        order.append(len(unique_clusters) + 1) # Terakhir boundary
        
    # Filter handles yang valid
    valid_handles = [handles[i] for i in order if i < len(handles)]
    valid_labels = [labels_legend[i] for i in order if i < len(labels_legend)]
    
    ax.legend(valid_handles, valid_labels, loc='upper right', fontsize=10, framealpha=0.9)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[Visualization] Saved to {save_path}")
    else:
        plt.show()
        
    plt.close()

import numpy as np
import pandas as pd
from collections import Counter
from copy import deepcopy

# ---------------------------------------------------
# Fungsi bantu
# ---------------------------------------------------
def mixed_distance(x, c, numeric_cols, cat_cols):
    """
    Menghitung jarak mixed-type seperti k-prototypes.
    Missing values (None/np.nan) diabaikan.
    """
    dist = 0.0
    
    # Bagian numerik: Euclidean
    for col in numeric_cols:
        val_x = x[col]
        val_c = c[col]
        if pd.notna(val_x) and pd.notna(val_c):
            # Pastikan keduanya numeric — aman karena sudah dikonversi di awal
            dist += (float(val_x) - float(val_c)) ** 2

    # Bagian kategorikal: simple matching
    for col in cat_cols:
        val_x = x[col]
        val_c = c[col]
        if pd.notna(val_x) and pd.notna(val_c):
            dist += 1 if str(val_x) != str(val_c) else 0
    
    return dist


def compute_centroid(df, cluster_members, numeric_cols, cat_cols):
    """
    Menghitung pusat cluster:
    - numerik: mean dari nilai yang tidak missing
    - kategorikal: modus (fallback ke nilai pertama jika semua NaN)
    """
    centroid = {}
    cluster_df = df.loc[cluster_members]

    for col in numeric_cols:
        # Ambil nilai non-null dan hitung mean
        values = pd.to_numeric(cluster_df[col], errors='coerce').dropna()
        centroid[col] = values.mean() if len(values) > 0 else np.nan

    for col in cat_cols:
        # Mode dari nilai non-null
        values = cluster_df[col].dropna()
        if len(values) > 0:
            mode_series = values.mode()
            centroid[col] = mode_series.iloc[0] if len(mode_series) > 0 else np.nan
        else:
            centroid[col] = np.nan

    return centroid


# ---------------------------------------------------
# Implementasi One-Step Imputation (imp.onestep)
# ---------------------------------------------------

def one_step_kprototypes(df, k, numeric_cols, cat_cols, max_iter=20):
    """
    One-step imputation using k-prototypes.
    
    Parameters:
        df (pd.DataFrame): Input data (may contain missing values)
        k (int): Number of clusters
        numeric_cols (list): List of numeric column names
        cat_cols (list): List of categorical column names
        max_iter (int): Max iterations
    
    Returns:
        df_imputed (pd.DataFrame): Imputed DataFrame
        clusters (dict): {cluster_id: [indices]}
        centroids (list of dict): Final centroids
    """
    # ✅ Ensure inputs are lists (not dicts!)
    if not isinstance(numeric_cols, list):
        raise TypeError("`numeric_cols` must be a list.")
    if not isinstance(cat_cols, list):
        raise TypeError("`cat_cols` must be a list.")
    
    df = df.copy()
    n = len(df)

    # ✅ 🔑 PERBAIKAN UTAMA: Konversi kolom numerik ke numeric (float), aman terhadap string/missing
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            raise KeyError(f"Column '{col}' in numeric_cols not found in DataFrame.")

    # Pastikan centroid juga akan berisi numeric (kami konversi saat inisialisasi & update)

    # 1. Inisialisasi centroid secara acak
    centroids = []
    # Hindari error jika n < k
    if k > n:
        raise ValueError(f"k={k} > number of samples ({n}). Reduce k.")
        
    chosen = np.random.choice(n, k, replace=False)
    for idx in chosen:
        row = df.loc[idx].to_dict()
        # Konversi numeric cols di centroid juga
        for col in numeric_cols:
            if col in row:
                val = row[col]
                row[col] = float(val) if pd.notna(val) else np.nan
        centroids.append(row)

    clusters = {}
    for iteration in range(max_iter):
        # 2. Assignment: tentukan cluster setiap objek
        clusters = {i: [] for i in range(k)}

        for i in range(n):
            x = df.loc[i].to_dict()  # aman: Series → dict
            distances = [
                mixed_distance(x, centroids[c], numeric_cols, cat_cols)
                for c in range(k)
            ]
            cluster_id = int(np.argmin(distances))  # pastikan int
            clusters[cluster_id].append(i)

        # 3. Update centroid
        new_centroids = []
        for c in range(k):
            members = clusters[c]
            if len(members) > 0:
                cent = compute_centroid(df, members, numeric_cols, cat_cols)
                # Pastikan numeric cols di centroid adalah float
                for col in numeric_cols:
                    if col in cent and pd.notna(cent[col]):
                        cent[col] = float(cent[col])
                new_centroids.append(cent)
            else:
                # Jika cluster kosong, pilih random baru & konversi numeric
                new_row = df.loc[np.random.choice(n)].to_dict()
                for col in numeric_cols:
                    if col in new_row:
                        val = new_row[col]
                        new_row[col] = float(val) if pd.notna(val) else np.nan
                new_centroids.append(new_row)

        # 4. Cek konvergensi (hanya numeric cols)
        converged = True
        for j in range(k):
            old_num = np.array([centroids[j].get(col, np.nan) for col in numeric_cols], dtype=float)
            new_num = np.array([new_centroids[j].get(col, np.nan) for col in numeric_cols], dtype=float)
            if not np.allclose(old_num, new_num, equal_nan=True):
                converged = False
                break

        if converged:
            print(f"✅ Converged at iteration {iteration + 1}")
            centroids = new_centroids
            break

        centroids = deepcopy(new_centroids)
    else:
        print(f"⚠️ Max iterations ({max_iter}) reached without full convergence.")

    # ---------------------------------------------------
    # ONE-STEP IMPUTATION
    # ---------------------------------------------------
    df_imputed = df.copy()

    for cluster_id, members in clusters.items():
        center = centroids[cluster_id]
        for i in members:
            for col in numeric_cols + cat_cols:
                if pd.isna(df_imputed.loc[i, col]) and col in center:
                    impute_val = center[col]
                    # Pastikan tipe data sesuai (hindari float untuk kolom kategorikal integer)
                    if col in cat_cols and not isinstance(impute_val, str):
                        # Konversi ke str kecuali sudah NaN
                        if pd.notna(impute_val):
                            impute_val = str(impute_val)
                    df_imputed.loc[i, col] = impute_val

    return df_imputed, clusters, centroids
