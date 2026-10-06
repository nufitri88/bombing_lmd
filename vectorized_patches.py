"""
Patch tervektorisasi (numpy) untuk empat titik O(n^2)/O(V^2*K*n) berbasis
loop Python murni yang menjadi bottleneck utama LMDBombingHybrid.fit().
Setiap fungsi di sini menghasilkan nilai numerik yang MATEMATIS IDENTIK
dengan versi asli (bukan pendekatan) -- hanya cara komputasinya yang beda
(vektorisasi numpy vs loop python). Diuji ekuivalensinya di test_equivalence.py
sebelum dipakai untuk audit Heart-Disease/NPHA.

Cara pakai: apply_patches(ModuleObject) -- menimpa method di kelas
LMDBombingHybrid milik modul tsb (module_lama atau module_revisi).
"""
import numpy as np
import pandas as pd


def _compute_distance_matrix_vec(self, X: pd.DataFrame) -> np.ndarray:
    n = len(X)
    D = np.zeros((n, n), dtype=float)
    for col in X.columns:
        t = self.attr_types[col]
        if t == "Nominal":
            vals = list(pd.unique(X[col]))
            val_to_idx = {v: i for i, v in enumerate(vals)}
            codes = X[col].map(val_to_idx).to_numpy()
            C = len(vals)
            cat_mat = np.zeros((C, C), dtype=float)
            mp = self.nominal_dist[col]
            for i, a in enumerate(vals):
                for j, b in enumerate(vals):
                    if i == j:
                        continue
                    cat_mat[i, j] = mp.get((a, b), mp.get((b, a), self.dis_const))
            D += cat_mat[codes[:, None], codes[None, :]]
        elif t == "Ordinal":
            order = list(self.ordinal_orders[col])
            L = len(order)
            adj = self.ordinal_adj_dist[col]
            step = np.array([adj.get(order[i], self.dis_const) for i in range(L - 1)], dtype=float)
            prefix = np.concatenate([[0.0], np.cumsum(step)])  # prefix[k] = jarak kumulatif ke posisi k
            order_to_idx = {v: i for i, v in enumerate(order)}
            idx_of = X[col].map(lambda v: order_to_idx.get(v, None))
            codes = idx_of.to_numpy()
            valid = np.array([c is not None for c in codes])
            codes_safe = np.array([c if c is not None else 0 for c in codes], dtype=int)
            prefx = prefix[codes_safe]
            d_ord = np.abs(prefx[:, None] - prefx[None, :])
            # a==b (indeks sama) -> 0 secara alami dari abs diff; nilai yang tidak
            # ditemukan di order (ValueError pada versi asli) diberi jarak 0, sesuai
            # perilaku except ValueError: return 0.0 pada _dist_attr asli.
            mask_invalid = (~valid[:, None]) | (~valid[None, :])
            d_ord[mask_invalid] = 0.0
            D += d_ord
        elif t == "Numeric":
            vmin, vmax = self.num_min[col], self.num_max[col]
            if vmax <= vmin:
                continue
            v = pd.to_numeric(X[col], errors="coerce").to_numpy(dtype=float)
            z = np.clip((v - vmin) / (vmax - vmin), 0.0, 1.0)
            delta = np.abs(z[:, None] - z[None, :])
            psi_vec = self.num_bin_psi[col]
            if psi_vec is None or len(psi_vec) == 0:
                D += delta
                continue
            B = self.n_bins_numeric
            bidx = np.clip((delta * B).astype(int), 0, B - 1)
            scale = np.maximum(0.1, 1.0 + psi_vec[bidx])
            D += delta * scale
    np.fill_diagonal(D, 0.0)
    return D


def _local_neighbor_density_vec(self, D_scaled: np.ndarray, r: float, valid_mask: np.ndarray) -> np.ndarray:
    counts = ((D_scaled <= r) & valid_mask[None, :]).sum(axis=1)
    rho_r = np.where(valid_mask, counts / self.k2, 0.0)
    return rho_r


def _update_nominal_ordinal_phi_vec(self, X: pd.DataFrame, labels: np.ndarray):
    eps = 1e-12
    k = self.K_auto_
    for col in X.columns:
        if self.attr_types[col] not in ("Nominal", "Ordinal"):
            continue
        vals = list(pd.unique(X[col]))
        V = len(vals)
        if V < 2:
            continue
        val_to_idx = {v: i for i, v in enumerate(vals)}
        codes = X[col].map(val_to_idx).to_numpy()

        # freq[c, v] = fraksi anggota klaster c yang bernilai vals[v]  (P(v|c))
        freq = np.zeros((k, V), dtype=float)
        for c in range(k):
            idx_c = np.where(labels == c)[0]
            size_c = len(idx_c)
            if size_c == 0:
                continue
            codes_c = codes[idx_c]
            counts = np.bincount(codes_c, minlength=V).astype(float)
            freq[c, :] = counts / (size_c + eps)

        # P[a,b] = sum_c P(a|c) * P(b|c)  -> perkalian matriks freq.T @ freq
        P_mat = freq.T @ freq  # (V, V)
        np.fill_diagonal(P_mat, -np.inf)  # a==b tidak dipakai (dilewati di versi asli)

        mask = ~np.eye(V, dtype=bool)
        P_vals = P_mat[mask]
        if P_vals.size == 0:
            continue
        P_max, P_min = P_vals.max(), P_vals.min()
        if abs(P_max - P_min) < eps:
            continue

        phi_old = self.nominal_phi[col] if self.attr_types[col] == "Nominal" else self.ordinal_phi[col]
        phi_new = {}
        for i, a in enumerate(vals):
            for j, b in enumerate(vals):
                if i == j:
                    continue
                v = P_mat[i, j]
                if v == P_max:
                    phi_raw = -P_max / self.eta_cat
                elif v == P_min:
                    phi_raw = (1 - P_min) / self.eta_cat
                else:
                    phi_raw = 0.0
                key = (a, b)
                old = phi_old.get(key, 0.0)
                phi_new[key] = float(np.clip(self.alpha_reg * old + (1 - self.alpha_reg) * phi_raw, -1, 1))

        if self.attr_types[col] == "Nominal":
            self.nominal_phi[col] = phi_new
            self.nominal_dist[col] = {kk: self.dis_const + vv for kk, vv in phi_new.items()}
        else:
            self.ordinal_phi[col] = phi_new
            self.ordinal_adj_dist[col] = {kk: self.dis_const + vv for kk, vv in phi_new.items()}


def _update_numeric_phi_vec(self, X: pd.DataFrame, labels: np.ndarray):
    eps = 1e-12
    B = self.n_bins_numeric
    k = self.K_auto_
    for col in X.columns:
        if self.attr_types[col] != "Numeric":
            continue
        if self.num_min[col] >= self.num_max[col]:
            continue
        z_full = np.clip(
            (pd.to_numeric(X[col], errors="coerce").fillna(X[col].mean()).to_numpy() - self.num_min[col])
            / (self.num_max[col] - self.num_min[col]),
            0.0, 1.0,
        )
        P_sum = np.zeros(B)

        for c in range(k):
            idx = np.where(labels == c)[0]
            m = len(idx)
            if m < 2:
                continue
            if m > self.max_pairs_per_cluster:
                idx_sample = self._rng.choice(idx, size=self.max_pairs_per_cluster, replace=False)
            else:
                idx_sample = idx
            m_s = len(idx_sample)
            if m_s < 2:
                continue

            zc = z_full[idx_sample]
            delta_mat = np.abs(zc[:, None] - zc[None, :])
            iu = np.triu_indices(m_s, k=1)
            deltas = delta_mat[iu]
            bidx = np.clip((deltas * B).astype(int), 0, B - 1)
            bin_counts = np.bincount(bidx, minlength=B).astype(float)
            total_pairs = bin_counts.sum()
            if total_pairs <= 0:
                continue
            probs = bin_counts / (total_pairs + eps)
            P_sum += probs * probs

        if np.all(P_sum <= eps):
            continue
        P_max, P_min = P_sum.max(), P_sum.min()
        if abs(P_max - P_min) < eps:
            continue

        phi_raw_vec = np.zeros(B, dtype=float)
        for bidx_ in range(B):
            val = P_sum[bidx_]
            if val == P_max:
                phi = -P_max / self.eta_num
            elif val == P_min:
                phi = (1.0 - P_min) / self.eta_num
            else:
                phi = 0.0
            phi_raw_vec[bidx_] = phi

        phi_old_vec = self.numeric_phi[col]
        if phi_old_vec.shape[0] != B:
            phi_old_vec = np.zeros(B, dtype=float)
        phi_new_vec = np.clip(self.alpha_reg * phi_old_vec + (1.0 - self.alpha_reg) * phi_raw_vec, -1.0, 1.0)
        self.numeric_phi[col] = phi_new_vec
        self.num_bin_psi[col] = phi_new_vec.copy()


def apply_patches(module):
    """Tempel method tervektorisasi ke class LMDBombingHybrid milik `module`."""
    import inspect
    cls = module.LMDBombingHybrid
    cls._compute_distance_matrix = _compute_distance_matrix_vec
    cls._local_neighbor_density = _local_neighbor_density_vec
    cls._update_nominal_ordinal_phi = _update_nominal_ordinal_phi_vec
    cls._update_numeric_phi = _update_numeric_phi_vec

    # _fit_vec memakai nama-nama global (hybrid_selection_conservative,
    # PostHocMerger, dll.) yang hidup di namespace module_lama/module_revisi,
    # bukan di vectorized_patches -- maka di-compile ulang dengan globals()
    # milik `module` supaya nama-nama itu ter-resolve dengan benar.
    src = inspect.getsource(_fit_vec)
    ns = {}
    exec(compile(src, f"<fit_vec for {module.__name__}>", "exec"), module.__dict__, ns)
    cls.fit = ns["_fit_vec"]
    return module
def _fit_vec(self, df: pd.DataFrame):
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
        
        # print("[Phase 3] Computing density-excluded core centroids...")
        # if trigger:
        #     for lbl in np.unique(labels_refined):
        #         orig_lbls = np.unique(labels_bombing[labels_refined == lbl])
        #         r_idx = []
        #         for ol in orig_lbls:
        #             r_idx.extend(self._reachable_indices.get(ol, []))
        #         self._reachable_indices[lbl] = r_idx
                
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
                D_sample = self._compute_distance_matrix(X.iloc[sample_idx].reset_index(drop=True))  # [VEKTORISASI] identik dgn loop asli
                
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
