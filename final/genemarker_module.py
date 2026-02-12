"""
Author: Fatemeh Monfared
Module: Gene/Marker Analysis Module
Purpose: Utility functions for preparing, integrating, analysing, and visualising
         genetic marker data alongside potato aroma (VOC) and sensory traits.
Thesis: Data-Driven Analysis of Potato Aroma and Flavor Using TD-GC-MS and Machine Learning
Affiliation: Hanze University of Applied Sciences / HZPC 
"""


from sklearn.model_selection import ParameterGrid
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from scipy.stats import f
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm
import os
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.exceptions import ConvergenceWarning
import warnings
from sklearn.model_selection import KFold
warnings.filterwarnings("ignore", category=ConvergenceWarning)
from sklearn.metrics import r2_score
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV, KFold
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import time
import networkx as nx

def merge_genetic_aroma_sensory(genetic_data, aroma_df, sensory_pcs):
    """
    Clean, align, and merge genetic, aroma, and sensory PCA data for matching varieties.

    Parameters
    ----------
    genetic_data : pd.DataFrame
        Genetic marker table (varieties × SNPs or markers).
    aroma_df : pd.DataFrame
        Aroma compound intensity data (with a 'Variety' column).
    sensory_pcs : pd.DataFrame
        Sensory PCA scores (with a 'Variety' column).

    Returns
    -------
    merged_snp_aroma : pd.DataFrame
        Genetic + Aroma merged table.
    merged_snp_sensory : pd.DataFrame
        Genetic + Sensory merged table.
    merged_all : pd.DataFrame
        Combined Genetic + Aroma + Sensory table.
    """

    #  Helper functions 
    def to_numeric(df):
        out = df.copy()
        for c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        return out

    def norm_key(s):
        """Normalize variety names to a comparable key"""
        s = (str(s).upper().replace(".", "").replace(" ", ""))
        s = re.sub(r"\bR\b", "", s)
        s = re.sub(r"RUSSET", "", s)
        return s.split("_")[0]


    #   Genetic data processing
    mrk_col = "TAGLO_ID" if "TAGLO_ID" in genetic_data.columns else genetic_data.columns[0]

    G = (
        genetic_data.dropna(subset=[mrk_col])
        .drop_duplicates(subset=[mrk_col])
        .set_index(mrk_col)
    )

    G = to_numeric(G)
    gd = G.T
    gd.index.name = "Variety_raw"
    gd.index = gd.index.map(norm_key)

    # Collapse duplicates (mean per variety)
    gd = gd.groupby(gd.index, observed=True).mean()
    gd.columns = ["GEN__" + str(c) for c in gd.columns]
    print(f" Genetic table after reshape: {gd.shape}")


    #   Aroma data processing
    af = aroma_df.copy()
    assert "Variety" in af.columns, " aroma_df must have a 'Variety' column"

    af["__key__"] = af["Variety"].map(norm_key)
    af = af.set_index("__key__")

    aroma_cols = [c for c in af.columns if c not in ["Variety", "index"]]
    A_raw = to_numeric(af[aroma_cols])

    # Drop all-NaN columns only
    n_before = len(A_raw.columns)
    dropped_allnan = [c for c in A_raw.columns if A_raw[c].isna().all()]
    A_step1 = A_raw.drop(columns=dropped_allnan)

    # Keep all remaining columns (fill NaNs with 0)
    A = A_step1.fillna(0)
    A = A.loc[:, A.std(ddof=0) > 0]  # Remove constant columns

    print(f"\n# Aroma data cleanup summary:")
    print(f"  Total columns before: {n_before}")
    print(f"  Dropped (all NaN): {len(dropped_allnan)}")
    print(f"  Final usable aroma columns: {A.shape[1]}")
    print("  Sample aroma cols:", list(A.columns[:8]))


    #   Sensory PCA processing
    assert "Variety" in sensory_pcs.columns, "sensory_pcs must have a 'Variety' column"

    sensory_pcs["__key__"] = sensory_pcs["Variety"].map(norm_key)
    sensory_pcs = sensory_pcs.set_index("__key__")

    S = to_numeric(sensory_pcs.drop(columns=["Variety", "is_panel"], errors="ignore")).fillna(0)
    S = S.loc[:, S.std(ddof=0) > 0]
    print(f"\n# Sensory PCs: {S.shape[1]}")
    print("Sample sensory cols:", list(S.columns[:8]))

    #   Merge all tables
    gk = gd.copy()
    gk.index.name = "Variety"

    merged_snp_aroma = gk.join(A, how="inner")
    merged_snp_sensory = gk.join(S, how="inner")
    merged_all = gk.join(pd.concat([A, S], axis=1), how="inner")

    print(f"\n# Merge results:")
    print(f"  Genetic + Aroma: {merged_snp_aroma.shape}")
    print(f"  Genetic + Sensory: {merged_snp_sensory.shape}")
    print(f"  Genetic + Aroma + Sensory: {merged_all.shape}")
    return merged_snp_aroma, merged_snp_sensory, merged_all






#  PCA + Outlier Detection + Loadings
def pca_with_feature_contribution(
    data,
    target_group=["ALTUS", "FESTIEN", "AVARNA", "ROYAL","DONALD"],
    n_components=5,
):
    """Run PCA on numeric columns, report PC2-driving features, and compute mean feature
    differences between `target_group` rows and all other rows.

    Returns
    -------
    pca_df : DataFrame
        PC1/PC2 scores per sample.
    loadings : DataFrame
        Feature loadings for PCs 1..n_components.
    difference : Series
        Mean(target_group) - Mean(others) per feature.
        """
    # Select only numeric columns
    X = data.select_dtypes(include=[np.number])
    feature_names = X.columns.tolist()


    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    pca = PCA(n_components=n_components)
    pcs = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame(
        pcs[:, :2],
        columns=["PC1", "PC2"],
        index=data.index
    )


    # PCA Loadings
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f"PC{i+1}" for i in range(n_components)],
        index=feature_names
    )

    # DIFFERENTIATING COMPOUNDS
    top_PC2_positive = loadings["PC2"].sort_values(ascending=False).head(15)
    top_PC2_negative = loadings["PC2"].sort_values().head(15)

    print("\n====================================================")
    print(" COMPOUNDS HIGH IN ALTUS / FESTIN / AVARNA / ROYAL DONALD")
    print("====================================================")
    print(top_PC2_positive)

    print("\n====================================================")
    print(" COMPOUNDS LOW IN THOSE (but high in other varieties)")
    print("====================================================")
    print(top_PC2_negative)

    # -------------------------
    # Compare mean concentrations
    # -------------------------
    numeric_cols = X.columns

    group_mean = data.loc[target_group, numeric_cols].mean()
    others_mean = data.drop(target_group, axis=0)[numeric_cols].mean()

    difference = (group_mean - others_mean).sort_values(ascending=False)

    print("\n====================================================")
    print(" ACTUALLY HIGHER IN TARGET GROUP (Mean concentration difference)")
    print("====================================================")
    print(difference.head(20))

    print("\n====================================================")
    print(" ACTUALLY LOWER IN TARGET GROUP")
    print("====================================================")
    print(difference.tail(20))

    return pca_df, loadings, difference




#  PCA Plot 
def plot_pca(pca_df, data, title="PCA of Genetic and Aroma Profiles"):
    plt.figure(figsize=(7, 5))
    plt.scatter(pca_df["PC1"], pca_df["PC2"], alpha=0.7)

    # Label outliers
    for name in data.index:
        if np.abs(pca_df.loc[name, "PC1"]) > 150 or np.abs(pca_df.loc[name, "PC2"]) > 150:
            plt.text(
                pca_df.loc[name, "PC1"],
                pca_df.loc[name, "PC2"],
                name, fontsize=8, color="blue", weight="bold"
            )

    plt.axhline(0, color="gray", lw=0.7)
    plt.axvline(0, color="gray", lw=0.7)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.show()





def plot_top_aroma_loadings_horizontal(top_aromas_df, title="Top 20 Aroma Compounds Influencing PLS Component 1"):
    """
    Plot top aroma compounds influencing PLS Component 1 (horizontal bar chart).
    Matches the style shown in  example image.
    """
    plt.figure(figsize=(7, 5))
    
    # Bars
    plt.barh(
        top_aromas_df.index,
        top_aromas_df.values,
        color="#f4a261",
        edgecolor="orange",
        alpha=0.9
    )
    
    # Red dots at the end of bars
    plt.scatter(
        top_aromas_df.values,
        top_aromas_df.index,
        color="darkred",
        s=50,
        zorder=3
    )

    # Titles and labels
    plt.title(title, fontsize=13, weight="bold", pad=10)
    plt.xlabel("Absolute Loading Strength", fontsize=11)
    plt.ylabel("")
    # Grid and formatting
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()




def perform_gwas_lmm_for_traits_fast(
    merged_df: pd.DataFrame,
    snp_prefix: str = "GEN__",
    ploidy: int = 4,
    n_pcs: int = 5,
    covariate_cols=None,
    traits: list[str] | None = None,          
    trait_transform: str | None = None,       
    trait_missing: str = "mean",              
    output_path: str | None = None,
    fdr_method: str = "fdr_bh",
    maf_min: float = 0.02,
    missing_max: float = 0.2,
    ld_threshold: float = 0.9,
    max_snps: int | None = None,             
    top_n_snps: int = 500,
    delta_grid: int = 40,
    block_size: int = 50000,
    precompute_UX: bool = False,              
    kinship_snps: int | None = 20000,         
    verbose: bool = True,
    random_state: int = 0,
):
    """
    LMM GWAS (EMMAX-style):
 
    """

    t0 = time.perf_counter()

    def log(msg: str):
        if verbose:
            dt = time.perf_counter() - t0
            print(f"[{dt:8.2f}s] {msg}", flush=True)

    rng = np.random.default_rng(random_state)

    df = merged_df.copy()

    
    #  SNP columns
    log("Stage 1/9: Detect SNP columns")
    snp_cols = [c for c in df.columns if str(c).startswith(snp_prefix)]
    if not snp_cols:
        raise ValueError(f"No SNP columns found with prefix '{snp_prefix}'.")

    if max_snps is not None and len(snp_cols) > max_snps:
        snp_cols = snp_cols[:max_snps]
        log(f"  max_snps applied -> using {len(snp_cols)} SNPs")

    # Extract SNP block ONCE (no per-column assignment!)
    log("Stage 2/9: Extract SNP matrix (single shot)")

    snp_block = df.loc[:, snp_cols]

  
    all_numeric = all(np.issubdtype(dt, np.number) for dt in snp_block.dtypes)

    if all_numeric:
        X = snp_block.to_numpy(dtype=np.float32, copy=True)
    else:
      
        snp_block = snp_block.apply(pd.to_numeric, errors="coerce")
        X = snp_block.to_numpy(dtype=np.float32, copy=True)

    snp_names = list(snp_cols)
    n, m = X.shape
    log(f"  Raw SNP matrix: n={n}, m={m}")

  
    #  SNP QC: missingness filter
    log("Stage 3/9: SNP QC (missingness / impute / monomorphic / MAF)")
    miss = np.isnan(X).mean(axis=0)
    keep = miss <= float(missing_max)
    if not np.any(keep):
        raise ValueError("All SNPs removed by missingness filter.")
    X = X[:, keep]
    snp_names = [s for s, k in zip(snp_names, keep) if k]

    # impute (median) —
    col_med = np.nanmedian(X, axis=0)
   
    col_med = np.where(np.isnan(col_med), 0.0, col_med)
    nan_mask = np.isnan(X)
    if nan_mask.any():
        X[nan_mask] = np.take(col_med, np.where(nan_mask)[1])

    # drop monomorphic
    std = X.std(axis=0)
    keep = std > 0
    if not np.any(keep):
        raise ValueError("All SNPs are monomorphic after QC.")
    X = X[:, keep]
    snp_names = [s for s, k in zip(snp_names, keep) if k]

    # MAF
    p = X.mean(axis=0) / float(ploidy)
    maf = np.minimum(p, 1.0 - p)
    keep = maf >= float(maf_min)
    if not np.any(keep):
        raise ValueError("All SNPs removed by MAF filter.")
    X = X[:, keep]
    snp_names = [s for s, k in zip(snp_names, keep) if k]

    n, m = X.shape
    log(f"  After QC: n={n}, m={m} SNPs")

    #  Traits selection
    log("Stage 4/9: Detect trait columns")
    df.columns = df.columns.astype(str).str.strip() 
    snp_set = set(snp_cols)
    if traits is None:
        trait_cols = []
        for c in df.columns:
            if c in snp_set:
                continue
            if np.issubdtype(df[c].dtype, np.number):
                trait_cols.append(c)
    else:
        trait_cols = [t for t in traits if t in df.columns]

    log(f"  Traits to test: {len(trait_cols)}")


    #  Build PCs + covariates (Intercept + PCs + extras)
    log("Stage 5/9: Build PCs + covariates")

    if kinship_snps is not None and kinship_snps < m:
        idx = rng.choice(m, size=int(kinship_snps), replace=False)
        Xk = X[:, idx]
        log(f"  Using {kinship_snps} SNPs to build K/PCs (subset)")
    else:
        Xk = X

    mean = Xk.mean(axis=0, dtype=np.float64)
    std = Xk.std(axis=0, dtype=np.float64)
    std[std == 0] = 1.0
    G_std = ((Xk - mean) / std).astype(np.float64, copy=False)

    pcs_to_use = int(min(n_pcs, n - 1)) if n_pcs else 0
    if pcs_to_use > 0:
        Cov = (G_std @ G_std.T) / float(G_std.shape[1])  # (n x n)
        evals_cov, evecs_cov = np.linalg.eigh(Cov)
    
        idx_sorted = np.argsort(evals_cov)[::-1]
        evals_cov = evals_cov[idx_sorted]
        evecs_cov = evecs_cov[:, idx_sorted]
        PCs = evecs_cov[:, :pcs_to_use]
    else:
        PCs = np.zeros((n, 0), dtype=float)

    covariate_cols = covariate_cols or []
    covariate_cols = [c for c in covariate_cols if c in df.columns]

    if covariate_cols:
        C_extra = df[covariate_cols].select_dtypes(include="number").copy()
        C_extra = C_extra.fillna(C_extra.median(numeric_only=True)).to_numpy(dtype=float)
    else:
        C_extra = np.zeros((n, 0), dtype=float)

    C = np.column_stack([np.ones(n), PCs, C_extra]).astype(float)
    p_fix = C.shape[1]
    log(f"  Fixed effects cols={p_fix} (1 intercept + {pcs_to_use} PCs + {C_extra.shape[1]} extras)")

 
    #  Kinship (K) + eigendecomposition (on subset)
    log("Stage 6/9: Kinship (K) + eigendecomposition")
    K = (G_std @ G_std.T) / float(G_std.shape[1])
    K = K + np.eye(n) * 1e-6

    evals, U = np.linalg.eigh(K)
    evals = np.clip(evals, 0, None)

    UC = (U.T @ C).astype(float)

    UX = None
    if precompute_UX:
        log("  Precomputing UX = U.T @ X (this may take a bit, but speeds trait loop)")
        #  U (n x n) و X (n x m)
        UX = (U.T @ X.astype(np.float64)).astype(np.float32, copy=False)

    #  helpers
 
    def choose_delta(Uy, UC_use, evals_use, grid=40):
        deltas = np.logspace(-4, 4, int(grid))
        best_delta, best_ll = deltas[0], -np.inf

        for d in deltas:
            w = 1.0 / np.sqrt(evals_use + d)
            y_star = Uy * w
            C_star = UC_use * w[:, None]

            beta = np.linalg.lstsq(C_star, y_star, rcond=None)[0]
            r = y_star - C_star @ beta
            rss = float(r.T @ r)
            dof = max(len(y_star) - np.linalg.matrix_rank(C_star), 1)
            sigma2 = rss / dof

            ll = -0.5 * (dof * np.log(sigma2 + 1e-30) + np.sum(np.log(evals_use + d)))
            if ll > best_ll:
                best_ll, best_delta = ll, d

        return float(best_delta)

    def ld_prune_greedy(X_sub: np.ndarray, snp_names_sub: list[str], pvals: np.ndarray, thr: float):
        order = np.argsort(pvals)
        keep_idx = []
        for idx in order:
            x = X_sub[:, idx]
            ok = True
            for kept in keep_idx:
                r = np.corrcoef(x, X_sub[:, kept])[0, 1]
                if np.isnan(r):
                    continue
                if abs(r) >= thr:
                    ok = False
                    break
            if ok:
                keep_idx.append(idx)
        keep_idx = np.array(keep_idx, dtype=int)
        return [snp_names_sub[i] for i in keep_idx]


    #  GWAS per trait
    log("Stage 7/9: GWAS trait loop")
    all_out = []

    for ti, trait in enumerate(trait_cols, 1):
        t_trait0 = time.perf_counter()

        y = pd.to_numeric(df[trait], errors="coerce").to_numpy(dtype=float)

        if np.isnan(y).any():
            if trait_missing == "mean":
                mu = np.nanmean(y)
                y = np.where(np.isnan(y), mu, y)
                mask = np.ones(n, dtype=bool)
            else:
                mask = ~np.isnan(y)
        else:
            mask = np.ones(n, dtype=bool)

        n_use = int(mask.sum())
        if n_use < (p_fix + 3):
            log(f"  [{ti}/{len(trait_cols)}] {trait}: skipped (n too small: {n_use})")
            continue

        if trait_transform == "log1p":
            y = np.log1p(np.clip(y, a_min=0, a_max=None))
        elif trait_transform == "zscore":
            mu = y[mask].mean()
            sd = y[mask].std() + 1e-12
            y = (y - mu) / sd

        if mask.all():
            Uy = (U.T @ y).astype(float)
            delta = choose_delta(Uy, UC, evals, grid=delta_grid)

            w = 1.0 / np.sqrt(evals + delta)
            y_star = Uy * w
            C_star = UC * w[:, None]

            CtC_inv = np.linalg.pinv(C_star.T @ C_star)

            beta_y = CtC_inv @ (C_star.T @ y_star)
            yr = y_star - C_star @ beta_y
            yy = float(yr.T @ yr)

            df_snp = int(n - np.linalg.matrix_rank(C_star) - 1)
            if df_snp <= 1:
                log(f"  [{ti}/{len(trait_cols)}] {trait}: skipped (df too small)")
                continue

            beta_all = np.full(m, np.nan, dtype=np.float32)
            t_all    = np.full(m, np.nan, dtype=np.float32)
            p_all    = np.full(m, np.nan, dtype=np.float64)

            for start in range(0, m, int(block_size)):
                end = min(m, start + int(block_size))

                if UX is not None:
                    G_eig = UX[:, start:end].astype(np.float64, copy=False)
                else:
                    G_eig = (U.T @ X[:, start:end].astype(np.float64)).astype(np.float64, copy=False)

                G_star = G_eig * w[:, None]

                beta_G = CtC_inv @ (C_star.T @ G_star)
                Gr = G_star - C_star @ beta_G

                xx = np.sum(Gr * Gr, axis=0)
                xy = np.sum(Gr * yr[:, None], axis=0)

                valid = xx > 0
                if not np.any(valid):
                    continue

                b = np.full(end - start, np.nan, dtype=np.float64)
                b[valid] = xy[valid] / (xx[valid] + 1e-30)

                rss = np.full(end - start, np.nan, dtype=np.float64)
                rss[valid] = yy - (xy[valid] ** 2) / (xx[valid] + 1e-30)

                mse = np.clip(rss / df_snp, 1e-30, None)

                se = np.full(end - start, np.nan, dtype=np.float64)
                se[valid] = np.sqrt(mse[valid] / (xx[valid] + 1e-30))

                t_stat = b / (se + 1e-30)
                p_val = 2 * stats.t.sf(np.abs(t_stat), df=df_snp)

                beta_all[start:end] = b.astype(np.float32, copy=False)
                t_all[start:end]    = t_stat.astype(np.float32, copy=False)
                p_all[start:end]    = p_val

        else:
            # SLOW path (trait_missing="drop")
            y_use = y[mask]
            C_use = C[mask, :]

            K_use = K[np.ix_(mask, mask)]
            evals_use, U_use = np.linalg.eigh(K_use)
            evals_use = np.clip(evals_use, 0, None)

            UC_use = (U_use.T @ C_use).astype(float)
            Uy_use = (U_use.T @ y_use).astype(float)

            delta = choose_delta(Uy_use, UC_use, evals_use, grid=delta_grid)

            w = 1.0 / np.sqrt(evals_use + delta)
            y_star = Uy_use * w
            C_star = UC_use * w[:, None]

            CtC_inv = np.linalg.pinv(C_star.T @ C_star)

            beta_y = CtC_inv @ (C_star.T @ y_star)
            yr = y_star - C_star @ beta_y
            yy = float(yr.T @ yr)

            df_snp = int(len(y_star) - np.linalg.matrix_rank(C_star) - 1)
            if df_snp <= 1:
                log(f"  [{ti}/{len(trait_cols)}] {trait}: skipped (df too small)")
                continue

            X_use = X[mask, :]

            beta_all = np.full(m, np.nan, dtype=np.float32)
            t_all    = np.full(m, np.nan, dtype=np.float32)
            p_all    = np.full(m, np.nan, dtype=np.float64)

            for start in range(0, m, int(block_size)):
                end = min(m, start + int(block_size))

                G_eig = (U_use.T @ X_use[:, start:end].astype(np.float64)).astype(np.float64, copy=False)
                G_star = G_eig * w[:, None]

                beta_G = CtC_inv @ (C_star.T @ G_star)
                Gr = G_star - C_star @ beta_G

                xx = np.sum(Gr * Gr, axis=0)
                xy = np.sum(Gr * yr[:, None], axis=0)

                valid = xx > 0
                if not np.any(valid):
                    continue

                b = np.full(end - start, np.nan, dtype=np.float64)
                b[valid] = xy[valid] / (xx[valid] + 1e-30)

                rss = np.full(end - start, np.nan, dtype=np.float64)
                rss[valid] = yy - (xy[valid] ** 2) / (xx[valid] + 1e-30)

                mse = np.clip(rss / df_snp, 1e-30, None)

                se = np.full(end - start, np.nan, dtype=np.float64)
                se[valid] = np.sqrt(mse[valid] / (xx[valid] + 1e-30))

                t_stat = b / (se + 1e-30)
                p_val = 2 * stats.t.sf(np.abs(t_stat), df=df_snp)

                beta_all[start:end] = b.astype(np.float32, copy=False)
                t_all[start:end]    = t_stat.astype(np.float32, copy=False)
                p_all[start:end]    = p_val

        ok = np.isfinite(p_all)
        if ok.sum() == 0:
            log(f"  [{ti}/{len(trait_cols)}] {trait}: no valid p-values")
            continue

        q_all = np.full(m, np.nan, dtype=np.float64)
        q_all[ok] = multipletests(p_all[ok], method=fdr_method)[1]

        order = np.argsort(q_all)
        order = order[np.isfinite(q_all[order])]
        top_idx = order[:int(top_n_snps)]

        top = pd.DataFrame({
            "Trait": trait,
            "SNP": [snp_names for snp_names in (np.array(snp_names, dtype=object)[top_idx])],
            "beta": beta_all[top_idx].astype(float),
            "t": t_all[top_idx].astype(float),
            "p_value": p_all[top_idx].astype(float),
            "p_fdr": q_all[top_idx].astype(float),
            "n": int(n_use),
            "df": int(df_snp),
            "n_pcs": int(pcs_to_use),
            "delta": float(delta),
        }).sort_values("p_fdr", kind="mergesort")

    
        if len(top) > 1 and ld_threshold is not None:
            top_snps = top["SNP"].tolist()
            idx_map = {name: i for i, name in enumerate(snp_names)}
            cols_idx = [idx_map[s] for s in top_snps if s in idx_map]

            X_top = X[:, cols_idx] if mask.all() else X[mask, :][:, cols_idx]
            keep_snps = ld_prune_greedy(X_top.astype(float), top_snps, top["p_fdr"].to_numpy(), thr=float(ld_threshold))
            top = top[top["SNP"].isin(keep_snps)].copy()

        all_out.append(top)

        dt_trait = time.perf_counter() - t_trait0
        log(f"  [{ti}/{len(trait_cols)}] Trait={trait} done | n={n_use} | top={len(top)} | {dt_trait:.2f}s")


    #  finalize
    
    log("Stage 8/9: Finalize output")
    if not all_out:
        log("No GWAS results generated.")
        return pd.DataFrame()

    all_gwas_df = pd.concat(all_out, ignore_index=True)

    if output_path:
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        all_gwas_df.to_csv(output_path, index=False)
        log(f"Stage 9/9: Saved results -> {output_path}")

    return all_gwas_df





def plot_gwas_full_report(
    gwas_df=None,
    p_col="p_value",
    fdr_col="p_fdr",
    effect_col=None,
    title_prefix="GWAS Results",
    fdr_threshold=0.05,
):
    """
    Display Manhattan, QQ, and Volcano plots for GWAS results.
    """

    # Custom Manhattan Plot using file_path 
    file_path = r"C:\Users\fatemehm\OneDrive - Royal HZPC Group\documents\GitHub\internship\mixed biorep\gwas_aroma_all_traits.csv"
    all_gwas_df = pd.read_csv(file_path)

    plt.figure(figsize=(10, 5))
    plt.scatter(all_gwas_df.index, -np.log10(all_gwas_df[p_col]), c="gray", alpha=0.6, s=10)
    plt.scatter(
        all_gwas_df.index[all_gwas_df[fdr_col] < fdr_threshold],
        -np.log10(all_gwas_df.loc[all_gwas_df[fdr_col] < fdr_threshold, p_col]),
        c="red", s=15, label=f"Significant SNPs (FDR < {fdr_threshold})"
    )
    plt.axhline(-np.log10(fdr_threshold), color="blue", linestyle="--")
    plt.xlabel("SNP Index")
    plt.ylabel("-log10(p-value)")
    plt.title("Manhattan Plot: GWAS for Aroma Traits")
    plt.legend()
    plt.tight_layout()
    plt.show()

    #  QQ Plot 
    if gwas_df is None:
        gwas_df = all_gwas_df.copy()

    observed = -np.log10(np.sort(gwas_df[p_col].clip(lower=1e-300)))
    expected = -np.log10(np.linspace(1 / len(observed), 1, len(observed)))

    plt.figure(figsize=(6, 5))
    plt.scatter(expected, observed, c="black", s=10)
    plt.plot([0, max(expected)], [0, max(expected)], "r--")
    plt.xlabel("Expected -log10(p)")
    plt.ylabel("Observed -log10(p)")
    plt.title(f"{title_prefix} – QQ Plot", fontsize=13, weight="bold")
    plt.tight_layout()
    plt.show()

    #  Volcano Plot 
    df = gwas_df.copy()
    df["neg_log_p"] = -np.log10(df[p_col].clip(lower=1e-300))
    if effect_col is None or effect_col not in df.columns:
        df["effect_size"] = np.random.randn(len(df))
        effect_col = "effect_size"

    plt.figure(figsize=(7, 5))
    plt.scatter(
        df[effect_col], df["neg_log_p"],
        c=df[fdr_col] < fdr_threshold, cmap="coolwarm", alpha=0.7, s=20
    )
    plt.axhline(-np.log10(fdr_threshold), color="gray", linestyle="--")
    plt.xlabel("Effect Size")
    plt.ylabel("-log10(p-value)")
    plt.title(f"{title_prefix} – Volcano Plot", fontsize=13, weight="bold")
    plt.tight_layout()
    plt.show()







def perform_pca_with_outliers(data, n_components=3, outlier_percent=5, title="PCA on Gene–Aroma Combined Data"):
    """
    Perform PCA on numeric data, detect outliers, and visualize the first two components.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset (must contain numeric columns).
    n_components : int, default=3
        Number of PCA components to compute.
    outlier_percent : float, default=5
        Percentage (0–100) of points farthest from origin to label as outliers.
    title : str, default="PCA on Gene–Aroma Combined Data"
        Title for the PCA plot.

    Returns
    -------
    pca_df : pd.DataFrame
        DataFrame containing PCA scores.
    outliers : list
        List of outlier variety names or indices.
    pca_model : PCA
        Fitted PCA model.
    """

    # Select numeric features
    X = data.select_dtypes(include=[np.number])
    if X.empty:
        raise ValueError("No numeric columns found for PCA.")

    # Standardize data
    X_scaled = StandardScaler().fit_transform(X)

    #  Perform PCA 
    pca = PCA(n_components=n_components)
    pcs = pca.fit_transform(X_scaled)

    # Identify outliers (top X% farthest from origin in PC1–PC2 space) 
    distances = np.sqrt(pcs[:, 0]**2 + pcs[:, 1]**2)
    threshold = np.percentile(distances, 100 - outlier_percent)
    outlier_idx = np.where(distances > threshold)[0]
    outliers = data.index[outlier_idx].tolist()

    # Create PCA DataFrame 
    pca_df = pd.DataFrame(
        pcs[:, :2],
        columns=["PC1", "PC2"],
        index=data.index
    )

    plt.figure(figsize=(7, 5))
    plt.scatter(pca_df["PC1"], pca_df["PC2"], alpha=0.7)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    plt.title(title, fontsize=13, weight="bold")
    plt.axhline(0, color='gray', lw=0.8)
    plt.axvline(0, color='gray', lw=0.8)

    # Label outliers 
    for idx in outlier_idx:
        plt.text(
            pcs[idx, 0], pcs[idx, 1],
            str(data.index[idx]),
            fontsize=8, fontweight='bold', color='darkblue'
        )

    plt.tight_layout()
    plt.show()
    print(f" PCA completed with {n_components} components.")
    print(f"Explained variance (first 2 PCs): {pca.explained_variance_ratio_[:2]}")
    print(f"Top {outlier_percent}% outliers: {outliers}")
    return pca_df, outliers, pca





def analyze_key_variety_aroma_profiles(
    merged_snp_aroma,
    gwas_path,
    key_varieties=["NADINE", "ARGOS", "OSPREY"],
    fdr_threshold=0.05,
    n_pca_components=50,
    n_pls_components=2,
    top_n_compounds=10,
    save_path="key_variety_top_aroma_compounds.csv"
):
    """
    Analyze aroma compound deviations for key potato varieties
    using GWAS-selected SNPs, PCA reduction, and PLSR.

    Parameters
    ----------
    merged_snp_aroma : pd.DataFrame
        Combined SNP + aroma dataset (indexed by variety names).
    gwas_path : str
        Path to the GWAS results CSV file.
    key_varieties : list
        List of key potato varieties to analyze.
    fdr_threshold : float, default=0.05
        FDR threshold for selecting significant SNPs.
    n_pca_components : int, default=50
        Number of PCA components for SNP reduction.
    n_pls_components : int, default=2
        Number of PLS components to fit.
    top_n_compounds : int, default=10
        Number of top differentiating aroma compounds per variety.
    save_path : str, default="key_variety_top_aroma_compounds.csv"
        Path to save the summary CSV.

    Returns
    -------
    summary_df : pd.DataFrame
        Summary table of key varieties and their top aroma compounds.
    """

    #  Load GWAS and select significant SNPs
    gwas_df = pd.read_csv(gwas_path)
    top_snps = gwas_df.loc[gwas_df['p_fdr'] < fdr_threshold, 'SNP'].unique().tolist()
    print(f" Selected {len(top_snps)} SNPs after GWAS feature selection (FDR < {fdr_threshold}).")

    if len(top_snps) == 0:
        print(" No SNPs passed the threshold.")
        return None

    #  Separate SNP (X) and Aroma (Y) matrices
    snp_cols = [c for c in merged_snp_aroma.columns if c.startswith("GEN__")]
    aroma_cols = [c for c in merged_snp_aroma.columns if c not in snp_cols]

    X = merged_snp_aroma[snp_cols].fillna(0).astype(float)
    Y = merged_snp_aroma[aroma_cols].select_dtypes(include=[np.number]).fillna(0)

    # Normalize index names
    variety_names = merged_snp_aroma.index.str.upper().str.replace(" ", "")
    X.index = variety_names
    Y.index = variety_names

 
    #  Standardize and apply PCA + PLSR
    X_scaled = StandardScaler().fit_transform(X)
    Y_scaled = StandardScaler().fit_transform(Y)

    pca = PCA(n_components=n_pca_components)
    X_reduced = pca.fit_transform(X_scaled)

    pls = PLSRegression(n_components=n_pls_components)
    pls.fit(X_reduced, Y_scaled)

    print("\n PLSR model fitted successfully.")


    #  Compute deviations from mean profile for key varieties
    Y_df = pd.DataFrame(Y_scaled, index=variety_names, columns=Y.columns)
    key_varieties = [v for v in [v.upper().replace(" ", "") for v in key_varieties] if v in Y_df.index]

    print(f"\n Found {len(key_varieties)} key varieties: {key_varieties}")

    mean_profile = Y_df.mean()
    diff_profiles = Y_df.loc[key_varieties] - mean_profile
    abs_diff = diff_profiles.abs()

    #  Identify top differentiating aroma compounds
    top_differences = {
        variety: abs_diff.loc[variety].sort_values(ascending=False).head(top_n_compounds)
        for variety in key_varieties
    }

    for variety, diffs in top_differences.items():
        print(f"\n Top {top_n_compounds} differentiating aroma compounds for {variety}:")
        display(diffs)

    #  Visualization for each key variety
    for variety in key_varieties:
        plt.figure(figsize=(8, 4))
        sns.barplot(
            x=diff_profiles.loc[variety, top_differences[variety].index],
            y=top_differences[variety].index,
            palette="coolwarm",
            orient="h"
        )
        plt.title(f"{variety}: Top Aroma Compound Deviations from Mean")
        plt.xlabel("Deviation (Standardized Units)")
        plt.ylabel("Aroma Compound")
        plt.axvline(0, color='gray', lw=0.8)
        plt.tight_layout()
        plt.show()

    #  Summary Table and Save
    summary_table = []
    for variety in key_varieties:
        for compound in top_differences[variety].index:
            direction = "↑ Higher" if diff_profiles.loc[variety, compound] > 0 else "↓ Lower"
            summary_table.append({
                "Variety": variety,
                "Aroma Compound": compound,
                "Direction": direction,
                "Deviation (std units)": diff_profiles.loc[variety, compound]
            })

    summary_df = pd.DataFrame(summary_table)
    summary_df.to_csv(save_path, index=False)
    print(f"\n Results saved as: {save_path}")

    return summary_df






def plot_sensory_profiles_of_pca_outliers(results, sensory_reconstructed_df):
    """
    Automatically detects PCA outlier varieties from GWAS PCA results
    and plots their sensory profiles on a radar chart.

    Parameters
    ----------
    results : dict
        Output dictionary from `analyze_gwas_snps_pca()` containing:
        - "pca_outliers": list of outlier names from PCA

    sensory_reconstructed_df : pd.DataFrame
        DataFrame containing reconstructed sensory scores for each variety.
        Rows = varieties (indexed by name), Columns = sensory traits.

    Returns
    -------
    None
        Displays a radar chart comparing normalized sensory profiles of PCA-detected outliers.
    """

    #  Extract PCA outliers 
    pca_outliers = results.get("pca_outliers", [])

    if not pca_outliers:
        raise ValueError("No PCA outliers detected in the analysis results.")

    print(f"Detected PCA outliers for radar plot: {pca_outliers}")

    #  Extract sensory profiles of detected outliers 
    Y_outliers = sensory_reconstructed_df.loc[
        sensory_reconstructed_df.index.intersection(pca_outliers)
    ]

    if Y_outliers.empty:
        raise ValueError("Detected PCA outliers not found in sensory_reconstructed_df index.")

    #  Normalize traits to (-1, 1) for comparability 
    Y_norm = (Y_outliers - Y_outliers.min()) / (Y_outliers.max() - Y_outliers.min()) * 2 - 1

    #  Define radar structure 
    traits = Y_norm.columns.tolist()
    angles = np.linspace(0, 2 * np.pi, len(traits), endpoint=False).tolist()
    angles += angles[:1]  # close circle

    #   Plot radar chart 
    plt.figure(figsize=(9, 9))
    ax = plt.subplot(111, polar=True)

    # Assign colors automatically
    color_palette = plt.cm.tab10.colors
    color_map = {name: color_palette[i % len(color_palette)] for i, name in enumerate(Y_norm.index)}

    for variety in Y_norm.index:
        values = Y_norm.loc[variety].tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=variety, color=color_map[variety])
        ax.fill(angles, values, alpha=0.15, color=color_map[variety])

    #  Aesthetic formatting 
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(traits, fontsize=10)
    ax.set_yticklabels([])
    plt.title("Sensory Profile Comparison of PCA-Detected Outlier Varieties", fontsize=14, pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.show()






def _acat_pvalue(pvals, weights=None):
    """
    ACAT p-value combination.
    pvals: array-like of p-values (0..1)
    weights: optional weights (same length)
    """
    p = np.asarray(pvals, dtype=float)
    p = p[np.isfinite(p)]
    if p.size == 0:
        return np.nan

    # clip to avoid inf tan()
    eps = 1e-15
    p = np.clip(p, eps, 1 - eps)

    if weights is None:
        w = np.ones_like(p)
    else:
        w = np.asarray(weights, dtype=float)
        w = w[np.isfinite(w)]
        if w.size != p.size:
            # fallback: equal weights if mismatch
            w = np.ones_like(p)
    t = np.sum(w * np.tan((0.5 - p) * np.pi)) / np.sum(w)
    pacat = 0.5 - np.arctan(t) / np.pi
    return float(np.clip(pacat, 0.0, 1.0))


def perform_gwas_lmm_multitrait_acat_fast(
    merged_df: pd.DataFrame,
    traits: list,
    snp_prefix: str = "GEN__",
    ploidy: int = 4,
    n_pcs: int = 5,
    trait_transform: str = "zscore",
    trait_missing: str = "mean",
    top_n_snps: int = 500,
    ld_threshold: float = 0.9,
    kinship_snps: int = 20000,
    precompute_UX: bool = True,
    output_dir: str = None,
    output_basename: str = "gwas_multitrait_acat",
    verbose: bool = True,
):
    """
    Multi-trait GWAS:
      - runs per-trait LMM GWAS (population structure control via PCs + kinship) using perform_gwas_lmm_for_traits_fast
      - combines per-trait p-values per SNP with ACAT
      - returns (mt_top, mt_all)
    Notes:
      - ACAT is computed over the available per-trait p-values in the per-trait output.
      - Best practice (heavy) would be computing p for ALL traits for each candidate SNP; this version is the practical one.
    """
    if output_dir is None:
        output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)

    per_trait_path = os.path.join(output_dir, f"{output_basename}_per_trait.csv")
    mt_path = os.path.join(output_dir, f"{output_basename}.csv")

    if verbose:
        print(f"[multi-trait GWAS] merged_df shape: {merged_df.shape}")
        snp_cols = [c for c in merged_df.columns if c.startswith(snp_prefix)]
        print(f"[multi-trait GWAS] #SNPs: {len(snp_cols)} | #traits: {len(traits)}")
        print(f"[multi-trait GWAS] n_pcs={n_pcs} | top_n_snps={top_n_snps} | ld_threshold={ld_threshold}")
        print(f"[multi-trait GWAS] per-trait out: {per_trait_path}")
        print(f"[multi-trait GWAS] multi-trait out: {mt_path}")

    #  Per-trait GWAS (LMM + PCs + kinship)
    per_trait_df = perform_gwas_lmm_for_traits_fast(
        merged_df=merged_df,
        traits=traits,
        snp_prefix=snp_prefix,
        ploidy=ploidy,
        n_pcs=n_pcs,
        trait_transform=trait_transform,
        trait_missing=trait_missing,
        kinship_snps=kinship_snps,
        top_n_snps=top_n_snps,
        ld_threshold=ld_threshold,
        output_path=per_trait_path,
        precompute_UX=precompute_UX,
        verbose=verbose
    )

    # Expected columns: Trait, SNP, p_value, p_fdr, beta, ...
    if not isinstance(per_trait_df, pd.DataFrame) or per_trait_df.empty:
        raise ValueError("per_trait_df is empty. Per-trait GWAS produced no results.")

    needed = {"Trait", "SNP", "p_value"}
    missing_cols = needed - set(per_trait_df.columns)
    if missing_cols:
        raise ValueError(f"per_trait_df is missing columns: {missing_cols}. "
                         f"Available columns: {list(per_trait_df.columns)[:30]}")

    # ACAT per SNP over traits
    grouped = per_trait_df.groupby("SNP", sort=False)

    mt_rows = []
    for snp, g in grouped:
        pvals = g["p_value"].values
        pacat = _acat_pvalue(pvals)
        mt_rows.append({
            "SNP": snp,
            "ACAT_p": pacat,
            "n_traits_used": int(len(pvals)),
            "min_p": float(np.nanmin(pvals)),
        })

    mt_all = pd.DataFrame(mt_rows).sort_values("ACAT_p", ascending=True).reset_index(drop=True)

    # FDR on ACAT p-values
    mt_all["ACAT_fdr"] = multipletests(mt_all["ACAT_p"].values, method="fdr_bh")[1]

    #  Top results
    mt_top = mt_all.head(top_n_snps).copy()

    # Save
    mt_all.to_csv(mt_path, index=False)

    if verbose:
        print(f"[multi-trait GWAS] saved: {mt_path}")
        print(f"[multi-trait GWAS] mt_all shape: {mt_all.shape} | mt_top shape: {mt_top.shape}")
    return mt_top, mt_all


def run_gwas_multitrait_with_population_structure(
    merged_df: pd.DataFrame,
    traits: list,
    snp_prefix: str = "GEN__",
    ploidy: int = 4,
    n_pcs: int = 5,
    trait_transform: str = "zscore",
    trait_missing: str = "mean",
    top_n_snps: int = 500,
    ld_threshold: float = 0.9,
    output_dir: str = None,
    output_basename: str = "gwas_aroma_multitrait_acat",
    precompute_UX: bool = True,
    verbose: bool = True
):
    return perform_gwas_lmm_multitrait_acat_fast(
        merged_df=merged_df,
        traits=traits,
        snp_prefix=snp_prefix,
        ploidy=ploidy,
        n_pcs=n_pcs,
        trait_transform=trait_transform,
        trait_missing=trait_missing,
        top_n_snps=top_n_snps,
        ld_threshold=ld_threshold,
        output_dir=output_dir,
        output_basename=output_basename,
        precompute_UX=precompute_UX,
        verbose=verbose
    )





# Helpers
def _ensure_dir(path: str):
    if path:
        os.makedirs(path, exist_ok=True)


def _get_snp_cols(df: pd.DataFrame, snp_prefix: str):
    return [c for c in df.columns if str(c).startswith(snp_prefix)]


def _build_population_pcs(
    merged_df: pd.DataFrame,
    snp_prefix: str = "GEN__",
    ploidy: int = 4,
    n_pcs: int = 5,
    kinship_snps: int = 20000,
    random_state: int = 0,
):
    """
    PCs from genotype to control population structure.
    Returns: pcs_df with columns PC1..PCk and same index as merged_df
    """
    snp_cols = _get_snp_cols(merged_df, snp_prefix)
    if len(snp_cols) == 0 or n_pcs <= 0:
        return pd.DataFrame(index=merged_df.index)

    rng = np.random.default_rng(random_state)
    use_cols = snp_cols
    if kinship_snps is not None and kinship_snps < len(snp_cols):
        use_cols = list(rng.choice(snp_cols, size=int(kinship_snps), replace=False))

    X = merged_df[use_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    # impute per SNP
    col_med = np.nanmedian(X, axis=0)
    col_med = np.where(np.isnan(col_med), 0.0, col_med)
    nan_mask = np.isnan(X)
    if nan_mask.any():
        X[nan_mask] = np.take(col_med, np.where(nan_mask)[1])

    # standardize SNPs
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    G = (X - mean) / std

    # PCA in sample-space (n x n) fast for n=94
    Cov = (G @ G.T) / G.shape[1]
    evals, evecs = np.linalg.eigh(Cov)
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:, idx]

    k = int(min(n_pcs, merged_df.shape[0] - 1))
    pcs = evecs[:, :k]
    pcs_df = pd.DataFrame(pcs, index=merged_df.index, columns=[f"PC{i+1}" for i in range(k)])
    return pcs_df


def _load_mt_top(mt_top):
    """
    mt_top can be:
      - DataFrame with column 'SNP'
      - path to csv
      - list of SNP names
    """
    if isinstance(mt_top, pd.DataFrame):
        if "SNP" not in mt_top.columns:
            raise ValueError("mt_top DataFrame must have column 'SNP'")
        return mt_top["SNP"].astype(str).tolist()

    if isinstance(mt_top, str):
        df = pd.read_csv(mt_top)
        if "SNP" not in df.columns:
            raise ValueError("mt_top csv must have column 'SNP'")
        return df["SNP"].astype(str).tolist()

    if isinstance(mt_top, (list, tuple, np.ndarray)):
        return [str(x) for x in mt_top]

    raise TypeError("mt_top must be a DataFrame, csv path, or a list of SNPs.")


def _cv_std_for_best(search_obj):
    """Extract std of test score for the best params (if available)."""
    try:
        idx = search_obj.best_index_
        return float(search_obj.cv_results_["std_test_score"][idx])
    except Exception:
        return np.nan


def evaluate_regression(y_true, y_pred):
    """Compute key regression metrics."""
    return {
        "R2_in_sample": r2_score(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
    }


def ld_prune(X_df, threshold=0.95, max_snps=2000):
    """
    Perform LD pruning on SNP matrix (remove highly correlated SNPs).
    Keeps at most max_snps columns (for speed) and drops highly correlated SNPs.
    """
    X_df = X_df.loc[:, X_df.std() > 0]
    if X_df.shape[1] > max_snps:
        X_df = X_df.iloc[:, :max_snps]
    if X_df.shape[1] <= 1:
        return X_df

    corr = X_df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    return X_df.drop(columns=to_drop)

def _inner_cv_r2(model, X, y, n_splits=2):
   

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []

    for tr, va in cv.split(X):
        model.fit(X[tr], y[tr])
        scores.append(r2_score(y[va], model.predict(X[va])))

    return np.mean(scores)
def run_linear_models_cv_nested_light(
    gwas_file,
    snp_df,
    n_snps=300,
    prune_threshold=0.95,
    n_splits=3,
    random_state=42
):


    gwas = pd.read_csv(gwas_file)
    results = []

    outer_cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    models = {
        "OLS": (LinearRegression, {}),
        "Ridge": (Ridge, {"alpha": [0.1, 1.0, 10.0]}),
        "Lasso": (Lasso, {"alpha": [0.01, 0.05, 0.1]}),
        "ElasticNet": (ElasticNet, {
            "alpha": [0.05, 0.1],
            "l1_ratio": [0.3, 0.5, 0.7]
        })
    }

    for trait in gwas["Trait"].unique():
        if trait not in snp_df.columns:
            continue

        ranked_snps = (
            gwas[gwas["Trait"] == trait]
            .sort_values("p_fdr")["SNP"]
            .tolist()
        )

        y_all = snp_df[trait].astype(float).values

        for model_name, (ModelCls, param_grid) in models.items():
            y_true_all, y_pred_all = [], []

            for tr, te in outer_cv.split(y_all):
                X_tr = snp_df.iloc[tr][ranked_snps[:n_snps]].fillna(0)
                X_te = snp_df.iloc[te][ranked_snps[:n_snps]].fillna(0)
                y_tr, y_te = y_all[tr], y_all[te]

                X_tr = ld_prune(X_tr, threshold=prune_threshold)
                X_te = X_te[X_tr.columns]

                if X_tr.shape[1] < 20:
                    continue

                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_tr)
                X_te = scaler.transform(X_te)

                if param_grid:
                    best_model, best_score = None, -np.inf
                    for params in ParameterGrid(param_grid):
                        model = ModelCls(max_iter=200000, **params)
                        score = _inner_cv_r2(model, X_tr, y_tr)
                        if score > best_score:
                            best_score, best_model = score, model
                    best_model.fit(X_tr, y_tr)
                else:
                    best_model = ModelCls()
                    best_model.fit(X_tr, y_tr)

                y_pred = best_model.predict(X_te)

                y_true_all.extend(y_te)
                y_pred_all.extend(y_pred)

            if y_true_all:
                results.append({
                    "Trait": trait,
                    "Model": model_name,
                    "CV_R2": r2_score(y_true_all, y_pred_all),
                    "RMSE": np.sqrt(mean_squared_error(y_true_all, y_pred_all)),
                    "MAE": mean_absolute_error(y_true_all, y_pred_all)
                })

    return pd.DataFrame(results)



def run_linear_models_independent_generalisation_with_params(
    df_linear_cv,
    gwas_file,
    snp_train,
    snp_indep,
    n_snps=300,
    prune_threshold=0.95
):
    best_models = (
        df_linear_cv
        .sort_values("CV_R2", ascending=False)
        .groupby("Trait", as_index=False)
        .first()
    )

    gwas = pd.read_csv(gwas_file)

    results = []

    for _, row in best_models.iterrows():
        trait = row["Trait"]
        model_name = row["Model"]

        if trait not in snp_train.columns or trait not in snp_indep.columns:
            continue

        ranked_snps = (
            gwas[gwas["Trait"] == trait]
            .sort_values("p_fdr")["SNP"]
            .tolist()
        )

        X_tr = snp_train[ranked_snps[:n_snps]].fillna(0)
        X_te = snp_indep[ranked_snps[:n_snps]].fillna(0)

        y_tr = snp_train[trait].astype(float).values
        y_te = snp_indep[trait].astype(float).values

        X_tr = ld_prune(X_tr, threshold=prune_threshold)
        X_te = X_te[X_tr.columns]

        if X_tr.shape[1] < 20:
            continue

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

        # build model with best params (if present)
        if model_name == "OLS":
            model = LinearRegression()

        elif model_name == "Ridge":
            alpha = row.get("alpha", 1.0)
            model = Ridge(alpha=float(alpha), max_iter=200000)

        elif model_name == "Lasso":
            alpha = row.get("alpha", 0.01)
            model = Lasso(alpha=float(alpha), max_iter=200000)

        elif model_name == "ElasticNet":
            alpha = row.get("alpha", 0.1)
            l1_ratio = row.get("l1_ratio", 0.5)
            model = ElasticNet(alpha=float(alpha), l1_ratio=float(l1_ratio), max_iter=200000)

        else:
            continue

        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)

        results.append({
            "Trait": trait,
            "Model": model_name,
            "R2": r2_score(y_te, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_te, y_pred)),
            "MAE": mean_absolute_error(y_te, y_pred)
        })

    return pd.DataFrame(results)





def run_krr_models_cv_nested_light(
    gwas_file,
    snp_df,
    n_snps=300,
    prune_threshold=0.95,
    n_splits=3,
    random_state=42
):

    gwas = pd.read_csv(gwas_file)
    results = []

    outer_cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    param_grid = {
        "alpha": [0.1, 1.0],
        "gamma": [1e-3, 1e-2]
    }

    for trait in gwas["Trait"].unique():
        if trait not in snp_df.columns:
            continue

        ranked_snps = (
            gwas[gwas["Trait"] == trait]
            .sort_values("p_fdr")["SNP"]
            .tolist()
        )

        y_all = snp_df[trait].astype(float).values
        y_true_all, y_pred_all = [], []

        for tr, te in outer_cv.split(y_all):
            X_tr = snp_df.iloc[tr][ranked_snps[:n_snps]].fillna(0)
            X_te = snp_df.iloc[te][ranked_snps[:n_snps]].fillna(0)
            y_tr, y_te = y_all[tr], y_all[te]

            X_tr = ld_prune(X_tr, threshold=prune_threshold)
            X_te = X_te[X_tr.columns]

            if X_tr.shape[1] < 30:
                continue

            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_te = scaler.transform(X_te)

            best_model, best_score = None, -np.inf
            for params in ParameterGrid(param_grid):
                model = KernelRidge(kernel="rbf", **params)
                score = _inner_cv_r2(model, X_tr, y_tr)
                if score > best_score:
                    best_score, best_model = score, model

            best_model.fit(X_tr, y_tr)
            y_pred = best_model.predict(X_te)

            y_true_all.extend(y_te)
            y_pred_all.extend(y_pred)

        if y_true_all:
            results.append({
                "Trait": trait,
                "Model": "KRR_RBF",
                "CV_R2": r2_score(y_true_all, y_pred_all),
                "RMSE": np.sqrt(mean_squared_error(y_true_all, y_pred_all)),
                "MAE": mean_absolute_error(y_true_all, y_pred_all)
            })

    return pd.DataFrame(results)




def run_nonlinear_models_cv_nested_light(
    gwas_file,
    snp_df,
    n_snps=300,
    prune_threshold=0.95,
    n_splits=3,
    random_state=42
):
    gwas = pd.read_csv(gwas_file)
    results = []

    outer_cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for trait in gwas["Trait"].unique():
        if trait not in snp_df.columns:
            continue

        ranked_snps = (
            gwas[gwas["Trait"] == trait]
            .sort_values("p_fdr")["SNP"]
            .tolist()
        )

        y_all = snp_df[trait].astype(float).values

        for model_name in ["SVR", "RF", "XGB"]:
            y_true_all, y_pred_all = [], []

            for tr, te in outer_cv.split(y_all):
                X_tr = snp_df.iloc[tr][ranked_snps[:n_snps]].fillna(0)
                X_te = snp_df.iloc[te][ranked_snps[:n_snps]].fillna(0)
                y_tr, y_te = y_all[tr], y_all[te]

                X_tr = ld_prune(X_tr, threshold=prune_threshold)
                X_te = X_te[X_tr.columns]

                if X_tr.shape[1] < 30:
                    continue

                if model_name == "SVR":
                    scaler = StandardScaler()
                    X_tr_s = scaler.fit_transform(X_tr)
                    X_te_s = scaler.transform(X_te)

                    best_model, best_score = None, -np.inf
                    for params in ParameterGrid({"C": [1, 3], "epsilon": [0.05, 0.1]}):
                        m = SVR(gamma="scale", **params)
                        s = _inner_cv_r2(m, X_tr_s, y_tr)
                        if s > best_score:
                            best_score, best_model = s, m

                    best_model.fit(X_tr_s, y_tr)
                    y_pred = best_model.predict(X_te_s)

                elif model_name == "RF":
                    best_model, best_score = None, -np.inf
                    for d in [6, 8]:
                        m = RandomForestRegressor(
                            n_estimators=120,
                            max_depth=d,
                            min_samples_leaf=5,
                            random_state=42,
                            n_jobs=-1
                        )
                        s = _inner_cv_r2(m, X_tr.values, y_tr)
                        if s > best_score:
                            best_score, best_model = s, m

                    best_model.fit(X_tr, y_tr)
                    y_pred = best_model.predict(X_te)

                else:  # XGB
                    best_model, best_score = None, -np.inf
                    for d in [3, 4]:
                        m = XGBRegressor(
                            objective="reg:squarederror",
                            n_estimators=120,
                            max_depth=d,
                            learning_rate=0.05,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            random_state=42,
                            n_jobs=-1,
                            verbosity=0
                        )
                        s = _inner_cv_r2(m, X_tr.values, y_tr)
                        if s > best_score:
                            best_score, best_model = s, m

                    best_model.fit(X_tr, y_tr)
                    y_pred = best_model.predict(X_te)

                y_true_all.extend(y_te)
                y_pred_all.extend(y_pred)

            if y_true_all:
                results.append({
                    "Trait": trait,
                    "Model": model_name,
                    "CV_R2": r2_score(y_true_all, y_pred_all),
                    "RMSE": np.sqrt(mean_squared_error(y_true_all, y_pred_all)),
                    "MAE": mean_absolute_error(y_true_all, y_pred_all)
                })

    return pd.DataFrame(results)





def build_linear_predictions_from_best_models(
    best_linear: pd.DataFrame,
    gwas_file: str,
    snp_df: pd.DataFrame,
    n_snps: int = 300,
    prune_threshold: float = 0.95
):
    """
    Generate gene-based predictions for aroma traits using the
    best-performing linear model per trait (biorep1 only).

    Returns
    -------
    pd.DataFrame
        Predicted aroma values (rows = genotypes, columns = traits)
    """
    # Load GWAS
    gwas = pd.read_csv(gwas_file)

    # Output predictions
    dfP = pd.DataFrame(index=snp_df.index)

    for _, row in best_linear.iterrows():
        trait = row["Trait"]
        model_name = row["Model"]

        if trait not in snp_df.columns:
            continue

        # --- SNP prioritisation from GWAS
        ranked_snps = (
            gwas[gwas["Trait"] == trait]
            .sort_values("p_fdr")["SNP"]
            .tolist()
        )

        X = snp_df[ranked_snps[:n_snps]].fillna(0)

        # LD pruning (training only)
        X = ld_prune(X, threshold=prune_threshold)

        if X.shape[1] == 20:
            continue

        y = snp_df[trait].astype(float).values

        # Scale
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        # Select model
        if model_name == "OLS":
            model = LinearRegression()
        elif model_name == "Ridge":
            model = Ridge()
        elif model_name == "Lasso":
            model = Lasso(max_iter=200000)
        elif model_name == "ElasticNet":
            model = ElasticNet(max_iter=200000)
        else:
            continue

        # Fit + predict
        model.fit(Xs, y)
        dfP[trait] = model.predict(Xs)

    print(f"Built predictions for {dfP.shape[1]} aroma traits")
    return dfP



def correlate_gene_predictions_with_sensory_pcs(
    predictions_df: pd.DataFrame,
    merged_snp_sensory: pd.DataFrame,
    save_dir: str,
    r2_threshold: float = 0.5,
    tag: str = "linear_models"
):
    """
    Correlate genetically predicted aroma compounds (from best linear models)
    with sensory principal components (PC1–PC3) using the training dataset only.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        DataFrame containing gene-based predictions for aroma traits.
        Rows = genotypes, columns = traits.
        Must also contain a column 'CV_R2' per trait (or be pre-filtered).
    merged_snp_sensory : pd.DataFrame
        DataFrame containing sensory PCs (PC1, PC2, PC3) for the same genotypes.
    save_dir : str
        Directory to save correlation results.
    r2_threshold : float, optional
        Minimum CV R² threshold to retain predicted traits.
    tag : str, optional
        Label for output files.

    Returns
    -------
    pd.DataFrame
        Correlation matrix between predicted aroma traits and sensory PCs.
    """

    #  Select robust predicted traits
    if "CV_R2" in predictions_df.columns:
        robust_traits = predictions_df.columns[
            predictions_df.loc["CV_R2"] >= r2_threshold
        ].tolist()
        predictions = predictions_df.drop(index="CV_R2")
    else:
        robust_traits = predictions_df.columns.tolist()
        predictions = predictions_df.copy()

    print(f" Using {len(robust_traits)} genetically predicted traits for correlation")
    predictions = predictions[robust_traits]
    #  Extract sensory PCs
    required_pcs = ["PC1", "PC2", "PC3"]
    if not all(pc in merged_snp_sensory.columns for pc in required_pcs):
        raise KeyError("Sensory PC columns (PC1–PC3) not found.")

    sensory_pcs = merged_snp_sensory[required_pcs].copy()

    #  Normalize genotype indices
    def normalize_index(df):
        df.index = (
            df.index.astype(str)
            .str.strip()
            .str.upper()
            .str.replace(r"[^A-Z0-9]", "", regex=True)
        )
        return df

    predictions = normalize_index(predictions)
    sensory_pcs = normalize_index(sensory_pcs)
    #  Match samples
    common_samples = predictions.index.intersection(sensory_pcs.index)
    predictions = predictions.loc[common_samples]
    sensory_pcs = sensory_pcs.loc[common_samples]

    print(f" Matched samples for correlation: {len(common_samples)}")

    #  Compute correlation
    merged = pd.concat([predictions, sensory_pcs], axis=1)
    corr_matrix = merged.corr().loc[robust_traits, required_pcs]

    #  Save results
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(
        save_dir, f"linear_gene_predicted_sensory_correlation_{tag}.csv"
    )
    corr_matrix.to_csv(out_path)
    print(f" Correlation matrix saved to: {out_path}")
    return corr_matrix


def build_gene_compound_sensory_triangle(
    gwas_file: str,
    corr_matrix: pd.DataFrame,
    corr_threshold: float = 0.1,
    top_compounds: int = 5,
    top_snps_per_compound: int = 4000,
    network_snps_per_compound: int = 30,
    save_dir: str = None
):
    """
    Build and visualize the Gene–Compound–Sensory PC triangle network
    based on correlations between gene-predicted aroma compounds
    and sensory principal components.
    """
    # Load GWAS data
    gwas_all = pd.read_csv(gwas_file)
    if not {"Trait", "SNP", "p_fdr"}.issubset(gwas_all.columns):
        raise ValueError("GWAS file must contain columns: Trait, SNP, p_fdr")
    #  Select compounds correlated with sensory PCs
  
    corr_filtered = corr_matrix.copy()
    corr_filtered = corr_filtered.loc[
        corr_filtered.abs().max(axis=1) >= corr_threshold
    ]

    if corr_filtered.empty:
        raise ValueError(
            "No compounds passed the correlation threshold. "
            "Lower corr_threshold or check corr_matrix."
        )

    avg_corr = corr_filtered.abs().mean(axis=1)
    selected_compounds = (
        avg_corr.sort_values(ascending=False)
        .head(top_compounds)
        .index
        .tolist()
    )

    print("\nTop gene-predicted compounds associated with sensory PCs:")
    display(avg_corr.loc[selected_compounds])

    #  Build Gene–Compound–Sensory links
    triangle_links = []

    for compound in selected_compounds:
        sensory_pc = corr_matrix.loc[compound].abs().idxmax()
        corr_value = corr_matrix.loc[compound, sensory_pc]

        top_snps = (
            gwas_all[gwas_all["Trait"] == compound]
            .sort_values("p_fdr")
            .head(top_snps_per_compound)["SNP"]
            .tolist()
        )

        for snp in top_snps:
            triangle_links.append({
                "Gene (SNP)": snp,
                "Compound": compound,
                "Sensory PC": sensory_pc,
                "Compound–Sensory Corr": corr_value
            })

    triangle_df = pd.DataFrame(triangle_links)
    print(f"\nTriangle built for {len(selected_compounds)} compounds")
    display(triangle_df.head())
    # Compute SNP overlap matrix

    compound_snps = {
        c: set(triangle_df.loc[triangle_df["Compound"] == c, "Gene (SNP)"])
        for c in selected_compounds
    }

    overlap_matrix = pd.DataFrame(
        0,
        index=selected_compounds,
        columns=selected_compounds
    )

    for c1 in selected_compounds:
        for c2 in selected_compounds:
            overlap_matrix.loc[c1, c2] = len(
                compound_snps[c1].intersection(compound_snps[c2])
            )

    # Overlap heatmap (SAFE)
    if overlap_matrix.shape[0] >= 2 and overlap_matrix.values.sum() > 0:
        mask = np.triu(np.ones_like(overlap_matrix, dtype=bool))

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            overlap_matrix,
            mask=mask,
            annot=True,
            fmt="d",
            cmap="YlOrRd",
            cbar_kws={"label": "Shared SNPs"}
        )
        plt.title("SNP overlap between sensory-associated aroma compounds")
        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(
                os.path.join(save_dir, "snp_overlap_triangle_heatmap.png"),
                dpi=300,
                bbox_inches="tight"
            )
        plt.show()
    else:
        print(
            "\n[INFO] No meaningful SNP overlap detected between selected compounds. "
            "Overlap heatmap skipped."
        )

    # Network visualization (clean & interpretable)
    G = nx.Graph()

    for compound in selected_compounds:
        G.add_node(compound, layer="Compound", color="lightgreen")

    for pc in corr_matrix.columns:
        G.add_node(pc, layer="Sensory", color="salmon")

    for compound in selected_compounds:
        sensory_pc = corr_matrix.loc[compound].abs().idxmax()
        G.add_edge(compound, sensory_pc, weight=abs(corr_matrix.loc[compound, sensory_pc]))

    for compound in selected_compounds:
        snps = (
            gwas_all[gwas_all["Trait"] == compound]
            .sort_values("p_fdr")
            .head(network_snps_per_compound)["SNP"]
            .tolist()
        )

        for snp in snps:
            G.add_node(snp, layer="Gene", color="skyblue")
            G.add_edge(snp, compound)

    pos = {}
    y_gap = 1.0
    genes = [n for n, d in G.nodes(data=True) if d["layer"] == "Gene"]
    compounds = [n for n, d in G.nodes(data=True) if d["layer"] == "Compound"]
    sensory = [n for n, d in G.nodes(data=True) if d["layer"] == "Sensory"]
    for i, g in enumerate(genes): pos[g] = (0, i * y_gap)
    for i, c in enumerate(compounds): pos[c] = (1, i * y_gap)
    for i, s in enumerate(sensory): pos[s] = (2, i * y_gap)
    plt.figure(figsize=(14, 10))
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color=[G.nodes[n]["color"] for n in G.nodes],
        node_size=1200,
        font_size=9,
        edge_color="gray"
    )
    plt.title("Gene–Compound–Sensory PC Triangle Network", fontsize=15, weight="bold")
    plt.axis("off")
    if save_dir:
        plt.savefig(
            os.path.join(save_dir, "gene_compound_sensory_triangle.png"),
            dpi=300,
            bbox_inches="tight"
        )
    plt.show()
    return triangle_df, overlap_matrix


def analyze_marker_effects_for_top_correlated_traits(
    gwas_file: str,
    df_aroma: pd.DataFrame,
    corr_matrix: pd.DataFrame,
    corr_threshold: float = 0.5,
    top_snps_per_trait: int = 20,
    save_dir: str = None
):
    """
    Identify gene markers (SNPs) associated with highly predicted aroma compounds
    that show strong correlation with sensory PCs, and visualize their effects
    using linear regression coefficients.

    Parameters
    ----------
    gwas_file : str
        Path to the full GWAS results file (must have columns ['Trait', 'SNP', 'p_fdr'])
    df_aroma : pd.DataFrame
        DataFrame containing genotype and predicted aroma compound values (index = variety)
    corr_matrix : pd.DataFrame
        Correlation matrix from `correlate_gene_predictions_with_sensory_pcs`
    corr_threshold : float, optional
        Minimum absolute correlation with sensory PCs to include a trait
    top_snps_per_trait : int, optional
        Number of top SNPs (lowest p_fdr) to include for each trait
    save_dir : str, optional
        Directory to save generated plots (optional)

    Returns
    -------
    dict
        Mapping of {trait: coefficient DataFrame} for all analyzed compounds
    """
    #  Load GWAS and filter top correlated compounds
  
    gwas_all = pd.read_csv(gwas_file)
    if not {"Trait", "SNP", "p_fdr"}.issubset(gwas_all.columns):
        raise ValueError("GWAS file must contain columns: Trait, SNP, p_fdr")
    # Find traits with |correlation| ≥ threshold
    corr_abs = corr_matrix.abs()
    top_compounds = corr_abs.max(axis=1).sort_values(ascending=False)
    high_corr_traits = top_compounds[top_compounds >= corr_threshold].index.tolist()

    print(f"\n Found {len(high_corr_traits)} high-correlation compounds (|r| ≥ {corr_threshold}):")
    print("→", high_corr_traits)

    #  Identify top SNPs for each high-correlation trait
    important_genes = (
        gwas_all[gwas_all["Trait"].isin(high_corr_traits)]
        .sort_values("p_fdr")
        .groupby("Trait")["SNP"]
        .apply(lambda x: x.head(top_snps_per_trait).tolist())
    )
    # Store outputs
    all_effects = {}
    #  Compute SNP effects via Linear Regression
    for trait, top_snps in important_genes.items():
        print(f"\n Analyzing {trait} ({len(top_snps)} SNPs)...")

        # Skip traits not present in your data
        if trait not in df_aroma.columns:
            print(f" Trait {trait} not found in df_aroma — skipping.")
            continue
        # Select SNP data and target
        available_snps = [s for s in top_snps if s in df_aroma.columns]
        if not available_snps:
            print(f" No matching SNPs found in df_aroma for {trait}.")
            continue

        X = df_aroma[available_snps].fillna(0).astype(float)
        y = df_aroma[trait].astype(float).values

        # Scale features
        X_scaled = StandardScaler().fit_transform(X)

        # Fit linear regression
        model = LinearRegression()
        model.fit(X_scaled, y)

        coef_df = pd.DataFrame({
            "SNP": available_snps,
            "Effect": model.coef_,
            "Direction": ["Positive" if c > 0 else "Negative" for c in model.coef_],
        })
        coef_df["|Effect|"] = coef_df["Effect"].abs()
        coef_df = coef_df.sort_values("|Effect|", ascending=False).head(20)
        all_effects[trait] = coef_df

        #  Plot SNP effects
        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=coef_df,
            x="Effect",
            y="SNP",
            hue="Direction",
            palette={"Positive": "firebrick", "Negative": "royalblue"},
            dodge=False,
        )
        plt.axvline(0, color="black", lw=1)
        plt.title(f"SNP Effects on {trait}", fontsize=14, weight="bold")
        plt.xlabel("Effect Size (Linear Regression Coefficient)")
        plt.ylabel("Gene Marker (SNP)")
        plt.legend(title="Direction", loc="lower right")
        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(
                os.path.join(save_dir, f"snp_effects_{trait.replace(' ', '_')}.png"),
                dpi=300, bbox_inches="tight"
            )
        plt.show()
    print("\n Finished analyzing SNP effects for high-correlation traits.")
    return all_effects




def reconstruct_sensory_traits_from_pca(
    projection_path: str,
    loadings_path: str,
    save_path: str,
    n_components: int = 3
):
    """
    Reconstruct sensory trait values (per variety) from PCA projection and loadings.

    Parameters
    ----------
    projection_path : str
        Path to sensory PCA projection file (must contain columns like PC1, PC2, PC3, ...).
    loadings_path : str
        Path to sensory PCA loadings file (traits as rows, PCs as columns).
    save_path : str
        Path to save the reconstructed sensory trait CSV.
    n_components : int, optional
        Number of PCs to use for reconstruction (default = 3).
    Returns
    -------
    pd.DataFrame
        Reconstructed sensory trait values per variety.
    """
    # --- Load files
    projection = pd.read_csv(projection_path)
    loadings = pd.read_csv(loadings_path, index_col=0)

    # --- Extract the first n_components PCs
    pcs = [f"PC{i+1}" for i in range(n_components)]
    X_scores = projection[pcs].values
    load_mat = loadings[pcs].values

    # --- Reconstruct trait matrix
    sensory_reconstructed = np.dot(X_scores, load_mat.T)

    # --- Create DataFrame
    sensory_reconstructed_df = pd.DataFrame(
        sensory_reconstructed,
        index=projection["Variety"],
        columns=loadings.index
    )

    print(f" Reconstructed sensory traits using {n_components} PCs:")
    display(sensory_reconstructed_df.head())

    # --- Save reconstructed sensory traits
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    sensory_reconstructed_df.to_csv(save_path)
    print(f" Saved reconstructed sensory traits → {save_path}")
    return sensory_reconstructed_df



def run_plsr_gene_predictable_vs_sensory(
    aroma_df: pd.DataFrame,
    sensory_reconstructed_path: str,
    high_r2_traits: list,
    save_dir: str,
    n_components: int = 5
):
    """
    Perform PLSR between genetically predictable aroma compounds
    (selected based on high CV R²) and reconstructed sensory attributes.
    """
    #  Load reconstructed sensory traits
    sensory_df = pd.read_csv(sensory_reconstructed_path, index_col=0)

    # Normalize and align sample names
    sensory_df.index = sensory_df.index.astype(str).str.upper().str.replace(" ", "")
    aroma_df.index = aroma_df.index.astype(str).str.upper().str.replace(" ", "")

    common_varieties = aroma_df.index.intersection(sensory_df.index)
    print(f" Common varieties found: {len(common_varieties)}")

    if len(common_varieties) < 10:
        raise ValueError("Too few overlapping samples for reliable PLSR.")

    #  Subset high-predictive compounds
    valid_traits = [t for t in high_r2_traits if t in aroma_df.columns]

    if len(valid_traits) == 0:
        raise ValueError("No high-R² aroma compounds found in aroma_df.")

    X = aroma_df.loc[common_varieties, valid_traits].select_dtypes(include=[np.number])
    Y = sensory_df.loc[common_varieties].select_dtypes(include=[np.number])

    print(f" Using {X.shape[1]} genetically predictable compounds for PLSR")

    # Standardize
    Xs = StandardScaler().fit_transform(X)
    Ys = StandardScaler().fit_transform(Y)

    #  Fit PLS model--
    n_components = min(n_components, Xs.shape[1], Ys.shape[1])
    pls = PLSRegression(n_components=n_components)
    pls.fit(Xs, Ys)

    print(f" Fitted PLS model with {n_components} components")

    #  Correlation: PLS components vs sensory traits
    corr_df = pd.DataFrame(
        index=[f"PLS{i+1}" for i in range(n_components)],
        columns=Y.columns,
        dtype=float
    )

    for i in range(n_components):
        for col in Y.columns:
            corr_df.loc[f"PLS{i+1}", col] = np.corrcoef(
                pls.x_scores_[:, i], Ys[:, Y.columns.get_loc(col)]
            )[0, 1]

    # Heatmap
    plt.figure(figsize=(14, 6))
    sns.heatmap(corr_df, cmap="coolwarm", center=0, annot=True, fmt=".2f")
    plt.title("PLS components vs sensory attributes\n(genetically predictable aroma compounds)")
    plt.tight_layout()
    plt.show()

    #  Compound loadings
    loadings_df = pd.DataFrame(
        pls.x_loadings_,
        index=X.columns,
        columns=[f"PLS{i+1}" for i in range(n_components)]
    )
    for comp in loadings_df.columns:
        top_loadings = loadings_df[comp].abs().sort_values(ascending=False).head(15)
        plt.figure(figsize=(9, 4))
        sns.barplot(x=top_loadings.values, y=top_loadings.index)
        plt.title(f"Top contributors to {comp}")
        plt.xlabel("Absolute loading")
        plt.tight_layout()
        plt.show()


    #  Save outputs
    os.makedirs(save_dir, exist_ok=True)
    corr_df.to_csv(os.path.join(save_dir, "pls_sensory_correlations.csv"))
    loadings_df.to_csv(os.path.join(save_dir, "pls_compound_loadings.csv"))

    return {
        "corr_df": corr_df,
        "loadings_df": loadings_df,
    }



#   Function to prepare aligned & scaled data
def prepare_plsr_data(predictions, high_r2_traits, sensory_reconstructed_path):
    """
    Align and standardize genetically predictable aroma compounds (X)
    and reconstructed sensory traits (Y) for PLSR.
    """
    # Load sensory reconstructed data
    sensory_df = pd.read_csv(sensory_reconstructed_path, index_col=0)

    # Normalize sample names
    sensory_df.index = sensory_df.index.astype(str).str.upper().str.replace(" ", "")
    predictions.index = predictions.index.astype(str).str.upper().str.replace(" ", "")

    # Find common varieties
    common_varieties = predictions.index.intersection(sensory_df.index)
    print(f" Common varieties found: {len(common_varieties)}")

    if len(common_varieties) < 10:
        raise ValueError("Too few overlapping samples for PLSR.")

    # Subset to shared samples
    valid_traits = [t for t in high_r2_traits if t in predictions.columns]
    if len(valid_traits) == 0:
        raise ValueError("No high-R² traits found in predictions.")

    X = predictions.loc[common_varieties, valid_traits].select_dtypes(include=[np.number])
    Y = sensory_df.loc[common_varieties].select_dtypes(include=[np.number])

    # Standardize and keep DataFrame structure
    X_scaled = pd.DataFrame(
        StandardScaler().fit_transform(X),
        index=X.index,
        columns=X.columns
    )
    Y_scaled = pd.DataFrame(
        StandardScaler().fit_transform(Y),
        index=Y.index,
        columns=Y.columns
    )
    print(f"X_scaled shape: {X_scaled.shape}, Y_scaled shape: {Y_scaled.shape}")
    return X, Y, X_scaled, Y_scaled




def plot_plsr_biplot(
    pls_model,
    X_scaled: pd.DataFrame,
    pc1: int = 1,
    pc2: int = 2,
    top_n_loadings: int = 15,
    rotate: str = 'none',
    scale_arrows: float = 8,
    show_labels: bool = False
):
    """
    Plot a PLS biplot showing sample scores and top contributing compound loadings.
    """
    # Extract scores
    x_scores = pls_model.x_scores_[:, pc1 - 1]
    y_scores = pls_model.x_scores_[:, pc2 - 1]

    # Rotation
    if rotate == '180':
        x_scores, y_scores = -x_scores, -y_scores
    elif rotate == '90cw':
        x_scores, y_scores = y_scores, -x_scores
    elif rotate == '90ccw':
        x_scores, y_scores = -y_scores, x_scores

    # Plot samples
    plt.figure(figsize=(10, 8))
    plt.scatter(x_scores, y_scores, c='steelblue', alpha=0.6)

    if show_labels:
        for i, txt in enumerate(X_scaled.index):
            plt.text(x_scores[i], y_scores[i], txt, fontsize=7, alpha=0.6)

    # Loadings
    x_loadings = pls_model.x_loadings_[:, pc1 - 1]
    y_loadings = pls_model.x_loadings_[:, pc2 - 1]

    if rotate == '180':
        x_loadings, y_loadings = -x_loadings, -y_loadings
    elif rotate == '90cw':
        x_loadings, y_loadings = y_loadings, -x_loadings
    elif rotate == '90ccw':
        x_loadings, y_loadings = -y_loadings, x_loadings

    loadings_df = pd.DataFrame(
        {'x': x_loadings, 'y': y_loadings},
        index=X_scaled.columns
    )
    top_loadings = (
        loadings_df.abs()
        .sum(axis=1)
        .sort_values(ascending=False)
        .head(top_n_loadings)
        .index
    )
    for compound in top_loadings:
        plt.arrow(
            0, 0,
            loadings_df.loc[compound, 'x'] * scale_arrows,
            loadings_df.loc[compound, 'y'] * scale_arrows,
            color='crimson', alpha=0.5, head_width=0.03
        )
        plt.text(
            loadings_df.loc[compound, 'x'] * scale_arrows * 1.05,
            loadings_df.loc[compound, 'y'] * scale_arrows * 1.05,
            compound, color='crimson', fontsize=9
        )
    plt.axhline(0, color='grey', linestyle='--', linewidth=1)
    plt.axvline(0, color='grey', linestyle='--', linewidth=1)
    plt.xlabel(f"PLS{pc1}")
    plt.ylabel(f"PLS{pc2}")
    plt.title(f"PLS Biplot: PLS{pc1} vs PLS{pc2}")
    plt.tight_layout()
    plt.show()








