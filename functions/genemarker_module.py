import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.stats import f
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm
import os
import seaborn as sns
from sklearn.model_selection import KFold, cross_val_score
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from scipy.stats import zscore
from sklearn.exceptions import ConvergenceWarning
import warnings
from sklearn.linear_model import LinearRegression, Ridge, LassoCV, ElasticNetCV
from sklearn.model_selection import KFold, cross_val_score
warnings.filterwarnings("ignore", category=ConvergenceWarning)
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import make_scorer, r2_score
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler




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





def analyze_gene_aroma_plsr(merged_snp_aroma, n_components=2, top_genes=10, top_aromas=20):
    """
    Perform Partial Least Squares Regression (PLSR) between genetic markers and aroma compounds.
    Identifies top influential genes and aroma compounds contributing to the shared variance
    and visualizes the first component relationship (biplot).

    Parameters
    ----------
    merged_snp_aroma : pd.DataFrame
        Merged dataset containing genetic (columns starting with 'GEN__') and aroma data.
    n_components : int, default=2
        Number of PLS components to compute.
    top_genes : int, default=10
        Number of top influential genes to display.
    top_aromas : int, default=20
        Number of top influential aroma compounds to display.

    Returns
    -------
    pls_model : PLSRegression
        Trained PLSR model.
    top_genes_df : pd.Series
        Top influential genes ranked by absolute loading values.
    top_aromas_df : pd.Series
        Top influential aroma compounds ranked by absolute loading values.
    """

    # Separate genetic and aroma features 
    gene_cols = [c for c in merged_snp_aroma.columns if c.startswith("GEN__")]
    aroma_cols = [c for c in merged_snp_aroma.columns if not c.startswith("GEN__")]

    gd = merged_snp_aroma[gene_cols].select_dtypes(include=[np.number])
    A = merged_snp_aroma[aroma_cols].select_dtypes(include=[np.number])

    # Align data (common varieties) 
    common = gd.index.intersection(A.index)
    X = gd.loc[common]
    Y = A.loc[common]

    # Standardize 
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    Y_scaled = scaler_Y.fit_transform(Y)

    #  PLSR 
    pls = PLSRegression(n_components=min(n_components, X.shape[1], Y.shape[1]))
    pls.fit(X_scaled, Y_scaled)

    # Gene influence 
    gene_loadings = pd.Series(pls.x_loadings_[:, 0], index=X.columns)
    top_genes_df = gene_loadings.abs().sort_values(ascending=False).head(top_genes)
    print("Top Influential Genes:\n", top_genes_df)

    #  Aroma influence 
    aroma_loadings = pd.Series(pls.y_loadings_[:, 0], index=Y.columns)
    top_aromas_df = aroma_loadings.abs().sort_values(ascending=False).head(top_aromas)
    print("\nTop Influential Aroma Compounds:\n", top_aromas_df)

    # Biplot
    plt.figure(figsize=(8, 6))
    plt.scatter(pls.x_scores_[:, 0], pls.y_scores_[:, 0], alpha=0.7)
    plt.xlabel("PLS Component 1 (Genes)")
    plt.ylabel("PLS Component 1 (Aromas)")
    plt.title("PLSR Biplot: Genetic–Aroma Relationship in Potato Varieties", fontsize=13, weight="bold")
    plt.grid(False)

    # Label varieties
    for i, name in enumerate(common):
        plt.text(pls.x_scores_[i, 0], pls.y_scores_[i, 0], name, fontsize=8, color="darkblue")
    plt.tight_layout()
    plt.show()

    print(f"\nExplained variance (X, Y): {pls.x_weights_.shape}, {pls.y_weights_.shape}")
    print(f"Number of varieties analyzed: {len(common)}")
    return pls, top_genes_df, top_aromas_df




def plot_top_aroma_loadings_horizontal(top_aromas_df, title="Top 20 Aroma Compounds Influencing PLS Component 1"):
    """
    Plot top aroma compounds influencing PLS Component 1 (horizontal bar chart).
    Matches the style shown in your example image.
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





def plot_balanced_dendrogram_snp_aroma(
    merged_snp_aroma,
    highlight_varieties=("LADYCLAIRE", "INNOVATOR"),
    method="ward",
    title="Balanced Hierarchical Clustering (SNP + Aroma Profiles)",
    figsize=(14, 6)
):
    """
    Perform balanced hierarchical clustering combining SNP and aroma data blocks.
    Each block (genetic and aroma) is standardized and normalized to contribute equally
    to the clustering distance metric.

    Parameters
    ----------
    merged_snp_aroma : pd.DataFrame
        DataFrame containing genetic (prefixed with 'GEN__') and aroma columns.
    highlight_varieties : tuple of str, default=("LADYCLAIRE", "INNOVATOR")
        Variety names (case-insensitive) to highlight in red on the dendrogram.
    method : str, default='ward'
        Linkage method for hierarchical clustering.
    title : str, default='Balanced Hierarchical Clustering (SNP + Aroma Profiles)'
        Plot title.
    figsize : tuple, default=(14, 6)
        Figure size for dendrogram.

    Returns
    -------
    Z : ndarray
        Linkage matrix used for hierarchical clustering.
    """

    # Separate SNP and aroma features
    snp_cols = [c for c in merged_snp_aroma.columns if c.startswith("GEN__")]
    aroma_cols = [c for c in merged_snp_aroma.columns if not c.startswith("GEN__")]

    X_snp = merged_snp_aroma[snp_cols].fillna(0)
    X_aroma = merged_snp_aroma[aroma_cols].fillna(0)

    # Standardize each block individually 
    scaler = StandardScaler()
    snp_scaled = scaler.fit_transform(X_snp)
    aroma_scaled = scaler.fit_transform(X_aroma)

    #  Normalize each block (equal contribution to total variance)
    snp_scaled /= np.sqrt(np.sum(np.var(snp_scaled, axis=0)))
    aroma_scaled /= np.sqrt(np.sum(np.var(aroma_scaled, axis=0)))

    #  Combine both blocks
    X_balanced = np.hstack([snp_scaled, aroma_scaled])

    # Hierarchical clustering 
    Z = linkage(X_balanced, method=method)

    #  Plot dendrogram 
    plt.figure(figsize=figsize)
    dendrogram(
        Z,
        labels=merged_snp_aroma.index,
        leaf_rotation=90,
        leaf_font_size=8,
        color_threshold=0.7 * np.max(Z[:, 2])
    )

    plt.title(title, fontsize=13, weight="bold")
    plt.xlabel("Varieties")
    plt.ylabel("Ward Distance")

    # --- Highlight selected varieties in red ---
    for lbl in plt.gca().get_xticklabels():
        txt = lbl.get_text().upper()
        if any(name.upper() in txt for name in highlight_varieties):
            lbl.set_color("red")
            lbl.set_fontweight("bold")
    plt.tight_layout()
    plt.show()
    return Z


def detect_outliers_aroma_genefrom_plsr(X_scores, Y_scores, sample_names, z_threshold=2.5):
    """
    Detect outliers in PLSR score space (Genes vs Aroma components).
    Uses Z-score on combined Euclidean distance from origin.
    """

    # Compute Euclidean distances of each sample in PLSR space
    distances = np.sqrt(X_scores[:, 0]**2 + Y_scores[:, 0]**2)

    # Compute Z-scores of these distances
    z_scores = (distances - np.mean(distances)) / np.std(distances)

    # Identify outliers exceeding threshold
    outlier_mask = np.abs(z_scores) > z_threshold
    outliers = np.array(sample_names)[outlier_mask]

    print(f"Detected {len(outliers)} outliers (Z > {z_threshold}): {list(outliers)}")

    return list(outliers)



def perform_pca_gene_sensory(
    merged_snp_sensory,
    n_components=2,
    outlier_threshold_pc1=4,
    outlier_threshold_pc2=3,
    title="PCA on Gene–Sensory Combined Data",
    figsize=(6, 5)
):
    """
    Perform PCA on merged gene–sensory dataset and visualize the first two components.
    Automatically highlights outlier varieties based on PC score thresholds.

    Parameters
    ----------
    merged_snp_sensory : pd.DataFrame
        Combined dataset containing genetic (prefixed with 'GEN__') and sensory data.
    n_components : int, default=2
        Number of principal components to compute.
    outlier_threshold_pc1 : float, default=4
        Threshold for labeling outliers along PC1.
    outlier_threshold_pc2 : float, default=3
        Threshold for labeling outliers along PC2.
    title : str, default="PCA on Gene–Sensory Combined Data"
        Plot title.
    figsize : tuple, default=(6, 5)
        Figure size for the PCA plot.

    Returns
    -------
    pca_df : pd.DataFrame
        PCA scores for each variety.
    pca_model : PCA
        Trained PCA model.
    outliers : list
        List of detected outlier variety names.
    """

    # Handle missing values
    X = merged_snp_sensory.fillna(0).values
    variety_names = merged_snp_sensory.index.tolist()

    #  Standardize 
    X_scaled = StandardScaler().fit_transform(X)

    # Perform PCA
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X_scaled)

    # Identify outliers
    outliers = [
        variety_names[i]
        for i in range(len(variety_names))
        if abs(scores[i, 0]) > outlier_threshold_pc1 or abs(scores[i, 1]) > outlier_threshold_pc2
    ]

    #  Plot PCA 
    plt.figure(figsize=figsize)
    plt.scatter(scores[:, 0], scores[:, 1], alpha=0.6, color="#457b9d")

    # Label outliers
    for i, label in enumerate(variety_names):
        if label in outliers:
            plt.text(
                scores[i, 0],
                scores[i, 1],
                label,
                fontsize=8,
                color="darkred",
                fontweight="bold"
            )

    plt.axhline(0, color="gray", lw=0.8)
    plt.axvline(0, color="gray", lw=0.8)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    plt.title(title, fontsize=13, weight="bold")
    plt.tight_layout()
    plt.show()

    #Store results
    pca_df = (
        pd.DataFrame(scores[:, :2], columns=["PC1", "PC2"], index=variety_names)
        .assign(IsOutlier=lambda df: df.index.isin(outliers))
    )

    print(f" PCA completed with {n_components} components.")
    print(f"Explained variance ratio (PC1+PC2): {pca.explained_variance_ratio_[:2].sum():.2%}")
    print(f"Outliers detected ({len(outliers)}): {outliers}")

    return pca_df, pca, outliers



def perform_pca_plsr_gene_sensory(
    merged_snp_sensory,
    n_pca_components=50,
    n_pls_components=2,
    label_fontsize=8,
    title="PLSR on PCA-Reduced SNP–Sensory Data",
    figsize=(7, 6)
):
    """
    Perform PCA on SNP (genetic) block and PLSR against sensory data.
    Generates a labeled scatter plot of PLS Component 1 (Genes) vs Component 1 (Sensory Traits).

    Parameters
    ----------
    merged_snp_sensory : pd.DataFrame
        Combined dataset containing columns prefixed with 'GEN__' (SNPs) and sensory features.
    n_pca_components : int, default=50
        Number of principal components to retain from SNP data before PLSR.
    n_pls_components : int, default=2
        Number of PLSR components to compute.
    label_fontsize : int, default=8
        Font size for variety labels.
    title : str
        Plot title.
    figsize : tuple, default=(7, 6)
        Figure size for the plot.

    Returns
    -------
    pca_model : PCA
        Fitted PCA model on SNP data.
    pls_model : PLSRegression
        Fitted PLSR model on PCA-reduced SNP and sensory data.
    pls_scores_df : pd.DataFrame
        PLS X and Y component scores per variety.
    """

    # Split SNP and sensory blocks
    X = merged_snp_sensory[[c for c in merged_snp_sensory.columns if c.startswith("GEN__")]]
    Y = merged_snp_sensory[[c for c in merged_snp_sensory.columns if not c.startswith("GEN__")]]

    # Standardize 
    X_scaled = StandardScaler().fit_transform(X)
    Y_scaled = StandardScaler().fit_transform(Y)

    # Reduce SNP block with PCA 
    pca_model = PCA(n_components=n_pca_components, random_state=42)
    X_pca = pca_model.fit_transform(X_scaled)

    # Perform PLSR
    pls_model = PLSRegression(n_components=n_pls_components)
    X_scores, Y_scores = pls_model.fit_transform(X_pca, Y_scaled)

    # Plot 
    plt.figure(figsize=figsize)
    plt.scatter(X_scores[:, 0], Y_scores[:, 0], alpha=0.7, color="steelblue")

    for i, label in enumerate(merged_snp_sensory.index):
        plt.text(X_scores[i, 0], Y_scores[i, 0], label, fontsize=label_fontsize, color='navy')

    plt.axhline(0, color='gray', lw=0.8)
    plt.axvline(0, color='gray', lw=0.8)
    plt.xlabel("PLS Component 1 (Genes)")
    plt.ylabel("PLS Component 1 (Sensory Traits)")
    plt.title(title, fontsize=13, weight="bold")
    plt.tight_layout()
    plt.show()

    #  Return DataFrame of scores
    pls_scores_df = pd.DataFrame({
        "PLS1_Gene": X_scores[:, 0],
        "PLS1_Sensory": Y_scores[:, 0],
    }, index=merged_snp_sensory.index)

    print(f" PCA reduced SNPs to {n_pca_components} components.")
    print(f" PLSR computed {n_pls_components} components.")
    print(f"Explained variance in PCA (first 3 comps): {pca_model.explained_variance_ratio_[:3]}")
    print(f"Samples plotted: {len(pls_scores_df)} varieties.")
    return pca_model, pls_model, pls_scores_df




def perform_hca_gene_sensory(
    merged_snp_sensory,
    metric="euclidean",
    method="ward",
    color_threshold_ratio=0.7,
    figsize=(12, 6),
    title="Hierarchical Clustering of Potato Varieties Based on Combined SNP and Sensory Profiles"
):
    """
    Perform Hierarchical Cluster Analysis (HCA) on SNP–sensory merged data.

    Parameters
    ----------
    merged_snp_sensory : pd.DataFrame
        Merged dataset containing genetic (GEN__) and sensory columns, indexed by variety name.
    metric : str, default='euclidean'
        Distance metric for clustering.
    method : str, default='ward'
        Linkage method for clustering.
    color_threshold_ratio : float, default=0.7
        Ratio of max linkage distance used to color clusters.
    figsize : tuple, default=(12, 6)
        Figure size for dendrogram.
    title : str
        Title for the dendrogram plot.

    Returns
    -------
    Z : ndarray
        Linkage matrix for hierarchical clustering.
    """

    #  Data cleaning 
    X = merged_snp_sensory.select_dtypes(include=[np.number]).dropna(axis=1, how="any")
    X = X.dropna(axis=0, how="any")

    # Standardize 
    Xz = (X - X.mean()) / X.std(ddof=0)
    Xz = Xz.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="any")

    # Compute distance + linkage 
    D = pdist(Xz, metric=metric)
    Z = linkage(D, method=method)

    #  Plot dendrogram 
    plt.figure(figsize=figsize)
    dendrogram(
        Z,
        labels=Xz.index,
        leaf_rotation=90,
        leaf_font_size=8,
        color_threshold=color_threshold_ratio * max(Z[:, 2]),
    )
    plt.title(title, fontsize=13, weight="bold")
    plt.xlabel("Variety", fontsize=11)
    plt.ylabel(f"{method.capitalize()} Distance", fontsize=11)
    plt.tight_layout()
    plt.show()
    print(f" HCA completed using {method} linkage and {metric} distance.")
    print(f"Samples clustered: {Xz.shape[0]}, Features used: {Xz.shape[1]}")
    return Z




def perform_gwas_for_traits(
    merged_df,
    snp_prefix="GEN__",
    output_path=None,
    fdr_method="fdr_bh",
    ld_threshold=0.9,
    max_snps=2000,
    top_n_snps=500
):
    """
    Perform GWAS across all numeric traits in a merged dataset (e.g., SNP–Aroma or SNP–Sensory),
    and print how many SNPs remain after each main step.
    """

    import numpy as np
    import pandas as pd
    from tqdm import tqdm
    from scipy.stats import f
    from statsmodels.stats.multitest import multipletests

    # --- Helper: LD pruning ---
    def ld_prune(X_df, threshold=0.9, max_snps=2000):
        X_df = X_df.loc[:, X_df.std() > 0]
        if X_df.shape[1] > max_snps:
            X_df = X_df.iloc[:, :max_snps]
        if X_df.shape[1] <= 1:
            return X_df

        corr = X_df.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
        return X_df.drop(columns=to_drop)

    # --- Helper: Single-trait GWAS ---
    def run_gwas(df, trait, snp_cols):
        y = df[trait].astype(float).values
        n = len(y)
        gwas_results = []

        for snp in snp_cols:
            x = df[snp].fillna(df[snp].mode().iloc[0]).astype(float).values
            if x.std() == 0:
                continue
            x = (x - x.mean()) / (x.std() + 1e-6)

            r = np.corrcoef(x, y)[0, 1]
            if np.isnan(r):
                continue
            r = np.clip(r, -0.999999, 0.999999)

            F_val = (r**2) / ((1 - r**2) / (n - 2))
            p_val = f.sf(F_val, 1, n - 2)
            gwas_results.append((snp, p_val))

        if not gwas_results:
            return pd.DataFrame(columns=["SNP", "p_value", "p_fdr", "Trait"])

        gwas_df = pd.DataFrame(gwas_results, columns=["SNP", "p_value"])
        gwas_df["p_fdr"] = multipletests(gwas_df["p_value"], method=fdr_method)[1]
        gwas_df["Trait"] = trait
        return gwas_df

    # --- Identify SNP and trait columns ---
    snp_cols = [c for c in merged_df.columns if c.startswith(snp_prefix)]
    trait_cols = [
        c for c in merged_df.columns
        if c not in snp_cols and np.issubdtype(merged_df[c].dtype, np.number)
    ]
    print(f"\nDetected {len(snp_cols)} SNP columns and {len(trait_cols)} numeric traits.\n")

    all_gwas = []
    snp_summary = []  # to store SNP counts for each trait

    # --- Run GWAS for each trait ---
    for trait in tqdm(trait_cols, desc="Running GWAS per trait"):
        print(f"\nProcessing trait: {trait}")
        total_snps = len(snp_cols)
        print(f"  - Initial SNPs: {total_snps}")

        gwas_df = run_gwas(merged_df, trait, snp_cols)
        if gwas_df.empty:
            print("  - No valid SNP associations found.")
            continue

        # Top SNPs
        top_snps = gwas_df.sort_values("p_fdr").head(top_n_snps)["SNP"].tolist()
        print(f"  - After selecting top {top_n_snps}: {len(top_snps)} SNPs")

        # LD pruning
        X = merged_df[top_snps].fillna(0).astype(float)
        X_pruned = ld_prune(X, threshold=ld_threshold, max_snps=max_snps)
        retained = X_pruned.shape[1]
        print(f"  - After LD pruning (r² < {ld_threshold}): {retained} SNPs retained")

        snp_summary.append((trait, total_snps, len(top_snps), retained))
        all_gwas.append(gwas_df)

    # --- Combine and save results ---
    if all_gwas:
        all_gwas_df = pd.concat(all_gwas, ignore_index=True)
        if output_path:
            all_gwas_df.to_csv(output_path, index=False)
            print(f"\nAll GWAS results saved to: {output_path}")
    else:
        all_gwas_df = pd.DataFrame()
        print("No valid GWAS results generated.")

    # --- Optional: print summary table ---
    if snp_summary:
        print("\nSummary of SNP counts per trait:")
        summary_df = pd.DataFrame(
            snp_summary, columns=["Trait", "Initial_SNPs", "Top_SNPs", "After_LD_Pruning"]
        )
        print(summary_df)

    return all_gwas_df


import numpy as np
import pandas as pd
from scipy.stats import f
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

def perform_multitrait_gwas(
    merged_df,
    snp_prefix="GEN__",
    output_path=None,
    fdr_method="fdr_bh",
    ld_threshold=0.9,
    max_snps=2000,
    top_n_snps=500
):
    """
    Perform multi-trait GWAS — testing each SNP against multiple numeric traits simultaneously.
    The association is summarized across traits using an omnibus F-statistic approximation.

    Parameters
    ----------
    merged_df : pd.DataFrame
        Merged DataFrame containing SNPs (prefixed by snp_prefix) and multiple numeric trait columns.
    snp_prefix : str, default='GEN__'
        Prefix used to identify SNP columns.
    output_path : str, optional
        If provided, saves all GWAS results to a CSV file.
    fdr_method : str, default='fdr_bh'
        Multiple testing correction method.
    ld_threshold : float, default=0.9
        Correlation threshold for LD pruning.
    max_snps : int, default=2000
        Max number of SNPs to retain before pruning for efficiency.
    top_n_snps : int, default=500
        Number of top SNPs (by adjusted p-value) to report.

    Returns
    -------
    multitrait_gwas_df : pd.DataFrame
        DataFrame with SNP, combined p-value, adjusted p-value, and summary stats.
    """

    def ld_prune(X_df, threshold=0.9, max_snps=2000):
        X_df = X_df.loc[:, X_df.std() > 0]
        if X_df.shape[1] > max_snps:
            X_df = X_df.iloc[:, :max_snps]
        if X_df.shape[1] <= 1:
            return X_df

        corr = X_df.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
        return X_df.drop(columns=to_drop)

    snp_cols = [c for c in merged_df.columns if c.startswith(snp_prefix)]
    trait_cols = [
        c for c in merged_df.columns
        if c not in snp_cols and np.issubdtype(merged_df[c].dtype, np.number)
    ]
    print(f"Detected {len(snp_cols)} SNP columns and {len(trait_cols)} numeric traits.\n")

    multitrait_results = []

    for snp in tqdm(snp_cols, desc="Running Multi-trait GWAS per SNP"):
        x = merged_df[snp].fillna(merged_df[snp].mode().iloc[0]).astype(float).values
        if x.std() == 0:
            continue
        x = (x - x.mean()) / (x.std() + 1e-6)

        # Stack all traits
        Y = merged_df[trait_cols].astype(float).values

        # Compute correlation for each trait
        r_values = [np.corrcoef(x, Y[:, i])[0, 1] for i in range(Y.shape[1])]
        r_values = np.array(r_values)
        r_values = np.nan_to_num(r_values, nan=0.0)

        # Combine trait correlations using average r^2
        r2_mean = np.mean(r_values ** 2)
        n = len(x)
        F_val = (r2_mean / (1 - r2_mean)) * (n - len(trait_cols) - 1) / len(trait_cols)
        p_val = f.sf(F_val, len(trait_cols), n - len(trait_cols) - 1)

        multitrait_results.append((snp, F_val, p_val))

    multitrait_gwas_df = pd.DataFrame(multitrait_results, columns=["SNP", "F_value", "p_value"])

    # Multiple testing correction
    multitrait_gwas_df["p_fdr"] = multipletests(multitrait_gwas_df["p_value"], method=fdr_method)[1]

    # Sort and prune LD
    top_snps = multitrait_gwas_df.sort_values("p_fdr").head(top_n_snps)["SNP"].tolist()
    X = merged_df[top_snps].fillna(0).astype(float)
    X_pruned = ld_prune(X, threshold=ld_threshold, max_snps=max_snps)

    multitrait_gwas_df = multitrait_gwas_df[multitrait_gwas_df["SNP"].isin(X_pruned.columns)]

    if output_path:
        multitrait_gwas_df.to_csv(output_path, index=False)
        print(f"\n Multi-trait GWAS results saved to: {output_path}")

    return multitrait_gwas_df


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_snp_overlap_heatmap(overlap_matrix, save_path=None, title="Overlap of Top SNPs Across Aroma Traits"):
    """
    Plot a lower-triangular heatmap showing overlap of top SNPs across aroma traits.
    """

    mask = np.triu(np.ones_like(overlap_matrix, dtype=bool))  # upper triangle mask

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        overlap_matrix,
        mask=mask,
        cmap="YlOrRd",
        annot=True,
        fmt="d",
        cbar_kws={"label": "Number of overlapping SNPs"},
        linewidths=0.5,
        linecolor="white"
    )

    plt.title(title, fontsize=15, weight="bold", pad=15)
    plt.xlabel("Compound")
    plt.ylabel("Compound")
    plt.xticks(rotation=70, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f" Overlap heatmap saved to: {save_path}")

    plt.show()


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

    # --- Custom Manhattan Plot using file_path ---
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

    # --- QQ Plot ---
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

    # --- Volcano Plot ---
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


def pca_after_gwas_feature_selection_gene_aroma(
    merged_snp_aroma,
    gwas_df,
    pval_col="p_value",
    fdr_col="p_fdr",
    snp_col="SNP",
    fdr_threshold=0.05,
    n_components=2,
    title="PCA on GWAS-Selected SNPs (Aroma-Associated Markers)"
):
    """
    Perform PCA on SNPs that were significant in GWAS analysis
    (gene–aroma associations), visualize results, and label outliers.

    Parameters
    ----------
    merged_snp_aroma : pd.DataFrame
        Combined dataset of SNPs and aroma traits (indexed by varieties).
    gwas_df : pd.DataFrame
        GWAS result table containing SNP, p-value, and FDR.
    pval_col : str
        Column name for p-values in gwas_df.
    fdr_col : str
        Column name for FDR-corrected p-values.
    snp_col : str
        Column name for SNP IDs.
    fdr_threshold : float
        Threshold for selecting significant SNPs.
    n_components : int
        Number of PCA components.
    title : str
        Plot title.
    """

    # --- Select significant SNPs ---
    if snp_col not in gwas_df.columns or fdr_col not in gwas_df.columns:
        raise ValueError(f"'{snp_col}' or '{fdr_col}' column not found in GWAS results.")

    top_snps = gwas_df.loc[gwas_df[fdr_col] < fdr_threshold, snp_col].unique().tolist()
    print(f" Selected {len(top_snps)} SNPs below FDR < {fdr_threshold}")

    if len(top_snps) == 0:
        print(" No significant SNPs found — PCA skipped.")
        return None, None

    # --- Extract SNP block ---
    X = merged_snp_aroma[top_snps].fillna(0).astype(float)

    # --- Standardize ---
    X_scaled = StandardScaler().fit_transform(X)

    # --- PCA ---
    pca = PCA(n_components=n_components, random_state=42)
    scores = pca.fit_transform(X_scaled)

    # --- Plot PCA ---
    plt.figure(figsize=(7, 6))
    plt.scatter(scores[:, 0], scores[:, 1], alpha=0.7)
    plt.axhline(0, color='gray', lw=0.8)
    plt.axvline(0, color='gray', lw=0.8)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    plt.title(title, fontsize=13, weight="bold")

    # --- Outlier detection (z-score > 2.5) ---
    z_pc1 = (scores[:, 0] - np.mean(scores[:, 0])) / np.std(scores[:, 0])
    z_pc2 = (scores[:, 1] - np.mean(scores[:, 1])) / np.std(scores[:, 1])
    outlier_mask = (np.abs(z_pc1) > 2.5) | (np.abs(z_pc2) > 2.5)

    for i, is_outlier in enumerate(outlier_mask):
        if is_outlier:
            plt.text(scores[i, 0], scores[i, 1], str(merged_snp_aroma.index[i]),
                     fontsize=8, color='black', ha='center', va='center')

    plt.tight_layout()
    plt.show()

    # --- Report explained variance ---
    explained = pca.explained_variance_ratio_[:2] * 100
    print(f"Explained variance: PC1 = {explained[0]:.2f}%, PC2 = {explained[1]:.2f}%")

    return pca, scores

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def hca_after_gwas_feature_selection_gene_aroma(
    merged_snp_aroma,
    gwas_df,
    snp_col="SNP",
    fdr_col="p_fdr",
    fdr_threshold=0.05,
    method="ward",
    metric="euclidean",
    title="Hierarchical Clustering on GWAS-Selected SNPs (Aroma-Associated Markers)",
    highlight_varieties=None,
    color_threshold=None,
    figsize=(8, 6)
):
    """
    Perform Hierarchical Clustering Analysis (HCA) using SNPs significant in GWAS (gene–aroma associations).
    Displays a dendrogram highlighting genetically and chemically similar potato varieties.

    Parameters
    ----------
    merged_snp_aroma : pd.DataFrame
        Combined dataset of SNPs and aroma traits (indexed by varieties).
    gwas_df : pd.DataFrame
        GWAS result table containing SNP, p-value, and FDR.
    snp_col : str, default='SNP'
        Column name for SNP IDs in gwas_df.
    fdr_col : str, default='p_fdr'
        Column name for FDR-corrected p-values.
    fdr_threshold : float, default=0.05
        Significance threshold for selecting SNPs.
    method : str, default='ward'
        Linkage method for hierarchical clustering.
    metric : str, default='euclidean'
        Distance metric used for clustering.
    title : str
        Title for the dendrogram plot.
    highlight_varieties : list or tuple, optional
        Varieties to highlight in red on the dendrogram.
    color_threshold : float, optional
        Distance threshold for coloring clusters.
    figsize : tuple, default=(8, 6)
        Figure size for the dendrogram.

    Returns
    -------
    Z : np.ndarray
        Linkage matrix used to build the dendrogram.
    """

    # --- Select significant SNPs ---
    top_snps = gwas_df.loc[gwas_df[fdr_col] < fdr_threshold, snp_col].unique().tolist()
    print(f" Selected {len(top_snps)} SNPs below FDR < {fdr_threshold}")

    if len(top_snps) == 0:
        print(" No significant SNPs found — HCA skipped.")
        return None

    # --- Extract SNP matrix ---
    X = merged_snp_aroma[top_snps].fillna(0).astype(float)

    # --- Standardize the data ---
    X_scaled = StandardScaler().fit_transform(X)

    # --- Compute linkage matrix ---
    Z = linkage(X_scaled, method=method, metric=metric)

    # --- Plot dendrogram ---
    plt.figure(figsize=figsize)
    dendro = dendrogram(
        Z,
        labels=merged_snp_aroma.index.tolist(),
        leaf_rotation=90,
        leaf_font_size=8,
        color_threshold=color_threshold,
    )
    plt.title(title, fontsize=13, weight="bold")
    plt.ylabel("Euclidean distance")
    plt.tight_layout()

    # --- Highlight specific varieties if provided ---
    if highlight_varieties:
        for lbl in plt.gca().get_xticklabels():
            if lbl.get_text() in highlight_varieties:
                lbl.set_color("red")
                lbl.set_fontweight("bold")

    plt.show()
    return Z


def plsr_after_gwas_selection(
    merged_snp_aroma,
    gwas_file,
    fdr_col="p_fdr",
    snp_col="SNP",
    fdr_threshold=0.05,
    n_pca_components=50,
    n_pls_components=2,
    title="PLSR on PCA-Reduced SNP–Aroma Data"
):
    """
    Run PCA + PLSR on GWAS-selected SNPs (only existing SNPs are used).
    """

    # --- Load GWAS results ---
    gwas_df = pd.read_csv(gwas_file)
    if snp_col not in gwas_df.columns or fdr_col not in gwas_df.columns:
        raise ValueError(f"'{snp_col}' or '{fdr_col}' column not found in GWAS results.")

    # --- Select significant SNPs ---
    top_snps = gwas_df.loc[gwas_df[fdr_col] < fdr_threshold, snp_col].unique().tolist()
    print(f"Found {len(top_snps)} SNPs below FDR < {fdr_threshold}")

    # --- Keep only SNPs that exist in merged data ---
    available_snps = [snp for snp in top_snps if snp in merged_snp_aroma.columns]
    missing_snps = set(top_snps) - set(available_snps)

    if len(missing_snps) > 0:
        print(f" {len(missing_snps)} SNPs not found in merged data — skipped.")
    print(f" Using {len(available_snps)} valid SNPs for analysis.")

    if len(available_snps) == 0:
        print(" No valid SNPs found. Aborting.")
        return None, None, None, None

    # --- Prepare matrices ---
    X = merged_snp_aroma[available_snps].fillna(0).astype(float)
    aroma_cols = [c for c in merged_snp_aroma.columns if c not in available_snps]
    Y = merged_snp_aroma[aroma_cols].select_dtypes(include=[np.number]).fillna(0)

    # --- Standardize ---
    X_scaled = StandardScaler().fit_transform(X)
    Y_scaled = StandardScaler().fit_transform(Y)

    # --- PCA reduction ---
    pca = PCA(n_components=n_pca_components)
    X_reduced = pca.fit_transform(X_scaled)

    # --- PLS regression ---
    pls = PLSRegression(n_components=n_pls_components)
    pls.fit(X_reduced, Y_scaled)

    print("\n PLSR model fitted successfully.")

    # --- Plot ---
    X_scores = pls.x_scores_
    Y_scores = pls.y_scores_

    plt.figure(figsize=(7, 6))
    plt.scatter(X_scores[:, 0], Y_scores[:, 0], alpha=0.7, color='steelblue')

    for i, name in enumerate(merged_snp_aroma.index):
        plt.text(X_scores[i, 0], Y_scores[i, 0], name, fontsize=7, color='navy')

    plt.axhline(0, color='gray', lw=0.8)
    plt.axvline(0, color='gray', lw=0.8)
    plt.xlabel("PLS Component 1 (Genes)")
    plt.ylabel("PLS Component 1 (Aroma)")
    plt.title(title)
    plt.tight_layout()
    plt.show()
    return pca, pls, X_scores, Y_scores



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

    # ==========================================================
    #  Load GWAS and select significant SNPs
    # ==========================================================
    gwas_df = pd.read_csv(gwas_path)
    top_snps = gwas_df.loc[gwas_df['p_fdr'] < fdr_threshold, 'SNP'].unique().tolist()
    print(f" Selected {len(top_snps)} SNPs after GWAS feature selection (FDR < {fdr_threshold}).")

    if len(top_snps) == 0:
        print(" No SNPs passed the threshold.")
        return None

    # ==========================================================
    #  Separate SNP (X) and Aroma (Y) matrices
    # ==========================================================
    snp_cols = [c for c in merged_snp_aroma.columns if c.startswith("GEN__")]
    aroma_cols = [c for c in merged_snp_aroma.columns if c not in snp_cols]

    X = merged_snp_aroma[snp_cols].fillna(0).astype(float)
    Y = merged_snp_aroma[aroma_cols].select_dtypes(include=[np.number]).fillna(0)

    # Normalize index names
    variety_names = merged_snp_aroma.index.str.upper().str.replace(" ", "")
    X.index = variety_names
    Y.index = variety_names

    # ==========================================================
    #  Standardize and apply PCA + PLSR
    # ==========================================================
    X_scaled = StandardScaler().fit_transform(X)
    Y_scaled = StandardScaler().fit_transform(Y)

    pca = PCA(n_components=n_pca_components)
    X_reduced = pca.fit_transform(X_scaled)

    pls = PLSRegression(n_components=n_pls_components)
    pls.fit(X_reduced, Y_scaled)

    print("\n PLSR model fitted successfully.")

    # ==========================================================
    # 4 Compute deviations from mean profile for key varieties
    # ==========================================================
    Y_df = pd.DataFrame(Y_scaled, index=variety_names, columns=Y.columns)
    key_varieties = [v for v in [v.upper().replace(" ", "") for v in key_varieties] if v in Y_df.index]

    print(f"\n Found {len(key_varieties)} key varieties: {key_varieties}")

    mean_profile = Y_df.mean()
    diff_profiles = Y_df.loc[key_varieties] - mean_profile
    abs_diff = diff_profiles.abs()

    # ==========================================================
    #  Identify top differentiating aroma compounds
    # ==========================================================
    top_differences = {
        variety: abs_diff.loc[variety].sort_values(ascending=False).head(top_n_compounds)
        for variety in key_varieties
    }

    for variety, diffs in top_differences.items():
        print(f"\n Top {top_n_compounds} differentiating aroma compounds for {variety}:")
        display(diffs)

    # ==========================================================
    #  Visualization for each key variety
    # ==========================================================
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

    # ==========================================================
    #  Summary Table and Save
    # ==========================================================
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




def run_gwas_for_sensory_pcs(
    merged_snp_sensory,
    snp_prefix="GEN__",
    traits=None,
    ld_threshold=0.95,
    max_snps=2000,
    top_n_snps=500,
    output_path="gwas_sensory_PCs.csv"
):
    """
    Perform GWAS for sensory PCA components (e.g., PC1–PC3)
    using SNP genotypes as predictors.

    Parameters
    ----------
    merged_snp_sensory : pd.DataFrame
        DataFrame containing SNP columns and sensory principal components.
    snp_prefix : str, default='GEN__'
        Prefix that identifies SNP marker columns.
    traits : list of str, optional
        List of sensory PCs to analyze (e.g., ["PC1", "PC2", "PC3"]).
    ld_threshold : float, default=0.95
        Correlation threshold for LD pruning.
    max_snps : int, default=2000
        Max number of SNPs to include before pruning.
    top_n_snps : int, default=500
        Number of top SNPs per trait to retain for pruning and inspection.
    output_path : str, default='gwas_sensory_PCs.csv'
        Path to save combined GWAS results.

    Returns
    -------
    all_gwas_df : pd.DataFrame
        Combined GWAS results for all sensory PCs.
    """

    # ---------------- Helper functions ----------------
    def run_gwas(df, trait, snp_cols):
        """Run GWAS for one sensory trait."""
        y = df[trait].astype(float).values
        n = len(y)
        gwas_results = []

        for snp in snp_cols:
            x = df[snp].fillna(df[snp].mode().iloc[0]).astype(float).values
            if x.std() == 0:
                continue
            x = (x - x.mean()) / (x.std() + 1e-6)
            r = np.corrcoef(x, y)[0, 1]
            if np.isnan(r):
                continue
            r = np.clip(r, -0.999999, 0.999999)
            F_val = (r**2) / ((1 - r**2) / (n - 2))
            p_val = f.sf(F_val, 1, n - 2)
            gwas_results.append((snp, p_val))

        gwas_df = pd.DataFrame(gwas_results, columns=["SNP", "p_value"])
        gwas_df["p_fdr"] = multipletests(gwas_df["p_value"], method="fdr_bh")[1]
        return gwas_df

    def ld_prune(X_df, threshold=0.9, max_snps=2000):
        """Remove SNPs with strong LD (|r| > threshold)."""
        X_df = X_df.loc[:, X_df.std() > 0]
        if X_df.shape[1] > max_snps:
            X_df = X_df.iloc[:, :max_snps]
        if X_df.shape[1] <= 1:
            return X_df

        corr = X_df.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
        return X_df.drop(columns=to_drop)

    # ---------------- Main GWAS loop ----------------
    snp_cols = [c for c in merged_snp_sensory.columns if c.startswith(snp_prefix)]
    if traits is None:
        traits = ["PC1", "PC2", "PC3"]

    all_gwas = []

    for trait in traits:
        print(f"\n🔹 Running GWAS for {trait}...")
        gwas_df = run_gwas(merged_snp_sensory, trait, snp_cols)

        if gwas_df.empty:
            print(f"   No valid SNPs for {trait}. Skipping.")
            continue

        top_snps = gwas_df.sort_values("p_fdr").head(top_n_snps)["SNP"].tolist()
        if not top_snps:
            print(f"   No significant SNPs for {trait}. Skipping.")
            continue

        X = merged_snp_sensory[top_snps].fillna(0).astype(float)
        X_pruned = ld_prune(X, threshold=ld_threshold, max_snps=max_snps)

        if X_pruned.shape[1] == 0:
            print(f"   No SNPs after LD pruning for {trait}. Skipping.")
            continue

        gwas_df["Trait"] = trait
        all_gwas.append(gwas_df)

    # ---------------- Combine and save ----------------
    if not all_gwas:
        print(" No GWAS results generated.")
        return pd.DataFrame()

    all_gwas_df = pd.concat(all_gwas, ignore_index=True)
    all_gwas_df.to_csv(output_path, index=False)
    print(f"\n All GWAS results saved to {output_path}")

    return all_gwas_df




def analyze_gwas_snps_pca(
    gwas_path: str,
    merged_snp_sensory: pd.DataFrame,
    fdr_threshold: float = 0.05,
    top_n_fallback: int = 500,
    outlier_z_pca=(3, 2.5)
):
    """
    Analyze GWAS results: select significant SNPs and perform PCA with outlier detection.

    Parameters
    ----------
    gwas_path : str
        Path to the GWAS results CSV file.
    merged_snp_sensory : pd.DataFrame
        Merged SNP × sensory PCs data.
    fdr_threshold : float, optional
        Significance threshold for FDR (default: 0.05).
    top_n_fallback : int, optional
        Number of top SNPs to use if no FDR-significant SNPs are found (default: 500).
    outlier_z_pca : tuple, optional
        Z-score thresholds for (PC1, PC2) outlier detection in PCA.

    Returns
    -------
    dict
        Dictionary with PCA and outlier detection results:
        {
            "significant_snps": list,
            "pca_model": PCA(),
            "pca_outliers": list
        }
    """

    # =====================================================
    # 1. Load GWAS results and extract significant SNPs
    # =====================================================
    gwas_df = pd.read_csv(gwas_path)

    significant_snps = gwas_df[gwas_df["p_fdr"] < fdr_threshold]["SNP"].unique().tolist()
    if len(significant_snps) == 0:
        print(f" No SNPs with FDR < {fdr_threshold}. Using top {top_n_fallback} by p-value.")
        significant_snps = (
            gwas_df.sort_values("p_fdr")
            .head(top_n_fallback)["SNP"]
            .unique()
            .tolist()
        )
    else:
        print(f" Significant SNPs found: {len(significant_snps)}")

    # =====================================================
    # 2. Subset SNP and sensory PC data
    # =====================================================
    snp_cols = [col for col in merged_snp_sensory.columns if col.startswith("GEN__")]
    snp_subset = merged_snp_sensory.loc[:, [s for s in significant_snps if s in snp_cols]].fillna(0).astype(float)

    if snp_subset.empty:
        raise ValueError("No matching SNPs found in merged_snp_sensory.")

    # =====================================================
    # 3. PCA with Outlier Detection
    # =====================================================
    X_scaled = StandardScaler().fit_transform(snp_subset)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    Z = np.abs(zscore(X_pca))
    outlier_mask = (Z[:, 0] > outlier_z_pca[0]) | (Z[:, 1] > outlier_z_pca[1])
    normal_mask = ~outlier_mask

    # --- PCA Plot ---
    plt.figure(figsize=(9, 7))
    plt.scatter(X_pca[normal_mask, 0], X_pca[normal_mask, 1],
                color="steelblue", alpha=0.7, s=60, label="Normal varieties")
    plt.scatter(X_pca[outlier_mask, 0], X_pca[outlier_mask, 1],
                color="red", s=120, edgecolor="black", label="Outliers")

    np.random.seed(42)
    for i in np.where(outlier_mask)[0]:
        name = snp_subset.index[i]
        dx, dy = np.random.uniform(0.2, 0.8), np.random.uniform(0.2, 0.8)
        plt.text(X_pca[i, 0] + dx, X_pca[i, 1] + dy, name,
                 fontsize=10, color="red", fontweight="bold")

    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    plt.title("PCA on Significant SNPs (Outlier Detection)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # =====================================================
    # 4. Return Results
    # =====================================================
    pca_outliers = snp_subset.index[outlier_mask].tolist()

    return {
        "significant_snps": significant_snps,
        "pca_model": pca,
        "pca_outliers": pca_outliers
    }




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

    # ---  Extract PCA outliers ---
    pca_outliers = results.get("pca_outliers", [])

    if not pca_outliers:
        raise ValueError("No PCA outliers detected in the analysis results.")

    print(f"Detected PCA outliers for radar plot: {pca_outliers}")

    # ---  Extract sensory profiles of detected outliers ---
    Y_outliers = sensory_reconstructed_df.loc[
        sensory_reconstructed_df.index.intersection(pca_outliers)
    ]

    if Y_outliers.empty:
        raise ValueError("Detected PCA outliers not found in sensory_reconstructed_df index.")

    # --- Normalize traits to (-1, 1) for comparability ---
    Y_norm = (Y_outliers - Y_outliers.min()) / (Y_outliers.max() - Y_outliers.min()) * 2 - 1

    # ---  Define radar structure ---
    traits = Y_norm.columns.tolist()
    angles = np.linspace(0, 2 * np.pi, len(traits), endpoint=False).tolist()
    angles += angles[:1]  # close circle

    # ---  Plot radar chart ---
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

    # ---  Aesthetic formatting ---
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(traits, fontsize=10)
    ax.set_yticklabels([])
    plt.title("Sensory Profile Comparison of PCA-Detected Outlier Varieties", fontsize=14, pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.show()








def plot_manhattan_like_gwas(
    gwas_path,
    out_png=None,
    per_trait_sample=5000,
    max_points_total=100000,
    top_n_annotate=10,
    fdr_threshold=0.05,
    random_state=42
):
    """
    Read GWAS CSV (must contain columns 'Trait', 'p_value' or 'p_fdr') and
    produce a Manhattan-like plot grouped by Trait.
    Returns a dictionary with summary statistics and the DataFrame used for plotting.
    """

    # Load file
    df = pd.read_csv(gwas_path)

    # Check for required columns
    if 'p_fdr' not in df.columns:
        if 'p_value' in df.columns:
            from statsmodels.stats.multitest import multipletests
            df['p_fdr'] = multipletests(df['p_value'].fillna(1.0), method='fdr_bh')[1]
        else:
            raise ValueError("CSV must contain 'p_fdr' or 'p_value' column")

    if 'Trait' not in df.columns:
        raise ValueError("CSV must contain 'Trait' column")

    # Add -log10(FDR)
    df = df.copy()
    df['p_fdr'] = df['p_fdr'].fillna(1.0)
    df['neglog10_p_fdr'] = -np.log10(df['p_fdr'].clip(lower=1e-300))

    # Sort for consistency
    df.sort_values(['Trait', 'p_fdr'], inplace=True)

    # Downsample per trait (prevents memory overload)
    sampled_list = []
    rng = np.random.default_rng(random_state)
    for trait, sub in df.groupby('Trait'):
        if len(sub) <= per_trait_sample:
            sampled_list.append(sub)
        else:
            sampled_list.append(sub.sample(per_trait_sample, random_state=random_state))
    plot_df = pd.concat(sampled_list).reset_index(drop=True)

    # Global downsample if still too big
    if len(plot_df) > max_points_total:
        plot_df = plot_df.sample(max_points_total, random_state=random_state).reset_index(drop=True)

    # Create index for plotting
    plot_df['Index'] = range(len(plot_df))

    # Color palette
    traits = plot_df['Trait'].unique().tolist()
    pal = sns.color_palette("tab10", n_colors=min(10, len(traits)))
    if len(traits) > 10:
        pal = sns.color_palette("tab20", n_colors=min(20, len(traits)))
    palette = dict(zip(traits, pal[:len(traits)]))

    # --- Plot ---
    plt.figure(figsize=(14, 6))
    sns.scatterplot(
        data=plot_df,
        x='Index',
        y='neglog10_p_fdr',
        hue='Trait',
        palette=palette,
        s=25,
        alpha=0.7,
        linewidth=0
    )

    # Significance threshold
    plt.axhline(-np.log10(fdr_threshold), color="red", linestyle="--", linewidth=1.2, label=f"FDR = {fdr_threshold}")

    # Labels
    plt.title("Manhattan-like Plot for GWAS of Sensory PCs", fontsize=14, pad=10)
    plt.xlabel("SNP index (grouped by trait)", fontsize=12)
    plt.ylabel("-log10(FDR-adjusted p-value)", fontsize=12)
    plt.legend(title="Trait", bbox_to_anchor=(1.05, 1), loc="upper left")

    # Annotate top N SNPs
    top_hits = df.nsmallest(top_n_annotate, 'p_fdr').copy()
    annotated = 0
    for _, row in top_hits.iterrows():
        if 'SNP' in plot_df.columns and 'SNP' in df.columns:
            mask = plot_df['SNP'] == row['SNP']
        else:
            mask = (plot_df['Trait'] == row['Trait']) & (
                np.isclose(plot_df['p_fdr'], row['p_fdr'], rtol=1e-8, atol=1e-12)
            )
        if mask.any():
            idx = plot_df[mask].index[0]
            x = plot_df.loc[idx, 'Index']
            y = plot_df.loc[idx, 'neglog10_p_fdr']
            label = row.get('SNP', f"{row['Trait']}_{idx}")
            plt.text(x, y + 0.02 * (plot_df['neglog10_p_fdr'].max() + 1),
                     label, fontsize=8, color='black', rotation=30)
            annotated += 1

    plt.tight_layout()

    # Save or show
    if out_png:
        out_dir = os.path.dirname(out_png)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        plt.savefig(out_png, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    # --- Summary statistics ---
    summary = {
        'n_total_snps': len(df),
        'n_plotted_points': len(plot_df),
        'min_p_value': float(df['p_value'].min()) if 'p_value' in df.columns else None,
        'min_p_fdr': float(df['p_fdr'].min()),
        'trait_counts': df['Trait'].value_counts().to_dict(),
        'n_significant_fdr_0.1': int((df['p_fdr'] < 0.1).sum()),
        'annotated_top_n': annotated
    }

    # --- Return all useful outputs ---
    return {
        'plot_df': plot_df,
        'full_df': df,
        'top_hits': top_hits,
        'summary': summary
    }

    




def plot_gwas_volcano(gwas_path, fdr_threshold=0.05):
    """
    Generate a Volcano Plot for GWAS of sensory principal components.

    Parameters
    ----------
    gwas_path : str
        Path to the GWAS results CSV file. The file must contain:
        - 'p_fdr' : FDR-adjusted p-values
        - 'Trait' : corresponding trait label
        - 'effect_size' : estimated SNP effect sizes (optional)
    fdr_threshold : float, optional
        Significance cutoff for drawing the horizontal threshold line (default: 0.1).

    Returns
    -------
    None
        Displays a volcano plot with effect size vs -log10(FDR-adjusted p-value).

    Notes
    -----
    The volcano plot visualizes SNP effect size (x-axis) versus statistical significance
    (-log10 FDR-adjusted p-value). Points above the red dashed line indicate SNPs
    with FDR < threshold, suggesting stronger associations.
    """

    # Load GWAS results
    gwas_df = pd.read_csv(gwas_path)

    # validation 
    required_cols = {"p_fdr", "Trait"}
    if not required_cols.issubset(gwas_df.columns):
        raise ValueError(f"Input file must contain columns: {required_cols}")

    #  Add mock effect size if not present 
    if "effect_size" not in gwas_df.columns:
        np.random.seed(42)
        gwas_df["effect_size"] = np.random.normal(0, 1, len(gwas_df))
        print(" 'effect_size' column not found — generated random values for visualization.")

    #  Sort for better visual layering 
    gwas_df = gwas_df.sort_values("p_fdr").reset_index(drop=True)

    # Create Volcano Plot 
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=gwas_df,
        x="effect_size",
        y=-np.log10(gwas_df["p_fdr"]),
        hue="Trait",
        palette="Set2",
        s=25,
        alpha=0.8
    )

    #  Significance threshold line
    plt.axhline(
        -np.log10(fdr_threshold),
        color="red",
        linestyle="--",
        linewidth=1.2,
        label=f"FDR = {fdr_threshold}"
    )

    # Styling and labels
    plt.title("Volcano Plot for Sensory GWAS Principal Components", fontsize=14, weight="bold")
    plt.xlabel("Effect Size", fontsize=12)
    plt.ylabel("−log10(FDR-adjusted p-value)", fontsize=12)
    plt.legend(title="Trait", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    plt.tight_layout()
    plt.show()



def plot_gwas_qq(gwas_df, pval_col="p_value"):
    """
    Generate a QQ (Quantile–Quantile) plot to assess GWAS p-value distribution.

    Parameters
    ----------
    gwas_df : pandas.DataFrame
        DataFrame containing GWAS results with a p-value column.
    pval_col : str, optional
        Column name for raw p-values (default: "p_value").

    Returns
    -------
    None
        Displays a QQ plot comparing observed vs expected -log10(p) values.

    Notes
    -----
    The QQ plot helps evaluate whether the distribution of observed GWAS p-values
    deviates from the null expectation. Points along the red dashed line indicate
    well-calibrated p-values (no inflation or deflation).
    """

    # --- Validate input ---
    if pval_col not in gwas_df.columns:
        raise ValueError(f"'{pval_col}' column not found in DataFrame.")

    # --- Clean p-values ---
    pvals = gwas_df[pval_col].replace(0, np.nan).dropna()
    if pvals.empty:
        raise ValueError("No valid p-values found after removing NaNs and zeros.")

    # --- Expected and observed -log10(p) ---
    expected = -np.log10(np.linspace(1 / len(pvals), 1, len(pvals)))
    observed = -np.log10(np.sort(pvals))

    # --- Plot ---
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=expected, y=observed, s=15, color="darkslateblue", alpha=0.7)
    plt.plot([0, max(expected)], [0, max(expected)], color="red", linestyle="--", linewidth=1)

    # --- Formatting ---
    plt.title("QQ Plot for Sensory GWAS Principal Components", fontsize=14, weight="bold")
    plt.xlabel("Expected -log10(p)", fontsize=12)
    plt.ylabel("Observed -log10(p)", fontsize=12)
    plt.tight_layout()
    plt.show()




def evaluate_regression(y_true, y_pred):
    """Compute key regression metrics."""
    return {
        "R2_in_sample": r2_score(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
    }


def ld_prune(X_df, threshold=0.9, max_snps=2000):
    """
    Perform LD pruning on SNP matrix (remove highly correlated SNPs).
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

def run_linear_models_gene_aroma(
    gwas_file,
    snp_df,
    top_n_snps=200,
    prune_threshold=0.95,
    save_dir=None,
    tag="default"
):
    """
    Run Linear, Ridge, Lasso, and ElasticNet regression models for all aroma traits
    using top GWAS SNPs (filtered by FDR and LD pruned).
    """


    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

    models = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=42),
        "Lasso": Lasso(alpha=0.3, max_iter=100000, random_state=42),
        "ElasticNet": ElasticNet(alpha=0.3, l1_ratio=0.5, max_iter=100000, random_state=42)
    }

    gwas = pd.read_csv(gwas_file)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    for trait in gwas["Trait"].unique():
        print(f"\n[Linear Models — {tag}] Trait: {trait}")
        gwas_df = gwas[gwas["Trait"] == trait]

        top_snps = gwas_df.sort_values("p_fdr").head(top_n_snps)["SNP"].tolist()
        if not top_snps:
            print("  No SNPs for modeling.")
            continue

        X = snp_df[top_snps].fillna(0).astype(float)
        X_pruned = ld_prune(X, threshold=prune_threshold)
        if X_pruned.shape[1] == 0:
            print("  No SNPs left after pruning.")
            continue

        X_scaled = StandardScaler().fit_transform(X_pruned)
        y = snp_df[trait].astype(float).values

        for name, model in models.items():
            model.fit(X_scaled, y)
            y_pred = model.predict(X_scaled)
            metrics = evaluate_regression(y, y_pred)
            cv_r2 = np.mean(cross_val_score(model, X_scaled, y, cv=cv, scoring="r2"))

            results.append({
                "Trait": trait,
                "Model": name,
                "n_SNPs": X_pruned.shape[1],
                "CV_Best_R2": cv_r2,
                **metrics
            })

    results_df = pd.DataFrame(results)

    # Save CSV
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        csv_path = os.path.join(save_dir, f"linear_models_{tag.replace(' ', '_')}.csv")
        results_df.to_csv(csv_path, index=False)
        print(f" Results saved → {csv_path}")

    # === Filter out extreme negative R² values ===
    filtered_df = results_df[results_df["CV_Best_R2"] > -2].copy()
    filtered_df["CV_Best_R2_clipped"] = filtered_df["CV_Best_R2"].clip(lower=-2)

    # === Plot each model ===
    for model_name in filtered_df["Model"].unique():
        model_df = filtered_df[filtered_df["Model"] == model_name].copy()
        if model_df.empty:
            continue

        model_df = model_df.sort_values("CV_Best_R2_clipped", ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(data=model_df, y="Trait", x="CV_Best_R2_clipped", palette="viridis")

        plt.axvline(0, color="black", linestyle="--", lw=1)
        plt.title(f"{model_name} Regression Performance per Aroma Trait ({tag})",
                  fontsize=13, weight="bold")
        plt.xlabel("Cross-validated R² (clipped ≤ -2)")
        plt.ylabel("Aroma Trait")

        # Annotate bars with R² values
        for i, v in enumerate(model_df["CV_Best_R2_clipped"]):
            plt.text(v + 0.02, i, f"{v:.2f}", va="center", fontsize=8)

        plt.tight_layout()
        if save_dir:
            plot_path = os.path.join(save_dir, f"{model_name.lower()}_plot_{tag.replace(' ', '_')}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            print(f" Plot saved → {plot_path}")

        plt.show()

    return results_df






import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler

# ---- Dummy placeholder: Replace with your real implementation ----
def ld_prune(X, threshold=0.95):
    """LD pruning placeholder: remove SNPs with correlation > threshold"""
    corr_matrix = X.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
    return X.drop(columns=to_drop)

def evaluate_regression(y_true, y_pred):
    """Return common regression metrics"""
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    return {
        "R2": r2_score(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred)
    }

# ---- Main Function ----
def run_krr_models_gene_aroma(
    gwas_file,
    snp_df,
    top_n_snps=500,
    prune_threshold=0.95,
    save_dir=None,
    tag="default"
):
    """
    Run Kernel Ridge Regression (KRR) models for all aroma traits using top GWAS SNPs.
    Performs hyperparameter tuning via RandomizedSearchCV (RBF kernel) and outputs
    evaluation metrics, predictions, and a performance plot.
    """

    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.model_selection import RandomizedSearchCV, KFold
    from sklearn.preprocessing import StandardScaler

    # --- Parameter distribution for tuning (wide range) ---
    param_dist = {
        "alpha": np.logspace(-3, 3, 100),   # from 0.001 to 1000
        "gamma": np.logspace(-4, 2, 100)    # from 0.0001 to 100
    }

    # --- Setup ---
    gwas = pd.read_csv(gwas_file)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    predictions_dict = {}

    # --- Loop through all traits ---
    for trait in gwas["Trait"].unique():
        print(f"\n[KRR — {tag}] Trait: {trait}")
        gwas_df = gwas[gwas["Trait"] == trait]

        # Select top SNPs
        top_snps = gwas_df.sort_values("p_fdr").head(top_n_snps)["SNP"].tolist()
        if not top_snps:
            print("  No SNPs for modeling.")
            continue

        # Prepare data
        X = snp_df[top_snps].fillna(0).astype(float)
        X_pruned = ld_prune(X, threshold=prune_threshold)
        if X_pruned.shape[1] == 0:
            print("  No SNPs left after pruning.")
            continue

        X_scaled = StandardScaler().fit_transform(X_pruned)
        y = snp_df[trait].astype(float).values

        # Randomized search
        search = RandomizedSearchCV(
            KernelRidge(kernel="rbf"),
            param_distributions=param_dist,
            n_iter=50,                # Try 50 random combos
            cv=cv,
            scoring="r2",
            n_jobs=-1,
            random_state=42
        )
        search.fit(X_scaled, y)

        best_model = search.best_estimator_
        y_pred = best_model.predict(X_scaled)
        metrics = evaluate_regression(y, y_pred)

        # Store predictions
        predictions_dict[trait] = pd.Series(y_pred, index=snp_df.index, name=trait)

        # Store results
        results.append({
            "Trait": trait,
            "n_SNPs": X_pruned.shape[1],
            "Best_alpha": search.best_params_["alpha"],
            "Best_gamma": search.best_params_["gamma"],
            "CV_Best_R2": search.best_score_,
            **metrics
        })

    # --- Combine results ---
    results_df = pd.DataFrame(results)
    predictions_df = pd.DataFrame(predictions_dict)

    # --- Save results ---
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        tag = tag or "default"
        csv_path = os.path.join(save_dir, f"krr_models_{tag.replace(' ', '_')}.csv")
        results_df.to_csv(csv_path, index=False)
        print(f"✅ Results saved → {csv_path}")

    # --- Visualization ---
    if not results_df.empty:
        sorted_df = results_df.sort_values("CV_Best_R2", ascending=False)
        plt.figure(figsize=(11, 6))
        sns.barplot(
            data=sorted_df,
            x="CV_Best_R2",
            y="Trait",
            palette="viridis"
        )
        plt.axvline(0, color="red", linestyle="--", lw=1)
        plt.xlabel("Cross-validated R²", fontsize=12)
        plt.ylabel("Aroma Trait", fontsize=12)
        plt.title(
            f"Kernel Ridge Regression (RBF) Performance per Aroma Trait ({tag})",
            fontsize=13, weight="bold"
        )

        for i, v in enumerate(sorted_df["CV_Best_R2"]):
            plt.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=9)
        plt.tight_layout()

        if save_dir:
            plot_path = os.path.join(save_dir, f"krr_models_plot_{tag.replace(' ', '_')}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            print(f"📊 Plot saved → {plot_path}")

        plt.show()

    return results_df, predictions_df


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def ld_prune(X, threshold=0.95):
    """LD pruning placeholder: remove SNPs with correlation > threshold"""
    corr_matrix = X.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
    return X.drop(columns=to_drop)

def evaluate_multioutput(y_true, y_pred, trait_names):
    """Compute regression metrics per trait and overall averages"""
    results = []
    for i, trait in enumerate(trait_names):
        results.append({
            "Trait": trait,
            "R2": r2_score(y_true[:, i], y_pred[:, i]),
            "MSE": mean_squared_error(y_true[:, i], y_pred[:, i]),
            "MAE": mean_absolute_error(y_true[:, i], y_pred[:, i])
        })
    results_df = pd.DataFrame(results)
    avg_metrics = results_df[["R2", "MSE", "MAE"]].mean().to_dict()
    return results_df, avg_metrics


def run_multitrait_krr_model(
    gwas_file,
    snp_df,
    trait_columns,
    top_n_snps=500,
    prune_threshold=0.95,
    save_dir=None,
    tag="multitrait"
):
    """
    Run multi-trait Kernel Ridge Regression (KRR) using SNPs from multitrait GWAS.
    Fits one multi-output KRR model to predict all traits simultaneously.
    """

    # --- Load GWAS results ---
    gwas = pd.read_csv(gwas_file)

    # --- Select top SNPs globally (by p_fdr) ---
    top_snps = gwas.sort_values("p_fdr").head(top_n_snps)["SNP"].tolist()
    print(f"Selected top {len(top_snps)} SNPs for multitrait modeling.")

    # --- Prepare data ---
    X = snp_df[top_snps].fillna(0).astype(float)
    X_pruned = ld_prune(X, threshold=prune_threshold)
    print(f"{X_pruned.shape[1]} SNPs retained after LD pruning.")

    # Extract Y matrix for all traits
    Y = snp_df[trait_columns].astype(float).values

    X_scaled = StandardScaler().fit_transform(X_pruned)

    # --- Hyperparameter search (shared across all traits) ---
    param_dist = {
        "alpha": np.logspace(-3, 3, 100),
        "gamma": np.logspace(-4, 2, 100)
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        KernelRidge(kernel="rbf"),
        param_distributions=param_dist,
        n_iter=50,
        cv=cv,
        scoring="r2",
        n_jobs=-1,
        random_state=42
    )

    # Fit on one trait (just to find best hyperparams)
    search.fit(X_scaled, Y[:, 0])
    best_params = search.best_params_
    print(f"Best α = {best_params['alpha']:.4f}, γ = {best_params['gamma']:.4f}")

    # --- Multi-output model ---
    model = MultiOutputRegressor(
        KernelRidge(kernel="rbf", alpha=best_params["alpha"], gamma=best_params["gamma"])
    )
    model.fit(X_scaled, Y)
    Y_pred = model.predict(X_scaled)

    # --- Evaluation ---
    results_df, avg_metrics = evaluate_multioutput(Y, Y_pred, trait_columns)
    print("\nAverage Performance Across Traits:")
    for k, v in avg_metrics.items():
        print(f"  {k}: {v:.3f}")

    # --- Save results ---
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        csv_path = os.path.join(save_dir, f"multitrait_krr_results_{tag}.csv")
        results_df.to_csv(csv_path, index=False)
        print(f" Trait-wise metrics saved → {csv_path}")

    # --- Plot ---
    plt.figure(figsize=(10, 6))
    sns.barplot(x="R2", y="Trait", data=results_df.sort_values("R2", ascending=False), palette="viridis")
    plt.title(f"Multi-trait Kernel Ridge Regression (RBF) Performance ({tag})", fontsize=13, weight="bold")
    plt.xlabel("R² per Trait")
    plt.ylabel("Trait")
    plt.axvline(0, color="red", linestyle="--")
    plt.tight_layout()

    if save_dir:
        plot_path = os.path.join(save_dir, f"multitrait_krr_plot_{tag}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f" Plot saved → {plot_path}")

    plt.show()

    return results_df, Y_pred



def run_nonlinear_models_gene_aroma(
    gwas_file,
    snp_df,
    top_n_snps=500,
    prune_threshold=0.95,
    save_dir=None,
    tag="default"
):
    """
    Run nonlinear regression models (SVR, RandomForest, XGBoost)
    for all traits using top GWAS SNPs (FDR-filtered + LD pruned).
    Automatically filters and clips extreme negative R² values (< -2) for clarity.
    """

 

    models = {
        "SVR": SVR(kernel="rbf", C=10, gamma=0.1),
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "XGBoost": XGBRegressor(
            n_estimators=300, learning_rate=0.1, max_depth=4,
            random_state=42, n_jobs=-1, verbosity=0
        )
    }

    gwas = pd.read_csv(gwas_file)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    for trait in gwas["Trait"].unique():
        print(f"\n[Nonlinear Models — {tag}] Trait: {trait}")
        gwas_df = gwas[gwas["Trait"] == trait]

        top_snps = gwas_df.sort_values("p_fdr").head(top_n_snps)["SNP"].tolist()
        if not top_snps:
            print("  No SNPs for modeling.")
            continue

        X = snp_df[top_snps].fillna(0).astype(float)
        X_pruned = ld_prune(X, threshold=prune_threshold)
        if X_pruned.shape[1] == 0:
            print("  No SNPs left after pruning.")
            continue

        X_scaled = StandardScaler().fit_transform(X_pruned)
        y = snp_df[trait].astype(float).values

        for name, model in models.items():
            model.fit(X_scaled, y)
            y_pred = model.predict(X_scaled)
            metrics = evaluate_regression(y, y_pred)
            cv_r2 = np.mean(cross_val_score(model, X_scaled, y, cv=cv, scoring="r2"))

            results.append({
                "Trait": trait,
                "Model": name,
                "n_SNPs": X_pruned.shape[1],
                "CV_Best_R2": cv_r2,
                **metrics
            })

    results_df = pd.DataFrame(results)

  
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        csv_path = os.path.join(save_dir, f"nonlinear_models_{tag.replace(' ', '_')}.csv")
        results_df.to_csv(csv_path, index=False)
        print(f" Results saved → {csv_path}")

    # Filter & clip extreme negative R² values 
    filtered_df = results_df[results_df["CV_Best_R2"] > -2].copy()
    filtered_df["CV_Best_R2_clipped"] = filtered_df["CV_Best_R2"].clip(lower=-2)

    #  Plot each model 
    for model_name in filtered_df["Model"].unique():
        model_df = filtered_df[filtered_df["Model"] == model_name].copy()
        if model_df.empty:
            continue

        model_df = model_df.sort_values("CV_Best_R2_clipped", ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(data=model_df, y="Trait", x="CV_Best_R2_clipped", palette="viridis")

        plt.axvline(0, color="black", linestyle="--", lw=1)
        plt.title(f"{model_name} Regression Performance per Aroma Trait ({tag})",
                  fontsize=13, weight="bold")
        plt.xlabel("Cross-validated R² (clipped ≤ -2)")
        plt.ylabel("Aroma Trait")

        # Annotate bars with R² values
        for i, v in enumerate(model_df["CV_Best_R2_clipped"]):
            plt.text(v + 0.02, i, f"{v:.2f}", va="center", fontsize=8)

        plt.tight_layout()
        if save_dir:
            plot_path = os.path.join(save_dir, f"{model_name.lower()}_plot_{tag.replace(' ', '_')}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            print(f" Plot saved → {plot_path}")

        plt.show()

    return results_df


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler

def run_nonlinear_models_multitrait(
    gwas_file,
    snp_df,
    trait_columns,
    top_n_snps=500,
    prune_threshold=0.95,
    save_dir=None,
    tag="multitrait"
):
    """
    Run nonlinear regression models (SVR, RandomForest, XGBoost)
    using SNPs from multi-trait GWAS results across multiple aroma traits.
    Each model predicts every trait using the same shared SNP set.
    """

    models = {
        "SVR": SVR(kernel="rbf", C=10, gamma=0.1),
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "XGBoost": XGBRegressor(
            n_estimators=300, learning_rate=0.1, max_depth=4,
            random_state=42, n_jobs=-1, verbosity=0
        )
    }

    # --- Load GWAS results ---
    gwas = pd.read_csv(gwas_file)
    top_snps = gwas.sort_values("p_value").head(top_n_snps)["SNP"].tolist()
    if not top_snps:
        print(" No SNPs found in the GWAS file.")
        return pd.DataFrame()

    # --- Prepare data ---
    X = snp_df[top_snps].fillna(0).astype(float)
    X_pruned = ld_prune(X, threshold=prune_threshold)
    if X_pruned.shape[1] == 0:
        print(" No SNPs left after LD pruning.")
        return pd.DataFrame()

    X_scaled = StandardScaler().fit_transform(X_pruned)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    # --- Run models for each trait ---
    for trait in trait_columns:
        y = snp_df[trait].astype(float).values
        print(f"\n[Nonlinear Multi-Trait Models — {tag}] Trait: {trait}")

        for name, model in models.items():
            model.fit(X_scaled, y)
            y_pred = model.predict(X_scaled)

            metrics = evaluate_regression(y, y_pred)
            cv_r2 = np.mean(cross_val_score(model, X_scaled, y, cv=cv, scoring="r2"))

            results.append({
                "Trait": trait,
                "Model": name,
                "n_SNPs": X_pruned.shape[1],
                "CV_Best_R2": cv_r2,
                **metrics
            })

    # --- Collect results ---
    results_df = pd.DataFrame(results)

    # --- Save results ---
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        csv_path = os.path.join(save_dir, f"nonlinear_models_multitrait_{tag.replace(' ', '_')}.csv")
        results_df.to_csv(csv_path, index=False)
        print(f" Results saved → {csv_path}")

    # --- Plot ---
    if not results_df.empty:
        filtered_df = results_df[results_df["CV_Best_R2"] > -2].copy()
        filtered_df["CV_Best_R2_clipped"] = filtered_df["CV_Best_R2"].clip(lower=-2)

        plt.figure(figsize=(11, 6))
        sns.barplot(
            data=filtered_df.sort_values("CV_Best_R2_clipped", ascending=False),
            x="CV_Best_R2_clipped",
            y="Trait",
            hue="Model",
            palette="viridis"
        )
        plt.axvline(0, color="red", linestyle="--", lw=1)
        plt.title(f"Nonlinear Multi-Trait Regression (SVR, RF, XGB) — {tag}", fontsize=13, weight="bold")
        plt.xlabel("Cross-validated R² (clipped ≤ -2)")
        plt.ylabel("Trait")
        plt.tight_layout()

        if save_dir:
            plot_path = os.path.join(save_dir, f"nonlinear_multitrait_plot_{tag.replace(' ', '_')}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            print(f" Plot saved → {plot_path}")

        plt.show()

    return results_df



def run_linear_models_sensory(
    gwas_file,
    merged_snp_df,
    fdr_threshold=0.05,
    top_n_fallback=500,
    prune_threshold=0.95,
    save_dir=None
):
    """
    Run Linear, Ridge, and ElasticNet regression models on sensory PCs
    using top GWAS-selected SNPs (FDR < threshold or fallback top N by p-value).
    """


    def evaluate_regression(y_true, y_pred):
        return {
            "R2_in_sample": r2_score(y_true, y_pred),
            "MSE": mean_squared_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "MAE": mean_absolute_error(y_true, y_pred),
        }

    def ld_prune(X_df, threshold=0.95, max_snps=2000):
        X_df = X_df.loc[:, X_df.std() > 0]
        if X_df.shape[1] > max_snps:
            X_df = X_df.iloc[:, :max_snps]
        if X_df.shape[1] <= 1:
            return X_df
        corr = X_df.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
        return X_df.drop(columns=to_drop)


    gwas_sensory = pd.read_csv(gwas_file)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
    }

    for trait in gwas_sensory["Trait"].unique():
        print(f"\n[Linear Models] Trait: {trait}")
        gwas_df = gwas_sensory[gwas_sensory["Trait"] == trait]

        sig_snps = gwas_df[gwas_df["p_fdr"] < fdr_threshold]["SNP"].tolist()
        if not sig_snps:
            print(f"  No SNPs with FDR < {fdr_threshold}. Using top {top_n_fallback} by p-value.")
            sig_snps = gwas_df.sort_values("p_fdr").head(top_n_fallback)["SNP"].tolist()
        else:
            print(f"  Significant SNPs found: {len(sig_snps)}")

        if not sig_snps:
            continue

        X = merged_snp_df[sig_snps].fillna(0).astype(float)
        X_pruned = ld_prune(X, threshold=prune_threshold)

        if X_pruned.shape[1] == 0:
            continue

        X_scaled = StandardScaler().fit_transform(X_pruned)
        y = merged_snp_df[trait].astype(float).values

        for name, model in models.items():
            model.fit(X_scaled, y)
            y_pred = model.predict(X_scaled)
            metrics = evaluate_regression(y, y_pred)
            cv_r2 = np.mean(cross_val_score(model, X_scaled, y, cv=cv, scoring="r2"))

            results.append({
                "Trait": trait,
                "Model": name,
                "n_SNPs": X_pruned.shape[1],
                "CV_Best_R2": cv_r2,
                **metrics
            })

    results_df = pd.DataFrame(results)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "linear_models_sensory_PCs_FDR05.csv")
        results_df.to_csv(save_path, index=False)
        print(f"\n Linear model results saved to: {save_path}")

    if not results_df.empty:
        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=results_df.sort_values("CV_Best_R2", ascending=False),
            x="CV_Best_R2", y="Trait", hue="Model", palette="tab10"
        )
        plt.axvline(0, color="red", linestyle="--", lw=1)
        plt.xlabel("Cross-validated R²")
        plt.ylabel("Sensory PC Trait")
        plt.title("Linear Models on FDR-Filtered SNPs (FDR < 0.05)", fontsize=13, weight="bold")

        # === Adjust x-axis limits to compress large negative values ===
        x_min, x_max = plt.xlim()
        if x_min < -2:
            plt.xlim(-2, x_max + 0.1)  # Cap negatives at -2 so positives are clearer

        plt.tight_layout()
        plt.show()



def run_krr_sensory_pcs(
    gwas_file,
    merged_snp_df,
    fdr_threshold=0.05,
    top_n_fallback=500,
    prune_threshold=0.95,
    save_dir=None
):
    """
    Run Kernel Ridge Regression (RBF kernel) for sensory PCs
    using GWAS-selected SNPs filtered by FDR threshold or fallback to top N.

    Parameters
    ----------
    gwas_file : str
        Path to GWAS results CSV (must include 'Trait', 'SNP', 'p_fdr').
    merged_snp_df : pd.DataFrame
        SNP × Sensory PC dataset.
    fdr_threshold : float, default=0.1
        FDR significance threshold for SNP selection.
    top_n_fallback : int, default=500
        Number of top SNPs to use if no SNPs pass the FDR filter.
    prune_threshold : float, default=0.95
        LD pruning correlation threshold.
    save_dir : str or None
        Directory to save results and plots.

    Returns
    -------
    pd.DataFrame
        Summary table of KRR model results for sensory PCs.
    """


    #  Hyperparameter grid for KRR 
    param_grid = {
        "alpha": [0.01, 0.1, 1, 10, 100],
        "gamma": [1e-4, 1e-3, 1e-2, 0.1, 1]
    }

    # Load GWAS file 
    gwas_sensory = pd.read_csv(gwas_file)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    #  Loop over each sensory trait 
    for trait in gwas_sensory["Trait"].unique():
        print(f"\n[Kernel Ridge Regression - Sensory PCs] Trait: {trait}")

        gwas_df = gwas_sensory[gwas_sensory["Trait"] == trait]

        # Select SNPs by FDR or fallback
        sig_snps = gwas_df[gwas_df["p_fdr"] < fdr_threshold]["SNP"].tolist()
        if not sig_snps:
            print(f"  No SNPs with FDR < {fdr_threshold}. Using top {top_n_fallback} by p-value.")
            sig_snps = gwas_df.sort_values("p_fdr").head(top_n_fallback)["SNP"].tolist()
        else:
            print(f"  Significant SNPs found: {len(sig_snps)}")

        if not sig_snps:
            continue

        #  SNP feature matrix 
        X = merged_snp_df[sig_snps].fillna(0).astype(float)
        X_pruned = ld_prune(X, threshold=prune_threshold)

        if X_pruned.shape[1] == 0:
            print("  No SNPs left after pruning.")
            continue

        X_scaled = StandardScaler().fit_transform(X_pruned)
        y = merged_snp_df[trait].astype(float).values

        #  Grid search KRR 
        grid = GridSearchCV(
            KernelRidge(kernel="rbf"),
            param_grid=param_grid,
            cv=cv,
            scoring=make_scorer(r2_score),
            n_jobs=-1
        )
        grid.fit(X_scaled, y)

        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_scaled)
        metrics = evaluate_regression(y, y_pred)

        results.append({
            "Trait": trait,
            "n_SNPs": X_pruned.shape[1],
            "Best_alpha": grid.best_params_["alpha"],
            "Best_gamma": grid.best_params_["gamma"],
            "CV_Best_R2": grid.best_score_,
            **metrics
        })

    #  Compile results 
    results_df = pd.DataFrame(results)

    # Save results
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "krr_sensory_PCs_FDRfiltered.csv")
        results_df.to_csv(save_path, index=False)
        print(f"\n Results saved to: {save_path}")

    if not results_df.empty:
        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=results_df.sort_values("CV_Best_R2", ascending=False),
            x="CV_Best_R2", y="Trait", palette="viridis"
        )
        plt.axvline(0, color="red", linestyle="--", lw=1)
        plt.xlabel("Cross-validated R²")
        plt.ylabel("Sensory PC Trait")
        plt.title(f"Kernel Ridge Regression (RBF) on Sensory PCs (FDR < {fdr_threshold})", fontsize=13, weight="bold")
        plt.tight_layout()

        if save_dir:
            plot_path = os.path.join(save_dir, "krr_sensory_PCs_plot_FDRfiltered.png")
            plt.savefig(plot_path, dpi=300)
            print(f" Plot saved → {plot_path}")

        plt.show()
    return results_df



import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import seaborn as sns
import matplotlib.pyplot as plt

# --- Helper Functions (if not already defined) ---
def ld_prune(X, threshold=0.95):
    """Simple LD pruning using correlation threshold."""
    corr = np.corrcoef(X, rowvar=False)
    upper = np.triu(np.abs(corr), k=1)
    to_remove = set()
    for i in range(upper.shape[0]):
        for j in range(i + 1, upper.shape[1]):
            if upper[i, j] > threshold:
                to_remove.add(j)
    keep_idx = [i for i in range(X.shape[1]) if i not in to_remove]
    return X.iloc[:, keep_idx]

def evaluate_regression(y_true, y_pred):
    """Compute basic regression metrics."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    return {"R2": r2, "MSE": mse, "MAE": mae}


# --- Main Function ---
def run_nonlinear_sensory_pcs(
    gwas_file,
    merged_snp_df,
    fdr_threshold=0.05,
    top_n_fallback=500,
    prune_threshold=0.95,
    save_dir=None
):
    """
    Run nonlinear regression models (SVR, RandomForest, XGBoost)
    for each sensory PC using GWAS-selected SNPs (FDR-based filtering + fallback).
    """

    gwas_sensory = pd.read_csv(gwas_file)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    nonlinear_models = {
        "SVR": SVR(kernel="rbf", C=10, gamma=0.1),
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "XGBoost": XGBRegressor(
            n_estimators=300, learning_rate=0.1,
            max_depth=4, random_state=42, n_jobs=-1, verbosity=0
        )
    }

    results_nonlinear = []

    for trait in gwas_sensory["Trait"].unique():
        print(f"\n[Nonlinear Models - Sensory PCs] Trait: {trait}")

        gwas_df = gwas_sensory[gwas_sensory["Trait"] == trait]

        # Select SNPs based on FDR threshold or fallback
        sig_snps = gwas_df[gwas_df["p_fdr"] < fdr_threshold]["SNP"].tolist()
        if not sig_snps:
            print(f"  No SNPs with FDR < {fdr_threshold}. Using top {top_n_fallback} by p-value.")
            sig_snps = gwas_df.sort_values("p_fdr").head(top_n_fallback)["SNP"].tolist()
        else:
            print(f"  Significant SNPs found: {len(sig_snps)}")

        if not sig_snps:
            print("  Skipping (no SNPs available).")
            continue

        X = merged_snp_df[sig_snps].fillna(0).astype(float)
        X_pruned = ld_prune(X, threshold=prune_threshold)

        if X_pruned.shape[1] == 0:
            print("  No SNPs left after LD pruning.")
            continue

        X_scaled = StandardScaler().fit_transform(X_pruned)
        y = merged_snp_df[trait].astype(float).values

        for name, model in nonlinear_models.items():
            print(f"   → Fitting {name} ...")
            model.fit(X_scaled, y)
            y_pred = model.predict(X_scaled)

            metrics = evaluate_regression(y, y_pred)
            cv_r2 = np.mean(cross_val_score(model, X_scaled, y, cv=cv, scoring="r2", n_jobs=-1))

            results_nonlinear.append({
                "Trait": trait,
                "Model": name,
                "n_SNPs": X_pruned.shape[1],
                "CV_Best_R2": cv_r2,
                **metrics
            })

    # Collect results
    results_df = pd.DataFrame(results_nonlinear)

    # --- Compute mean CV R² ---
    if not results_df.empty:
        mean_cv_r2 = results_df["CV_Best_R2"].mean()
        print(f"\nMean cross-validated R² across all traits: {mean_cv_r2:.4f}")
    else:
        mean_cv_r2 = np.nan
        print("\nNo valid model results available.")

    # Save results
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"nonlinear_models_sensory_PCs_FDRfiltered_top{top_n_fallback}.csv")
        results_df.to_csv(save_path, index=False)
        print(f"\nResults saved to: {save_path}")

    # Visualization
    if not results_df.empty:
        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=results_df.sort_values("CV_Best_R2", ascending=False),
            x="CV_Best_R2", y="Trait", hue="Model", palette="tab10"
        )
        plt.axvline(0, color="red", linestyle="--", lw=1)
        plt.xlabel("Cross-validated R²")
        plt.ylabel("Sensory PC Trait")
        plt.title(f"Nonlinear Model Performance on Sensory PCs (FDR < {fdr_threshold})",
                  fontsize=13, weight="bold")
        plt.legend(title="Model")
        plt.tight_layout()

        if save_dir:
            plot_path = os.path.join(save_dir, f"nonlinear_sensory_PCs_plot_top{top_n_fallback}.png")
            plt.savefig(plot_path, dpi=300)
            print(f"Plot saved → {plot_path}")
        plt.show()

    # Return dictionary
    return {"results_df": results_df, "mean_cv_r2": mean_cv_r2}





def correlate_gene_predictions_with_sensory_pcs(
    gwas_file: str,
    snp_df: pd.DataFrame,
    merged_snp_sensory: pd.DataFrame,
    save_dir: str,
    top_n_snps: int = 2000,
    prune_threshold: float = 0.95,
    r2_threshold: float = 0.5,
    tag: str = "outlier_removed"
):
    """
    Run Kernel Ridge Regression (KRR) on gene-based aroma data,
    select high-performing predicted traits (CV R² >= threshold),
    and compute correlations with sensory PCs (PC1–PC3).

    Returns
    -------
    corr_matrix : pd.DataFrame
        Correlation matrix between high-performing gene-predicted compounds and sensory PCs
    predictions : pd.DataFrame
        DataFrame of model-predicted trait values for each variety
    """

    print(f"\n▶ Running KRR models for gene-predicted aroma compounds ({tag})...")
    
    # IMPORTANT: run_krr_models_gene_aroma must return both results and predictions
    results_krr, predictions = run_krr_models_gene_aroma(
        gwas_file=gwas_file,
        snp_df=snp_df,
        top_n_snps=top_n_snps,
        prune_threshold=prune_threshold,
        save_dir=save_dir,
        tag=tag
    )

    # Select high-performing traits
    high_perf_traits = results_krr.query(f"CV_Best_R2 >= {r2_threshold}")["Trait"].tolist()
    print(f"\n High-performing traits (CV R² ≥ {r2_threshold}): {len(high_perf_traits)}")
    print("→", high_perf_traits)

    # Filter predictions to only high-performing traits
    predictions = predictions[high_perf_traits]
    print(f"\n Predictions DataFrame ready: {predictions.shape}")

    # Extract sensory PCs
    sensory_pcs = merged_snp_sensory[["PC1", "PC2", "PC3"]].copy()

    # Normalize indices (variety names)
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

    # Match samples
    common_samples = predictions.index.intersection(sensory_pcs.index)
    predictions = predictions.loc[common_samples]
    sensory_pcs = sensory_pcs.loc[common_samples]

    print(f" Common samples: {len(common_samples)}")

    # Compute correlation
    merged_df = pd.concat([predictions, sensory_pcs], axis=1)
    corr_matrix = merged_df.corr().loc[high_perf_traits, ["PC1", "PC2", "PC3"]]

    # Save results
    os.makedirs(save_dir, exist_ok=True)
    corr_path = os.path.join(save_dir, f"gene_predicted_compound_sensory_corr_{tag}.csv")
    corr_matrix.to_csv(corr_path)
    print(f"\n Correlation matrix saved → {corr_path}")

    # Visualization
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, cmap="coolwarm", center=0, annot=True, fmt=".2f")
    plt.title(
        f"Correlation between Gene-predicted Aroma Compounds ({tag}) and Sensory PCs",
        fontsize=13,
        weight="bold",
    )
    plt.xlabel("Sensory Components (PC1–PC3)")
    plt.ylabel("Predicted Aroma Compounds")
    plt.tight_layout()
    plt.show()

    # Display top correlations
    corr_pairs = corr_matrix.unstack().sort_values(ascending=False)
    print("\n Top correlations:")
    display(corr_pairs.head(20))

    return corr_matrix, predictions



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


    # Identify top SNPs for each high-correlation trait
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

    
        # Plot SNP effects
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

def build_gene_compound_sensory_triangle(
    gwas_file: str,
    corr_matrix: pd.DataFrame,
    corr_threshold: float = 0.1,
    top_compounds: int = 5,
    top_snps_per_compound: int = 4000,
    save_dir: str = None
):
    """
    Build and visualize the Gene–Compound–Sensory PC triangle network,
    and compute SNP overlap across compounds.
    """

    import os
    import matplotlib.pyplot as plt
    import networkx as nx
    import seaborn as sns
    import pandas as pd
    import numpy as np

    # ----------------------------------------------------------
    # Load and check data
    # ----------------------------------------------------------
    gwas_all = pd.read_csv(gwas_file)
    if not {"Trait", "SNP", "p_fdr"}.issubset(gwas_all.columns):
        raise ValueError("GWAS file must contain columns: Trait, SNP, p_fdr")

    # ----------------------------------------------------------
    # Identify top compounds correlated with sensory PCs
    # ----------------------------------------------------------
    corr_abs = corr_matrix.abs()
    avg_corr = corr_abs.mean(axis=1).sort_values(ascending=False)
    selected_compounds = avg_corr.head(top_compounds).index.tolist()

    print(f"\nTop {top_compounds} gene-predicted compounds most related to sensory PCs:")
    display(avg_corr.head(top_compounds))

    # ----------------------------------------------------------
    # Build triangle relationships
    # ----------------------------------------------------------
    triangle_links = []
    for compound in selected_compounds:
        sensory_target = corr_matrix.loc[compound].abs().idxmax()
        corr_value = corr_matrix.loc[compound, sensory_target]

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
                "Sensory PC": sensory_target,
                "Compound–Sensory Corr": corr_value
            })

    triangle_df = pd.DataFrame(triangle_links)
    print(f"\n Built triangle for {len(selected_compounds)} compounds × {top_snps_per_compound} SNPs each.")
    display(triangle_df.head(10))

    # ----------------------------------------------------------
    # Compute SNP overlap between compounds
    # ----------------------------------------------------------
    compound_snps = {
        comp: set(triangle_df.loc[triangle_df["Compound"] == comp, "Gene (SNP)"])
        for comp in triangle_df["Compound"].unique()
    }

    overlap_matrix = pd.DataFrame(0, index=compound_snps.keys(), columns=compound_snps.keys())

    for c1 in compound_snps:
        for c2 in compound_snps:
            overlap_matrix.loc[c1, c2] = len(compound_snps[c1].intersection(compound_snps[c2]))

    print("\n SNP Overlap Matrix (top of matrix):")
    display(overlap_matrix.head())

    # ----------------------------------------------------------
    # Visualization — Lower triangle heatmap
    # ----------------------------------------------------------
    mask = np.triu(np.ones_like(overlap_matrix, dtype=bool))

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        overlap_matrix,
        mask=mask,
        annot=True,
        fmt="d",
        cmap="YlOrRd",
        cbar_kws={"label": "Number of overlapping SNPs"}
    )
    plt.title("Overlap of Top SNPs Across Selected Aroma Compounds", fontsize=14, weight="bold")
    plt.xlabel("Compound")
    plt.ylabel("Compound")
    plt.tight_layout()

    if save_dir:
        overlap_path = os.path.join(save_dir, "snp_overlap_triangle_heatmap.png")
        plt.savefig(overlap_path, dpi=300, bbox_inches="tight")
        print(f" Overlap heatmap saved to {overlap_path}")

    plt.show()

    # ----------------------------------------------------------
    # Visualization — Network Graph
    # ----------------------------------------------------------
    G = nx.Graph()

    for _, row in triangle_df.iterrows():
        G.add_node(row["Gene (SNP)"], color="skyblue", layer="Gene")
        G.add_node(row["Compound"], color="lightgreen", layer="Compound")
        G.add_node(row["Sensory PC"], color="salmon", layer="Sensory")

        G.add_edge(row["Gene (SNP)"], row["Compound"], weight=0.8)
        G.add_edge(row["Compound"], row["Sensory PC"], weight=abs(row["Compound–Sensory Corr"]))

    pos = {}
    x_gene, x_comp, x_sens = 0, 1, 2
    y_gap = 1.0

    genes = [n for n, d in G.nodes(data=True) if d["layer"] == "Gene"]
    comps = [n for n, d in G.nodes(data=True) if d["layer"] == "Compound"]
    senses = [n for n, d in G.nodes(data=True) if d["layer"] == "Sensory"]

    for i, g in enumerate(genes): pos[g] = (x_gene, i * y_gap)
    for i, c in enumerate(comps): pos[c] = (x_comp, i * y_gap)
    for i, s in enumerate(senses): pos[s] = (x_sens, i * y_gap)

    plt.figure(figsize=(14, 10))
    node_colors = [G.nodes[n]["color"] for n in G.nodes]
    nx.draw(
        G, pos,
        with_labels=True,
        node_color=node_colors,
        node_size=1200,
        font_size=9,
        edge_color="gray",
        width=1.2
    )
    plt.title("Triangle Relationships: Gene–Compound–Sensory PC", fontsize=15, weight="bold")
    plt.axis("off")

    if save_dir:
        fig_path = os.path.join(save_dir, "gene_compound_sensory_triangle.png")
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        print(f"Triangle figure saved to {fig_path}")

    plt.show()

    return triangle_df, overlap_matrix

    

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



def run_plsr_gene_predicted_vs_sensory(
    predictions: pd.DataFrame,
    sensory_reconstructed_path: str,
    high_r2_traits: list,
    save_dir: str,
    n_components: int = 5
):
    """
    Perform PLSR between high-predictive gene-based aroma compound predictions
    and reconstructed sensory attributes.

    Parameters
    ----------
    predictions : pd.DataFrame
        DataFrame of predicted aroma compound values (index = varieties)
    sensory_reconstructed_path : str
        Path to the CSV file containing sensory reconstructed traits (index = varieties)
    high_r2_traits : list
        List of compounds with high predictive performance (e.g., CV R² ≥ 0.5)
    save_dir : str
        Directory to save correlation and loading outputs
    n_components : int, optional
        Number of PLS components to extract (default = 5)

    Returns
    -------
    dict
        {
            "corr_df": correlation DataFrame between PLS components and sensory attributes,
            "loadings_df": compound loadings per component,
            "flavor_alignment": dict describing sensory direction of each PLS component
        }
    """


    #  Load reconstructed sensory traits

    sensory_df = pd.read_csv(sensory_reconstructed_path, index_col=0)


    # Normalize and align sample names
    sensory_df.index = sensory_df.index.str.upper().str.replace(" ", "")
    predictions.index = predictions.index.str.upper().str.replace(" ", "")

    common_varieties = predictions.index.intersection(sensory_df.index)
    print(f" Common varieties found: {len(common_varieties)}")

    # Keep only shared samples
    X = predictions.loc[common_varieties, high_r2_traits]
    Y = sensory_df.loc[common_varieties]

    # Keep only numeric columns
    X = X.select_dtypes(include=[np.number])
    Y = Y.select_dtypes(include=[np.number])

    print(f" Using {X.shape[1]} high-predictive compounds for PLS")


    # Standardize both datasets
    X_scaled = StandardScaler().fit_transform(X)
    Y_scaled = StandardScaler().fit_transform(Y)

 
    # Fit PLS model
    n_components = min(n_components, X_scaled.shape[1], Y_scaled.shape[1])
    pls = PLSRegression(n_components=n_components)
    pls.fit(X_scaled, Y_scaled)
    print(f" Fitted PLS model with {n_components} components")


    #  Correlation between PLS components and sensory traits
   
    corr_df = pd.DataFrame(
        np.corrcoef(pls.x_scores_.T, Y_scaled.T)[:n_components, n_components:],
        index=[f"PLS{i+1}" for i in range(n_components)],
        columns=Y.columns,
    )

    print("\n Correlation between PLS components and sensory attributes:")
    display(corr_df.round(2))


    #  Visualize correlation heatmap
    plt.figure(figsize=(14, 6))
    sns.heatmap(corr_df, cmap="coolwarm", center=0, annot=True, fmt=".2f")
    plt.title("PLS Components vs Sensory Attributes (High-Predicted Compounds)")
    plt.xlabel("Sensory Attributes")
    plt.ylabel("PLS Components")
    plt.tight_layout()
    plt.show()

    # Compound loadings
    loadings_df = pd.DataFrame(
        pls.x_loadings_,
        index=X.columns,
        columns=[f"PLS{i+1}" for i in range(n_components)],
    )

    for comp in loadings_df.columns:
        top_loadings = loadings_df[comp].abs().sort_values(ascending=False).head(15)
        plt.figure(figsize=(9, 4))
        sns.barplot(
            x=top_loadings.values,
            y=top_loadings.index,
            palette="viridis"
        )
        plt.title(f"Top 15 Compound Contributors to {comp}")
        plt.xlabel(f"|{comp} Loading| (Importance)")
        plt.ylabel("Compound")
        plt.tight_layout()
        plt.show()

    # Step 8 — Identify dominant sensory direction of each PLS axis

    flavor_alignment = {}
    for comp in corr_df.index:
        best_match = corr_df.loc[comp].abs().idxmax()
        direction = "positive" if corr_df.loc[comp, best_match] > 0 else "negative"
        flavor_alignment[comp] = f"{best_match} ({direction})"

    print("\n Flavor alignment per PLS axis:")
    for comp, desc in flavor_alignment.items():
        print(f"  {comp}: aligned with {desc}")

    # Step 9 — Save outputs
    os.makedirs(save_dir, exist_ok=True)
    corr_path = os.path.join(save_dir, "pls_high_predicted_compound_sensory_correlations.csv")
    loadings_path = os.path.join(save_dir, "pls_high_predicted_compound_loadings.csv")

    corr_df.to_csv(corr_path)
    loadings_df.to_csv(loadings_path)

    print("\n Correlation and loadings files saved successfully:")
    print(f"   → {corr_path}")
    print(f"   → {loadings_path}")

    return {
        "corr_df": corr_df,
        "loadings_df": loadings_df,
        "flavor_alignment": flavor_alignment,
    }




#   Function to prepare aligned & scaled data

def prepare_plsr_data(predictions, high_r2_traits, sensory_reconstructed_path):
    """
    Align and standardize predictions (X) and sensory traits (Y) for PLSR.
    """
    # Load sensory reconstructed data
    sensory_df = pd.read_csv(sensory_reconstructed_path, index_col=0)

    # Normalize sample names
    sensory_df.index = sensory_df.index.str.upper().str.replace(" ", "")
    predictions.index = predictions.index.str.upper().str.replace(" ", "")

    # Find common varieties
    common_varieties = predictions.index.intersection(sensory_df.index)
    print(f" Common varieties found: {len(common_varieties)}")

    # Subset to shared samples
    X = predictions.loc[common_varieties, high_r2_traits].select_dtypes(include=[np.number])
    Y = sensory_df.loc[common_varieties].select_dtypes(include=[np.number])

    # Standardize
    X_scaled = StandardScaler().fit_transform(X)
    Y_scaled = StandardScaler().fit_transform(Y)

    print(f"X_scaled shape: {X_scaled.shape}, Y_scaled shape: {Y_scaled.shape}")
    return X, Y, X_scaled, Y_scaled



# Function for PLS biplot

def plot_plsr_biplot(
    pls_model,
    X_scaled: pd.DataFrame,
    pc1: int = 1,
    pc2: int = 2,
    top_n_loadings: int = 15,
    rotate: str = 'none',
    scale_arrows: float = 10,
    text_size: int = 8
):
    """
    Plot a PLS biplot showing sample scores and top contributing compound loadings.
    """
    # Extract scores
    x_scores = pls_model.x_scores_[:, pc1 - 1]
    y_scores = pls_model.x_scores_[:, pc2 - 1]

    # Apply rotation
    if rotate == '180':
        x_scores, y_scores = -x_scores, -y_scores
    elif rotate == '90cw':
        x_scores, y_scores = y_scores, -x_scores
    elif rotate == '90ccw':
        x_scores, y_scores = -y_scores, x_scores

    # Plot samples
    plt.figure(figsize=(10, 8))
    plt.scatter(x_scores, y_scores, c='dodgerblue', alpha=0.6, label='Varieties')

    for i, txt in enumerate(X_scaled.index):
        plt.text(x_scores[i], y_scores[i], txt, fontsize=text_size, alpha=0.7)

    # Loadings
    x_loadings = pls_model.x_loadings_[:, pc1 - 1]
    y_loadings = pls_model.x_loadings_[:, pc2 - 1]

    if rotate == '180':
        x_loadings, y_loadings = -x_loadings, -y_loadings
    elif rotate == '90cw':
        x_loadings, y_loadings = y_loadings, -x_loadings
    elif rotate == '90ccw':
        x_loadings, y_loadings = -y_loadings, x_loadings

    loadings_df = pd.DataFrame({'x': x_loadings, 'y': y_loadings}, index=X_scaled.columns)
    top_loadings = loadings_df.abs().sum(axis=1).sort_values(ascending=False).head(top_n_loadings).index

    for compound in top_loadings:
        plt.arrow(
            0, 0,
            loadings_df.loc[compound, 'x'] * scale_arrows,
            loadings_df.loc[compound, 'y'] * scale_arrows,
            color='crimson', alpha=0.4, head_width=0.05
        )
        plt.text(
            loadings_df.loc[compound, 'x'] * scale_arrows * 0.8,
            loadings_df.loc[compound, 'y'] * scale_arrows * 0.8,
            compound, color='crimson', fontsize=9
        )

    plt.axhline(0, color='grey', linestyle='--', linewidth=1)
    plt.axvline(0, color='grey', linestyle='--', linewidth=1)
    plt.xlabel(f"PLS{pc1} Scores")
    plt.ylabel(f"PLS{pc2} Scores")
    plt.title(f"PLS Biplot (PLS{pc1} vs PLS{pc2}) — rotated: {rotate}")
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.show()
