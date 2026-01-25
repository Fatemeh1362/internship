# compound_analysis_module.py


"""
Author: Fatemeh Monfared
Module: compound_analysis_module.py (GC–MS Compound & Sensory Analysis Utilities)
Thesis: Data-Driven Analysis of Potato Aroma and Flavor Using TD–GC–MS and Machine Learning
Affiliation: Hanze University of Applied Sciences
Year: 2025

Purpose
-------
This module provides reusable functions for GC–MS aroma compound identification, filtering,
and exploratory analysis, including integration with sensory profiling. It is designed to be
imported by notebooks/pipelines so the same compound-level logic is applied consistently
across batches and projects (company + thesis).

What this module supports (high level)
--------------------------------------
1) File discovery & dataset construction
   - Find GC–MS compound CSV files (case-insensitive search)
   - Combine sample/QC/env/blank compound tables into one dataset with FileType metadata

2) Contamination filtering
   - Remove compounds detected in blanks/environment/QC that overlap with sample entries
     within a retention-time tolerance (RT ± tolerance)

3) Compound ↔ peak matching (annotation)
   - Match NIST-identified compounds to quantified peak tables (from preprocessing outputs)
     using retention time (RT) and (optionally) m/z tolerances
   - Build aroma intensity outputs:
       * long-format matched-per-variety table
       * wide Variety × Compound intensity matrix

   Note: The peak table input is typically produced by the data preparation workflow
   (drift-corrected + normalised peak intensities).

4) Canonical aroma library & name harmonisation
   - Curated potato aroma list (core + optional)
   - Alias handling and canonical name mapping (exact/alias/substring/fuzzy)
   - Save reference RT/mz tables and report unmapped/missing aroma names

5) Aroma chemistry interpretation helpers
   - Assign chemical families (e.g., Aldehyde, Alcohol, Ketone, Sulfur, Ester, Acid, Pyrazine)
   - Assign aroma descriptions (curated where available; otherwise family-based defaults)
   - Summarise chemical-family distributions

6) Multivariate analysis & clustering
   - PCA-based exploration of aroma matrices
   - Clustering workflows (KMeans / Gaussian Mixture Models) with model selection metrics
   - Outlier detection (Mahalanobis distance rules) and visual summaries
   - Feature importance interpretation from PCA loadings

7) Sensory integration
   - Clean and summarise sensory traits per Variety
   - Merge aroma matrices with sensory summaries (Variety alignment/normalisation)
   - Downstream correlation / prediction utilities (where used in notebooks)
"""

from __future__ import annotations

#  library
import os
import glob
import json
import warnings
from typing import Optional, Tuple, Dict, List
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from difflib import get_close_matches
from scipy.spatial import distance

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    r2_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.neighbors import NearestNeighbors
from matplotlib.patches import Ellipse
import re
from difflib import get_close_matches
from collections import OrderedDict, defaultdict
from typing import List, Dict, Tuple




def _ensure_variety_col(df: pd.DataFrame, desired: str = "Variety") -> pd.DataFrame:
    """
    Ensure a DataFrame contains a variety column named exactly `desired`.
    Accepts 'Variety'/'variety'/'VARIETY' etc.
    """
    df = df.copy()
    for c in df.columns:
        if c.strip().lower() == "variety":
            if c != desired:
                df.rename(columns={c: desired}, inplace=True)
            return df
    # if not found, assume first column is variety index-like
    df = df.reset_index()
    df.rename(columns={df.columns[0]: desired}, inplace=True)
    return df


def _safe_numeric(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(pd.to_numeric, errors="coerce")



 # File discovery & combining

def find_files_case_insensitive(base_dir: str, keyword: str) -> List[str]:
    """Recursively search for CSV files containing a keyword (case-insensitive)."""
    found = []
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.lower().endswith(".csv") and keyword.lower() in f.lower():
                found.append(os.path.join(root, f))
    return found


def combine_compound_data(
    sample_files: List[str],
    base_dir: str,
    output_path: str,
    qc_files: Optional[List[str]] = None,
    env_files: Optional[List[str]] = None,
    blank_files: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Combines GC–MS compound data from sample, QC, env, and blank CSV files into one DataFrame.
    Adds columns: FileType, FileName
    """

    def load_file(file_path: str, file_type: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(file_path)
            df["FileType"] = file_type
            df["FileName"] = os.path.basename(file_path)
            return df
        except Exception as e:
            print(f"Error reading {file_type} file: {file_path}\n{e}")
            return pd.DataFrame()

    all_data = []
    for f in sample_files:
        all_data.append(load_file(f, "sample"))
    if qc_files:
        for f in qc_files:
            all_data.append(load_file(f, "qc"))
    if env_files:
        for f in env_files:
            all_data.append(load_file(f, "env"))
    if blank_files:
        for f in blank_files:
            all_data.append(load_file(f, "blank"))

    combined_df = pd.concat(all_data, ignore_index=True)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    combined_df.to_csv(output_path, index=False)

    print(f"\nCombined file saved to: {output_path}")
    if "FileType" in combined_df.columns:
        print(f"\nFileType counts:\n{combined_df['FileType'].value_counts()}")
    print("\nPreview:")
    print(combined_df.head())
    return combined_df




def plot_filetype_counts(df: pd.DataFrame, column: str = "FileType", title: str = "Number of Files per Type") -> None:
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")

    counts = df[column].value_counts()
    plt.figure(figsize=(6, 4))
    plt.bar(counts.index, counts.values)
    plt.title(title)
    plt.xlabel("File Type")
    plt.ylabel("Count")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()
    print(counts)


# Contamination filtering

def filter_real_compounds(compound_df: pd.DataFrame, rt_tolerance: float = 0.4) -> pd.DataFrame:
    """
    Removes sample compounds that also appear in blank/env/qc within RT tolerance and same compound name.
    Needs: 'Component RT', 'Compound Name', 'FileType'
    """
    sample_df = compound_df[compound_df["FileType"] == "sample"].copy()
    contam_df = compound_df[compound_df["FileType"].isin(["blank", "env", "qc"])].copy()

    print(f"Sample compounds before filtering: {len(sample_df)}")

    sample_df["Component RT"] = pd.to_numeric(sample_df["Component RT"], errors="coerce")
    contam_df["Component RT"] = pd.to_numeric(contam_df["Component RT"], errors="coerce")

    sample_df = sample_df.sort_values("Component RT")
    contam_df = contam_df.sort_values("Component RT")

    contaminated_idx = set()
    for _, c in contam_df.iterrows():
        rt = c["Component RT"]
        name = str(c["Compound Name"])
        matches = sample_df[
            (np.abs(sample_df["Component RT"] - rt) <= rt_tolerance)
            & (sample_df["Compound Name"].astype(str).str.lower() == name.lower())
        ].index
        contaminated_idx.update(matches.tolist())

    filtered_df = sample_df.drop(index=list(contaminated_idx))
    print(f"Sample compounds after filtering: {len(filtered_df)}")
    print(f"Removed {len(contaminated_idx)} sample peaks (ΔRT ≤ {rt_tolerance})")
    return filtered_df




def plot_rt_distribution(compound_df: pd.DataFrame, real_compounds: pd.DataFrame, bins: int = 40) -> None:
    plt.figure(figsize=(6, 4))
    plt.hist(
        compound_df[compound_df["FileType"] == "sample"]["Component RT"],
        bins=bins, alpha=0.5, label="Before filtering"
    )
    plt.hist(real_compounds["Component RT"], bins=bins, alpha=0.7, label="After filtering")
    plt.xlabel("Retention Time (min)")
    plt.ylabel("Count")
    plt.title("RT Distribution Before and After Filtering")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()




def plot_filtering_effect(compound_df: pd.DataFrame, real_compounds: pd.DataFrame, rt_tolerance: float = 0.5) -> None:
    before = compound_df[compound_df["FileType"] == "sample"].shape[0]
    after = real_compounds.shape[0]
    plt.figure(figsize=(5, 4))
    plt.bar(["Before filtering", "After filtering"], [before, after])
    plt.ylabel("Number of sample compound entries")
    plt.title(f"Effect of Contamination Filtering (RT ±{rt_tolerance} min)")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()
    print(f"Before: {before}, After: {after}")


#  Matching compounds to peaks + aroma extraction
def match_compounds_to_peaks(
    compound_df: pd.DataFrame,
    samples_filtered_path: str,
    output_path: str,
    rt_tolerance: float = 0.4
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Match NIST-identified compounds with GC–MS peaks based on retention time (RT).
    Expects samples_filtered CSV has columns: Variety, Peak, tR_best, m/z, Intensity_corrected
    """

    samples_filtered = pd.read_csv(samples_filtered_path)

    # Create Variety from FileName
    compound_df = compound_df.copy()
    compound_df["Variety"] = compound_df["FileName"].astype(str).str.replace(".csv", "", regex=False).str.strip()

    def find_closest_peak(rt, peaks_rt, tol=rt_tolerance):
        diffs = np.abs(peaks_rt - rt)
        if diffs.empty:
            return None
        min_diff = diffs.min()
        return diffs.idxmin() if min_diff <= tol else None

    matches, unmatched = [], []

    for _, row in compound_df.iterrows():
        variety = row["Variety"]
        sample_data = samples_filtered[samples_filtered["Variety"] == variety]
        if sample_data.empty:
            unmatched.append({**row, "Reason": "No rows for this variety"})
            continue

        compound_rt = row.get("Component RT", np.nan)
        peak_idx = find_closest_peak(compound_rt, sample_data["tR_best"], tol=rt_tolerance)

        if peak_idx is None:
            unmatched.append({**row, "Reason": "No matching RT"})
            continue

        peak_row = sample_data.loc[peak_idx]
        matches.append({
            "Compound Name": row.get("Compound Name"),
            "FileName": row.get("FileName"),
            "Variety": variety,
            "Component RT": compound_rt,
            "Peak": peak_row.get("Peak"),
            "tR_best": peak_row.get("tR_best"),
            "m/z": peak_row.get("m/z"),
            "Intensity": peak_row.get("Intensity_corrected", peak_row.get("Intensity", np.nan)),
        })

    compound_intensity_df = pd.DataFrame(matches)
    unmatched_df = pd.DataFrame(unmatched)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    compound_intensity_df.to_csv(output_path, index=False)
    print(f"\nMatched compounds saved to: {output_path}")

    return compound_intensity_df, unmatched_df






def match_potato_aromas_strict(
    combined_df,
    samples_filtered_path,
    out_dir="/Volumes/bmqg/default_bronze/fatemeh/aroma_flavor_project/Results",
    paths=None,
    rt_tol=0.4,
    mz_tol=0.5,
):
    """
    COMPLETE NOTEBOOK LOGIC AS A FUNCTION (module-friendly)

    Parameters
    ----------
    combined_df : pd.DataFrame
        Your combined dataframe with at least:
        - Compound Name (or Name/Compound)
        - Component RT (or RT/ComponentRT)
        - Base Peak MZ (or MZ/m/z)
        - optional FileType (will filter to FileType == 'sample' if present)
    samples_filtered_path : str
        CSV path for peaks input.
        Works with:
          - LONG: columns include Peak,tR_best,m/z + (Intensity_corrected or Intensity/Area)
                  + (Variety or Filename or Sample or SampleCol)
          - WIDE: columns Peak,tR_best,m/z + many variety columns
    out_dir : str
        Output directory (must be writable)
    paths : dict-like or None
        If provided, will use:
          - paths.get("rt_tol", rt_tol)
          - paths.get("mz_tol", mz_tol)
    rt_tol, mz_tol : float
        Default tolerances if paths not provided (or missing keys)

    Returns
    -------
    dict with:
      - matched_per_variety (pd.DataFrame)
      - aroma_matrix (pd.DataFrame)
      - ref (pd.DataFrame)
      - paths (dict of output file paths)
    """

    #  Output directory
    # Potato aroma list 
    POTATO_AROMAS_CORE = [
        "Hexanal","Heptanal","Octanal","Nonanal","Decanal",
        "Pentanal","Propanal",
        "2-Methylbutanal","3-Methylbutanal","Butanal, 3-methyl-",
        "4-Heptenal","2-Pentenal","2-Hexenal","2-Octenal","2-Nonenal","2-Heptenal",
        "2-Trans-nonenal","2-Trans-octenal",
        "2,4-Heptadienal","2,4-Octadienal",
        "(E, E)-2,4-Nonadienal","(E, Z)-2,4-Decadienal","(E, E)-2,4-Decadienal","2,4-Nonadienal",
        "Benzaldehyde","2-methyl-Benzaldehyde","4-methyl-Benzaldehyde",
        "Benzeneacetaldehyde","Phenylacetaldehyde",

        "1-Heptanol","1-Nonanol","1-Octen-3-ol","1-Pentanol","3-Nonen-1-ol",
        "Hexanol","Phenylethyl alcohol","1-Phenylethanol",
        "2-Methyl-1-propanol","3-Methyl-1-butanol","Ethanol","Benzyl alcohol",

        "2-Heptanone","2-Nonanone","3-Heptanone","3-Octanone","3-Octen-2-one","1-Penten-3-one",
        "2,3-Butanedione","2,3-Pentanedione","Acetophenone","4-Methyl-2-pentanone",
        "(E, E)-3,5-Octadien-2-one","3,5-Octadien-2-one",

        "Acetic acid","Butanoic acid","Hexanoic acid","Isobutyric acid","Isovaleric acid",
        "Propanoic acid","Octanoic acid","2-Octenoic acid","Decanoic acid","nonanoic acid",

        "Methanethiol","Methional",
        "Dimethyl sulfide","Dimethyl disulfide","Dimethyl trisulfide","Dimethyl tetrasulfide",
        "Carbon disulfide","Hydrogen sulfide",
        "2-Methyl-3-furanthiol","3-Methylthiopropanal","3-(Methylthio)propanal",
        "2-Acetylthiazole","2-Methyl-3-thiazoline","2-Methylthiazole","Allyl methyl sulfide",

        "Ethyl acetate","Isoamyl acetate","Ethyl butanoate","Ethyl hexanoate","Methyl butanoate",
        "Ethyl 2-methylbutanoate","Ethyl propanoate","Methyl propanoate","Ethyl pentanoate",

        "2-Acetylpyrazine","2-Ethyl-3,5-dimethylpyrazine",
        "2-Isobutyl-3-methoxypyrazine","2-Isopropyl-3-methoxypyrazine",
        "2-Ethyl-3-methylpyrazine","Trimethylpyrazine","Tetramethylpyrazine",
        "2,5-Dimethylpyrazine","2,6-Dimethylpyrazine","2-Methylpyrazine",

        "Furfural","5-Methylfurfural","3-Furaldehyde",
        "methyl-Cyclopentane","n-Hexane",
        "2-n-Butyl furan","cis-2-(2-Pentenyl) furan",
        "2-pentyl-Furan","Furan, 2-pentyl-","2-Pentylfuran","2-(2-propenyl)-Furan",
        "1,2-dimethoxy-Benzene","2-methoxy-Phenol","Methyl salicylate","Mequinol",

        "Copaene","trans-beta-Ionone","Linalool","alpha-Terpineol",".alpha.-Terpineol",
        "Dimethyl phthalate",
        "Undecanal","Dodecanal",
        "(E)-2-Hexenal","(E)-2-Heptenal","(E)-2-Decenal",
        "2-Octenal, (E)-","2-Nonenal, (E)-",
        "2(3H)-Furanone, dihydro-5-pentyl-",
    ]

    POTATO_AROMAS_OPTIONAL = [
        "Phenol",
        "Benzoic acid",
        "Benzoic acid, methyl ester",

        "Eucalyptol",
        "Limonene",
        "3-Carene",

        "Tridecanal",
        "Tetradecanal",
        "2-Undecenal",
        "2-Tridecenal",

        "6-Methyl-5-hepten-2-one",
        "2-Methylpropanal",
        "1-Octyn-3-ol",
        "3-Methylfuran",
        "2-Ethylfuran",
        "2-Acetyl-5-methylfuran",

        "Acetic acid, ethenyl ester",
        "Acetic acid, methyl ester",
        "Butanoic acid, butyl ester",
        "Hexanoic acid, methyl ester",
        "Octanoic acid, methyl ester",
        "Dodecanoic acid",
        "Dodecanoic acid, 1-methylethyl ester",

        "1-Hexanol, 2-ethyl-",
        "Styrene",
        "n-Hexadecanoic acid",
        "Ethanol, 2-phenoxy-",
        "Tridecane",
        "Tetradecane",
    
        "2-Propanone, 1-methoxy-",
        "2-Decanone",
        "3-Heptanone, 4-methyl-",
        "Methyl 8-oxooctanoate",
        "Nonanoic acid, 9-oxo-, methyl ester",
    ]

    potato_aromas = list(OrderedDict.fromkeys(POTATO_AROMAS_CORE + POTATO_AROMAS_OPTIONAL))

    ALIASES = {
        "Phenylacetaldehyde": ["benzeneacetaldehyde"],
        "Phenylethyl alcohol": ["2phenylethanol", "phenethyl alcohol", "phenylethanol", "2-phenylethanol"],
        "alpha-Terpineol": [".alpha.-terpineol", "alphaterpineol", "alpha terpineol"],
        "Furan, 2-pentyl-": ["2pentylfuran", "2-pentylfuran", "2-pentyl furan", "furan, 2-pentyl"],

        "Dimethyl disulfide": ["disulfide, dimethyl"],
        "Dimethyl sulfide": ["sulfide, dimethyl"],
        "Dimethyl trisulfide": ["trisulfide, dimethyl"],

        "Butanal, 3-methyl-": ["3-methylbutanal", "butanal,3-methyl-", "3-methyl butanal"],
        "2-Octenal, (E)-": ["(e)-2-octenal", "2-octenal (e)", "2-octenal, (e)-"],
        "2-Nonenal, (E)-": ["(e)-2-nonenal", "2-nonenal (e)", "2-nonenal, (e)-"],

        "2-Methylbutanal": ["butanal, 2-methyl-", "2-methyl butanal", "2-methylbutyraldehyde"],
        "3-Methylbutanal": ["butanal, 3-methyl-", "isovaleraldehyde", "3-methyl butanal"],

        "6-Methyl-5-hepten-2-one": ["5-hepten-2-one, 6-methyl-", "sulcatone"],
        "2-Methylpropanal": ["propanal, 2-methyl-", "isobutyraldehyde"],

        "2-Undecenal": ["(e)-2-undecenal", "2-undecenal (e)", "2-undecenal (e)-", "2-undecenal, (e)-"],
        "2-Tridecenal": ["(e)-2-tridecenal", "2-tridecenal (e)", "2-tridecenal (e)-", "2-tridecenal, (e)-"],

        "3-Methylfuran": ["furan, 3-methyl-"],
        "2-Ethylfuran": ["furan, 2-ethyl-"],

        "Acetic acid, ethenyl ester": ["vinyl acetate"],
        "Acetic acid, methyl ester": ["methyl acetate"],
        "Butanoic acid, butyl ester": ["butyl butanoate"],
        "Hexanoic acid, methyl ester": ["methyl hexanoate"],
        "Octanoic acid, methyl ester": ["methyl octanoate"],
        "Dodecanoic acid, 1-methylethyl ester": ["isopropyl dodecanoate"],

        "1-Hexanol, 2-ethyl-": ["2-ethyl-1-hexanol", "2-ethylhexanol"],
        "n-Hexadecanoic acid": ["palmitic acid", "hexadecanoic acid"],
        "Ethanol, 2-phenoxy-": ["phenoxyethanol", "2-phenoxyethanol"],

        "Benzoic acid, methyl ester": ["methyl benzoate"],
        "Eucalyptol": ["1,8-cineole", "cineole"],
        "Limonene": ["d-limonene", "dl-limonene"],
        "3-Carene": ["delta-3-carene", "δ-3-carene"],

        "2-Propanone, 1-methoxy-": ["1-methoxy-2-propanone", "propylene glycol monomethyl ether"],
    }

    #  Helpers (unchanged)
    def strip_stereo(s: str) -> str:
        s = "" if pd.isna(s) else str(s)
        s = re.sub(r"\(\s*[EeZz]\s*,\s*[EeZz]\s*\)", "", s)
        s = re.sub(r"\(\s*[EeZz]\s*\)", "", s)
        s = re.sub(r"\b(cis|trans)\b\s*-?", "", s, flags=re.IGNORECASE)
        return s

    def normalize(s: str) -> str:
        s = strip_stereo(s).lower().strip()
        return re.sub(r"[^a-z0-9]+", "", s)

    by_norm = defaultdict(list)
    for x in potato_aromas:
        by_norm[normalize(x)].append(x)

    def pick_canonical(names):
        prefs = ["(E, E)", "(E, Z)", "(E)-", "alpha-", ".alpha.-", ", (E)-"]
        for p in prefs:
            for n in names:
                if p in n:
                    return n
        return sorted(names, key=lambda s: (len(s), s))[0]

    canon_norm = {k: pick_canonical(v) for k, v in by_norm.items()}

    syn_to_canon = {}
    for norm_key, names in by_norm.items():
        canon = canon_norm[norm_key]
        for n in names:
            syn_to_canon[normalize(n)] = canon

    for canon, alts in ALIASES.items():
        for a in alts:
            syn_to_canon[normalize(a)] = canon

    canon_keys = list(canon_norm.keys())

    def map_to_canonical(name: str, cutoff: float = 0.86):
        n = normalize(name)
        if n in syn_to_canon:
            return syn_to_canon[n], "exact/alias"

        for ck in canon_keys:
            if len(ck) >= 6 and ck in n:
                return canon_norm[ck], "substring"

        hit = get_close_matches(n, canon_keys, n=1, cutoff=cutoff)
        if hit:
            return canon_norm[hit[0]], "fuzzy"
        return None, "no"

    #  Load peaks (LONG + WIDE)
  
    peaks_raw = pd.read_csv(samples_filtered_path, sep=None, engine="python").copy()
    peaks_raw.columns = [str(c).strip() for c in peaks_raw.columns]

    rt_col = "tR_best"
    mz_col = "m/z"
    for c in ["Peak", rt_col, mz_col]:
        if c not in peaks_raw.columns:
            raise KeyError(f"peaks must contain '{c}'. Available: {peaks_raw.columns.tolist()}")

    long_int_candidates = ["Intensity_corrected", "Intensity", "Area", "peak_Intensity"]
    id_candidates = ["Variety", "Filename", "Sample", "SampleCol"]

    is_long = any(c in peaks_raw.columns for c in long_int_candidates) and any(
        c in peaks_raw.columns for c in id_candidates
    )

    if is_long:
        INT_COL = next(c for c in long_int_candidates if c in peaks_raw.columns)

        if "Variety" in peaks_raw.columns:
            ID_COL = "Variety"
        elif "Filename" in peaks_raw.columns:
            ID_COL = "Filename"
        elif "Sample" in peaks_raw.columns:
            ID_COL = "Sample"
        else:
            ID_COL = "SampleCol"

        peaks = peaks_raw.copy()
        peaks["Variety"] = peaks[ID_COL].astype(str).str.strip()

        if ID_COL == "Filename":
            peaks["Variety"] = peaks["Variety"].map(lambda x: os.path.splitext(os.path.basename(x))[0])

    else:
        id_vars = ["Peak", rt_col, mz_col]
        value_vars = [c for c in peaks_raw.columns if c not in id_vars]
        if not value_vars:
            raise ValueError(
                "WIDE peaks table has no variety intensity columns. "
                "Your file might be LONG but missing Intensity_corrected."
            )

        peaks = peaks_raw.melt(
            id_vars=id_vars,
            value_vars=value_vars,
            var_name="Variety",
            value_name="peak_Intensity",
        )
        INT_COL = "peak_Intensity"

    peaks[rt_col] = pd.to_numeric(peaks[rt_col], errors="coerce")
    peaks[mz_col] = pd.to_numeric(peaks[mz_col], errors="coerce")
    peaks[INT_COL] = pd.to_numeric(peaks[INT_COL], errors="coerce")

    peaks = peaks.dropna(subset=["Variety", rt_col, mz_col, INT_COL]).copy()
    peaks = peaks[peaks[INT_COL] > 0].copy()
    peaks["_mz_round"] = peaks[mz_col].round(0)

    # Prepare compound reference from combined_df

    cdf = combined_df.copy()
    cdf.columns = [str(c).strip() for c in cdf.columns]

    compound_rt_col = next((c for c in ["Component RT", "RT", "ComponentRT"] if c in cdf.columns), None)
    compound_mz_col = next((c for c in ["Base Peak MZ", "BasePeakMZ", "MZ", "m/z"] if c in cdf.columns), None)
    name_col = next((c for c in ["Compound Name", "Compound", "Name"] if c in cdf.columns), None)

    if compound_rt_col is None or compound_mz_col is None or name_col is None:
        raise KeyError(
            "combined_df must contain Name + RT + MZ columns.\n"
            f"Detected name={name_col}, rt={compound_rt_col}, mz={compound_mz_col}\n"
            f"Available columns: {cdf.columns.tolist()}"
        )

    if "FileType" in cdf.columns:
        cdf = cdf[cdf["FileType"].eq("sample")].copy()

    cdf[name_col] = cdf[name_col].astype(str)
    cdf[compound_rt_col] = pd.to_numeric(cdf[compound_rt_col], errors="coerce")
    cdf[compound_mz_col] = pd.to_numeric(cdf[compound_mz_col], errors="coerce")
    cdf = cdf.dropna(subset=[name_col, compound_rt_col, compound_mz_col]).copy()

    mapped = cdf[name_col].map(map_to_canonical)
    cdf["Compound Name"] = [m[0] for m in mapped]
    cdf["_match_type"] = [m[1] for m in mapped]

    unmapped = cdf[cdf["Compound Name"].isna()].copy()
    unmapped_path = os.path.join(out_dir, "compound_df_unmapped_names.csv")
    if not unmapped.empty:
        unmapped[[name_col]].drop_duplicates().to_csv(unmapped_path, index=False)

    cdf = cdf[cdf["Compound Name"].notna()].copy()

    ref = (cdf.groupby("Compound Name", as_index=False)
           .agg(rt_ref=(compound_rt_col, "median"),
                mz_ref=(compound_mz_col, "median"),
                n_hits=("Compound Name", "size")))

    ref_path = os.path.join(out_dir, "aroma_reference_rt_mz.csv")
    ref.to_csv(ref_path, index=False)

    missing_list = [x for x in potato_aromas if normalize(x) not in set(ref["Compound Name"].map(normalize))]
    missing_ref_path = os.path.join(out_dir, "missing_aromas_from_reference.csv")
    pd.DataFrame({"Compound Name": missing_list}).to_csv(missing_ref_path, index=False)

    #  Strict matching per Variety (RT + MZ)

    if paths is not None:
        rt_tol_use = float(paths.get("rt_tol", rt_tol))
        mz_tol_use = float(paths.get("mz_tol", mz_tol))
    else:
        rt_tol_use = float(rt_tol)
        mz_tol_use = float(mz_tol)

    rows = []
    for variety, pv in peaks.groupby("Variety", sort=False):
        pv = pv.copy()

        for _, r in ref.iterrows():
            cname = r["Compound Name"]
            rt_ref = float(r["rt_ref"])
            mz_ref = float(r["mz_ref"])
            mz_round_ref = round(mz_ref)

            cand = pv[np.abs(pv["_mz_round"] - mz_round_ref) <= 1].copy()
            if cand.empty:
                continue

            cand["_rt_diff"] = (cand[rt_col] - rt_ref).abs()
            cand["_mz_diff"] = (cand[mz_col] - mz_ref).abs()

            strict = cand[
                (cand["_rt_diff"] <= rt_tol_use) &
                (cand["_mz_diff"] <= mz_tol_use)
            ].copy()

            if strict.empty:
                continue

            strict["_score"] = (
                strict["_rt_diff"] / max(rt_tol_use, 1e-9) +
                strict["_mz_diff"] / max(mz_tol_use, 1e-9)
            )

            best = strict.sort_values("_score").iloc[0]

            rows.append({
                "Variety": variety,
                "Compound Name": cname,
                "tR_best": float(best[rt_col]),
                "m/z": float(best[mz_col]),
                "peak_Intensity": float(best[INT_COL]),
                "_rt_diff": float(best["_rt_diff"]),
                "_mz_diff": float(best["_mz_diff"]),
            })

    matched_per_variety = pd.DataFrame(rows)
    matched_out = os.path.join(out_dir, "matched_per_variety_STRICT_ONLY.csv")
    matched_per_variety.to_csv(matched_out, index=False)


    if matched_per_variety.empty:
        aroma_matrix = pd.DataFrame()
    else:
        aroma_matrix = (matched_per_variety
                        .pivot_table(index="Variety",
                                     columns="Compound Name",
                                     values="peak_Intensity",
                                     aggfunc="max",
                                     fill_value=0)
                        .reset_index())

    matrix_out = os.path.join(out_dir, "aroma_intensity_matrix_STRICT.csv")
    aroma_matrix.to_csv(matrix_out, index=False)

    return {
        "matched_per_variety": matched_per_variety,
        "aroma_matrix": aroma_matrix,
        "ref": ref,
        "paths": {
            "out_dir": out_dir,
            "ref_path": ref_path,
            "missing_ref_path": missing_ref_path,
            "unmapped_path": unmapped_path if os.path.exists(unmapped_path) else None,
            "matched_out": matched_out,
            "matrix_out": matrix_out,
        }}








def assign_aroma_descriptions_and_families(aroma_sample_df):
    """
    Assign aroma descriptions and chemical families to each compound in the DataFrame.

    Parameters
    ----------
    aroma_sample_df : pd.DataFrame
        Must contain a column named 'Compound Name'.

    Returns
    -------
    aroma_sample_df : pd.DataFrame
        Updated DataFrame with added 'Aroma_Description' and 'Chemical_Family' columns.
    aroma_descriptions : dict
        Mapping of compounds to sensory aroma notes.
    chemical_family_map : dict
        Mapping of compounds to chemical family classification.
    """
  

    if "Compound Name" not in aroma_sample_df.columns:
        raise KeyError("Input DataFrame must include a 'Compound Name' column.")


    aroma_descriptions = {
        # Aldehydes
        "Propanal": "Sharp, pungent, green",
        "Butanal, 3-methyl-": "Malty, chocolate, cocoa-like (Maillard reaction product)",
        "Pentanal": "Fatty, green, pungent",
        "2-Methylbutanal": "Malty, chocolate, roasted (amino acid-derived)",
        "3-Methylbutanal": "Malty, chocolate, roasted (Leu-derived Strecker aldehyde)",
        "Hexanal": "Green, grassy, fresh-cut potato",
        "Heptanal": "Fatty, citrus-like, slightly green",
        "Octanal": "Citrus, fruity, green",
        "Nonanal": "Waxy, citrus, floral (lipid oxidation product)",
        "Decanal": "Citrus, fatty, waxy, potato-like",
        "2-Trans-nonenal": "Cardboard, fatty, old potato note",
        "2-Trans-octenal": "Green, fatty, cucumber-like",
        "4-Heptenal": "Fatty, oily, green",
        "2,4-Decadienal": "Fatty, fried, potato-chip-like",
        "Phenylacetaldehyde": "Honey-like, floral, sweet potato note",
        "Benzaldehyde": "Almond, cherry, sweet",

        # Alcohols
        "1-Heptanol": "Floral, herbal, fatty",
        "1-Nonanol": "Waxy, floral, oily",
        "1-Octen-3-ol": "Mushroom, earthy, raw potato",
        "1-Phenylethanol": "Floral, rose, sweet",
        "Hexanol": "Green, grassy, herbaceous",
        "2-Methyl-1-propanol": "Fusel, solvent-like",
        "3-Methyl-1-butanol": "Whiskey-like, malty, fusel",
        "Ethanol": "Alcoholic, sweet",

        # Ketones
        "2-Heptanone": "Fruity, blue cheese, creamy",
        "2-Nonanone": "Fatty, floral, soapy",
        "3-Heptanone": "Cheesy, earthy, fruity",
        "3-Octanone": "Mushroom, earthy, fatty",
        "2,3-Butanedione": "Buttery, creamy, cooked potato",
        "2,3-Pentanedione": "Creamy, buttery, sweet",
        "Acetophenone": "Floral, sweet, almond-like",
        "2-Propanone, 1-methoxy-": "Solvent, ether-like",
        "4-Methyl-2-pentanone": "Solvent, fruity, pungent",
        "2-Undecanone": "Fatty, waxy, green (potato tuber-like note)",

        # Sulfur compounds
        "Methanethiol": "Sulfurous, cabbage, cooked potato",
        "Methional": "Cooked potato, meaty, sulfurous (key potato note)",
        "Dimethyl disulfide": "Garlic, onion, sulfurous",
        "Dimethyl trisulfide": "Cabbage, onion, sulfurous",
        "Dimethyl tetrasulfide": "Strongly sulfurous, boiled vegetable",
        "Carbon disulfide": "Sulfurous, chemical",
        "Hydrogen sulfide": "Rotten egg, sulfurous",
        "2-Methyl-3-furanthiol": "Meaty, roasted, savory",
        "3-Methylthiopropanal": "Cooked potato, malty, onion-like",
        "2-Acetylthiazole": "Roasted, popcorn-like",
        "2-Methylthiazole": "Nutty, roasted",
        "2-Methyl-3-thiazoline": "Meaty, roasted",
        "3-(Methylthio)propanal": "Boiled potato, onion-like",
        "Allyl methyl sulfide": "Garlic, onion-like",

        # Acids
        "Acetic acid": "Vinegar-like, sour",
        "Butanoic acid": "Rancid, cheesy, sweaty",
        "Hexanoic acid": "Fatty, sweaty, rancid",
        "Isobutyric acid": "Cheesy, rancid, sour",
        "Isovaleric acid": "Sweaty, cheesy, foot-like",
        "Octanoic acid": "Fatty, soapy, rancid",
        "Propanoic acid": "Sour, pungent",

        # Esters
        "Ethyl acetate": "Fruity, solvent-like",
        "Isoamyl acetate": "Banana, fruity, sweet",
        "Ethyl butanoate": "Pineapple, fruity, sweet",
        "Ethyl hexanoate": "Apple, pineapple, sweet",
        "Methyl butanoate": "Apple, fruity, sweet",
        "Ethyl 2-methylbutanoate": "Apple, sweet, fruity",
        "Ethyl propanoate": "Fruity, rum-like",
        "Methyl propanoate": "Fruity, sweet",
        "Ethyl pentanoate": "Fruity, sweet",

        # Pyrazines
        "2-Acetylpyrazine": "Nutty, roasted, popcorn-like",
        "2-Ethyl-3,5-dimethylpyrazine": "Nutty, earthy, roasted",
        "2-Isobutyl-3-methoxypyrazine": "Green bell pepper, earthy",
        "2-Isopropyl-3-methoxypyrazine": "Green, earthy, potato peel-like",
        "2-Ethyl-3-methylpyrazine": "Nutty, roasted, earthy",
        "Trimethylpyrazine": "Roasted, cocoa, nutty",
        "Tetramethylpyrazine": "Roasted, nutty, cocoa",
        "2,5-Dimethylpyrazine": "Roasted, nutty",
        "2,6-Dimethylpyrazine": "Roasted, earthy",
        "2-Methylpyrazine": "Roasted, nutty, chocolate-like",

        # Aromatics/Furans
        "Benzeneacetaldehyde": "Honey, floral, sweet",
        "Benzyl alcohol": "Floral, sweet, mild",
        "Phenylethyl alcohol": "Rose-like, floral, sweet",
        "Furfural": "Sweet, almond, caramel",
        "5-Methylfurfural": "Caramel, baked, sweet"
    }

   
    # 2) Normalization helpers (handles (E)-, cis/trans, punctuation)
    def _strip_stereo(x: str) -> str:
        x = "" if pd.isna(x) else str(x)
        x = re.sub(r"\(\s*[EeZz]\s*,\s*[EeZz]\s*\)", "", x)
        x = re.sub(r"\(\s*[EeZz]\s*\)", "", x)
        x = re.sub(r"\b(cis|trans)\b\s*-?", "", x, flags=re.IGNORECASE)
        return x

    def _norm(x: str) -> str:
        x = _strip_stereo(x).lower().strip()
        return re.sub(r"[^a-z0-9]+", "", x)

    # normalized lookup for descriptions
    aroma_desc_norm = {_norm(k): v for k, v in aroma_descriptions.items()}


    #  Chemical family rules (covers ALL your potato aroma list)
    # Order matters: first match wins.
    FAMILY_RULES = [
        # Sulfur family (strong signals)
        ("Sulfur Compound", [
            "methanethiol", "methional", "dimethylsulfide", "dimethyldisulfide",
            "dimethyltrisulfide", "dimethyltetrasulfide", "carbondisulfide",
            "hydrogensulfide", "furanthiol", "methylthio", "thiazole", "thiazoline",
            "allylmethylsulfide"
        ]),

        # Acids
        ("Acid", [
            "acid"
        ]),

        # Esters (common suffixes)
        ("Ester", [
            "acetate", "butanoate", "propanoate", "pentanoate", "hexanoate", "benzoate",
            "salicylate", "ester"
        ]),

        # Aldehydes (suffixes + known tokens)
        ("Aldehyde", [
            "al", "aldehyde", "furaldehyde", "hexenal", "heptenal", "octenal", "nonenal", "decenal",
            "undecenal", "tridecenal", "pentanal", "hexanal", "heptanal", "octanal", "nonanal", "decanal",
            "undecanal", "dodecanal", "tridecanal", "tetradecanal"
        ]),

        # Alcohols
        ("Alcohol", [
            "ol", "alcohol", "ethanol", "hexanol", "heptanol", "nonanol", "phenylethanol", "benzylalcohol"
        ]),

        # Ketones
        ("Ketone", [
            "one", "butanedione", "pentanedione", "acetophenone", "propanone"
        ]),

        # Pyrazines
        ("Pyrazine", [
            "pyrazine", "methoxypyrazine"
        ]),

        # Furans / aromatics / phenolics / terpenes / hydrocarbons
        ("Aromatic/Furan", [
            "furfural", "furan", "benzene", "phenol", "styrene", "methoxyphenol", "dimethoxybenzene",
            "linalool", "terpineol", "eucalyptol", "limonene", "carene", "ionone",
            "copaene", "hexane", "cyclopentane", "tridecane", "tetradecane", "hexadecane",
            "phthalate"
        ]),
    ]

    def classify_family(compound_name: str) -> str:
        n = _norm(compound_name)

        # Special-case: some "al" endings could be alcohols; keep it conservative.
        # Use explicit known aldehyde tokens above; otherwise fall through.
        for fam, keys in FAMILY_RULES:
            for k in keys:
                if k in n:
                    # A tiny guard: "ol" appears in many; don't misclassify aldehydes as alcohol
                    if fam == "Alcohol" and ("al" in n and n.endswith("al")):
                        continue
                    return fam
        return "Other"

    #  Default description per family 
    DEFAULT_DESC_BY_FAMILY = {
        "Aldehyde": "Green/fatty notes; lipid-oxidation related",
        "Alcohol": "Green/floral notes; fresh/vegetal nuances",
        "Ketone": "Buttery/creamy/mushroom-like notes",
        "Sulfur Compound": "Sulfurous cooked-potato / onion-cabbage notes",
        "Acid": "Sour/rancid/cheesy notes",
        "Ester": "Fruity/sweet notes",
        "Pyrazine": "Roasted/nutty/earthy notes",
        "Aromatic/Furan": "Sweet/toasty/spicy/floral notes (aromatic/furan/terpene/hydrocarbon)",
        "Other": "General aroma compound (no curated descriptor yet)",
    }

    #  Apply mappings
    comp_series = aroma_sample_df["Compound Name"].astype(str)

    aroma_sample_df["Chemical_Family"] = comp_series.map(classify_family)

    def get_description(name: str) -> str:
        n = _norm(name)
        if n in aroma_desc_norm:
            return aroma_desc_norm[n]
        fam = classify_family(name)
        return DEFAULT_DESC_BY_FAMILY.get(fam, "General aroma compound")

    aroma_sample_df["Aroma_Description"] = comp_series.map(get_description)

    # also return explicit maps for your export / reproducibility
    chemical_family_map = {c: classify_family(c) for c in comp_series.unique()}
    # build a full description map (curated + defaults)
    full_desc_map = {c: get_description(c) for c in comp_series.unique()}

    return aroma_sample_df, full_desc_map, chemical_family_map


# Public pipeline

def run_full_aroma_matching_pipeline(
    samples_filtered_path: str,
    compound_df: pd.DataFrame,
    rt_tol: float = 0.4,
    mz_tol: float = 0.5,
) -> pd.DataFrame:

    potato_aromas = build_potato_aroma_list()
    map_to_canonical = build_canonical_mapper(potato_aromas, ALIASES)

    peaks, mz_col, int_col = load_peaks(samples_filtered_path)
    ref = build_reference_from_compound_df(compound_df, map_to_canonical)

    return match_aromas_per_variety_strict(
        peaks, ref, mz_col, int_col, rt_tol, mz_tol
    )

def infer_chemical_family_from_name(name: str) -> str:
    n = name.lower()

    if any(x in n for x in ["al", "enal", "anal", "aldehyde"]):
        return "Aldehyde"

    if any(x in n for x in ["ol", "alcohol"]):
        return "Alcohol"

    if any(x in n for x in ["one", "ketone"]):
        return "Ketone"

    if any(x in n for x in ["acid"]):
        return "Acid"

    if any(x in n for x in ["ate", "ester"]):
        return "Ester"

    if any(x in n for x in ["thiol", "sulfide", "thio", "sulfur"]):
        return "Sulfur Compound"

    if any(x in n for x in ["pyrazine"]):
        return "Pyrazine"

    if any(x in n for x in ["furan", "furfural"]):
        return "Aromatic/Furan"

    if any(x in n for x in ["benz", "phenyl", "aromatic"]):
        return "Aromatic/Furan"

    return "Other"







def plot_aroma_family_distribution_unique(aroma_df: pd.DataFrame, save_path: Optional[str] = None) -> pd.Series:
    """
    Canonical version (ONLY ONE) of aroma family distribution plot.
    Counts UNIQUE compounds per family.
    """
    if "Compound Name" not in aroma_df.columns or "Chemical_Family" not in aroma_df.columns:
        raise KeyError("Need 'Compound Name' and 'Chemical_Family' columns.")

    family_counts = (
        aroma_df[["Compound Name", "Chemical_Family"]]
        .drop_duplicates()
        .groupby("Chemical_Family")
        .size()
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(family_counts.index, family_counts.values)
    ax.invert_yaxis()
    ax.set_title("Distribution of Identified Aroma Compound Families", fontsize=14, fontweight="bold")
    ax.set_xlabel("Number of Unique Compounds")
    ax.set_ylabel("Chemical Family")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
    return family_counts







def plot_outlier_vs_normal_aromas(aroma_matrix, outliers, chemical_family_map, output_dir="aroma_cluster_results"):
    
    os.makedirs(output_dir, exist_ok=True)

    # Prepare Data 
    df = aroma_matrix.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    if "variety" not in df.columns:
        df.rename(columns={df.columns[0]: "variety"}, inplace=True)

    df["group"] = df["variety"].apply(lambda x: "Outlier" if x.upper() in [o.upper() for o in outliers] else "Normal")

    features = df.drop(columns=["variety", "group"]).select_dtypes(include="number")
    mean_profiles = df.groupby("group")[features.columns].mean().T
    mean_profiles["Difference"] = mean_profiles["Outlier"] - mean_profiles["Normal"]

    mean_profiles = mean_profiles.reset_index().rename(columns={"index": "Compound Name"})
    mean_profiles["compound_norm"] = mean_profiles["Compound Name"].str.strip().str.lower()
    chem_map_norm = {k.strip().lower(): v for k, v in chemical_family_map.items()}
    mean_profiles["Chemical Family"] = mean_profiles["compound_norm"].map(chem_map_norm).fillna("Other")

    #  Select top 5 up/down regulated 
    top_up = mean_profiles.nlargest(5, "Difference")
    top_down = mean_profiles.nsmallest(5, "Difference")
    top_diff = pd.concat([top_up, top_down]).sort_values("Difference", ascending=True)

    #  Define color palette
    family_colors = {
        "Alcohol": "#81C784",
        "Aldehyde": "#FBC02D",
        "Ketone": "#4DB6AC",
        "Ester": "#BA68C8",
        "Acid": "#E57373",
        "Furan": "#90A4AE",
        "Aromatic": "#F48FB1",
        "Aromatic/Furan": "#F48FB1",
        "Sulfur": "#8D6E63",
        "Sulfur Compound": "#8D6E63",
        "Pyrazine": "#6D4C41",
        "Other": "#C0C0C0"
    }

    # Auto-assign neutral gray to unseen families
    for fam in top_diff["Chemical Family"].unique():
        if fam not in family_colors:
            family_colors[fam] = "#BDBDBD"

    plt.figure(figsize=(9, 6))
    sns.set_style("whitegrid")

    sns.barplot(
        data=top_diff,
        y="Compound Name",
        x="Difference",
        hue="Chemical Family",
        palette=family_colors,
        dodge=False,
        edgecolor="none"
    )

    plt.axvline(0, color='gray', linestyle='--', lw=1)
    plt.title("Key Aroma Compounds by Chemical Family (Outliers vs Normal)",
              fontsize=14, weight="bold", pad=12)
    plt.xlabel("Mean Intensity Difference (Outliers – Normal)", fontsize=12)
    plt.ylabel("Compound Name", fontsize=12)
    plt.legend(title="Chemical Family", frameon=True, fontsize=10, title_fontsize=11)
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, "outlier_vs_normal_aromas.png"), dpi=300, bbox_inches="tight")
    plt.show()







#  Matrices + clustering (PCA/KMeans, PCA/GMM)

def create_aroma_intensity_matrix(aroma_sample_df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates Variety × Compound matrix using mean intensity.
    Needs columns: FileName (optional), Variety, Compound Name, Intensity
    """
    df = aroma_sample_df.copy()
    if "Variety" not in df.columns:
        df = _ensure_variety_col(df, "Variety")

    aroma_pivot = df.pivot_table(
        index=["Variety"],
        columns="Compound Name",
        values="Intensity",
        aggfunc="mean"
    ).reset_index()

    aroma_pivot.columns.name = None
    print(f"Aroma matrix shape: {aroma_pivot.shape}")
    return aroma_pivot




def pca_kmeans_aroma(
    aroma_matrix: pd.DataFrame,
    output_dir: str = "aroma_pca_kmeans",
    n_components: int = 2,
    k_min: int = 2,
    k_max: int = 7,
    ellipse_min_points: int = 3,
    outlier_iqr_mult: float = 2.0
):
    """
    PCA + KMeans + outlier detection (Mahalanobis) + ellipses.
    """
    os.makedirs(output_dir, exist_ok=True)

    df = _ensure_variety_col(aroma_matrix, "Variety")
    varieties = df["Variety"].astype(str).values

    X = df.drop(columns=["Variety"], errors="ignore").select_dtypes(include=[np.number]).fillna(0)
    X_scaled = StandardScaler().fit_transform(X)

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    print(f"Explained variance ({n_components} PCs): {pca.explained_variance_ratio_.sum()*100:.2f}%")

    mean_vec = np.mean(X_pca, axis=0)
    cov_matrix = np.cov(X_pca, rowvar=False)
    inv_cov = np.linalg.pinv(cov_matrix)
    mahal = np.array([distance.mahalanobis(x, mean_vec, inv_cov) for x in X_pca])

    Q1, Q3 = np.percentile(mahal, [25, 75])
    IQR = Q3 - Q1
    threshold = Q3 + outlier_iqr_mult * IQR
    outlier_mask = mahal > threshold

    K_range = range(k_min, k_max + 1)
    sil_scores = []
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_pca)
        if len(set(labels)) < 2:
            sil_scores.append(-1)
        else:
            sil_scores.append(silhouette_score(X_pca, labels))

    best_k = list(K_range)[int(np.argmax(sil_scores))]
    print(f"Optimal k (silhouette): {best_k}")

    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_pca)
    centers = kmeans.cluster_centers_

    results = pd.DataFrame({
        "Variety": varieties,
        "Cluster": cluster_labels,
        "Mahalanobis": mahal,
        "Outlier": outlier_mask
    })
    results.to_csv(os.path.join(output_dir, "aroma_pca_kmeans_results.csv"), index=False)

    def _draw_ellipse(position, cov2x2, ax, color):
        if cov2x2 is None or cov2x2.shape != (2, 2) or not np.all(np.isfinite(cov2x2)):
            return
        eigvals, eigvecs = np.linalg.eigh(cov2x2)
        eigvals = np.maximum(eigvals, 0)
        order = eigvals.argsort()[::-1]
        eigvals, eigvecs = eigvals[order], eigvecs[:, order]
        angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
        width, height = 2 * np.sqrt(eigvals)
        for nsig in range(1, 4):
            e = Ellipse(
                xy=position,
                width=nsig * width,
                height=nsig * height,
                angle=angle,
                facecolor=color,
                edgecolor="none",
                alpha=0.15
            )
            ax.add_patch(e)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.grid(False)
    colors = plt.cm.Set2(np.linspace(0, 1, best_k))

    for i, color in enumerate(colors):
        pts = X_pca[cluster_labels == i]
        ax.scatter(pts[:, 0], pts[:, 1], color=color, s=70, alpha=0.8,
                   edgecolor="k", label=f"Cluster {i+1}")

        if pts.shape[0] >= ellipse_min_points:
            cov = np.cov(pts[:, :2], rowvar=False)
            _draw_ellipse(centers[i], cov, ax=ax, color=color)

    ax.scatter(centers[:, 0], centers[:, 1], c="black", s=180,
               marker="X", edgecolor="white", linewidth=1.5, label="Centroids")
    ax.scatter(X_pca[outlier_mask, 0], X_pca[outlier_mask, 1],
               color="red", edgecolor="black", s=130, label="Outliers", zorder=5)

    ax.axhline(0, color="gray", lw=0.8, alpha=0.6)
    ax.axvline(0, color="gray", lw=0.8, alpha=0.6)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=12, weight="bold")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=12, weight="bold")
    ax.set_title("PCA + K-Means Clustering of Aroma Varieties", fontsize=14, weight="bold", pad=15)
    ax.legend(frameon=True, fontsize=10, loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pca_kmeans_clusters.png"), dpi=300)
    plt.show()

    return results, X_pca, cluster_labels





def analyze_aroma_clusters_gmm(
    aroma_matrix: pd.DataFrame,
    output_dir: str = "aroma_cluster_results",
    outlier_space: str = "2d",
    outlier_rule: str = "iqr",
    iqr_multiplier: float = 1.5,
    percentile: float = 97.5
):
    """
    PCA (90% var) + GMM + metrics + outlier detection
    """
    os.makedirs(output_dir, exist_ok=True)

    df = _ensure_variety_col(aroma_matrix, "Variety")
    varieties = df["Variety"].astype(str).values

    X = df.drop(columns=["Variety"], errors="ignore").select_dtypes(include=[np.number]).fillna(0)
    X_scaled = StandardScaler().fit_transform(X)

    pca_full = PCA().fit(X_scaled)
    explained = np.cumsum(pca_full.explained_variance_ratio_)
    n_comp = int(np.argmax(explained >= 0.9) + 1)
    print(f"PCA comps for ≥90% variance: {n_comp} ({explained[n_comp-1]*100:.1f}%)")

    pca = PCA(n_components=n_comp)
    X_pca = pca.fit_transform(X_scaled)

    bics, aics = [], []
    k_range = range(2, 11)
    for k in k_range:
        gmm_tmp = GaussianMixture(n_components=k, covariance_type="full", random_state=42)
        gmm_tmp.fit(X_pca)
        bics.append(gmm_tmp.bic(X_pca))
        aics.append(gmm_tmp.aic(X_pca))

    plt.figure(figsize=(7, 4))
    plt.plot(list(k_range), bics, marker="o", label="BIC")
    plt.plot(list(k_range), aics, marker="s", label="AIC")
    plt.title("GMM Model Selection via BIC/AIC")
    plt.xlabel("k")
    plt.ylabel("criterion")
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "gmm_model_selection.png"), dpi=300)
    plt.show()

    best_k = list(k_range)[int(np.argmin(bics))]
    print(f"Best k by BIC: {best_k}")

    gmm = GaussianMixture(n_components=best_k, covariance_type="full", random_state=42)
    cluster_labels = gmm.fit_predict(X_pca)

    if len(set(cluster_labels)) > 1:
        silhouette = silhouette_score(X_pca, cluster_labels)
        dbi = davies_bouldin_score(X_pca, cluster_labels)
        chi = calinski_harabasz_score(X_pca, cluster_labels)
    else:
        silhouette, dbi, chi = np.nan, np.nan, np.nan

    print(f"Silhouette={silhouette:.3f} | DBI={dbi:.3f} | CHI={chi:.1f}")

    scores2d = X_pca[:, :2]
    outlier_data = scores2d if outlier_space.lower() == "2d" else X_pca
    mean_vec = np.mean(outlier_data, axis=0)
    cov_matrix = np.cov(outlier_data, rowvar=False)
    inv_cov = np.linalg.pinv(cov_matrix)
    mahal = np.array([distance.mahalanobis(x, mean_vec, inv_cov) for x in outlier_data])

    if outlier_rule == "percentile":
        thr = np.percentile(mahal, percentile)
        outlier_mask = mahal > thr
    else:
        Q1, Q3 = np.percentile(mahal, [25, 75])
        IQR = Q3 - Q1
        thr = Q3 + iqr_multiplier * IQR
        outlier_mask = mahal > thr

    results = pd.DataFrame({
        "Variety": varieties,
        "Cluster": cluster_labels,
        "Mahalanobis": mahal,
        "is_outlier": outlier_mask
    })
    results.to_csv(os.path.join(output_dir, "gmm_cluster_results.csv"), index=False)
        # ---- Print outliers ----
    outliers_df = results[results["is_outlier"]].sort_values("Mahalanobis", ascending=False)

    print(f"Outliers detected: {outliers_df.shape[0]} / {results.shape[0]}")
    if not outliers_df.empty:
        print("Outlier varieties (sorted by Mahalanobis distance):")
        for _, row in outliers_df.iterrows():
            print(f" - {row['Variety']} | Cluster={int(row['Cluster'])+1} | Mahalanobis={row['Mahalanobis']:.3f}")

    # PC1/PC2 plot
    plt.figure(figsize=(12, 9))
    plt.grid(False)
    colors = plt.cm.tab10(np.linspace(0, 1, best_k))
    for i, color in enumerate(colors):
        pts = scores2d[cluster_labels == i]
        plt.scatter(pts[:, 0], pts[:, 1], color=color, s=80, alpha=0.85, edgecolor="k", label=f"Cluster {i+1}")

    plt.scatter(scores2d[outlier_mask, 0], scores2d[outlier_mask, 1],
                color="red", edgecolor="black", s=130, label="Outliers", zorder=5)
    plt.axhline(0, color="gray", lw=0.8, alpha=0.6)
    plt.axvline(0, color="gray", lw=0.8, alpha=0.6)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA + GMM Clustering (Outliers Highlighted)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pca_gmm_clusters.png"), dpi=300)
    plt.show()

    # t-SNE (on PCA)
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1500, random_state=42)
    X_tsne = tsne.fit_transform(X_pca)
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=cluster_labels, palette="tab10", s=90, alpha=0.9, edgecolor="k")
    plt.scatter(X_tsne[outlier_mask, 0], X_tsne[outlier_mask, 1], color="red", edgecolor="black", s=130, label="Outliers")
    plt.title("t-SNE Visualization of GMM Clusters")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "tsne_gmm.png"), dpi=300)
    plt.show()

    cluster_means = pd.DataFrame(X, index=varieties).groupby(cluster_labels).mean()
    return results, cluster_means




def plot_outlier_vs_normal_aromas(
    aroma_matrix: pd.DataFrame,
    outliers: List[str],
    chemical_family_map: Dict[str, str],
    output_dir: str = "aroma_cluster_results",
    top_n: int = 5
) -> pd.DataFrame:
    """
    Canonical version: outlier vs normal mean profile difference barplot.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = _ensure_variety_col(aroma_matrix, "Variety")
    df["Variety"] = df["Variety"].astype(str).str.strip()

    outlier_set = set(str(o).strip().upper() for o in outliers)
    df["Group"] = df["Variety"].apply(lambda x: "Outlier" if x.upper() in outlier_set else "Normal")

    feature_cols = [c for c in df.columns if c not in ["Variety", "Group"]]
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")

    mean_profiles = pd.concat([df[["Group"]], X], axis=1).groupby("Group").mean().T
    if "Outlier" not in mean_profiles.columns or "Normal" not in mean_profiles.columns:
        raise ValueError(f"Both groups must exist. Found: {mean_profiles.columns.tolist()}")

    mean_profiles["Difference"] = mean_profiles["Outlier"] - mean_profiles["Normal"]
    mean_profiles = mean_profiles.reset_index().rename(columns={"index": "Compound Name"})

    chem_map_norm = {str(k).strip().lower(): v for k, v in chemical_family_map.items()}
    mean_profiles["Chemical Family"] = mean_profiles["Compound Name"].astype(str).str.lower().map(chem_map_norm).fillna("Other")

    top_up = mean_profiles[mean_profiles["Difference"] > 0].nlargest(top_n, "Difference")
    top_down = mean_profiles[mean_profiles["Difference"] < 0].nsmallest(top_n, "Difference")
    top_diff = pd.concat([top_down, top_up]).sort_values("Difference", ascending=True)

    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.figure(figsize=(10, 7))
    sns.barplot(data=top_diff, y="Compound Name", x="Difference", hue="Chemical Family", dodge=False, edgecolor="none")
    plt.axvline(0, color="gray", linestyle="--", lw=1)
    plt.title("Key Aroma Compounds by Chemical Family (Outliers vs Normal)", fontsize=14, weight="bold", pad=12)
    plt.xlabel("Mean Intensity Difference (Outliers – Normal)")
    plt.ylabel("Compound Name")
    plt.tight_layout()
    out_png = os.path.join(output_dir, "outlier_vs_normal_aromas.png")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.show()

    print("Saved plot:", out_png)
    return top_diff





#  Heatmaps / summaries / PCA feature importance

def summarize_aroma_intensities(aroma_matrix: pd.DataFrame, top_n: int = 10):
    numeric_data = aroma_matrix.select_dtypes(include=["number"])
    desc_stats = numeric_data.describe().T
    top_compounds = desc_stats.sort_values(by="mean", ascending=False).head(top_n)

    top_plot = top_compounds[["mean", "std"]]
    print(top_plot.round(2))

    plt.figure(figsize=(10, 6))
    plt.barh(top_plot.index, top_plot["mean"], xerr=top_plot["std"])
    plt.xlabel("Mean Intensity (±1 SD)")
    plt.title(f"Top {top_n} Aroma Compounds by Mean Intensity")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    return desc_stats, top_compounds





def plot_top_aroma_heatmap(aroma_matrix: pd.DataFrame, top_n: int = 5, mode: str = "per_variety") -> None:
    df = _ensure_variety_col(aroma_matrix, "Variety")

    aroma_long = df.melt(id_vars="Variety", var_name="Compound", value_name="Intensity")
    if mode == "per_variety":
        top_aromas = (
            aroma_long.groupby("Variety", group_keys=False)
            .apply(lambda g: g.nlargest(top_n, "Intensity"))
            .reset_index(drop=True)
        )
        title = f"Top {top_n} per Variety"
    elif mode == "overall":
        top_compounds = aroma_long.groupby("Compound")["Intensity"].mean().nlargest(top_n).index
        top_aromas = aroma_long[aroma_long["Compound"].isin(top_compounds)]
        title = f"Top {top_n} overall"
    else:
        raise ValueError("mode must be 'per_variety' or 'overall'")

    heatmap_data = top_aromas.pivot_table(index="Variety", columns="Compound", values="Intensity", fill_value=0)
    plt.figure(figsize=(12, 10))
    sns.heatmap(heatmap_data, cmap="YlOrRd", linewidths=0.2, cbar_kws={"label": "Intensity"})
    plt.title(f"Aroma Intensity Heatmap — {title}")
    plt.tight_layout()
    plt.show()





def compute_feature_importance_from_pca(aroma_matrix: pd.DataFrame, n_components: int = 2) -> pd.DataFrame:
    df = _ensure_variety_col(aroma_matrix, "Variety")
    X = df.drop(columns=["Variety"], errors="ignore").fillna(0)

    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=n_components).fit(X_scaled)

    loadings = pd.DataFrame(
        np.abs(pca.components_.T),
        index=X.columns,
        columns=[f"PC{i+1}" for i in range(n_components)]
    )
    loadings["Total_Importance"] = loadings.sum(axis=1)
    imp = loadings.sort_values("Total_Importance", ascending=False)

    top_n = 15
    plt.figure(figsize=(10, 6))
    plt.barh(imp.head(top_n).index[::-1], imp["Total_Importance"].head(top_n)[::-1])
    plt.xlabel("Total PCA Loading (|contribution|)")
    plt.title(f"Top {top_n} Compounds Contributing to PCA Variance")
    plt.tight_layout()
    plt.show()

    return imp



#  Sensory side (cleaning + merge + correlation + prediction)

SENSORY_COLS = [
    "Metallic Flavour","Bitter Flavour","Earthy Flavour","Sour Flavour",
    "Fresh Flavour","Sweet Flavour","Root/ Vegetable Flavour",
    "Farmyard (grass/hay) flavour","Bitter Aftertaste","Sour Aftertaste","Sweet Aftertaste",
    "Earthy Aroma","Farmyard","Sweet","Starchy","Damp","Buttery","Fresh","Potato Starch","Metalic"
]


def summarize_sensory_traits(sensory_df: pd.DataFrame, sensory_cols: Optional[List[str]] = None) -> pd.DataFrame:
    df = sensory_df.copy()
    sensory_cols = sensory_cols or SENSORY_COLS

    if "VARIETY" not in df.columns:
        # accept any case
        for c in df.columns:
            if c.strip().lower() == "variety":
                df.rename(columns={c: "VARIETY"}, inplace=True)
                break

    df["VARIETY"] = df["VARIETY"].astype(str).str.strip()
    df = df.drop(columns=[c for c in df.columns if str(c).startswith("Unnamed")], errors="ignore")

    traits = df[sensory_cols].apply(pd.to_numeric, errors="coerce")
    traits = traits.assign(VARIETY=df["VARIETY"])

    summary = traits.groupby("VARIETY", as_index=False).mean(numeric_only=True)

    print(f"Rows (unique varieties): {summary['VARIETY'].nunique()}")
    print(f"Traits averaged: {len(sensory_cols)}")
    print(summary.head())
    return summary





def merge_aroma_with_sensory(aroma_df: pd.DataFrame, sensory_summary: pd.DataFrame) -> pd.DataFrame:
    aroma_df = _ensure_variety_col(aroma_df, "Variety")
    sensory_summary = sensory_summary.copy()

    # sensory variety column normalize
    if "Variety" not in sensory_summary.columns:
        for c in sensory_summary.columns:
            if c.strip().lower() == "variety":
                sensory_summary.rename(columns={c: "Variety"}, inplace=True)
                break
        if "Variety" not in sensory_summary.columns and "VARIETY" in sensory_summary.columns:
            sensory_summary.rename(columns={"VARIETY": "Variety"}, inplace=True)

    aroma_df["Variety"] = aroma_df["Variety"].astype(str).str.strip().str.upper()
    sensory_summary["Variety"] = sensory_summary["Variety"].astype(str).str.strip().str.upper()

    common = set(aroma_df["Variety"]).intersection(set(sensory_summary["Variety"]))
    print(f"Aroma varieties: {aroma_df['Variety'].nunique()}")
    print(f"Sensory varieties: {sensory_summary['Variety'].nunique()}")
    print(f"Common: {len(common)}")

    merged = pd.merge(aroma_df, sensory_summary, on="Variety", how="inner")
    print(f"Merged varieties: {merged['Variety'].nunique()}")
    return merged







def plot_aroma_sensory_correlation(
    merged_aroma_sensory: pd.DataFrame,
    sensory_cols: List[str],
    aroma_descriptions: Optional[Dict[str, str]] = None,
    threshold: float = 0.1,
    min_compounds: int = 15,
    save_path: Optional[str] = None
) -> pd.DataFrame:
    df = merged_aroma_sensory.copy()
    df.columns = df.columns.str.strip().str.lower()
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.loc[:, df.nunique() > 1]

    sensory_cols = [c.strip().lower() for c in sensory_cols]
    df = df.select_dtypes(include=["number"]).replace([np.inf, -np.inf], np.nan)
    df = df.dropna(axis=1, thresh=max(3, len(df) // 2)).dropna(axis=0, how="any")

    sensory_cols_found = [c for c in df.columns if c in sensory_cols]
    compound_cols = [c for c in df.columns if c not in sensory_cols_found]

    if not sensory_cols_found or not compound_cols:
        raise ValueError("No valid sensory/compound columns found for correlation.")

    corr = df[compound_cols + sensory_cols_found].corr()
    block = corr.loc[compound_cols, sensory_cols_found].fillna(0)
    strong = block[(block.abs() > threshold).any(axis=1)]

    if strong.shape[0] < min_compounds:
        top = block.abs().max(axis=1).sort_values(ascending=False).head(min_compounds).index
        strong = block.loc[top]

    if aroma_descriptions:
        annotated = []
        for c in strong.index:
            name = str(c).strip()
            desc = aroma_descriptions.get(name, aroma_descriptions.get(name.title(), "No description"))
            annotated.append(f"{name}\n({desc})")
        strong.index = annotated

    plt.figure(figsize=(16, 9))
    sns.heatmap(strong, cmap="coolwarm", center=0, annot=True, fmt=".2f", linewidths=0.5, cbar_kws={"label": "r"})
    plt.title("Correlation Between Aroma Compounds and Sensory Attributes", fontsize=16, pad=15)
    plt.tight_layout()
    plt.show()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        strong.to_csv(save_path)
        print("Saved:", save_path)

    return strong



# Dashboard 

def launch_aroma_dashboard(aroma_matrix: pd.DataFrame, port: int = 7031) -> None:
    """
    Dash/Plotly dashboard. Imports inside so module import doesn't fail if dash isn't installed.
    """
    try:
        import dash
        from dash import dcc, html
        from dash.dependencies import Input, Output
        import plotly.express as px
    except Exception as e:
        raise ImportError("dash/plotly not installed. Install them to use launch_aroma_dashboard().") from e

    df = _ensure_variety_col(aroma_matrix, "Variety")
    base = df.set_index("Variety").copy()
    base.index = base.index.astype(str)
    base.columns = [str(c) for c in base.columns]
    options = [{"label": v, "value": v} for v in base.index]

    app = dash.Dash(__name__)
    app.layout = html.Div([
        html.H1("Potato Aroma Compound Dashboard"),
        html.Label("Select Variety:"),
        dcc.Dropdown(id="variety-dropdown", options=options, value=base.index[0], clearable=False),
        dcc.Graph(id="compound-barplot"),
        dcc.Graph(id="compound-heatmap"),
    ])

    @app.callback(
        [Output("compound-barplot", "figure"), Output("compound-heatmap", "figure")],
        [Input("variety-dropdown", "value")]
    )
    def update_dashboard(selected_variety):
        s = base.loc[selected_variety]
        s = pd.to_numeric(s, errors="coerce").fillna(0).sort_values(ascending=False).reset_index()
        s.columns = ["Compound", "Relative Concentration"]

        fig_bar = px.bar(s, x="Compound", y="Relative Concentration", title=f"Aroma Profile for {selected_variety}")
        fig_bar.update_layout(xaxis_tickangle=-45, height=450, margin=dict(l=60, r=20, t=60, b=120))

        heat_df = base.apply(pd.to_numeric, errors="coerce").fillna(0).T
        fig_heat = px.imshow(heat_df, aspect="auto", title="Heatmap of Compounds Across Varieties")
        fig_heat.update_layout(height=600, margin=dict(l=90, r=20, t=60, b=120))
        return fig_bar, fig_heat

    print(f"\nLaunching dashboard at: http://127.0.0.1:{port}")
    app.run(debug=True, port=port)





def perform_knn_sensory_projection(
    aroma_matrix,
    sensory_summary,
    k=3,
    n_components=3,
    save_projection_path=None,
    save_loadings_path=None,
    scaling_factor=5
):
    """
    PCA on sensory PANEL varieties only.
    Non-panel varieties are projected into sensory PCA space using KNN
    based on aroma similarity.
    """
    #  Standardize inputs
    aroma_df = aroma_matrix.copy()
    sensory_df = sensory_summary.copy()

    aroma_df.columns = aroma_df.columns.str.strip().str.upper()
    sensory_df.columns = sensory_df.columns.str.strip().str.upper()

    if "VARIETY" not in aroma_df.columns:
        raise KeyError("'VARIETY' missing in aroma_matrix")
    if "VARIETY" not in sensory_df.columns:
        raise KeyError("'VARIETY' missing in sensory_summary")

    aroma_df["VARIETY"] = aroma_df["VARIETY"].astype(str).str.strip().str.upper()
    sensory_df["VARIETY"] = sensory_df["VARIETY"].astype(str).str.strip().str.upper()

    sensory_cols = [c for c in sensory_df.columns if c != "VARIETY"]
    aroma_cols   = [c for c in aroma_df.columns if c != "VARIETY"]

    aroma_df[aroma_cols] = aroma_df[aroma_cols].apply(
        pd.to_numeric, errors="coerce"
    ).fillna(0)

    sensory_df[sensory_cols] = sensory_df[sensory_cols].apply(
        pd.to_numeric, errors="coerce"
    )

    #  TRUE panel definition (intersection)
    panel_varieties = set(aroma_df["VARIETY"]) & set(sensory_df["VARIETY"])

    if len(panel_varieties) == 0:
        raise ValueError("No overlapping varieties between aroma and sensory!")

    aroma_panel = aroma_df[aroma_df["VARIETY"].isin(panel_varieties)].copy()
    aroma_nonpanel = aroma_df[~aroma_df["VARIETY"].isin(panel_varieties)].copy()
    sensory_panel = sensory_df[sensory_df["VARIETY"].isin(panel_varieties)].copy()

    print("Panel varieties:", len(panel_varieties))
    print("Aroma non-panel varieties:", aroma_nonpanel.shape[0])


    # PCA on sensory PANEL ONLY
    sensory_scaler = StandardScaler()
    Y_panel_scaled = sensory_scaler.fit_transform(
        sensory_panel[sensory_cols].values
    )

    pca = PCA(n_components=n_components, random_state=0)
    pcs_panel = pca.fit_transform(Y_panel_scaled)

    panel_pc_df = pd.DataFrame(
        pcs_panel,
        columns=[f"PC{i+1}" for i in range(n_components)]
    )
    panel_pc_df.insert(0, "VARIETY", sensory_panel["VARIETY"].values)
    panel_pc_df["is_panel"] = True

   
    # KNN projection for NON-PANEL (AROMA space)
   
    if aroma_nonpanel.empty:
        print("No aroma-only varieties → skipping KNN")
        pc_df = panel_pc_df.copy()

    else:
        aroma_scaler = StandardScaler()
        X_panel = aroma_scaler.fit_transform(aroma_panel[aroma_cols].values)
        X_nonpanel = aroma_scaler.transform(aroma_nonpanel[aroma_cols].values)

        k_eff = min(k, X_panel.shape[0])

        nn = NearestNeighbors(n_neighbors=k_eff, metric="euclidean")
        nn.fit(X_panel)

        _, indices = nn.kneighbors(X_nonpanel)

        panel_pc_mat = (
            panel_pc_df
            .set_index("VARIETY")
            .loc[aroma_panel["VARIETY"]]
            [[f"PC{i+1}" for i in range(n_components)]]
            .values
        )

        Y_nonpanel_pred = np.array([
            panel_pc_mat[idx].mean(axis=0) for idx in indices
        ])

        nonpanel_pc_df = pd.DataFrame(
            Y_nonpanel_pred,
            columns=[f"PC{i+1}" for i in range(n_components)]
        )
        nonpanel_pc_df.insert(0, "VARIETY", aroma_nonpanel["VARIETY"].values)
        nonpanel_pc_df["is_panel"] = False

        pc_df = pd.concat([panel_pc_df, nonpanel_pc_df], ignore_index=True)


    # Loadings
    loadings_df = pd.DataFrame(
        pca.components_.T,
        index=sensory_cols,
        columns=[f"PC{i+1}" for i in range(n_components)]
    )

    print("Explained variance ratio:",
          np.round(pca.explained_variance_ratio_, 3))
    print("Total projected varieties:", pc_df.shape[0])

    # -------------------------------------------------
    # 5) Save outputs
    # -------------------------------------------------
    if save_projection_path:
        os.makedirs(os.path.dirname(save_projection_path), exist_ok=True)
        pc_df.to_csv(save_projection_path, index=False)

    if save_loadings_path:
        os.makedirs(os.path.dirname(save_loadings_path), exist_ok=True)
        loadings_df.to_csv(save_loadings_path)

    return pc_df, pca, loadings_df




def plot_sensory_pca_biplot(pc_df, loadings_df, scaling_factor=15):
    plt.figure(figsize=(8, 7))

    # Panel vs non-panel
    plt.scatter(
        pc_df.loc[pc_df["is_panel"], "PC1"],
        pc_df.loc[pc_df["is_panel"], "PC2"],
        c="tab:blue",
        label="Sensory panel",
        alpha=0.8
    )

    plt.scatter(
        pc_df.loc[~pc_df["is_panel"], "PC1"],
        pc_df.loc[~pc_df["is_panel"], "PC2"],
        c="tab:orange",
        marker="*",
        s=120,
        label="Aroma-only (KNN-projected)"
    )

    # Loadings arrows
    for trait in loadings_df.index:
        x = loadings_df.loc[trait, "PC1"] * scaling_factor
        y = loadings_df.loc[trait, "PC2"] * scaling_factor
        plt.arrow(0, 0, x, y, color="red", alpha=0.6, head_width=0.15)
        plt.text(x * 1.1, y * 1.1, trait, fontsize=8, color="red")

    plt.axhline(0, color="grey", linestyle="--", linewidth=0.8)
    plt.axvline(0, color="grey", linestyle="--", linewidth=0.8)

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Sensory PCA biplot with aroma-based KNN projection")
    plt.legend()
    plt.tight_layout()
    plt.show()