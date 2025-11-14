
import os
import glob
import pandas as pd
from difflib import get_close_matches
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial import distance
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
from sklearn.neighbors import NearestNeighbors
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score



def find_files_case_insensitive(base_dir, keyword):
    """Recursively search for CSV files containing a keyword (case-insensitive)."""
    found = []
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.lower().endswith(".csv") and keyword.lower() in f.lower():
                found.append(os.path.join(root, f))
    return found


def combine_compound_data(sample_files, base_dir, output_path,
                          qc_files=None, env_files=None, blank_files=None):
    """
    Combines GC-MS compound data from sample, QC, env, and blank CSV files into one DataFrame.

    Parameters
    ----------
    sample_files : list
        List of file paths for sample compound CSVs.
    base_dir : str
        Base directory containing day_1, day_2, ... subfolders.
    output_path : str
        Path where the combined CSV will be saved.
    qc_files, env_files, blank_files : list, optional
        Lists of QC, environmental, and blank compound CSVs.

    Returns
    -------
    DataFrame
        Combined DataFrame of all compounds with a 'FileType' column.
    """

    # Helper function to load a CSV file safely
    def load_file(file_path, file_type):
        try:
            df = pd.read_csv(file_path)
            df["FileType"] = file_type
            df["FileName"] = os.path.basename(file_path)
            return df
        except Exception as e:
            print(f" Error reading {file_type} file: {file_path}\n{e}")
            return pd.DataFrame()

    # --- Load all datasets ---
    all_data = []

    # Load sample files
    for f in sample_files:
        all_data.append(load_file(f, "sample"))

    # Load QC, Env, and Blank files (if provided)
    if qc_files:
        for f in qc_files:
            all_data.append(load_file(f, "qc"))
    if env_files:
        for f in env_files:
            all_data.append(load_file(f, "env"))
    if blank_files:
        for f in blank_files:
            all_data.append(load_file(f, "blank"))

    # Combine everything
    combined_df = pd.concat(all_data, ignore_index=True)

    # Save combined data
    combined_df.to_csv(output_path, index=False)
    print(f"\nCombined file saved to: {output_path}")
    print(f"\nFileType counts:\n{combined_df['FileType'].value_counts()}")
    print("\nPreview of combined data:")
    print(combined_df.head())

    return combined_df

import matplotlib.pyplot as plt

def plot_filetype_counts(df, column="FileType", title="Number of Files per Type"):
    """
    Plot the count of different file types in a GC–MS dataset.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the file type information.
    column : str, optional
        Name of the column containing file type labels (default: "FileType").
    title : str, optional
        Title for the plot.
    """
    if column not in df.columns:
        raise ValueError(f" Column '{column}' not found in DataFrame")

    # Count file types
    file_type_counts = df[column].value_counts()

    # Bar plot
    plt.figure(figsize=(6, 4))
    plt.bar(
        file_type_counts.index,
        file_type_counts.values,
        color=["#4C72B0", "#FFA500", "#55A868", "#C44E52"]
    )
    plt.title(title, fontsize=12)
    plt.xlabel("File Type")
    plt.ylabel("Count")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

    print(" File type count plot generated successfully.")
    print(file_type_counts)



def filter_real_compounds(compound_df, rt_tolerance=0.4):
    """
    Removes sample compounds that also appear in blank, env, or QC files
    within a given RT (retention time) tolerance.

    Parameters
    ----------
    compound_df : DataFrame
        Combined compound data containing 'Component RT', 'Compound Name', and 'FileType'.
    rt_tolerance : float
        Allowed RT difference (in minutes) to consider a match (default = 0.3).

    Returns
    -------
    DataFrame
        Filtered DataFrame containing only sample compounds not matching contaminants.
    """

    # Separate by file type
    sample_df = compound_df[compound_df["FileType"] == "sample"].copy()
    contam_df = compound_df[compound_df["FileType"].isin(["blank", "env", "qc"])].copy()

    print(f"Sample compounds before filtering: {len(sample_df)}")

    # Ensure numeric RT
    sample_df["Component RT"] = pd.to_numeric(sample_df["Component RT"], errors="coerce")
    contam_df["Component RT"] = pd.to_numeric(contam_df["Component RT"], errors="coerce")

    # Sort to speed up comparisons
    sample_df = sample_df.sort_values("Component RT")
    contam_df = contam_df.sort_values("Component RT")

    # Identify contaminated peaks
    contaminated = []
    for _, c in contam_df.iterrows():
        rt = c["Component RT"]
        name = c["Compound Name"]
        # find sample peaks within RT tolerance and with same compound name
        matches = sample_df[
            (np.abs(sample_df["Component RT"] - rt) <= rt_tolerance)
            & (sample_df["Compound Name"].str.lower() == name.lower())
        ].index
        contaminated.extend(matches)

    contaminated = list(set(contaminated))

    # Remove contaminated ones
    filtered_df = sample_df.drop(index=contaminated)
    print(f"Sample compounds after filtering: {len(filtered_df)}")
    print(f"Removed {len(contaminated)} sample peaks (ΔRT ≤ {rt_tolerance})")

    return filtered_df

import matplotlib.pyplot as plt

def plot_rt_distribution(compound_df, real_compounds, bins=40):
    """
    Plot retention time (RT) distribution before and after contamination filtering.

    Parameters
    ----------
    compound_df : pandas.DataFrame
        Original dataset containing all sample entries.
    real_compounds : pandas.DataFrame
        Filtered dataset after removing contaminations.
    bins : int, optional
        Number of histogram bins (default: 40).
    """
    plt.figure(figsize=(6, 4))
    plt.hist(
        compound_df[compound_df["FileType"] == "sample"]["Component RT"],
        bins=bins, alpha=0.5, label="Before filtering", color="#4C72B0"
    )
    plt.hist(
        real_compounds["Component RT"],
        bins=bins, alpha=0.7, label="After filtering", color="#55A868"
    )
    plt.xlabel("Retention Time (min)")
    plt.ylabel("Count")
    plt.title("RT Distribution Before and After Filtering")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    print("✅ RT distribution plot generated successfully.")


def plot_filtering_effect(compound_df, real_compounds, rt_tolerance=0.5):
    """
    Plot the number of sample compounds before and after filtering.

    Parameters
    ----------
    compound_df : pandas.DataFrame
        Original dataset containing all samples.
    real_compounds : pandas.DataFrame
        Filtered dataset after removing contaminants.
    rt_tolerance : float, optional
        Retention time tolerance used for filtering (default: 0.5).
    """
    before = compound_df[compound_df["FileType"] == "sample"].shape[0]
    after = real_compounds.shape[0]

    plt.figure(figsize=(5, 4))
    plt.bar(
        ["Before filtering", "After filtering"],
        [before, after],
        color=["#4C72B0", "#55A868"]
    )
    plt.ylabel("Number of sample compound entries")
    plt.title(f"Effect of Contamination Filtering (RT ±{rt_tolerance} min)")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

    print(f" Filtering effect plot generated successfully. (Before: {before}, After: {after})")




def match_compounds_to_peaks(
    compound_df,
    samples_filtered_path,
    output_path,
    rt_tolerance=0.4
):
    """
    Match NIST-identified compounds with GC–MS peaks based on retention time (RT).
    """

    # Load filtered samples
    samples_filtered = pd.read_csv(samples_filtered_path)

    # Handle missing 'Filename' column
    if "Filename" in samples_filtered.columns:
        samples_filtered = samples_filtered[
            ~samples_filtered['Filename'].str.contains("QC|ENV|BLANK", case=False, na=False)
        ].copy()
    else:
        print(" 'Filename' column not found — skipping QC/blank filtering.")

    print(f"Samples kept after filtering: {samples_filtered['Variety'].nunique()} unique varieties")

    # Clean compound filenames
    compound_df["Variety"] = compound_df["FileName"].str.replace(".csv", "", regex=False).str.strip()

    # Helper: find closest peak
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

        compound_rt = row["Component RT"]
        peak_idx = find_closest_peak(compound_rt, sample_data["tR_best"], tol=rt_tolerance)

        if peak_idx is None:
            unmatched.append({**row, "Reason": "No matching RT"})
            continue

        peak_row = sample_data.loc[peak_idx]
        matches.append({
            "Compound Name": row["Compound Name"],
            "FileName": row["FileName"],
            "Variety": variety,
            "Component RT": compound_rt,
            "Peak": peak_row["Peak"],
            "tR_best": peak_row["tR_best"],
            "m/z": peak_row["m/z"],
            "Intensity": peak_row["Intensity_corrected"],
        })

    compound_intensity_df = pd.DataFrame(matches)
    unmatched_df = pd.DataFrame(unmatched)

    compound_intensity_df.to_csv(output_path, index=False)
    print(f"\n Matched {len(compound_intensity_df)} compounds — saved to: {output_path}")

    return compound_intensity_df, unmatched_df

# --- 2. Function definition ---
def extract_aroma_compounds(compound_intensity_df, potato_aromas, output_path):
    """
    Identify and extract known potato aroma compounds from the matched intensity dataset.

    Parameters
    ----------
    compound_intensity_df : pd.DataFrame
        DataFrame containing matched compound intensity information.
    potato_aromas : list of str
        List of known potato aroma compound names (from literature).
    output_path : str
        Full path (including filename) where the aroma compound file will be saved.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing only aroma compounds.
    """

    # Helper function for name normalization
    def normalize(name: str) -> str:
        return str(name).lower().replace(" ", "").replace("-", "").replace(",", "")

    # Create normalized reference dictionary for exact and fuzzy matching
    ref_norm = {normalize(x): x for x in potato_aromas}

    # Normalize names in the dataset
    compound_intensity_df["norm_name"] = compound_intensity_df["Compound Name"].apply(normalize)

    # Exact matches
    compound_intensity_df["is_aroma"] = compound_intensity_df["norm_name"].isin(ref_norm)

    # Fuzzy matching for non-exact names
    def fuzzy_match(n, refs, cutoff=0.6):
        matches = get_close_matches(n, refs, n=1, cutoff=cutoff)
        return matches[0] if matches else None

    mask = ~compound_intensity_df["is_aroma"]
    fuzzy_hits = compound_intensity_df.loc[mask, "norm_name"].map(
        lambda n: fuzzy_match(n, list(ref_norm.keys()))
    )

    # Update with fuzzy matches
    compound_intensity_df.loc[mask & fuzzy_hits.notna(), "is_aroma"] = True
    compound_intensity_df.loc[mask & fuzzy_hits.notna(), "Compound Name"] = (
        fuzzy_hits.map(ref_norm)
    )

    # Keep only aroma compounds
    aroma_sample_df = compound_intensity_df[compound_intensity_df["is_aroma"]].copy()

    # Print summary
    print(f"\nNumber of aroma compound records: {aroma_sample_df.shape[0]}")
    print(f"Number of unique samples: {aroma_sample_df['FileName'].nunique()}")
    print(f"Number of unique varieties: {aroma_sample_df['Variety'].nunique()}\n")
    print("Variety counts:")
    print(aroma_sample_df["Variety"].value_counts())

    # Save filtered data
    aroma_sample_df.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")

    return aroma_sample_df




# === Aroma Descriptions Dictionary (stored in module) ===
AROMA_DESCRIPTIONS = {
    # --- Aldehydes ---
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

    # --- Alcohols ---
    "1-Heptanol": "Floral, herbal, fatty",
    "1-Nonanol": "Waxy, floral, oily",
    "1-Octen-3-ol": "Mushroom, earthy, raw potato",
    "1-Phenylethanol": "Floral, rose, sweet",
    "Hexanol": "Green, grassy, herbaceous",
    "2-Methyl-1-propanol": "Fusel, solvent-like",
    "3-Methyl-1-butanol": "Whiskey-like, malty, fusel",
    "Ethanol": "Alcoholic, sweet",

    # --- Ketones ---
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

    # --- Sulfur compounds ---
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

    # --- Acids ---
    "Acetic acid": "Vinegar-like, sour",
    "Butanoic acid": "Rancid, cheesy, sweaty",
    "Hexanoic acid": "Fatty, sweaty, rancid",
    "Isobutyric acid": "Cheesy, rancid, sour",
    "Isovaleric acid": "Sweaty, cheesy, foot-like",
    "Octanoic acid": "Fatty, soapy, rancid",
    "Propanoic acid": "Sour, pungent",

    # --- Esters ---
    "Ethyl acetate": "Fruity, solvent-like",
    "Isoamyl acetate": "Banana, fruity, sweet",
    "Ethyl butanoate": "Pineapple, fruity, sweet",
    "Ethyl hexanoate": "Apple, pineapple, sweet",
    "Methyl butanoate": "Apple, fruity, sweet",
    "Ethyl 2-methylbutanoate": "Apple, sweet, fruity",
    "Ethyl propanoate": "Fruity, rum-like",
    "Methyl propanoate": "Fruity, sweet",
    "Ethyl pentanoate": "Fruity, sweet",

    # --- Pyrazines ---
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

    # --- Aromatics & Furans ---
    "Benzeneacetaldehyde": "Honey, floral, sweet",
    "Benzyl alcohol": "Floral, sweet, mild",
    "Phenylethyl alcohol": "Rose-like, floral, sweet",
    "Furfural": "Sweet, almond, caramel",
    "5-Methylfurfural": "Caramel, baked, sweet"
}

def add_aroma_descriptions(aroma_sample_df, aroma_descriptions=None):
    """
    Adds human-readable aroma descriptions to aroma compound data.

    Parameters
    ----------
    aroma_sample_df : pd.DataFrame
        DataFrame containing matched aroma compounds (must include 'Compound Name').
    aroma_descriptions : dict, optional
        Custom dictionary mapping compound names to their sensory descriptions.
        If not provided, uses the default AROMA_DESCRIPTIONS defined in this module.

    Returns
    -------
    pd.DataFrame
        Updated DataFrame with a new column 'Aroma Description'.
    """

    # Use default dictionary if none provided
    if aroma_descriptions is None:
        aroma_descriptions = AROMA_DESCRIPTIONS

    # Normalize helper
    def normalize_name(name: str) -> str:
        return str(name).lower().replace("-", "").replace(",", "").replace(" ", "")

    # Normalize dataframe compound names
    aroma_sample_df["norm_name"] = aroma_sample_df["Compound Name"].apply(normalize_name)

    # Normalize dictionary keys
    aroma_descriptions_norm = {normalize_name(k): v for k, v in aroma_descriptions.items()}

    # Map descriptions
    aroma_sample_df["Aroma Description"] = aroma_sample_df["norm_name"].map(aroma_descriptions_norm)

    # Check for missing ones
    missing = set(aroma_sample_df["norm_name"].unique()) - set(aroma_descriptions_norm.keys())
    if missing:
        print("\nCompounds without descriptions:")
        print(missing)
    else:
        print("\n All compounds have descriptions!")

    print(f"Number of unique aroma compounds: {aroma_sample_df['Compound Name'].nunique()}")

    return aroma_sample_df



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
    import pandas as pd

    if "Compound Name" not in aroma_sample_df.columns:
        raise KeyError("Input DataFrame must include a 'Compound Name' column.")

    # --- Aroma Descriptions ---
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

    # --- Chemical Family Mapping ---
    chemical_family_map = {}
    for compound in aroma_descriptions.keys():
        if compound in [
            "Propanal", "Butanal, 3-methyl-", "Pentanal", "2-Methylbutanal", "3-Methylbutanal",
            "Hexanal", "Heptanal", "Octanal", "Nonanal", "Decanal", "2-Trans-nonenal",
            "2-Trans-octenal", "4-Heptenal", "2,4-Decadienal", "Phenylacetaldehyde", "Benzaldehyde"
        ]:
            family = "Aldehyde"
        elif compound in [
            "1-Heptanol", "1-Nonanol", "1-Octen-3-ol", "1-Phenylethanol", "Hexanol",
            "2-Methyl-1-propanol", "3-Methyl-1-butanol", "Ethanol"
        ]:
            family = "Alcohol"
        elif compound in [
            "2-Heptanone", "2-Nonanone", "3-Heptanone", "3-Octanone", "2,3-Butanedione",
            "2,3-Pentanedione", "Acetophenone", "2-Propanone, 1-methoxy-", "4-Methyl-2-pentanone",
            "2-Undecanone"
        ]:
            family = "Ketone"
        elif compound in [
            "Methanethiol", "Methional", "Dimethyl disulfide", "Dimethyl trisulfide",
            "Dimethyl tetrasulfide", "Carbon disulfide", "Hydrogen sulfide",
            "2-Methyl-3-furanthiol", "3-Methylthiopropanal", "2-Acetylthiazole",
            "2-Methylthiazole", "2-Methyl-3-thiazoline", "3-(Methylthio)propanal", "Allyl methyl sulfide"
        ]:
            family = "Sulfur Compound"
        elif compound in [
            "Acetic acid", "Butanoic acid", "Hexanoic acid", "Isobutyric acid",
            "Isovaleric acid", "Octanoic acid", "Propanoic acid"
        ]:
            family = "Acid"
        elif compound in [
            "Ethyl acetate", "Isoamyl acetate", "Ethyl butanoate", "Ethyl hexanoate",
            "Methyl butanoate", "Ethyl 2-methylbutanoate", "Ethyl propanoate",
            "Methyl propanoate", "Ethyl pentanoate"
        ]:
            family = "Ester"
        elif compound in [
            "2-Acetylpyrazine", "2-Ethyl-3,5-dimethylpyrazine", "2-Isobutyl-3-methoxypyrazine",
            "2-Isopropyl-3-methoxypyrazine", "2-Ethyl-3-methylpyrazine", "Trimethylpyrazine",
            "Tetramethylpyrazine", "2,5-Dimethylpyrazine", "2,6-Dimethylpyrazine", "2-Methylpyrazine"
        ]:
            family = "Pyrazine"
        elif compound in [
            "Benzeneacetaldehyde", "Benzyl alcohol", "Phenylethyl alcohol", "Furfural", "5-Methylfurfural"
        ]:
            family = "Aromatic/Furan"
        else:
            family = "Other"

        chemical_family_map[compound] = family

    # --- Apply mappings ---
    aroma_sample_df["Chemical_Family"] = aroma_sample_df["Compound Name"].map(chemical_family_map).fillna("Unknown")
    aroma_sample_df["Aroma_Description"] = aroma_sample_df["Compound Name"].map(aroma_descriptions).fillna("Unknown")

    return aroma_sample_df, aroma_descriptions, chemical_family_map




def plot_aroma_family_distribution(aroma_df):
    """
    Plot the distribution of identified aroma compound families.

    Parameters
    ----------
    aroma_df : pd.DataFrame
        Must contain 'Compound Name' and 'Chemical_Family' columns.

    Returns
    -------
    family_counts : pd.Series
        Number of unique compounds per chemical family.
    """
    if "Compound Name" not in aroma_df.columns or "Chemical_Family" not in aroma_df.columns:
        raise KeyError("DataFrame must include 'Compound Name' and 'Chemical_Family' columns.")

    # Count unique compounds per family
    family_counts = (
        aroma_df[["Compound Name", "Chemical_Family"]]
        .drop_duplicates()
        .groupby("Chemical_Family")
        .size()
        .sort_values(ascending=False)
    )

    # --- Color palette for chemical families ---
    family_palette = {
        "Aldehyde": "#FDD835",          # yellow
        "Alcohol": "#81C784",           # green
        "Ketone": "#64B5F6",            # blue
        "Ester": "#FFB74D",             # orange
        "Pyrazine": "#A1887F",          # brown
        "Sulfur Compound": "#E57373",   # red
        "Acid": "#4DB6AC",              # teal
        "Aromatic/Furan": "#BA68C8",    # purple
        "Other": "#B0BEC5",             # gray
        "Unknown": "#E0E0E0"            # light gray
    }

    # Apply color for families present
    colors = [family_palette.get(fam, "#B0BEC5") for fam in family_counts.index]

    # --- Plot ---
    plt.figure(figsize=(9, 5))
    sns.barplot(
        x=family_counts.values,
        y=family_counts.index,
        palette=colors
    )
    plt.title("Distribution of Identified Aroma Compound Families", fontsize=14, weight="bold")
    plt.xlabel("Number of Unique Compounds", fontsize=12)
    plt.ylabel("Chemical Family", fontsize=12)
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.show()

    return family_counts


def classify_and_plot_families(aroma_sample_df, aroma_descriptions):
    """
    Classify aroma compounds into chemical families and plot their distribution.
    
    Parameters
    ----------
    aroma_sample_df : pd.DataFrame
        DataFrame containing at least a 'Compound Name' column.
    aroma_descriptions : dict
        Dictionary of compound descriptions (used to ensure consistent mapping).
    
    Returns
    -------
    pd.DataFrame
        Updated DataFrame with a new column 'Chemical_Family'.
    """
    chemical_family_map = {}

    for compound in aroma_descriptions.keys():
        if compound in [
            "Propanal", "Butanal, 3-methyl-", "Pentanal", "2-Methylbutanal", "3-Methylbutanal",
            "Hexanal", "Heptanal", "Octanal", "Nonanal", "Decanal", "2-Trans-nonenal",
            "2-Trans-octenal", "4-Heptenal", "2,4-Decadienal", "Phenylacetaldehyde", "Benzaldehyde"
        ]:
            family = "Aldehyde"
        elif compound in [
            "1-Heptanol", "1-Nonanol", "1-Octen-3-ol", "1-Phenylethanol", "Hexanol",
            "2-Methyl-1-propanol", "3-Methyl-1-butanol", "Ethanol"
        ]:
            family = "Alcohol"
        elif compound in [
            "2-Heptanone", "2-Nonanone", "3-Heptanone", "3-Octanone", "2,3-Butanedione",
            "2,3-Pentanedione", "Acetophenone", "2-Propanone, 1-methoxy-", "4-Methyl-2-pentanone",
            "2-Undecanone"
        ]:
            family = "Ketone"
        elif compound in [
            "Methanethiol", "Methional", "Dimethyl disulfide", "Dimethyl trisulfide",
            "Dimethyl tetrasulfide", "Carbon disulfide", "Hydrogen sulfide",
            "2-Methyl-3-furanthiol", "3-Methylthiopropanal", "2-Acetylthiazole",
            "2-Methylthiazole", "2-Methyl-3-thiazoline", "3-(Methylthio)propanal", "Allyl methyl sulfide"
        ]:
            family = "Sulfur Compound"
        elif compound in [
            "Acetic acid", "Butanoic acid", "Hexanoic acid", "Isobutyric acid",
            "Isovaleric acid", "Octanoic acid", "Propanoic acid"
        ]:
            family = "Acid"
        elif compound in [
            "Ethyl acetate", "Isoamyl acetate", "Ethyl butanoate", "Ethyl hexanoate",
            "Methyl butanoate", "Ethyl 2-methylbutanoate", "Ethyl propanoate",
            "Methyl propanoate", "Ethyl pentanoate"
        ]:
            family = "Ester"
        elif compound in [
            "2-Acetylpyrazine", "2-Ethyl-3,5-dimethylpyrazine", "2-Isobutyl-3-methoxypyrazine",
            "2-Isopropyl-3-methoxypyrazine", "2-Ethyl-3-methylpyrazine", "Trimethylpyrazine",
            "Tetramethylpyrazine", "2,5-Dimethylpyrazine", "2,6-Dimethylpyrazine", "2-Methylpyrazine"
        ]:
            family = "Pyrazine"
        elif compound in [
            "Benzeneacetaldehyde", "Benzyl alcohol", "Phenylethyl alcohol", "Furfural", "5-Methylfurfural"
        ]:
            family = "Aromatic/Furan"
        else:
            family = "Other"

        chemical_family_map[compound] = family

    # Map family to DataFrame
    aroma_sample_df["Chemical_Family"] = aroma_sample_df["Compound Name"].map(chemical_family_map)

    # Count unique families
    family_counts = (
        aroma_sample_df[["Compound Name", "Chemical_Family"]]
        .drop_duplicates()["Chemical_Family"]
        .value_counts()
    )

    # --- Visualization ---
    plt.figure(figsize=(8, 5))
    family_counts.plot(kind="bar", color="#CBB680")
    plt.title("Distribution of Identified Aroma Compound Families")
    plt.ylabel("Number of Unique Compounds")
    plt.xlabel("Chemical Family")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

    print("\nChemical Family Counts:\n", family_counts)
    return aroma_sample_df


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_aroma_family_distribution(aroma_df):
    """
    Plot the distribution of identified aroma compound families.

    Parameters
    ----------
    aroma_df : pd.DataFrame
        Must contain 'Compound Name' and 'Chemical_Family' columns.

    Returns
    -------
    family_counts : pd.Series
        Number of unique compounds per chemical family.
    """
    if "Compound Name" not in aroma_df.columns or "Chemical_Family" not in aroma_df.columns:
        raise KeyError("DataFrame must include 'Compound Name' and 'Chemical_Family' columns.")

    # Count unique compounds per family
    family_counts = (
        aroma_df[["Compound Name", "Chemical_Family"]]
        .drop_duplicates()
        .groupby("Chemical_Family")
        .size()
        .sort_values(ascending=False)
    )

    # --- Color palette for chemical families ---
    family_palette = {
        "Aldehyde": "#FDD835",          # yellow
        "Alcohol": "#81C784",           # green
        "Ketone": "#64B5F6",            # blue
        "Ester": "#FFB74D",             # orange
        "Pyrazine": "#A1887F",          # brown
        "Sulfur Compound": "#E57373",   # red
        "Acid": "#4DB6AC",              # teal
        "Aromatic/Furan": "#BA68C8",    # purple
        "Other": "#B0BEC5",             # gray
        "Unknown": "#E0E0E0"            # light gray
    }

    # Apply color for families present
    colors = [family_palette.get(fam, "#B0BEC5") for fam in family_counts.index]

    # --- Plot ---
    plt.figure(figsize=(9, 5))
    sns.barplot(
        x=family_counts.values,
        y=family_counts.index,
        palette=colors
    )
    plt.title("Distribution of Identified Aroma Compound Families", fontsize=14, weight="bold")
    plt.xlabel("Number of Unique Compounds", fontsize=12)
    plt.ylabel("Chemical Family", fontsize=12)
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.show()

    return family_counts

def plot_aroma_filtering_funnel(aroma_sample_df, value_labels=True):
    """
    Visualize the filtering workflow of aroma identification as a simple horizontal funnel chart.

    Parameters
    ----------
    aroma_sample_df : pd.DataFrame
        DataFrame containing aroma compound information. Must include
        columns: 'Compound Name', 'Component RT', and 'm/z'.
    value_labels : bool, optional
        Whether to show numerical values on the bars (default: True).

    Returns
    -------
    pd.DataFrame
        A filtered DataFrame with duplicate peaks removed.
    """

    # --- Step 1: Remove duplicates based on compound + RT + m/z ---
    unique_aroma_df = aroma_sample_df.drop_duplicates(subset=["Compound Name", "Component RT", "m/z"])

    total_before = len(aroma_sample_df)
    total_after = len(unique_aroma_df)
    unique_names = unique_aroma_df["Compound Name"].nunique()

    print(f"Total aroma peaks before removing duplicates: {total_before}")
    print(f"Unique aroma peaks after filtering: {total_after}")
    print(f"Unique compound names: {unique_names}")

    # --- Step 2: Prepare data for visualization ---
    stages = ["Unique after filtering", "Unique compound names"]
    values = [total_after, unique_names]
    colors = ["#26A69A", "#00796B"]

    # --- Step 3: Plot horizontal funnel (simple bar chart) ---
    plt.figure(figsize=(8, 5))
    plt.barh(stages, values, color=colors)
    plt.gca().invert_yaxis()  # Display top-to-bottom

    if value_labels:
        for i, v in enumerate(values):
            plt.text(v + (0.02 * max(values)), i, f"{v}", va="center", fontsize=11, fontweight="bold")

    plt.title("Aroma Identification Filtering Workflow", fontsize=14, pad=10)
    plt.xlabel("Number of Peaks / Compounds")
    plt.grid(False)
    plt.tight_layout()
    plt.show()

    return unique_aroma_df

import json
import pandas as pd
import matplotlib.pyplot as plt

def summarize_and_visualize_aroma_compounds(
    aroma_sample_df,
    aroma_json_path="aroma_descriptions.json",
    family_json_path="chemical_family_map.json",
    output_path=None
):
    """
    Summarize and visualize aroma compounds by chemical family and precursor pathways.

    Parameters
    ----------
    aroma_sample_df : pd.DataFrame
        DataFrame containing aroma compounds with 'Compound Name', 'Component RT', and 'm/z' columns.
    aroma_json_path : str
        Path to JSON file containing aroma descriptions.
    family_json_path : str
        Path to JSON file containing chemical family mappings.
    output_path : str, optional
        Path to save the summary table (e.g., 'aroma_summary.xlsx').

    Returns
    -------
    summary_df : pd.DataFrame
        DataFrame summarizing key aroma compounds and their statistics.
    """
    # --- Load mappings ---
    with open(aroma_json_path, "r", encoding="utf-8") as f:
        AROMA_DESCRIPTIONS = json.load(f)
    with open(family_json_path, "r", encoding="utf-8") as f:
        CHEMICAL_FAMILY_MAP = json.load(f)

    # --- Remove duplicates ---
    unique_aroma_df = aroma_sample_df.drop_duplicates(subset=["Compound Name", "Component RT", "m/z"])

    # --- Add aroma descriptions and chemical families ---
    unique_aroma_df["Chemical_Family"] = unique_aroma_df["Compound Name"].map(CHEMICAL_FAMILY_MAP)
    unique_aroma_df["Aroma_Description"] = unique_aroma_df["Compound Name"].map(AROMA_DESCRIPTIONS)

    # --- Precursor / Pathway mapping ---
    precursor_map = {
        "Nonanal": "Fatty acids",
        "2-Isopropyl-3-methoxypyrazine": "Amino acids",
        "3-Heptanone": "Fatty acids",
        "2-Nonanone": "Fatty acids",
        "Butanal, 3-methyl-": "BCAA (Leucine)",
        "2-Propanone, 1-methoxy-": "Sugar degradation",
        "Isoamyl acetate": "BCAA (Leu/Ile)",
        "Hexanal": "Fatty acids",
        "Octanal": "Fatty acids",
        "Pentanal": "Fatty acids",
        "Acetic acid": "Sugar/Fatty acids",
        "Propanal": "Fatty acids",
        "Tetramethylpyrazine": "Sugar + Amino acids",
        "Ethanol": "Carbohydrate fermentation",
        "Decanal": "Fatty acids",
        "Hexanoic acid": "Fatty acids",
        "Octanoic acid": "Fatty acids",
        "3-Octanone": "Fatty acids",
        "Methyl butanoate": "Fatty acids / Esterification",
        "Phenylacetaldehyde": "Phenylalanine"
    }
    unique_aroma_df["Precursor/Pathway"] = unique_aroma_df["Compound Name"].map(precursor_map)

    # --- Determine abundance column ---
    for possible_col in ["Intensity", "Ratio_normalized", "Log10_normalized", "Intensity_corrected"]:
        if possible_col in unique_aroma_df.columns:
            abundance_col = possible_col
            break
    else:
        raise KeyError("No valid abundance column found!")

    print(f"Using '{abundance_col}' for abundance calculations.")

    # --- Compute statistics ---
    summary_df = (
        unique_aroma_df
        .groupby(["Compound Name", "Chemical_Family", "Precursor/Pathway", "Aroma_Description"])
        .agg(
            Median_Intensity=(abundance_col, "median"),
            Mean_Intensity=(abundance_col, "mean"),
            Std_Intensity=(abundance_col, "std")
        )
        .reset_index()
    )

    # Add Coefficient of Variation
    summary_df["Coefficient of Variation (%)"] = (
        (summary_df["Std_Intensity"] / summary_df["Mean_Intensity"]) * 100
    ).round(1)

    # Sort and rename
    summary_df = (
        summary_df
        .sort_values(by="Median_Intensity", ascending=False)
        .rename(columns={
            "Compound Name": "Compound",
            "Chemical_Family": "Functional Group",
            "Aroma_Description": "Aroma Description",
            "Median_Intensity": "Median (relative intensity)"
        })
    )

    # --- Optional: Save to Excel ---
    if output_path:
        summary_df.to_excel(output_path, index=False)
        print(f"✅ Summary saved to: {output_path}")

    # --- Visualize table ---
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis("off")

    header_color = "#D35400"
    row_colors = ["#FDF2E9", "#FBEEE6"]

    table = ax.table(
        cellText=summary_df[[
            "Compound", "Functional Group", "Precursor/Pathway",
            "Aroma Description", "Coefficient of Variation (%)"
        ]].values,
        colLabels=["Compound", "Functional Group", "Precursor/Pathway",
                   "Aroma Description", "CV (%)"],
        cellLoc="center",
        loc="center",
        colColours=[header_color]*5
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.4)

    for i in range(len(summary_df)):
        color = row_colors[i % 2]
        for j in range(5):
            table[(i+1, j)].set_facecolor(color)

    for key, cell in table.get_celld().items():
        cell.set_edgecolor("white")

    plt.tight_layout()
    plt.show()

    return summary_df


def plot_outlier_vs_normal_aromas(aroma_matrix, outliers, chemical_family_map, output_dir="aroma_cluster_results"):
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    os.makedirs(output_dir, exist_ok=True)

    # --- Prepare Data ---
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

    # --- Select top 5 up/down regulated ---
    top_up = mean_profiles.nlargest(5, "Difference")
    top_down = mean_profiles.nsmallest(5, "Difference")
    top_diff = pd.concat([top_up, top_down]).sort_values("Difference", ascending=True)

    # --- Define color palette ---
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

    # --- Plot ---
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






def create_aroma_intensity_matrix(aroma_sample_df):
    """
    Creates an aroma compound intensity matrix (samples × compounds).

    Parameters
    ----------
    aroma_sample_df : pd.DataFrame
        DataFrame containing aroma compound data with columns:
        ['FileName', 'Variety', 'Compound Name', 'Intensity'].

    Returns
    -------
    aroma_matrix : pd.DataFrame
        Pivoted DataFrame where each row represents a variety/sample
        and each column represents an aroma compound's mean intensity.
    """

    # --- Pivot to create matrix ---
    aroma_pivot = aroma_sample_df.pivot_table(
        index=["FileName", "Variety"],
        columns="Compound Name",
        values="Intensity",
        aggfunc="mean"
    ).reset_index()

    # --- Clean up column names ---
    aroma_pivot.columns.name = None

    # --- Drop filename column (optional, if you only want variety + compounds) ---
    aroma_matrix = aroma_pivot.drop(columns=["FileName"], errors="ignore")

    # --- Summary ---
    print(f"Aroma matrix created with shape: {aroma_matrix.shape}")
    print("\nPreview:")
    print(aroma_matrix.head())

    return aroma_matrix



from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial import distance
from matplotlib.patches import Ellipse
import os

def pca_kmeans_aroma(aroma_matrix, output_dir="aroma_pca_kmeans"):
    """
    PCA + K-Means clustering + Outlier detection + Confidence Ellipses
    Handles both 'Variety' and 'variety' column names automatically.
    """
    os.makedirs(output_dir, exist_ok=True)

    # --- Prepare data ---
    df = aroma_matrix.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    if "variety" in df.columns:
        varieties = df["variety"].astype(str).values
    else:
        df = df.reset_index()
        df.rename(columns={df.columns[0]: "variety"}, inplace=True)
        varieties = df["variety"].astype(str).values

    X = df.drop(columns=["variety"], errors="ignore").select_dtypes(include=[np.number]).fillna(0)
    X_scaled = StandardScaler().fit_transform(X)

    # --- PCA ---
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    print(f"Explained variance (2 PCs): {sum(pca.explained_variance_ratio_)*100:.2f}%")

    # --- Outlier detection (Mahalanobis) ---
    mean_vec = np.mean(X_pca, axis=0)
    cov_matrix = np.cov(X_pca, rowvar=False)
    inv_cov = np.linalg.inv(cov_matrix)
    mahal = [distance.mahalanobis(x, mean_vec, inv_cov) for x in X_pca]

    Q1, Q3 = np.percentile(mahal, [25, 75])
    IQR = Q3 - Q1
    threshold = Q3 + 2 * IQR
    outlier_mask = np.array(mahal) > threshold

    # --- Determine best k for K-Means ---
    sil_scores = []
    K_range = range(2, 8)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X_pca)
        sil_scores.append(silhouette_score(X_pca, labels))

    best_k = K_range[np.argmax(sil_scores)]
    print(f"Optimal number of clusters (based on silhouette): {best_k}")

    # --- Final K-Means ---
    kmeans = KMeans(n_clusters=best_k, random_state=42)
    cluster_labels = kmeans.fit_predict(X_pca)
    centers = kmeans.cluster_centers_

    # --- Create results table ---
    results = pd.DataFrame({
        "Variety": varieties,
        "Cluster": cluster_labels,
        "Mahalanobis": mahal,
        "Outlier": outlier_mask
    })
    results.to_csv(os.path.join(output_dir, "aroma_pca_kmeans_results.csv"), index=False)

    # --- Helper: draw confidence ellipse ---
        # --- Helper: draw confidence ellipse ---
    def draw_ellipse(position, covariance, ax=None, color='gray'):
        if ax is None:
            ax = plt.gca()
        if covariance.shape == (2, 2):
            eigvals, eigvecs = np.linalg.eigh(covariance)
            order = eigvals.argsort()[::-1]
            eigvals, eigvecs = eigvals[order], eigvecs[:, order]
            angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
            width, height = 2 * np.sqrt(eigvals)
        else:
            width, height, angle = 0, 0, 0
        for nsig in range(1, 4):  # 1σ, 2σ, 3σ
            ellipse = Ellipse(position, nsig * width, nsig * height,
                              angle=angle, facecolor=color, edgecolor='none', alpha=0.15)
            ax.add_patch(ellipse)


    # --- Plot PCA with K-Means clusters, centroids, and ellipses ---
    plt.figure(figsize=(10, 7))
    plt.grid(False)
    colors = plt.cm.Set2(np.linspace(0, 1, best_k))

    for i, color in enumerate(colors):
        cluster_points = X_pca[cluster_labels == i]
        cov = np.cov(cluster_points, rowvar=False)
        draw_ellipse(centers[i], cov, color=color)
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                    color=color, s=70, alpha=0.8, edgecolor="k", label=f"Cluster {i+1}")

    # --- Plot centroids ---
    plt.scatter(centers[:, 0], centers[:, 1], c="black", s=180,
                marker="X", edgecolor="white", linewidth=1.5, label="Centroids")

    # --- Highlight outliers ---
    plt.scatter(X_pca[outlier_mask, 0], X_pca[outlier_mask, 1],
                color="red", edgecolor="black", s=130, label="Outliers", zorder=5)
    for i, name in enumerate(varieties):
        if outlier_mask[i]:
            plt.text(X_pca[i, 0]+0.3, X_pca[i, 1], name,
                     fontsize=8, color='darkred', weight='bold')

    # --- Aesthetics ---
    plt.axhline(0, color='gray', lw=0.8, alpha=0.6)
    plt.axvline(0, color='gray', lw=0.8, alpha=0.6)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=12, weight='bold')
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=12, weight='bold')
    plt.title("PCA + K-Means Clustering of Aroma Varieties\n(Centroids, Ellipses, and Outliers Highlighted)",
              fontsize=14, weight='bold', pad=15)
    plt.legend(frameon=True, fontsize=10, loc="best")
    plt.tight_layout()
    plt.grid(False)
    plt.savefig(os.path.join(output_dir, "pca_kmeans_clusters_ellipses.png"),
                dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\nResults saved in: {output_dir}")
    return results, X_pca, cluster_labels



from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.spatial import distance
from sklearn.manifold import TSNE
from matplotlib.patches import Ellipse
import os


def analyze_aroma_clusters_gmm(aroma_matrix, output_dir="aroma_cluster_results"):
    """
    Perform PCA + GMM clustering + outlier detection for aroma compounds.
    Handles both 'Variety' and 'variety' column names automatically.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = aroma_matrix.copy()
    df.columns = [c.strip().lower() for c in df.columns]  # normalize column names

    # --- Handle variety names ---
    if "variety" in df.columns:
        varieties = df["variety"].astype(str).values
    else:
        df = df.reset_index()
        df.rename(columns={df.columns[0]: "variety"}, inplace=True)
        varieties = df["variety"].astype(str).values

    # --- Extract numeric data ---
    X = (
        df.drop(columns=["variety"], errors="ignore")
        .select_dtypes(include=[np.number])
        .fillna(0)
    )
    X_scaled = StandardScaler().fit_transform(X)

    # --- PCA ---
    pca_full = PCA().fit(X_scaled)
    explained = np.cumsum(pca_full.explained_variance_ratio_)
    n_comp = np.argmax(explained >= 0.9) + 1
    print(f"Optimal PCA components: {n_comp} (explains {explained[n_comp-1]*100:.1f}% variance)")

    pca = PCA(n_components=n_comp)
    X_pca = pca.fit_transform(X_scaled)

    # --- GMM model selection ---
    bics, aics = [], []
    k_range = range(2, 11)
    for k in k_range:
        gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=42)
        gmm.fit(X_pca)
        bics.append(gmm.bic(X_pca))
        aics.append(gmm.aic(X_pca))

    plt.figure(figsize=(7, 4))
    plt.plot(k_range, bics, marker='o', label='BIC')
    plt.plot(k_range, aics, marker='s', label='AIC')
    plt.title("GMM Model Selection via BIC and AIC")
    plt.xlabel("Number of Components (k)")
    plt.ylabel("Criterion Value")
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "gmm_model_selection.png"), dpi=300)
    plt.show()

    best_k = k_range[np.argmin(bics)]
    print(f"Best number of clusters (based on BIC): {best_k}")

    # --- Fit final GMM ---
    gmm = GaussianMixture(n_components=best_k, covariance_type='full', random_state=42)
    cluster_labels = gmm.fit_predict(X_pca)
    # --- Evaluate clustering ---
    silhouette = silhouette_score(X_pca, cluster_labels)
    dbi = davies_bouldin_score(X_pca, cluster_labels)
    chi = calinski_harabasz_score(X_pca, cluster_labels)
    print(f"\nClustering metrics for k={best_k}:")
    print(f" Silhouette Score:       {silhouette:.3f}")
    print(f" Davies–Bouldin Index:   {dbi:.3f}")
    print(f" Calinski–Harabasz:      {chi:.1f}")

    # --- Outlier detection ---
    mean_vec = np.mean(X_pca, axis=0)
    cov_matrix = np.cov(X_pca, rowvar=False)
    inv_cov = np.linalg.inv(cov_matrix)
    mahal = [distance.mahalanobis(x, mean_vec, inv_cov) for x in X_pca]
    Q1, Q3 = np.percentile(mahal, [25, 75])
    IQR = Q3 - Q1
    threshold = Q3 + 2 * IQR
    outlier_mask = np.array(mahal) > threshold

    results = pd.DataFrame({
        "Variety": varieties,
        "Cluster": cluster_labels,
        "Mahalanobis": mahal,
        "is_outlier": outlier_mask
    })
    outliers = results.loc[outlier_mask, "Variety"].tolist()
    print(f"\nOutliers detected ({len(outliers)}): {outliers}")

    # --- Helper function for ellipses ---
    def draw_ellipse(position, covariance, ax=None, color='gray'):
        if ax is None:
            ax = plt.gca()
        if covariance.shape == (2, 2):
            eigvals, eigvecs = np.linalg.eigh(covariance)
            order = eigvals.argsort()[::-1]
            eigvals, eigvecs = eigvals[order], eigvecs[:, order]
            angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
            width, height = 2 * np.sqrt(eigvals)
            for nsig in range(1, 3):
                ellipse = Ellipse(position, nsig * width, nsig * height,
                                  angle=angle, color=color, alpha=0.15)
                ax.add_patch(ellipse)

    # --- Enhanced PCA biplot ---
    scores = X_pca[:, :2]
    plt.figure(figsize=(12, 9))
    plt.grid(False)
    colors = plt.cm.tab10(np.linspace(0, 1, best_k))

    for i, color in enumerate(colors):
        cluster_points = scores[cluster_labels == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                    color=color, s=80, alpha=0.85, edgecolor="k", label=f"Cluster {i+1}")
        cov = np.cov(cluster_points, rowvar=False)
        draw_ellipse(cluster_points.mean(axis=0), cov, color=color)
        centroid = cluster_points.mean(axis=0)
        plt.scatter(centroid[0], centroid[1], marker="X", c="black", s=200, edgecolor="white", lw=1.2)
        plt.text(centroid[0]+0.3, centroid[1], f"C{i+1}", fontsize=12, color="black", weight="bold")

    plt.scatter(scores[outlier_mask, 0], scores[outlier_mask, 1],
                color="red", edgecolor="black", s=130, label="Outliers", zorder=5)
    for i, name in enumerate(varieties):
        if outlier_mask[i]:
            plt.text(scores[i, 0], scores[i, 1], name, fontsize=8, color='darkred', weight='bold')

    plt.axhline(0, color='gray', lw=0.8, alpha=0.6)
    plt.axvline(0, color='gray', lw=0.8, alpha=0.6)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)", fontsize=12, weight='bold')
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)", fontsize=12, weight='bold')
    plt.title("PCA + GMM Clustering of Aroma Varieties\n(Centroids, Ellipses, and Outliers Highlighted)",
              fontsize=14, weight='bold', pad=15)
    plt.legend(frameon=True, fontsize=10, loc="best")
    plt.tight_layout()
    plt.grid(False)
    plt.savefig(os.path.join(output_dir, "pca_biplot_clusters_enhanced.png"), dpi=300)
    plt.show()
    # --- Enhanced t-SNE visualization ---
    tsne = TSNE(
    n_components=2,
    perplexity=30,      # امتحان بین 20 تا 40
    learning_rate=200,
    n_iter=1500,
    random_state=42
)
    X_tsne = tsne.fit_transform(X_pca)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=cluster_labels,
                    palette="tab10", s=90, alpha=0.9, edgecolor="k", legend="full")

    plt.scatter(X_tsne[outlier_mask, 0], X_tsne[outlier_mask, 1],
                color="red", edgecolor="black", s=130, label="Outliers")

    tsne_centers = np.array([X_tsne[cluster_labels == i].mean(axis=0) for i in range(best_k)])
    for i, (x, y) in enumerate(tsne_centers):
        plt.text(x, y, f"C{i+1}", fontsize=12, color="black", weight="bold")

    plt.title("t-SNE Visualization of GMM Clusters\n(Centroids + Outliers Highlighted)",
              fontsize=14, weight='bold', pad=15)
    plt.legend(frameon=True, fontsize=10, loc="best")
    plt.tight_layout()
    plt.grid(False)
    plt.savefig(os.path.join(output_dir, "tsne_clusters_enhanced.png"), dpi=300)
    plt.show()

    # --- Compute mean intensity per cluster ---
    cluster_means = pd.DataFrame(X, index=varieties).groupby(cluster_labels).mean()

    # --- Top volatiles per cluster ---
    top_features = {}
    for cluster in cluster_means.index:
        sorted_feats = cluster_means.loc[cluster].sort_values(ascending=False).head(5)
        top_features[cluster] = list(sorted_feats.index)

    pd.DataFrame.from_dict(top_features, orient="index").to_csv(
        os.path.join(output_dir, "top_features_per_cluster.csv")
    )

    print(f"\nResults saved in: {output_dir}")
    return results, cluster_means





def summarize_aroma_intensities(aroma_matrix, top_n=10):
    """
    Summarizes and visualizes aroma compound intensities across potato varieties.

    Parameters
    ----------
    aroma_matrix : pd.DataFrame
        DataFrame containing aroma compound intensity data (with 'Variety' column).
    top_n : int, optional
        Number of top compounds to display based on average intensity (default = 10).

    Returns
    -------
    desc_stats : pd.DataFrame
        DataFrame containing descriptive statistics (mean, std, min, max) for each compound.
    top_compounds : pd.DataFrame
        Top N aroma compounds ranked by mean intensity.
    """
    # ---  Keep only numeric columns (exclude 'Variety' etc.) ---
    numeric_data = aroma_matrix.select_dtypes(include=['number'])

    # ---  Compute descriptive statistics ---
    desc_stats = numeric_data.describe().T  # Transpose for compounds as rows
    # --- 3 Identify top compounds by mean intensity ---
    top_compounds = desc_stats.sort_values(by='mean', ascending=False).head(top_n)
    top_compounds_plot = top_compounds[['mean', 'std']]
    # ---  Print summary ---
    print(f" Total compounds analyzed: {numeric_data.shape[1]}")
    print(f" Top {top_n} aroma compounds by average intensity:\n")
    print(top_compounds_plot.round(2))
    # ---  Visualization: bar plot with error bars ---
    plt.figure(figsize=(10, 6))
    plt.barh(
        top_compounds_plot.index,
        top_compounds_plot['mean'],
        xerr=top_compounds_plot['std'],
        color='skyblue',
        edgecolor='black'
    )
    plt.xlabel("Mean Intensity (±1 SD)")
    plt.title(f"Top {top_n} Aroma Compounds in Potato Varieties by Mean Intensity")
    plt.gca().invert_yaxis()  # highest at top
    plt.tight_layout()
    plt.show()

    return desc_stats, top_compounds


def plot_top_aroma_heatmap(aroma_matrix, top_n=5, mode="per_variety"):
    """
    Plots a heatmap showing the most intense aroma compounds in potato varieties.

    Parameters
    ----------
    aroma_matrix : pd.DataFrame
        DataFrame containing aroma compound intensities with columns:
        ['Variety', compound_1, compound_2, ...]
    top_n : int, optional
        Number of top compounds to select (default = 5).
    mode : str, optional
        'per_variety' → selects top_n compounds separately for each variety
        'overall' → selects top_n compounds based on overall mean intensity across varieties

    Returns
    -------
    None
        Displays a heatmap visualization of aroma intensities.
    """

    # --- Step 1: Prepare data ---
    aroma_df = aroma_matrix.copy()
    if "Variety" not in aroma_df.columns:
        raise ValueError("DataFrame must contain a 'Variety' column.")

    # --- Step 2: Melt into long format ---
    aroma_long = aroma_df.melt(
        id_vars="Variety",
        var_name="Compound",
        value_name="Intensity"
    )
    # --- Step 3: Select top aromas based on mode ---
    if mode == "per_variety":
        # Top N compounds per variety
        top_aromas = (
            aroma_long.groupby("Variety", group_keys=False)
            .apply(lambda g: g.nlargest(top_n, "Intensity"))
            .reset_index(drop=True)
        )
        title_suffix = f"Top {top_n} Compounds per Variety"

    elif mode == "overall":
        # Top N compounds overall
        top_compounds = (
            aroma_long.groupby("Compound")["Intensity"].mean()
            .nlargest(top_n)
            .index
        )
        top_aromas = aroma_long[aroma_long["Compound"].isin(top_compounds)]
        title_suffix = f"Top {top_n} Overall Compounds Across Varieties"

    else:
        raise ValueError("mode must be either 'per_variety' or 'overall'")

    # --- Step 4: Pivot for heatmap ---
    heatmap_data = top_aromas.pivot_table(
        index="Variety",
        columns="Compound",
        values="Intensity",
        fill_value=0
    )

    # --- Step 5: Plot heatmap ---
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        heatmap_data,
        cmap="YlOrRd",
        cbar_kws={'label': 'Intensity'},
        linewidths=0.2
    )
    plt.title(f"Aroma Intensity Heatmap – {title_suffix}")
    plt.ylabel("Potato Variety")
    plt.xlabel("Aroma Compounds")
    plt.tight_layout()
    plt.show()

def launch_aroma_dashboard(aroma_matrix, port=7031):
    """
    Launches an interactive dashboard to explore aroma compound intensities across potato varieties.

    Parameters
    ----------
    aroma_matrix : pd.DataFrame
        A DataFrame where rows represent potato varieties and columns represent aroma compound intensities.
        Must contain a 'Variety' column.
    port : int, optional
        Port number to run the dashboard (default = 8097).

    Description
    -----------
    The dashboard provides:
    - A dropdown menu to select a potato variety.
    - A bar chart showing the aroma profile (compound intensities) of the selected variety.
    - A heatmap showing compound intensities across all varieties.
    """

    # --- Prepare data ---
    if "Variety" in aroma_matrix.columns:
        BASE_REL = aroma_matrix.set_index("Variety").copy()
    else:
        BASE_REL = aroma_matrix.copy()

    BASE_REL.index = BASE_REL.index.astype(str)
    BASE_REL.columns = [str(c) for c in BASE_REL.columns]

    variety_options = [{"label": v, "value": v} for v in BASE_REL.index]

    # --- Initialize Dash app ---
    app = dash.Dash(__name__)

    app.layout = html.Div([
        html.H1(" Potato Aroma Compound Dashboard"),

        html.Label("Select Variety:"),
        dcc.Dropdown(
            id="variety-dropdown",
            options=variety_options,
            value=BASE_REL.index[0],
            clearable=False
        ),

        dcc.Graph(id="compound-barplot"),
        dcc.Graph(id="compound-heatmap"),
    ])

    # --- Define callback ---
    @app.callback(
        [Output("compound-barplot", "figure"),
         Output("compound-heatmap", "figure")],
        [Input("variety-dropdown", "value")]
    )
    def update_dashboard(selected_variety):
        # Extract selected variety data
        s = BASE_REL.loc[selected_variety]
        s = pd.to_numeric(s, errors="coerce").fillna(0)
        s = s.sort_values(ascending=False).reset_index()
        s.columns = ["Compound", "Relative Concentration"]

        # Bar chart
        fig_bar = px.bar(
            s, x="Compound", y="Relative Concentration",
            title=f"Aroma Profile for {selected_variety}",
            color="Relative Concentration",
            color_continuous_scale="Blues"
        )
        fig_bar.update_layout(
            xaxis_tickangle=-45,
            height=450,
            margin=dict(l=60, r=20, t=60, b=120)
        )

        # Heatmap (all varieties × compounds)
        heat_df = BASE_REL.apply(pd.to_numeric, errors="coerce").fillna(0).T
        fig_heat = px.imshow(
            heat_df,
            aspect="auto",
            color_continuous_scale="Viridis",
            labels={"x": "Variety", "y": "Compound", "color": "Relative Conc."},
            title="Heatmap of Compounds Across Varieties"
        )
        fig_heat.update_layout(
            height=600,
            margin=dict(l=90, r=20, t=60, b=120)
        )

        return fig_bar, fig_heat

    # --- Run app ---
    print(f"\n Launching Potato Aroma Dashboard at: http://127.0.0.1:{port}")
    app.run(debug=True, port=port)




def compute_feature_importance_from_pca(aroma_matrix, n_components=2):
    """
    Computes feature importance of aroma compounds based on PCA loadings.

    Parameters
    ----------
    aroma_matrix : pd.DataFrame
        Aroma intensity matrix with 'Variety' column and compound intensity columns.
    n_components : int, optional
        Number of PCA components to include when computing feature importance (default = 2).

    Returns
    -------
    importance_df : pd.DataFrame
        DataFrame showing compounds ranked by absolute PCA loading (importance).
    """
    # ---  Prepare Data ---
    X = aroma_matrix.drop(columns=["Variety"], errors="ignore").fillna(0)
    X_scaled = StandardScaler().fit_transform(X)

    # ---  PCA ---
    pca = PCA(n_components=n_components)
    pca.fit(X_scaled)
    loadings = pd.DataFrame(
        np.abs(pca.components_.T),
        index=X.columns,
        columns=[f"PC{i+1}" for i in range(n_components)]
    )

    # ---  Compute overall importance (sum of abs loadings across PCs) ---
    loadings["Total_Importance"] = loadings.sum(axis=1)
    importance_df = loadings.sort_values(by="Total_Importance", ascending=False)

    # ---  Visualization ---
    top_n = 15
    plt.figure(figsize=(10,6))
    plt.barh(importance_df.head(top_n).index[::-1],
              importance_df["Total_Importance"].head(top_n)[::-1],
              color='goldenrod')
    plt.xlabel("Total PCA Loading (|contribution|)")
    plt.title(f"Top {top_n} Aroma Compounds Contributing to PCA Variance")
    plt.tight_layout()
    plt.show()

    print(f" Top {top_n} most contributing compounds:")
    print(importance_df.head(top_n))

    return importance_df




sensory_cols = [
    "Metallic Flavour","Bitter Flavour","Earthy Flavour","Sour Flavour",
    "Fresh Flavour","Sweet Flavour","Root/ Vegetable Flavour",
    "Farmyard (grass/hay) flavour","Bitter Aftertaste","Sour Aftertaste","Sweet Aftertaste", 'Earthy Aroma',
    'Farmyard', 'Sweet', 'Starchy', 'Damp', 'Buttery', 'Fresh', 'Potato Starch','Metalic'
]
def summarize_sensory_traits(sensory_df, flavor_columns):
    """
    Cleans and summarizes sensory (flavour and aroma) data per potato variety.

    Parameters
    ----------
    sensory_df : pd.DataFrame
        Raw sensory data containing columns for flavour intensity and a 'VARIETY' column.
    flavor_columns : list
        List of sensory-related column names to include in the summary.

    Returns
    -------
    sensory_summary : pd.DataFrame
        DataFrame containing the mean sensory trait values for each variety.
    """

    #  Clean variety names 
    sensory_df = sensory_df.copy()
    sensory_df["VARIETY"] = sensory_df["VARIETY"].astype(str).str.strip()

    # --- Step 2: Remove any unnamed or empty columns 
    sensory_df = sensory_df.drop(
        columns=[c for c in sensory_df.columns if str(c).startswith("Unnamed")],
        errors="ignore"
    )

    # Keep only numeric sensory columns + variety ---
    sensory_traits = sensory_df[sensory_cols].apply(pd.to_numeric, errors="coerce")
    sensory_traits = sensory_traits.assign(VARIETY=sensory_df["VARIETY"])

    #  Group by variety and average traits 
    sensory_summary = (
        sensory_traits
        .groupby("VARIETY", as_index=False)
        .mean(numeric_only=True)
    )

    # Print summary info 
    print(f" Rows (unique varieties): {sensory_summary['VARIETY'].nunique()}")
    print(f" Traits averaged: {len(flavor_columns)}")
    print("\n Example of summarized sensory data:")
    print(sensory_summary.head())

    return sensory_summary

def plot_sensory_vs_flavour(df, base_trait="POTATO_FLAVOUR"):
    """
    Creates scatter + regression plots between potato flavour
    and all other sensory traits (based on PCA sensory features).
    """
    # --- Standardize column names ---
    df = df.copy()
    df.columns = df.columns.str.strip().str.upper()

    if base_trait not in df.columns:
        raise KeyError(f"'{base_trait}' not found in dataframe columns!")

    # --- Sensory traits (from your PCA biplot) ---
    sensory_traits = [
        "SWEET FLAVOUR", "BITTER FLAVOUR", "EARTHY FLAVOUR", 
        "SOUR FLAVOUR", "FRESH FLAVOUR", "SWEET AFTERTASTE", "BITTER AFTERTASTE",
        "SOUR AFTERTASTE", "FARMYARD (GRASS/HAY) FLAVOUR", "ROOT/ VEGETABLE FLAVOUR",
        "METALLIC FLAVOUR", "DAMP", "BUTTERY", "POTATO STARCH", "EARTHY AROMA"
    ]
    sensory_traits = [t for t in sensory_traits if t in df.columns]

    # --- Color mapping similar to your reference image ---
    colors = {
        "SWEET FLAVOUR": "royalblue",
        "TEXTURE": "red",
        "SOUR FLAVOUR": "green",
        "BITTER FLAVOUR": "orange",
        "SWEET AFTERTASTE": "purple",
        "BUTTERY": "crimson",
        "POTATO STARCH": "darkred",
        "EARTHY FLAVOUR": "brown",
        "METALLIC FLAVOUR": "darkslategray"
    }

    plt.figure(figsize=(10, 6))

    for trait in sensory_traits:
        sns.regplot(
            data=df,
            x=base_trait,
            y=trait,
            scatter_kws={"alpha": 0.7, "s": 45},
            line_kws={"lw": 2},
            color=colors.get(trait, None),
            label=trait.title()
        )

    plt.xlabel("Potato Flavour", fontsize=12, weight="bold")
    plt.ylabel("Sensory Score", fontsize=12)
    plt.title("Relationship between Potato Flavour and Sensory Traits", fontsize=14, weight="bold")
    plt.legend(title="Sensory Traits", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.grid(False)
    plt.show()



def merge_aroma_with_sensory(aroma_sample_df, sensory_summary):
    """
    Merge aroma compound data with sensory panel data based on potato variety names.
    Handles case differences ('VARIETY', 'variety', etc.) automatically.
    """

    # --- Drop redundant columns from sensory data ---
    cols_to_drop = [c for c in ["level_0", "index"] if c in sensory_summary.columns]
    sensory_summary = sensory_summary.drop(columns=cols_to_drop, errors="ignore")

    # --- Identify variety column in both datasets ---
    def find_variety_col(df):
        for c in df.columns:
            if c.strip().lower() == "variety":
                return c
        raise KeyError(" No 'Variety' column found in DataFrame!")

    variety_col_aroma = find_variety_col(aroma_sample_df)
    variety_col_sensory = find_variety_col(sensory_summary)

    # --- Standardize variety column names ---
    aroma_sample_df = aroma_sample_df.rename(columns={variety_col_aroma: "Variety"})
    sensory_summary = sensory_summary.rename(columns={variety_col_sensory: "Variety"})

    # --- Clean and normalize names ---
    aroma_sample_df["Variety"] = aroma_sample_df["Variety"].astype(str).str.strip().str.upper()
    sensory_summary["Variety"] = sensory_summary["Variety"].astype(str).str.strip().str.upper()

    # --- Report overlaps ---
    common = set(aroma_sample_df["Variety"]).intersection(set(sensory_summary["Variety"]))
    print(f" Varieties in Aroma Data: {len(aroma_sample_df['Variety'].unique())}")
    print(f" Varieties in Sensory Data: {len(sensory_summary['Variety'].unique())}")
    print(f" Common Varieties for Merging: {len(common)}\n")

    # --- Merge ---
    merged_aroma_sensory = pd.merge(aroma_sample_df, sensory_summary, on="Variety", how="inner")

    print(f" Number of merged varieties: {merged_aroma_sensory['Variety'].nunique()}")
    print(" Preview of merged data:")
    display(merged_aroma_sensory.head())

    return merged_aroma_sensory






import os

def plot_aroma_sensory_correlation(
    merged_aroma_sensory,
    sensory_cols,
    aroma_descriptions=None,
    threshold=0.1,
    min_compounds=15,
    save_path=None
):
    """
    Compute and visualize correlations between aroma compound intensities
    and sensory traits, ensuring at least `min_compounds` are displayed.

    Parameters
    ----------
    merged_aroma_sensory : pd.DataFrame
        Combined dataset with both aroma intensities and sensory scores.
    sensory_cols : list
        List of sensory attribute column names.
    aroma_descriptions : dict, optional
        Mapping of compound names to aroma descriptions.
    threshold : float, default=0.1
        Minimum absolute correlation value for inclusion.
    min_compounds : int, default=15
        Minimum number of compounds to display (select strongest if fewer pass threshold).
    save_path : str, optional
        Path to save correlation matrix (CSV).
    """

    df = merged_aroma_sensory.copy()

    # --- Standardize column names ---
    df.columns = df.columns.str.strip().str.lower()
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.loc[:, df.nunique() > 1]  # remove constant columns

    # --- Normalize sensory column names ---
    sensory_cols = [c.strip().lower() for c in sensory_cols]

    # --- Keep numeric data ---
    df = df.select_dtypes(include=["number"]).replace([np.inf, -np.inf], np.nan)
    df = df.dropna(axis=1, thresh=max(3, len(df) // 2))
    df = df.dropna(axis=0, how="any")

    # --- Identify sensory vs aroma features ---
    sensory_cols_found = [c for c in df.columns if c in sensory_cols]
    compound_cols = [c for c in df.columns if c not in sensory_cols_found]

    print(f" Found {len(compound_cols)} aroma compounds and {len(sensory_cols_found)} sensory traits.")

    if not sensory_cols_found or not compound_cols:
        raise ValueError(" No valid sensory or compound columns found for correlation analysis.")

    # --- Compute correlation matrix ---
    corr_matrix = df[compound_cols + sensory_cols_found].corr()
    corr_block = corr_matrix.loc[compound_cols, sensory_cols_found].fillna(0)

    # --- Filter by correlation threshold ---
    strong_corr = corr_block[(corr_block.abs() > threshold).any(axis=1)]

    # --- Ensure at least `min_compounds` are shown ---
    if strong_corr.shape[0] < min_compounds:
        top_compounds = (
            corr_block.abs().max(axis=1)
            .sort_values(ascending=False)
            .head(min_compounds)
            .index
        )
        strong_corr = corr_block.loc[top_compounds]

    print(f" Displaying {strong_corr.shape[0]} compounds (threshold={threshold})")

    # --- Add aroma descriptions ---
    if aroma_descriptions:
        annotated_index = []
        for compound in strong_corr.index:
            clean_name = compound.strip().title()
            desc = aroma_descriptions.get(clean_name, "No description")
            annotated_index.append(f"{clean_name}\n({desc})")
        strong_corr.index = annotated_index

    # --- Plot ---
    plt.figure(figsize=(16, 9))
    sns.heatmap(
        strong_corr,
        cmap="coolwarm",
        center=0,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={'label': 'Correlation (r)'}
    )
    plt.title("Correlation Between Aroma Compounds and Sensory Attributes", fontsize=16, pad=15)
    plt.ylabel("Aroma Compounds (Top Correlated with Sensory Attributes)")
    plt.xlabel("Sensory Attributes")
    plt.tight_layout()
    plt.show()

    # --- Optional: save results ---
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        strong_corr.to_csv(save_path)
        print(f" Correlation matrix saved to: {save_path}")

    return strong_corr


def plot_key_aroma_sensory_relationships(
    df,
    x_col="Correlation_Strength",
    y_col="Compound",
    sensory_col="Key_Sensory_Associations",
    desc_col="Description",
    title="Key Aroma–Sensory Relationships in Potato Aroma Profile",
    save_path=None
):
    """
    Create a bubble plot showing key aroma–sensory relationships.

    Parameters
    ----------
    df : pd.DataFrame
        Must include ['Compound', 'Key_Sensory_Associations', 'Description', 'Correlation_Strength'].
    x_col : str
        Column for correlation values.
    y_col : str
        Column for aroma compound names.
    sensory_col : str
        Column for sensory associations.
    desc_col : str
        Column for aroma descriptions.
    title : str
        Plot title.
    save_path : str, optional
        If provided, the figure is saved to this location.

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt.figure(figsize=(12, 7))
    sns.set(style="whitegrid")

    # Bubble plot
    sns.scatterplot(
        data=df,
        x=x_col,
        y=y_col,
        size=x_col,
        sizes=(200, 900),
        hue=x_col,
        palette="YlOrRd",
        edgecolor="black",
        legend=False
    )

    # Add text annotations
    for i, row in df.iterrows():
        plt.text(
            row[x_col] + 0.015,
            i,
            f"{row[sensory_col]}\n({row[desc_col]})",
            fontsize=9.5,
            va='center',
            ha='left',
            color="#333333"
        )

    # Titles and layout
    plt.title(title, fontsize=16, weight="bold", pad=15)
    plt.xlabel("Correlation Strength (r)", fontsize=12)
    plt.ylabel("Aroma Compounds", fontsize=12)
    plt.xlim(0.6, 1.02)
    plt.grid(axis='x', linestyle='--', alpha=0.4)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f" Figure saved to: {save_path}")

    plt.show()
    return plt.gcf()




from sklearn.model_selection import cross_val_predict, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

def predict_sensory_from_models(
    merged_aroma_sensory,
    aroma_matrix,
    sensory_cols,
    model_type="PLSR", 
    n_components=3,
    cv_folds=5,
    alpha=1.0
):
    """
    Predict sensory flavor attributes from aroma compound intensities
    using various linear models (PLSR, Linear, Ridge, Lasso, ElasticNet).

    Parameters
    ----------
    merged_aroma_sensory : pd.DataFrame
        Combined dataset with both aroma compound and sensory data.
    aroma_matrix : pd.DataFrame
        Aroma compound intensity matrix for all varieties.
    sensory_cols : list
        List of sensory attribute column names.
    model_type : str, default='PLSR'
        Type of model to use ('PLSR', 'Linear', 'Ridge', 'Lasso', or 'ElasticNet').
    n_components : int, default=3
        Number of PLS components (used only for PLSR).
    cv_folds : int, default=5
        Number of folds for cross-validation.
    alpha : float, default=1.0
        Regularization strength (used for Ridge, Lasso, ElasticNet).

    Returns
    -------
    predicted_df : pd.DataFrame
        Predicted sensory values and mean aroma–flavor (MAF) score.
    r2_summary : pd.DataFrame
        Table of R² and CVR² for each sensory trait.
    """

    # --- Select features ---
    compound_cols = [
        c for c in merged_aroma_sensory.columns
        if c not in sensory_cols and c.lower() != "variety"
    ]

    X = merged_aroma_sensory[compound_cols].select_dtypes(include=[np.number])
    Y = merged_aroma_sensory[sensory_cols].select_dtypes(include=[np.number])

    X = X.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="any")
    Y = Y.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="any")
    X = X.loc[:, X.std() > 0]
    Y = Y.loc[:, Y.std() > 0]

    print(f"Training with {X.shape[0]} samples, {X.shape[1]} aroma features, and {Y.shape[1]} sensory traits.")

    # --- Standardize ---
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    Y_scaled = scaler_Y.fit_transform(Y)

    # --- Choose model ---
    if model_type == "PLSR":
        model = PLSRegression(n_components=min(n_components, X.shape[1], Y.shape[1]))
    elif model_type == "Linear":
        model = LinearRegression()
    elif model_type == "Ridge":
        model = Ridge(alpha=alpha)
    elif model_type == "Lasso":
        model = Lasso(alpha=alpha)
    elif model_type == "ElasticNet":
        model = ElasticNet(alpha=alpha, l1_ratio=0.5)
    else:
        raise ValueError("Invalid model_type. Choose 'PLSR', 'Linear', 'Ridge', 'Lasso', or 'ElasticNet'.")

    print(f"\nFitting model: {model_type}")

    # --- Fit model ---
    model.fit(X_scaled, Y_scaled)

    # --- In-sample prediction ---
    Y_pred_scaled = model.predict(X_scaled)
    Y_pred = scaler_Y.inverse_transform(Y_pred_scaled)

    # --- Cross-validation prediction ---
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    Y_cv_pred_scaled = cross_val_predict(model, X_scaled, Y_scaled, cv=kf)
    Y_cv_pred = scaler_Y.inverse_transform(Y_cv_pred_scaled)

    # --- Compute R² and CVR² ---
    r2_scores = {col: r2_score(Y[col], Y_pred[:, i]) for i, col in enumerate(Y.columns)}
    cvr2_scores = {col: r2_score(Y[col], Y_cv_pred[:, i]) for i, col in enumerate(Y.columns)}

    r2_df = pd.DataFrame({
        "Sensory Attribute": Y.columns,
        "R²": [r2_scores[c] for c in Y.columns],
        "CVR²": [cvr2_scores[c] for c in Y.columns]
    }).sort_values("CVR²", ascending=False)

    print("\nCross-validated performance summary:")
    print(r2_df)

    # --- Predict all varieties ---
    aroma_reset = aroma_matrix.reset_index(drop=True)
    common_cols = [c for c in X.columns if c in aroma_reset.columns]
    all_X = aroma_reset[common_cols].fillna(0)
    all_X_scaled = scaler_X.transform(all_X)
    Y_all_pred = scaler_Y.inverse_transform(model.predict(all_X_scaled))

    predicted_df = pd.DataFrame(Y_all_pred, columns=Y.columns)
    predicted_df["Variety"] = aroma_reset.get("Variety", aroma_reset.index)
    predicted_df["MAF_score"] = predicted_df[Y.columns].mean(axis=1)

    # --- Visualization ---
    plt.figure(figsize=(8, 4))
    heat_df = r2_df.set_index("Sensory Attribute")[["R²", "CVR²"]]
    sns.heatmap(heat_df.T, annot=True, fmt=".2f", cmap="YlOrRd", cbar=False)
    plt.title(f"Model Performance per Sensory Attribute ({model_type})", fontsize=14, pad=15)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    return predicted_df, r2_df





from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def predict_sensory_with_models_and_cvr2(
    merged_aroma_sensory,
    aroma_matrix,
    sensory_cols,
    n_estimators=500,
    n_components=3,
    cv_folds=5
):
    """
    Compare linear and nonlinear models for predicting sensory attributes
    from aroma compound intensities.
    (Optimized for small datasets — excludes neural networks)

    Linear model:
        - PLSR (Partial Least Squares Regression)

    Nonlinear models:
        - Random Forest
        - Extra Trees
        - XGBoost
        - SVR (RBF)
    """

    # --- Clean column names ---
    merged_aroma_sensory.columns = merged_aroma_sensory.columns.str.strip().str.lower()
    aroma_matrix.columns = aroma_matrix.columns.str.strip().str.lower()
    sensory_cols = [c.lower() for c in sensory_cols]

    # --- Split data ---
    compound_cols = [c for c in merged_aroma_sensory.columns if c not in sensory_cols and c != "variety"]
    X = merged_aroma_sensory[compound_cols].select_dtypes(include=[np.number]).fillna(0)
    Y = merged_aroma_sensory[sensory_cols].select_dtypes(include=[np.number]).fillna(0)

    # --- Standardize ---
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    Y_scaled = scaler_Y.fit_transform(Y)

    # --- Define models ---
    models = {
        "Random Forest": RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1),
        "Extra Trees": ExtraTreesRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1),
        "XGBoost": XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42, n_jobs=-1),
        "SVR (RBF)": SVR(kernel="rbf", C=1.0, epsilon=0.1)
    }

    # --- Prepare results ---
    r2_results = {m: [] for m in models.keys()}
    cvr2_results = {m: [] for m in models.keys()}
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

    # --- Train & evaluate ---
    for trait_idx, trait in enumerate(Y.columns):
        y = Y_scaled[:, trait_idx]
        for name, model in models.items():
            try:
                model.fit(X_scaled, y)
                y_pred = model.predict(X_scaled)
                y_cv_pred = cross_val_predict(model, X_scaled, y, cv=kf)

                r2_results[name].append(r2_score(y, y_pred))
                cvr2_results[name].append(r2_score(y, y_cv_pred))
            except Exception as e:
                print(f" {name} failed for {trait}: {e}")
                r2_results[name].append(np.nan)
                cvr2_results[name].append(np.nan)

    # --- Combine results ---
    r2_df = pd.DataFrame(r2_results, index=Y.columns)
    cvr2_df = pd.DataFrame(cvr2_results, index=Y.columns)
    summary = pd.concat({"R²": r2_df.T, "CVR²": cvr2_df.T}).round(2)

    print("\nModel Performance Summary (R² and CVR²):")
    print(summary)

    # --- Visualization ---
    plt.figure(figsize=(14, 6))
    sns.heatmap(summary, annot=True, fmt=".2f", cmap="YlOrRd", linewidths=0.5)
    plt.title("Model Performance per Sensory Attribute (Linear vs Nonlinear)", fontsize=14)
    plt.xlabel("Sensory Attribute")
    plt.ylabel("Metric / Model")
    plt.tight_layout()
    plt.show()

    # --- Average performance ---
    avg_scores = pd.DataFrame({
        "R²": r2_df.mean(axis=0),
        "CVR²": cvr2_df.mean(axis=0)
    }).sort_values("CVR²", ascending=False)

    avg_scores.plot(kind="bar", figsize=(9, 5), color=["#1d3557", "#e63946"])
    plt.title("Average Model Performance Across Sensory Traits", fontsize=14)
    plt.ylabel("Average Score")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="")
    plt.tight_layout()
    plt.show()

    return r2_df, cvr2_df, summary




from sklearn.linear_model import ElasticNet

def predict_sensory_from_elasticnet(merged_aroma_sensory, aroma_matrix, sensory_cols,
                                    alpha=0.1, l1_ratio=0.5, max_iter=10000):
    """
    Predicts sensory traits from aroma compound intensities using ElasticNet Regression.
    Handles small naming mismatches automatically.
    """

    # ---  Normalize column names for reliable matching ---
    merged_aroma_sensory.columns = merged_aroma_sensory.columns.str.strip().str.lower()
    aroma_matrix.columns = aroma_matrix.columns.str.strip().str.lower()
    sensory_cols_norm = [c.lower().strip() for c in sensory_cols]

    # ---  Select compound and sensory data ---
    compound_cols = [
        c for c in merged_aroma_sensory.columns 
        if c not in sensory_cols_norm and c != "variety"
    ]

    X = merged_aroma_sensory[compound_cols].select_dtypes(include=[np.number]).fillna(0)
    Y = merged_aroma_sensory[[c for c in sensory_cols_norm if c in merged_aroma_sensory.columns]].fillna(0)

    if Y.empty:
        raise ValueError(" None of the sensory columns were found in merged_aroma_sensory!")

    # --- Standardize ---
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    Y_scaled = scaler_Y.fit_transform(Y)

    # --- Train ElasticNet on matched varieties ---
    elastic = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42, max_iter=max_iter)
    elastic.fit(X_scaled[:len(Y_scaled)], Y_scaled)

    # --- Predict for all varieties in aroma_matrix ---
    aroma_reset = aroma_matrix.reset_index(drop=True)
    common_cols = [c for c in X.columns if c in aroma_reset.columns]

    if not common_cols:
        raise ValueError(" No matching compound columns found between aroma_matrix and training data!")

    all_X = aroma_reset[common_cols].fillna(0)
    all_X_scaled = scaler_X.transform(all_X)
    elastic_pred_scaled = elastic.predict(all_X_scaled)

    # Fix shape if single output
    if elastic_pred_scaled.ndim == 1:
        elastic_pred_scaled = elastic_pred_scaled.reshape(-1, 1)

    elastic_pred = scaler_Y.inverse_transform(elastic_pred_scaled)

    # ---  Build results dataframe ---
    elastic_df = pd.DataFrame(elastic_pred, columns=Y.columns)
    if "variety" in aroma_reset.columns:
        elastic_df["Variety"] = aroma_reset["variety"].values
    else:
        elastic_df["Variety"] = aroma_reset.index.astype(str)

    elastic_df["MAF_score"] = elastic_df[Y.columns].mean(axis=1)

    print("\n ElasticNet - Top 10 Predicted Varieties by Mean Flavor Score:")
    print(elastic_df[["Variety", "MAF_score"]].sort_values("MAF_score", ascending=False).head(10))

    # ---  Plot measured vs predicted ---
    Y_true = Y.values
    Y_pred = elastic_pred[:len(Y_true)]

    n_traits = len(Y.columns)
    ncols = 4
    nrows = (n_traits + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
    axes = axes.ravel()

    for i, trait in enumerate(Y.columns):
        ax = axes[i]
        x = Y_true[:, i]
        y = Y_pred[:, i]
        ax.scatter(x, y, alpha=0.7)
        ax.plot([x.min(), x.max()], [x.min(), x.max()], "r--")
        r2 = r2_score(x, y)
        ax.set_title(f"{trait}\nR² = {r2:.2f}")
        ax.set_xlabel("Measured")
        ax.set_ylabel("Predicted")

    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle("ElasticNet Model Performance (Measured vs Predicted)", fontsize=16)
    plt.tight_layout()
    plt.show()

    return elastic_df

from sklearn.neighbors import NearestNeighbors
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
    Merge aroma + sensory data, impute missing sensory traits using KNN,
    perform PCA on sensory traits, visualize projection, loadings heatmap, and biplot.
    """

    #  Standardize column names ---
    aroma_matrix = aroma_matrix.copy()
    sensory_summary = sensory_summary.copy()
    aroma_matrix.columns = aroma_matrix.columns.str.strip().str.upper()
    sensory_summary.columns = sensory_summary.columns.str.strip().str.upper()

    # Check for Variety column ---
    if "VARIETY" not in aroma_matrix.columns:
        raise KeyError(" 'VARIETY' column not found in aroma_matrix!")
    if "VARIETY" not in sensory_summary.columns:
        raise KeyError(" 'VARIETY' column not found in sensory_summary!")

    #Merge datasets 
    merged = aroma_matrix.merge(sensory_summary, on="VARIETY", how="left")
    merged["IS_PANEL"] = merged["VARIETY"].isin(sensory_summary["VARIETY"])

    # Split data
    panel_df = merged[merged["IS_PANEL"]].reset_index(drop=True)
    nonpanel_df = merged[~merged["IS_PANEL"]].reset_index(drop=True)

    # Define columns 
    sensory_cols = [c for c in sensory_summary.columns if c != "VARIETY"]
    aroma_cols = [c for c in aroma_matrix.columns if c != "VARIETY"]

    #  Prepare data matrices 
    Y_panel = panel_df[sensory_cols].fillna(0)
    X_panel = panel_df[aroma_cols].fillna(0)
    X_nonpanel = nonpanel_df[aroma_cols].fillna(0)

    #KNN imputation for sensory traits
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(X_panel)
    distances, indices = nn.kneighbors(X_nonpanel)
    Y_nonpanel_pred = np.array([Y_panel.iloc[idx].mean(axis=0) for idx in indices])

    # PCA projection on sensory traits
    scaler = StandardScaler().fit(Y_panel.values)
    X_panel_scaled = scaler.transform(Y_panel.values)
    X_nonpanel_scaled = scaler.transform(Y_nonpanel_pred)

    pca = PCA(n_components=n_components, random_state=0)
    pcs_panel = pca.fit_transform(X_panel_scaled)
    pcs_nonpanel = pca.transform(X_nonpanel_scaled)

    # Combine results 
    panel_pcs = pd.DataFrame(pcs_panel, columns=[f"PC{i+1}" for i in range(n_components)])
    panel_pcs["Variety"] = panel_df["VARIETY"].values
    panel_pcs["is_panel"] = True

    nonpanel_pcs = pd.DataFrame(pcs_nonpanel, columns=[f"PC{i+1}" for i in range(n_components)])
    nonpanel_pcs["Variety"] = nonpanel_df["VARIETY"].values
    nonpanel_pcs["is_panel"] = False

    pc_df = pd.concat([panel_pcs, nonpanel_pcs], ignore_index=True)

    # Compute loadings 
    loadings_df = pd.DataFrame(
        pca.components_.T,
        columns=[f"PC{i+1}" for i in range(n_components)],
        index=sensory_cols
    )

    print("\n PCA explained variance ratio:", np.round(pca.explained_variance_ratio_, 3))
    print(" Total projected varieties:", pc_df["Variety"].nunique())

    # Plot loadings heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(loadings_df, annot=True, cmap="coolwarm", center=0)
    plt.title("PCA Loadings of Sensory Traits")
    plt.ylabel("Sensory Traits")
    plt.xlabel("Principal Components")
    plt.tight_layout()
    plt.show()

    # PCA Biplot 
    plt.figure(figsize=(8, 7))

    # Scatter varieties
    plt.scatter(pc_df.loc[pc_df["is_panel"], "PC1"],
                pc_df.loc[pc_df["is_panel"], "PC2"],
                c="blue", label="Panel", alpha=0.7)
    plt.scatter(pc_df.loc[~pc_df["is_panel"], "PC1"],
                pc_df.loc[~pc_df["is_panel"], "PC2"],
                c="orange", marker="*", s=120, label="Non-panel")

    # Add variety names
    for i, row in pc_df.iterrows():
        if row["is_panel"]:
            plt.text(row["PC1"], row["PC3"], row["Variety"], fontsize=7)

    # Arrows for sensory loadings
    for feature in sensory_cols:
        x = loadings_df.loc[feature, "PC1"] * scaling_factor
        y = loadings_df.loc[feature, "PC2"] * scaling_factor
        plt.arrow(0, 0, x, y, color="red", alpha=0.5, head_width=0.1, length_includes_head=True)
        plt.text(x * 1.1, y * 1.1, feature, color="red", fontsize=8)

    plt.axhline(0, color="gray", linestyle="--")
    plt.axvline(0, color="gray", linestyle="--")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[2]*100:.1f}%)")
    plt.title("PCA Biplot: Varieties vs Sensory Traits")
    plt.legend()
    plt.tight_layout()
    plt.grid(False)
    plt.show()

    # Save results
    if save_projection_path:
        os.makedirs(os.path.dirname(save_projection_path), exist_ok=True)
        pc_df.to_csv(save_projection_path, index=False)
        print(f" Projection data saved to: {save_projection_path}")

    if save_loadings_path:
        os.makedirs(os.path.dirname(save_loadings_path), exist_ok=True)
        loadings_df.to_csv(save_loadings_path)
        print(f" Loadings saved to: {save_loadings_path}")

    return pc_df, pca, loadings_df



from sklearn.neighbors import NearestNeighbors

def perform_knn_sensory_projection_rotatable_3D(
    aroma_matrix,
    sensory_summary,
    k=3,
    n_components=3,
    scaling_factor=4,
    loading_threshold=0.2,
    rotation_angle_12=0,   # rotation for PC1–PC2
    rotation_angle_13=0,   # rotation for PC1–PC3
    rotation_angle_23=0,   # rotation for PC2–PC3
    save_projection_path=None,
    save_loadings_path=None
):
    """
    Merge aroma + sensory data, impute missing sensory traits using KNN,
    perform PCA on sensory traits, visualize projections for PC1–PC2, PC1–PC3, PC2–PC3,
    and show loadings for all sensory traits.
    """

    # --- Step 1: Standardize column names ---
    aroma_matrix = aroma_matrix.copy()
    sensory_summary = sensory_summary.copy()
    aroma_matrix.columns = aroma_matrix.columns.str.strip().str.upper()
    sensory_summary.columns = sensory_summary.columns.str.strip().str.upper()

    # --- Step 2: Check for Variety column ---
    if "VARIETY" not in aroma_matrix.columns:
        raise KeyError("'VARIETY' column not found in aroma_matrix!")
    if "VARIETY" not in sensory_summary.columns:
        raise KeyError("'VARIETY' column not found in sensory_summary!")

    # --- Step 3: Merge datasets ---
    merged = aroma_matrix.merge(sensory_summary, on="VARIETY", how="left")
    merged["IS_PANEL"] = merged["VARIETY"].isin(sensory_summary["VARIETY"])

    panel_df = merged[merged["IS_PANEL"]].reset_index(drop=True)
    nonpanel_df = merged[~merged["IS_PANEL"]].reset_index(drop=True)

    sensory_cols = [c for c in sensory_summary.columns if c != "VARIETY"]
    aroma_cols = [c for c in aroma_matrix.columns if c != "VARIETY"]

    Y_panel = panel_df[sensory_cols].fillna(0)
    X_panel = panel_df[aroma_cols].fillna(0)
    X_nonpanel = nonpanel_df[aroma_cols].fillna(0)

    # --- Step 4: KNN imputation ---
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(X_panel)
    distances, indices = nn.kneighbors(X_nonpanel)
    Y_nonpanel_pred = np.array([Y_panel.iloc[idx].mean(axis=0) for idx in indices])

    # --- Step 5: PCA ---
    scaler = StandardScaler().fit(Y_panel.values)
    X_panel_scaled = scaler.transform(Y_panel.values)
    X_nonpanel_scaled = scaler.transform(Y_nonpanel_pred)

    pca = PCA(n_components=n_components, random_state=0)
    pcs_panel = pca.fit_transform(X_panel_scaled)
    pcs_nonpanel = pca.transform(X_nonpanel_scaled)

    panel_pcs = pd.DataFrame(pcs_panel, columns=[f"PC{i+1}" for i in range(n_components)])
    panel_pcs["Variety"] = panel_df["VARIETY"].values
    panel_pcs["is_panel"] = True

    nonpanel_pcs = pd.DataFrame(pcs_nonpanel, columns=[f"PC{i+1}" for i in range(n_components)])
    nonpanel_pcs["Variety"] = nonpanel_df["VARIETY"].values
    nonpanel_pcs["is_panel"] = False

    pc_df = pd.concat([panel_pcs, nonpanel_pcs], ignore_index=True)

    loadings_df = pd.DataFrame(
        pca.components_.T,
        columns=[f"PC{i+1}" for i in range(n_components)],
        index=sensory_cols
    )

    print("\nPCA explained variance ratio:", np.round(pca.explained_variance_ratio_, 3))

    # --- Step 6: Plot loadings heatmap for all 3 PCs ---
    plt.figure(figsize=(12, 6))
    sns.heatmap(loadings_df, annot=True, cmap="coolwarm", center=0)
    plt.title("PCA Loadings of Sensory Traits (PC1–PC3)")
    plt.tight_layout()
    plt.show()

    # --- Helper for rotation and plotting ---
    def rotate_and_plot(pc_df, loadings_df, pc_x, pc_y, angle, title):
        theta = np.deg2rad(angle)
        rot = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta),  np.cos(theta)]])
        scores = pc_df[[pc_x, pc_y]].values @ rot
        loads = loadings_df[[pc_x, pc_y]].values @ rot
        pc_df[f"{pc_x}_rot{pc_y}"], pc_df[f"{pc_y}_rot{pc_x}"] = scores[:, 0], scores[:, 1]
        loadings_df[f"{pc_x}_rot{pc_y}"], loadings_df[f"{pc_y}_rot{pc_x}"] = loads[:, 0], loads[:, 1]

        plt.figure(figsize=(8, 7))
        plt.scatter(pc_df.loc[pc_df["is_panel"], f"{pc_x}_rot{pc_y}"],
                    pc_df.loc[pc_df["is_panel"], f"{pc_y}_rot{pc_x}"],
                    c="blue", label="Panel", alpha=0.7)
        plt.scatter(pc_df.loc[~pc_df["is_panel"], f"{pc_x}_rot{pc_y}"],
                    pc_df.loc[~pc_df["is_panel"], f"{pc_y}_rot{pc_x}"],
                    c="orange", marker="*", s=120, label="Non-panel")

        # Text labels for panel samples
        for i, row in pc_df.iterrows():
            if row["is_panel"]:
                plt.text(row[f"{pc_x}_rot{pc_y}"], row[f"{pc_y}_rot{pc_x}"], row["Variety"], fontsize=7)

        strong = loadings_df[
            (loadings_df[pc_x].abs() > loading_threshold) |
            (loadings_df[pc_y].abs() > loading_threshold)
        ]

        for feature in strong.index:
            x = loadings_df.loc[feature, f"{pc_x}_rot{pc_y}"] * scaling_factor
            y = loadings_df.loc[feature, f"{pc_y}_rot{pc_x}"] * scaling_factor
            plt.arrow(0, 0, x, y, color="red", alpha=0.6, head_width=0.08, length_includes_head=True)
            plt.text(x * 1.1, y * 1.1, feature, color="red", fontsize=8)

        plt.axhline(0, color="gray", linestyle="--")
        plt.axvline(0, color="gray", linestyle="--")
        plt.xlabel(f"{pc_x} (rotated {angle}°)")
        plt.ylabel(f"{pc_y} (rotated {angle}°)")
        plt.title(title)
        plt.legend()
        plt.grid(False)
        plt.tight_layout()
        plt.show()

    # -- Generate all pairwise plots ---
    rotate_and_plot(pc_df, loadings_df, "PC1", "PC2", rotation_angle_12, "PC1–PC2 Sensory Space")
    rotate_and_plot(pc_df, loadings_df, "PC1", "PC3", rotation_angle_13, "PC1–PC3 Sensory Space")
    rotate_and_plot(pc_df, loadings_df, "PC2", "PC3", rotation_angle_23, "PC2–PC3 Sensory Space")

    # --- Save outputs ---
    if save_projection_path:
        pc_df.to_csv(save_projection_path, index=False)
        print(f" Saved projection to: {save_projection_path}")
    if save_loadings_path:
        loadings_df.to_csv(save_loadings_path)
        print(f" Saved loadings to: {save_loadings_path}")

    return pc_df, pca, loadings_df




# --- Step 10: Sensory profile interpretation table ---
def interpret_variety_sensory(pc_df, loadings_df, top_n=3):
    # Determine which sensory traits contribute most to each PC
    loadings_abs = loadings_df.abs()
    top_traits_pc1 = loadings_abs["PC1"].nlargest(top_n).index.tolist()
    top_traits_pc2 = loadings_abs["PC2"].nlargest(top_n).index.tolist()
    top_traits_pc3 = loadings_abs["PC3"].nlargest(top_n).index.tolist()

    summary_rows = []
    for _, row in pc_df.iterrows():
        name = row["Variety"]
        if row["PC1"] > 0:
            pc1_traits = top_traits_pc1
        else:
            pc1_traits = [f"low {t}" for t in top_traits_pc1]

        if row["PC2"] > 0:
            pc2_traits = top_traits_pc2
        else:
            pc2_traits = [f"low {t}" for t in top_traits_pc2]

        if row["PC3"] > 0:
            pc3_traits = top_traits_pc3
        else:
            pc3_traits = [f"low {t}" for t in top_traits_pc3]

        summary_rows.append({
            "Variety": name,
            "Dominant traits (PC1)": ", ".join(pc1_traits),
            "Dominant traits (PC2)": ", ".join(pc2_traits),
            "Dominant traits (PC3)": ", ".join(pc3_traits)
        })

    return pd.DataFrame(summary_rows)

from sklearn.mixture import GaussianMixture
from matplotlib.patches import Ellipse
def analyze_sensory_clusters(
    pc_df,
    sensory_summary,
    max_clusters=10,
    top_n_traits=3,
    scale_ellipse=2.5,
    save_path=None,
    random_state=42
):
    """
    Perform Gaussian Mixture clustering on PCA scores and interpret dominant sensory traits.

    Parameters
    ----------
    pc_df : pd.DataFrame
        DataFrame containing PCA scores (must include 'PC1', 'PC2', 'PC3', and 'Variety').
    sensory_summary : pd.DataFrame
        DataFrame with sensory data (must include 'VARIETY' column matching pc_df).
    max_clusters : int, default=10
        Maximum number of clusters to evaluate.
    top_n_traits : int, default=3
        Number of top sensory traits to display per cluster.
    scale_ellipse : float, default=2.5
        Scaling factor for cluster ellipse size.
    save_path : str, optional
        Path to save updated pc_df (with clusters). If None, file is not saved.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    pc_df : pd.DataFrame
        Input DataFrame with an added 'SensoryCluster' column.
    cluster_means : pd.DataFrame
        Mean sensory profile for each cluster.
    dominant_traits : dict
        Dictionary of top traits per cluster.
    """

    # --- Extract PCA scores for clustering ---
    X = pc_df[["PC1", "PC2", "PC3"]].values

    # --- Determine optimal number of clusters using BIC/AIC ---
    bic_scores, aic_scores = [], []
    n_values = range(1, max_clusters + 1)

    for n in n_values:
        gmm = GaussianMixture(n_components=n, covariance_type="full", random_state=random_state)
        gmm.fit(X)
        bic_scores.append(gmm.bic(X))
        aic_scores.append(gmm.aic(X))

    # --- Plot model selection ---
    plt.figure(figsize=(8, 5))
    plt.plot(n_values, bic_scores, marker="o", label="BIC")
    plt.plot(n_values, aic_scores, marker="o", label="AIC")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Score (Lower = Better)")
    plt.title("Model Selection for GMM (BIC & AIC)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- Best cluster count (BIC minimum) ---
    best_n = n_values[np.argmin(bic_scores)]
    print(f" Optimal number of sensory clusters (by BIC): {best_n}")

    # --- Fit final GMM model ---
    gmm = GaussianMixture(n_components=best_n, covariance_type="full", random_state=random_state)
    pc_df["SensoryCluster"] = gmm.fit_predict(X)

    # --- Merge sensory data ---
    merged = sensory_summary.merge(
        pc_df[["Variety", "SensoryCluster"]],
        left_on="VARIETY",
        right_on="Variety",
        how="inner"
    )

    # --- Compute cluster means ---
    numeric_cols = merged.select_dtypes(include=[np.number]).columns
    cluster_means = merged.groupby("SensoryCluster")[numeric_cols].mean()

    # --- Identify dominant traits per cluster ---
    dominant_traits = {
        c: cluster_means.loc[c].sort_values(ascending=False).head(top_n_traits).index.tolist()
        for c in cluster_means.index
    }

    print("\n Dominant Sensory Traits per Cluster:")
    for c, traits in dominant_traits.items():
        print(f"  Cluster {c+1}: {', '.join(traits)}")

    # --- Ellipse helper ---
    def draw_ellipse(position, covariance, ax, color, scale=scale_ellipse):
        if covariance.shape == (2, 2):
            eigvals, eigvecs = np.linalg.eigh(covariance)
            order = eigvals.argsort()[::-1]
            eigvals, eigvecs = eigvals[order], eigvecs[:, order]
            angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
            width, height = scale * np.sqrt(eigvals)
            ellipse = Ellipse(
                xy=position,
                width=width,
                height=height,
                angle=angle,
                edgecolor=color,
                facecolor="none",
                lw=2.5
            )
            ax.add_patch(ellipse)

    # --- Plot PCA cluster map ---
    plt.figure(figsize=(10, 8))
    colors = sns.color_palette("Set2", best_n)

    for i, color in enumerate(colors):
        cluster_points = pc_df[pc_df["SensoryCluster"] == i]
        traits_label = ", ".join(dominant_traits[i])
        legend_label = f"Cluster {i+1} — {traits_label}"

        plt.scatter(
            cluster_points["PC1"], cluster_points["PC2"],
            s=70, c=[color], label=legend_label,
            alpha=0.8, edgecolor="k"
        )

        if len(cluster_points) > 2:
            cov = np.cov(cluster_points[["PC1", "PC2"]].values.T)
            draw_ellipse(cluster_points[["PC1", "PC2"]].mean().values, cov, plt.gca(), color=color)

        # Variety labels
        for _, row in cluster_points.iterrows():
            plt.text(row["PC1"] + 0.1, row["PC2"] + 0.1, row["Variety"],
                     fontsize=8, color="black", alpha=0.8)

    plt.xlabel("PC1 (Sensory Component 1)")
    plt.ylabel("PC2 (Sensory Component 2)")
    plt.title("Sensory Clusters of Potato Varieties (Based on PCA)", fontsize=13, weight="bold")
    plt.legend(title="Clusters & Dominant Flavors", loc="best", fontsize=9)
    plt.grid(False)
    plt.tight_layout()
    plt.show()

    # --- Save results ---
    if save_path:
        pc_df.to_csv(save_path, index=False)
        print(f" Clustered PCA data saved to: {save_path}")

    return pc_df, cluster_means, dominant_traits
