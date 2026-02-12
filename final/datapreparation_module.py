
"""
Author: Fatemeh Monfared
Module: datapreparation_module.py (Data Preparation Utilities)
Thesis: Data-Driven Analysis of Potato Aroma and Flavor Using TD–GC–MS and Machine Learning
Affiliation: Hanze University of Applied Sciences
Year: 2026

Purpose
-------
This module provides reusable helper functions for preparing TD–GC–MS peak tables
(MsMetrix outputs) into analysis-ready datasets. It is designed to be imported by
the biorep1/biorep2 preparation notebooks and to standardize preprocessing steps
for company and thesis workflows.

What this module supports 
--------------------------------------
1) Input handling & cleaning
   - Read QC/sample files (Excel or delimited text)
   - Clean/standardize columns, compute retention-time summary and tR_best
   - Filter/reshape wide tables to tidy/long formats when needed

2) QC validation (alkanes / QC samples)
   - Match alkane standards using expected tR and m/z tolerances
   - Evaluate QC linearity (e.g., R²) across QC levels/days
   - Optional interactive Dash dashboard for QC linear trends

3) Drift correction & normalisation
   - Estimate injection-order drift using LOESS (LOWESS) regression
     (x = injection order across the run; y = log-intensity)
   - Apply drift correction to QC + sample data
   - QC-based normalisation (e.g., using QC1 medians per peak)

4) Export & downstream readiness
   - Extract/attach Variety metadata from sample naming
   - Export cleaned/corrected/normalised outputs for PCA/ML/GWAS pipelines
"""




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os
import statsmodels.api as sm
from sklearn.decomposition import PCA
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import seaborn as sns
import plotly.graph_objs as go
from dash import Dash, dcc, html, Input, Output
import yaml
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import plotly.express as px
from statsmodels.nonparametric.smoothers_lowess import lowess
from plotly.subplots import make_subplots


def read_qc_file(file_path, sheet_index=0, header_row=12, preview_rows=5):
    """
    Reads a QC data file, handling both Excel (.xlsx, .xls) and text (.csv, .txt) files.
    Automatically detects the correct format and prints a preview.
    """
    try:
        # Try reading as Excel first
        qc_data = pd.read_excel(file_path, sheet_name=sheet_index, header=header_row, engine="openpyxl")
        print(" File successfully read as Excel (.xlsx)")
    except Exception as e:
        print(f" Excel read failed ({e}). Trying as CSV/TSV instead...")
        try:
            qc_data = pd.read_csv(file_path, sep=None, header=header_row, engine="python")  # auto-detect delimiter
            print(" File successfully read as delimited text (CSV/TSV)")
        except Exception as e2:
            print(f" Could not read file as CSV either: {e2}")
            raise e2
    print("\n Preview of QC data:")
    print(qc_data.head(preview_rows))
    return qc_data






def clean_qc_data(qc_data: pd.DataFrame, output_path: str) -> pd.DataFrame:
    """
    Cleans and processes QC data derived from MsMetrix output.
    
    Steps performed:
    1. Cleans column names (removes extra spaces).
    2. Selects metadata columns and T1 sample columns.
    3. Calculates differences between retention times (tR1, tR2, tR).
    4. Defines 'tR_best' as the average of tR1 and tR2.
    5. Removes unnecessary retention time columns.
    6. Reorders columns so 'tR_best' appears after 'Peak'.
    7. Saves the cleaned DataFrame as a CSV for further analysis.
    
    Parameters
    ----------
    qc_data : pd.DataFrame
        The raw QC data loaded from MsMetrix output.
    output_path : str
        Path to save the cleaned CSV file.

    Returns
    -------
    pd.DataFrame
        Cleaned QC data with 'tR_best' column.
    """
    #Clean column names
    qc_data.columns = qc_data.columns.str.strip()
    # Select metadata and T1 sample columns
    metadata_cols = ['Peak', 'tR1', 'tR2', 'tR', 'm/z']
    t1_cols = [c for c in qc_data.columns if c.strip().endswith("T1")]
    qc_data_T1 = qc_data[metadata_cols + t1_cols]

    print(f"Selected T1 columns ({len(t1_cols)}): {t1_cols}")
    print(f"Data shape before cleaning: {qc_data_T1.shape}")

    #  Calculate retention time differences
    diff_t_vs_1 = (qc_data_T1['tR'] - qc_data_T1['tR1']).abs()
    diff_t_vs_2 = (qc_data_T1['tR'] - qc_data_T1['tR2']).abs()
    diff_1_vs_2 = (qc_data_T1['tR1'] - qc_data_T1['tR2']).abs()
    avg_t = (qc_data_T1['tR1'] + qc_data_T1['tR2']) / 2
    diff_avg_vs_t = (avg_t - qc_data_T1['tR']).abs()

    # summary statistics
    print("\nRetention time differences summary:")
    print("abs(tR - tR1):\n", diff_t_vs_1.describe(), "\n")
    print("abs(tR - tR2):\n", diff_t_vs_2.describe(), "\n")
    print("abs(tR1 - tR2):\n", diff_1_vs_2.describe(), "\n")
    print("abs(avg(tR1,tR2) - tR):\n", diff_avg_vs_t.describe(), "\n")
    # Add averaged retention time
    qc_data_T1['tR_best'] = avg_t
    # Remove redundant columns
    to_keep = [col for col in qc_data_T1.columns if col not in ['tR1', 'tR2', 'tR']]
    cleaned_qc = qc_data_T1[to_keep]
    # Reorder columns to put 'tR_best' after 'Peak'
    cols = cleaned_qc.columns.tolist()
    cols.insert(cols.index('Peak') + 1, cols.pop(cols.index('tR_best')))
    cleaned_qc = cleaned_qc[cols]
    # Save the cleaned QC data
    cleaned_qc.to_csv(output_path, index=False)
    print(f"\n Cleaned QC data saved to: {output_path}")
    print(f"Final shape: {cleaned_qc.shape}")

    return cleaned_qc





def merge_qc_with_master(master_path, qc_path, output_path):

    """
    Merges the cleaned QC data with the Master GC–MS table for compound validation.
    Adds progress output and peak consistency reporting.
    """
    print("\n Merging Master and Cleaned QC files...")
    print(f" Master path: {master_path}")
    print(f" QC path: {qc_path}\n")

    # Load data
    master_df = pd.read_csv(master_path)
    qc_df = pd.read_csv(qc_path)

    print(f" Loaded master data: {master_df.shape}")
    print(f" Loaded QC data: {qc_df.shape}")

    # Standardize column names for safety
    master_df.columns = master_df.columns.str.strip()
    qc_df.columns = qc_df.columns.str.strip()

    # Ensure Peak column exists
    if "Peak" not in master_df.columns or "Peak" not in qc_df.columns:
        raise KeyError("Both master and QC files must contain a 'Peak' column.")

    # Peak consistency check
    master_peaks = set(master_df["Peak"])
    qc_peaks = set(qc_df["Peak"])
    common_peaks = master_peaks.intersection(qc_peaks)
    missing_in_qc = master_peaks - qc_peaks
    missing_in_master = qc_peaks - master_peaks

    print(f"\n Peak matching summary:")
    print(f"  • Peaks in Master: {len(master_peaks)}")
    print(f"  • Peaks in QC: {len(qc_peaks)}")
    print(f"  • Common Peaks: {len(common_peaks)}")
    print(f"  • Missing in QC: {len(missing_in_qc)}")
    print(f"  • Missing in Master: {len(missing_in_master)}")

    # Merge based on shared 'Peak'
    merged_df = pd.merge(master_df, qc_df, left_on="Filename", right_on="ID Names", how="inner")
    print(f"\n Merged DataFrame shape: {merged_df.shape}")

    # Save merged file
    merged_df.to_csv(output_path, index=False)
    print(f" Saved merged QC–Master file to:\n{output_path}\n")

    # Save unmatched peaks
    if missing_in_qc:
        pd.DataFrame(sorted(missing_in_qc), columns=["Missing_in_QC"]).to_csv(
            output_path.replace(".csv", "_missing_in_QC.csv"), index=False
        )
        print(f" Missing peaks in QC saved to: {output_path.replace('.csv', '_missing_in_QC.csv')}")
    if missing_in_master:
        pd.DataFrame(sorted(missing_in_master), columns=["Missing_in_Master"]).to_csv(
            output_path.replace(".csv", "_missing_in_Master.csv"), index=False
        )
        print(f" Missing peaks in Master saved to: {output_path.replace('.csv', '_missing_in_Master.csv')}")
    return merged_df





def filter_potato_data(input_path, output_path, final_normalized_ids):
    """
    Reads a potato GC-MS Excel file, finds the header row dynamically, selects metadata
    and specific sample IDs, and saves a filtered dataset.
    """
    print(f"\n Reading input file: {input_path}")
    df_raw = pd.read_excel(input_path, header=None)

    #  Find header row automatically
    header_row = df_raw[df_raw.iloc[:, 0].astype(str).str.strip() == "Peak"].index[0]
    potato_data = pd.read_excel(input_path, header=header_row)

    #  Metadata columns
    metadata_cols = ['Peak', 'tR1', 'tR2', 'tR', 'm/z']

    #  Get sample columns (exclude metadata)
    sample_cols = [c for c in potato_data.columns if c not in metadata_cols and c != "ID Names"]

    #  Normalize column names
    def normalize_id(col):
        parts = col.split()
        if len(parts) < 2:
            return col
        sid = parts[1]
        sid = re.sub(r'_T[0-9]+_', '_', sid)
        cleaned = []
        for p in sid.split("_"):
            if re.fullmatch(r'T[0-9]+', p) and p not in ['T1', 'T2']:
                continue
            cleaned.append(p)
        return "_".join(cleaned)

    norm_to_col = {normalize_id(c): c for c in sample_cols}
    # Filter for desired IDs
    selected_columns = [norm_to_col[nid] for nid in final_normalized_ids if nid in norm_to_col]
    #  Combine metadata + selected columns
    potato_data_filtered = potato_data[metadata_cols + selected_columns]
    # Save filtered dataset
    potato_data_filtered.to_excel(output_path, index=False)
    print(f" Dataset saved with {len(selected_columns)} samples + metadata columns at:\n{output_path}")
    return potato_data_filtered


def clean_potato_data(potato_data_filtered, output_path):
    """
    Cleans filtered potato GC-MS data by analyzing retention time consistency,
    computing tR_best, and selecting essential columns for further analysis.
    """
    print("\n Cleaning potato dataset and computing tR_best...")

    #  Compute differences between retention times
    diff_t_vs_1 = (potato_data_filtered['tR'] - potato_data_filtered['tR1']).abs()
    diff_t_vs_2 = (potato_data_filtered['tR'] - potato_data_filtered['tR2']).abs()
    diff_1_vs_2 = (potato_data_filtered['tR1'] - potato_data_filtered['tR2']).abs()
    avg_t = (potato_data_filtered['tR1'] + potato_data_filtered['tR2']) / 2
    diff_avg_vs_t = (avg_t - potato_data_filtered['tR']).abs()

    #  summary statistics
    print("abs(tR - tR1):\n", diff_t_vs_1.describe(), "\n")
    print("abs(tR - tR2):\n", diff_t_vs_2.describe(), "\n")
    print("abs(tR1 - tR2):\n", diff_1_vs_2.describe(), "\n")
    print("abs(avg(tR1,tR2) - tR):\n", diff_avg_vs_t.describe(), "\n")

    #  Rename tR to tR_best
    selected_data = potato_data_filtered.rename(columns={'tR': 'tR_best'})
    #  Keep only relevant columns
    columns_to_keep = ['Peak', 'tR_best', 'm/z'] + [c for c in selected_data.columns if c.startswith('s')]
    selected_data = selected_data[columns_to_keep].copy()
    #  Save cleaned dataset
    selected_data.to_excel(output_path, index=False)
    print(f" Cleaned dataset saved at:\n{output_path}")
    print("Selected data shape:", selected_data.shape)
    return selected_data




def match_alkanes(cleaned_data, rt_tolerance=0.05, mz_tolerance=0.2):
    """
    Matches alkane standards (C8–C20) in cleaned GC-MS data based on expected
    retention times (tR_best) and m/z values within defined tolerances.

    Parameters
    ----------
    cleaned_data : pd.DataFrame
        Cleaned GC-MS dataset containing at least 'Peak', 'tR_best', and 'm/z' columns.
    rt_tolerance : float
        Retention time matching tolerance window (default: 0.05).
    mz_tolerance : float
        m/z value matching tolerance window (default: 0.2).

    Returns
    -------
    pd.DataFrame
        DataFrame listing all detected alkane matches with expected and observed
        retention times and m/z values.
    """

    # Define alkane reference standards with tR and m/z
    alkanes = pd.DataFrame([
        {'Compound': 'Octane',      'tR': 5.34,   'm/z': 43},
        {'Compound': 'Decane',      'tR': 10.154, 'm/z': 57},
        {'Compound': 'Dodecane',    'tR': 13.626, 'm/z': 57},
        {'Compound': 'Tetradecane', 'tR': 16.388, 'm/z': 57},
        {'Compound': 'Pentadecane', 'tR': 17.62,  'm/z': 57},
        {'Compound': 'Octadecane',  'tR': 20.906, 'm/z': 57},
        {'Compound': 'Eicosane',    'tR': 22.825, 'm/z': 57},
    ])

    matches = []
    #  Iterate through each reference alkane and find matches
    for _, alk in alkanes.iterrows():
        subset = cleaned_data[
            (cleaned_data["m/z"].between(alk["m/z"] - mz_tolerance, alk["m/z"] + mz_tolerance)) &
            (cleaned_data["tR_best"].between(alk["tR"] - rt_tolerance, alk["tR"] + rt_tolerance))
        ]

        if not subset.empty:
            for _, row in subset.iterrows():
                matches.append({
                    "Compound": alk["Compound"],
                    "Expected_tR": alk["tR"],
                    "Expected_m/z": alk["m/z"],
                    "Peak": row["Peak"],
                    "Observed_tR": row["tR_best"],
                    "Observed_m/z": row["m/z"]
                })

    alkane_matches = pd.DataFrame(matches)

    #  summary
    if alkane_matches.empty:
        print(" No alkane matches found within tolerance limits.")
    else:
        print(f" Found {len(alkane_matches)} alkane matches within "
              f"±{rt_tolerance} tR and ±{mz_tolerance} m/z tolerance.")

    return alkane_matches


def plot_alkane_alignment(alkane_matches):
    """
    Plots expected vs observed retention times for matched alkanes.
    Shows alignment, deviation, and ideal 1:1 line.
    """
    if alkane_matches.empty:
        print(" No alkane matches to plot.")
        return

    plt.figure(figsize=(6, 5))
    plt.scatter(
        alkane_matches["Expected_tR"],
        alkane_matches["Observed_tR"],
        color="teal",
        edgecolors="black",
        s=80,
        label="Alkane match"
    )

    # expected = observed
    plt.plot(
        [alkane_matches["Expected_tR"].min(), alkane_matches["Expected_tR"].max()],
        [alkane_matches["Expected_tR"].min(), alkane_matches["Expected_tR"].max()],
        'r--',
        label="Ideal match (y = x)"
    )

    # Annotate points with compound names and deviation
    for _, row in alkane_matches.iterrows():
        delta = row["Observed_tR"] - row["Expected_tR"]
        plt.text(
            row["Expected_tR"] + 0.1,
            row["Observed_tR"] + 0.05,
            f"{row['Compound']} (Δ={delta:.2f})",
            fontsize=8
        )

    plt.xlabel("Expected RT (min)")
    plt.ylabel("Observed RT (min)")
    plt.title("Alkane Retention Time Alignment")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()




def merge_qc_with_master(file_master, file_qc):
    """
    Merges cleaned QC GC-MS data with the experiment master table metadata.

    Steps:
    1. Reads the master table (CSV) and cleaned QC file (CSV).
    2. Converts the wide QC table into long (tidy) format using melt().
    3. Extracts sample filenames from column headers.
    4. Merges with metadata (Sample, Day, Injection order).
    5. Filters only QC samples (Sample names starting with 'QC').

    Parameters
    ----------
    file_master : str
        Path to the master metadata table CSV file.
    file_qc : str
        Path to the cleaned QC dataset CSV file.

    Returns
    -------
    pd.DataFrame
        A merged DataFrame containing QC samples with associated metadata.
    """

    # Load both input files
    master_table = pd.read_csv(file_master)
    cleaned_qc = pd.read_csv(file_qc)

    # Reshape from wide to long format
    df_long_qc = cleaned_qc.melt(
        id_vars=["Peak", "tR_best", "m/z"],
        var_name="SampleCol",
        value_name="Intensity"
    )

    # Extract actual sample filenames (second token in column name)
    df_long_qc["Filename"] = df_long_qc["SampleCol"].str.split().str[1]

    # Merge with metadata from master table
    df_merged_qc = df_long_qc.merge(
        master_table[["Filename", "Sample", "Day", "Injection_order"]],
        on="Filename",
        how="left"
    )
    # Keep only QC samples
    df_qc = df_merged_qc[df_merged_qc["Sample"].str.startswith("QC")].copy()

    print(f" Merged QC dataset shape: {df_qc.shape}")
    print(f"Included metadata columns: {', '.join(['Sample', 'Day', 'Injection_order'])}")
    return df_qc




def evaluate_qc_linearity(alkane_matches, df_qc):
    """
    Evaluates linearity (R²) of QC sample intensities across days for each alkane peak.

    For each compound–peak pair:
      - Extracts QC samples with the same peak.
      - Groups by day.
      - Fits a simple linear regression: Intensity ~ QC replicate number (QC1–QC9).
      - Computes R² as a measure of linearity/repeatability.

    Parameters
    ----------
    alkane_matches : pd.DataFrame
        DataFrame with at least ['Compound', 'Peak'] columns from alkane matching.
    df_qc : pd.DataFrame
        Long-format QC data merged with metadata (from `merge_qc_with_master`).

    Returns
    -------
    pd.DataFrame
        Summary table with columns:
        ['Compound', 'Peak', 'Day', 'tR_best', 'm/z', 'R2']
    """
    results = []
    for (compound, peak), sub_matches in alkane_matches.groupby(["Compound", "Peak"]):
        # Filter QC data for this peak
        for day, sub in df_qc[df_qc["Peak"] == peak].groupby("Day"):
            sub = sub.copy()

            # Extract QC replicate number from 'Sample' (e.g., QC1 → 1)
            sub["QC_num"] = sub["Sample"].str.extract(r"QC(\d+)").astype(int)

            # Only fit if we have enough replicates
            if len(sub) >= 3:
                X = sub["QC_num"].values.reshape(-1, 1)
                y = sub["Intensity"].values

                model = LinearRegression().fit(X, y)
                r2 = model.score(X, y)

                results.append({
                    "Compound": compound,
                    "Peak": peak,
                    "Day": day,
                    "tR_best": sub["tR_best"].iloc[0],
                    "m/z": sub["m/z"].iloc[0],
                    "R2": r2
                })
    results_df = pd.DataFrame(results)
    print(f" Evaluated QC linearity for {len(results_df)} (Compound, Day) combinations.")
    return results_df





def launch_qc_linearity_dashboard(df_qc, results_df, port=8040):
    """
    Launch an interactive Dash dashboard to visualize QC linearity per day.

    Parameters
    ----------
    df_qc : pd.DataFrame
        Long-format QC data. Must contain at least: ['Day', 'Peak', 'Sample', 'Intensity'].
    results_df : pd.DataFrame
        Linearity results per compound/peak/day. Must contain at least: ['Day', 'Peak', 'Compound'].
    port : int, default=8040
        Port to run the dashboard.
    """
    qc_concentrations = {"QC1": 312.5, "QC2": 625, "QC3": 1250, "QC4": 2500}

    # Basic sanity checks (fail fast, clearer errors)
    required_qc = {"Day", "Peak", "Sample", "Intensity"}
    required_res = {"Day", "Peak", "Compound"}
    missing_qc = required_qc - set(df_qc.columns)
    missing_res = required_res - set(results_df.columns)
    if missing_qc:
        raise ValueError(f"df_qc missing columns: {missing_qc}")
    if missing_res:
        raise ValueError(f"results_df missing columns: {missing_res}")

    # Clean Day values
    days = sorted(pd.Series(df_qc["Day"].dropna().unique()).tolist())
    if not days:
        raise ValueError("df_qc has no valid 'Day' values.")

    app = Dash(__name__)
    app.title = "QC Linear Trend Dashboard"

    app.layout = html.Div(
        [
            html.H1("QC Linear Trend Dashboard", style={"textAlign": "center"}),

            html.Div(
                [
                    html.Label("Select Day:"),
                    dcc.Dropdown(
                        id="day-dropdown",
                        options=[{"label": str(day), "value": day} for day in days],
                        value=days[0],
                        clearable=False,
                    ),
                ],
                style={"width": "40%", "margin": "auto"},
            ),

            dcc.Graph(id="qc-trend-plots", style={"marginTop": "30px"}),
        ]
    )

    @app.callback(Output("qc-trend-plots", "figure"), Input("day-dropdown", "value"))
    def update_plots(selected_day):
        day_results = results_df[results_df["Day"] == selected_day].copy()

        if day_results.empty:
            fig = go.Figure()
            fig.update_layout(title=f"No results for Day {selected_day}")
            return fig

        # Optional: limit number of panels for readability/performance
        day_results = day_results.head(9)

        traces = []
        subplot_titles = []

        for _, row_data in day_results.iterrows():
            peak = row_data["Peak"]

            sub = df_qc[(df_qc["Peak"] == peak) & (df_qc["Day"] == selected_day)].copy()
            if sub.empty:
                continue

            sub["QC_label"] = sub["Sample"].astype(str).str.extract(r"(QC\d+)")
            sub["Concentration"] = sub["QC_label"].map(qc_concentrations)

            sub = sub.dropna(subset=["Concentration", "Intensity"])
            if sub.empty:
                continue

            X = sub["Concentration"].to_numpy(dtype=float).reshape(-1, 1)
            y = sub["Intensity"].to_numpy(dtype=float)

            if len(sub) > 1 and np.nanstd(y) > 0:
                model = LinearRegression().fit(X, y)
                r2 = model.score(X, y)

                # sort by x so the fitted line is not zig-zag
                order = np.argsort(sub["Concentration"].to_numpy(dtype=float))
                x_sorted = sub["Concentration"].to_numpy(dtype=float)[order]
                y_sorted = y[order]
                y_pred_sorted = model.predict(x_sorted.reshape(-1, 1))
            else:
                r2 = np.nan
                order = np.argsort(sub["Concentration"].to_numpy(dtype=float))
                x_sorted = sub["Concentration"].to_numpy(dtype=float)[order]
                y_sorted = y[order]
                y_pred_sorted = y_sorted

            subplot_titles.append(
                f"{row_data['Compound']} (Peak {int(peak)}, R²={r2:.2f})"
                if np.isfinite(r2)
                else f"{row_data['Compound']} (Peak {int(peak)}, R²=NA)"
            )

            traces.append((x_sorted, y_sorted, y_pred_sorted, row_data))

        if not traces:
            fig = go.Figure()
            fig.update_layout(title=f"No plottable QC data for Day {selected_day}")
            return fig

        # grid size (up to 9 panels)
        n_panels = len(traces)
        n_cols = 3
        n_rows = int(np.ceil(n_panels / n_cols))

        fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=subplot_titles)

        row_idx, col_idx = 1, 1
        for x_vals, y_vals, y_pred, row_data in traces:
            fig.add_trace(
                go.Scatter(x=x_vals, y=y_vals, mode="markers"),
                row=row_idx, col=col_idx,
            )
            fig.add_trace(
                go.Scatter(x=x_vals, y=y_pred, mode="lines"),
                row=row_idx, col=col_idx,
            )

            fig.update_xaxes(title_text="Concentration (pg/µL)", row=row_idx, col=col_idx)
            fig.update_yaxes(title_text="Intensity", row=row_idx, col=col_idx)

            col_idx += 1
            if col_idx > n_cols:
                col_idx = 1
                row_idx += 1

        fig.update_layout(
            height=320 * n_rows + 200,
            width=1100,
            title_text=f"QC Linear Trends – Day {selected_day}",
            showlegend=False,
        )
        return fig

    print(f"Launching QC Linear Trend Dashboard at http://127.0.0.1:{port}")
    app.run(debug=True, port=port)


def filter_stable_peaks_by_r2(results_df, r2_threshold=0.9):
    """
    Filters peaks (compounds) that show stable QC linearity based on an R² threshold.
    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame containing regression results from evaluate_qc_linearity(),
        with columns ['Compound', 'Peak', 'Day', 'tR_best', 'm/z', 'R2'].
    r2_threshold : float
        Minimum acceptable R² value for a compound/day to be considered stable.
        Default = 0.9.
    Returns
    -------
    pd.DataFrame
        Filtered DataFrame of stable peaks (R² ≥ threshold), sorted by Day and Compound.
    """
    # Apply threshold filter
    stable_peaks = results_df[results_df["R2"] >= r2_threshold].copy()
    # Sort for readability
    stable_peaks = stable_peaks.sort_values(["Day", "Compound"])
    # Display summary
    print(f" Found {stable_peaks['Peak'].nunique()} unique stable peaks "
          f"across {stable_peaks['Day'].nunique()} days (R² ≥ {r2_threshold})")
    print(stable_peaks[["Day", "Compound", "Peak", "tR_best", "m/z", "R2"]])

    return stable_peaks






def plot_loess_trends_for_stable_peaks(df_qc, stable_peaks, max_plots=9, frac=0.3):
    """
    Plots LOESS-smoothed intensity trends across injection order 
    for stable QC peaks (R²-filtered).
    Parameters
    ----------
    df_qc : pd.DataFrame
        Long-format QC dataset containing at least:
        ['Peak', 'Intensity', 'Injection_order', 'tR_best', 'm/z'].
    stable_peaks : pd.DataFrame
        DataFrame of stable peaks returned by filter_stable_peaks_by_r2(),
        must include a 'Peak' column.
    max_plots : int
        Maximum number of peaks to plot (default: 9).
    frac : float
        Fraction of data used for LOESS smoothing (default: 0.3).

    Returns
    -------
    
        Displays a grid of scatter plots with LOESS fits.
    """
    stable_ids = stable_peaks["Peak"].unique()[:max_plots]
    plt.figure(figsize=(14, 10))
    for i, peak in enumerate(stable_ids, 1):
        sub = df_qc[df_qc["Peak"] == peak].dropna(subset=["Intensity", "Injection_order"])
        if sub.empty:
            continue

        x = sub["Injection_order"].values
        y = sub["Intensity"].values

        # LOESS smoothing for visualization
        loess_fit = lowess(y, x, frac=frac, return_sorted=True)
        plt.subplot(3, 3, i)
        plt.scatter(x, y, alpha=0.6, label=f"Peak {peak}", color="steelblue")
        plt.plot(loess_fit[:, 0], loess_fit[:, 1], color="red", linewidth=2, label="LOESS")
        plt.title(f"Peak {peak}\n tR={sub['tR_best'].iloc[0]:.2f}, m/z={sub['m/z'].iloc[0]}")
        plt.xlabel("Injection order")
        plt.ylabel("Intensity")
        plt.legend()
    plt.tight_layout()
    plt.show()
    print(f" Displayed LOESS trends for {len(stable_ids)} stable peaks.")







def plot_alkane_stability_colored(df_qc, stable_peaks, qc_levels=None, r2_threshold=0.2, log_transform=True):
    """
    Visualizes alkane QC stability across injections with distinct colors per QC level.
    Adds human-readable stability status  based on R² threshold.
    Parameters
    ----------
    df_qc : pd.DataFrame
        QC merged dataset containing 'Peak', 'Injection_order', 'Sample', and 'Intensity'.
    stable_peaks : pd.DataFrame
        DataFrame with alkane peaks (includes columns 'Peak', 'tR_best', 'm/z').
    qc_levels : list of str
        QC identifiers to include, default: ["QC1", "QC2", "QC3", "QC4"].
    r2_threshold : float
        R² threshold for determining stability (default 0.2).
    log_transform : bool
        If True, applies log10 transformation to intensity.

    Returns
    -------
    stability_results : pd.DataFrame
        R² values and stability labels for each peak and QC level.
    """
    if qc_levels is None:
        qc_levels = ["QC1", "QC2", "QC3", "QC4"]

    colors = {
        "QC1": "#1f77b4",  # blue
        "QC2": "#ff7f0e",  # orange
        "QC3": "#2ca02c",  # green
        "QC4": "#d62728"   # red
    }

    alkane_ids = stable_peaks["Peak"].unique()
    alkan_qc = df_qc[df_qc["Peak"].isin(alkane_ids)].copy()

    if log_transform:
        alkan_qc["Log_Intensity"] = np.log10(alkan_qc["Intensity"] + 1)
        y_col = "Log_Intensity"
        y_label = "Log₁₀ Intensity"
    else:
        y_col = "Intensity"
        y_label = "Intensity"
    plt.figure(figsize=(14, 10))
    stability_records = []

    for i, peak in enumerate(alkane_ids, 1):
        sub = alkan_qc[alkan_qc["Peak"] == peak].dropna(subset=[y_col, "Injection_order"]).copy()
        if sub.empty:
            continue

        plt.subplot(3, 3, i)
        r2_values = []
        for qc in qc_levels:
            sub_qc = sub[sub["Sample"].str.startswith(qc)].copy()
            if sub_qc.empty:
                continue

            x = sub_qc["Injection_order"].values
            y = sub_qc[y_col].values

            if len(sub_qc) > 1:
                model = LinearRegression().fit(x.reshape(-1, 1), y)
                y_pred = model.predict(x.reshape(-1, 1))
                r2 = model.score(x.reshape(-1, 1), y)
            else:
                y_pred = y
                r2 = np.nan

            r2_values.append(r2)
            #  Correct stability condition
            stability = "Stable" if (not np.isnan(r2) and r2 < r2_threshold) else "Unstable "
            color = colors.get(qc, "gray")

            plt.scatter(x, y, alpha=0.7, label=f"{qc} (R²={r2:.2f}, {stability})", color=color)
            linestyle = "-" if "Stable" in stability else "--"
            plt.plot(x, y_pred, linestyle=linestyle, color=color)
            stability_records.append({
                "Peak": peak,
                "QC_Level": qc,
                "R2": r2,
                "Stability": stability,
                "tR_best": sub["tR_best"].iloc[0],
                "m/z": sub["m/z"].iloc[0]
            })

        #  Determine overall stability for this peak
        max_r2 = np.nanmax(r2_values) if r2_values else np.nan
        overall_status = "Stable " if max_r2 < r2_threshold else "Unstable"
        title_color = "green" if "Stable" in overall_status else "red"

        plt.title(
            f"Alkane Peak {peak} ({overall_status})\n"
            f"tR={sub['tR_best'].iloc[0]:.2f}, m/z={sub['m/z'].iloc[0]}",
            color=title_color
        )
        plt.xlabel("Injection order")
        plt.ylabel(y_label)
        plt.legend(fontsize=8)

    plt.tight_layout()
    plt.show()
    stability_df = pd.DataFrame(stability_records)
    return stability_df






def apply_alkane_drift_correction_global(df_qc, selected_data, stable_peaks, config_path, loess_frac=0.3, dataset_key="biorep1"):

    """
    Applies LOESS-based global drift correction using alkane QC peaks.
    Shows before/after evaluation as boxplots (no per-peak metrics).
    Parameters
    ----------
    df_qc : pd.DataFrame
        QC dataset (must contain 'Injection_order', 'Peak', 'Intensity', and 'Sample').
    selected_data : pd.DataFrame
        Sample dataset in wide format (metadata + intensity columns).
    stable_peaks : pd.DataFrame
        DataFrame of alkane peaks ['Peak', 'tR_best', 'm/z'].
    config_path : str
        Path to YAML configuration file containing 'master_path'.
    loess_frac : float
        Fraction of data used for LOESS smoothing (default=0.3).

    Returns
    -------
    dict
        {
            "df_corrected": corrected data,
            "qc_r2_before": global R² before correction,
            "qc_r2_after": global R² after correction,
            "var_before": overall variance before correction,
            "var_after": overall variance after correction
        }
    """
    #  Load master table from config 
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

        try:
            master_path = config[dataset_key]["qc_merge"]["master_path"]
        except KeyError:
            raise KeyError(f"master_path not found at config['{dataset_key}']['qc_merge']['master_path']")

        print(f"Loaded master table from: {master_path}")
        master_table = pd.read_csv(master_path)   # <-- REQUIRED


    #  Estimate drift from alkane QCs 
    alkane_ids = stable_peaks["Peak"].unique()
    alkan_qc = df_qc[df_qc["Peak"].isin(alkane_ids)].copy()
    alkan_qc["QC_label"] = alkan_qc["Sample"].str.extract(r"(QC\d+)")
    alkan_qc["Log_Intensity"] = np.log10(alkan_qc["Intensity"] + 1)

    x = alkan_qc["Injection_order"].values
    y = alkan_qc["Log_Intensity"].values
    loess_fit = lowess(y, x, frac=loess_frac, return_sorted=True)

    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, alpha=0.3, label="Alkane QCs (log-intensity)")
    plt.plot(loess_fit[:, 0], loess_fit[:, 1], color="red", linewidth=2, label="Global drift (LOESS)")
    plt.xlabel("Injection order")
    plt.ylabel("Log₁₀ Intensity")
    plt.title("Global drift estimated from alkane QCs")
    plt.legend()
    plt.show()
    #Prepare sample data 
    metadata_cols = ['Peak', 'tR_best', 'm/z']
    sample_cols = [c for c in selected_data.columns if c not in metadata_cols]
    rename_map = {col: col.split()[-1] for col in sample_cols}
    selected_data = selected_data.rename(columns=rename_map)
    sample_cols = [rename_map[c] for c in sample_cols]

    long_samples = selected_data.melt(
        id_vars=metadata_cols,
        value_vars=sample_cols,
        var_name="Filename",
        value_name="Intensity"
    )

    samples_merged = master_table.merge(long_samples, on="Filename", how="inner")
    samples_merged["Intensity"] = pd.to_numeric(samples_merged["Intensity"], errors="coerce")
    samples_merged = samples_merged.dropna(subset=["Intensity", "Injection_order"])

    all_data = pd.concat([samples_merged, df_qc], ignore_index=True)
    # Apply global drift correction
    corrected_records = []
    for peak in all_data["Peak"].unique():
        sub = all_data[all_data["Peak"] == peak].copy()
        drift_factor = np.interp(sub["Injection_order"], loess_fit[:, 0], loess_fit[:, 1])
        log_intensity = np.log10(sub["Intensity"] + 1)
        log_corrected = log_intensity - (drift_factor - drift_factor.mean())
        sub["Intensity_corrected"] = 10 ** log_corrected - 1
        corrected_records.append(sub)

    df_corrected = pd.concat(corrected_records, ignore_index=True)

    # Global QC evaluation
    qc_raw = df_qc[df_qc["Sample"].astype(str).str.startswith("QC")]
    qc_corr = df_corrected[df_corrected["Sample"].astype(str).str.startswith("QC")]

    X_raw = qc_raw["Injection_order"].values.reshape(-1, 1)
    y_raw = np.log10(qc_raw["Intensity"].values + 1)
    qc_r2_before = LinearRegression().fit(X_raw, y_raw).score(X_raw, y_raw)

    X_corr = qc_corr["Injection_order"].values.reshape(-1, 1)
    y_corr = np.log10(qc_corr["Intensity_corrected"].values + 1)
    qc_r2_after = LinearRegression().fit(X_corr, y_corr).score(X_corr, y_corr)

    print(f" Global QC R² before: {qc_r2_before:.4f}")
    print(f" Global QC R² after : {qc_r2_after:.4f}")

    # Boxplot for QC drift 
    qc_box = pd.DataFrame({
        "Stage": ["Before"] * len(y_raw) + ["After"] * len(y_corr),
        "Log_Intensity": np.concatenate([y_raw, y_corr])
    })
    plt.figure(figsize=(6, 5))
    sns.boxplot(data=qc_box, x="Stage", y="Log_Intensity", palette=["salmon", "seagreen"])
    plt.title("QC Log₁₀ Intensities Before vs After Correction")
    plt.ylabel("Log₁₀ Intensity")
    plt.show()
    # Global sample variance before/after 
    samples_raw = samples_merged["Intensity"]
    samples_corr = df_corrected[
        ~df_corrected["Sample"].astype(str).str.startswith("QC")
    ]["Intensity_corrected"]

    var_before = np.var(np.log10(samples_raw + 1))
    var_after = np.var(np.log10(samples_corr + 1))

    print(f" Global sample variance before: {var_before:.6f}")
    print(f" Global sample variance after : {var_after:.6f}")

    #Boxplot for sample intensities 
    sample_box = pd.DataFrame({
        "Stage": ["Before"] * len(samples_raw) + ["After"] * len(samples_corr),
        "Log_Intensity": np.concatenate([np.log10(samples_raw + 1), np.log10(samples_corr + 1)])
    })
    plt.figure(figsize=(6, 5))
    sns.boxplot(data=sample_box, x="Stage", y="Log_Intensity", palette=["salmon", "seagreen"])
    plt.title("Sample Log₁₀ Intensities Before vs After Correction")
    plt.ylabel("Log₁₀ Intensity")
    plt.show()
    print(" Drift correction and global boxplot evaluation completed.")
    return {
        "df_corrected": df_corrected,
        "qc_r2_before": qc_r2_before,
        "qc_r2_after": qc_r2_after,
        "var_before": var_before,
        "var_after": var_after
    }





def plot_qc_drift_slopes(df_qc, df_corrected):
    """
    Calculate and visualize QC drift slopes before and after correction.

    Parameters
    ----------
    df_qc : pd.DataFrame
        Original QC data before correction.
        Must include columns: ['Peak', 'Sample', 'Injection_order', 'Intensity'].
    df_corrected : pd.DataFrame
        Corrected data after drift correction.
        Must include columns: ['Peak', 'Sample', 'Injection_order', 'Intensity_corrected'].

    Returns
    -------
    slopes_df : pd.DataFrame
        DataFrame with slope per QC before and after correction.
    """
    # Calculate slopes before correction 
    slopes_before = []
    for peak in df_qc["Peak"].unique():
        sub = df_qc[df_qc["Peak"] == peak].copy()
        sub["QC_label"] = sub["Sample"].str.extract(r"(QC\d+)")
        for qc in sub["QC_label"].dropna().unique():
            sub_qc = sub[sub["QC_label"] == qc]
            if len(sub_qc) > 2:
                X = sub_qc["Injection_order"].values.reshape(-1, 1)
                y = np.log10(sub_qc["Intensity"].values + 1)
                model = LinearRegression().fit(X, y)
                slope = model.coef_[0]
                slopes_before.append((qc, slope))

    slopes_before_df = pd.DataFrame(slopes_before, columns=["QC", "Slope_before"])

    # Calculate slopes after correction 
    slopes_after = []
    for peak in df_corrected["Peak"].unique():
        sub = df_corrected[df_corrected["Peak"] == peak].copy()
        sub["QC_label"] = sub["Sample"].str.extract(r"(QC\d+)")
        for qc in sub["QC_label"].dropna().unique():
            sub_qc = sub[sub["QC_label"] == qc]
            if len(sub_qc) > 2:
                X = sub_qc["Injection_order"].values.reshape(-1, 1)
                y = np.log10(sub_qc["Intensity_corrected"].values + 1)
                model = LinearRegression().fit(X, y)
                slope = model.coef_[0]
                slopes_after.append((qc, slope))

    slopes_after_df = pd.DataFrame(slopes_after, columns=["QC", "Slope_after"])

    # Merge before/after slopes 
    slopes_df = pd.merge(slopes_before_df, slopes_after_df, on="QC", how="inner")

    # Melt for plotting 
    slopes_melt = pd.melt(
        slopes_df,
        id_vars="QC",
        value_vars=["Slope_before", "Slope_after"],
        var_name="Stage",
        value_name="Slope"
    )

    # Boxplot visualization 
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=slopes_melt, x="QC", y="Slope", hue="Stage",
                palette=["#FF9999", "#66CC99"], linewidth=1.2)
    plt.axhline(0, color="red", linestyle="--", label="No drift")
    plt.title("Distribution of Drift Slopes per QC Level (Before vs After Correction)", fontsize=12)
    plt.ylabel("Slope (log₁₀ intensity / injection order)")
    plt.xlabel("QC Level")
    plt.legend(title="Stage", loc="upper right")
    plt.tight_layout()
    plt.show()
    # summary stats
    print("\n Mean slope before correction:", slopes_df["Slope_before"].mean())
    print(" Mean slope after correction :", slopes_df["Slope_after"].mean())

    print("\n Drift slope evaluation complete.")
    return slopes_df





def optimize_loess_drift_correction(df_qc, stable_peaks, frac_values=[0.2, 0.3, 0.4],
                                    alpha_values=[0.01, 0.05, 0.1],
                                    delta_values=[0.0, 1.0, 5.0]):
    """
    Optimize LOESS parameters (frac, alpha, delta) for signal drift correction 
    in QC chromatographic data using alkanes as stable reference peaks.
    This function performs a grid search over specified LOESS parameters,
    fits the LOESS model to log-transformed QC alkanes, applies drift correction
    across the dataset, and evaluates performance by comparing the R² 
    improvement (before vs. after correction) for each peak.
    Parameters
    ----------
    df_qc : pandas.DataFrame
        QC dataset containing 'Peak', 'Injection_order', and 'Intensity' columns.
    stable_peaks : pandas.DataFrame
        Subset of stable alkanes used as drift reference peaks.
    frac_values : list of float, optional
        List of LOESS smoothing fractions to test.
    alpha_values : list of float, optional
        List of significance levels for evaluation (kept for future extensions).
    delta_values : list of float, optional
        List of delta values for LOESS distance weighting to test.
    Returns
    -------
    best_params : pandas.Series
        Row from the optimization results containing the best frac, alpha, delta, 
        and mean improvement.
    opt_df : pandas.DataFrame
        Full results table of parameter combinations and mean R² improvements.
    heatmap : matplotlib.figure.Figure
        A heatmap visualization of the mean improvements across frac and delta.
    """
    results = []
    # only QC alkanes for LOESS fitting
    alkan_qc = df_qc[df_qc["Peak"].isin(stable_peaks["Peak"].unique())].copy()
    alkan_qc["Log_Intensity"] = np.log10(alkan_qc["Intensity"] + 1)

    # Grid search over parameters
    for frac in frac_values:
        for alpha in alpha_values:
            for delta in delta_values:
                # Fit LOESS on alkanes
                smoothed = lowess(
                    endog=alkan_qc["Log_Intensity"],
                    exog=alkan_qc["Injection_order"],
                    frac=frac,
                    delta=delta,
                    return_sorted=False
                )

                # Apply correction to full dataset
                df_tmp = df_qc.copy()
                df_tmp["Log_Intensity"] = np.log10(df_tmp["Intensity"] + 1)
                df_tmp["Drift"] = np.interp(
                    df_tmp["Injection_order"],
                    alkan_qc["Injection_order"],
                    smoothed
                )
                df_tmp["Corrected"] = df_tmp["Log_Intensity"] - df_tmp["Drift"]

                # Evaluate correction performance
                improvements = []
                for peak in df_tmp["Peak"].unique():
                    sub = df_tmp[df_tmp["Peak"] == peak]
                    if len(sub) < 5:
                        continue
                    X_raw = sub["Injection_order"].values.reshape(-1, 1)
                    y_raw = sub["Log_Intensity"].values
                    r2_before = LinearRegression().fit(X_raw, y_raw).score(X_raw, y_raw)
                    y_corr = sub["Corrected"].values
                    r2_after = LinearRegression().fit(X_raw, y_corr).score(X_raw, y_corr)
                    improvements.append(r2_before - r2_after)

                if len(improvements) > 0:
                    mean_improvement = np.mean(improvements)
                    results.append((frac, alpha, delta, mean_improvement))

    # results
    opt_df = pd.DataFrame(results, columns=["frac", "alpha", "delta", "mean_improvement"])
    best_params = opt_df.loc[opt_df["mean_improvement"].idxmax()]

    # Plot heatmap (frac vs delta)
    heatmap_df = opt_df.groupby(["frac", "delta"])["mean_improvement"].mean().unstack()
    plt.figure(figsize=(7, 5))
    sns.heatmap(heatmap_df, annot=True, cmap="coolwarm", center=0)
    plt.title("Optimization of LOESS Parameters for QC Drift Correction")
    plt.ylabel("frac (smoothing parameter)")
    plt.xlabel("delta (distance weighting)")
    plt.show()

    return best_params, opt_df




def evaluate_qc_reproducibility(df_corrected):
    """
    Evaluate QC sample reproducibility after drift correction.

    This function computes key QC quality metrics — including median intensity,
    coefficient of variation (CV), and data completeness — for each QC sample.
    It also derives a composite score based on signal stability and intensity,
    ranking QCs from most to least reproducible.
    Parameters
    ----------
    df_corrected : pd.DataFrame
        Drift-corrected dataset containing at least:
        ['Sample', 'Intensity_corrected'] columns.

    Returns
    -------
    qc_stats : pd.DataFrame
        DataFrame containing per-QC summary statistics:
        ['Sample', 'Median_intensity', 'CV', 'Completeness', 'Score', 'Rank']
    """

    # Extract only QC rows 
    qc_corrected = df_corrected[df_corrected["Sample"].astype(str).str.startswith("QC")].copy()

    # Compute per-QC metrics 
    qc_stats = qc_corrected.groupby("Sample").agg(
        Median_intensity=("Intensity_corrected", "median"),
        CV=("Intensity_corrected", lambda x: np.std(x) / np.mean(x) if np.mean(x) > 0 else np.nan),
        Completeness=("Intensity_corrected", lambda x: x.notna().mean())
    ).reset_index()

    # Compute composite score 
    qc_stats["Score"] = qc_stats["Median_intensity"].apply(np.log1p) / qc_stats["CV"]

    # Rank QCs (higher score = better reproducibility)
    qc_stats["Rank"] = qc_stats["Score"].rank(ascending=False).astype(int)

    # Display sorted summary 
    qc_sorted = qc_stats.sort_values("Rank").reset_index(drop=True)
    print("QC reproducibility ranking (best → worst):")
    print(qc_sorted)

    return qc_sorted




def plot_qc_reproducibility(qc_stats):
    """
    Visualize QC reproducibility ranking based on computed scores.

    Parameters
    ----------
    qc_stats : pd.DataFrame
        Output from evaluate_qc_reproducibility(), containing columns:
        ['Sample', 'Score', 'Rank'].

    Returns
    -------
    None
    """
    qc_sorted = qc_stats.sort_values("Rank")

    plt.figure(figsize=(7, 5))
    sns.barplot(data=qc_sorted, x="Sample", y="Score", palette="viridis")
    plt.title("QC Reproducibility Ranking (Higher = Better)", fontsize=13)
    plt.xlabel("QC Sample")
    plt.ylabel("Composite Reproducibility Score (log(Median) / CV)")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()




def normalize_samples(df_corrected, qc_label="QC1"):
    """
    Normalize sample intensities using a reference QC (default = QC1).

    Parameters
    ----------
    df_corrected : pd.DataFrame
        Drift-corrected dataset containing:
        ['Sample', 'Peak', 'Intensity_corrected'] columns.
    qc_label : str
        QC sample name used as normalization reference (default='QC1').

    Returns
    -------
    pd.DataFrame
        A copy of the sample data with added normalization columns:
        ['QC1_median', 'Ratio_normalized', 'Log10_normalized'].
    """


    #  Separate QCs and samples 
    qcs = df_corrected[df_corrected["Sample"].astype(str).str.startswith("QC")].copy()
    samples = df_corrected[~df_corrected["Sample"].astype(str).str.startswith("QC")].copy()

    # Compute median intensity per peak from reference QC
    qc_ref = (
        qcs[qcs["Sample"].astype(str).str.startswith(qc_label)]
        .groupby("Peak")["Intensity_corrected"]
        .median()
        .rename(f"{qc_label}_median")
    )

    # Merge QC reference with sample data 
    samples = samples.merge(qc_ref, on="Peak", how="left")

    #  Normalize using QC reference 
    samples["Ratio_normalized"] = samples["Intensity_corrected"] / samples[f"{qc_label}_median"]

    # Log10-transform normalized ratios 
    samples["Log10_normalized"] = np.log10(samples["Ratio_normalized"] + 1)

    print(f" Samples normalized relative to {qc_label}.")
    print(f"Number of samples normalized: {len(samples)}")
    print(f"Number of peaks referenced: {qc_ref.shape[0]}")
    return samples





def extract_and_export_samples(samples, output_path="samples_filtered.csv"):
    """
    Extract the 'Variety' name from the Sample column, filter relevant columns,
    and export the cleaned dataset to a CSV file.

    Parameters
    ----------
    samples : pd.DataFrame
        DataFrame containing at least the following columns:
        ['Filename', 'Sample', 'Peak', 'tR_best', 'm/z',
         'Intensity_corrected', 'Ratio_normalized', 'Log10_normalized'].
    output_path : str, optional
        Path to save the filtered CSV file (default = 'samples_filtered.csv').

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with added 'Variety' column.
    """

    # Extract Variety 
    samples = samples.copy()
    samples["Variety"] = samples["Sample"].astype(str).str.split(";").str[-1]

    # Filter relevant columns 
    keep_cols = [
        "Filename", "Sample", "Peak", "Variety",
        "tR_best", "m/z", "Intensity_corrected",
        "Ratio_normalized", "Log10_normalized"
    ]
    samples_filtered = samples[keep_cols].copy()

    # Save 
    samples_filtered.to_csv(output_path, index=False)
    print(f" Filtered samples saved to: {output_path}")
    print(f"Number of rows exported: {len(samples_filtered)}")

    return samples_filtered





def evaluate_normalization_effect(samples, output_path=None):
    """
    Evaluate and visualize the effect of normalization on sample intensities.

    Parameters
    ----------
    samples : pd.DataFrame
        Must contain 'Intensity_corrected' (before normalization)
        and 'Ratio_normalized' (after normalization) columns.
    output_path : str, optional
        If provided, saves the plot to this file (e.g., 'normalization_effect.png').

    Returns
    -------
    dict
        {
            "variance_before": float,
            "variance_after": float,
            "reduction_percent": float
        }
    """

    # compute log10 values before and after normalization 
    samples_plot = samples.copy()
    samples_plot["Log10_before"] = np.log10(samples_plot["Intensity_corrected"] + 1)
    samples_plot["Log10_after"] = np.log10(samples_plot["Ratio_normalized"] + 1)

    #  Combine for plotting
    plot_df = samples_plot.melt(
        value_vars=["Log10_before", "Log10_after"],
        var_name="Stage",
        value_name="Log10 Intensity"
    )

    plot_df["Stage"] = plot_df["Stage"].map({
        "Log10_before": "Before normalization",
        "Log10_after": "After normalization"
    })

    # Plot boxplot 
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=plot_df, x="Stage", y="Log10 Intensity",
                palette=["salmon", "seagreen"])
    plt.title("Effect of Normalization on Sample Intensities (log₁₀ scale)")
    plt.ylabel("Log₁₀ Intensity")
    plt.xlabel("")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    # Save plot 
    if output_path:
        plt.savefig(output_path, dpi=300)
        print(f" Plot saved to {output_path}")
    else:
        plt.show()

    # Compute and print variance reduction 
    var_before = np.var(np.log10(samples["Intensity_corrected"] + 1))
    var_after = np.var(np.log10(samples["Ratio_normalized"] + 1))
    reduction = (1 - var_after / var_before) * 100
    print("\n Normalization Effect Summary")
    print(f"Variance before normalization : {var_before:.6f}")
    print(f"Variance after normalization  : {var_after:.6f}")
    print(f"Reduction in variance         : {reduction:.2f}%")

    return {
        "variance_before": var_before,
        "variance_after": var_after,
        "reduction_percent": reduction
    }





def attach_variety_info(samples):
    """
    Extract and attach Variety information to the samples DataFrame
    based on the last token in the 'Sample' column (split by ';').

    Parameters
    ----------
    samples : pd.DataFrame
        Must contain 'Filename' and 'Sample' columns.

    Returns
    -------
    pd.DataFrame
        A cleaned DataFrame containing 'Filename', 'Sample', and 'Variety' columns,
        with duplicate entries removed and consistent column naming.
    """

    # Extract sample info 
    sample_info = samples[["Filename", "Sample"]].drop_duplicates().copy()

    # Extract Variety (last token after ';')
    sample_info["Variety"] = sample_info["Sample"].astype(str).str.split(";").str[-1]

    # If Variety columns already exist and need merging 
    if "Variety_x" in samples.columns or "Variety_y" in samples.columns:
        samples = samples.rename(columns={"Variety_x": "Variety"}).drop(
            columns=["Variety_y"], errors="ignore"
        )

    # Merge back with main dataset if needed 
    samples_with_variety = samples.merge(sample_info, on=["Filename", "Sample"], how="left")
    print(f" Attached 'Variety' information for {samples_with_variety['Variety'].nunique()} varieties.")
    print(f"Total samples with assigned Variety: {len(samples_with_variety)}")
    return samples_with_variety





def perform_pca_interactive_biplot(samples_with_variety, 
                                          feature_col="Log10_normalized", 
                                          n_components=2, 
                                          top_n_loadings=5, 
                                          scale_factor=20,
                                          point_color="blue"):
    """
     Perform PCA and visualize results as an interactive biplot (uniform color, no variety grouping).

    Parameters
    ----------
    samples_with_variety : pd.DataFrame
        DataFrame with at least ['Filename', 'Peak', feature_col].
    feature_col : str, optional
        Column used as feature values for PCA (default: 'Log10_normalized').
    n_components : int, optional
        Number of PCA components to compute (default: 2).
    top_n_loadings : int, optional
        Number of top contributing loadings (peaks) to display as arrows (default: 5).
    scale_factor : float, optional
        Scaling factor for loadings arrow length (default: 20).
    point_color : str, optional
        Color of all sample points (default: 'blue').

    Returns
    -------
    dict
        {
            "pca_model": PCA object,
            "scores_df": DataFrame with PCA scores,
            "loadings_df": DataFrame with PCA loadings,
            "fig": Plotly figure object
        }
    """
 
    print("\n Performing PCA (uniform color mode)...")

    # Pivot to samples × peaks matrix 
    X = samples_with_variety.pivot_table(
        index="Filename",
        columns="Peak",
        values=feature_col,
        aggfunc="mean"
    )

    # Impute missing values 
    imputer = SimpleImputer(strategy="mean")
    X_filled = imputer.fit_transform(X)

    # Standardize 
    X_scaled = StandardScaler().fit_transform(X_filled)

    # PCA 
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X_scaled)

    # Create DataFrame for PCA scores 
    scores_df = pd.DataFrame(scores, index=X.index, columns=[f"PC{i+1}" for i in range(n_components)]).reset_index()

    #  Compute loadings 
    loadings = pca.components_.T[:, :2]
    loading_df = pd.DataFrame(loadings, columns=["PC1", "PC2"], index=X.columns)
    loading_df["abs_contrib"] = np.sqrt(loading_df["PC1"]**2 + loading_df["PC2"]**2)
    top_loadings = loading_df.nlargest(top_n_loadings, "abs_contrib")

    # Build the plot 
    fig = go.Figure()

    # Add samples (all one color)
    fig.add_trace(go.Scatter(
        x=scores_df["PC1"], 
        y=scores_df["PC2"],
        mode="markers",
        marker=dict(size=7, color=point_color, opacity=0.7),
        text=scores_df["Filename"],
        name="Samples"
    ))

    # Add loadings arrows
    for i, row in top_loadings.iterrows():
        fig.add_trace(go.Scatter(
            x=[0, row["PC1"] * scale_factor],
            y=[0, row["PC2"] * scale_factor],
            mode="lines+markers+text",
            line=dict(color="red", width=2),
            marker=dict(size=6, color="red"),
            text=[None, str(i)],
            textposition="top center",
            name=f"Peak {i}",
            hovertext=f"PC1: {row['PC1']:.3f}, PC2: {row['PC2']:.3f}"
        ))

    # Add axis lines
    fig.add_shape(type="line", x0=min(scores_df["PC1"]), x1=max(scores_df["PC1"]), y0=0, y1=0,
                  line=dict(color="gray", dash="dash"))
    fig.add_shape(type="line", x0=0, x1=0, y0=min(scores_df["PC2"]), y1=max(scores_df["PC2"]),
                  line=dict(color="gray", dash="dash"))

    fig.update_layout(
        title="PCA Biplot (Uniform Color)",
        xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)",
        yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)",
        width=900, height=900,
        template="simple_white"
    )

    print(f" PCA completed. Explained variance: PC1 = {pca.explained_variance_ratio_[0]*100:.2f}%, "
          f"PC2 = {pca.explained_variance_ratio_[1]*100:.2f}%")

    fig.show()

    return {
        "pca_model": pca,
        "scores_df": scores_df,
        "loadings_df": loading_df,
        "fig": fig
    }





def perform_hierarchical_clustering(samples_with_variety,
                                    feature_col="Log10_normalized",
                                    method="ward",
                                    n_clusters=5,
                                    plot_labels="Variety",
                                    figsize=(14, 6)):
    """
     Perform Hierarchical Clustering (HCA) and visualize dendrogram.

    This function:
    - Pivots data into a samples × peaks matrix.
    - Scales the features using StandardScaler.
    - Computes hierarchical clustering using the specified linkage method.
    - Assigns cluster labels to each sample.
    - Plots a dendrogram with sample or variety labels.

    Parameters
    ----------
    samples_with_variety : pd.DataFrame
        Must contain ['Filename', 'Peak', feature_col, 'Variety'].
    feature_col : str, optional
        Column used for clustering (default: 'Log10_normalized').
    method : str, optional
        Linkage method for hierarchical clustering ('ward', 'average', 'complete', etc.).
    n_clusters : int, optional
        Number of clusters to cut the dendrogram into (default: 5).
    plot_labels : str, optional
        Label to display on the dendrogram ('Variety' or 'Filename').
    figsize : tuple, optional
        Figure size for dendrogram plot (default: (14, 6)).

    Returns
    -------
    dict
        {
            "linkage_matrix": Z,
            "cluster_assignments": pd.DataFrame with ['Filename', 'Cluster', 'Variety'],
            "fig": matplotlib Figure object
        }
    """
    print("\n Performing Hierarchical Clustering...")

    # Pivot to samples × peaks matrix 
    X = samples_with_variety.pivot_table(
        index="Filename",
        columns="Peak",
        values=feature_col,
        aggfunc="mean"
    ).fillna(0)

    #  Scale features 
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Compute linkage matrix 
    Z = linkage(X_scaled, method=method)

    # Assign clusters 
    clusters = fcluster(Z, t=n_clusters, criterion="maxclust")
    cluster_df = pd.DataFrame({
        "Filename": X.index,
        "Cluster": clusters
    })

    # --- Step 5: Merge with metadata ---
    cluster_df = cluster_df.merge(
        samples_with_variety[["Filename", "Variety"]].drop_duplicates(),
        on="Filename", how="left"
    )

    print(f" Clustering completed using '{method}' linkage with {n_clusters} clusters.")
    print("First few cluster assignments:")
    print(cluster_df.head(10))

    # Plot dendrogram 
    plt.figure(figsize=figsize)
    dendrogram(
        Z,
        labels=cluster_df[plot_labels].tolist(),
        leaf_rotation=90,
        leaf_font_size=6,
        color_threshold=0
    )
    plt.title(f"Hierarchical Clustering Dendrogram ({plot_labels} labels)")
    plt.xlabel(f"{plot_labels}s")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.show()

    return {
        "linkage_matrix": Z,
        "cluster_assignments": cluster_df,
        "fig": plt.gcf()
    }





def evaluate_hierarchical_clustering_full(samples_with_variety,
                                          feature_col="Log10_normalized",
                                          method="ward",
                                          cluster_range=(2, 10)):
    """
     Comprehensive Evaluation of Hierarchical Clustering Quality
    --------------------------------------------------------------
    Computes multiple metrics to assess clustering stability and separation:
    - Cophenetic Correlation (structure preservation)
    - Silhouette Score (cohesion vs separation)
    - Calinski–Harabasz Index (cluster compactness)
    - Davies–Bouldin Index (lower is better)

    Parameters
    ----------
    samples_with_variety : pd.DataFrame
        DataFrame with ['Filename', 'Peak', feature_col].
    feature_col : str, optional
        Column for clustering features (default: 'Log10_normalized').
    method : str, optional
        Linkage method for hierarchical clustering (default: 'ward').
    cluster_range : tuple, optional
        Range of cluster numbers to test (default: (2, 10)).

    Returns
    -------
    dict
        {
            "linkage_matrix": Z,
            "cophenetic_corr": float,
            "metrics": pd.DataFrame with metrics across cluster counts
        }
    """
    print("\n Evaluating Hierarchical Clustering Across Multiple Metrics...")

    #Prepare data matrix 
    X = samples_with_variety.pivot_table(
        index="Filename",
        columns="Peak",
        values=feature_col,
        aggfunc="mean"
    ).fillna(0)

    #Scale features 
    X_scaled = StandardScaler().fit_transform(X)

    # Compute linkage and cophenetic correlation 
    Z = linkage(X_scaled, method=method)
    coph_corr, _ = cophenet(Z, pdist(X_scaled))
    print(f" Cophenetic correlation coefficient: {coph_corr:.3f}\n")

    # Evaluate multiple cluster counts 
    metrics = []
    for k in range(cluster_range[0], cluster_range[1]):
        labels = fcluster(Z, k, criterion="maxclust")

        sil = silhouette_score(X_scaled, labels)
        ch = calinski_harabasz_score(X_scaled, labels)
        db = davies_bouldin_score(X_scaled, labels)

        metrics.append({
            "Clusters": k,
            "Silhouette": sil,
            "Calinski-Harabasz": ch,
            "Davies-Bouldin": db
        })
        print(f"k={k}: silhouette={sil:.3f}, calinski={ch:.1f}, davies={db:.3f}")

    metrics_df = pd.DataFrame(metrics)

    # Plot all metrics 
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(metrics_df["Clusters"], metrics_df["Silhouette"], 'o-', label="Silhouette", color="blue")
    ax1.set_ylabel("Silhouette Score", color="blue")
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_xlabel("Number of Clusters")

    ax2 = ax1.twinx()
    ax2.plot(metrics_df["Clusters"], metrics_df["Calinski-Harabasz"], 's--', label="Calinski–Harabasz", color="green")
    ax2.plot(metrics_df["Clusters"], metrics_df["Davies-Bouldin"], 'd-.', label="Davies–Bouldin", color="red")
    ax2.set_ylabel("Other Metrics", color="black")

    fig.suptitle("Hierarchical Clustering Evaluation Metrics", fontsize=14)
    fig.legend(loc="upper right", bbox_to_anchor=(0.85, 0.85))
    plt.tight_layout()
    plt.show()

    print("\n Evaluation complete. Use Silhouette (↑), Calinski–Harabasz (↑), and Davies–Bouldin (↓) to choose optimal k.")

    return {
        "linkage_matrix": Z,
        "cophenetic_corr": coph_corr,
        "metrics": metrics_df
    }




def plot_pca_clusters(pca_results, clustering_results, optimal_k=2, color_sequence=None):
    """
     Visualize Hierarchical Clustering Results on PCA Plot (Interactive)
    This function:
    - Uses PCA scores from `pca_results`
    - Uses the linkage matrix from hierarchical clustering results
    - Assigns cluster labels for the specified optimal cluster number (k)
    - Displays an interactive 2D PCA plot colored by cluster

    Parameters
    ----------
    pca_results : dict
        Output dictionary from the PCA function.
        Must contain keys: ['pca_model', 'scores_df'].
    clustering_results : dict
        Output dictionary from the hierarchical clustering evaluation.
        Must contain key: ['linkage_matrix'].
    optimal_k : int
        Number of clusters to display (default: 2).
    color_sequence : list
        Custom list of colors for the clusters (default: Plotly categorical colors).

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive PCA scatter plot colored by clusters.
    """

    print(f"\n Generating PCA plot with {optimal_k} hierarchical clusters...")

    #  clusters 
    Z = clustering_results["linkage_matrix"]
    cluster_labels = fcluster(Z, t=optimal_k, criterion="maxclust")

    #Add cluster info to PCA scores 
    pca_scores = pca_results["scores_df"].copy()
    pca_scores["Cluster"] = cluster_labels

    # Define colors
    if color_sequence is None:
        # Default to Plotly palette
        color_sequence = px.colors.qualitative.Set1[:optimal_k]

    # Create interactive PCA scatter plot 
    fig = px.scatter(
        pca_scores,
        x="PC1",
        y="PC2",
        color=pca_scores["Cluster"].astype(str),
        symbol=pca_scores["Cluster"].astype(str),
        hover_data=["Filename"],
        title=f"PCA Plot Colored by Hierarchical Clusters (k={optimal_k})",
        labels={
            "PC1": f"PC1 ({pca_results['pca_model'].explained_variance_ratio_[0]*100:.1f}% var)",
            "PC2": f"PC2 ({pca_results['pca_model'].explained_variance_ratio_[1]*100:.1f}% var)",
            "Cluster": "Cluster"
        },
        color_discrete_sequence=color_sequence
    )

    fig.update_traces(marker=dict(size=10, opacity=0.8))
    fig.update_layout(
        width=900, height=800,
        legend_title_text="Cluster",
        template="simple_white"
    )

    print(" PCA cluster visualization ready.")
    fig.show()

    return fig