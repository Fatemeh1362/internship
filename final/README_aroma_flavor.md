# Potato Aroma & Flavor (TD–GC–MS) Pipeline — Biorep1 + Biorep2

## 1) Overview
This project implements a **config-driven** workflow for **potato aroma and flavor analysis** using **TD–GC–MS** data, with optional integration of **sensory profiling** and **genetic marker (taglo/SNP) data**. The pipeline is designed to be reusable for  **thesis work**, and is executed for two independent batches:

- **Biorep1** — development batch (main analysis)
- **Biorep2** — independent batch validation (generalisation check)

Core engineering challenges addressed:
- QC validation and reproducible preprocessing
- injection-order drift correction (LOWESS/LOESS)
- QC-based normalisation for robust intensity comparability
- compound identification integration (NIST compound tables)
- contamination filtering using blank/environment/QC references
- aroma matrix construction (Variety × Compound)
- multivariate analysis (PCA/clustering)
- GWAS-style association analysis and SNP/taglo prioritisation (single-trait and multi-trait)
-  predictive modelling
- genotype × VOC × sensory integration 

---

## 2) Purpose
The primary objective is to create a **reproducible, end-to-end workflow** that converts raw TD–GC–MS outputs into analysis-ready datasets and interpretable results.

Key goals include:
- Standardising preprocessing across batches (Biorep1/Biorep2)
- Producing consistent **aroma phenotype matrices** for downstream statistics/ML
- Linking aroma traits to **sensory attributes** and **genetic markers**
- Supporting both exploratory analysis (heatmap,clustering/outliers), multivariate analysis(PLSR, PCA) and modelling (CV-based prediction)

---

## 3) Methodology (Pipeline)

### Step 1 — Configuration & project setup
All notebooks read from `config.yaml`, which centralises:
- input file paths (QC, peak tables, compound lists, sensory, genetics)
- output paths (CSV exports, figures)
- parameters (e.g., retention-time tolerance, clustering settings)

**Config file:** `config.yaml`

---

### Step 2 — Data preparation (QC → drift-corrected + normalised peak tables)
This step transforms raw MsMetrix peak tables into clean, analysis-ready intensity tables.

**Notebooks**
- `final_data_prepration_biorep1.ipynb`
- `final_data_prepration_biorep2.ipynb`

**Module**
- `datapreparation_module.py`

**What happens here**
- Read QC/sample peak tables (Excel or delimited text)
- Standardise columns and compute retention-time summaries (e.g., `tR_best`)
- QC validation (e.g., linearity trends; optional dashboard)
- Select stable peaks
- Correct injection-order drift using LOWESS/LOESS
- QC-based normalisation
- Export corrected/normalised outputs for downstream compound matching


### Step 3 — Compound analysis (NIST IDs → matched intensity → aroma matrix)
This step integrates compound identification tables with corrected peak intensities and produces compound-level outputs.

**Notebooks**
- `final_compound_analyis_biorep1.ipynb`
- `final_compound_analyis_biorep2.ipynb`

**Module**
- `compound_analysis_module.py`

**What happens here**
- Combine compound identification tables for **samples + blank + environment**
- Apply contamination filtering (remove overlaps with blank/env within RT tolerance)
- Match compounds to quantified peaks using retention time (RT ± tolerance)
- Construct outputs:
  - long-format matched table (per variety/compound)
  - wide **aroma matrix** (Variety × Compound)
-  chemical-family annotation and exploratory multivariate analysis (PCA/clustering/outliers)



### Step 4 — Sensory integration & exploratory analysis 
If sensory profiling is available, this step aligns sensory summaries with VOC profiles.

**Implemented via**
- functions in `compound_analysis_module.py`
- notebook-level steps inside the compound/gene notebooks

Typical actions:
- clean sensory tables and aggregate per Variety
- merge sensory PCA scores with aroma matrices
- compute correlations and summary visualisations


### Step 5 — Genetics integration & modelling (genotype × VOC × sensory)
This step merges genetic markers with aroma phenotypes (and optionally sensory) and runs modelling / association-style summaries.

**Notebooks**
- `final_gene_compounds_biorep1.ipynb`
- `final_gene_compounds_biorep2.ipynb`

**Module**
- `genemarker_module.py`

**What happens here**
- Align Variety identifiers across datasets
- Merge:
  - genetic + aroma
  - genetic + aroma + sensory 
- Exploratory PCA / PLSR where relevant
- GWAS-style reporting/visualisation 
- Predictive modelling with CV (e.g., linear models, Kernel Ridge, RF/SVR/XGBoost)
- Use **Biorep2** as an independent generalisation check when configured


## 4) Execution

### Run order (recommended)

**Biorep1**
1. `final_data_prepration_biorep1.ipynb`
2. `final_compound_analyis_biorep1.ipynb`
3. `final_gene_compounds_biorep1.ipynb`

**Biorep2 (independent validation)**
1. `final_data_prepration_biorep2.ipynb`
2. `final_compound_analyis_biorep2.ipynb`
3. `final_gene_compounds_biorep2.ipynb`

> Note:  Behaviour is controlled by selecting `config["biorep1"]` vs `config["biorep2"]`.


## 5) Requirements (Dependencies)

- `python>=3.9`
import os
import sys
import re
import glob
import io
import base64
import tempfile
import importlib
from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
import yaml
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse
import matplotlib.patches as mpatches
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingRegressor,
)
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    LogisticRegression,
    RidgeCV,
    ElasticNetCV,
    MultiTaskElasticNetCV,
)
from sklearn.svm import SVR, SVC
from sklearn.kernel_ridge import KernelRidge
from xgboost import XGBRegressor
import networkx as nx
from IPython.display import display, HTML
import genemarker_module
import compound_analysis_module
import datapreparation_module
importlib.reload(genemarker_module)
importlib.reload(compound_analysis_module)
importlib.reload(datapreparation_module)
```


## 6) Conclusion
This pipeline provides a structured way to go from **raw TD–GC–MS outputs** to **analysis-ready VOC phenotypes**, and to connect those phenotypes to **sensory** and **genetic** information in a consistent, batch-aware way. Running both **Biorep1** and **Biorep2** enables stronger conclusions by checking whether signals generalise beyond a single batch.

---

"Author": Fatemeh Monfared