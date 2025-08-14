# app.py — Potato Flavor Dashboard (Plotly Dash)
# Run: pip install dash plotly pandas numpy
#      python app.py
# Or from notebook: import app; app.launch(X=X, df0=df0, dom_score=dom_score, share_desc=share_desc)

import os, re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, dash_table

# ------------------------ CONFIG ------------------------
P_SHARE, P_Z = 0.05, 0.50
D_SHARE, D_Z = 0.15, 1.00
FLAVOR_COLS = [
    'Sweet','Metallic Flavour','Bitter Flavour','Earthy Flavour','Sour Flavour',
    'Fresh Flavour','Sweet Flavour','Root/ Vegetable Flavour',
    'Farmyard (grass/hay) flavour','Bitter Aftertaste','Sour Aftertaste','Sweet Aftertaste'
]

# ------------------- UTILITIES -------------------------
def canon_name(s: str) -> str:
    return re.sub(r'\s*\(.*\)', '', str(s)).strip()

def _safe_load_csv(path):
    try:
        if os.path.exists(path):
            return pd.read_csv(path)
    except Exception:
        pass
    return None

def compute_X_from_compound_matrix(compound_matrix: pd.DataFrame):
    X = compound_matrix.copy()
    X.index = X.index.astype(str)
    X = X.apply(pd.to_numeric, errors='coerce').clip(lower=0)
    X.columns = [canon_name(c) for c in X.columns]
    X = X.T.groupby(level=0).sum().T
    return X

def compute_overview_from_X(X: pd.DataFrame):
    row_sum = X.sum(axis=1).replace(0, np.nan)
    share = X.div(row_sum, axis=0)
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=0).replace(0, np.nan)
    z = (X - mu) / sd
    present  = (share.ge(P_SHARE)) | (z.ge(P_Z))
    dominant = (share.ge(D_SHARE)) | (z.ge(D_Z))
    dom_score = (0.6*share + 0.4*np.tanh(z/2)).fillna(0.0)
    return share, z, present, dominant, dom_score

def flavor_table_from_df0(df0: pd.DataFrame):
    if df0 is None:
        return pd.DataFrame()
    df = df0.copy()
    if 'Variety' in df.columns:
        df = df.dropna(subset=['Variety']).set_index('Variety')
    df.index = df.index.astype(str)
    cols = [c for c in FLAVOR_COLS if c in df.columns]
    if not cols:
        return pd.DataFrame(index=df.index)
    df = df[cols].apply(pd.to_numeric, errors='coerce')
    df = df.loc[:, df.notna().any(axis=0)].fillna(0.0)
    if df.columns.duplicated().any():
        df = df.T.groupby(level=0).mean().T
    return df

def make_radar(series: pd.Series, title="Flavor profile"):
    cats = list(series.index)
    vals = series.values.astype(float).tolist()
    fig = go.Figure()
    if not cats:
        fig.update_layout(title=title)
        return fig
    cats_closed = cats + [cats[0]]
    vals_closed = vals + [vals[0]]
    fig.add_trace(go.Scatterpolar(r=vals_closed, theta=cats_closed, mode='lines+markers', fill='toself'))
    fig.update_layout(title=title, polar=dict(radialaxis=dict(visible=True)),
                      margin=dict(l=20, r=20, t=60, b=20))
    return fig

def make_bar(names, vals, title, ylab="Value"):
    fig = go.Figure(go.Bar(x=names, y=vals))
    fig.update_layout(title=title, yaxis_title=ylab, margin=dict(l=20, r=20, t=60, b=60))
    return fig

def topk_series(s: pd.Series, k: int):
    if s is None or s.empty:
        return [], []
    s = s.dropna().sort_values(ascending=False).head(k)
    return list(s.index), list(s.values)

# ------------------- DATA HUB --------------------------
class DataHub:
    def __init__(self, X=None, df0=None, dom_score=None, share_desc=None, preds=None):
        self.X = X
        self.df0 = df0
        self.dom_score = dom_score
        self.share_desc = share_desc
        self.preds = preds

        # 1) Load preds / df0 / X from disk if not provided
        if self.preds is None:
            self.preds = _safe_load_csv("rankfirst_predictions_all_varieties_CALIBRATED.csv")
            if self.preds is not None:
                if "Variety" in self.preds.columns:
                    self.preds = self.preds.set_index("Variety")
                else:
                    first = self.preds.columns[0]
                    self.preds = self.preds.rename(columns={first: "Variety"}).set_index("Variety")

        if self.df0 is None:
            df0_csv = _safe_load_csv("merged_aroma_sensory.csv")
            if df0_csv is not None:
                self.df0 = df0_csv

        if self.X is None:
            cm = _safe_load_csv("compound_matrix.csv")
            if cm is None and os.path.exists("Variety_Dominance_Report.xlsx"):
                try:
                    cm = pd.read_excel("Variety_Dominance_Report.xlsx",
                                       sheet_name="Raw_Compound_Intensity", index_col=0)
                except Exception:
                    cm = None
            if cm is not None:
                self.X = compute_X_from_compound_matrix(cm)

        # 2) Compute overview metrics
        self.share = self.z = self.present = self.dominant = None
        if isinstance(self.X, pd.DataFrame) and self.X.shape[1] > 0:
            self.share, self.z, self.present, self.dominant, self.dom_score_calc = compute_overview_from_X(self.X)
            if self.dom_score is None:
                self.dom_score = self.dom_score_calc

        # 3) Panel flavor table
        self.flav_tbl = flavor_table_from_df0(self.df0)

        # 4) Descriptor recovery (auto-build share_desc if missing)
        if self.share_desc is None and isinstance(self.X, pd.DataFrame) and self.X.shape[1] > 0:
            def _pick_desc(colname: str):
                m = re.search(r'\(([^)]+)\)', str(colname))
                return m.group(1).strip() if m else None

            # Prefer a "pretty" matrix with parentheses in headers
            cm_pretty = None
            cm_csv = _safe_load_csv("compound_matrix.csv")
            if cm_csv is not None:
                cm_pretty = cm_csv.copy()
                if 'Variety' in cm_pretty.columns:
                    cm_pretty = cm_prety.set_index('Variety')  # <-- will be fixed below

            elif os.path.exists("Variety_Dominance_Report.xlsx"):
                try:
                    cm_pretty = pd.read_excel("Variety_Dominance_Report.xlsx",
                                              sheet_name="Raw_Compound_Intensity", index_col=0)
                except Exception:
                    cm_pretty = None

            # If not found, try safe->pretty mapping
            if cm_pretty is None:
                try:
                    name_map = pd.read_json("compound_name_map.json", typ="series").to_dict()
                except Exception:
                    name_map = None
                if name_map:
                    cm_pretty = self.X.copy()
                    cm_pretty.columns = [name_map.get(c, c) for c in cm_pretty.columns]
                else:
                    cm_pretty = self.X.copy()

            # FIX: typo and normalization
            if 'Variety' in getattr(cm_pretty, 'columns', []):
                cm_pretty = cm_pretty.set_index('Variety')

            cm_pretty = cm_pretty.apply(pd.to_numeric, errors='coerce').clip(lower=0)
            cm_pretty.index = cm_pretty.index.astype(str)

            comp2desc = {c: _pick_desc(c) for c in cm_pretty.columns}
            desc_names = sorted({d for d in comp2desc.values() if d and str(d).strip().lower() != 'nan'})

            if desc_names:
                row_sum_raw = cm_pretty.sum(axis=1).replace(0, np.nan)
                share_raw = cm_pretty.div(row_sum_raw, axis=0)

                target_idx = self.dom_score.index if isinstance(self.dom_score, pd.DataFrame) else share_raw.index
                share_raw = share_raw.reindex(target_idx)

                share_desc = pd.DataFrame(index=share_raw.index, columns=desc_names, dtype=float)
                for d in desc_names:
                    cols = [c for c in cm_pretty.columns if comp2desc.get(c) == d]
                    share_desc[d] = share_raw[cols].sum(axis=1) if cols else np.nan

                self.share_desc = share_desc

        # 5) Varieties list + demo fallback
        idxs = []
        for item in [self.dom_score, self.flav_tbl, self.X]:
            if isinstance(item, pd.DataFrame) and item.shape[0] > 0:
                idxs.append(set(item.index.astype(str)))
        self.varieties = sorted(set.union(*idxs)) if idxs else []

        if len(self.varieties) == 0:
            rng = np.random.default_rng(123)
            demo_var = [f"V{i:02d}" for i in range(1, 11)]
            demo_comp = [f"C{i}" for i in range(1, 16)]
            self.X = pd.DataFrame(rng.uniform(0, 100, (len(demo_var), len(demo_comp))),
                                  index=demo_var, columns=demo_comp)
            self.share, self.z, self.present, self.dominant, self.dom_score = compute_overview_from_X(self.X)
            self.flav_tbl = pd.DataFrame(rng.uniform(0, 40, (len(demo_var), len(FLAVOR_COLS))),
                                         index=demo_var, columns=FLAVOR_COLS)
            self.share_desc = None
            self.varieties = demo_var

    def dom_row(self, v):
        try:
            return self.dom_score.loc[v]
        except Exception:
            return pd.Series(dtype=float)

    def desc_row(self, v):
        if isinstance(self.share_desc, pd.DataFrame) and v in self.share_desc.index:
            return self.share_desc.loc[v]
        return pd.Series(dtype=float)

    def panel_row(self, v):
        if isinstance(self.flav_tbl, pd.DataFrame) and v in self.flav_tbl.index:
            return self.flav_tbl.loc[v]
        return pd.Series(dtype=float)

    def model_row(self, v):
        if self.preds is None or v not in self.preds.index:
            return pd.Series(dtype=float)
        cols = [c for c in self.preds.columns if c.endswith("__pred_cal")] or \
               [c for c in self.preds.columns if c.endswith("__pred")]
        if not cols:
            return pd.Series(dtype=float)
        s = self.preds.loc[v, cols].rename(lambda c: c.replace("__pred_cal", "").replace("__pred", ""))
        return s

# ------------------- DASH APP --------------------------
def build_app(hub: DataHub):
    app = Dash(__name__)
    app.title = "Potato Flavor Dashboard"

    _summary_df = _safe_load_csv("Variety_Dominance_Summary.csv")
    summary_data = _summary_df.to_dict("records") if isinstance(_summary_df, pd.DataFrame) else []
    summary_cols = [{"name": c, "id": c} for c in (_summary_df.columns if isinstance(_summary_df, pd.DataFrame) else [])]

    # Status banner so you know data actually loaded
    status_text = (
        f"Varieties: {len(hub.varieties)} | "
        f"Compounds: {hub.X.shape[1] if isinstance(hub.X, pd.DataFrame) else 0} | "
        f"Descriptors: {hub.share_desc.shape[1] if isinstance(hub.share_desc, pd.DataFrame) else 0} | "
        f"Radar source: panel{' + model' if isinstance(hub.preds, pd.DataFrame) else ''}"
    )

    app.layout = html.Div([
        html.H2("Potato Flavor & Aroma Dominance — Interactive Dashboard"),
        html.Div(status_text, style={"marginBottom":"8px","fontSize":"13px","opacity":0.8}),
        html.Div([
            html.Div([
                html.Label("Variety"),
                dcc.Dropdown(
                    id="variety",
                    options=[{"label": v, "value": v} for v in hub.varieties],
                    value=hub.varieties[0] if hub.varieties else None,
                    clearable=False
                ),
            ], style={"flex": "2", "minWidth": "220px", "marginRight": "12px"}),

            html.Div([
                html.Label("Radar source"),
                dcc.RadioItems(
                    id="radar_source",
                    options=[
                        {"label": "Panel (df0)", "value": "panel"},
                        {"label": "Model predictions", "value": "model"}
                    ],
                    value="panel",
                    inline=True
                )
            ], style={"flex": "2", "minWidth": "260px", "marginRight": "12px"}),

            html.Div([
                html.Label("Top-K compounds"),
                dcc.Slider(id="k_comp", min=3, max=10, step=1, value=5,
                           marks={i: str(i) for i in range(3, 11)})
            ], style={"flex": "3", "minWidth": "250px", "marginRight": "12px"}),

            html.Div([
                html.Label("Top-K descriptors"),
                dcc.Slider(id="k_desc", min=3, max=10, step=1, value=5,
                           marks={i: str(i) for i in range(3, 11)})
            ], style={"flex": "3", "minWidth": "250px"}),

        ], style={"display": "flex", "flexWrap": "wrap", "gap": "8px", "marginBottom": "16px"}),

        dcc.Tabs([
            dcc.Tab(label="Variety view", children=[
                html.Div([
                    html.Div([dcc.Graph(id="radar_plot")], style={"flex": "1", "minWidth": "350px"}),
                    html.Div([dcc.Graph(id="bar_compounds")], style={"flex": "1", "minWidth": "350px"}),
                    html.Div([dcc.Graph(id="bar_descriptors")], style={"flex": "1", "minWidth": "350px"}),
                ], style={"display": "flex", "flexWrap": "wrap", "gap": "10px"}),

                html.Div(id="variety_text", style={"marginTop": "6px", "fontSize": "14px"})
            ]),

            dcc.Tab(label="Overview", children=[
                html.Div([
                    html.Div([dcc.Graph(id="heat_share")], style={"flex": "1", "minWidth": "450px"}),
                    html.Div([dcc.Graph(id="heat_domrate")], style={"flex": "1", "minWidth": "450px"}),
                ], style={"display": "flex", "flexWrap": "wrap", "gap": "12px", "marginTop": "10px"})
            ]),

            dcc.Tab(label="Tables", children=[
                html.H4("Per-variety summary (if loaded)"),
                dash_table.DataTable(
                    id="summary_table",
                    data=summary_data,
                    columns=summary_cols,
                    page_size=12,
                    filter_action="native",
                    sort_action="native",
                    style_table={"overflowX": "auto"}
                )
            ])
        ])
    ], style={"maxWidth": "1400px", "margin": "auto", "padding": "12px"})

    @app.callback(
        Output("radar_plot", "figure"),
        Output("bar_compounds", "figure"),
        Output("bar_descriptors", "figure"),
        Output("variety_text", "children"),
        Input("variety", "value"), Input("radar_source", "value"),
        Input("k_comp", "value"), Input("k_desc", "value")
    )
    def update_variety(v, source, k_comp, k_desc):
        if v is None:
            return go.Figure(), go.Figure(), go.Figure(), "No variety selected."

        if source == "model":
            s = hub.model_row(v)
            fig_radar = make_radar(s, f"Flavor profile (model): {v}") if not s.empty \
                        else make_radar(pd.Series(dtype=float), f"Flavor profile (model): {v} (n/a)")
        else:
            s = hub.panel_row(v)
            fig_radar = make_radar(s, f"Flavor profile (panel): {v}") if not s.empty \
                        else make_radar(pd.Series(dtype=float), f"Flavor profile (panel): {v} (n/a)")

        comp_row = hub.dom_row(v)
        names, vals = topk_series(comp_row, int(k_comp))
        fig_comp = make_bar(names, vals, f"Top-{k_comp} compounds — {v}", "Dominance score")

        drow = hub.desc_row(v)
        if not drow.empty:
            dnames, dvals = topk_series(drow, int(k_desc))
            fig_desc = make_bar(dnames, dvals, f"Top-{k_desc} descriptors — {v}", "Descriptor share")
        else:
            fig_desc = make_bar([], [], "Top descriptors (n/a)")

        sshow = hub.model_row(v) if source == "model" else hub.panel_row(v)
        top_flavs = list(sshow.dropna().sort_values(ascending=False).index[:3]) if not sshow.empty else []
        txt = f"Top flavors ({'model' if source=='model' else 'panel'}): {', '.join(top_flavs) if top_flavs else '—'}"
        return fig_radar, fig_comp, fig_desc, txt

    @app.callback(
        Output("heat_share", "figure"),
        Output("heat_domrate", "figure"),
        Input("variety", "value")
    )
    def update_overview(_):
        if hub.share is None or hub.share.empty:
            return go.Figure(), go.Figure()

        mean_share = hub.share.mean(axis=0).sort_values(ascending=False)
        top_cols = list(mean_share.head(min(30, mean_share.shape[0])).index)
        share_top = hub.share.loc[hub.varieties, top_cols]
        heat1 = go.Figure(data=go.Heatmap(z=share_top.values, x=top_cols, y=share_top.index, coloraxis="coloraxis"))
        heat1.update_layout(title="Share heatmap — Top 30 compounds (mean share)",
                            coloraxis={"colorscale": "Viridis"}, margin=dict(l=40, r=10, t=60, b=40))

        if hub.dominant is None or hub.dominant.empty:
            return heat1, go.Figure()

        dom_rate = hub.dominant.mean(axis=0).sort_values(ascending=False)
        dcols = list(dom_rate.head(min(30, dom_rate.shape[0])).index)
        dom_mask_top = hub.dominant.loc[hub.varieties, dcols].astype(float)
        heat2 = go.Figure(data=go.Heatmap(z=dom_mask_top.values, x=dcols, y=dom_mask_top.index, coloraxis="coloraxis"))
        heat2.update_layout(title="Dominance mask — Top 30 compounds (dominance rate)",
                            coloraxis={"colorscale": "Viridis"}, margin=dict(l=40, r=10, t=60, b=40))
        return heat1, heat2

    return app

# --------------- ENTRY POINTS ----------------
def launch(X=None, df0=None, dom_score=None, share_desc=None, preds=None,
           host="127.0.0.1", port=8050, debug=True):
    hub = DataHub(X=X, df0=df0, dom_score=dom_score, share_desc=share_desc, preds=preds)
    app = build_app(hub)
    print(f"Starting dashboard on http://{host}:{port}  (Ctrl+C to stop)")
    app.run(debug=debug, host=host, port=port)

if __name__ == "__main__":
    launch()
