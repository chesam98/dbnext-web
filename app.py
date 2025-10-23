# app.py — dBNext Web (v7.2) — version interactive (sélection + seuil + undo/redo)
import base64
from io import StringIO
from datetime import datetime
from typing import List, Dict

import dash
from dash import Dash, dcc, html, Input, Output, State, ctx
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from dbnext_core import (
    lire_fichier_dBNext_from_text, build_intervals, concat_series,
    LAeq, LN, rolling_LAeq, total_duration, build_workbook
)

app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# ---------- Helpers ----------
def parse_contents(contents)->pd.DataFrame:
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    text = decoded.decode('utf-8', errors='ignore')
    return lire_fichier_dBNext_from_text(text)

def default_situations(t0: pd.Timestamp, t1: pd.Timestamp):
    days = ["lundi","mardi","mercredi","jeudi","vendredi","samedi","dimanche"]
    return {
        "Diurne": build_intervals(t0, t1, 7, 22, days),
        "Nocturne": build_intervals(t0, t1, 22, 7, days),
    }

def make_editor_figure(df: pd.DataFrame, excluded_idx: List[pd.Timestamp], threshold_y: float|None):
    """Retourne une figure avec points valides (bleu) / exclus (rouge) + seuil (ligne pointillée)."""
    fig = go.Figure()
    if df is None or df.empty:
        fig.add_annotation(text="Aucune donnée", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(margin=dict(l=40,r=20,t=40,b=40))
        return fig

    excluded_mask = df.index.isin(pd.to_datetime(excluded_idx)) if excluded_idx else np.zeros(len(df), bool)
    valid = df.loc[~excluded_mask, "LAeq"]
    excl  = df.loc[ excluded_mask, "LAeq"]

    # courbe fine de fond pour la tendance
    fig.add_trace(go.Scatter(
        x=df.index, y=df["LAeq"], mode="lines", name="Mesures",
        line=dict(width=1), opacity=0.25, hoverinfo="skip", showlegend=False
    ))

    # points valides (bleu)
    if not valid.empty:
        fig.add_trace(go.Scatter(
            x=valid.index, y=valid.values, mode="markers", name="Valide",
            marker=dict(size=5), hovertemplate="%{x|%d/%m/%Y %H:%M:%S} | %{y:.1f} dB(A)"
        ))
    # points exclus (rouge)
    if not excl.empty:
        fig.add_trace(go.Scatter(
            x=excl.index, y=excl.values, mode="markers", name="Perturbations",
            marker=dict(size=6, color="red"),
            hovertemplate="%{x|%d/%m/%Y %H:%M:%S} | %{y:.1f} dB(A)"
        ))

    # lignes verticales de minuit
    if not df.empty:
        start = pd.Timestamp(df.index.min()).floor("D")
        end = pd.Timestamp(df.index.max()).ceil("D")
        shapes = []
        j = start
        while j <= end:
            shapes.append(dict(
                type="line", x0=j, x1=j, y0=0, y1=1, xref="x", yref="paper",
                line=dict(color="rgba(0,0,0,0.35)", dash="dash", width=1)
            ))
            j += pd.Timedelta(days=1)
        # seuil horizontal si actif
        if threshold_y is not None:
            shapes.append(dict(
                type="line", x0=df.index.min(), x1=df.index.max(),
                y0=threshold_y, y1=threshold_y, xref="x", yref="y",
                line=dict(color="rgba(200,0,0,0.8)", dash="dash")
            ))
        fig.update_layout(shapes=shapes)

    fig.update_layout(
        hovermode="x",
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(orientation="h", x=0, y=1.12),
        xaxis=dict(title="Date / Heure", tickformat="%d/%m\n%H:%M", showspikes=True, spikemode="across", spikesnap="cursor"),
        yaxis=dict(title="LAeq (dB(A))", dtick=5, showspikes=True, spikemode="across", spikesnap="cursor"),
    )
    return fig

# ---------- Layout ----------
app.layout = html.Div([
    html.H1("dBNext — Analyse acoustique (v7.2) — Web"),
    html.Div("Compatible fichiers dBNext Cleaner / tout CSV horodaté", className="small"),

    dcc.Tabs([
        dcc.Tab(label="1) Import & Paramétrage", children=[
            html.Div(className="card", children=[
                html.H3("Importer vos points (CSV)"),
                dcc.Upload(
                    id="uploader", multiple=True,
                    children=html.Div(["Glissez-déposez ou ", html.A("sélectionnez des fichiers CSV")]),
                    style={"width":"100%","height":"120px","lineHeight":"120px","borderWidth":"2px",
                           "borderStyle":"dashed","borderRadius":"12px","textAlign":"center"}
                ),
                html.Div(id="upload-status", style={"marginTop":"10px"}),
            ]),
            html.Div(className="card", children=[
                html.H3("Points détectés"),
                html.Div(id="points-config"),
            ]),
        ]),
        dcc.Tab(label="2) Nettoyage interactif", children=[
            html.Div(className="card", children=[
                html.Div([
                    html.Div([html.Label("Point"), dcc.Dropdown(id="sel-point")],
                             style={"flex":"1","marginRight":"8px"}),
                    html.Div([html.Label("Type"), dcc.Dropdown(id="sel-type",
                                                               options=[{"label":x,"value":x} for x in ["LP","ZER","HYBRIDE"]])],
                             style={"width":"200px","marginRight":"8px"}),
                    html.Div([html.Label("Situation"), dcc.Dropdown(id="sel-situation")],
                             style={"flex":"1"}),
                ], style={"display":"flex"}),

                html.Div([
                    html.Button("🧹 Gomme (sélection rect./lasso)", id="btn-erase", n_clicks=0),
                    html.Button("🔺 Placer seuil (clic)", id="btn-threshold-mode", n_clicks=0, style={"marginLeft":"8px"}),
                    dcc.Input(id="input-threshold", type="number", placeholder="Seuil dB(A)", style={"width":"140px","marginLeft":"6px"}),
                    html.Button("↩️ Undo", id="btn-undo", n_clicks=0, style={"marginLeft":"8px"}),
                    html.Button("↪️ Redo", id="btn-redo", n_clicks=0, style={"marginLeft":"6px"}),
                    html.Button("🔄 Reset", id="btn-reset", n_clicks=0, style={"marginLeft":"6px"}),
                    html.Span(id="excluded-count", style={"marginLeft":"12px", "fontWeight":"600"}),
                ], style={"marginTop":"10px","marginBottom":"6px"}),

                dcc.Graph(
                    id="editor-graph", clear_on_unhover=True,
                    config={"displaylogo": False, "modeBarButtonsToAdd": ["select2d","lasso2d"], "scrollZoom": True}
                ),
                html.Div("Astuce : utilisez l'outil 'Select' ou 'Lasso' pour dessiner sur le graphe. "
                         "Clique sur le graphe après « Placer seuil » pour exclure tous les points ≥ seuil.",
                         className="small")
            ]),
        ]),
        dcc.Tab(label="3) Export Excel", children=[
            html.Div(className="card", children=[
                html.P("Quand tout est prêt, exportez l'Excel complet (feuille par point)."),
                html.Button("📦 Générer l'Excel", id="btn-export", n_clicks=0),
                dcc.Download(id="dl-excel"),
                html.Div(id="export-status", style={"marginTop":"8px"})
            ])
        ])
    ]),

    # --- Stores (état global) ---
    dcc.Store(id="points-store"),    # {name: {type, df_json, situations, cleaned{sit:[iso]}, ...}}
    dcc.Store(id="editor-state"),    # {"threshold_mode":bool, "threshold_y":float|None, "undo":{(pt,sit):[...past]}, "redo":{...}}
])

# ---------- Upload & config ----------
@app.callback(
    Output("points-store","data"),
    Output("upload-status","children"),
    Input("uploader","contents"),
    State("uploader","filename"),
    prevent_initial_call=True
)
def on_upload(list_contents, filenames):
    pts = {}
    msgs=[]
    if not list_contents:
        return dash.no_update, ""
    for contents, name in zip(list_contents, filenames):
        try:
            df = parse_contents(contents)
            t0, t1 = df.index.min(), df.index.max()
            sit = default_situations(t0, t1)
            pts[name] = {
                "name": name,
                "type": "HYBRIDE",
                "df_json": df.reset_index().to_json(date_format="iso"),
                "t0": str(t0), "t1": str(t1),
                "situations": {k: [(str(s), str(e)) for s,e in v] for k,v in sit.items()},
                "cleaned": {k: [] for k in sit.keys()}
            }
            msgs.append(html.Div(f"✅ {name}: {len(df)} points | {t0} → {t1}"))
        except Exception as e:
            msgs.append(html.Div(f"❌ {name}: {e}"))
    return pts, msgs

@app.callback(
    Output("points-config","children"),
    Input("points-store","data"),
)
def render_points_config(data):
    if not data:
        return html.Div("Aucun point importé pour l'instant.")
    rows=[]
    for name, meta in data.items():
        rows.append(html.Div(className="card", children=[
            html.H4(name),
            html.Div([
                html.Div([html.Label("Type"),
                          dcc.Dropdown(id={"type":"point-type","name":name},
                                       options=[{"label":x,"value":x} for x in ["LP","ZER","HYBRIDE"]],
                                       value=meta.get("type","HYBRIDE"),
                                       style={"width":"200px"})],
                         style={"marginRight":"16px"}),
                html.Div([html.Label("Situations (Jour/Nuit par défaut)"),
                          html.Div(f"Période : {meta['t0']} → {meta['t1']}", className="small")]),
            ], style={"display":"flex","alignItems":"center"}),
        ]))
    return rows

@app.callback(
    Output("points-store","data", allow_duplicate=True),
    Input({"type":"point-type","name":dash.ALL},"value"),
    State("points-store","data"),
    prevent_initial_call=True
)
def update_point_types(types, data):
    if not data:
        return dash.no_update
    names = list(data.keys())
    for name, t in zip(names, types):
        data[name]["type"] = t or "HYBRIDE"
    return data

# ---------- Dropdowns ----------
@app.callback(
    Output("sel-point","options"),
    Output("sel-point","value"),
    Input("points-store","data")
)
def populate_points(data):
    if not data:
        return [], None
    opts=[{"label":k, "value":k} for k in data.keys()]
    return opts, opts[0]["value"]

@app.callback(
    Output("sel-type","value"),
    Output("sel-situation","options"),
    Output("sel-situation","value"),
    Input("sel-point","value"),
    State("points-store","data"),
)
def populate_type_and_sit(point, data):
    if not point or not data:
        return None, [], None
    meta = data[point]
    situations = list(meta["situations"].keys())
    sit_opts = [{"label":s, "value":s} for s in situations]
    return meta.get("type","HYBRIDE"), sit_opts, (situations[0] if situations else None)

# ---------- Graph render ----------
@app.callback(
    Output("editor-graph","figure"),
    Output("editor-state","data"),
    Output("excluded-count","children"),
    Input("sel-point","value"),
    Input("sel-situation","value"),
    Input("points-store","data"),
    State("editor-state","data"),
)
def update_graph(point, situation, data, est):
    if not point or not situation or not data:
        return go.Figure(), est, ""
    meta = data[point]
    df = pd.read_json(StringIO(meta["df_json"]))
    df["time_block"] = pd.to_datetime(df["time_block"])
    df = df.rename(columns={"time_block":"time"}).set_index("time").sort_index()

    excluded_idx = [pd.to_datetime(x) for x in meta["cleaned"].get(situation, [])]
    threshold_y = (est or {}).get("threshold_y") if est else None
    fig = make_editor_figure(df, excluded_idx, threshold_y)

    # init/editor state per (point,sit)
    est = est or {"threshold_mode": False, "threshold_y": None, "undo": {}, "redo": {}}
    key = f"{point}||{situation}"
    est["undo"].setdefault(key, [])
    est["redo"].setdefault(key, [])
    return fig, est, f"{len(excluded_idx)} point(s) exclu(s)"

# ---------- Gomme via sélection rectangle/lasso ----------
@app.callback(
    Output("points-store","data", allow_duplicate=True),
    Output("editor-state","data", allow_duplicate=True),
    Input("btn-erase","n_clicks"),
    State("editor-graph","selectedData"),
    State("sel-point","value"),
    State("sel-situation","value"),
    State("points-store","data"),
    State("editor-state","data"),
    prevent_initial_call=True
)
def on_erase(n, selected, point, situation, data, est):
    if not n or not selected or not data or not point:
        return dash.no_update, dash.no_update
    meta = data[point]
    # les x sont des timestamps ISO dans selectedData["points"]
    xvals = [p["x"] for p in selected["points"]]
    # historique
    est = est or {"threshold_mode": False, "threshold_y": None, "undo": {}, "redo": {}}
    key = f"{point}||{situation}"
    old = list(meta["cleaned"].get(situation, []))
    est["undo"].setdefault(key, []).append(old)
    est["redo"][key] = []

    new_set = set(old)
    for x in xvals:
        try:
            new_set.add(pd.to_datetime(x).isoformat())
        except Exception:
            pass
    data[point]["cleaned"][situation] = sorted(list(new_set))
    return data, est

# ---------- Toggle mode seuil ----------
@app.callback(
    Output("editor-state","data", allow_duplicate=True),
    Input("btn-threshold-mode","n_clicks"),
    State("editor-state","data"),
    prevent_initial_call=True
)
def toggle_threshold(n, est):
    est = est or {"threshold_mode": False, "threshold_y": None, "undo": {}, "redo": {}}
    est["threshold_mode"] = not est.get("threshold_mode", False)
    return est

# ---------- Clic pour seuil : ajoute tous les points >= seuil ----------
@app.callback(
    Output("points-store","data", allow_duplicate=True),
    Output("editor-state","data", allow_duplicate=True),
    Input("editor-graph","clickData"),
    State("input-threshold","value"),
    State("editor-state","data"),
    State("sel-point","value"),
    State("sel-situation","value"),
    State("points-store","data"),
    prevent_initial_call=True
)
def on_threshold_click(click, manual_thr, est, point, situation, data):
    if not click or not est or not est.get("threshold_mode"):
        return dash.no_update, dash.no_update
    y_click = click["points"][0]["y"]
    thr = float(manual_thr) if manual_thr is not None else float(y_click)

    # df global
    meta = data[point]
    df = pd.read_json(StringIO(meta["df_json"]))
    df["time_block"] = pd.to_datetime(df["time_block"])
    df = df.rename(columns={"time_block":"time"}).set_index("time").sort_index()

    # sous-série de la situation sélectionnée
    sits = [(pd.to_datetime(s), pd.to_datetime(e)) for s,e in meta["situations"][situation]]
    segs = [df.loc[(df.index>=s)&(df.index<=e), "LAeq"] for s,e in sits]
    s = pd.concat(segs).sort_index() if segs else pd.Series(dtype=float)

    cand = s[s >= thr]
    # historique
    key = f"{point}||{situation}"
    old = list(meta["cleaned"].get(situation, []))
    est["undo"].setdefault(key, []).append(old)
    est["redo"][key] = []

    new_set = set(old) | set([t.isoformat() for t in cand.index])
    data[point]["cleaned"][situation] = sorted(list(new_set))
    # afficher la ligne seuil tant que le mode est actif
    est["threshold_mode"] = True
    est["threshold_y"] = thr
    return data, est

# ---------- Undo / Redo / Reset ----------
@app.callback(
    Output("points-store","data", allow_duplicate=True),
    Output("editor-state","data", allow_duplicate=True),
    Input("btn-undo","n_clicks"),
    Input("btn-redo","n_clicks"),
    Input("btn-reset","n_clicks"),
    State("sel-point","value"),
    State("sel-situation","value"),
    State("points-store","data"),
    State("editor-state","data"),
    prevent_initial_call=True
)
def on_undo_redo_reset(n_undo, n_redo, n_reset, point, situation, data, est):
    if not data or not point or not situation:
        return dash.no_update, dash.no_update
    est = est or {"threshold_mode": False, "threshold_y": None, "undo": {}, "redo": {}}
    key = f"{point}||{situation}"
    est["undo"].setdefault(key, [])
    est["redo"].setdefault(key, [])
    current = list(data[point]["cleaned"].get(situation, []))

    trig = ctx.triggered_id
    if trig == "btn-undo" and est["undo"][key]:
        prev = est["undo"][key].pop()
        est["redo"][key].append(current)
        data[point]["cleaned"][situation] = prev
    elif trig == "btn-redo" and est["redo"][key]:
        nxt = est["redo"][key].pop()
        est["undo"][key].append(current)
        data[point]["cleaned"][situation] = nxt
    elif trig == "btn-reset":
        est["undo"][key].append(current)
        est["redo"][key] = []
        data[point]["cleaned"][situation] = []
        est["threshold_y"] = None
    else:
        return dash.no_update, dash.no_update
    return data, est

# ---------- Export ----------
@app.callback(
    Output("dl-excel","data"),
    Output("export-status","children"),
    Input("btn-export","n_clicks"),
    State("points-store","data"),
    prevent_initial_call=True
)
def do_export(n, data):
    if not n or not data:
        return dash.no_update, ""
    points = []
    for name, meta in data.items():
        df = pd.read_json(StringIO(meta["df_json"]))
        df["time_block"] = pd.to_datetime(df["time_block"])
        df = df.rename(columns={"time_block":"time"}).set_index("time").sort_index()
        situ = {k: [(pd.to_datetime(s), pd.to_datetime(e)) for s,e in v] for k,v in meta["situations"].items()}
        points.append({
            "name": name,
            "type": meta.get("type", "HYBRIDE"),
            "df": df,
            "situations": situ,
            "cleaned_masks": {k: [pd.to_datetime(x) for x in meta["cleaned"].get(k, [])] for k in meta["cleaned"].keys()}
        })
    xlsx = build_workbook(points)
    return dcc.send_bytes(xlsx, filename="Analyse_Acoustique_dBNext_v7_2.xlsx"), html.Div("✅ Export prêt")

if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8050, debug=True)
