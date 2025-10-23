\
import base64, io, json, datetime as dt
from datetime import datetime, timedelta
from typing import List, Dict

import dash
from dash import Dash, callback, dcc, html, Input, Output, State, ctx, dash_table, ALL, MATCH, Patch
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from dbnext_core import (
    lire_fichier_dBNext_from_text, build_intervals, concat_series,
    LAeq, LN, rolling_LAeq, total_duration,
    build_workbook
)

app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# ------------------ Helpers ------------------
def parse_contents(contents)->pd.DataFrame:
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    text = decoded.decode('utf-8', errors='ignore')
    return lire_fichier_dBNext_from_text(text)

def human_date(ts: pd.Timestamp)->str:
    return ts.strftime("%d/%m/%Y %H:%M:%S")

def default_situations(t0: pd.Timestamp, t1: pd.Timestamp):
    days = ["lundi","mardi","mercredi","jeudi","vendredi","samedi","dimanche"]
    return {
        "Diurne": build_intervals(t0, t1, 7, 22, days),
        "Nocturne": build_intervals(t0, t1, 22, 7, days),
    }

def fig_for_series(df: pd.DataFrame, removed_index: List[pd.Timestamp]):
    fig = go.Figure()
    if df is None or df.empty:
        fig.add_annotation(text="Aucune donnée", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

    # Main line
    fig.add_trace(go.Scatter(
        x=df.index, y=df["LAeq"], mode="lines", name="Mesures",
        line=dict(width=1.2)
    ))

    # Removed points
    if removed_index:
        removed = df.loc[df.index.isin(removed_index), "LAeq"]
        fig.add_trace(go.Scatter(
            x=removed.index, y=removed.values, mode="markers", name="Perturbations",
            marker=dict(size=6), marker_color="red"
        ))

    # Spikelines (crosshair-like)
    fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor", showline=True)
    fig.update_yaxes(showspikes=True, spikemode="across", spikesnap="cursor", zeroline=False)

    # Midnight vertical lines
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
        fig.update_layout(shapes=shapes)

    fig.update_layout(
        hovermode="x",
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(orientation="h", x=0, y=1.1),
        xaxis=dict(title="Date / Heure", tickformat="%d/%m\n%H:%M"),
        yaxis=dict(title="LAeq (dB(A))", dtick=5)
    )
    # Hover template
    fig.update_traces(hovertemplate="%{x|%d/%m/%Y %H:%M:%S} | %{y:.1f} dB(A)")
    return fig

# ------------------ Layout ------------------
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
                    style={
                        "width":"100%","height":"120px","lineHeight":"120px",
                        "borderWidth":"2px","borderStyle":"dashed","borderRadius":"12px","textAlign":"center"
                    }
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
                    html.Div([
                        html.Label("Point"), dcc.Dropdown(id="sel-point", placeholder="Choisir un point"),
                    ], style={"flex":"1","marginRight":"8px"}),
                    html.Div([
                        html.Label("Type"), dcc.Dropdown(id="sel-type", options=[
                            {"label":"LP","value":"LP"},{"label":"ZER","value":"ZER"},{"label":"HYBRIDE","value":"HYBRIDE"}
                        ]),
                    ], style={"width":"200px","marginRight":"8px"}),
                    html.Div([
                        html.Label("Situation"), dcc.Dropdown(id="sel-situation", placeholder="Diurne / Nocturne / ..."),
                    ], style={"flex":"1"}),
                ], style={"display":"flex"}),

                html.Div([
                    html.Button("🧹 Gomme (sélection rect.)", id="btn-erase", n_clicks=0),
                    html.Button("🔺 Placer seuil (depuis un clic)", id="btn-threshold-mode", n_clicks=0, style={"marginLeft":"8px"}),
                    dcc.Input(id="input-threshold", type="number", placeholder="Seuil dB(A)", style={"width":"140px","marginLeft":"6px"}),
                    html.Button("↩️ Undo", id="btn-undo", n_clicks=0, style={"marginLeft":"8px"}),
                    html.Button("↪️ Redo", id="btn-redo", n_clicks=0, style={"marginLeft":"6px"}),
                    html.Button("🔄 Reset", id="btn-reset", n_clicks=0, style={"marginLeft":"6px"}),
                ], style={"marginTop":"10px","marginBottom":"6px"}),

                dcc.Graph(id="editor-graph", clear_on_unhover=True, config={
                    "displaylogo": False,
                    "modeBarButtonsToAdd": ["select2d", "lasso2d"],
                    "scrollZoom": True,
                }),
                html.Div("Astuce : utilisez l'outil 'Select' de la barre du graphique pour dessiner un rectangle. Le seuil peut être défini en cliquant sur le graphe après avoir appuyé sur « Placer seuil ».",
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

    # -------------- State --------------
    dcc.Store(id="points-store"),   # {name: {type, df_json, t0, t1, situations{...}, cleaned{sit: [timestamps]}}}
    dcc.Store(id="editor-state"),   # {"threshold_mode": bool, "undo": [...], "redo":[...]}
])

# ------------------ Points upload & config ------------------
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
                html.Div([
                    html.Label("Situations (Jour/Nuit par défaut)"),
                    html.Div(f"Période : {meta['t0']} → {meta['t1']}", className="small")
                ]),
            ], style={"display":"flex","alignItems":"center"}),
        ]))
    return rows

@app.callback(
    Output("points-store","data", allow_duplicate=True),
    Input({"type":"point-type","name":ALL},"value"),
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

# ------------------ Editor dropdowns ------------------
@app.callback(
    Output("sel-point","options"),
    Output("sel-point","value"),
    Input("points-store","data")
)
def populate_points(data):
    if not data:
        return [], None
    opts=[{"label":k, "value":k} for k in data.keys()]
    first=opts[0]["value"]
    return opts, first

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
    default_sit = situations[0] if situations else None
    return meta.get("type","HYBRIDE"), sit_opts, default_sit

# ------------------ Graph + interactions ------------------
@app.callback(
    Output("editor-graph","figure"),
    Output("editor-state","data"),
    Input("sel-point","value"),
    Input("sel-situation","value"),
    Input("points-store","data"),
    State("editor-state","data"),
)
def update_graph(point, situation, data, est):
    if not point or not situation or not data:
        return go.Figure(), est
    meta = data[point]
    df = pd.read_json(meta["df_json"])
    df["time_block"] = pd.to_datetime(df["time_block"])
    df = df.rename(columns={"time_block":"time"}).set_index("time").sort_index()

    removed_idx = [pd.to_datetime(x) for x in meta["cleaned"].get(situation, [])]
    fig = fig_for_series(df, removed_idx)

    # init editor state
    if not est:
        est = {"threshold_mode": False, "undo": [], "redo": []}
    return fig, est

# handle selection erase
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
    if not n: return dash.no_update, dash.no_update
    if not selected or not data or not point:
        return dash.no_update, dash.no_update
    xvals = [p["x"] for p in selected["points"]]
    # push history
    old = list(data[point]["cleaned"].get(situation, []))
    est = est or {"threshold_mode": False, "undo": [], "redo": []}
    est["undo"].append(old)
    est["redo"] = []

    new_set = set(old)
    for x in xvals:
        try:
            new_set.add(pd.to_datetime(x).isoformat())
        except Exception:
            pass
    data[point]["cleaned"][situation] = sorted(list(new_set))
    return data, est

# toggle threshold mode
@app.callback(
    Output("editor-state","data", allow_duplicate=True),
    Input("btn-threshold-mode","n_clicks"),
    State("editor-state","data"),
    prevent_initial_call=True
)
def toggle_threshold(n, est):
    if est is None: est={"threshold_mode": False, "undo": [], "redo": []}
    est["threshold_mode"] = not est.get("threshold_mode", False)
    return est

# when graph clicked in threshold mode, add all points >= threshold
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
    thr = manual_thr if manual_thr is not None else y_click
    # obtain df
    meta = data[point]
    df = pd.read_json(meta["df_json"])
    df["time_block"] = pd.to_datetime(df["time_block"])
    df = df.rename(columns={"time_block":"time"}).set_index("time").sort_index()
    # get situation subset
    sits = [(pd.to_datetime(s), pd.to_datetime(e)) for s,e in meta["situations"][situation]]
    segs = [df.loc[(df.index>=s)&(df.index<=e), "LAeq"] for s,e in sits]
    s = pd.concat(segs).sort_index() if segs else pd.Series(dtype=float)

    cand = s[s >= thr]
    # history
    old = list(meta["cleaned"].get(situation, []))
    est["undo"].append(old)
    est["redo"] = []

    new_set = set(old) | set([t.isoformat() for t in cand.index])
    data[point]["cleaned"][situation] = sorted(list(new_set))
    # disable threshold mode after placement
    est["threshold_mode"] = False
    return data, est

# Undo/Redo/Reset
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
    trig = ctx.triggered_id
    if not data or not point or not situation:
        return dash.no_update, dash.no_update
    est = est or {"threshold_mode": False, "undo": [], "redo": []}
    current = list(data[point]["cleaned"].get(situation, []))

    if trig == "btn-undo" and est["undo"]:
        prev = est["undo"].pop()
        est["redo"].append(current)
        data[point]["cleaned"][situation] = prev
    elif trig == "btn-redo" and est["redo"]:
        nxt = est["redo"].pop()
        est["undo"].append(current)
        data[point]["cleaned"][situation] = nxt
    elif trig == "btn-reset":
        est["undo"].append(current)
        est["redo"] = []
        data[point]["cleaned"][situation] = []
    else:
        return dash.no_update, dash.no_update
    return data, est

# ------------------ Export ------------------
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
        df = pd.read_json(meta["df_json"])
        df["time_block"] = pd.to_datetime(df["time_block"])
        df = df.rename(columns={"time_block":"time"}).set_index("time").sort_index()
        # situations
        situ = {k: [(pd.to_datetime(s), pd.to_datetime(e)) for s,e in v] for k,v in meta["situations"].items()}
        points.append({
            "name": name,
            "type": meta.get("type", "HYBRIDE"),
            "df": df,
            "situations": situ,
            "cleaned_masks": {k: [pd.to_datetime(x) for x in meta["cleaned"].get(k, [])] for k in meta["cleaned"].keys()}
        })
    xlsx = build_workbook(points)
    fname = f"Analyse_Acoustique_dBNext_v7_2.xlsx"
    return dcc.send_bytes(xlsx, filename=fname), html.Div("✅ Export prêt")

if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8050, debug=True)
