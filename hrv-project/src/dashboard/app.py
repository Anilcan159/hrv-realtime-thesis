# src/dashboard/app.py
import os
import sys
from typing import Any, Dict

import numpy as np
import pandas as pd                     # <<< YENİ
from pathlib import Path               # <<< YENİ
import requests
from dash import Dash, html, dcc, Input, Output
import plotly.graph_objects as go

# ----------------- PROJE KÖKÜ ----------------- #

CURRENT_DIR = os.path.dirname(__file__)                    # .../src/dashboard
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))  # .../hrv-project
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.config.settings import settings
from src.hrv_metrics.service_hrv import (
    get_time_domain_metrics,
    get_available_subject_codes,
    get_hr_timeseries,
    get_poincare_data,
    get_subject_info,
    get_signal_status,
    get_vmdon_components,
)


# ----------------- SPARK AVMD ÇIKTISI ----------------- #

SPARK_PARQUET_PATH = (
    Path(PROJECT_ROOT) / "data" / "processed" / "hrv_bands_avmd.parquet"
)

try:
    if SPARK_PARQUET_PATH.exists():
        df_avmd_spark = pd.read_parquet(SPARK_PARQUET_PATH)
        print(f"[dashboard] Loaded Spark AVMD parquet: {SPARK_PARQUET_PATH}")
    else:
        print(f"[dashboard] Spark parquet not found: {SPARK_PARQUET_PATH}")
        df_avmd_spark = pd.DataFrame()
except Exception as e:
    print(f"[dashboard] Failed to load Spark parquet: {e}")
    df_avmd_spark = pd.DataFrame()


# ----------------- RENK PALETİ (PASTEL) ----------------- #

PALETTE: Dict[str, str] = {
    "bg": "#F5F7FB",
    "bg_soft": "#EDF2FF",
    "card": "#FFFFFF",
    "card_alt": "#FDFEFF",
    "border": "#E2E8F0",
    "primary": "#4C6FFF",
    "primary_soft": "#E3E8FF",
    "accent": "#FF8BA7",
    "accent_soft": "#FFE5EC",
    "text": "#1F2933",
    "muted": "#718096",
    "muted_soft": "#A0AEC0",
    "good": "#38A169",
    "warning": "#F6AD55",
}

PANEL_STYLE = {
    "backgroundColor": PALETTE["card"],
    "border": f"1px solid {PALETTE['border']}",
    "borderRadius": "18px",
    "padding": "18px",
    "display": "flex",
    "flexDirection": "column",
    "gap": "12px",
    "boxShadow": "0 10px 30px rgba(15, 23, 42, 0.06)",
}

METRIC_CARD_STYLE = {
    "backgroundColor": PALETTE["card_alt"],
    "borderRadius": "14px",
    "padding": "14px 16px",
    "display": "flex",
    "flexDirection": "column",
    "gap": "6px",
    "minHeight": "70px",
    "border": f"1px solid {PALETTE['border']}",
    "alignItems": "center",
    "justifyContent": "center",
    "textAlign": "center",
}


# ----------------- YARDIMCI FONKSİYONLAR ----------------- #

def _fetch_from_api(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    FastAPI HRV servisinden JSON veri çeker.
    Her türlü hata durumunda boş dict döner (dashboard çökmesin).
    """
    base_url = settings.api.base_url.rstrip("/")
    url = f"{base_url}{path}"

    try:
        resp = requests.get(url, params=params, timeout=1.0)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"[dashboard] API request failed: {e} (url={url}, params={params})")
        return {}


DEFAULT_WINDOW_S = settings.dashboard.default_window_s  # 5 dakikalık pencere


def _fmt(v: Any, nd: int = 1) -> str:
    """Metrik sayısını güvenli şekilde string'e çevirir."""
    try:
        if v is None:
            return "N/A"
        if isinstance(v, float) and np.isnan(v):
            return "N/A"
        return f"{float(v):.{nd}f}"
    except Exception:
        return "N/A"


def metric_card(title: str, value: str, unit: str = "") -> html.Div:
    return html.Div(
        style=METRIC_CARD_STYLE,
        children=[
            html.Span(
                title,
                style={
                    "fontSize": "13px",
                    "fontWeight": "600",
                    "color": PALETTE["primary"],
                    "letterSpacing": "0.06em",
                    "textTransform": "uppercase",
                    "textAlign": "center",
                },
            ),
            html.Div(
                style={
                    "display": "flex",
                    "alignItems": "baseline",
                    "justifyContent": "center",
                    "gap": "4px",
                },
                children=[
                    html.Span(
                        value,
                        style={
                            "fontSize": "24px",
                            "fontWeight": "600",
                            "color": PALETTE["text"],
                        },
                    ),
                    html.Span(
                        unit,
                        style={
                            "fontSize": "11px",
                            "color": PALETTE["muted_soft"],
                        },
                    ),
                ],
            ),
        ],
    )


def _empty_figure(title: str, message: str) -> go.Figure:
    """Boş durumlar için sade bir figure."""
    fig = go.Figure()
    fig.update_layout(
        title=title,
        annotations=[
            dict(
                text=message,
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(color=PALETTE["muted"]),
            )
        ],
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


# ----------------- DASH APP & LAYOUT ----------------- #

app = Dash(__name__)
subject_codes = get_available_subject_codes()

app.layout = html.Div(
    style={
        "backgroundColor": PALETTE["bg"],
        "minHeight": "100vh",
        "padding": "22px 26px",
        "color": PALETTE["text"],
        "fontFamily": "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
    },
    children=[
        dcc.Interval(id="refresh", interval=1000, n_intervals=0),

        # ---------- HEADER ----------
        html.Div(
            style={
                "display": "flex",
                "justifyContent": "space-between",
                "alignItems": "center",
                "marginBottom": "24px",
            },
            children=[
                # SOL: Başlık + ikon + tagline
                html.Div(
                    children=[
                        # Başlık + PNG aynı satırda
                        html.Div(
                            style={
                                "display": "flex",
                                "alignItems": "center",
                                "gap": "10px",
                            },
                            children=[
                                html.H1(
                                    "HRV Live Dashboard",
                                    style={
                                        "margin": 0,
                                        "fontSize": "75px",
                                        "fontWeight": "700",
                                        "letterSpacing": "0.02em",
                                    },
                                ),
                                html.Img(
                                    src="/assets/hrv_wave.png",
                                    style={
                                        "height": "75px",
                                        "display": "block",
                                    },
                                ),
                            ],
                        ),
                        # Alt satır: tagline
                        html.Span(
                            "Real-time heart rate variability monitoring",
                            style={
                                "fontSize": "20px",
                                "color": PALETTE["muted"],
                            },
                        ),
                    ],
                ),

                # SAĞ: Subject figürü + dropdown / method card
                html.Div(
                    style={
                        "display": "flex",
                        "alignItems": "center",
                        "gap": "18px",
                    },
                    children=[
                        # Sol: subject figürü
                        html.Img(
                            id="subject-figure",
                            src=app.get_asset_url("Man.png"),
                            style={
                                "height": "160px",
                                "width": "auto",
                                "display": "block",
                            },
                        ),

                        # Sağ: subject + method kartı
                        html.Div(
                            style={
                                "minWidth": "360px",
                                "backgroundColor": PALETTE["card"],
                                "borderRadius": "18px",
                                "padding": "12px 14px",
                                "boxShadow": "0 10px 26px rgba(15, 23, 42, 0.06)",
                                "border": f"1px solid {PALETTE['border']}",
                                "display": "flex",
                                "flexDirection": "column",
                                "gap": "10px",
                            },
                            children=[
                                html.Span(
                                    "Recording & decomposition",
                                    style={
                                        "fontSize": "11px",
                                        "letterSpacing": "0.12em",
                                        "textTransform": "uppercase",
                                        "color": PALETTE["muted"],
                                        "fontWeight": "600",
                                    },
                                ),
                                html.Div(
                                    style={
                                        "display": "grid",
                                        "gridTemplateColumns": "1.1fr 0.9fr",
                                        "gridGap": "8px",
                                    },
                                    children=[
                                        # Subject dropdown
                                        html.Div(
                                            children=[
                                                html.Label(
                                                    "Subject / Session",
                                                    style={
                                                        "fontSize": "12px",
                                                        "marginBottom": "4px",
                                                        "color": PALETTE["muted"],
                                                    },
                                                ),
                                                dcc.Dropdown(
                                                    id="subject-dropdown",
                                                    options=[
                                                        {
                                                            "label": f"Subject {code}",
                                                            "value": code,
                                                        }
                                                        for code in subject_codes
                                                    ],
                                                    value=subject_codes[0]
                                                    if subject_codes
                                                    else None,
                                                    clearable=False,
                                                    style={
                                                        "color": "#111111",
                                                        "backgroundColor": "#FFFFFF",
                                                        "fontSize": "13px",
                                                    },
                                                ),
                                            ]
                                        ),
                                        # Method seçimi (VMDON / AVMD Spark)
                                        html.Div(
                                            children=[
                                                html.Label(
                                                    "Decomposition method",
                                                    style={
                                                        "fontSize": "12px",
                                                        "marginBottom": "4px",
                                                        "color": PALETTE["muted"],
                                                    },
                                                ),
                                                dcc.RadioItems(
                                                    id="method-radio",
                                                    options=[
                                                        {
                                                            "label": "VMDON (online)",
                                                            "value": "vmdon",
                                                        },
                                                        {
                                                            "label": "AVMD Spark (offline)",
                                                            "value": "avmd_spark",
                                                        },
                                                    ],
                                                    value="vmdon",
                                                    inline=True,
                                                    labelStyle={
                                                        "marginRight": "10px",
                                                        "display": "inline-flex",
                                                        "alignItems": "center",
                                                        "gap": "4px",
                                                        "fontSize": "12px",
                                                    },
                                                    inputStyle={
                                                        "marginRight": "4px",
                                                    },
                                                ),
                                            ]
                                        ),
                                    ],
                                ),
                                html.Div(
                                    id="subject-info",
                                    style={
                                        "fontSize": "11px",
                                        "color": PALETTE["muted"],
                                        "marginTop": "2px",
                                        "lineHeight": "1.4",
                                    },
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),

        # ---------- ANA GRID (3 PANEL) ---------- #
        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "1fr 1.6fr 1fr",
                "gridGap": "20px",
                "marginBottom": "18px",
            },
            children=[
                # TIME-DOMAIN PANEL
                html.Div(
                    style=PANEL_STYLE,
                    children=[
                        html.Div(
                            style={
                                "display": "flex",
                                "justifyContent": "space-between",
                                "alignItems": "center",
                            },
                            children=[
                                html.Div(
                                    children=[
                                        html.H3(
                                            "Time-domain metrics",
                                            style={
                                                "marginBottom": "2px",
                                                "fontSize": "16px",
                                            },
                                        ),
                                        html.Span(
                                            "Short-term HRV indices over recent window",
                                            style={
                                                "fontSize": "12px",
                                                "color": PALETTE["muted"],
                                            },
                                        ),
                                    ]
                                ),
                            ],
                        ),
                        html.Div(
                            id="metrics-grid",
                            style={
                                "display": "grid",
                                "gridTemplateColumns": "repeat(3, 1fr)",
                                "gridGap": "10px",
                                "marginTop": "8px",
                            },
                        ),
                        html.Div(
                            style={"marginTop": "12px"},
                            children=[
                                html.Span(
                                    "Heart rate over time",
                                    style={
                                        "fontSize": "12px",
                                        "color": PALETTE["muted"],
                                    },
                                ),
                                dcc.Graph(
                                    id="hr-graph",
                                    style={"height": "260px"},
                                    config={"displayModeBar": False},
                                ),
                            ],
                        ),
                    ],
                ),

                # ORTA PANEL: VMDON veya AVMD SPARK
                html.Div(
                    style=PANEL_STYLE,
                    children=[
                        html.H3(
                            "Frequency decomposition & band power",
                            style={"marginBottom": "2px", "fontSize": "16px"},
                        ),
                        html.Span(
                            "Online VMDON vs. offline PySpark AVMD",
                            style={"fontSize": "12px", "color": PALETTE["muted"]},
                        ),

                        dcc.Graph(
                            id="vmdon-components-graph",
                            style={"height": "260px", "marginTop": "10px"},
                            config={"displayModeBar": False},
                        ),

                        dcc.Graph(
                            id="band-pie-graph",
                            style={"height": "230px", "marginTop": "6px"},
                            config={"displayModeBar": False},
                        ),
                    ],
                ),

                # POINCARÉ PANELİ
                html.Div(
                    style=PANEL_STYLE,
                    children=[
                        html.H3(
                            "Poincaré & non-linear indices",
                            style={"marginBottom": "2px", "fontSize": "16px"},
                        ),
                        html.Span(
                            "Beat-to-beat dynamics and SD1/SD2 balance",
                            style={"fontSize": "12px", "color": PALETTE["muted"]},
                        ),
                        dcc.Graph(
                            id="poincare-graph",
                            style={"height": "260px"},
                            config={"displayModeBar": False},
                        ),
                        html.Div(
                            id="poincare-metrics",
                            style={
                                "display": "grid",
                                "gridTemplateColumns": "repeat(2, 1fr)",
                                "gridGap": "10px",
                            },
                        ),
                    ],
                ),
            ],
        ),

        # ---------- ALT STATUS BAR ---------- #
        html.Div(
            style={
                "backgroundColor": PALETTE["card"],
                "borderRadius": "16px",
                "padding": "10px 14px",
                "display": "flex",
                "justifyContent": "space-between",
                "alignItems": "center",
                "border": f"1px solid {PALETTE['border']}",
                "boxShadow": "0 6px 18px rgba(15, 23, 42, 0.04)",
            },
            children=[
                html.Div(
                    id="status-bar",
                    children=[
                        html.Span(
                            "Status: ",
                            style={
                                "fontWeight": "600",
                                "marginRight": "4px",
                            },
                        ),
                        html.Span(
                            id="status-text",
                            children="Waiting for stream...",
                            style={"color": PALETTE["muted"]},
                        ),
                    ],
                ),
                html.Span(
                    id="signal-quality-text",
                    children="Signal quality: - · Source: Kafka stream",
                    style={"fontSize": "12px", "color": PALETTE["muted"]},
                ),
            ],
        ),
    ],
)


# ----------------- CALLBACKS ----------------- #

@app.callback(
    Output("metrics-grid", "children"),
    Input("refresh", "n_intervals"),
    Input("subject-dropdown", "value"),
    Input("method-radio", "value"),  # method input, şimdilik sadece tetiklemek için
)
def update_metrics_grid(n, subject_code, method_value):
    # Hep kısa pencere (örn. son 5 dk) kullanıyoruz
    window_s = float(DEFAULT_WINDOW_S)

    params = {"subject": subject_code, "window_s": window_s}
    metrics = _fetch_from_api("/metrics/time", params)
    if not metrics:
        metrics = get_time_domain_metrics(subject_code, window_length_s=window_s)

    cards = [
        metric_card("SDNN", _fmt(metrics.get("sdnn"), 1), "ms"),
        metric_card("RMSSD", _fmt(metrics.get("rmssd"), 1), "ms"),
        metric_card("pNN50", _fmt(metrics.get("pnn50"), 1), "%"),
        metric_card("Mean HR", _fmt(metrics.get("mean_hr"), 1), "bpm"),
        metric_card("HR max", _fmt(metrics.get("hr_max"), 1), "bpm"),
        metric_card("HR min", _fmt(metrics.get("hr_min"), 1), "bpm"),
    ]
    return cards


@app.callback(
    Output("hr-graph", "figure"),
    Input("refresh", "n_intervals"),
    Input("subject-dropdown", "value"),
    Input("method-radio", "value"),
)
def update_hr_graph(n, subject_code, method_value):
    window_s = float(DEFAULT_WINDOW_S)

    params = {"subject": subject_code, "window_s": window_s}
    data = _fetch_from_api("/metrics/hr_timeseries", params)

    if data and data.get("t_sec") and data.get("hr_bpm"):
        t_sec = np.array(data["t_sec"], dtype=float)
        hr_bpm = np.array(data["hr_bpm"], dtype=float)
    else:
        t_sec, hr_bpm = get_hr_timeseries(subject_code, window_length_s=window_s)

    fig = go.Figure()
    if t_sec.size > 0 and hr_bpm.size > 0:
        fig.add_trace(
            go.Scatter(
                x=t_sec,
                y=hr_bpm,
                mode="lines",
                name="Heart Rate",
                line=dict(width=2.4, color=PALETTE["primary"]),
            )
        )

    fig.update_layout(
        title="Heart rate over time",
        xaxis_title="Time (s)",
        yaxis_title="Heart Rate (bpm)",
        margin=dict(l=40, r=20, t=40, b=40),
        height=260,
        paper_bgcolor=PALETTE["bg"],
        plot_bgcolor="#FFFFFF",
        xaxis=dict(showgrid=True, gridcolor="#EDF2F7"),
        yaxis=dict(showgrid=True, gridcolor="#EDF2F7"),
    )
    return fig


@app.callback(
    Output("vmdon-components-graph", "figure"),
    Output("band-pie-graph", "figure"),
    Input("refresh", "n_intervals"),
    Input("subject-dropdown", "value"),
    Input("method-radio", "value"),
)
def update_middle_panel(n, subject_code, method_value):
    """
    Orta panel:
      - VMDON (online) => eski grafikleri göster
      - AVMD Spark (offline) => Spark parquet'ten band / metrik bar + pie
    """
    # ----------------- VMDON ONLINE MODU ----------------- #
    if method_value == "vmdon":
        window_s = float(DEFAULT_WINDOW_S)
        params = {"subject": subject_code, "window_s": window_s}

        data = _fetch_from_api("/metrics/vmdon", params)
        if not data or not data.get("t_sec"):
            data = get_vmdon_components(subject_code, window_length_s=window_s)

        t_sec = np.array(data.get("t_sec", []), dtype=float)
        hf = np.array(data.get("hf", []), dtype=float)
        lf = np.array(data.get("lf", []), dtype=float)
        vlf = np.array(data.get("vlf", []), dtype=float)
        ulf = np.array(data.get("ulf", []), dtype=float)

        comp_fig = go.Figure()

        if t_sec.size > 0 and hf.size == t_sec.size:
            comp_fig.add_trace(
                go.Scatter(
                    x=t_sec,
                    y=hf,
                    mode="lines",
                    name="HF",
                    line=dict(width=1.8, color=PALETTE["accent"]),
                )
            )
        if t_sec.size > 0 and lf.size == t_sec.size:
            comp_fig.add_trace(
                go.Scatter(
                    x=t_sec,
                    y=lf,
                    mode="lines",
                    name="LF",
                    line=dict(width=1.4),
                )
            )
        if t_sec.size > 0 and vlf.size == t_sec.size:
            comp_fig.add_trace(
                go.Scatter(
                    x=t_sec,
                    y=vlf,
                    mode="lines",
                    name="VLF",
                    line=dict(width=1.2),
                )
            )
        if t_sec.size > 0 and ulf.size == t_sec.size:
            comp_fig.add_trace(
                go.Scatter(
                    x=t_sec,
                    y=ulf,
                    mode="lines",
                    name="ULF",
                    line=dict(width=1.0, dash="dot"),
                )
            )

        comp_fig.update_layout(
            title="VMDON components (HF / LF / VLF / ULF)",
            xaxis_title="Time (s)",
            yaxis_title="Component amplitude (RR, s)",
            template="plotly_white",
            margin=dict(l=40, r=20, t=30, b=40),
            height=260,
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            xaxis=dict(showgrid=True, gridcolor="#EDF2F7"),
            yaxis=dict(showgrid=True, gridcolor="#EDF2F7"),
            legend=dict(orientation="h", y=-0.25),
        )

        def _band_power(x: np.ndarray) -> float:
            if x.size == 0:
                return 0.0
            x = x - np.mean(x)
            return float(np.var(x))

        vlf_p = _band_power(vlf)
        lf_p = _band_power(lf)
        hf_p = _band_power(hf)
        ulf_p = _band_power(ulf)

        labels = ["VLF", "LF", "HF", "ULF"]
        values = [vlf_p, lf_p, hf_p, ulf_p]

        pie_fig = go.Figure(
            data=[go.Pie(labels=labels, values=values, hole=0.55)]
        )
        pie_fig.update_layout(
            title="VMDON band power distribution",
            template="plotly_white",
            margin=dict(l=10, r=10, t=30, b=20),
            height=220,
            showlegend=True,
        )

        return comp_fig, pie_fig

    # ----------------- AVMD SPARK OFFLINE MODU ----------------- #
    # df_avmd_spark global
    if df_avmd_spark is None or df_avmd_spark.empty:
        msg = "Spark AVMD results not found. Run PySpark job first."
        empty1 = _empty_figure("AVMD Spark band metrics", msg)
        empty2 = _empty_figure("AVMD Spark distribution", msg)
        return empty1, empty2

    df_sub = df_avmd_spark[df_avmd_spark["subject"] == subject_code]
    if df_sub.empty:
        msg = f"No Spark AVMD row for subject {subject_code}."
        empty1 = _empty_figure("AVMD Spark band metrics", msg)
        empty2 = _empty_figure("AVMD Spark distribution", msg)
        return empty1, empty2

    row = df_sub.iloc[0]

    # subject / method dışındaki numerik kolonları al
    numeric_cols = []
    numeric_vals = []
    for col in df_sub.columns:
        if col in ("subject", "method"):
            continue
        try:
            if np.issubdtype(df_sub[col].dtype, np.number):
                numeric_cols.append(col)
                numeric_vals.append(float(row[col]))
        except Exception:
            continue

    if not numeric_cols:
        msg = "No numeric AVMD band/metric columns to display."
        empty1 = _empty_figure("AVMD Spark band metrics", msg)
        empty2 = _empty_figure("AVMD Spark distribution", msg)
        return empty1, empty2

    labels = [c.replace("_", " ") for c in numeric_cols]

    # 1) Bar chart
    bar_fig = go.Figure(
        data=[go.Bar(x=labels, y=numeric_vals)]
    )
    bar_fig.update_layout(
        title=f"AVMD Spark metrics for Subject {subject_code}",
        xaxis_title="Band / metric",
        yaxis_title="Value",
        template="plotly_white",
        margin=dict(l=40, r=20, t=30, b=40),
        height=260,
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        xaxis=dict(showgrid=True, gridcolor="#EDF2F7", tickangle=-30),
        yaxis=dict(showgrid=True, gridcolor="#EDF2F7"),
    )

    # 2) Pie chart (oran)
    total = sum(numeric_vals)
    if total <= 0:
        pie_fig = _empty_figure(
            "AVMD Spark distribution",
            "Non-positive values; cannot build pie chart.",
        )
    else:
        pie_fig = go.Figure(
            data=[go.Pie(labels=labels, values=numeric_vals, hole=0.55)]
        )
        pie_fig.update_layout(
            title="AVMD Spark relative distribution",
            template="plotly_white",
            margin=dict(l=10, r=10, t=30, b=20),
            height=220,
            showlegend=True,
        )

    return bar_fig, pie_fig


@app.callback(
    Output("poincare-graph", "figure"),
    Input("refresh", "n_intervals"),
    Input("subject-dropdown", "value"),
    Input("method-radio", "value"),
)
def update_poincare_graph(n, subject_code, method_value):
    window_s = float(DEFAULT_WINDOW_S)

    params = {"subject": subject_code, "window_s": window_s}
    data = _fetch_from_api("/metrics/poincare", params)
    if not data:
        data = get_poincare_data(subject_code, window_length_s=window_s)

    fig = go.Figure()
    if data.get("x") and data.get("y"):
        fig.add_trace(
            go.Scatter(
                x=data["x"],
                y=data["y"],
                mode="markers",
                marker=dict(
                    size=5,
                    opacity=0.65,
                    color=PALETTE["primary"],
                ),
                name="RRn vs RRn+1",
            )
        )

    fig.update_layout(
        xaxis_title="RRₙ (ms)",
        yaxis_title="RRₙ₊₁ (ms)",
        template="plotly_white",
        margin=dict(l=40, r=20, t=10, b=40),
        height=260,
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        xaxis=dict(showgrid=True, gridcolor="#EDF2F7"),
        yaxis=dict(showgrid=True, gridcolor="#EDF2F7"),
    )
    return fig


@app.callback(
    Output("poincare-metrics", "children"),
    Input("refresh", "n_intervals"),
    Input("subject-dropdown", "value"),
    Input("method-radio", "value"),
)
def update_poincare_metrics(n, subject_code, method_value):
    window_s = float(DEFAULT_WINDOW_S)

    params = {"subject": subject_code, "window_s": window_s}
    data = _fetch_from_api("/metrics/poincare", params)
    if not data:
        data = get_poincare_data(subject_code, window_length_s=window_s)

    cards = [
        metric_card("SD1", _fmt(data.get("sd1"), 1), "ms"),
        metric_card("SD2", _fmt(data.get("sd2"), 1), "ms"),
        metric_card("SD1/SD2 ratio", _fmt(data.get("sd1_sd2_ratio"), 2), ""),
        metric_card("Stress index", _fmt(data.get("stress_index"), 2), ""),
    ]
    return cards


@app.callback(
    Output("subject-info", "children"),
    Output("subject-figure", "src"),
    Input("refresh", "n_intervals"),
    Input("subject-dropdown", "value"),
)
def update_subject_info(n, subject_code):
    info = get_subject_info(subject_code)

    age = info.get("age")
    sex = info.get("sex")
    group = info.get("group")

    if age is None:
        age_str = "Unknown"
    else:
        try:
            age_str = str(int(float(age)))
        except Exception:
            age_str = str(age)

    sex_str = "-" if sex in (None, "", float("nan")) else str(sex)
    group_str = "" if group in (None, "") else f" · Group: {group}"

    sex_norm = str(sex).strip().lower() if sex is not None else ""

    if sex_norm.startswith("m"):
        filename = "Man.png"
    elif sex_norm.startswith("f"):
        filename = "woman.png"
    else:
        filename = "woman.png"

    figure_src = app.get_asset_url(filename)

    info_children = [
        html.Span(
            f"Subject {info.get('code', subject_code)}",
            style={"fontWeight": "600", "color": PALETTE["text"]},
        ),
        html.Br(),
        html.Span(f"Age: {age_str} · Sex: {sex_str}{group_str}"),
    ]

    return info_children, figure_src


@app.callback(
    Output("status-text", "children"),
    Output("signal-quality-text", "children"),
    Input("refresh", "n_intervals"),
    Input("subject-dropdown", "value"),
    Input("method-radio", "value"),
)
def update_status_bar(n, subject_code, method_value):
    window_s = float(DEFAULT_WINDOW_S)

    params = {"subject": subject_code, "window_s": window_s}
    status = _fetch_from_api("/metrics/status", params)
    if not status:
        status = get_signal_status(subject_code, window_length_s=window_s)

    quality_label = status.get("quality_label", "Unknown")
    status_text = status.get("status_text", "No status available")
    outlier_percent = status.get("outlier_percent", 0.0)

    status_msg = status_text
    quality_msg = (
        f"Signal quality: {quality_label}"
        f" · Irregular beats ≈ {outlier_percent:.1f}%"
        f" · Source: Kafka stream"
    )

    return status_msg, quality_msg


# ----------------- MAIN ----------------- #

if __name__ == "__main__":
    # Sadece Dash’i çalıştır. Kafka consumer FastAPI içinde çalışacak.
    app.run(debug=True)

