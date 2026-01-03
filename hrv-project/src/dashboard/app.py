# src/dashboard/app.py
import os
import sys
from typing import Any, Dict

import numpy as np
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
    get_freq_domain_metrics,
    get_signal_status,
)
from src.streaming.rr_consumer import start_consumer_background


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
    "minHeight": "90px",
    "border": f"1px solid {PALETTE['border']}",
    "alignItems": "center",       # ortala
    "justifyContent": "center",   # dikeyde ortala
    "textAlign": "center",        # metni ortala
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


DEFAULT_WINDOW_S = settings.dashboard.default_window_s
VLF_BAND = settings.hrv.vlf_band
LF_BAND = settings.hrv.lf_band
HF_BAND = settings.hrv.hf_band


def _window_to_seconds(window_value: str | None) -> float | None:
    """
    'full' -> None (tüm kayıt),
    'last_5min' -> DEFAULT_WINDOW_S
    """
    if window_value == "last_5min":
        return float(DEFAULT_WINDOW_S)
    return None


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
                    "color": PALETTE["primary"],     # <-- daha belirgin renk
                    "letterSpacing": "0.06em",
                    "textTransform": "uppercase",
                    "textAlign": "center",
                },
            ),
            html.Div(
                style={
                    "display": "flex",
                    "alignItems": "baseline",
                    "justifyContent": "center",      # <-- ortala
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
                # SOL: Başlık + tagline
                html.Div(
                    children=[
                        html.H1(
                            "HRV Live Dashboard",
                            style={
                                "margin": 0,
                                "fontSize": "28px",
                                "fontWeight": "700",
                                "letterSpacing": "0.02em",
                            },
                        ),
                        html.Span(
                            "Real-time heart rate variability monitoring",
                            style={
                                "fontSize": "13px",
                                "color": PALETTE["muted"],
                            },
                        ),
                    ]
                ),

                # SAĞ: Dropdown'lar + en sağda PNG ikon
                html.Div(
                    style={
                        "display": "flex",
                        "alignItems": "center",
                        "gap": "24px",
                    },
                    children=[
                        # Dropdown'lar
                        html.Div(
                            style={
                                "minWidth": "280px",
                                "display": "flex",
                                "flexDirection": "column",
                                "gap": "8px",
                            },
                            children=[
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
                                            },
                                        ),
                                    ]
                                ),
                                html.Div(
                                    children=[
                                        html.Label(
                                            "Analysis window",
                                            style={
                                                "fontSize": "12px",
                                                "marginBottom": "4px",
                                                "color": PALETTE["muted"],
                                            },
                                        ),
                                        dcc.Dropdown(
                                            id="window-dropdown",
                                            options=[
                                                {
                                                    "label": "Full recording",
                                                    "value": "full",
                                                },
                                                {
                                                    "label": "Last 5 minutes",
                                                    "value": "last_5min",
                                                },
                                            ],
                                            value="last_5min",
                                            clearable=False,
                                            style={
                                                "color": "#111111",
                                                "backgroundColor": "#FFFFFF",
                                            },
                                        ),
                                    ]
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

                        # En sağ: HRV dalga PNG
                        html.Img(
                            src="/assets/hrv_wave.png",   # assets klasöründeki dosya
                            style={
                                "height": "40px",
                                "display": "block",
                            },
                        ),
                    ],
                ),
            ],
        ),


        # ---------- ANA GRID (3 PANEL) ---------- #
        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "1.4fr 1fr 1fr",
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

                # FREQUENCY-DOMAIN PANEL
                html.Div(
                    style=PANEL_STYLE,
                    children=[
                        html.H3(
                            "Frequency-domain metrics",
                            style={"marginBottom": "2px", "fontSize": "16px"},
                        ),
                        html.Span(
                            "Spectral power (Welch) and VLF / LF / HF distribution",
                            style={"fontSize": "12px", "color": PALETTE["muted"]},
                        ),
                        html.Div(
                            style={
                                "display": "grid",
                                "gridTemplateColumns": "1.6fr 1fr",
                                "gridGap": "10px",
                                "marginTop": "10px",
                                "alignItems": "stretch",
                            },
                            children=[
                                dcc.Graph(
                                    id="lf-hf-graph",
                                    style={"height": "260px"},
                                    config={"displayModeBar": False},
                                ),
                                dcc.Graph(
                                    id="band-pie-graph",
                                    style={"height": "260px"},
                                    config={"displayModeBar": False},
                                ),
                            ],
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
    Input("window-dropdown", "value"),
)
def update_metrics_grid(n, subject_code, window_value):
    window_s = _window_to_seconds(window_value)

    params = {"subject": subject_code}
    if window_s is not None:
        params["window_s"] = window_s

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
    Input("window-dropdown", "value"),
)
def update_hr_graph(n, subject_code, window_value):
    window_s = _window_to_seconds(window_value)

    # 1) API'den dene
    params = {"subject": subject_code}
    if window_s is not None:
        params["window_s"] = window_s
    data = _fetch_from_api("/metrics/hr_timeseries", params)

    if data and data.get("t_sec") and data.get("hr_bpm"):
        t_sec = np.array(data["t_sec"], dtype=float)
        hr_bpm = np.array(data["hr_bpm"], dtype=float)
    else:
        # 2) API hata verirse / endpoint yoksa lokal servise fallback
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
    Output("lf-hf-graph", "figure"),
    Output("band-pie-graph", "figure"),
    Input("refresh", "n_intervals"),
    Input("subject-dropdown", "value"),
    Input("window-dropdown", "value"),
)
def update_frequency_domain_graphs(n, subject_code, window_value):
    window_s = _window_to_seconds(window_value)

    params = {"subject": subject_code}
    if window_s is not None:
        params["window_s"] = window_s

    fd = _fetch_from_api("/metrics/freq", params)
    if not fd:
        fd = get_freq_domain_metrics(subject_code, window_length_s=window_s)

    freq = np.array(fd.get("freq", []), dtype=float)
    psd = np.array(fd.get("psd", []), dtype=float)
    band_powers = fd.get("band_powers", {})

    # Boş durum
    if freq.size == 0 or psd.size == 0:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            template="plotly_white",
            margin=dict(l=40, r=20, t=20, b=40),
            height=230,
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            xaxis=dict(showgrid=True, gridcolor="#EDF2F7"),
            yaxis=dict(showgrid=True, gridcolor="#EDF2F7"),
        )

        pie_fig = go.Figure(
            data=[
                go.Pie(labels=["VLF", "LF", "HF"], values=[0, 0, 0], hole=0.55)
            ]
        )
        pie_fig.update_layout(
            title="VLF / LF / HF power distribution",
            template="plotly_white",
            margin=dict(l=10, r=10, t=30, b=20),
            height=220,
            showlegend=True,
        )
        return empty_fig, pie_fig

    max_psd = float(psd.max())

    # PSD grafiği
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=freq,
            y=psd,
            mode="lines",
            name="PSD (Welch)",
            line=dict(width=2.5, color=PALETTE["primary"]),
        )
    )

    bands = {"VLF": VLF_BAND, "LF": LF_BAND, "HF": HF_BAND}
    colors = {
        "VLF": "rgba(76, 111, 255, 0.08)",
        "LF": "rgba(56, 161, 105, 0.10)",
        "HF": "rgba(255, 139, 167, 0.12)",
    }

    shapes = []
    for name, (f_low, f_high) in bands.items():
        shapes.append(
            dict(
                type="rect",
                xref="x",
                yref="y",
                x0=f_low,
                x1=f_high,
                y0=0,
                y1=max_psd * 1.05,
                fillcolor=colors[name],
                line=dict(width=0),
                layer="below",
            )
        )

    fig.update_layout(
        title="HRV power spectrum (Welch)",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Power (ms²/Hz)",
        template="plotly_white",
        margin=dict(l=40, r=20, t=30, b=40),
        height=230,
        shapes=shapes,
        legend=dict(orientation="h", y=-0.25),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        xaxis=dict(showgrid=True, gridcolor="#EDF2F7"),
        yaxis=dict(showgrid=True, gridcolor="#EDF2F7"),
    )

    # Pasta grafiği
    labels = ["VLF", "LF", "HF"]
    values = [
        band_powers.get("VLF", 0.0),
        band_powers.get("LF", 0.0),
        band_powers.get("HF", 0.0),
    ]

    pie_fig = go.Figure(
        data=[go.Pie(labels=labels, values=values, hole=0.55)]
    )
    pie_fig.update_layout(
        title="VLF / LF / HF power distribution",
        template="plotly_white",
        margin=dict(l=10, r=10, t=30, b=20),
        height=220,
        showlegend=True,
    )

    return fig, pie_fig


@app.callback(
    Output("poincare-graph", "figure"),
    Input("refresh", "n_intervals"),
    Input("subject-dropdown", "value"),
    Input("window-dropdown", "value"),
)
def update_poincare_graph(n, subject_code, window_value):
    window_s = _window_to_seconds(window_value)

    params = {"subject": subject_code}
    if window_s is not None:
        params["window_s"] = window_s

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
    Input("window-dropdown", "value"),
)
def update_poincare_metrics(n, subject_code, window_value):
    window_s = _window_to_seconds(window_value)

    params = {"subject": subject_code}
    if window_s is not None:
        params["window_s"] = window_s

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

    return [
        html.Span(
            f"Subject {info.get('code', subject_code)}",
            style={"fontWeight": "600", "color": PALETTE["text"]},
        ),
        html.Br(),
        html.Span(f"Age: {age_str} · Sex: {sex_str}{group_str}"),
    ]


@app.callback(
    Output("status-text", "children"),
    Output("signal-quality-text", "children"),
    Input("refresh", "n_intervals"),
    Input("subject-dropdown", "value"),
    Input("window-dropdown", "value"),
)
def update_status_bar(n, subject_code, window_value):
    window_s = _window_to_seconds(window_value)

    params = {"subject": subject_code}
    if window_s is not None:
        params["window_s"] = window_s

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
    start_consumer_background()
    app.run(debug=True)
