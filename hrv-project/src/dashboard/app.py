import os
import sys
import numpy as np
from dash import Dash, html, dcc, Input, Output
import plotly.graph_objects as go
import requests 

# --- PROJE KÖKÜNÜ sys.path'E EKLE (ÖNCE BUNU YAP) --- #
CURRENT_DIR = os.path.dirname(__file__)                  # .../src/dashboard
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))  # .../hrv-project
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Artık src.* importları güvenli
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


app = Dash(__name__)

subject_codes = get_available_subject_codes()

PANEL_STYLE = {
    "backgroundColor": "#131F39",
    "border": "1px solid #223459",
    "borderRadius": "12px",
    "padding": "15px",
    "display": "flex",
    "flexDirection": "column",
    "gap": "12px",
}


def _fetch_from_api(path: str, params: dict) -> dict:
    """
    FastAPI HRV servisinden JSON veri çeker.
    Hata olursa bos dict dondurur.
    """
    base_url = settings.api.base_url.rstrip("/")
    url = f"{base_url}{path}"
    try:
        resp = requests.get(url, params=params, timeout=1.0)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        # Buraya istersen logging de ekleyebilirsin
        print(f"[dashboard] API request failed: {e} (url={url}, params={params})")
        return {}


# Config-driven defaults (dashboard-level)
DEFAULT_WINDOW_S = settings.dashboard.default_window_s  # e.g. 5 * 60
VLF_BAND = settings.hrv.vlf_band
LF_BAND = settings.hrv.lf_band
HF_BAND = settings.hrv.hf_band


def metric_card(title: str, value: str, unit: str = "") -> html.Div:
    return html.Div(
        style={
            "backgroundColor": "#223459",
            "borderRadius": "10px",
            "padding": "10px 12px",
            "display": "flex",
            "flexDirection": "column",
            "gap": "4px",
            "minHeight": "70px",
        },
        children=[
            html.Span(
                title,
                style={
                    "fontSize": "12px",
                    "color": "#A0AEC0",
                    "letterSpacing": "0.05em",
                },
            ),
            html.Div(
                style={"display": "flex", "alignItems": "baseline", "gap": "4px"},
                children=[
                    html.Span(value, style={"fontSize": "20px", "fontWeight": "bold"}),
                    html.Span(unit, style={"fontSize": "11px", "color": "#CBD5F5"}),
                ],
            ),
        ],
    )


def _window_to_seconds(window_value: str | None) -> float | None:
    """
    UI'dan gelen pencere seçimini (full / last_5min) saniye cinsine çevirir.

    "last_5min" değeri, settings.dashboard.default_window_s üzerinden yönetilir.
    """
    if window_value == "last_5min":
        return float(DEFAULT_WINDOW_S)
    # "full" veya bilinmeyen değerler -> None (full buffer/recording)
    return None


def _fmt(v, nd: int = 1) -> str:
    try:
        if v is None:
            return "N/A"
        if isinstance(v, float) and np.isnan(v):
            return "N/A"
        return f"{float(v):.{nd}f}"
    except Exception:
        return "N/A"


app.layout = html.Div(
    style={
        "backgroundColor": "#131F39",
        "minHeight": "100vh",
        "padding": "20px",
        "color": "white",
        "fontFamily": "Arial, sans-serif",
    },
    children=[
        # LIVE REFRESH (1s)
        dcc.Interval(id="refresh", interval=1000, n_intervals=0),

        html.Div(
            style={
                "display": "flex",
                "justifyContent": "space-between",
                "alignItems": "center",
                "marginBottom": "20px",
            },
            children=[
                html.Div(
                    children=[
                        html.H1("HRV Live Dashboard", style={"marginBottom": "5px"}),
                        html.Span(
                            "Real-time heart rate variability monitoring",
                            style={"fontSize": "13px", "color": "#A0AEC0"},
                        ),
                    ]
                ),
                html.Div(
                    style={
                        "minWidth": "260px",
                        "display": "flex",
                        "flexDirection": "column",
                        "gap": "8px",
                    },
                    children=[
                        html.Div(
                            children=[
                                html.Label(
                                    "Subject / Session",
                                    style={"fontSize": "12px", "marginBottom": "4px"},
                                ),
                                dcc.Dropdown(
                                    id="subject-dropdown",
                                    options=[
                                        {"label": f"Subject {code}", "value": code}
                                        for code in subject_codes
                                    ],
                                    value=subject_codes[0] if subject_codes else None,
                                    clearable=False,
                                    style={"color": "#000000"},
                                ),
                            ]
                        ),
                        html.Div(
                            children=[
                                html.Label(
                                    "Analysis window",
                                    style={"fontSize": "12px", "marginBottom": "4px"},
                                ),
                                dcc.Dropdown(
                                    id="window-dropdown",
                                    options=[
                                        {"label": "Full recording", "value": "full"},
                                        {
                                            "label": "Last 5 minutes",
                                            "value": "last_5min",
                                        },
                                    ],
                                    value="last_5min",  # canlıda default daha mantıklı
                                    clearable=False,
                                    style={"color": "#000000"},
                                ),
                            ]
                        ),
                        html.Div(
                            id="subject-info",
                            style={
                                "fontSize": "11px",
                                "color": "#A0AEC0",
                                "marginTop": "2px",
                                "lineHeight": "1.4",
                            },
                        ),
                    ],
                ),
            ],
        ),

        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "1.3fr 1fr 1fr",
                "gridGap": "20px",
                "marginBottom": "20px",
            },
            children=[
                html.Div(
                    style=PANEL_STYLE,
                    children=[
                        html.H3("Time-domain metrics", style={"marginBottom": "5px"}),
                        html.Span(
                            "Short-term HRV indices computed over recent window",
                            style={"fontSize": "12px", "color": "#A0AEC0"},
                        ),
                        html.Div(
                            id="metrics-grid",
                            style={
                                "display": "grid",
                                "gridTemplateColumns": "repeat(3, 1fr)",
                                "gridGap": "10px",
                                "marginTop": "5px",
                            },
                            children=[],
                        ),
                        html.Div(
                            style={"marginTop": "10px"},
                            children=[
                                html.Span(
                                    "Heart rate over time",
                                    style={"fontSize": "12px", "color": "#A0AEC0"},
                                ),
                                dcc.Graph(id="hr-graph", style={"height": "260px"}),
                            ],
                        ),
                    ],
                ),
                html.Div(
                    style=PANEL_STYLE,
                    children=[
                        html.H3("Frequency-domain metrics"),
                        html.Span(
                            "LF / HF power and band distribution",
                            style={"fontSize": "12px", "color": "#A0AEC0"},
                        ),
                        dcc.Graph(id="lf-hf-graph", style={"height": "230px"}),
                        dcc.Graph(id="band-pie-graph", style={"height": "220px"}),
                    ],
                ),
                html.Div(
                    style=PANEL_STYLE,
                    children=[
                        html.H3("Poincaré & non-linear indices"),
                        html.Span(
                            "Beat-to-beat dynamics and SD1/SD2 balance",
                            style={"fontSize": "12px", "color": "#A0AEC0"},
                        ),
                        dcc.Graph(id="poincare-graph", style={"height": "260px"}),
                        html.Div(
                            id="poincare-metrics",
                            style={
                                "display": "grid",
                                "gridTemplateColumns": "repeat(2, 1fr)",
                                "gridGap": "10px",
                            },
                            children=[],
                        ),
                    ],
                ),
            ],
        ),

        html.Div(
            style={
                "backgroundColor": "#223459",
                "borderRadius": "10px",
                "padding": "10px 5px",
                "display": "flex",
                "justifyContent": "space-between",
                "alignItems": "center",
            },
            children=[
                html.Div(
                    id="status-bar",
                    children=[
                        html.Span("Status: ", style={"fontWeight": "bold"}),
                        html.Span(
                            id="status-text", children="Waiting for stream..."
                        ),
                    ],
                ),
                html.Span(
                    id="signal-quality-text",
                    children="Signal quality: - · Source: Kafka stream",
                    style={"fontSize": "12px", "color": "#A0AEC0"},
                ),
            ],
        ),
    ],
)

# ----------------- CALLBACKS (NOW LIVE) ----------------- #


@app.callback(
    Output("metrics-grid", "children"),
    Input("refresh", "n_intervals"),
    Input("subject-dropdown", "value"),
    Input("window-dropdown", "value"),
)
def update_metrics_grid(n, subject_code, window_value):
    window_s = _window_to_seconds(window_value)
    # 1) API'den dene
    params = {"subject": subject_code}
    if window_s is not None:
        params["window_s"] = window_s
    metrics = _fetch_from_api("/metrics/time", params)
    # 2) API bos donerse (hata vs.), eski lokal fonksiyona fallback
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
    t_sec, hr_bpm = get_hr_timeseries(subject_code, window_length_s=window_s)

    fig = go.Figure()
    if len(t_sec) > 0 and len(hr_bpm) > 0:
        fig.add_trace(
            go.Scatter(x=t_sec, y=hr_bpm, mode="lines", name="Heart Rate")
        )

    fig.update_layout(
        title="Heart rate over time",
        xaxis_title="Time (s)",
        yaxis_title="Heart Rate (bpm)",
        template="plotly_dark",
        margin=dict(l=40, r=20, t=40, b=40),
        height=260,
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

    if freq.size == 0 or psd.size == 0:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            template="plotly_dark",
            margin=dict(l=40, r=20, t=40, b=40),
            height=230,
        )

        pie_fig = go.Figure(
            data=[
                go.Pie(labels=["VLF", "LF", "HF"], values=[0, 0, 0], hole=0.3)
            ]
        )
        pie_fig.update_layout(
            title="VLF / LF / HF power distribution (Welch)",
            template="plotly_dark",
            margin=dict(l=20, r=20, t=30, b=30),
            height=220,
        )
        return empty_fig, pie_fig

    max_psd = float(psd.max())

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=freq, y=psd, mode="lines", name="PSD (Welch)")
    )

    # Band sınırlarını konfigürasyondan al
    bands = {
        "VLF": VLF_BAND,
        "LF": LF_BAND,
        "HF": HF_BAND,
    }
    colors = {
        "VLF": "rgba(56, 161, 105, 0.25)",
        "LF": "rgba(66, 153, 225, 0.25)",
        "HF": "rgba(237, 100, 166, 0.25)",
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
                opacity=0.25,
                line=dict(width=0),
                layer="below",
            )
        )

    fig.update_layout(
        title="HRV power spectrum (Welch)",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Power (ms²/Hz)",
        template="plotly_dark",
        margin=dict(l=40, r=20, t=40, b=40),
        height=230,
        shapes=shapes,
        legend=dict(orientation="h", y=-0.2),
    )

    labels = ["VLF", "LF", "HF"]
    values = [
        band_powers.get("VLF", 0.0),
        band_powers.get("LF", 0.0),
        band_powers.get("HF", 0.0),
    ]

    pie_fig = go.Figure(
        data=[go.Pie(labels=labels, values=values, hole=0.3)]
    )
    pie_fig.update_layout(
        title="VLF / LF / HF power distribution (Welch)",
        template="plotly_dark",
        margin=dict(l=20, r=20, t=30, b=30),
        height=220,
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
                marker=dict(size=4, opacity=0.6),
                name="RRn vs RRn+1",
            )
        )

    fig.update_layout(
        title="Poincaré plot (RRn vs RRn+1)",
        xaxis_title="RRₙ (ms)",
        yaxis_title="RRₙ₊₁ (ms)",
        template="plotly_dark",
        margin=dict(l=40, r=20, t=40, b=40),
        height=260,
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

    # Pretty age
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
            style={"fontWeight": "bold"},
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


if __name__ == "__main__":
    app.run(debug=True)
