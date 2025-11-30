import os
import sys
import numpy as np
import pandas as pd
from dash import Dash, html, dcc, Input, Output
import plotly.graph_objects as go

# Proje kökünü sys.path'e ekle (src içinden import için)
CURRENT_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.dirname(CURRENT_DIR)
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from hrv_metrics.service_hrv import (
    get_time_domain_metrics,
    get_available_subject_codes,
    get_hr_timeseries,
    get_poincare_data,
    get_subject_info,
    get_freq_domain_metrics,
    get_signal_status,      # 👈 YENİ
)

# Dash uygulaması
app = Dash(__name__)

# Uygulama açılışında bir kez okunacak subject listesi
subject_codes = get_available_subject_codes()


# ----------------- KÜÇÜK YARDIMCI: METRIC CARD ----------------- #
def metric_card(title: str, value: str, unit: str = "") -> html.Div:
    """Dashboard üzerinde metrik gösterimi için basit kart bileşeni."""
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
                    html.Span(
                        value,
                        style={"fontSize": "20px", "fontWeight": "bold"},
                    ),
                    html.Span(
                        unit,
                        style={"fontSize": "11px", "color": "#CBD5F5"},
                    ),
                ],
            ),
        ],
    )

# ----------------- APP LAYOUT ----------------- #
# layout'tan ÖNCE bir yerde:
# Ortak panel stili (3 sütun için)
PANEL_STYLE = {
    "backgroundColor": "#131F39",
    "border": "1px solid #223459",
    "borderRadius": "12px",
    "padding": "15px",
    "display": "flex",
    "flexDirection": "column",
    "gap": "12px",
}

# ----------------- APP LAYOUT ----------------- #
app.layout = html.Div(
    style={
        "backgroundColor": "#131F39",
        "minHeight": "100vh",
        "padding": "20px",
        "color": "white",
        "fontFamily": "Arial, sans-serif",
    },
    children=[
        # ---------------- ÜST BAŞLIK + DROPDOWN ---------------- #
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
                        html.H1(
                            "HRV Live Dashboard",
                            style={"marginBottom": "5px"},
                        ),
                        html.Span(
                            "Real-time heart rate variability monitoring",
                            style={"fontSize": "13px", "color": "#A0AEC0"},
                        ),
                    ]
                ),
                html.Div(
                    style={"minWidth": "260px", "display": "flex", "flexDirection": "column", "gap": "8px"},
                    children=[
                        # Subject selection
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
                        # Analysis window selection
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
                                        {"label": "Last 5 minutes", "value": "last_5min"},
                                    ],
                                    value="full",
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
        # ---------------- ORTA BÖLÜM: 3 SÜTUN ---------------- #
        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "1.3fr 1fr 1fr",
                "gridGap": "20px",
                "marginBottom": "20px",
            },
            children=[
                # SOL SÜTUN: TIME-DOMAIN + HR
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
                                dcc.Graph(
                                    id="hr-graph",
                                    style={"height": "260px"},
                                ),
                            ],
                        ),
                    ],
                ),
        
                # ORTA SÜTUN: FREQUENCY-DOMAIN
                html.Div(
                    style=PANEL_STYLE,
                    children=[
                        html.H3("Frequency-domain metrics"),
                        html.Span(
                            "LF / HF power evolution and band distribution",
                            style={"fontSize": "12px", "color": "#A0AEC0"},
                        ),
                        dcc.Graph(
                            id="lf-hf-graph",
                            style={"height": "230px"},
                        ),
                        dcc.Graph(
                            id="band-pie-graph",
                            style={"height": "220px"},
                        ),
                    ],
                ),

                # SAĞ SÜTUN: POINCARÉ + NON-LINEAR
                html.Div(
                    style=PANEL_STYLE,
                    children=[
                        html.H3("Poincaré & non-linear indices"),
                        html.Span(
                            "Beat-to-beat dynamics and SD1/SD2 balance",
                            style={"fontSize": "12px", "color": "#A0AEC0"},
                        ),
                        dcc.Graph(
                            id="poincare-graph",
                            style={"height": "260px"},
                        ),
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

        # ---------------- ALT STATUS BAR ---------------- #
        html.Div(
            style={
                "backgroundColor": "#223459",
                "borderRadius": "10px",
                "padding": "10px"
                "5px",
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
                            id="status-text",
                            children="Normal (no alerts detected in the last window)",
                        ),
                    ],
                ),
                html.Span(
                    id="signal-quality-text",
                    children="Signal quality: OK · Source: recorded RR file",
                    style={"fontSize": "12px", "color": "#A0AEC0"},
                ),
            ],
        ),
    ],
)
# ----------------- window_to_seconds ----------------- #
def _window_to_seconds(window_value: str | None) -> float | None:
    """
    UI'daki 'Analysis window' seçimini saniyeye çevirir.
    - "full"       -> None  (tüm kayıt)
    - "last_5min"  -> 300.0
    Gerekirse ileride başka seçenekler de eklenebilir.
    """
    if window_value == "last_5min":
        return 5 * 60.0
    # default: full recording
    return None




# ----------------- CALLBACKS ----------------- #

# ---------------- CALLBACKS ---------------- #

@app.callback(
    Output("metrics-grid", "children"),
    Input("subject-dropdown", "value"),
)
def update_metrics_grid(subject_code):
    """
    Sol sütundaki time-domain metrik kartlarını günceller.
    Şu an sadece temel metrikler gösteriliyor; backend tarafında
    daha fazla metrik hesaplanıyor olsa da burada çekirdek seti kullanıyoruz.
    """
    metrics = get_time_domain_metrics(subject_code)

    cards = [
        metric_card("SDNN", f"{metrics['sdnn']:.1f}", "ms"),
        metric_card("RMSSD", f"{metrics['rmssd']:.1f}", "ms"),
        metric_card("pNN50", f"{metrics['pnn50']:.1f}", "%"),
        metric_card("Mean HR", f"{metrics['mean_hr']:.1f}", "bpm"),
        metric_card("HR max", f"{metrics['hr_max']:.1f}", "bpm"),
        metric_card("HR min", f"{metrics['hr_min']:.1f}", "bpm"),
    ]
    return cards


@app.callback(
    Output("hr-graph", "figure"),
    Input("subject-dropdown", "value"),
)
def update_hr_graph(subject_code):
    """
    Kalp hızı zaman serisini çizer.
    Şu anda tam kayıt üzerinden çalışıyor (max_points ile kısaltılıyor).
    """
    t_sec, hr_bpm = get_hr_timeseries(subject_code)

    fig = go.Figure()

    if len(t_sec) > 0 and len(hr_bpm) > 0:
        fig.add_trace(
            go.Scatter(
                x=t_sec,
                y=hr_bpm,
                mode="lines",
                name="Heart Rate",
            )
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
    Input("subject-dropdown", "value"),
)
def update_frequency_domain_graphs(subject_code):
    # 1) Backend'den frekans domeni metriklerini çek (Welch)
    fd = get_freq_domain_metrics(subject_code)

    freq = np.array(fd.get("freq", []), dtype=float)
    psd = np.array(fd.get("psd", []), dtype=float)
    band_powers = fd.get("band_powers", {})

    # Veri yoksa: boş figürler döndür
    if freq.size == 0 or psd.size == 0:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            template="plotly_dark",
            margin=dict(l=40, r=20, t=40, b=40),
            height=230,
        )

        pie_fig = go.Figure(
            data=[
                go.Pie(
                    labels=["VLF", "LF", "HF"],
                    values=[0, 0, 0],
                    hole=0.3,
                )
            ]
        )
        pie_fig.update_layout(
            title="VLF / LF / HF power distribution",
            template="plotly_dark",
            margin=dict(l=20, r=20, t=30, b=30),
            height=220,
        )
        return empty_fig, pie_fig

    max_psd = float(psd.max())

    # 2) PSD grafiği (tam spektrum) + arkaplanda band shading
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=freq,
            y=psd,
            mode="lines",
            name="PSD (Welch)",
        )
    )

    bands = {
        "VLF": (0.0033, 0.04),
        "LF": (0.04, 0.15),
        "HF": (0.15, 0.40),
    }

    colors = {
        "VLF": "rgba(56, 161, 105, 0.25)",   # yeşilimsi
        "LF": "rgba(66, 153, 225, 0.25)",   # mavi
        "HF": "rgba(237, 100, 166, 0.25)",  # pembe
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

    # 3) Pie chart: VLF / LF / HF dağılımı
    labels = ["VLF", "LF", "HF"]
    values = [
        band_powers.get("VLF", 0.0),
        band_powers.get("LF", 0.0),
        band_powers.get("HF", 0.0),
    ]

    pie_fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.3,
            )
        ]
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
    Input("subject-dropdown", "value"),
)
def update_poincare_graph(subject_code):
    """
    Sağ sütundaki Poincaré scatter grafiğini günceller.
    """
    data = get_poincare_data(subject_code)

    fig = go.Figure()

    if data["x"] and data["y"]:
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
    Input("subject-dropdown", "value"),
)
def update_poincare_metrics(subject_code):
    """
    Sağ sütundaki SD1 / SD2 / oran / stress kartlarını günceller.
    """
    data = get_poincare_data(subject_code)

    sd1 = data["sd1"]
    sd2 = data["sd2"]
    ratio = data["sd1_sd2_ratio"]
    stress = data["stress_index"]

    cards = [
        metric_card("SD1", f"{sd1:.1f}", "ms"),
        metric_card("SD2", f"{sd2:.1f}", "ms"),
        metric_card("SD1/SD2 ratio", f"{ratio:.2f}", ""),
        metric_card("Stress index", f"{stress:.2f}", ""),
    ]
    return cards


@app.callback(
    Output("subject-info", "children"),
    Input("subject-dropdown", "value"),
)
def update_subject_info(subject_code):
    """
    Üst sağdaki subject yaş / cinsiyet / grup bilgisini günceller.
    """
    info = get_subject_info(subject_code)

    age = info.get("age")
    sex = info.get("sex")
    group = info.get("group")

    # Age formatla (NaN / None durumlarına karşı)
    if age is None:
        age_str = "Unknown"
    elif isinstance(age, float) and age != age:  # NaN kontrolü
        age_str = "Unknown"
    else:
        if isinstance(age, (int, float)):
            age_str = str(int(age))
        else:
            age_str = str(age)

    sex_str = "-" if sex in (None, "", float("nan")) else str(sex)
    group_str = "" if group in (None, "") else f" · Group: {group}"

    return [
        html.Span(
            f"Subject {info['code']}",
            style={"fontWeight": "bold"},
        ),
        html.Br(),
        html.Span(f"Age: {age_str} · Sex: {sex_str}{group_str}"),
    ]


@app.callback(
    Output("status-text", "children"),
    Output("signal-quality-text", "children"),
    Input("subject-dropdown", "value"),
)
def update_status_bar(subject_code):
    """
    Alt bar'daki sinyal kalite özetini günceller.
    get_signal_status sadece subject_code alıyor, pencere almıyor.
    """
    status = get_signal_status(subject_code)

    quality_label = status.get("quality_label", "Unknown")
    status_text = status.get("status_text", "No status available")
    outlier_percent = status.get("outlier_percent", 0.0)

    # Status: ... kısmı
    status_msg = status_text

    # Signal quality: ... kısmı
    quality_msg = (
        f"Signal quality: {quality_label}"
        f" · Irregular beats ≈ {outlier_percent:.1f}%"
        f" · Source: recorded RR file"
    )

    return status_msg, quality_msg


if __name__ == "__main__":
    app.run(debug=True)

