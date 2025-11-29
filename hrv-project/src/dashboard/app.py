import os
import sys

CURRENT_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.dirname(CURRENT_DIR)

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from dash import Dash, html, dcc, Input, Output
import plotly.graph_objects as go
import math

from hrv_metrics.service_hrv import (
    get_time_domain_metrics,
    get_available_subject_codes,
    get_hr_timeseries,
    get_poincare_data,
    get_subject_info,          # 👈 BUNU EKLE
)

app = Dash(__name__)
# ----------------- GLOBAL DEGISKENLER ----------------- #
subject_codes = get_available_subject_codes()



# ----------------- DUMMY VERI (sadece layout görmek için) ----------------- #

time_points = list(range(60))

lf_values = [500 + 80 * math.sin(t / 10) for t in time_points]
hf_values = [400 + 60 * math.cos(t / 8) for t in time_points]

# Pie için örnek değerler
vlf_power = 200
lf_power = 450
hf_power = 350

# Poincaré plot için örnek RR serisi
rr_base = 0.8  # 800 ms
rr_series = [rr_base + 0.05 * math.sin(t / 4) for t in range(200)]
rr_n = rr_series[:-1]
rr_n1 = rr_series[1:]

# ----------------- PLOTLY FIGURE'LERI ----------------- #


# LF/HF line chart
lfhf_fig = go.Figure()
lfhf_fig.add_trace(go.Scatter(x=time_points, y=lf_values, mode="lines", name="LF"))
lfhf_fig.add_trace(go.Scatter(x=time_points, y=hf_values, mode="lines", name="HF"))
lfhf_fig.update_layout(
    margin=dict(l=20, r=20, t=30, b=30),
    template="plotly_dark",
    xaxis_title="Time (s)",
    yaxis_title="Power (a.u.)",
)

# VLF / LF / HF pie chart
pie_fig = go.Figure(
    data=[
        go.Pie(
            labels=["VLF", "LF", "HF"],
            values=[vlf_power, lf_power, hf_power],
            hole=0.3,
        )
    ]
)
pie_fig.update_layout(
    margin=dict(l=20, r=20, t=30, b=30),
    template="plotly_dark",
)

# Poincaré scatter
poincare_fig = go.Figure()
poincare_fig.add_trace(
    go.Scatter(
        x=rr_n,
        y=rr_n1,
        mode="markers",
        marker=dict(size=5),
        name="RR_n vs RR_n+1",
    )
)
poincare_fig.update_layout(
    margin=dict(l=20, r=20, t=30, b=30),
    template="plotly_dark",
    xaxis_title="RR_n (s)",
    yaxis_title="RR_n+1 (s)",
)


# ----------------- KÜÇÜK YARDIMCI: METRIC CARD ----------------- #
def metric_card(title: str, value: str, unit: str = ""):
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
                style={"fontSize": "12px", "color": "#A0AEC0", "letterSpacing": "0.05em"},
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
subject_codes = get_available_subject_codes()

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
                    style={"minWidth": "260px"},
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
                        html.Div(
                            id="subject-info",
                            style={
                                "fontSize": "11px",
                                "color": "#A0AEC0",
                                "marginTop": "6px",
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
                    style={
                        "backgroundColor": "#131F39",
                        "border": "1px solid #223459",
                        "borderRadius": "12px",
                        "padding": "15px",
                        "display": "flex",
                        "flexDirection": "column",
                        "gap": "12px",
                    },
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
                    style={
                        "backgroundColor": "#131F39",
                        "border": "1px solid #223459",
                        "borderRadius": "12px",
                        "padding": "15px",
                        "display": "flex",
                        "flexDirection": "column",
                        "gap": "12px",
                    },
                    children=[
                        html.H3("Frequency-domain metrics"),
                        html.Span(
                            "LF / HF power evolution and band distribution",
                            style={"fontSize": "12px", "color": "#A0AEC0"},
                        ),
                        dcc.Graph(
                            id="lf-hf-graph",
                            figure=lfhf_fig,
                            style={"height": "230px"},
                        ),
                        dcc.Graph(
                            id="band-pie-graph",
                            figure=pie_fig,
                            style={"height": "220px"},
                        ),
                    ],
                ),

                # SAĞ SÜTUN: POINCARÉ + NON-LINEAR
                html.Div(
                    style={
                        "backgroundColor": "#131F39",
                        "border": "1px solid #223459",
                        "borderRadius": "12px",
                        "padding": "15px",
                        "display": "flex",
                        "flexDirection": "column",
                        "gap": "12px",
                    },
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
                "padding": "10px 15px",
                "display": "flex",
                "justifyContent": "space-between",
                "alignItems": "center",
            },
            children=[
                html.Div(
                    children=[
                        html.Span("Status: ", style={"fontWeight": "bold"}),
                        html.Span("Normal (no alerts detected in the last window)"),
                    ]
                ),
                html.Span(
                    "Signal quality: OK · Update source: dummy data",
                    style={"fontSize": "12px", "color": "#A0AEC0"},
                ),
            ],
        ),
    ],
)
# ---------------- CALLBACKS ---------------- #

@app.callback(
    Output("metrics-grid", "children"),
    Input("subject-dropdown", "value"),
)
def update_metrics_grid(subject_code):
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
    t_sec, hr_bpm = get_hr_timeseries(subject_code)

    fig = go.Figure()
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
    Output("poincare-graph", "figure"),
    Input("subject-dropdown", "value"),
)
def update_poincare_graph(subject_code):
    data = get_poincare_data(subject_code)

    fig = go.Figure()
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
    Seçilen subject için yaş / cinsiyet / grup bilgisini gösterir.
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
        # 53.0 -> 53
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


if __name__ == "__main__":
    app.run(debug=True)