import os
import sys

CURRENT_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.dirname(CURRENT_DIR)

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from dash import Dash, html, dcc
import plotly.graph_objects as go
import math

from hrv_metrics.service_hrv import get_time_domain_metrics

app = Dash(__name__)

metrics = get_time_domain_metrics("000")

# ----------------- DUMMY VERI (sadece layout görmek için) ----------------- #
time_points = list(range(60))  # 60 saniyelik örnek aks
hr_values = [70 + 5 * math.sin(t / 5) for t in time_points]  # bpm

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
# HR line chart
hr_fig = go.Figure()
hr_fig.add_trace(
    go.Scatter(
        x=time_points,
        y=hr_values,
        mode="lines",
        name="HR (bpm)",
    )
)
hr_fig.update_layout(
    margin=dict(l=20, r=20, t=30, b=30),
    template="plotly_dark",
    xaxis_title="Time (s)",
    yaxis_title="Heart Rate (bpm)",
)

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
app.layout = html.Div(
    style={
        "backgroundColor": "#131F39",
        "minHeight": "100vh",
        "padding": "20px",
        "color": "white",
        "fontFamily": "Arial, sans-serif",
    },
    children=[
        # ÜST BAŞLIK + DROPDOWN
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
                    style={"minWidth": "220px"},
                    children=[
                        html.Label(
                            "Subject / Session",
                            style={"fontSize": "12px", "marginBottom": "4px"},
                        ),
                        dcc.Dropdown(
                            id="subject-dropdown",
                            options=[
                                {"label": "Subject 001 - Rest", "value": "sub001_rest"},
                                {"label": "Subject 001 - Stress test", "value": "sub001_stress"},
                                {"label": "Subject 002 - Rest", "value": "sub002_rest"},
                            ],
                            value="sub001_rest",
                            style={"color": "#000000"},
                        ),
                    ],
                ),
            ],
        ),

        # ORTA BÖLÜM: 3 SÜTUN
        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "1.3fr 1fr 1fr",
                "gridGap": "20px",
                "marginBottom": "20px",
            },
            children=[
                # ---------------- SOL SÜTUN: TIME-DOMAIN + HR LINE ---------------- #
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
                        # 2x3 grid metric kartları
                        html.Div(
                            style={
                                "display": "grid",
                                "gridTemplateColumns": "repeat(3, 1fr)",
                                "gridGap": "10px",
                                "marginTop": "5px",
                            },
                            children=[
                                metric_card("SDNN", f"{metrics['sdnn']:.1f}", "ms"),
                                metric_card("RMSSD", f"{metrics['rmssd']:.1f}", "ms"),
                                metric_card("pNN50", f"{metrics['pnn50']:.1f}", "%"),
                                metric_card("Mean HR", f"{metrics['mean_hr']:.1f}", "bpm"),
                                metric_card("HR max", f"{metrics['hr_max']:.1f}", "bpm"),
                                metric_card("HR min", f"{metrics['hr_min']:.1f}", "bpm"),
                            ],
                        ),

                        # HR çizgi grafiği
                        html.Div(
                            style={"marginTop": "10px"},
                            children=[
                                html.Span(
                                    "Heart rate over time",
                                    style={"fontSize": "12px", "color": "#A0AEC0"},
                                ),
                                dcc.Graph(
                                    id="hr-graph",
                                    figure=hr_fig,
                                    style={"height": "260px"},
                                ),
                            ],
                        ),
                    ],
                ),

                # ---------------- ORTA SÜTUN: FREQUENCY-DOMAIN ---------------- #
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

                # ---------------- SAĞ SÜTUN: POINCARÉ + INDEX ---------------- #
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
                            figure=poincare_fig,
                            style={"height": "260px"},
                        ),
                        html.Div(
                            style={
                                "display": "grid",
                                "gridTemplateColumns": "repeat(2, 1fr)",
                                "gridGap": "10px",
                            },
                            children=[
                                metric_card("SD1", "24", "ms"),
                                metric_card("SD2", "55", "ms"),
                                metric_card("SD1/SD2 ratio", "0.44", ""),
                                metric_card("Stress index", "Normal", ""),
                            ],
                        ),
                    ],
                ),
            ],
        ),

        # ALT STATUS BAR
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
                        html.Span(
                            "Status: ",
                            style={"fontWeight": "bold"},
                        ),
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


if __name__ == "__main__":
    app.run(debug=True)
