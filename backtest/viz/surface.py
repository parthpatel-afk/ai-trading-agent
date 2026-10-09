"""
The 3D landscape + the robustness curve (interactive plotly HTML).

surface_3d: the landscape — X = parameter value, Y (height) = the outcome, Z = date-window.
  Peaks = parameter values that produced the best outcome in that window. Read the SHAPE, not just
  the summit: a broad, gentle ridge that stays high across many date-windows is a ROBUST setting;
  a lone spike in one window is overfit luck that won't repeat.

robustness_2d: the all-time view — outcome vs parameter (bars) with the trade-count overlaid — the
  quick "is the good value a stable plateau or a lucky spike, and does it even trade enough?" read.
"""
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from backtest.analysis.landscape import to_grid


def surface_3d(cube: pd.DataFrame, param_label: str, metric_label: str,
               out_html: str, title: str = "") -> str:
    """Render the long-form cube as an interactive 3D surface saved to out_html. Returns the path."""
    grid = to_grid(cube)
    if grid.empty:
        raise ValueError("empty cube — nothing to plot (did the sweep produce any trades?)")
    x_vals = [float(c) for c in grid.columns]            # parameter values (X)
    periods = [p.strftime("%Y-%m") for p in grid.index]  # date windows (Z-as-axis label)
    y_idx = list(range(len(periods)))
    z = grid.values                                      # outcome height, rows=period × cols=param

    fig = go.Figure(go.Surface(
        x=x_vals, y=y_idx, z=z, colorscale="RdYlGn", cmid=0,
        colorbar=dict(title=metric_label),
        hovertemplate=(f"{param_label}=%{{x}}<br>window=%{{customdata}}<br>"
                       f"{metric_label}=%{{z:,.0f}}<extra></extra>"),
        customdata=[[p] * len(x_vals) for p in periods],
    ))
    fig.update_layout(
        title=title or f"{metric_label} landscape — {param_label} × date-window",
        scene=dict(
            xaxis_title=param_label,
            yaxis=dict(title="date window", tickmode="array",
                       tickvals=y_idx[::max(1, len(y_idx) // 12)],
                       ticktext=periods[::max(1, len(periods) // 12)]),
            zaxis_title=metric_label,
        ),
        margin=dict(l=0, r=0, t=40, b=0), height=720,
    )
    Path(out_html).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_html, include_plotlyjs="cdn")
    return out_html


def robustness_2d(table: pd.DataFrame, param_label: str, out_html: str,
                  metric: str = "total_pnl", title: str = "") -> str:
    """All-time outcome (bars) + trade count (line) vs the parameter. table = SweepResult.table."""
    t = table.sort_values("param_value")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in t[metric]]
    fig.add_bar(x=t["param_value"], y=t[metric], marker_color=colors,
                name=metric, hovertemplate=f"{param_label}=%{{x}}<br>{metric}=%{{y:,.0f}}<extra></extra>")
    fig.add_scatter(x=t["param_value"], y=t["n_trades"], name="# trades", mode="lines+markers",
                    line=dict(color="#1f77b4", dash="dot"), secondary_y=True)
    fig.update_layout(title=title or f"{metric} & trade count vs {param_label} (all-time)",
                      xaxis_title=param_label, height=460, bargap=0.35)
    fig.update_yaxes(title_text=metric, secondary_y=False)
    fig.update_yaxes(title_text="# trades", secondary_y=True)
    Path(out_html).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_html, include_plotlyjs="cdn")
    return out_html
