"""
HRP_Pipeline.py
===============
Project: Quant Trader Lab - Hierarchical Risk Parity 3D Visualization
Author: quant.traderr (Instagram)
License: MIT

Description:
    A production-ready pipeline visualizing Hierarchical Risk Parity (HRP)
    portfolio optimization. Features a 3D surface of sorted cumulative
    returns across assets over time — the "return landscape."

    HRP Algorithm (Marcos Lopez de Prado, 2016):
    1. Compute correlation/distance matrix from asset returns.
    2. Hierarchical clustering (single-linkage) to build a dendrogram.
    3. Quasi-diagonalize the covariance matrix via dendrogram ordering.
    4. Recursive bisection to allocate weights by inverse variance.

    Pipeline Steps:
    1. DATA — Fetch multi-asset data, compute HRP weights & sorted cum returns.
    2. VISUALIZATION — Static 3D Plotly surface snapshot.

    NOTE: Video rendering has been removed for pipeline efficiency.

Dependencies:
    pip install numpy pandas yfinance plotly kaleido scipy

Usage:
    python HRP_Pipeline.py
"""

import os
import time
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

# --- CONFIGURATION ---

CONFIG = {
    "ASSETS": [
        "BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "AVAX-USD",
        "LINK-USD", "DOT-USD", "MATIC-USD", "UNI-USD", "ATOM-USD",
        "NEAR-USD", "FTM-USD", "ALGO-USD", "XLM-USD", "AAVE-USD",
        "DOGE-USD", "SHIB-USD", "LTC-USD", "BCH-USD", "XRP-USD",
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
        "META", "TSLA", "JPM", "V", "UNH",
    ],
    "PERIOD": "1y",
    "INTERVAL": "1d",
    "RESOLUTION": (1920, 1080),
    "OUTPUT_IMAGE": "hrp_static.png",
}

THEME = {
    "BG": "#0b0b0b",
    "GRID": "#1a1a1a",
    "ACCENT": "#00f2ff",
    "TEXT": "#ffffff",
    "TEXT_MUTED": "#888888",
}

# --- UTILS ---

def log(msg):
    """Simple logger."""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}")

# --- MODULE 1: DATA ---

def _hrp_sort(link, n):
    """Extract leaf ordering from linkage matrix (quasi-diagonalization)."""
    return list(leaves_list(link))


def _hrp_weights(cov, sort_idx):
    """Recursive bisection to assign weights by inverse cluster variance."""
    w = pd.Series(1.0, index=sort_idx)
    cluster_items = [sort_idx]

    while len(cluster_items) > 0:
        next_clusters = []
        for subset in cluster_items:
            if len(subset) <= 1:
                continue
            mid = len(subset) // 2
            left = subset[:mid]
            right = subset[mid:]

            cov_left = cov.iloc[left, left]
            cov_right = cov.iloc[right, right]

            var_left = _get_cluster_var(cov_left)
            var_right = _get_cluster_var(cov_right)

            alpha = 1.0 - var_left / (var_left + var_right + 1e-16)

            w.iloc[left] *= alpha
            w.iloc[right] *= (1.0 - alpha)

            if len(left) > 1:
                next_clusters.append(left)
            if len(right) > 1:
                next_clusters.append(right)

        cluster_items = next_clusters

    return w / w.sum()


def _get_cluster_var(cov):
    """Inverse-variance portfolio variance for a cluster."""
    ivp = 1.0 / np.diag(cov)
    ivp /= ivp.sum()
    return float(np.dot(ivp, np.dot(cov, ivp)))


def fetch_and_process_data():
    """
    Fetches multi-asset data, computes HRP weights, and returns:
    - sorted_cum_pct: DataFrame of cumulative % change sorted by final return
    - weights: HRP allocation weights
    - tickers: list of ticker names in sorted order
    """
    log("[Data] Fetching multi-asset data...")

    tickers = CONFIG["ASSETS"]

    try:
        raw = yf.download(tickers, period=CONFIG["PERIOD"],
                          interval=CONFIG["INTERVAL"], progress=False)
    except Exception as e:
        log(f"[Error] Download failed: {e}")
        return None, None, None

    if raw.empty:
        log("[Error] No data returned.")
        return None, None, None

    # Extract Close prices
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        close = raw[["Close"]]

    close = close.dropna(axis=1, how="all").ffill().dropna()
    actual_tickers = list(close.columns)
    n_assets = len(actual_tickers)
    log(f"[Data] Got {n_assets} assets, {len(close)} days.")

    if n_assets < 5:
        log("[Error] Too few assets survived filtering.")
        return None, None, None

    # --- HRP Algorithm ---
    returns = close.pct_change().dropna()
    corr = returns.corr()
    cov = returns.cov()

    # 1. Distance matrix from correlation
    dist = ((1 - corr) / 2.0) ** 0.5
    np.fill_diagonal(dist.values, 0)
    condensed = squareform(dist.values, checks=False)

    # 2. Hierarchical clustering
    link = linkage(condensed, method="single")
    sort_idx = _hrp_sort(link, n_assets)

    # 3. HRP weights
    weights = _hrp_weights(cov, sort_idx)
    weights.index = [actual_tickers[i] for i in sort_idx]

    log("[Data] HRP weights computed:")
    for t, w in weights.items():
        log(f"[Data]   {t:<12s} {w:.4f}")

    # 4. Sorted cumulative percentage change (for 3D surface)
    cum_ret = (1 + returns).cumprod() - 1
    cum_pct = cum_ret * 100

    # Sort columns by final cumulative return (descending) — creates the mountain shape
    final_returns = cum_pct.iloc[-1].sort_values(ascending=False)
    sorted_cols = final_returns.index.tolist()
    sorted_cum_pct = cum_pct[sorted_cols]

    log(f"[Data] Surface: {sorted_cum_pct.shape[0]} time steps x {sorted_cum_pct.shape[1]} assets")
    return sorted_cum_pct, weights, sorted_cols


# --- MODULE 2: STATIC VISUALIZATION ---

def visualize(surface_df, weights, tickers):
    """Generates a static 3D surface snapshot using Plotly."""
    log("[Visual] Generating static 3D snapshot...")

    surface_data = surface_df.values.astype(np.float64)
    n_assets = surface_data.shape[1]
    n_time = surface_data.shape[0]

    x = np.arange(n_assets)
    y = np.arange(n_time)
    X, Y = np.meshgrid(x, y)

    fig = go.Figure()

    # Main surface
    fig.add_trace(go.Surface(
        x=X, y=Y, z=surface_data,
        colorscale=[
            [0.0,  "#0033cc"],
            [0.15, "#0066ff"],
            [0.3,  "#00ccff"],
            [0.45, "#00ff88"],
            [0.55, "#aaff00"],
            [0.7,  "#ffcc00"],
            [0.85, "#ff6600"],
            [1.0,  "#cc0000"],
        ],
        showscale=False,
        opacity=0.95,
        lighting=dict(ambient=0.5, diffuse=0.6, specular=0.3, roughness=0.5),
        contours=dict(
            z=dict(show=True, color="#ffffff", width=1, usecolormap=False,
                   start=0, end=float(np.nanmax(surface_data)), size=2),
        ),
    ))

    # Zero-plane
    fig.add_trace(go.Surface(
        x=X, y=Y, z=np.zeros_like(surface_data),
        colorscale=[[0, "#1a1a1a"], [1, "#1a1a1a"]],
        showscale=False,
        opacity=0.25,
    ))

    # Camera — dramatic 3/4 view
    camera_eye = dict(x=1.6, y=-1.8, z=0.7)

    fig.update_layout(
        title=dict(
            text=(
                f"<b>HIERARCHICAL RISK PARITY</b><br>"
                f"<span style='font-size:18px;color:#888'>"
                f"Sorted Cumulative % Change  |  {n_assets} Assets  |  {n_time} Days</span>"
            ),
            font=dict(family="Roboto Mono, Consolas, monospace", size=28, color="white"),
            y=0.93, x=0.5, xanchor="center",
        ),
        width=CONFIG["RESOLUTION"][0],
        height=CONFIG["RESOLUTION"][1],
        scene=dict(
            xaxis=dict(
                title="Asset",
                showgrid=True, gridcolor=THEME["GRID"],
                backgroundcolor=THEME["BG"], color="#aaaaaa",
                showticklabels=False,
                range=[0, n_assets - 1],
            ),
            yaxis=dict(
                title="Days",
                showgrid=True, gridcolor=THEME["GRID"],
                backgroundcolor=THEME["BG"], color="#aaaaaa",
                range=[0, n_time],
            ),
            zaxis=dict(
                title="Cumulative Return %",
                showgrid=True, gridcolor=THEME["GRID"],
                backgroundcolor=THEME["BG"], color="#aaaaaa",
                range=[float(np.nanmin(surface_data)) - 2,
                       float(np.nanmax(surface_data)) + 2],
            ),
            bgcolor=THEME["BG"],
            camera=dict(
                eye=camera_eye,
                center=dict(x=0, y=0, z=-0.15),
                up=dict(x=0, y=0, z=1),
            ),
            aspectratio=dict(x=1.2, y=1.8, z=0.7),
        ),
        paper_bgcolor=THEME["BG"],
        margin=dict(l=0, r=0, b=0, t=100),
        showlegend=False,
        annotations=[
            dict(
                text="@quant.traderr",
                xref="paper", yref="paper",
                x=0.5, y=0.01,
                showarrow=False,
                font=dict(size=12, color="#2a2a2a",
                          family="Roboto Mono, Consolas, monospace"),
            )
        ],
    )

    # Save
    out_path = os.path.join(os.path.dirname(__file__), CONFIG["OUTPUT_IMAGE"])
    fig.write_image(out_path)
    log(f"[Visual] Saved to {out_path}")


# --- MAIN ---

def main():
    log("=== HRP PIPELINE ===")

    # 1. Data & HRP
    surface_df, weights, sorted_tickers = fetch_and_process_data()

    if surface_df is not None:
        # 2. Visualize
        visualize(surface_df, weights, sorted_tickers)

    log("=== PIPELINE FINISHED ===")


if __name__ == "__main__":
    main()
