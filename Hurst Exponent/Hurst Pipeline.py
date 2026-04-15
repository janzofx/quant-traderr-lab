"""
Hurst_Pipeline.py
==================
Project: Quant Trader Lab - Time Series Analysis
Author: quant.traderr (Instagram)
License: MIT

Description:
    Production-ready pipeline for Hurst Exponent analysis on market data.

    Estimates the Hurst exponent via Rescaled Range (R/S) analysis across
    multiple lookback windows to detect regime shifts between trending
    (H > 0.5) and mean-reverting (H < 0.5) behavior.

    Pipeline Steps:
    1.  **Data Acquisition**: Fetches historical data (BTC-USD) via yfinance.
    2.  **Hurst Estimation**: Computes rolling H via R/S analysis.
    3.  **Regime Detection**: Classifies periods as trending, random, or mean-reverting.

Dependencies:
    pip install numpy pandas yfinance matplotlib
"""

import time
import warnings
import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════
CONFIG = {
    # Data
    "TICKER":           "BTC-USD",
    "LOOKBACK_YEARS":   3,
    "INTERVAL":         "1d",

    # Hurst Estimation
    "RS_MIN_LAG":       10,
    "RS_MAX_LAG":       200,
    "ROLLING_WINDOWS":  [30, 60, 90, 120, 180, 252],
    "HEATMAP_WINDOWS":  list(range(20, 260, 10)),

    # Output
    "OUTPUT_IMAGE":     "Hurst_Output.png",
    "RESOLUTION":       (1920, 1080),
    "DPI":              100,

    # Theme (Bloomberg Dark)
    "BG":               "#0b0b0b",
    "PANEL_BG":         "#0e0e0e",
    "GRID":             "#1a1a1a",
    "TEXT":             "#ffffff",
    "TEXT_DIM":         "#aaaaaa",
    "ORANGE":           "#ff9500",
    "CYAN":             "#00f2ff",
    "MAGENTA":          "#ff1493",
    "YELLOW":           "#ffd400",
    "GREEN":            "#00ff41",
    "RED":              "#ff3050",
    "FONT":             "Arial",
}

# ═══════════════════════════════════════════════════════════════════
# UTILS
# ═══════════════════════════════════════════════════════════════════
def log(msg):
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}")


# ═══════════════════════════════════════════════════════════════════
# MODULE 1: DATA
# ═══════════════════════════════════════════════════════════════════
def fetch_market_data():
    """Fetch historical price data."""
    log(f"[Data] Fetching {CONFIG['TICKER']} ({CONFIG['LOOKBACK_YEARS']}y)...")

    try:
        data = yf.download(
            CONFIG["TICKER"],
            period=f"{CONFIG['LOOKBACK_YEARS']}y",
            interval=CONFIG["INTERVAL"],
            progress=False,
        )
        if isinstance(data.columns, pd.MultiIndex):
            data = data.xs(CONFIG["TICKER"], axis=1, level=1)

        if data.empty:
            raise ValueError("No data returned from yfinance.")

        prices = data["Close"].values.flatten()
        dates = data.index
        log_returns = np.diff(np.log(prices))

        log(f"[Data] Loaded {len(prices)} days of prices.")
        return prices, dates, log_returns

    except Exception as e:
        log(f"[Error] Data fetch failed: {e}")
        log("[Warning] Using synthetic random walk.")
        n = 756
        log_returns = np.random.normal(0.0003, 0.02, n)
        prices = 100 * np.exp(np.cumsum(np.r_[0, log_returns]))
        dates = pd.date_range(end=pd.Timestamp.today(), periods=n + 1, freq="D")
        return prices, dates, log_returns


# ═══════════════════════════════════════════════════════════════════
# MODULE 2: HURST ESTIMATION ENGINE
# ═══════════════════════════════════════════════════════════════════
def hurst_rs(series, min_lag=None, max_lag=None):
    """Estimate Hurst exponent via Rescaled Range (R/S) analysis."""
    min_lag = min_lag or CONFIG["RS_MIN_LAG"]
    max_lag = max_lag or min(CONFIG["RS_MAX_LAG"], len(series) // 2)

    if max_lag <= min_lag or len(series) < min_lag * 2:
        return np.nan

    lags = range(min_lag, max_lag)
    rs_values, valid_lags = [], []

    for lag in lags:
        n_chunks = max(1, len(series) // lag)
        chunks = np.array_split(series[:n_chunks * lag], n_chunks)
        chunk_rs = []

        for chunk in chunks:
            if len(chunk) < 2:
                continue
            mean = chunk.mean()
            dev = np.cumsum(chunk - mean)
            R = dev.max() - dev.min()
            S = chunk.std(ddof=1)
            if S > 0:
                chunk_rs.append(R / S)

        if chunk_rs:
            rs_values.append(np.mean(chunk_rs))
            valid_lags.append(lag)

    if len(valid_lags) < 3:
        return np.nan

    log_rs = np.log(rs_values)
    log_n = np.log(valid_lags)
    H = np.polyfit(log_n, log_rs, 1)[0]
    return np.clip(H, 0, 1)


def compute_rolling_hurst(log_returns, windows=None):
    """Compute rolling Hurst exponent for multiple window sizes."""
    windows = windows or CONFIG["ROLLING_WINDOWS"]
    results = {}

    for w in windows:
        log(f"[Hurst] Computing rolling H (window={w})...")
        n = len(log_returns)
        h_series = np.full(n, np.nan)

        for i in range(w, n):
            h_series[i] = hurst_rs(log_returns[i - w:i], min_lag=5, max_lag=w // 2)

        results[w] = h_series

    return results


def compute_hurst_heatmap(log_returns, windows=None):
    """Compute H across a grid of lookback windows for heatmap."""
    windows = windows or CONFIG["HEATMAP_WINDOWS"]
    n = len(log_returns)

    # Sample dates (every 5th day for speed)
    step = 5
    date_indices = list(range(max(windows), n, step))
    heatmap = np.full((len(windows), len(date_indices)), np.nan)

    log(f"[Hurst] Computing heatmap ({len(windows)} windows x {len(date_indices)} dates)...")

    for i, w in enumerate(windows):
        for j, idx in enumerate(date_indices):
            if idx >= w:
                heatmap[i, j] = hurst_rs(log_returns[idx - w:idx],
                                          min_lag=5, max_lag=w // 2)

    return heatmap, date_indices, windows


# ═══════════════════════════════════════════════════════════════════
# MODULE 3: VISUALIZATION
# ═══════════════════════════════════════════════════════════════════
def visualize(prices, dates, log_returns, rolling_hurst, heatmap_data):
    """Generate Bloomberg-dark static visualization."""
    log("[Visual] Generating static snapshot...")

    heatmap, hm_date_idx, hm_windows = heatmap_data

    fig = plt.figure(
        figsize=(CONFIG["RESOLUTION"][0] / CONFIG["DPI"],
                 CONFIG["RESOLUTION"][1] / CONFIG["DPI"]),
        dpi=CONFIG["DPI"],
        facecolor=CONFIG["BG"],
    )

    gs = GridSpec(3, 1, height_ratios=[1.2, 1.5, 1.8], hspace=0.35,
                  left=0.06, right=0.94, top=0.92, bottom=0.06)

    # ── Title ──
    fig.text(0.50, 0.97,
             f"HURST EXPONENT ANALYSIS: {CONFIG['TICKER']}",
             ha="center", va="center", fontsize=18, fontweight="bold",
             color=CONFIG["ORANGE"], fontfamily=CONFIG["FONT"])
    fig.text(0.50, 0.94,
             "Rescaled Range (R/S) Analysis  |  Rolling Windows  |  Regime Detection",
             ha="center", va="center", fontsize=10,
             color=CONFIG["TEXT_DIM"], fontfamily=CONFIG["FONT"])
    fig.text(0.98, 0.01, "@quant.traderr",
             ha="right", va="bottom", fontsize=9,
             color=CONFIG["TEXT_DIM"], fontfamily=CONFIG["FONT"], alpha=0.6)

    # ── Panel 1: Price ──
    ax1 = fig.add_subplot(gs[0])
    _style_axis(ax1)
    ax1.set_title("Price", color=CONFIG["TEXT_DIM"], fontsize=10,
                  fontfamily=CONFIG["FONT"], loc="left")
    ax1.plot(dates, prices, color=CONFIG["CYAN"], linewidth=1.0, alpha=0.9)
    ax1.set_xlim(dates[0], dates[-1])
    ax1.set_ylabel("USD", color=CONFIG["TEXT_DIM"], fontsize=9)

    # ── Panel 2: Rolling Hurst ──
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    _style_axis(ax2)
    ax2.set_title("Rolling Hurst Exponent", color=CONFIG["TEXT_DIM"],
                  fontsize=10, fontfamily=CONFIG["FONT"], loc="left")

    colors = [CONFIG["CYAN"], CONFIG["ORANGE"], CONFIG["MAGENTA"],
              CONFIG["YELLOW"], CONFIG["GREEN"], CONFIG["RED"]]
    for i, (w, h_series) in enumerate(rolling_hurst.items()):
        c = colors[i % len(colors)]
        ax2.plot(dates[1:], h_series, color=c, linewidth=1.0,
                 alpha=0.8, label=f"{w}d")

    # H = 0.5 reference line
    ax2.axhline(0.5, color=CONFIG["TEXT_DIM"], linewidth=0.8,
                linestyle="--", alpha=0.5)
    ax2.text(dates[1], 0.51, "Random Walk (H=0.5)",
             color=CONFIG["TEXT_DIM"], fontsize=7, fontfamily=CONFIG["FONT"])

    # Regime shading
    ax2.axhspan(0.5, 1.0, alpha=0.03, color=CONFIG["GREEN"])
    ax2.axhspan(0.0, 0.5, alpha=0.03, color=CONFIG["RED"])
    ax2.text(dates[-1], 0.85, "TRENDING", ha="right",
             color=CONFIG["GREEN"], fontsize=8, alpha=0.6)
    ax2.text(dates[-1], 0.15, "MEAN-REVERTING", ha="right",
             color=CONFIG["RED"], fontsize=8, alpha=0.6)

    ax2.set_ylim(0, 1)
    ax2.set_ylabel("H", color=CONFIG["TEXT_DIM"], fontsize=9)
    leg = ax2.legend(loc="upper left", fontsize=7,
                     facecolor=CONFIG["BG"], edgecolor=CONFIG["GRID"])
    for text in leg.get_texts():
        text.set_color(CONFIG["TEXT_DIM"])

    # ── Panel 3: Heatmap ──
    ax3 = fig.add_subplot(gs[2])
    _style_axis(ax3)
    ax3.set_title("Hurst Heatmap (Window x Time)", color=CONFIG["TEXT_DIM"],
                  fontsize=10, fontfamily=CONFIG["FONT"], loc="left")

    # Custom colormap: blue (mean-rev) -> grey (random) -> orange (trending)
    cmap = LinearSegmentedColormap.from_list("hurst_regime", [
        (0.0, CONFIG["RED"]),
        (0.5, CONFIG["TEXT_DIM"]),
        (1.0, CONFIG["GREEN"]),
    ])

    hm_dates = [dates[1:][i] for i in hm_date_idx if i < len(dates) - 1]
    im = ax3.imshow(
        heatmap, aspect="auto", cmap=cmap, vmin=0.2, vmax=0.8,
        extent=[0, len(hm_dates), hm_windows[0], hm_windows[-1]],
        origin="lower", interpolation="bilinear",
    )

    # X-axis date labels
    n_ticks = 8
    tick_idx = np.linspace(0, len(hm_dates) - 1, n_ticks, dtype=int)
    ax3.set_xticks([int(t) for t in tick_idx])
    ax3.set_xticklabels([hm_dates[int(t)].strftime("%Y-%m") for t in tick_idx],
                        rotation=30, fontsize=7)
    ax3.set_ylabel("Lookback Window (days)", color=CONFIG["TEXT_DIM"], fontsize=9)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax3, fraction=0.02, pad=0.02)
    cbar.set_label("H", color=CONFIG["TEXT_DIM"], fontsize=9)
    cbar.ax.tick_params(colors=CONFIG["TEXT_DIM"], labelsize=7)

    # ── Save ──
    out_path = os.path.join(os.path.dirname(__file__), CONFIG["OUTPUT_IMAGE"])
    fig.savefig(out_path, dpi=CONFIG["DPI"], facecolor=CONFIG["BG"])
    plt.close(fig)
    log(f"[Visual] Saved to {out_path}")


def _style_axis(ax):
    """Apply Bloomberg Dark styling to an axis."""
    ax.set_facecolor(CONFIG["PANEL_BG"])
    ax.tick_params(colors=CONFIG["TEXT_DIM"], labelsize=8)
    ax.grid(True, color=CONFIG["GRID"], linewidth=0.3, alpha=0.5)
    for spine in ax.spines.values():
        spine.set_color(CONFIG["GRID"])
        spine.set_linewidth(0.5)


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    log("=== HURST EXPONENT PIPELINE ===")

    # 1. Data
    prices, dates, log_returns = fetch_market_data()

    # 2. Rolling Hurst
    rolling = compute_rolling_hurst(log_returns)

    # Current H values
    log("=== CURRENT HURST VALUES ===")
    for w, h_series in rolling.items():
        current_h = h_series[~np.isnan(h_series)][-1] if np.any(~np.isnan(h_series)) else np.nan
        regime = "TRENDING" if current_h > 0.55 else "MEAN-REVERTING" if current_h < 0.45 else "RANDOM WALK"
        log(f"  {w:>3}d window: H = {current_h:.3f}  [{regime}]")

    # 3. Heatmap
    heatmap_data = compute_hurst_heatmap(log_returns)

    # 4. Visualization
    visualize(prices, dates, log_returns, rolling, heatmap_data)

    log("=== PIPELINE FINISHED ===")


if __name__ == "__main__":
    main()
