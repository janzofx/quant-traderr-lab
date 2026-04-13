"""
RL_Trader_Pipeline.py
=====================
3D visualization of a tabular Q-Learning agent that learns to trade
a mean-reverting (Ornstein-Uhlenbeck) market.

State space  : (price_bin, position)   - 10 price bins x 3 positions {short, flat, long}
Action space : {HOLD, BUY, SELL}
Reward       : position * dPrice  (mark-to-market)
Update       : Q[s, a] <- Q[s, a] + alpha * (r + gamma * max_a' Q[s', a'] - Q[s, a])

Left panel  : 3D V*(s) = max_a Q(s, a) bar chart, colored by argmax action (policy)
Right panel : 4-stack 2D — Episode reward · Cumulative PnL · Epsilon decay · Action mix

Pipeline: TRAIN Q-LEARNING -> RENDER
Resolution: 1920x1080
Dependencies: pip install numpy matplotlib
"""

import os, time, warnings
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════
CONFIG = {
    "RESOLUTION": (1920, 1080),
    "OUTPUT_IMAGE": "RL_Trader_Output.png",
    "LOG_FILE": "rl_pipeline.log",
}

# Trading environment + Q-learning hyperparameters
RL = {
    "n_bins":     10,        # price bins
    "n_pos":      3,         # positions: -1, 0, +1
    "n_actions":  3,         # HOLD=0, BUY=1, SELL=2
    "T":          120,       # steps per episode
    "n_episodes": 600,
    "n_snaps":    140,       # Q-table snapshots saved

    # OU market params
    "mean":       100.0,
    "theta":      0.10,      # mean-reversion speed
    "sigma":      1.5,
    "S0":         100.0,
    "bin_half_range": 8.0,   # price bins span [mean - 8, mean + 8]

    # Q-learning
    "alpha":      0.18,
    "gamma":      0.96,
    "eps_start":  1.00,
    "eps_min":    0.05,

    "seed":       17,
}

# Action / position labels
ACTIONS = ["HOLD", "BUY", "SELL"]
POSITIONS = ["SHORT", "FLAT", "LONG"]

# ─── THEME ────────────────────────────────────────────────────────
THEME = {
    "BG":         "#000000",
    "PANEL_BG":   "#0a0a0a",
    "GRID":       "#222222",
    "GRID_ALT":   "#1f1f1f",
    "SPINE":      "#333333",
    "TEXT":       "#ffffff",
    "TEXT_DIM":   "#aaaaaa",
    "TEXT_SEC":   "#c0c0c0",
    "ORANGE":     "#ff9500",
    "YELLOW":     "#ffd400",
    "CYAN":       "#00f2ff",
    "GREEN":      "#00ff7f",
    "RED":        "#ff3050",
    "PINK":       "#ff2a9e",
    "BLUE":       "#00bfff",
    "FONT":       "Arial",
}

# Action colors (BUY = green, SELL = red, HOLD = cyan)
ACTION_COLORS = ["#00f2ff", "#00ff7f", "#ff3050"]


# ═══════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════

def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    try:
        with open(CONFIG["LOG_FILE"], "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def _hex_to_rgba(hexc, alpha=1.0):
    h = hexc.lstrip("#")
    return (int(h[0:2], 16) / 255,
            int(h[2:4], 16) / 255,
            int(h[4:6], 16) / 255,
            alpha)


# ═══════════════════════════════════════════════════════════════════
# MODULE 1 — ENV + Q-LEARNING
# ═══════════════════════════════════════════════════════════════════

def _price_bin(price):
    """Discretize price into bins."""
    p = RL
    edge = p["bin_half_range"]
    rel = (price - p["mean"]) / (2 * edge) + 0.5
    b = int(np.clip(rel * p["n_bins"], 0, p["n_bins"] - 1))
    return b


def step_env(price, position, action, rng):
    """Take an action; return (new_price, new_position, reward)."""
    p = RL
    new_pos = position
    if action == 1 and position < 1:           # BUY
        new_pos = position + 1
    elif action == 2 and position > -1:        # SELL
        new_pos = position - 1

    # OU update
    new_price = price + p["theta"] * (p["mean"] - price) + p["sigma"] * rng.standard_normal()

    # mark-to-market reward = position-after-trade * price change
    reward = new_pos * (new_price - price)
    return new_price, new_pos, reward


def train_q_learning():
    """Tabular Q-learning. Returns snapshots + training curves."""
    p = RL
    rng = np.random.default_rng(p["seed"])

    n_b, n_p, n_a = p["n_bins"], p["n_pos"], p["n_actions"]
    Q = np.zeros((n_b, n_p, n_a))

    eps_decay = (p["eps_min"] / p["eps_start"]) ** (1.0 / p["n_episodes"])
    epsilon = p["eps_start"]

    rewards          = np.zeros(p["n_episodes"])
    cum_rewards      = np.zeros(p["n_episodes"])
    eps_hist         = np.zeros(p["n_episodes"])
    action_counts    = np.zeros((p["n_episodes"], 3), dtype=int)

    save_every = max(p["n_episodes"] // p["n_snaps"], 1)
    snaps = []
    snap_eps = []
    snap_rewards = []     # (running mean reward at snapshot)

    running_total = 0.0
    for ep in range(p["n_episodes"]):
        price = p["S0"] + rng.standard_normal() * 0.5
        position = 0    # flat
        ep_reward = 0.0
        ep_actions = [0, 0, 0]

        for t in range(p["T"]):
            b = _price_bin(price)
            pi = position + 1     # 0,1,2

            # epsilon-greedy
            if rng.random() < epsilon:
                a = rng.integers(n_a)
            else:
                a = int(np.argmax(Q[b, pi]))

            new_price, new_position, reward = step_env(price, position, a, rng)
            ep_reward += reward
            ep_actions[a] += 1

            new_b = _price_bin(new_price)
            new_pi = new_position + 1
            target = reward + p["gamma"] * Q[new_b, new_pi].max()
            Q[b, pi, a] += p["alpha"] * (target - Q[b, pi, a])

            price, position = new_price, new_position

        rewards[ep]       = ep_reward
        running_total    += ep_reward
        cum_rewards[ep]   = running_total
        eps_hist[ep]      = epsilon
        action_counts[ep] = ep_actions
        epsilon          *= eps_decay

        if ep % save_every == 0 or ep == p["n_episodes"] - 1:
            snaps.append(Q.copy())
            snap_eps.append(epsilon)
            snap_rewards.append(rewards[max(0, ep - 9):ep + 1].mean())

    return dict(
        Q_stack=np.array(snaps),         # (n_snaps, n_bins, n_pos, n_actions)
        rewards=rewards,
        cum_rewards=cum_rewards,
        eps_hist=eps_hist,
        action_counts=action_counts,
        snap_eps=np.array(snap_eps),
        snap_rewards=np.array(snap_rewards),
        n_snaps=len(snaps),
    )


# ═══════════════════════════════════════════════════════════════════
# MODULE 2 — RENDERING
# ═══════════════════════════════════════════════════════════════════

def _style_2d(ax, xlabel=False):
    ax.set_facecolor(THEME["PANEL_BG"])
    for sp in ax.spines.values():
        sp.set_color(THEME["SPINE"]); sp.set_linewidth(0.6)
    ax.tick_params(axis="both", colors=THEME["TEXT_DIM"], labelsize=9,
                   direction="in", length=4)
    ax.yaxis.grid(True, lw=0.3, alpha=0.4, color=THEME["GRID_ALT"])
    ax.xaxis.grid(False)
    if not xlabel:
        ax.tick_params(axis="x", labelbottom=False)


def render_static(train, yr, out_path):
    try:
        elev, azim, dist, salpha = 34, 260, 1.15, 0.95

        Q_stack = train["Q_stack"]      # (n_snaps, n_b, n_p, n_a)
        n_snaps = Q_stack.shape[0]
        snap_idx = n_snaps - 1          # fully trained Q-table

        save_every = max(RL["n_episodes"] // RL["n_snaps"], 1)
        cur_ep = RL["n_episodes"] - 1
        ep_slice = slice(0, cur_ep + 1)

        Q_now = Q_stack[snap_idx]                      # (n_b, n_p, n_a)
        V_star = Q_now.max(axis=2)                     # (n_b, n_p)  value func
        argmax_a = Q_now.argmax(axis=2)                # (n_b, n_p)  policy

        # ── figure ────────────────────────────────────────────────
        fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor=THEME["BG"])
        gs = gridspec.GridSpec(
            4, 2, width_ratios=[1.15, 1],
            height_ratios=[1.2, 1, 1, 1.2],
            hspace=0.30, wspace=0.24,
            left=0.04, right=0.97, top=0.86, bottom=0.06,
        )

        # ═══ LEFT: 3D V*(s) bars colored by argmax policy ═════════
        ax = fig.add_subplot(gs[:, 0], projection="3d", computed_zorder=False)
        ax.set_facecolor(THEME["BG"])
        pane = (0.02, 0.02, 0.02, 1)
        ax.xaxis.set_pane_color(pane)
        ax.yaxis.set_pane_color(pane)
        ax.zaxis.set_pane_color(pane)
        for a in (ax.xaxis, ax.yaxis, ax.zaxis):
            a._axinfo["grid"]["color"]     = (0.13, 0.13, 0.13, 0.55)
            a._axinfo["grid"]["linewidth"] = 0.4

        n_b, n_p = RL["n_bins"], RL["n_pos"]

        # bar grid
        xs, ys = np.meshgrid(np.arange(n_b), np.arange(n_p), indexing="xy")
        xs = xs.flatten().astype(float)
        ys = ys.flatten().astype(float)
        zs = np.zeros(xs.size)
        dx = 0.65; dy = 0.55
        xs -= dx / 2.0
        ys -= dy / 2.0
        dz = V_star.T.flatten()                         # (n_p, n_b) flatten
        argmax_flat = argmax_a.T.flatten()

        bar_colors = []
        bar_edges  = []
        for k in range(xs.size):
            base = ACTION_COLORS[argmax_flat[k]]
            bar_colors.append(_hex_to_rgba(base, salpha))
            bar_edges.append(_hex_to_rgba(THEME["YELLOW"], 0.4))

        # avoid clipping when all Q's are still ~0
        bar_top_pad = max(abs(dz).max() * 1.20, 0.2)

        ax.bar3d(xs, ys, zs, dx, dy, dz,
                 color=bar_colors, edgecolor=bar_edges, linewidth=0.6,
                 shade=True, zorder=1)

        # action label colors at the top of each bar
        for k in range(xs.size):
            if dz[k] > 0.05 * bar_top_pad:
                ax.text(xs[k] + dx / 2, ys[k] + dy / 2, dz[k] + bar_top_pad * 0.04,
                        ACTIONS[argmax_flat[k]][0],   # H/B/S
                        color=THEME["TEXT"], fontsize=8, ha="center", va="bottom",
                        fontweight="bold", fontfamily=THEME["FONT"])

        ax.set_xticks(np.arange(n_b))
        ax.set_xticklabels([f"{i}" for i in range(n_b)],
                           color=THEME["TEXT_DIM"], fontsize=8)
        ax.set_yticks(np.arange(n_p))
        ax.set_yticklabels(POSITIONS, color=THEME["TEXT_DIM"], fontsize=9)
        ax.set_xlabel("PRICE BIN", fontsize=11, fontweight="bold",
                      color=THEME["TEXT_DIM"], labelpad=14, fontfamily=THEME["FONT"])
        ax.set_ylabel("POSITION", fontsize=11, fontweight="bold",
                      color=THEME["TEXT_DIM"], labelpad=14, fontfamily=THEME["FONT"])
        ax.set_zlabel(r"$V^{*}(s)$", fontsize=12, fontweight="bold",
                      color=THEME["TEXT_DIM"], labelpad=10, fontfamily=THEME["FONT"])
        ax.tick_params(axis="z", colors=THEME["TEXT_DIM"], labelsize=8)

        ax.set_xlim(-0.5, n_b - 0.5)
        ax.set_ylim(-0.5, n_p - 0.5)
        ax.set_zlim(min(0, dz.min() * 1.15), bar_top_pad)
        ax.set_box_aspect([1.55 * dist, 1.0 * dist, 0.85 * dist])
        ax.view_init(elev=elev, azim=azim)
        ax.set_title("LEARNED POLICY:  cyan=HOLD  green=BUY  red=SELL",
                     fontsize=12, fontweight="bold",
                     color=THEME["YELLOW"], fontfamily=THEME["FONT"], pad=8)

        # ═══ RIGHT: 4 panels ══════════════════════════════════════
        ep_arr = np.arange(RL["n_episodes"])

        # Panel 1 — Episode reward + running mean
        a1 = fig.add_subplot(gs[0, 1])
        _style_2d(a1)
        a1.plot(ep_arr[ep_slice], train["rewards"][ep_slice],
                color=THEME["BLUE"], lw=0.6, alpha=0.55)
        w = 20
        r = train["rewards"][ep_slice]
        kernel = np.ones(min(w, len(r))) / min(w, len(r))
        run = np.convolve(r, kernel, mode="same")
        a1.plot(ep_arr[ep_slice], run, color=THEME["CYAN"], lw=1.4)
        a1.axhline(0, color=THEME["SPINE"], lw=0.5, alpha=0.6)
        a1.set_xlim(0, RL["n_episodes"])
        a1.set_ylim(*yr["reward"])
        a1.set_title("Episode reward   (running mean cyan)",
                     fontsize=11, fontweight="bold",
                     color=THEME["TEXT_SEC"], loc="left", pad=4)

        # Panel 2 — Cumulative reward
        a2 = fig.add_subplot(gs[1, 1])
        _style_2d(a2)
        a2.plot(ep_arr[ep_slice], train["cum_rewards"][ep_slice],
                color=THEME["GREEN"], lw=1.1)
        a2.fill_between(ep_arr[ep_slice], 0, train["cum_rewards"][ep_slice],
                        where=train["cum_rewards"][ep_slice] >= 0,
                        color=THEME["GREEN"], alpha=0.12)
        a2.fill_between(ep_arr[ep_slice], 0, train["cum_rewards"][ep_slice],
                        where=train["cum_rewards"][ep_slice] < 0,
                        color=THEME["RED"], alpha=0.12)
        a2.axhline(0, color=THEME["SPINE"], lw=0.5, alpha=0.6)
        a2.set_xlim(0, RL["n_episodes"])
        a2.set_ylim(*yr["cum"])
        a2.set_title("Cumulative reward",
                     fontsize=11, fontweight="bold",
                     color=THEME["TEXT_SEC"], loc="left", pad=4)

        # Panel 3 — Epsilon decay
        a3 = fig.add_subplot(gs[2, 1])
        _style_2d(a3)
        a3.plot(ep_arr[ep_slice], train["eps_hist"][ep_slice],
                color=THEME["PINK"], lw=1.1)
        a3.fill_between(ep_arr[ep_slice], 0, train["eps_hist"][ep_slice],
                        color=THEME["PINK"], alpha=0.12)
        a3.set_xlim(0, RL["n_episodes"]); a3.set_ylim(0, 1.05)
        a3.set_title(r"$\epsilon$  (exploration schedule)",
                     fontsize=11, fontweight="bold",
                     color=THEME["TEXT_SEC"], loc="left", pad=4)

        # Panel 4 — Action mix (stacked area)
        a4 = fig.add_subplot(gs[3, 1])
        _style_2d(a4, xlabel=True)
        ac = train["action_counts"][ep_slice].astype(float)
        ac /= ac.sum(axis=1, keepdims=True) + 1e-12
        w = 10
        kk = np.ones(min(w, len(ac))) / min(w, len(ac))
        ac_s = np.column_stack([np.convolve(ac[:, i], kk, mode="same")
                                for i in range(3)])
        a4.stackplot(ep_arr[ep_slice],
                     ac_s[:, 0], ac_s[:, 1], ac_s[:, 2],
                     colors=ACTION_COLORS, alpha=0.78,
                     edgecolor="none",
                     labels=ACTIONS)
        a4.set_xlim(0, RL["n_episodes"]); a4.set_ylim(0, 1)
        a4.set_xlabel("Episode", fontsize=10,
                       color=THEME["TEXT_DIM"], fontfamily=THEME["FONT"])
        a4.set_title("Action mix per episode",
                     fontsize=11, fontweight="bold",
                     color=THEME["TEXT_SEC"], loc="left", pad=4)

        # ═══ Title bar ════════════════════════════════════════════
        fig.text(0.50, 0.955, "REINFORCEMENT  LEARNING  TRADER",
                 ha="center", va="center", fontsize=26, fontweight="bold",
                 color=THEME["ORANGE"], fontfamily=THEME["FONT"])
        fig.text(0.50, 0.918,
                 r"$Q(s, a) \;\leftarrow\; Q(s, a)"
                 r" \;+\; \alpha\,[\, r"
                 r" + \gamma \,\max_{a'} Q(s', a') \;-\; Q(s, a)\,]$",
                 ha="center", va="center", fontsize=14,
                 color=THEME["TEXT"], fontfamily=THEME["FONT"])
        fig.text(0.50, 0.886,
                 r"OU mean-revert market    "
                 r"$\alpha = 0.18$    $\gamma = 0.96$    "
                 r"$\epsilon : 1.0 \to 0.05$    "
                 r"600 episodes  ${\times}$  120 steps",
                 ha="center", va="center", fontsize=10, fontweight="bold",
                 color=THEME["TEXT_DIM"], fontfamily=THEME["FONT"])

        # ═══ HUD ══════════════════════════════════════════════════
        cur_eps = train["eps_hist"][cur_ep]
        cur_cum = train["cum_rewards"][cur_ep]
        last_mean = train["rewards"][max(0, cur_ep - 19):cur_ep + 1].mean()
        fig.text(0.97, 0.875,
                 f"ep {cur_ep:3d}/{RL['n_episodes']}    "
                 f"epsilon = {cur_eps:4.2f}    "
                 f"avg reward (20) = {last_mean:+5.1f}    "
                 f"cum = {cur_cum:+7.1f}",
                 ha="right", va="center", fontsize=11, fontweight="bold",
                 color=THEME["YELLOW"], fontfamily=THEME["FONT"])

        # ═══ Footer ═══════════════════════════════════════════════
        fig.text(0.98, 0.012, "@quant.traderr",
                 ha="right", va="bottom", fontsize=10,
                 color=THEME["TEXT_DIM"], fontfamily=THEME["FONT"], alpha=0.6)

        fig.savefig(out_path, dpi=100, facecolor=THEME["BG"])
        plt.close(fig)
        log(f"Saved image to {out_path}")
        return True

    except Exception:
        import traceback
        traceback.print_exc()
        return False


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    log("=" * 60)
    log("RL Trader Pipeline START")
    log("=" * 60)

    log("Training tabular Q-learning agent ...")
    train = train_q_learning()
    final_avg = train["rewards"][-50:].mean()
    early_avg = train["rewards"][:50].mean()
    log(f"  episodes: {RL['n_episodes']}  snapshots: {train['n_snaps']}")
    log(f"  avg reward first 50:  {early_avg:+.2f}")
    log(f"  avg reward last 50:   {final_avg:+.2f}")
    log(f"  total cumulative:      {train['cum_rewards'][-1]:+.2f}")

    yr = {
        "reward": (train["rewards"].min() * 1.1, train["rewards"].max() * 1.1),
        "cum":    (min(0, train["cum_rewards"].min()) * 1.1,
                   max(0, train["cum_rewards"].max()) * 1.1),
    }

    log("Rendering static visualization ...")
    render_static(train, yr, CONFIG["OUTPUT_IMAGE"])

    log(f"Pipeline complete in {time.time() - t0:.1f}s")
    log("=" * 60)


if __name__ == "__main__":
    main()
