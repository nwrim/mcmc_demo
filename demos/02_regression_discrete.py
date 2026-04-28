"""Discrete linear regression — MCMC over a 10-point β grid.

Extends the island demo: instead of 10 physical islands, the 10 "islands"
are β values on a grid [0.40 ... 0.85]. "Size of an island" = posterior
density (∝ likelihood under a flat prior). α = 0 and σ = 0.25 are known.

module load ffmpeg/5.1

Outputs (in ../out/):
    02_regression_discrete_data.png          scatter + true line (for "show the data" slide)
    02_regression_discrete_lines.png         scatter + 10 candidate lines ("the 10 islands")
    02_regression_discrete_step00..02.png    first few steps as standalone figures
    02_regression_discrete.mp4               animation of the first N_ANIM_FRAMES iterations
    02_regression_discrete_trace.png         full N_STEPS trace
    02_regression_discrete_convergence.png   histograms vs grid posterior at snapshots
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from samplers import metropolis

OUT = Path(__file__).resolve().parent.parent / "out"

# --- data-generating model ---
N_DATA = 100
TRUE_BETA = 0.6
SIGMA = 0.25
DATA_SEED = 1422

# --- β grid (the "10 islands") ---
N_BETAS = 10
BETA_MIN, BETA_MAX = 0.40, 0.85
betas = np.linspace(BETA_MIN, BETA_MAX, N_BETAS)
beta_indices = np.arange(N_BETAS)

# --- sampler settings ---
N_STEPS = 10000
N_ANIM_FRAMES = 1000
FPS = 60
N_WALKTHROUGH = 3
SNAPSHOT_STEPS = [50, 500, N_ANIM_FRAMES, N_STEPS]
MCMC_SEED = 20260418
INITIAL_INDEX = 8   # β = 0.80, deliberately away from the true 0.6

# --- simulate data ---
data_rng = np.random.default_rng(DATA_SEED)
x = data_rng.uniform(-1, 1, N_DATA)
y = TRUE_BETA * x + data_rng.normal(0, SIGMA, N_DATA)


def log_likelihood_at_index(idx):
    beta = betas[int(idx)]
    return float(np.sum(norm.logpdf(y, loc=beta * x, scale=SIGMA)))


# Grid-approx posterior under a flat prior: normalize likelihoods on the grid.
_log_liks = np.array([log_likelihood_at_index(i) for i in beta_indices])
_log_liks -= _log_liks.max()
grid_posterior = np.exp(_log_liks)
grid_posterior /= grid_posterior.sum()


def log_prob_on_grid(idx):
    i = int(idx)
    if i < 0 or i >= N_BETAS:
        return -np.inf
    return log_likelihood_at_index(i)


def adjacent_proposal(state, rng):
    """Left/right neighbor on the β grid. Out-of-bounds proposals are
    passed through unchanged; log_prob_on_grid returns -inf there, so the
    sampler rejects and the king stays — same 'rejection = staying' rule
    as the island demo, just without the (unphysical) ring wrap-around.
    """
    direction = 1 if rng.uniform() < 0.5 else -1
    return int(state) + direction


def visit_fractions(prefix):
    counts = np.bincount(prefix, minlength=N_BETAS)
    total = counts.sum()
    return counts / total if total > 0 else counts.astype(float)


def write_data_png():
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x, y, s=14, color="tab:green", alpha=0.7)
    xline = np.array([-1.0, 1.0])
    ax.plot(xline, TRUE_BETA * xline, "k--", linewidth=1.5,
            label=f"true line (β = {TRUE_BETA})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.1, 1.1)
    ax.legend(loc="upper left")
    fig.tight_layout()
    path = OUT / "02_regression_discrete_data.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def write_lines_png():
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x, y, s=14, color="tab:green", alpha=0.6)
    xline = np.array([-1.0, 1.0])
    for b in betas:
        ax.plot(xline, b * xline, color="steelblue", alpha=0.7, linewidth=1)
        ax.annotate(f"{b:.2f}", xy=(1.02, b), fontsize=9, va="center")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(-1.05, 1.2)
    ax.set_ylim(-1.1, 1.1)
    fig.tight_layout()
    path = OUT / "02_regression_discrete_lines.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def build_figure():
    fig = plt.figure(figsize=(15, 4.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.2, 1.0],
                          wspace=0.35, top=0.78, bottom=0.15)
    ax_lines = fig.add_subplot(gs[0, 0])
    ax_trace = fig.add_subplot(gs[0, 1])
    ax_hist = fig.add_subplot(gs[0, 2])

    # --- lines panel ---
    ax_lines.scatter(x, y, s=10, color="tab:green", alpha=0.55, zorder=1)
    xline = np.array([-1.0, 1.0])
    line_artists = []
    for b in betas:
        ln, = ax_lines.plot(xline, b * xline, color="steelblue",
                            alpha=0.4, linewidth=1, zorder=2)
        line_artists.append(ln)
    ax_lines.set_xlim(-1.05, 1.25)
    ax_lines.set_ylim(-1.1, 1.1)
    ax_lines.set_xlabel("x")
    ax_lines.set_ylabel("y")
    beta_label = ax_lines.text(1.03, 0, "", fontsize=11,
                               family="monospace", va="center", color="tab:orange")
    prob_text = ax_lines.text(
        0.5, 1.14, "", transform=ax_lines.transAxes,
        ha="center", va="bottom", fontsize=12, family="monospace",
    )
    verdict_text = ax_lines.text(
        0.5, 1.03, "", transform=ax_lines.transAxes,
        ha="center", va="bottom", fontsize=13,
        family="monospace", fontweight="bold",
    )

    # --- trace panel ---
    trace_line, = ax_trace.plot([], [], "-", color="steelblue",
                                markersize=3, linewidth=1)
    trace_current, = ax_trace.plot([], [], "o", color="tab:orange",
                                   markersize=6, zorder=3)
    ax_trace.set_xlim(-1, 20)
    ax_trace.set_ylim(-0.3, N_BETAS - 0.7)
    ax_trace.set_yticks(beta_indices)
    ax_trace.set_yticklabels([f"{b:.2f}" for b in betas])
    ax_trace.set_xlabel("iteration")
    ax_trace.set_ylabel("β")
    ax_trace.set_title("β sampled")

    # --- histogram panel ---
    bars = ax_hist.bar(beta_indices, np.zeros(N_BETAS),
                       color="steelblue", alpha=0.85)
    ax_hist.plot(beta_indices, grid_posterior, "k--o",
                 linewidth=1.5, markersize=5, label="grid posterior")
    ax_hist.set_xticks(beta_indices)
    ax_hist.set_xticklabels([f"{b:.2f}" for b in betas], rotation=45)
    ax_hist.set_xlabel("β")
    ax_hist.set_ylabel("visit fraction")
    ax_hist.set_title("visit fraction vs grid posterior")
    ax_hist.set_ylim(0, max(0.6, grid_posterior.max() * 1.15))
    ax_hist.legend(loc="upper right")
    iter_text = ax_hist.text(0.98, 0.88, "", transform=ax_hist.transAxes,
                             ha="right", va="top")

    artists = dict(
        fig=fig, ax_lines=ax_lines, ax_trace=ax_trace, ax_hist=ax_hist,
        line_artists=line_artists, trace_line=trace_line,
        trace_current=trace_current, bars=bars,
        beta_label=beta_label, prob_text=prob_text, verdict_text=verdict_text,
        iter_text=iter_text, arrow=None,
    )
    return artists


def render_step(artists, step, chain, proposals, accepts, show_accept_text=True):
    current_idx = int(chain[step])
    current_beta = betas[current_idx]

    for i, ln in enumerate(artists["line_artists"]):
        if i == current_idx:
            ln.set_color("tab:orange")
            ln.set_linewidth(2.5)
            ln.set_alpha(1.0)
            ln.set_zorder(3)
        else:
            ln.set_color("steelblue")
            ln.set_linewidth(1)
            ln.set_alpha(0.35)
            ln.set_zorder(2)

    artists["beta_label"].set_position((1.03, current_beta))
    artists["beta_label"].set_text(f"β = {current_beta:.2f}")

    if artists["arrow"] is not None:
        artists["arrow"].remove()
        artists["arrow"] = None

    if step < len(proposals):
        proposed_idx = int(proposals[step])
        in_bounds = 0 <= proposed_idx < N_BETAS
        accepted = bool(accepts[step])
        color = "green" if accepted else "red"

        if in_bounds:
            proposed_beta = betas[proposed_idx]
            artists["arrow"] = artists["ax_lines"].annotate(
                "",
                xy=(0.95, proposed_beta),
                xytext=(0.95, current_beta),
                arrowprops=dict(facecolor=color, edgecolor=color,
                                width=2, headwidth=10, shrink=0.05),
            )
            if show_accept_text:
                log_ratio = (log_likelihood_at_index(proposed_idx)
                             - log_likelihood_at_index(current_idx))
                alpha_disp = min(1.0, float(np.exp(min(log_ratio, 0.0))))
                artists["prob_text"].set_text(
                    f"min(1, L({proposed_beta:.2f})/L({current_beta:.2f})) = {alpha_disp:.2f}"
                )
                artists["verdict_text"].set_text("ACCEPTED" if accepted else "REJECTED")
                artists["verdict_text"].set_color(color)
            else:
                artists["prob_text"].set_text("")
                artists["verdict_text"].set_text("")
        else:
            if show_accept_text:
                artists["prob_text"].set_text("proposed β off grid → stay")
                artists["verdict_text"].set_text("REJECTED")
                artists["verdict_text"].set_color("red")
            else:
                artists["prob_text"].set_text("")
                artists["verdict_text"].set_text("")
    else:
        artists["prob_text"].set_text("")
        artists["verdict_text"].set_text("")

    artists["trace_line"].set_data(np.arange(step + 1), chain[: step + 1])
    artists["trace_current"].set_data([step], [chain[step]])
    artists["ax_trace"].set_xlim(-1, max(20, step + 5))

    fracs = visit_fractions(chain[: step + 1])
    for bar, h in zip(artists["bars"], fracs):
        bar.set_height(h)
    artists["ax_hist"].set_ylim(
        0, max(0.6, grid_posterior.max() * 1.15, fracs.max() * 1.1)
    )

    artists["iter_text"].set_text(f"iter {step}")


def write_walkthrough_pngs(chain, proposals, accepts):
    for step in range(N_WALKTHROUGH):
        artists = build_figure()
        render_step(artists, step, chain, proposals, accepts)
        path = OUT / f"02_regression_discrete_step{step:02d}.png"
        artists["fig"].savefig(path, dpi=150)
        plt.close(artists["fig"])
        print(f"wrote {path}")


def write_animation(chain, proposals, accepts):
    artists = build_figure()
    anim = animation.FuncAnimation(
        artists["fig"],
        lambda s: render_step(artists, s, chain, proposals, accepts,
                              show_accept_text=False),
        frames=N_ANIM_FRAMES + 1,
        interval=1000 / FPS,
        blit=False,
    )
    path = OUT / "02_regression_discrete.mp4"
    anim.save(path, writer=animation.FFMpegWriter(fps=FPS, bitrate=1800))
    plt.close(artists["fig"])
    print(f"wrote {path}")


def write_full_trace_png(chain):
    fig, ax = plt.subplots(figsize=(14, 3.5))
    ax.plot(np.arange(len(chain)), chain, color="steelblue",
            linewidth=0.3, alpha=0.7)
    ax.set_xlim(0, len(chain) - 1)
    ax.set_ylim(-0.3, N_BETAS - 0.7)
    ax.set_yticks(beta_indices)
    ax.set_yticklabels([f"{b:.2f}" for b in betas])
    ax.set_xlabel("iteration")
    ax.set_ylabel("β")
    ax.set_title(f"β sampled (all {len(chain) - 1} iterations)")
    fig.tight_layout()
    path = OUT / "02_regression_discrete_trace.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def write_convergence_grid(chain):
    fig, axes = plt.subplots(1, len(SNAPSHOT_STEPS),
                             figsize=(3 * len(SNAPSHOT_STEPS), 3.2), sharey=True)
    for ax, step in zip(axes, SNAPSHOT_STEPS):
        ax.bar(beta_indices, visit_fractions(chain[: step + 1]),
               color="steelblue", alpha=0.85)
        ax.plot(beta_indices, grid_posterior, "k--o", linewidth=1.5, markersize=5)
        ax.set_title(f"iter {step}")
        ax.set_xlabel("β")
        ax.set_xticks(beta_indices)
        ax.set_xticklabels([f"{b:.2f}" for b in betas], rotation=45)
    axes[0].set_ylabel("visit fraction")
    fig.tight_layout()
    path = OUT / "02_regression_discrete_convergence.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def main():
    OUT.mkdir(exist_ok=True)
    write_data_png()
    write_lines_png()
    chain, proposals, accepts = metropolis(
        log_prob=log_prob_on_grid,
        initial_state=np.array(INITIAL_INDEX),
        proposal=adjacent_proposal,
        n_steps=N_STEPS,
        rng=np.random.default_rng(MCMC_SEED),
    )
    print(f"acceptance rate: {accepts.mean():.2%}")
    write_walkthrough_pngs(chain, proposals, accepts)
    write_animation(chain, proposals, accepts)
    write_full_trace_png(chain)
    write_convergence_grid(chain)


if __name__ == "__main__":
    main()
