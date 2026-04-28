"""Continuous linear regression — Metropolis with a Gaussian random-walk proposal on β.

One-parameter version: α = 0 and σ = 0.25 are known, flat prior on β.
The proposal is β' ~ Normal(β, PROPOSAL_SD); since that's symmetric,
the accept formula is still min(1, π(β')/π(β)) — same as the discrete demo.

With a flat prior + known σ, the posterior is conjugate:
    β | data ~ Normal( Σx·y / Σx², σ² / Σx² )
so we can overlay the exact posterior as a smooth curve (the continuous
analog of the dashed grid-posterior dots from demo 2).

module load ffmpeg/5.1

Outputs (in ../out/):
    03_regression_continuous_step00..02.png    first few steps as standalone figures
    03_regression_continuous.mp4               animation of the first N_ANIM_FRAMES iterations
    03_regression_continuous_trace.png         full N_STEPS trace
    03_regression_continuous_convergence.png   histograms vs analytic posterior at snapshots
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

# --- sampler settings ---
N_STEPS = 10000
N_ANIM_FRAMES = 1000
FPS = 60
N_WALKTHROUGH = 3
SNAPSHOT_STEPS = [50, 500, N_ANIM_FRAMES, N_STEPS]
MCMC_SEED = 20260418
INITIAL_BETA = 0.80
PROPOSAL_SD = 0.1

# --- viewing window on β ---
BETA_MIN_VIEW = 0.30
BETA_MAX_VIEW = 0.90
N_HIST_BINS = 40

# --- simulate data ---
data_rng = np.random.default_rng(DATA_SEED)
x = data_rng.uniform(-1, 1, N_DATA)
y = TRUE_BETA * x + data_rng.normal(0, SIGMA, N_DATA)

# --- analytic posterior (flat prior, known σ, α = 0) ---
POSTERIOR_MEAN = float(np.sum(x * y) / np.sum(x ** 2))
POSTERIOR_SD = float(SIGMA / np.sqrt(np.sum(x ** 2)))


def log_likelihood_beta(beta):
    return float(np.sum(norm.logpdf(y, loc=float(beta) * x, scale=SIGMA)))


def gaussian_proposal(state, rng):
    return float(state) + float(rng.normal(0.0, PROPOSAL_SD))


def build_figure():
    fig = plt.figure(figsize=(15, 4.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.2, 1.0],
                          wspace=0.35, top=0.78, bottom=0.15)
    ax_lines = fig.add_subplot(gs[0, 0])
    ax_trace = fig.add_subplot(gs[0, 1])
    ax_hist = fig.add_subplot(gs[0, 2])

    # --- lines panel ---
    ax_lines.scatter(x, y, s=10, color="tab:green", alpha=0.55, zorder=1)
    current_line, = ax_lines.plot([], [], color="tab:orange",
                                  linewidth=2.5, zorder=3)
    proposed_line, = ax_lines.plot([], [], linestyle="--", linewidth=1.5,
                                   color="gray", alpha=0.9, zorder=2)
    ax_lines.set_xlim(-1.05, 1.25)
    ax_lines.set_ylim(-1.1, 1.1)
    ax_lines.set_xlabel("x")
    ax_lines.set_ylabel("y")
    beta_label = ax_lines.text(1.03, 0, "", fontsize=11,
                               family="monospace", va="center",
                               color="tab:orange")
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
    trace_line, = ax_trace.plot([], [], "-", color="steelblue", linewidth=0.8)
    trace_current, = ax_trace.plot([], [], "o", color="tab:orange",
                                   markersize=6, zorder=3)
    ax_trace.axhline(TRUE_BETA, color="black", linestyle=":",
                     linewidth=1, alpha=0.5)
    ax_trace.set_xlim(-1, 20)
    ax_trace.set_ylim(BETA_MIN_VIEW, BETA_MAX_VIEW)
    ax_trace.set_xlabel("iteration")
    ax_trace.set_ylabel("β")
    ax_trace.set_title("β sampled")

    # --- histogram panel ---
    bins = np.linspace(BETA_MIN_VIEW, BETA_MAX_VIEW, N_HIST_BINS + 1)
    bar_positions = 0.5 * (bins[:-1] + bins[1:])
    bar_width = bins[1] - bins[0]
    bars = ax_hist.bar(bar_positions, np.zeros(N_HIST_BINS),
                       width=bar_width, color="steelblue", alpha=0.85,
                       align="center")
    beta_grid = np.linspace(BETA_MIN_VIEW, BETA_MAX_VIEW, 400)
    posterior_curve = norm.pdf(beta_grid, loc=POSTERIOR_MEAN, scale=POSTERIOR_SD)
    ax_hist.plot(beta_grid, posterior_curve, "k-",
                 linewidth=1.5, label="analytic posterior")
    ax_hist.set_xlabel("β")
    ax_hist.set_ylabel("density")
    ax_hist.set_xlim(BETA_MIN_VIEW, BETA_MAX_VIEW)
    ax_hist.set_ylim(0, posterior_curve.max() * 1.3)
    ax_hist.legend(loc="upper right")
    iter_text = ax_hist.text(0.02, 0.88, "", transform=ax_hist.transAxes,
                             ha="left", va="top")

    artists = dict(
        fig=fig, ax_lines=ax_lines, ax_trace=ax_trace, ax_hist=ax_hist,
        current_line=current_line, proposed_line=proposed_line,
        trace_line=trace_line, trace_current=trace_current,
        bars=bars, bins=bins,
        beta_label=beta_label, prob_text=prob_text, verdict_text=verdict_text,
        iter_text=iter_text, arrow=None,
        posterior_ymax=float(posterior_curve.max()),
    )
    return artists


def render_step(artists, step, chain, proposals, accepts, show_accept_text=True):
    current_beta = float(chain[step])
    xline = np.array([-1.0, 1.0])
    artists["current_line"].set_data(xline, current_beta * xline)
    artists["beta_label"].set_position((1.03, current_beta))
    artists["beta_label"].set_text(f"β = {current_beta:.3f}")

    if artists["arrow"] is not None:
        artists["arrow"].remove()
        artists["arrow"] = None

    if step < len(proposals):
        proposed_beta = float(proposals[step])
        accepted = bool(accepts[step])
        color = "green" if accepted else "red"
        artists["proposed_line"].set_data(xline, proposed_beta * xline)
        artists["proposed_line"].set_color(color)
        artists["proposed_line"].set_visible(True)
        artists["arrow"] = artists["ax_lines"].annotate(
            "",
            xy=(0.95, proposed_beta),
            xytext=(0.95, current_beta),
            arrowprops=dict(facecolor=color, edgecolor=color,
                            width=2, headwidth=10, shrink=0.05),
        )
        if show_accept_text:
            log_ratio = (log_likelihood_beta(proposed_beta)
                         - log_likelihood_beta(current_beta))
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
        artists["proposed_line"].set_visible(False)
        artists["prob_text"].set_text("")
        artists["verdict_text"].set_text("")

    artists["trace_line"].set_data(np.arange(step + 1), chain[: step + 1])
    artists["trace_current"].set_data([step], [chain[step]])
    artists["ax_trace"].set_xlim(-1, max(20, step + 5))

    counts, _ = np.histogram(chain[: step + 1], bins=artists["bins"], density=True)
    for bar, h in zip(artists["bars"], counts):
        bar.set_height(h)
    max_h = float(counts.max()) if counts.size else 0.0
    artists["ax_hist"].set_ylim(
        0, max(artists["posterior_ymax"] * 1.3, max_h * 1.1)
    )

    artists["iter_text"].set_text(f"iter {step}")


def write_walkthrough_pngs(chain, proposals, accepts):
    for step in range(N_WALKTHROUGH):
        artists = build_figure()
        render_step(artists, step, chain, proposals, accepts)
        path = OUT / f"03_regression_continuous_step{step:02d}.png"
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
    path = OUT / "03_regression_continuous.mp4"
    anim.save(path, writer=animation.FFMpegWriter(fps=FPS, bitrate=1800))
    plt.close(artists["fig"])
    print(f"wrote {path}")


def write_full_trace_png(chain):
    fig, ax = plt.subplots(figsize=(14, 3.5))
    ax.plot(np.arange(len(chain)), chain, color="steelblue",
            linewidth=0.3, alpha=0.7)
    ax.axhline(TRUE_BETA, color="black", linestyle=":", linewidth=1,
               alpha=0.6, label=f"true β = {TRUE_BETA}")
    ax.set_xlim(0, len(chain) - 1)
    ax.set_ylim(BETA_MIN_VIEW, BETA_MAX_VIEW)
    ax.set_xlabel("iteration")
    ax.set_ylabel("β")
    ax.set_title(f"β sampled (all {len(chain) - 1} iterations)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    path = OUT / "03_regression_continuous_trace.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def write_convergence_grid(chain):
    fig, axes = plt.subplots(1, len(SNAPSHOT_STEPS),
                             figsize=(3 * len(SNAPSHOT_STEPS), 3.2), sharey=True)
    bins = np.linspace(BETA_MIN_VIEW, BETA_MAX_VIEW, N_HIST_BINS + 1)
    beta_grid = np.linspace(BETA_MIN_VIEW, BETA_MAX_VIEW, 400)
    posterior_curve = norm.pdf(beta_grid, loc=POSTERIOR_MEAN, scale=POSTERIOR_SD)
    for ax, step in zip(axes, SNAPSHOT_STEPS):
        ax.hist(chain[: step + 1], bins=bins, color="steelblue",
                alpha=0.85, density=True)
        ax.plot(beta_grid, posterior_curve, "k-", linewidth=1.5)
        ax.set_title(f"iter {step}")
        ax.set_xlabel("β")
        ax.set_xlim(BETA_MIN_VIEW, BETA_MAX_VIEW)
    axes[0].set_ylabel("density")
    fig.tight_layout()
    path = OUT / "03_regression_continuous_convergence.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def main():
    OUT.mkdir(exist_ok=True)
    chain, proposals, accepts = metropolis(
        log_prob=log_likelihood_beta,
        initial_state=np.array(INITIAL_BETA),
        proposal=gaussian_proposal,
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
