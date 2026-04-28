"""Continuous linear regression — 2 parameters (α and β), Metropolis with a
2D Gaussian random-walk proposal.

Model: y = α + β·x + ε,  ε ~ N(0, σ),  σ = 0.25 known, flat priors on (α, β).
Proposal: α' ~ N(α, PROPOSAL_SD_ALPHA), β' ~ N(β, PROPOSAL_SD_BETA),
independent — still symmetric, so the accept formula is min(1, π(θ')/π(θ)).

Analytic posterior (flat prior, known σ):
    (α, β) | data ~ N( (X'X)⁻¹ X'y,  σ² (X'X)⁻¹ )
where X = [1, x]. Used as ground truth in the contour overlay.

module load ffmpeg/5.1

Outputs (in ../out/):
    04_regression_2param_step00..02.png    first few steps as standalone figures
    04_regression_2param.mp4               animation of the first N_ANIM_FRAMES iterations
    04_regression_2param_trace.png         full N_STEPS traces (α and β stacked)
    04_regression_2param_convergence.png   chain scatter + posterior contours at snapshots
    04_regression_2param_convergence_1d.png 1D marginal histograms + analytic marginals
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
TRUE_ALPHA = 0.0
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
INITIAL_ALPHA = 0.10
INITIAL_BETA = 0.80
PROPOSAL_SD_ALPHA = 0.05
PROPOSAL_SD_BETA = 0.10

# --- viewing windows ---
ALPHA_MIN_VIEW, ALPHA_MAX_VIEW = -0.15, 0.20
BETA_MIN_VIEW, BETA_MAX_VIEW = 0.40, 0.85

# --- simulate data ---
data_rng = np.random.default_rng(DATA_SEED)
x = data_rng.uniform(-1, 1, N_DATA)
y = TRUE_ALPHA + TRUE_BETA * x + data_rng.normal(0, SIGMA, N_DATA)

# --- analytic bivariate-normal posterior ---
X = np.column_stack([np.ones(N_DATA), x])
XtX_inv = np.linalg.inv(X.T @ X)
POSTERIOR_MEAN = XtX_inv @ X.T @ y           # [α_hat, β_hat]
POSTERIOR_COV = SIGMA ** 2 * XtX_inv          # 2×2


def log_likelihood(state):
    a = float(state[0])
    b = float(state[1])
    return float(np.sum(norm.logpdf(y, loc=a + b * x, scale=SIGMA)))


def gaussian_proposal(state, rng):
    step = rng.normal(0.0, [PROPOSAL_SD_ALPHA, PROPOSAL_SD_BETA])
    return state + step


def _posterior_contour_grid():
    """Evaluate the bivariate-normal posterior on a grid, for contour overlays."""
    a_grid = np.linspace(ALPHA_MIN_VIEW, ALPHA_MAX_VIEW, 120)
    b_grid = np.linspace(BETA_MIN_VIEW, BETA_MAX_VIEW, 120)
    AA, BB = np.meshgrid(a_grid, b_grid)
    diff = np.stack([AA - POSTERIOR_MEAN[0], BB - POSTERIOR_MEAN[1]], axis=-1)
    cov_inv = np.linalg.inv(POSTERIOR_COV)
    quad = np.einsum("...i,ij,...j", diff, cov_inv, diff)
    density = np.exp(-0.5 * quad)
    return AA, BB, density


_AA, _BB, _DENSITY = _posterior_contour_grid()
# 68% / 95% / 99.7% credible ellipses for bivariate normal, ascending.
_CONTOUR_LEVELS = np.exp(-0.5 * np.array([11.83, 6.17, 2.30])) * _DENSITY.max()


def build_figure():
    fig = plt.figure(figsize=(16, 5.2))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.1, 1.0, 1.0],
                          hspace=0.18, wspace=0.32, top=0.82, bottom=0.12)
    ax_lines = fig.add_subplot(gs[:, 0])
    ax_trace_a = fig.add_subplot(gs[0, 1])
    ax_trace_b = fig.add_subplot(gs[1, 1], sharex=ax_trace_a)
    ax_post = fig.add_subplot(gs[:, 2])

    # --- lines panel ---
    ax_lines.scatter(x, y, s=10, color="tab:green", alpha=0.55, zorder=1)
    current_line, = ax_lines.plot([], [], color="tab:orange",
                                  linewidth=2.5, zorder=3)
    proposed_line, = ax_lines.plot([], [], linestyle="--", linewidth=1.5,
                                   color="gray", alpha=0.9, zorder=2)
    ax_lines.set_xlim(-1.05, 1.15)
    ax_lines.set_ylim(-1.1, 1.1)
    ax_lines.set_xlabel("x")
    ax_lines.set_ylabel("y")
    param_label = ax_lines.text(
        0.02, 0.02, "", transform=ax_lines.transAxes,
        fontsize=10, family="monospace", va="bottom", ha="left",
        color="tab:orange",
    )
    prob_text = ax_lines.text(
        0.5, 1.10, "", transform=ax_lines.transAxes,
        ha="center", va="bottom", fontsize=12, family="monospace",
    )
    verdict_text = ax_lines.text(
        0.5, 1.02, "", transform=ax_lines.transAxes,
        ha="center", va="bottom", fontsize=13,
        family="monospace", fontweight="bold",
    )

    # --- α trace (top middle) ---
    trace_line_a, = ax_trace_a.plot([], [], "-", color="steelblue", linewidth=0.8)
    trace_current_a, = ax_trace_a.plot([], [], "o", color="tab:orange",
                                       markersize=5, zorder=3)
    ax_trace_a.axhline(TRUE_ALPHA, color="black", linestyle=":",
                       linewidth=1, alpha=0.5)
    ax_trace_a.set_xlim(-1, 20)
    ax_trace_a.set_ylim(ALPHA_MIN_VIEW, ALPHA_MAX_VIEW)
    ax_trace_a.set_ylabel("α")
    ax_trace_a.tick_params(labelbottom=False)
    ax_trace_a.set_title("α, β sampled")

    # --- β trace (bottom middle) ---
    trace_line_b, = ax_trace_b.plot([], [], "-", color="steelblue", linewidth=0.8)
    trace_current_b, = ax_trace_b.plot([], [], "o", color="tab:orange",
                                       markersize=5, zorder=3)
    ax_trace_b.axhline(TRUE_BETA, color="black", linestyle=":",
                       linewidth=1, alpha=0.5)
    ax_trace_b.set_ylim(BETA_MIN_VIEW, BETA_MAX_VIEW)
    ax_trace_b.set_xlabel("iteration")
    ax_trace_b.set_ylabel("β")

    # --- posterior panel (scatter + contour + current + proposal + arrow) ---
    ax_post.contour(_AA, _BB, _DENSITY, levels=_CONTOUR_LEVELS,
                    colors="black", linewidths=1.0, alpha=0.6, zorder=2)
    post_scatter = ax_post.scatter([], [], s=6, color="steelblue",
                                   alpha=0.35, zorder=1)
    post_current, = ax_post.plot([], [], "o", color="tab:orange",
                                 markersize=9, zorder=4)
    post_proposal, = ax_post.plot([], [], "o", color="gray",
                                  markersize=7, alpha=0.9, zorder=3)
    ax_post.set_xlim(ALPHA_MIN_VIEW, ALPHA_MAX_VIEW)
    ax_post.set_ylim(BETA_MIN_VIEW, BETA_MAX_VIEW)
    ax_post.set_xlabel("α")
    ax_post.set_ylabel("β")
    ax_post.set_title("samples vs analytic posterior")
    iter_text = ax_post.text(0.02, 0.98, "", transform=ax_post.transAxes,
                             ha="left", va="top",
                             fontsize=10, family="monospace")

    artists = dict(
        fig=fig, ax_lines=ax_lines,
        ax_trace_a=ax_trace_a, ax_trace_b=ax_trace_b, ax_post=ax_post,
        current_line=current_line, proposed_line=proposed_line,
        trace_line_a=trace_line_a, trace_current_a=trace_current_a,
        trace_line_b=trace_line_b, trace_current_b=trace_current_b,
        post_scatter=post_scatter, post_current=post_current,
        post_proposal=post_proposal,
        param_label=param_label, prob_text=prob_text, verdict_text=verdict_text,
        iter_text=iter_text, arrow_lines=None, arrow_post=None,
    )
    return artists


def render_step(artists, step, chain, proposals, accepts, show_accept_text=True):
    current_alpha = float(chain[step, 0])
    current_beta = float(chain[step, 1])
    xline = np.array([-1.05, 1.05])

    artists["current_line"].set_data(xline, current_alpha + current_beta * xline)
    artists["param_label"].set_text(
        f"α = {current_alpha:+.3f}\nβ = {current_beta:+.3f}"
    )

    for key in ("arrow_lines", "arrow_post"):
        if artists[key] is not None:
            artists[key].remove()
            artists[key] = None

    if step < len(proposals):
        proposed_alpha = float(proposals[step, 0])
        proposed_beta = float(proposals[step, 1])
        accepted = bool(accepts[step])
        color = "green" if accepted else "red"

        artists["proposed_line"].set_data(
            xline, proposed_alpha + proposed_beta * xline
        )
        artists["proposed_line"].set_color(color)
        artists["proposed_line"].set_visible(True)

        artists["post_proposal"].set_data([proposed_alpha], [proposed_beta])
        artists["post_proposal"].set_color(color)
        artists["post_proposal"].set_visible(True)

        # arrow on the lines panel: at x = 1, current y -> proposed y
        artists["arrow_lines"] = artists["ax_lines"].annotate(
            "",
            xy=(1.0, proposed_alpha + proposed_beta * 1.0),
            xytext=(1.0, current_alpha + current_beta * 1.0),
            arrowprops=dict(facecolor=color, edgecolor=color,
                            width=2, headwidth=10, shrink=0.05),
        )
        # arrow on the posterior panel: current -> proposed in (α, β)
        artists["arrow_post"] = artists["ax_post"].annotate(
            "",
            xy=(proposed_alpha, proposed_beta),
            xytext=(current_alpha, current_beta),
            arrowprops=dict(facecolor=color, edgecolor=color,
                            width=1.5, headwidth=8, shrink=0.04),
        )

        if show_accept_text:
            log_ratio = (log_likelihood(np.array([proposed_alpha, proposed_beta]))
                         - log_likelihood(np.array([current_alpha, current_beta])))
            alpha_disp = min(1.0, float(np.exp(min(log_ratio, 0.0))))
            artists["prob_text"].set_text(
                f"min(1, L(α={proposed_alpha:+.2f}, β={proposed_beta:.2f})"
                f" / L(α={current_alpha:+.2f}, β={current_beta:.2f}))"
                f" = {alpha_disp:.2f}"
            )
            artists["verdict_text"].set_text("ACCEPTED" if accepted else "REJECTED")
            artists["verdict_text"].set_color(color)
        else:
            artists["prob_text"].set_text("")
            artists["verdict_text"].set_text("")
    else:
        artists["proposed_line"].set_visible(False)
        artists["post_proposal"].set_visible(False)
        artists["prob_text"].set_text("")
        artists["verdict_text"].set_text("")

    xs = np.arange(step + 1)
    artists["trace_line_a"].set_data(xs, chain[: step + 1, 0])
    artists["trace_current_a"].set_data([step], [current_alpha])
    artists["trace_line_b"].set_data(xs, chain[: step + 1, 1])
    artists["trace_current_b"].set_data([step], [current_beta])
    artists["ax_trace_a"].set_xlim(-1, max(20, step + 5))

    artists["post_scatter"].set_offsets(chain[: step + 1])
    artists["post_current"].set_data([current_alpha], [current_beta])
    artists["iter_text"].set_text(f"iter {step}")


def write_walkthrough_pngs(chain, proposals, accepts):
    for step in range(N_WALKTHROUGH):
        artists = build_figure()
        render_step(artists, step, chain, proposals, accepts)
        path = OUT / f"04_regression_2param_step{step:02d}.png"
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
    path = OUT / "04_regression_2param.mp4"
    anim.save(path, writer=animation.FFMpegWriter(fps=FPS, bitrate=1800))
    plt.close(artists["fig"])
    print(f"wrote {path}")


def write_full_trace_png(chain):
    fig, axes = plt.subplots(2, 1, figsize=(14, 5), sharex=True)
    axes[0].plot(np.arange(len(chain)), chain[:, 0], color="steelblue",
                 linewidth=0.3, alpha=0.7)
    axes[0].axhline(TRUE_ALPHA, color="black", linestyle=":", linewidth=1,
                    alpha=0.6, label=f"true α = {TRUE_ALPHA}")
    axes[0].set_ylim(ALPHA_MIN_VIEW, ALPHA_MAX_VIEW)
    axes[0].set_ylabel("α")
    axes[0].legend(loc="upper right")
    axes[0].set_title(f"α, β sampled (all {len(chain) - 1} iterations)")

    axes[1].plot(np.arange(len(chain)), chain[:, 1], color="steelblue",
                 linewidth=0.3, alpha=0.7)
    axes[1].axhline(TRUE_BETA, color="black", linestyle=":", linewidth=1,
                    alpha=0.6, label=f"true β = {TRUE_BETA}")
    axes[1].set_ylim(BETA_MIN_VIEW, BETA_MAX_VIEW)
    axes[1].set_ylabel("β")
    axes[1].set_xlabel("iteration")
    axes[1].legend(loc="upper right")

    axes[0].set_xlim(0, len(chain) - 1)
    fig.tight_layout()
    path = OUT / "04_regression_2param_trace.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def write_convergence_grid(chain):
    fig, axes = plt.subplots(1, len(SNAPSHOT_STEPS),
                             figsize=(3.2 * len(SNAPSHOT_STEPS), 3.4),
                             sharey=True)
    for ax, step in zip(axes, SNAPSHOT_STEPS):
        ax.contour(_AA, _BB, _DENSITY, levels=_CONTOUR_LEVELS,
                   colors="black", linewidths=1.0, alpha=0.6)
        ax.scatter(chain[: step + 1, 0], chain[: step + 1, 1],
                   s=4, color="steelblue", alpha=0.35)
        ax.set_xlim(ALPHA_MIN_VIEW, ALPHA_MAX_VIEW)
        ax.set_ylim(BETA_MIN_VIEW, BETA_MAX_VIEW)
        ax.set_title(f"iter {step}")
        ax.set_xlabel("α")
    axes[0].set_ylabel("β")
    fig.tight_layout()
    path = OUT / "04_regression_2param_convergence.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def write_convergence_1d_grid(chain):
    """1D marginal histograms at snapshots, overlaid with analytic marginals.
    Top row = α, bottom row = β. Direct 2-param analog of demo 03's convergence plot.
    """
    alpha_mean, beta_mean = POSTERIOR_MEAN
    alpha_sd = float(np.sqrt(POSTERIOR_COV[0, 0]))
    beta_sd = float(np.sqrt(POSTERIOR_COV[1, 1]))
    alpha_grid = np.linspace(ALPHA_MIN_VIEW, ALPHA_MAX_VIEW, 400)
    beta_grid = np.linspace(BETA_MIN_VIEW, BETA_MAX_VIEW, 400)
    alpha_curve = norm.pdf(alpha_grid, loc=alpha_mean, scale=alpha_sd)
    beta_curve = norm.pdf(beta_grid, loc=beta_mean, scale=beta_sd)
    alpha_bins = np.linspace(ALPHA_MIN_VIEW, ALPHA_MAX_VIEW, 31)
    beta_bins = np.linspace(BETA_MIN_VIEW, BETA_MAX_VIEW, 31)

    n = len(SNAPSHOT_STEPS)
    fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 5.0),
                             sharex="row", sharey="row")
    for col, step in enumerate(SNAPSHOT_STEPS):
        ax_a = axes[0, col]
        ax_b = axes[1, col]
        ax_a.hist(chain[: step + 1, 0], bins=alpha_bins,
                  color="steelblue", alpha=0.85, density=True)
        ax_a.plot(alpha_grid, alpha_curve, "k-", linewidth=1.5)
        ax_a.set_xlim(ALPHA_MIN_VIEW, ALPHA_MAX_VIEW)
        ax_a.set_title(f"iter {step}")

        ax_b.hist(chain[: step + 1, 1], bins=beta_bins,
                  color="steelblue", alpha=0.85, density=True)
        ax_b.plot(beta_grid, beta_curve, "k-", linewidth=1.5)
        ax_b.set_xlim(BETA_MIN_VIEW, BETA_MAX_VIEW)
        ax_b.set_xlabel("β")

    axes[0, 0].set_ylabel("α density")
    axes[1, 0].set_ylabel("β density")
    fig.tight_layout()
    path = OUT / "04_regression_2param_convergence_1d.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def main():
    OUT.mkdir(exist_ok=True)
    chain, proposals, accepts = metropolis(
        log_prob=log_likelihood,
        initial_state=np.array([INITIAL_ALPHA, INITIAL_BETA], dtype=float),
        proposal=gaussian_proposal,
        n_steps=N_STEPS,
        rng=np.random.default_rng(MCMC_SEED),
    )
    print(f"acceptance rate: {accepts.mean():.2%}")
    write_walkthrough_pngs(chain, proposals, accepts)
    write_animation(chain, proposals, accepts)
    write_full_trace_png(chain)
    write_convergence_grid(chain)
    write_convergence_1d_grid(chain)


if __name__ == "__main__":
    main()
