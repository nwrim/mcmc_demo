"""HMC walkthrough — 1D posterior.

Visualizes a single HMC proposal on the same 1-parameter regression target
as demo 03 (β with α = 0, σ = 0.25 known, flat prior). Target plotted as
an altitude landscape: x = β, y = -log π(β). The car (orange dot) sits on
the curve; its horizontal speed (purple arrow) = momentum.

Three moments:
    step00 — random momentum drawn, car at start
    step01 — one leapfrog step done (gradient → speed update → position update)
    step02 — full L-step trajectory done, proposal ready, MH decision shown

Outputs (in ../out/):
    05_hmc_walkthrough_step00..02.png
    05_hmc_walkthrough.mp4
    05_hmc_chain.mp4            several proposals in a row, so students can
                                see the chain settle into the valley
    05_hmc_walkthrough_trace.png        full N_STEPS trace of β
    05_hmc_walkthrough_convergence.png  histograms vs analytic posterior at
                                        snapshots (same style as demo 03)

module load ffmpeg/5.1
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import norm

OUT = Path(__file__).resolve().parent.parent / "out"

# --- regression target (identical to demo 03) ---
N_DATA = 100
TRUE_BETA = 0.6
SIGMA = 0.25
DATA_SEED = 1769

data_rng = np.random.default_rng(DATA_SEED)
x = data_rng.uniform(-1, 1, N_DATA)
y = TRUE_BETA * x + data_rng.normal(0, SIGMA, N_DATA)

# analytic posterior on β (flat prior, known σ, α = 0) — same form as demo 03
POSTERIOR_MEAN = float(np.sum(x * y) / np.sum(x ** 2))
POSTERIOR_SD = float(SIGMA / np.sqrt(np.sum(x ** 2)))

# --- HMC settings ---
STEP_SIZE = 0.002
N_LEAPFROG = 30
INITIAL_BETA = 0.80
HMC_SEED = 7
FPS = 15
HOLD_FRAMES = 12  # pause at start and end of single-proposal animation

# --- multi-sample chain animation ---
N_CHAIN_SAMPLES = 15
FPS_CHAIN = 30          # faster than single-proposal walkthrough
HOLD_FRAMES_KICK = 5    # pause on the new momentum before leapfrog begins
HOLD_FRAMES_VERDICT = 9 # pause on ACCEPTED/REJECTED before the next kick

# --- full chain (trace + convergence) ---
N_STEPS = 10000
SNAPSHOT_STEPS = [50, 500, 1000, N_STEPS]
N_HIST_BINS = 40

# --- viewing window ---
BETA_MIN_VIEW = 0.40
BETA_MAX_VIEW = 0.95


def log_posterior(beta):
    """log π(β) up to an additive constant (flat prior, known σ)."""
    return -0.5 * float(np.sum((y - beta * x) ** 2)) / SIGMA ** 2


def grad_log_posterior(beta):
    return float(np.sum(x * (y - beta * x)) / SIGMA ** 2)


def altitude(beta):
    """-log π(β). King Monty's landscape."""
    return -log_posterior(beta)


def leapfrog_step(q, p, step_size):
    """One leapfrog step: half-kick + full-drift + half-kick, bundled as a single
    conceptual 'move the car using the local slope' operation."""
    p = p + 0.5 * step_size * grad_log_posterior(q)
    q = q + step_size * p
    p = p + 0.5 * step_size * grad_log_posterior(q)
    return q, p


def leapfrog_trajectory(q0, p0, step_size, n_leapfrog):
    qs = np.empty(n_leapfrog + 1)
    ps = np.empty(n_leapfrog + 1)
    qs[0], ps[0] = q0, p0
    q, p = q0, p0
    for i in range(n_leapfrog):
        q, p = leapfrog_step(q, p, step_size)
        qs[i + 1], ps[i + 1] = q, p
    return qs, ps


def hamiltonian(q, p):
    """Total energy = altitude + kinetic."""
    return altitude(q) + 0.5 * p ** 2


def run_hmc_proposal(q0, rng):
    p0 = float(rng.normal())
    qs, ps = leapfrog_trajectory(q0, p0, STEP_SIZE, N_LEAPFROG)
    log_accept = hamiltonian(qs[0], ps[0]) - hamiltonian(qs[-1], ps[-1])
    accepted = bool(np.log(rng.uniform()) < log_accept)
    return qs, ps, accepted, log_accept


# --- precompute altitude curve (shifted so valley bottom = 0 for readability) ---
_beta_grid = np.linspace(BETA_MIN_VIEW, BETA_MAX_VIEW, 400)
_altitude_curve = np.array([altitude(b) for b in _beta_grid])
_altitude_baseline = float(_altitude_curve.min())
_altitude_curve_shifted = _altitude_curve - _altitude_baseline
_ALT_MAX = float(_altitude_curve_shifted.max() * 1.12)


def _altitude_at(beta):
    return altitude(beta) - _altitude_baseline


def build_figure():
    fig, ax = plt.subplots(figsize=(12, 5.5))
    fig.subplots_adjust(top=0.80, bottom=0.13, left=0.08, right=0.96)

    ax.plot(_beta_grid, _altitude_curve_shifted, "-", color="black",
            linewidth=1.8, alpha=0.85, zorder=2)
    ax.fill_between(_beta_grid, _altitude_curve_shifted, _ALT_MAX,
                    color="tab:cyan", alpha=0.10, zorder=1)

    # trajectory trail (builds up across leapfrog steps)
    trail, = ax.plot([], [], "-", color="steelblue",
                     linewidth=1.2, alpha=0.75, zorder=3)
    trail_dots, = ax.plot([], [], "o", color="steelblue",
                          markersize=3.5, alpha=0.6, zorder=3)

    # the car
    car, = ax.plot([], [], "o", color="tab:orange", markersize=16,
                   markeredgecolor="black", markeredgewidth=1, zorder=5)

    ax.set_xlim(BETA_MIN_VIEW, BETA_MAX_VIEW)
    ax.set_ylim(-_ALT_MAX * 0.05, _ALT_MAX)
    ax.set_xlabel("β  (position along the valley)")
    ax.set_ylabel("altitude = -log π(β)  (shifted so valley bottom = 0)")
    ax.set_title("HMC on a 1D posterior — King Monty drives along the valley")

    info_text = ax.text(
        0.02, 1.05, "", transform=ax.transAxes,
        ha="left", va="bottom", fontsize=12, family="monospace",
    )
    verdict_text = ax.text(
        0.98, 1.05, "", transform=ax.transAxes,
        ha="right", va="bottom", fontsize=13,
        family="monospace", fontweight="bold",
    )

    artists = dict(
        fig=fig, ax=ax,
        trail=trail, trail_dots=trail_dots, car=car,
        info_text=info_text, verdict_text=verdict_text,
        momentum_arrow=None,
    )
    return artists


def _draw_momentum_arrow(artists, beta, momentum, scale=0.025):
    ax = artists["ax"]
    if artists["momentum_arrow"] is not None:
        artists["momentum_arrow"].remove()
        artists["momentum_arrow"] = None
    if abs(momentum) < 1e-8:
        return
    dx = momentum * scale
    y = _altitude_at(beta)
    artists["momentum_arrow"] = ax.annotate(
        "",
        xy=(beta + dx, y), xytext=(beta, y),
        arrowprops=dict(facecolor="tab:purple", edgecolor="tab:purple",
                        width=2.5, headwidth=11, shrink=0.0),
        zorder=6,
    )


def render(artists, qs, ps, step_index, accepted, log_accept, phase):
    """phase in {'start', 'leapfrog', 'end'}"""
    if phase == "start":
        beta, p = qs[0], ps[0]
        artists["car"].set_data([beta], [_altitude_at(beta)])
        artists["trail"].set_data([], [])
        artists["trail_dots"].set_data([], [])
        _draw_momentum_arrow(artists, beta, p)
        artists["info_text"].set_text(
            f"step 0   β = {beta:.3f}   momentum = {p:+.3f}   (random kick drawn)"
        )
        artists["verdict_text"].set_text("")
    elif phase == "leapfrog":
        i = step_index
        beta, p = qs[i], ps[i]
        artists["car"].set_data([beta], [_altitude_at(beta)])
        trail_q = qs[: i + 1]
        trail_alt = [_altitude_at(q) for q in trail_q]
        artists["trail"].set_data(trail_q, trail_alt)
        artists["trail_dots"].set_data(trail_q, trail_alt)
        _draw_momentum_arrow(artists, beta, p)
        artists["info_text"].set_text(
            f"leapfrog step {i}/{N_LEAPFROG}   "
            f"β = {beta:.3f}   momentum = {p:+.3f}"
        )
        artists["verdict_text"].set_text("")
    elif phase == "end":
        beta, p = qs[-1], ps[-1]
        artists["car"].set_data([beta], [_altitude_at(beta)])
        trail_alt = [_altitude_at(q) for q in qs]
        artists["trail"].set_data(qs, trail_alt)
        artists["trail_dots"].set_data(qs, trail_alt)
        _draw_momentum_arrow(artists, beta, p)
        artists["info_text"].set_text(
            f"trajectory done   proposal β = {beta:.3f}   log α = {log_accept:+.3f}"
        )
        artists["verdict_text"].set_text("ACCEPTED" if accepted else "REJECTED")
        artists["verdict_text"].set_color("green" if accepted else "red")


def build_chain_figure():
    """Landscape on top (full width); lines | trace | histogram stacked below."""
    fig = plt.figure(figsize=(15, 8.4))
    gs = fig.add_gridspec(
        2, 3, width_ratios=[1, 1, 1], height_ratios=[2, 1],
        hspace=0.55, wspace=0.35,
        top=0.86, bottom=0.09, left=0.06, right=0.97,
    )
    ax = fig.add_subplot(gs[0, :])
    ax_lines = fig.add_subplot(gs[1, 0])
    ax_trace = fig.add_subplot(gs[1, 1])
    ax_hist = fig.add_subplot(gs[1, 2])

    # --- altitude landscape ---
    ax.plot(_beta_grid, _altitude_curve_shifted, "-", color="black",
            linewidth=1.8, alpha=0.85, zorder=2)
    ax.fill_between(_beta_grid, _altitude_curve_shifted, _ALT_MAX,
                    color="tab:cyan", alpha=0.10, zorder=1)
    trail, = ax.plot([], [], "-", color="steelblue",
                     linewidth=1.2, alpha=0.75, zorder=3)
    trail_dots, = ax.plot([], [], "o", color="steelblue",
                          markersize=3.5, alpha=0.6, zorder=3)
    car, = ax.plot([], [], "o", color="tab:orange", markersize=16,
                   markeredgecolor="black", markeredgewidth=1, zorder=5)
    ax.set_xlim(BETA_MIN_VIEW, BETA_MAX_VIEW)
    ax.set_ylim(-_ALT_MAX * 0.05, _ALT_MAX)
    ax.set_xlabel("β  (position along the valley)")
    ax.set_ylabel("altitude = -log π(β)")
    ax.set_title("HMC on a 1D posterior — King Monty drives along the valley")

    info_text = ax.text(
        0.02, 1.05, "", transform=ax.transAxes,
        ha="left", va="bottom", fontsize=12, family="monospace",
    )
    verdict_text = ax.text(
        0.98, 1.05, "", transform=ax.transAxes,
        ha="right", va="bottom", fontsize=13,
        family="monospace", fontweight="bold",
    )

    # --- regression-lines panel (β <-> regression line) ---
    ax_lines.scatter(x, y, s=10, color="tab:green", alpha=0.55, zorder=1)
    current_line, = ax_lines.plot([], [], color="tab:orange",
                                  linewidth=2.5, zorder=3)
    proposed_line, = ax_lines.plot([], [], linestyle="--", linewidth=1.5,
                                   color="gray", alpha=0.9, zorder=2)
    ax_lines.set_xlim(-1.05, 1.05)
    ax_lines.set_ylim(-1.1, 1.1)
    ax_lines.set_xlabel("x")
    ax_lines.set_ylabel("y")
    ax_lines.set_title("regression line at current β")

    # --- trace panel ---
    trace_line, = ax_trace.plot([], [], "-", color="steelblue", linewidth=1.0)
    trace_dots, = ax_trace.plot([], [], "o", color="steelblue",
                                markersize=4, alpha=0.7)
    trace_current, = ax_trace.plot([], [], "o", color="tab:orange",
                                   markersize=8, zorder=5)
    ax_trace.axhline(TRUE_BETA, color="black", linestyle=":",
                     linewidth=1, alpha=0.5, label=f"true β = {TRUE_BETA}")
    ax_trace.set_xlim(-0.5, N_CHAIN_SAMPLES + 0.5)
    ax_trace.set_ylim(BETA_MIN_VIEW, BETA_MAX_VIEW)
    ax_trace.set_xlabel("iteration")
    ax_trace.set_ylabel("β")
    ax_trace.set_title("trace")
    ax_trace.legend(loc="upper right", fontsize=9)

    # --- histogram panel ---
    bins = np.linspace(BETA_MIN_VIEW, BETA_MAX_VIEW, N_HIST_BINS + 1)
    bar_positions = 0.5 * (bins[:-1] + bins[1:])
    bar_width = bins[1] - bins[0]
    bars = ax_hist.bar(bar_positions, np.zeros(N_HIST_BINS),
                       width=bar_width, color="steelblue", alpha=0.85,
                       align="center")
    beta_grid_h = np.linspace(BETA_MIN_VIEW, BETA_MAX_VIEW, 400)
    posterior_curve = norm.pdf(beta_grid_h, loc=POSTERIOR_MEAN, scale=POSTERIOR_SD)
    ax_hist.plot(beta_grid_h, posterior_curve, "k-", linewidth=1.5,
                 label="analytic posterior")
    ax_hist.set_xlabel("β")
    ax_hist.set_ylabel("density")
    ax_hist.set_xlim(BETA_MIN_VIEW, BETA_MAX_VIEW)
    ax_hist.set_ylim(0, posterior_curve.max() * 1.3)
    ax_hist.set_title("samples so far")
    ax_hist.legend(loc="upper right", fontsize=9)

    artists = dict(
        fig=fig, ax=ax,
        ax_lines=ax_lines, ax_trace=ax_trace, ax_hist=ax_hist,
        trail=trail, trail_dots=trail_dots, car=car,
        current_line=current_line, proposed_line=proposed_line,
        trace_line=trace_line, trace_dots=trace_dots, trace_current=trace_current,
        bars=bars, bins=bins,
        info_text=info_text, verdict_text=verdict_text,
        momentum_arrow=None,
        posterior_ymax=float(posterior_curve.max()),
    )
    return artists


def run_hmc_chain(q0, n_samples, rng):
    """Run n_samples HMC iterations starting from q0.

    Returns a list of per-iteration records:
        (qs, ps, accepted, log_accept, q_next)
    where q_next is qs[-1] if accepted else qs[0] (the MCMC sample for that
    iteration — rejection means we keep the previous state as the sample).
    """
    records = []
    q = float(q0)
    for _ in range(n_samples):
        qs, ps, accepted, log_accept = run_hmc_proposal(q, rng)
        q_next = float(qs[-1]) if accepted else float(qs[0])
        records.append((qs, ps, accepted, log_accept, q_next))
        q = q_next
    return records


def _render_chain_frame(artists, records, sample_idx, local_frame):
    qs, ps, accepted, log_accept, q_next = records[sample_idx]
    total = len(records)

    # committed samples so far: iter 0 = INITIAL_BETA, iter k = records[k-1][4].
    # A new sample is appended once the current iteration enters its verdict hold.
    is_verdict_phase = local_frame >= HOLD_FRAMES_KICK + N_LEAPFROG
    n_new = sample_idx + (1 if is_verdict_phase else 0)
    committed = [float(INITIAL_BETA)] + [r[4] for r in records[:n_new]]

    # trace panel
    iters = np.arange(len(committed))
    artists["trace_line"].set_data(iters, committed)
    artists["trace_dots"].set_data(iters, committed)
    artists["trace_current"].set_data([iters[-1]], [committed[-1]])

    # histogram panel
    counts, _ = np.histogram(committed, bins=artists["bins"], density=True)
    for bar, h in zip(artists["bars"], counts):
        bar.set_height(h)
    max_h = float(counts.max()) if counts.size else 0.0
    artists["ax_hist"].set_ylim(
        0, max(artists["posterior_ymax"] * 1.3, max_h * 1.1)
    )

    # resolve which β the car is at this frame (drives the landscape + lines panel)
    xline = np.array([-1.0, 1.0])
    if local_frame < HOLD_FRAMES_KICK:
        beta, p = qs[0], ps[0]
        artists["car"].set_data([beta], [_altitude_at(beta)])
        artists["trail"].set_data([], [])
        artists["trail_dots"].set_data([], [])
        _draw_momentum_arrow(artists, beta, p)
        artists["info_text"].set_text(
            f"sample {sample_idx + 1}/{total}   β = {beta:.3f}   "
            f"momentum = {p:+.3f}   (new kick)"
        )
        artists["verdict_text"].set_text("")
        artists["current_line"].set_data(xline, beta * xline)
        artists["proposed_line"].set_visible(False)
    elif local_frame < HOLD_FRAMES_KICK + N_LEAPFROG:
        i = local_frame - HOLD_FRAMES_KICK + 1
        beta, p = qs[i], ps[i]
        artists["car"].set_data([beta], [_altitude_at(beta)])
        trail_q = qs[: i + 1]
        trail_alt = [_altitude_at(q) for q in trail_q]
        artists["trail"].set_data(trail_q, trail_alt)
        artists["trail_dots"].set_data(trail_q, trail_alt)
        _draw_momentum_arrow(artists, beta, p)
        artists["info_text"].set_text(
            f"sample {sample_idx + 1}/{total}   "
            f"leapfrog {i}/{N_LEAPFROG}   β = {beta:.3f}"
        )
        artists["verdict_text"].set_text("")
        artists["current_line"].set_data(xline, beta * xline)
        artists["proposed_line"].set_visible(False)
    else:
        # verdict hold: car rests at the committed sample (snap back if rejected)
        beta = q_next
        artists["car"].set_data([beta], [_altitude_at(beta)])
        trail_alt = [_altitude_at(q) for q in qs]
        artists["trail"].set_data(qs, trail_alt)
        artists["trail_dots"].set_data(qs, trail_alt)
        if artists["momentum_arrow"] is not None:
            artists["momentum_arrow"].remove()
            artists["momentum_arrow"] = None
        artists["info_text"].set_text(
            f"sample {sample_idx + 1}/{total}   "
            f"proposal β = {qs[-1]:.3f}   log α = {log_accept:+.3f}"
        )
        artists["verdict_text"].set_text("ACCEPTED" if accepted else "REJECTED")
        artists["verdict_text"].set_color("green" if accepted else "red")
        # lines panel: orange at committed β, dashed at trajectory endpoint
        color = "green" if accepted else "red"
        artists["current_line"].set_data(xline, beta * xline)
        artists["proposed_line"].set_data(xline, float(qs[-1]) * xline)
        artists["proposed_line"].set_color(color)
        artists["proposed_line"].set_visible(True)


def write_chain_animation(records):
    frames_per_sample = HOLD_FRAMES_KICK + N_LEAPFROG + HOLD_FRAMES_VERDICT
    total_frames = frames_per_sample * len(records)
    artists = build_chain_figure()

    def update(frame):
        sample_idx = frame // frames_per_sample
        local_frame = frame % frames_per_sample
        _render_chain_frame(artists, records, sample_idx, local_frame)

    anim = animation.FuncAnimation(
        artists["fig"], update,
        frames=total_frames,
        interval=1000 / FPS_CHAIN,
        blit=False,
    )
    path = OUT / "05_hmc_chain.mp4"
    anim.save(path, writer=animation.FFMpegWriter(fps=FPS_CHAIN, bitrate=1800))
    plt.close(artists["fig"])
    print(f"wrote {path}")


def write_walkthrough_pngs(qs, ps, accepted, log_accept):
    specs = [
        ("start", 0, "step00"),
        ("leapfrog", 1, "step01"),
        ("end", N_LEAPFROG, "step02"),
    ]
    for phase, idx, tag in specs:
        artists = build_figure()
        render(artists, qs, ps, idx, accepted, log_accept, phase)
        path = OUT / f"05_hmc_walkthrough_{tag}.png"
        artists["fig"].savefig(path, dpi=150)
        plt.close(artists["fig"])
        print(f"wrote {path}")


def write_animation(qs, ps, accepted, log_accept):
    total_frames = 2 * HOLD_FRAMES + N_LEAPFROG
    artists = build_figure()

    def update(frame):
        if frame < HOLD_FRAMES:
            render(artists, qs, ps, 0, accepted, log_accept, "start")
        elif frame < HOLD_FRAMES + N_LEAPFROG:
            i = frame - HOLD_FRAMES + 1
            render(artists, qs, ps, i, accepted, log_accept, "leapfrog")
        else:
            render(artists, qs, ps, N_LEAPFROG, accepted, log_accept, "end")

    anim = animation.FuncAnimation(
        artists["fig"], update,
        frames=total_frames,
        interval=1000 / FPS,
        blit=False,
    )
    path = OUT / "05_hmc_walkthrough.mp4"
    anim.save(path, writer=animation.FFMpegWriter(fps=FPS, bitrate=1800))
    plt.close(artists["fig"])
    print(f"wrote {path}")


def write_chain_trace_png(chain):
    fig, ax = plt.subplots(figsize=(14, 3.5))
    ax.plot(np.arange(len(chain)), chain, color="steelblue",
            linewidth=0.4, alpha=0.8)
    ax.axhline(TRUE_BETA, color="black", linestyle=":", linewidth=1,
               alpha=0.6, label=f"true β = {TRUE_BETA}")
    ax.set_xlim(0, len(chain) - 1)
    ax.set_ylim(BETA_MIN_VIEW, BETA_MAX_VIEW)
    ax.set_xlabel("iteration")
    ax.set_ylabel("β")
    ax.set_title(f"HMC β trace (all {len(chain) - 1} iterations)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    path = OUT / "05_hmc_walkthrough_trace.png"
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
    path = OUT / "05_hmc_walkthrough_convergence.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def main():
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(HMC_SEED)
    records = run_hmc_chain(INITIAL_BETA, N_STEPS, rng)

    # first iteration drives the single-proposal walkthrough (PNGs + mp4)
    qs, ps, accepted, log_accept, _ = records[0]
    print(f"initial β:  {qs[0]:.4f}")
    print(f"initial p:  {ps[0]:+.4f}")
    print(f"proposal β: {qs[-1]:.4f}")
    print(f"log α:      {log_accept:+.4f}")
    print(f"decision:   {'ACCEPT' if accepted else 'REJECT'}")

    n_accepted = sum(1 for r in records if r[2])
    print(f"chain acceptance: {n_accepted}/{N_STEPS} = {n_accepted / N_STEPS:.2%}")

    write_walkthrough_pngs(qs, ps, accepted, log_accept)
    write_animation(qs, ps, accepted, log_accept)
    write_chain_animation(records[:N_CHAIN_SAMPLES])

    # full-chain outputs: prepend INITIAL_BETA so chain[step] indexing matches
    # demo 03's convention (chain has N_STEPS+1 entries: iter 0..N_STEPS)
    chain = np.concatenate(([INITIAL_BETA], [r[4] for r in records]))
    write_chain_trace_png(chain)
    write_convergence_grid(chain)


if __name__ == "__main__":
    main()
