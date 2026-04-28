"""King Markov on 10 islands — the opening Metropolis demo from Statistical Rethinking ch. 9.

module load ffmpeg/5.1

Outputs (in ../out/):
    01_island.mp4               animation of the first N_ANIM_FRAMES weeks
    01_island_proposal.png      walk panel only, week 1's proposal arrow (intro slide)
    01_island_step00..05.png    first six steps as standalone figures (walkthrough)
    01_island_trace.png         full N_STEPS trace of the chain
    01_island_convergence.png   histogram snapshots at SNAPSHOT_WEEKS
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from samplers import metropolis

OUT = Path(__file__).resolve().parent.parent / "out"

N_ISLANDS = 10
N_STEPS = 10000          # long enough for the convergence grid to actually converge
N_ANIM_FRAMES = 1000     # mp4 animates this many frames
FPS = 60                 # video length ≈ N_ANIM_FRAMES / FPS seconds
N_WALKTHROUGH = 3
SNAPSHOT_WEEKS = [50, 500, N_ANIM_FRAMES, N_STEPS]
SEED = 0

islands = np.arange(1, N_ISLANDS + 1)
target = islands / islands.sum()

# Circular layout: island 1 at the right, going counterclockwise.
theta = 2 * np.pi * (islands - 1) / N_ISLANDS
island_x = np.cos(theta)
island_y = np.sin(theta)


def log_population(k):
    return np.log(k)


def coin_flip(state, rng):
    direction = 1 if rng.uniform() < 0.5 else -1
    proposed = int(state) + direction
    if proposed == 0:
        proposed = N_ISLANDS
    elif proposed == N_ISLANDS + 1:
        proposed = 1
    return proposed


def visit_fractions(prefix):
    counts = np.bincount(prefix, minlength=N_ISLANDS + 1)[1:]
    total = counts.sum()
    return counts / total if total > 0 else counts.astype(float)


def build_figure():
    fig = plt.figure(figsize=(14, 4.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.2, 1.0], wspace=0.3)
    ax_walk = fig.add_subplot(gs[0, 0])
    ax_trace = fig.add_subplot(gs[0, 1])
    ax_hist = fig.add_subplot(gs[0, 2])

    # --- walk panel (circular) ---
    ax_walk.scatter(island_x, island_y, s=islands * 120,
                    facecolor="steelblue", edgecolor="black", zorder=1)
    for k in islands:
        ax_walk.text(island_x[k - 1] * 1.35, island_y[k - 1] * 1.35, str(k),
                     ha="center", va="center", fontsize=12)
    king_dot, = ax_walk.plot([island_x[-1]], [island_y[-1]],
                             "o", color="orange", markersize=16, zorder=3)
    ax_walk.set_xlim(-1.5, 1.5)
    ax_walk.set_ylim(-1.5, 1.5)
    ax_walk.set_aspect("equal")
    ax_walk.axis("off")
    prob_text = ax_walk.text(
        0.5, 1.14, "", transform=ax_walk.transAxes,
        ha="center", va="bottom", fontsize=13, family="monospace",
    )
    verdict_text = ax_walk.text(
        0.5, 1.03, "", transform=ax_walk.transAxes,
        ha="center", va="bottom", fontsize=14,
        family="monospace", fontweight="bold",
    )

    # --- trace panel ---
    trace_line, = ax_trace.plot([], [], "-", color="steelblue",
                                markersize=3, linewidth=1)
    trace_current, = ax_trace.plot([], [], "o", color="orange",
                                   markersize=6, zorder=3)
    ax_trace.set_xlim(-1, 20)
    ax_trace.set_ylim(0.5, N_ISLANDS + 0.5)
    ax_trace.set_yticks(islands)
    ax_trace.set_xlabel("week")
    ax_trace.set_ylabel("island")
    ax_trace.set_title("island visited")

    # --- histogram panel ---
    bars = ax_hist.bar(islands, np.zeros(N_ISLANDS),
                       color="steelblue", alpha=0.85)
    ax_hist.plot(islands, target, "k--o", linewidth=1.5, markersize=5,
                 label="target ∝ k")
    ax_hist.set_xlim(0.5, N_ISLANDS + 0.5)
    ax_hist.set_ylim(0, 0.35)
    ax_hist.set_xticks(islands)
    ax_hist.set_xlabel("island")
    ax_hist.set_ylabel("visit fraction")
    ax_hist.set_title("visit fraction vs target")
    ax_hist.legend(loc="upper left")
    week_text = ax_hist.text(0.98, 0.95, "", transform=ax_hist.transAxes,
                             ha="right", va="top")

    artists = dict(
        fig=fig, ax_walk=ax_walk, ax_trace=ax_trace, ax_hist=ax_hist,
        king_dot=king_dot, trace_line=trace_line, trace_current=trace_current,
        bars=bars, prob_text=prob_text, verdict_text=verdict_text,
        week_text=week_text, arrow=None,
    )
    return artists


def render_step(artists, step, chain, proposals, accepts, show_accept_text=True):
    """Update all artists to show the state at week `step` and its proposal."""
    current_k = int(chain[step])
    artists["king_dot"].set_data(
        [island_x[current_k - 1]], [island_y[current_k - 1]]
    )

    if artists["arrow"] is not None:
        artists["arrow"].remove()
        artists["arrow"] = None

    if step < len(proposals):
        proposed_k = int(proposals[step])
        accepted = bool(accepts[step])
        color = "green" if accepted else "red"
        artists["arrow"] = artists["ax_walk"].annotate(
            "",
            xy=(island_x[proposed_k - 1], island_y[proposed_k - 1]),
            xytext=(island_x[current_k - 1], island_y[current_k - 1]),
            arrowprops=dict(facecolor=color, edgecolor=color,
                            width=2, headwidth=10, shrink=0.12),
        )
        if show_accept_text:
            alpha = min(1.0, proposed_k / current_k)
            artists["prob_text"].set_text(
                f"min(1, {proposed_k}/{current_k}) = {alpha:.2f}"
            )
            artists["verdict_text"].set_text("ACCEPTED" if accepted else "REJECTED")
            artists["verdict_text"].set_color(color)
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
    artists["ax_hist"].set_ylim(0, max(0.35, fracs.max() * 1.1))

    artists["week_text"].set_text(f"week {step}")


def write_islands_only_png():
    """Just the ring of islands, no king and no arrow. Used on the intro slide."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(island_x, island_y, s=islands * 120,
               facecolor="steelblue", edgecolor="black", zorder=1)
    for k in islands:
        ax.text(island_x[k - 1] * 1.35, island_y[k - 1] * 1.35, str(k),
                ha="center", va="center", fontsize=12)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")
    ax.axis("off")
    path = OUT / "01_island_only.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def write_proposal_only_png(chain, proposals):
    """Walk panel only, week 0 → 1: king + first proposal as a neutral arrow.
    Used in the talk to introduce the proposal step before accept/reject is on the table.
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(island_x, island_y, s=islands * 120,
               facecolor="steelblue", edgecolor="black", zorder=1)
    for k in islands:
        ax.text(island_x[k - 1] * 1.35, island_y[k - 1] * 1.35, str(k),
                ha="center", va="center", fontsize=12)
    current_k = int(chain[0])
    proposed_k = int(proposals[0])
    ax.plot([island_x[current_k - 1]], [island_y[current_k - 1]],
            "o", color="orange", markersize=16, zorder=3)
    ax.annotate(
        "",
        xy=(island_x[proposed_k - 1], island_y[proposed_k - 1]),
        xytext=(island_x[current_k - 1], island_y[current_k - 1]),
        arrowprops=dict(facecolor="black", edgecolor="black",
                        width=2, headwidth=10, shrink=0.12),
    )
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")
    ax.axis("off")
    path = OUT / "01_island_proposal.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def write_walkthrough_pngs(chain, proposals, accepts):
    for step in range(N_WALKTHROUGH):
        artists = build_figure()
        render_step(artists, step, chain, proposals, accepts)
        path = OUT / f"01_island_step{step:02d}.png"
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
    path = OUT / "01_island.mp4"
    anim.save(path, writer=animation.FFMpegWriter(fps=FPS, bitrate=1800))
    plt.close(artists["fig"])
    print(f"wrote {path}")


def write_full_trace_png(chain):
    fig, ax = plt.subplots(figsize=(14, 3.5))
    ax.plot(np.arange(len(chain)), chain, color="steelblue",
            linewidth=0.3, alpha=0.7)
    ax.set_xlim(0, len(chain) - 1)
    ax.set_ylim(0.5, N_ISLANDS + 0.5)
    ax.set_yticks(islands)
    ax.set_xlabel("week")
    ax.set_ylabel("island")
    ax.set_title(f"island visited (all {len(chain) - 1} weeks)")
    fig.tight_layout()
    path = OUT / "01_island_trace.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def write_convergence_grid(chain):
    fig, axes = plt.subplots(1, len(SNAPSHOT_WEEKS),
                             figsize=(3 * len(SNAPSHOT_WEEKS), 3), sharey=True)
    for ax, step in zip(axes, SNAPSHOT_WEEKS):
        ax.bar(islands, visit_fractions(chain[: step + 1]),
               color="steelblue", alpha=0.85)
        ax.plot(islands, target, "k--o", linewidth=1.5, markersize=5)
        ax.set_title(f"week {step}")
        ax.set_xlabel("island")
        ax.set_xticks(islands)
    axes[0].set_ylabel("visit fraction")
    fig.tight_layout()
    path = OUT / "01_island_convergence.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def main():
    OUT.mkdir(exist_ok=True)
    chain, proposals, accepts = metropolis(
        log_prob=log_population,
        initial_state=np.array(10),
        proposal=coin_flip,
        n_steps=N_STEPS,
        rng=np.random.default_rng(SEED),
    )
    print(f"acceptance rate: {accepts.mean():.2%}")
    write_islands_only_png()
    write_proposal_only_png(chain, proposals)
    write_walkthrough_pngs(chain, proposals, accepts)
    write_animation(chain, proposals, accepts)
    write_full_trace_png(chain)
    write_convergence_grid(chain)


if __name__ == "__main__":
    main()
