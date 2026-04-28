"""MCMC samplers."""
import numpy as np


def metropolis(log_prob, initial_state, proposal, n_steps, rng=None):
    """Random-walk Metropolis with a symmetric proposal.

    log_prob(state) -> log density up to a constant.
    proposal(state, rng) -> proposed next state (symmetric in state).

    Returns (chain, proposals, accepts):
        chain     : (n_steps + 1, *state.shape) — includes the initial state
        proposals : (n_steps,     *state.shape) — proposed state at each step
        accepts   : (n_steps,) bool             — whether each proposal was accepted
    """
    if rng is None:
        rng = np.random.default_rng()

    state = np.asarray(initial_state)
    chain = np.empty((n_steps + 1,) + state.shape, dtype=state.dtype)
    proposals = np.empty((n_steps,) + state.shape, dtype=state.dtype)
    accepts = np.zeros(n_steps, dtype=bool)

    chain[0] = state
    log_p = log_prob(state)

    for i in range(n_steps):
        proposed = np.asarray(proposal(state, rng))
        proposals[i] = proposed
        log_p_new = log_prob(proposed)
        if np.log(rng.uniform()) < log_p_new - log_p:
            state = proposed
            log_p = log_p_new
            accepts[i] = True
        chain[i + 1] = state

    return chain, proposals, accepts
