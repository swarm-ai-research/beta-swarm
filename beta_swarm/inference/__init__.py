"""Posterior inference over proxy parameters.

The proxy in :mod:`beta_swarm.proxy` maps observables to a Beta belief through
seven hand-set constants. Because the outcome ``v`` is continuous on ``[0, 1]``,
that map already *defines a likelihood* — the calibration harness scores it as
one (CRPS and PIT are proper scoring rules for exactly this predictive
distribution) but never inverts it.

This package inverts it: :mod:`beta_swarm.inference.proxy_posterior` writes the
log-posterior and its analytic gradient, and :mod:`beta_swarm.inference.hmc`
samples it with dynamic Hamiltonian Monte Carlo.

The sampler is hand-rolled numpy on purpose. The open question is whether a
posterior buys anything the existing 1-D grid sweep does not; adopting jax or
pymc to answer that would prejudge it. If the answer turns out to be yes, the
productionization step is to swap in numpyro or blackjax and delete
:mod:`~beta_swarm.inference.hmc`.

Reference: Betancourt, "A Conceptual Introduction to Hamiltonian Monte Carlo",
arXiv:1701.02434.
"""

from beta_swarm.inference.hmc import (
    HMCDiagnostics,
    SampleResult,
    ebfmi,
    effective_sample_size,
    sample,
    split_rhat,
)
from beta_swarm.inference.proxy_posterior import (
    ProxyDesign,
    ProxyPosterior,
    ProxyTheta,
    design_from_observables,
)

__all__ = [
    "HMCDiagnostics",
    "ProxyDesign",
    "ProxyPosterior",
    "ProxyTheta",
    "SampleResult",
    "design_from_observables",
    "ebfmi",
    "effective_sample_size",
    "sample",
    "split_rhat",
]
