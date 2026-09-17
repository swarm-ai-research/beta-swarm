"""The proxy, read as a likelihood, with an analytic gradient.

:class:`~beta_swarm.proxy.BetaProxyComputer` maps observables to a Beta belief
through seven constants. Since the outcome ``v`` is continuous on ``[0, 1]``,
that map *is* a conditional density::

    w      = softmax([z1, z2, z3, 0])      # weights on the simplex
    v_hat  = w . x                         # x in [-1, 1]^4, from observables
    mean   = sigmoid(k * v_hat)            # k  = exp(log_k)
    conc   = c0 + s * e                    # c0 = exp(log_c0), s = exp(log_s)
    v     ~ Beta(mean * conc, (1 - mean) * conc)

so fitting the proxy is just estimating ``theta = (z1, z2, z3, log_k, log_c0,
log_s)``. The existing grid sweep moves ``evidence_scale`` (``s``) alone; this
is the six-dimensional object that slice cuts through.

Three modeling decisions worth stating rather than burying:

**Weights live on the simplex.** ``compute_v_hat`` divides by ``sum(w)``, so
the four raw weights are only identified up to scale — a posterior over them
would have a perfectly flat ridge. Softmax with the last logit pinned to zero
removes exactly that redundancy and nothing else.

**The clip in ``compute_v_hat`` is a no-op here, by construction.** A convex
combination of values in ``[-1, 1]`` is already in ``[-1, 1]``. That is what
makes the map smooth enough to differentiate, so
:func:`design_from_observables` asserts the feature bound instead of trusting
it.

**Priors sit directly on the unconstrained coordinates**, not on the natural
parameters with a Jacobian correction. A Normal prior on ``log_k`` is a
log-normal prior on ``k``, which is what we want anyway for a positive scale
parameter. This is a choice, not an oversight: it means the priors below should
be read as statements about log-scales.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import special

from beta_swarm.proxy import BetaProxyComputer, ProxyObservables

N_PARAMS = 6
PARAM_NAMES = ("z_progress", "z_rework", "z_verifier", "log_k", "log_c0", "log_s")

# Human-readable names of the natural parameters, in the order
# `ProxyTheta.to_vector` / `natural_draws` produce them.
NATURAL_NAMES = (
    "w_progress", "w_rework", "w_verifier", "w_engagement",
    "sigmoid_k", "base_concentration", "evidence_scale",
)

# Outcomes exactly at 0 or 1 make the Beta log-density diverge. Nudge them
# inward — and count how many, so a dataset that is mostly boundary values
# cannot masquerade as a clean fit.
_OUTCOME_EPS = 1e-9


@dataclass(frozen=True)
class ProxyDesign:
    """Observables reduced to what the likelihood actually needs.

    ``features`` and ``evidence`` depend only on the observables, never on the
    parameters, so they are computed once and reused across every gradient
    evaluation — which is what makes the sampler affordable in pure numpy.
    """

    features: np.ndarray   # (n, 4) in [-1, 1]
    evidence: np.ndarray   # (n,) >= 0
    outcomes: np.ndarray   # (n,) in (0, 1)
    n_clamped: int = 0
    """Outcomes that sat exactly on 0 or 1 and were nudged inward."""

    def __post_init__(self) -> None:
        n = self.features.shape[0]
        if self.evidence.shape != (n,) or self.outcomes.shape != (n,):
            raise ValueError("features, evidence and outcomes must agree in length")
        if n == 0:
            raise ValueError("need at least one interaction")

    def __len__(self) -> int:
        return int(self.features.shape[0])


def design_from_observables(
    observables: Sequence[ProxyObservables],
    outcomes: Sequence[float],
    decay: float = 0.4,
) -> ProxyDesign:
    """Build the design from raw observables, mirroring ``BetaProxyComputer``.

    ``decay`` must match the proxy's ``_count_signal`` decay; it is a shape
    constant of the observable encoding, not a fitted parameter.
    """
    if len(observables) != len(outcomes):
        raise ValueError("observables and outcomes must have the same length")
    if not observables:
        raise ValueError("need at least one interaction")

    count_signal = BetaProxyComputer._count_signal
    feats = np.empty((len(observables), 4))
    evid = np.empty(len(observables))
    for i, obs in enumerate(observables):
        rejection = count_signal(obs.verifier_rejections, decay)
        misuse = count_signal(obs.tool_misuse_flags, decay)
        feats[i] = (
            float(np.clip(obs.task_progress_delta, -1.0, 1.0)),
            count_signal(obs.rework_count, decay),
            0.5 * (rejection + misuse),
            float(np.clip(obs.counterparty_engagement_delta, -1.0, 1.0)),
        )
        # Matches compute_concentration exactly, including its use of the
        # unclipped deltas.
        evid[i] = (
            abs(obs.task_progress_delta)
            + obs.rework_count
            + obs.verifier_rejections
            + obs.tool_misuse_flags
            + abs(obs.counterparty_engagement_delta)
        )

    if not np.all(np.abs(feats) <= 1.0 + 1e-12):
        raise ValueError(
            "features escaped [-1, 1]; the simplex parameterization assumes the "
            "clip in compute_v_hat never binds"
        )

    y = np.asarray(outcomes, dtype=float)
    if np.any(y < 0.0) or np.any(y > 1.0):
        raise ValueError("outcomes must lie in [0, 1]")
    clamped = int(np.sum((y <= 0.0) | (y >= 1.0)))
    y = np.clip(y, _OUTCOME_EPS, 1.0 - _OUTCOME_EPS)
    return ProxyDesign(features=feats, evidence=evid, outcomes=y, n_clamped=clamped)


@dataclass(frozen=True)
class ProxyTheta:
    """Proxy parameters in their natural (interpretable) scale."""

    w: np.ndarray            # (4,) on the simplex
    sigmoid_k: float
    base_concentration: float
    evidence_scale: float

    @classmethod
    def from_vector(cls, u: np.ndarray) -> "ProxyTheta":
        u = np.asarray(u, dtype=float)
        if u.shape != (N_PARAMS,):
            raise ValueError(f"expected {N_PARAMS} parameters, got {u.shape}")
        logits = np.concatenate([u[:3], [0.0]])
        logits = logits - logits.max()
        w = np.exp(logits)
        return cls(
            w=w / w.sum(),
            sigmoid_k=float(np.exp(u[3])),
            base_concentration=float(np.exp(u[4])),
            evidence_scale=float(np.exp(u[5])),
        )

    def to_vector(self) -> np.ndarray:
        """Natural-scale vector in :data:`NATURAL_NAMES` order."""
        return np.concatenate([
            self.w,
            [self.sigmoid_k, self.base_concentration, self.evidence_scale],
        ])

    def to_proxy(self, max_concentration: float | None = None) -> BetaProxyComputer:
        """Rebuild the production proxy so fitted parameters can be run as-is."""
        return BetaProxyComputer(
            w_progress=float(self.w[0]),
            w_rework=float(self.w[1]),
            w_verifier=float(self.w[2]),
            w_engagement=float(self.w[3]),
            sigmoid_k=self.sigmoid_k,
            base_concentration=self.base_concentration,
            evidence_scale=self.evidence_scale,
            max_concentration=max_concentration,
        )


# Weakly informative priors on the unconstrained coordinates, centred on the
# proxy's current hand-set defaults (uniform weights, k=2, c0=2, s=1.5). Wide
# enough that the data moves them; tight enough to keep the sampler out of
# regions where the Beta density underflows.
_PRIOR_MEAN = np.array([0.0, 0.0, 0.0, np.log(2.0), np.log(2.0), np.log(1.5)])
_PRIOR_SD = np.array([1.5, 1.5, 1.5, 0.75, 0.75, 1.0])


@dataclass
class ProxyPosterior:
    """Log-posterior over proxy parameters, with an analytic gradient.

    The gradient is hand-derived rather than autodiffed, which is the whole
    reason this runs without jax — and also the most likely place for a silent
    error, so :func:`tests.test_hmc` checks it against finite differences.
    """

    design: ProxyDesign
    prior_mean: np.ndarray = None  # type: ignore[assignment]
    prior_sd: np.ndarray = None    # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.prior_mean is None:
            self.prior_mean = _PRIOR_MEAN.copy()
        if self.prior_sd is None:
            self.prior_sd = _PRIOR_SD.copy()
        self.prior_mean = np.asarray(self.prior_mean, dtype=float)
        self.prior_sd = np.asarray(self.prior_sd, dtype=float)
        if self.prior_mean.shape != (N_PARAMS,) or self.prior_sd.shape != (N_PARAMS,):
            raise ValueError(f"priors must have {N_PARAMS} entries")
        if np.any(self.prior_sd <= 0):
            raise ValueError("prior standard deviations must be positive")
        d = self.design
        self._log_y = np.log(d.outcomes)
        self._log1m_y = np.log1p(-d.outcomes)

    # ------------------------------------------------------------------
    def _forward(self, u: np.ndarray) -> tuple:
        """Shared forward pass for the value and the gradient."""
        logits = np.concatenate([u[:3], [0.0]])
        logits = logits - logits.max()
        expw = np.exp(logits)
        w = expw / expw.sum()

        k = np.exp(u[3])
        c0 = np.exp(u[4])
        s = np.exp(u[5])

        v_hat = self.design.features @ w
        mu = special.expit(k * v_hat)
        conc = c0 + s * self.design.evidence
        return w, k, c0, s, v_hat, mu, conc

    def log_prob(self, u: np.ndarray) -> float:
        u = np.asarray(u, dtype=float)
        _, _, _, _, _, mu, conc = self._forward(u)
        alpha = mu * conc
        beta = (1.0 - mu) * conc
        if np.any(alpha <= 0) or np.any(beta <= 0):
            return -np.inf
        ll = np.sum(
            (alpha - 1.0) * self._log_y
            + (beta - 1.0) * self._log1m_y
            - special.betaln(alpha, beta)
        )
        lp = -0.5 * np.sum(((u - self.prior_mean) / self.prior_sd) ** 2)
        total = ll + lp
        return float(total) if np.isfinite(total) else -np.inf

    def log_prob_and_grad(self, u: np.ndarray) -> tuple[float, np.ndarray]:
        """Value and analytic gradient — the callable HMC drives.

        Chain rule from the Beta log-density::

            dl/dalpha = ln v       - (psi(alpha) - psi(alpha + beta))
            dl/dbeta  = ln(1 - v)  - (psi(beta)  - psi(alpha + beta))

        then through ``(alpha, beta) <- (mean, conc) <- (w, k, c0, s) <- u``.
        """
        u = np.asarray(u, dtype=float)
        w, k, c0, s, v_hat, mu, conc = self._forward(u)
        alpha = mu * conc
        beta = (1.0 - mu) * conc
        if np.any(alpha <= 0) or np.any(beta <= 0):
            return -np.inf, np.zeros(N_PARAMS)

        psi_sum = special.digamma(alpha + beta)
        d_alpha = self._log_y - (special.digamma(alpha) - psi_sum)
        d_beta = self._log1m_y - (special.digamma(beta) - psi_sum)

        # (alpha, beta) -> (mean, conc)
        d_mu = conc * (d_alpha - d_beta)
        d_conc = mu * d_alpha + (1.0 - mu) * d_beta

        # mean = sigmoid(k * v_hat)
        mu_prime = mu * (1.0 - mu)
        d_vhat = d_mu * k * mu_prime

        grad = np.empty(N_PARAMS)
        # v_hat = w . x with w = softmax(z): dv_hat/dz_j = w_j * (x_j - v_hat)
        for j in range(3):
            grad[j] = float(np.sum(d_vhat * w[j] * (self.design.features[:, j] - v_hat)))
        # k = exp(log_k): dmu/dlog_k = v_hat * k * mu'
        grad[3] = float(np.sum(d_mu * v_hat * k * mu_prime))
        # conc = c0 + s * e
        grad[4] = float(np.sum(d_conc) * c0)
        grad[5] = float(np.sum(d_conc * self.design.evidence) * s)

        ll = np.sum(
            (alpha - 1.0) * self._log_y
            + (beta - 1.0) * self._log1m_y
            - special.betaln(alpha, beta)
        )
        lp = -0.5 * np.sum(((u - self.prior_mean) / self.prior_sd) ** 2)
        grad -= (u - self.prior_mean) / (self.prior_sd ** 2)

        total = ll + lp
        if not np.isfinite(total) or not np.all(np.isfinite(grad)):
            return -np.inf, np.zeros(N_PARAMS)
        return float(total), grad

    def __call__(self, u: np.ndarray) -> tuple[float, np.ndarray]:
        return self.log_prob_and_grad(u)

    # ------------------------------------------------------------------
    def init_point(self) -> np.ndarray:
        """Prior mean — the proxy's current hand-set defaults."""
        return self.prior_mean.copy()

    def dispersed_inits(
        self, n_chains: int, rng: np.random.Generator, scale: float = 0.5
    ) -> np.ndarray:
        """Over-dispersed starting points, so R-hat can detect non-convergence."""
        jitter = rng.normal(size=(n_chains, N_PARAMS))
        return np.asarray(self.prior_mean + scale * self.prior_sd * jitter, dtype=float)


# ----------------------------------------------------------------------
# Posterior summaries
# ----------------------------------------------------------------------
def natural_draws(draws: np.ndarray) -> np.ndarray:
    """Map unconstrained draws to natural scale: ``(..., 6) -> (..., 7)``.

    Interpretation happens in natural units — a credible interval for
    ``evidence_scale`` is what a reader wants, not one for ``log_s``.
    """
    flat = np.asarray(draws, dtype=float).reshape(-1, N_PARAMS)
    out = np.array([ProxyTheta.from_vector(u).to_vector() for u in flat])
    shape = (*np.asarray(draws).shape[:-1], len(NATURAL_NAMES))
    return np.asarray(out.reshape(shape), dtype=float)


def posterior_tail_mass(
    draws: np.ndarray,
    observables: Sequence[ProxyObservables],
    tau: float = 0.4,
    max_draws: int = 400,
    seed: int = 0,
) -> np.ndarray:
    """``P(v < tau)`` per interaction, per posterior draw: ``(n_used, n_obs)``.

    This is the number governance triggers on. Computing it per draw is the
    point of the whole exercise: the spread across draws is the parameter
    uncertainty that a point estimate silently discards, and it is what turns
    "this interaction is over the line" into a claim with an error bar.

    Subsamples to ``max_draws`` because each draw costs a full pass of
    incomplete-beta evaluations.
    """
    flat = np.asarray(draws, dtype=float).reshape(-1, N_PARAMS)
    if flat.shape[0] > max_draws:
        idx = np.random.default_rng(seed).choice(
            flat.shape[0], size=max_draws, replace=False
        )
        flat = flat[idx]
    out = np.empty((flat.shape[0], len(observables)))
    for i, u in enumerate(flat):
        proxy = ProxyTheta.from_vector(u).to_proxy()
        out[i] = [proxy.compute_belief(o).tail_mass(tau) for o in observables]
    return out
