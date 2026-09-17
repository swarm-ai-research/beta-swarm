"""Dynamic Hamiltonian Monte Carlo, in numpy, with the diagnostics that matter.

Implemented from Betancourt, "A Conceptual Introduction to Hamiltonian Monte
Carlo" (arXiv:1701.02434). The design choices that paper argues for, and which
this module therefore makes:

**Dynamic trajectory length, not static.** A static HMC trajectory length is
exactly the kind of hand-tuned knob that inference is supposed to remove: too
short and the sampler random-walks, too long and it doubles back and wastes the
work. Trajectories here are built by multiplicative doubling and terminated by
a U-turn criterion on the sampled momenta (section 4.4 and appendix A.4), so
the length adapts to local geometry.

**Multinomial sampling from the trajectory**, rather than the slice sampling of
the original NUTS. The paper notes this as a strict improvement: every state in
the trajectory is a candidate weighted by its density, instead of only those
above a uniform slice.

**Divergences are a first-class output.** When the symplectic integrator hits a
region of high curvature, the energy error explodes and the trajectory is
silently truncated. The chain then looks perfectly healthy while never visiting
a whole neighbourhood of the typical set — a posterior that is wrong in a way no
amount of extra sampling fixes. ``HMCDiagnostics.divergences`` is a required
field, and :meth:`HMCDiagnostics.warnings` reports it.

**E-BFMI.** Momentum resampling has to be able to move the chain between energy
levels. When it cannot, the marginal energy distribution is explored far more
slowly than the position distribution, and R-hat will not necessarily notice.

**Split R-hat and ESS** are computed the standard way, across chains.

The sampler is deliberately dependency-free (numpy + scipy only). It exists to
answer whether a posterior is worth having for this model at all; if the answer
is yes, replace it with numpyro or blackjax rather than maintaining this.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np

# Energy error beyond which a trajectory is declared divergent. The scale is
# arbitrary but conventional (Stan uses 1000): any error this large means the
# integrator has fallen off the level set entirely, so the precise cutoff does
# not matter.
_MAX_ENERGY_ERROR = 1000.0

# Hard ceiling on doubling depth. A trajectory reaching this has 2**10 states
# and almost always means the step size is far too small rather than that the
# geometry genuinely needs that long a trajectory.
_MAX_TREE_DEPTH = 10


class LogProbFn(Protocol):
    """``theta -> (log density, gradient)``, both up to an additive constant."""

    def __call__(self, q: np.ndarray) -> tuple[float, np.ndarray]: ...


# ----------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------
@dataclass
class HMCDiagnostics:
    """Everything needed to decide whether to trust the draws.

    Every field here can indicate a failure. That is the point: a sampler that
    cannot report its own pathologies is worse than no sampler, because it
    produces confident wrong answers instead of obvious ones.
    """

    divergences: int
    """Post-warmup transitions whose energy error exceeded the threshold."""

    max_tree_depth_hits: int
    """Trajectories truncated by the depth ceiling rather than by a U-turn."""

    step_size: list[float]
    """Adapted step size per chain."""

    accept_rate: list[float]
    """Mean acceptance statistic per chain; should land near ``target_accept``."""

    ebfmi: list[float]
    """Energy BFMI per chain. Below ~0.3 means poor energy-level mixing."""

    rhat: np.ndarray
    """Split R-hat per parameter. Above 1.01 means the chains disagree."""

    ess: np.ndarray
    """Effective sample size per parameter."""

    n_chains: int
    n_draws: int
    n_warmup: int

    def warnings(self) -> list[str]:
        """Human-readable problems, empty if the run looks healthy.

        Thresholds follow common practice: R-hat 1.01, ESS 100/chain, E-BFMI
        0.3. They are conventions, not theorems — a run that trips one is
        suspect, not proven wrong.
        """
        out: list[str] = []
        if self.divergences:
            pct = 100.0 * self.divergences / max(1, self.n_chains * self.n_draws)
            out.append(
                f"{self.divergences} divergent transition(s) ({pct:.1f}%) — the "
                f"posterior has curvature the integrator cannot follow, so these "
                f"draws are biased, not merely noisy. Raise target_accept or "
                f"reparameterize."
            )
        if self.max_tree_depth_hits:
            out.append(
                f"{self.max_tree_depth_hits} trajectory(ies) hit max tree depth "
                f"{_MAX_TREE_DEPTH} — efficiency problem, not a validity problem."
            )
        bad_rhat = int(np.sum(self.rhat > 1.01))
        if bad_rhat:
            out.append(
                f"{bad_rhat} parameter(s) with split R-hat > 1.01 "
                f"(max {float(np.nanmax(self.rhat)):.3f}) — chains have not mixed."
            )
        low_ess = int(np.sum(self.ess < 100 * self.n_chains))
        if low_ess:
            out.append(
                f"{low_ess} parameter(s) with ESS < {100 * self.n_chains} "
                f"(min {float(np.nanmin(self.ess)):.0f}) — estimates are imprecise."
            )
        low_ebfmi = [i for i, e in enumerate(self.ebfmi) if e < 0.3]
        if low_ebfmi:
            out.append(
                f"chain(s) {low_ebfmi} have E-BFMI < 0.3 — momentum resampling is "
                f"not traversing the energy distribution; heavy tails likely."
            )
        return out

    def summary(self) -> str:
        lines = [
            f"chains={self.n_chains} draws={self.n_draws} warmup={self.n_warmup}",
            f"divergences={self.divergences} depth_hits={self.max_tree_depth_hits}",
            f"step_size={[round(s, 4) for s in self.step_size]}",
            f"accept_rate={[round(a, 3) for a in self.accept_rate]}",
            f"ebfmi={[round(e, 3) for e in self.ebfmi]}",
            f"rhat_max={float(np.nanmax(self.rhat)):.4f} "
            f"ess_min={float(np.nanmin(self.ess)):.0f}",
        ]
        warns = self.warnings()
        lines.append("OK" if not warns else "WARNINGS:\n  - " + "\n  - ".join(warns))
        return "\n".join(lines)


@dataclass
class SampleResult:
    """Posterior draws plus the diagnostics that say whether to believe them."""

    draws: np.ndarray
    """``(n_chains, n_draws, n_params)`` post-warmup draws."""

    log_prob: np.ndarray
    """``(n_chains, n_draws)`` log density at each draw."""

    energy: np.ndarray
    """``(n_chains, n_draws)`` Hamiltonian at each draw."""

    diagnostics: HMCDiagnostics

    inv_mass: np.ndarray = field(default_factory=lambda: np.empty(0))
    """``(n_chains, n_params)`` adapted diagonal inverse metric."""

    @property
    def flat(self) -> np.ndarray:
        """All chains stacked: ``(n_chains * n_draws, n_params)``."""
        c, d, p = self.draws.shape
        return self.draws.reshape(c * d, p)

    def mean(self) -> np.ndarray:
        return np.asarray(self.flat.mean(axis=0), dtype=float)

    def quantile(self, q: float | list[float]) -> np.ndarray:
        return np.asarray(np.quantile(self.flat, q, axis=0), dtype=float)

    def credible_interval(self, level: float = 0.9) -> np.ndarray:
        """Equal-tailed interval per parameter: ``(2, n_params)``."""
        lo = (1.0 - level) / 2.0
        return np.asarray(np.quantile(self.flat, [lo, 1.0 - lo], axis=0), dtype=float)


# ----------------------------------------------------------------------
# Integrator
# ----------------------------------------------------------------------
def _leapfrog(
    q: np.ndarray,
    p: np.ndarray,
    grad: np.ndarray,
    eps: float,
    inv_mass: np.ndarray,
    logp_and_grad: LogProbFn,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """One symplectic step. Symplecticity is what bounds the energy error."""
    p = p + 0.5 * eps * grad
    q = q + eps * (inv_mass * p)
    logp, grad = logp_and_grad(q)
    p = p + 0.5 * eps * grad
    return q, p, logp, grad


def _hamiltonian(logp: float, p: np.ndarray, inv_mass: np.ndarray) -> float:
    """``H = -log pi(q) + 0.5 p' M^-1 p``, the potential plus kinetic energy."""
    if not np.isfinite(logp):
        return np.inf
    return -logp + 0.5 * float(np.sum(inv_mass * p * p))


# ----------------------------------------------------------------------
# Dynamic trajectory expansion
# ----------------------------------------------------------------------
@dataclass
class _Tree:
    """A trajectory segment, tracked only at the edges it can be extended from."""

    q_minus: np.ndarray
    p_minus: np.ndarray
    grad_minus: np.ndarray
    q_plus: np.ndarray
    p_plus: np.ndarray
    grad_plus: np.ndarray
    q_sample: np.ndarray
    logp_sample: float
    p_sum: np.ndarray
    log_weight: float
    n_leapfrog: int
    sum_accept: float
    diverging: bool
    turning: bool


def _is_turning(p_sum: np.ndarray, p_minus: np.ndarray, p_plus: np.ndarray,
                inv_mass: np.ndarray) -> bool:
    """U-turn criterion in the metric.

    The trajectory has doubled back when the total momentum no longer points
    along the velocity at either end — extending further would revisit ground
    already covered.
    """
    return bool(
        np.dot(p_sum, inv_mass * p_minus) <= 0.0
        or np.dot(p_sum, inv_mass * p_plus) <= 0.0
    )


def _build_tree(
    q: np.ndarray,
    p: np.ndarray,
    grad: np.ndarray,
    direction: int,
    depth: int,
    eps: float,
    h0: float,
    inv_mass: np.ndarray,
    logp_and_grad: LogProbFn,
    rng: np.random.Generator,
) -> _Tree:
    if depth == 0:
        q1, p1, logp1, grad1 = _leapfrog(
            q, p, grad, direction * eps, inv_mass, logp_and_grad
        )
        h1 = _hamiltonian(logp1, p1, inv_mass)
        energy_error = h1 - h0
        diverging = not np.isfinite(h1) or energy_error > _MAX_ENERGY_ERROR
        # min(1, exp(-dH)) — the acceptance statistic dual averaging targets.
        accept = 1.0 if energy_error <= 0.0 else float(np.exp(-energy_error))
        if not np.isfinite(accept):
            accept = 0.0
        return _Tree(
            q_minus=q1, p_minus=p1, grad_minus=grad1,
            q_plus=q1, p_plus=p1, grad_plus=grad1,
            q_sample=q1, logp_sample=logp1,
            p_sum=p1.copy(),
            # Multinomial weight: log pi(q,p) = -H. Divergent states get no
            # weight so they can never be returned as the draw.
            log_weight=-np.inf if diverging else -h1,
            n_leapfrog=1, sum_accept=accept,
            diverging=diverging, turning=False,
        )

    left = _build_tree(q, p, grad, direction, depth - 1, eps, h0,
                       inv_mass, logp_and_grad, rng)
    if left.diverging or left.turning:
        return left

    # Extend from the edge the trajectory is growing toward.
    if direction < 0:
        right = _build_tree(left.q_minus, left.p_minus, left.grad_minus,
                            direction, depth - 1, eps, h0,
                            inv_mass, logp_and_grad, rng)
        q_minus, p_minus, grad_minus = right.q_minus, right.p_minus, right.grad_minus
        q_plus, p_plus, grad_plus = left.q_plus, left.p_plus, left.grad_plus
    else:
        right = _build_tree(left.q_plus, left.p_plus, left.grad_plus,
                            direction, depth - 1, eps, h0,
                            inv_mass, logp_and_grad, rng)
        q_minus, p_minus, grad_minus = left.q_minus, left.p_minus, left.grad_minus
        q_plus, p_plus, grad_plus = right.q_plus, right.p_plus, right.grad_plus

    total_weight = np.logaddexp(left.log_weight, right.log_weight)
    # Multinomial progressive sampling: pick from the new subtree with
    # probability proportional to its share of the combined weight.
    q_sample, logp_sample = left.q_sample, left.logp_sample
    if np.isfinite(total_weight) and np.log(rng.uniform()) < right.log_weight - total_weight:
        q_sample, logp_sample = right.q_sample, right.logp_sample

    p_sum = left.p_sum + right.p_sum
    turning = (
        right.turning
        or _is_turning(p_sum, p_minus, p_plus, inv_mass)
        # Cross-subtree checks (Stan's refinement): catch U-turns that the
        # combined-edge test misses when the trajectory loops within a subtree.
        or _is_turning(left.p_sum + right.p_minus, p_minus, right.p_minus, inv_mass)
        or _is_turning(left.p_plus + right.p_sum, left.p_plus, p_plus, inv_mass)
    )

    return _Tree(
        q_minus=q_minus, p_minus=p_minus, grad_minus=grad_minus,
        q_plus=q_plus, p_plus=p_plus, grad_plus=grad_plus,
        q_sample=q_sample, logp_sample=logp_sample,
        p_sum=p_sum, log_weight=total_weight,
        n_leapfrog=left.n_leapfrog + right.n_leapfrog,
        sum_accept=left.sum_accept + right.sum_accept,
        diverging=right.diverging, turning=turning,
    )


def _transition(
    q: np.ndarray,
    logp: float,
    grad: np.ndarray,
    eps: float,
    inv_mass: np.ndarray,
    logp_and_grad: LogProbFn,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float, np.ndarray, float, float, bool, bool, float]:
    """One dynamic HMC transition.

    Returns ``(q, logp, grad, accept_stat, energy, diverged, depth_hit, h0)``.
    """
    mass = 1.0 / inv_mass
    p = rng.normal(size=q.shape) * np.sqrt(mass)
    h0 = _hamiltonian(logp, p, inv_mass)

    q_minus = q_plus = q
    p_minus = p_plus = p
    grad_minus = grad_plus = grad
    q_sample, logp_sample = q, logp
    log_weight = -h0
    p_sum = p.copy()

    diverged = False
    depth_hit = False
    sum_accept = 0.0
    n_leapfrog = 0

    for depth in range(_MAX_TREE_DEPTH):
        direction = 1 if rng.uniform() < 0.5 else -1
        if direction < 0:
            subtree = _build_tree(q_minus, p_minus, grad_minus, direction, depth,
                                  eps, h0, inv_mass, logp_and_grad, rng)
            q_minus, p_minus, grad_minus = (
                subtree.q_minus, subtree.p_minus, subtree.grad_minus)
        else:
            subtree = _build_tree(q_plus, p_plus, grad_plus, direction, depth,
                                  eps, h0, inv_mass, logp_and_grad, rng)
            q_plus, p_plus, grad_plus = (
                subtree.q_plus, subtree.p_plus, subtree.grad_plus)

        sum_accept += subtree.sum_accept
        n_leapfrog += subtree.n_leapfrog

        if subtree.diverging:
            diverged = True
            break
        if subtree.turning:
            break

        # Biased progressive sampling: favour the newly built half, which moves
        # the chain further per trajectory than an unbiased choice would.
        if np.isfinite(subtree.log_weight) and (
            subtree.log_weight > log_weight
            or np.log(rng.uniform()) < subtree.log_weight - log_weight
        ):
            q_sample, logp_sample = subtree.q_sample, subtree.logp_sample

        log_weight = np.logaddexp(log_weight, subtree.log_weight)
        p_sum = p_sum + subtree.p_sum
        if _is_turning(p_sum, p_minus, p_plus, inv_mass):
            break
        if depth == _MAX_TREE_DEPTH - 1:
            depth_hit = True

    accept_stat = sum_accept / max(1, n_leapfrog)
    if q_sample is q:
        grad_sample = grad
    else:
        logp_sample, grad_sample = logp_and_grad(q_sample)
    return (q_sample, logp_sample, grad_sample, accept_stat, h0,
            diverged, depth_hit, h0)


# ----------------------------------------------------------------------
# Adaptation
# ----------------------------------------------------------------------
class _DualAveraging:
    """Nesterov dual averaging on ``log eps``, targeting an acceptance rate.

    Targeting acceptance rather than tuning ``eps`` directly is what makes the
    step size transfer across problems: the acceptance statistic is a
    dimensionless measure of how well the integrator is tracking the level set.
    """

    def __init__(self, eps0: float, target: float = 0.8,
                 gamma: float = 0.05, t0: float = 10.0, kappa: float = 0.75):
        self.mu = np.log(10.0 * eps0)
        self.target = target
        self.gamma, self.t0, self.kappa = gamma, t0, kappa
        self.h_bar = 0.0
        self.log_eps_bar = 0.0
        self.m = 0

    def update(self, accept_stat: float) -> float:
        self.m += 1
        eta = 1.0 / (self.m + self.t0)
        self.h_bar = (1.0 - eta) * self.h_bar + eta * (self.target - accept_stat)
        log_eps = self.mu - np.sqrt(self.m) / self.gamma * self.h_bar
        w = self.m ** (-self.kappa)
        self.log_eps_bar = w * log_eps + (1.0 - w) * self.log_eps_bar
        return float(np.exp(log_eps))

    def final(self) -> float:
        return float(np.exp(self.log_eps_bar))

    def restart(self, eps0: float) -> None:
        self.mu = np.log(10.0 * eps0)
        self.h_bar = 0.0
        self.log_eps_bar = 0.0
        self.m = 0


class _Welford:
    """Streaming diagonal variance, for the inverse metric."""

    def __init__(self, dim: int):
        self.n = 0
        self.mean = np.zeros(dim)
        self.m2 = np.zeros(dim)

    def add(self, x: np.ndarray) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (x - self.mean)

    def variance(self) -> np.ndarray:
        if self.n < 2:
            return np.ones_like(self.mean)
        var = self.m2 / (self.n - 1)
        # Stan's regularization toward 1: with few samples the raw variance is
        # noisy, and a bad metric costs more than a slightly stale one.
        var = (self.n / (self.n + 5.0)) * var + 1e-3 * (5.0 / (self.n + 5.0))
        return np.where(var > 0, var, 1.0)

    def reset(self) -> None:
        self.n = 0
        self.mean[:] = 0.0
        self.m2[:] = 0.0


def _find_initial_step_size(
    q: np.ndarray, logp: float, grad: np.ndarray,
    inv_mass: np.ndarray, logp_and_grad: LogProbFn, rng: np.random.Generator,
) -> float:
    """Heuristic search for a step size with roughly even odds of acceptance."""
    eps = 1.0
    p = rng.normal(size=q.shape) * np.sqrt(1.0 / inv_mass)
    h0 = _hamiltonian(logp, p, inv_mass)
    _, p1, logp1, _ = _leapfrog(q, p, grad, eps, inv_mass, logp_and_grad)
    h1 = _hamiltonian(logp1, p1, inv_mass)
    direction = 1.0 if (h0 - h1) > np.log(0.8) else -1.0
    for _ in range(50):
        eps = eps * (2.0 ** direction)
        _, p1, logp1, _ = _leapfrog(q, p, grad, eps, inv_mass, logp_and_grad)
        h1 = _hamiltonian(logp1, p1, inv_mass)
        delta = h0 - h1
        if direction > 0 and not delta > np.log(0.8):
            break
        if direction < 0 and not delta < np.log(0.8):
            break
        if eps > 1e7 or eps < 1e-10:
            break
    return float(eps)


def _adaptation_windows(n_warmup: int) -> tuple[int, int, list[int]]:
    """Stan's three-phase warmup schedule.

    Step size adapts throughout; the metric is estimated only in the middle
    phase, in doubling windows. The initial buffer lets the chain reach the
    typical set before its covariance is worth measuring, and the terminal
    buffer re-tunes the step size against the final metric.
    """
    if n_warmup < 20:
        return n_warmup, 0, []
    init_buffer, term_buffer, base_window = 75, 50, 25
    if init_buffer + term_buffer + base_window > n_warmup:
        init_buffer = int(0.15 * n_warmup)
        term_buffer = int(0.10 * n_warmup)
        base_window = n_warmup - init_buffer - term_buffer
    ends: list[int] = []
    start, window = init_buffer, base_window
    while start + window <= n_warmup - term_buffer:
        end = start + window
        # Absorb a remaining stub into the last window rather than running a
        # too-short one.
        if end + 2 * window > n_warmup - term_buffer:
            end = n_warmup - term_buffer
        ends.append(end)
        start, window = end, window * 2
    return init_buffer, term_buffer, ends


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------
def sample(
    logp_and_grad: LogProbFn,
    init: np.ndarray,
    n_draws: int = 1000,
    n_warmup: int = 1000,
    n_chains: int = 4,
    target_accept: float = 0.8,
    seed: int = 0,
    progress: Callable[[str], None] | None = None,
) -> SampleResult:
    """Run dynamic HMC and return draws plus diagnostics.

    Parameters
    ----------
    logp_and_grad:
        ``q -> (log density, gradient)``, unnormalized. Must accept any real
        vector; return ``-inf`` (with any finite gradient) out of support.
    init:
        Starting point, ``(n_params,)`` shared by all chains, or
        ``(n_chains, n_params)`` for dispersed starts. Dispersed starts are
        what make R-hat meaningful — chains started at the same point can agree
        without either having converged.
    target_accept:
        Dual-averaging target. Raise toward 0.95+ when divergences appear: a
        smaller step size tracks high curvature better.

    Notes
    -----
    Always read ``result.diagnostics.warnings()`` before using the draws.
    """
    init = np.asarray(init, dtype=float)
    if init.ndim == 1:
        inits = np.tile(init, (n_chains, 1))
    else:
        inits = init
        if inits.shape[0] != n_chains:
            raise ValueError(
                f"init has {inits.shape[0]} rows but n_chains={n_chains}"
            )
    if n_draws <= 0 or n_warmup < 0:
        raise ValueError("n_draws must be positive and n_warmup non-negative")
    n_params = inits.shape[1]

    draws = np.empty((n_chains, n_draws, n_params))
    logps = np.empty((n_chains, n_draws))
    energies = np.empty((n_chains, n_draws))
    step_sizes: list[float] = []
    accept_rates: list[float] = []
    inv_masses = np.empty((n_chains, n_params))
    total_div = 0
    total_depth = 0

    init_buffer, term_buffer, window_ends = _adaptation_windows(n_warmup)

    for chain in range(n_chains):
        rng = np.random.default_rng(seed + 1000 * chain)
        q = inits[chain].copy()
        logp, grad = logp_and_grad(q)
        if not np.isfinite(logp):
            raise ValueError(
                f"chain {chain}: initial log density is not finite "
                f"({logp}) at {q} — pick a starting point inside the support"
            )

        inv_mass = np.ones(n_params)
        eps = _find_initial_step_size(q, logp, grad, inv_mass, logp_and_grad, rng)
        da = _DualAveraging(eps, target=target_accept)
        welford = _Welford(n_params)

        for i in range(n_warmup):
            q, logp, grad, accept, energy, diverged, depth_hit, _ = _transition(
                q, logp, grad, eps, inv_mass, logp_and_grad, rng
            )
            eps = da.update(accept)
            if init_buffer <= i < n_warmup - term_buffer:
                welford.add(q)
            if i + 1 in window_ends:
                inv_mass = welford.variance()
                welford.reset()
                eps = _find_initial_step_size(
                    q, logp, grad, inv_mass, logp_and_grad, rng
                )
                da.restart(eps)
            if i + 1 == n_warmup - term_buffer and window_ends:
                # Freeze the metric; the terminal buffer re-tunes eps against it.
                eps = da.final() if da.m > 0 else eps
                da.restart(eps)
        if n_warmup > 0:
            eps = da.final()

        chain_accept = 0.0
        for i in range(n_draws):
            q, logp, grad, accept, energy, diverged, depth_hit, _ = _transition(
                q, logp, grad, eps, inv_mass, logp_and_grad, rng
            )
            draws[chain, i] = q
            logps[chain, i] = logp
            energies[chain, i] = energy
            chain_accept += accept
            total_div += int(diverged)
            total_depth += int(depth_hit)

        step_sizes.append(float(eps))
        accept_rates.append(chain_accept / n_draws)
        inv_masses[chain] = inv_mass
        if progress is not None:
            progress(f"chain {chain + 1}/{n_chains} done (eps={eps:.4g})")

    diagnostics = HMCDiagnostics(
        divergences=total_div,
        max_tree_depth_hits=total_depth,
        step_size=step_sizes,
        accept_rate=accept_rates,
        ebfmi=[ebfmi(energies[c]) for c in range(n_chains)],
        rhat=split_rhat(draws),
        ess=effective_sample_size(draws),
        n_chains=n_chains,
        n_draws=n_draws,
        n_warmup=n_warmup,
    )
    return SampleResult(
        draws=draws, log_prob=logps, energy=energies,
        diagnostics=diagnostics, inv_mass=inv_masses,
    )


# ----------------------------------------------------------------------
# Diagnostics
# ----------------------------------------------------------------------
def ebfmi(energy: np.ndarray) -> float:
    """Energy Bayesian fraction of missing information for one chain.

    Compares the step size of momentum-resampling moves between energy levels
    against the width of the marginal energy distribution. When this is small,
    the sampler needs many transitions to cross the energy range even if the
    position draws look well mixed — the pathology R-hat tends to miss.
    """
    energy = np.asarray(energy, dtype=float)
    if energy.size < 2:
        return np.nan
    var = float(np.var(energy))
    if var <= 0:
        return np.nan
    return float(np.mean(np.diff(energy) ** 2) / var)


def split_rhat(draws: np.ndarray) -> np.ndarray:
    """Split R-hat per parameter for ``(n_chains, n_draws, n_params)``.

    Splitting each chain in half before comparing catches within-chain trends
    that a plain between-chain comparison misses.
    """
    draws = np.asarray(draws, dtype=float)
    n_chains, n_draws, n_params = draws.shape
    half = n_draws // 2
    if half < 2:
        return np.full(n_params, np.nan)
    split = np.concatenate([draws[:, :half], draws[:, half: 2 * half]], axis=0)
    n = split.shape[1]

    chain_means = split.mean(axis=1)
    chain_vars = split.var(axis=1, ddof=1)
    w = chain_vars.mean(axis=0)
    b = n * chain_means.var(axis=0, ddof=1)
    var_plus = ((n - 1) / n) * w + b / n
    with np.errstate(divide="ignore", invalid="ignore"):
        rhat = np.sqrt(var_plus / w)
    # A parameter with zero within-chain variance is constant, not converged;
    # report nan rather than a spurious 1.0 or a divide-by-zero.
    return np.where(w > 0, rhat, np.nan)


def _autocovariance(x: np.ndarray) -> np.ndarray:
    """Autocovariance of a 1-D series via FFT."""
    n = x.size
    x = x - x.mean()
    n_fft = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(x, n_fft)
    acov = np.fft.irfft(f * np.conjugate(f), n_fft)[:n]
    return acov / n


def effective_sample_size(draws: np.ndarray) -> np.ndarray:
    """ESS per parameter, using Geyer's initial monotone positive sequence.

    The truncation rule matters: summing autocorrelations until the first
    negative estimate is noisy, so pairs are summed and the sequence is forced
    monotone, which is what keeps ESS from being inflated by noise in the tail.
    """
    draws = np.asarray(draws, dtype=float)
    n_chains, n_draws, n_params = draws.shape
    if n_draws < 4:
        return np.full(n_params, np.nan)

    ess = np.empty(n_params)
    for j in range(n_params):
        x = draws[:, :, j]
        chain_means = x.mean(axis=1)
        chain_vars = x.var(axis=1, ddof=1)
        w = chain_vars.mean()
        if w <= 0:
            ess[j] = np.nan
            continue
        if n_chains > 1:
            b = n_draws * chain_means.var(ddof=1)
            var_plus = ((n_draws - 1) / n_draws) * w + b / n_draws
        else:
            var_plus = w

        acov = np.mean([_autocovariance(x[c]) for c in range(n_chains)], axis=0)
        rho = 1.0 - (w - acov) / var_plus
        rho[0] = 1.0

        # Geyer: sum adjacent pairs, stop when a pair goes negative, and force
        # the pair sequence to be non-increasing.
        max_pairs = (n_draws - 2) // 2
        pair_sums = []
        for t in range(max_pairs):
            s = rho[2 * t + 1] + rho[2 * t + 2]
            if s < 0:
                break
            pair_sums.append(s)
        if not pair_sums:
            ess[j] = float(n_chains * n_draws)
            continue
        pair = np.array(pair_sums)
        pair = np.minimum.accumulate(pair)
        tau = -1.0 + 2.0 * float(pair.sum())
        tau = max(tau, 1.0 / np.log10(max(n_chains * n_draws, 11)))
        ess[j] = float(n_chains * n_draws) / tau
    return ess
