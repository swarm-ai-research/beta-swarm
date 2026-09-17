#!/usr/bin/env python
"""Does a posterior tell us anything the `evidence_scale` grid does not?

`beta_swarm.calibration.sweep_evidence_scale` fits one parameter on a 1-D grid
and reports a point estimate. `beta_swarm.inference` samples all six. This
script runs both on identical data and reports what separates them.

The comparison is not apples-to-apples, and pretending otherwise would be the
easy way to make the sampler look good. The grid selects on PIT deviation (a
calibration criterion); HMC targets the posterior (a likelihood criterion).
Both derive from proper scoring rules and should broadly agree on a
well-specified model, but "the posterior mean differs from the grid optimum" is
by itself evidence of nothing. What would be decisive is one of:

  1. Posterior correlation between parameters. A per-axis grid cannot find the
     joint optimum of a correlated posterior *by construction* — it moves one
     coordinate with the rest frozen at hand-set values.
  2. A tail-mass credible interval wide enough to change a governance call the
     point estimate would have made confidently.
  3. Flat or ridged marginals — the observables failing to identify a knob the
     proxy exposes as if it were meaningful.

Outcomes are generated from the archetype emission models, not from the proxy
itself, so the proxy is *misspecified* relative to the truth — which is the
actual operating condition, and a harder test than self-recovery.

Usage:
    python scripts/hmc_vs_grid_evidence_scale.py [--n 3000] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from beta_swarm.agents import Archetype, SimAgent  # noqa: E402
from beta_swarm.calibration import calibrate, sweep_evidence_scale  # noqa: E402
from beta_swarm.inference import hmc  # noqa: E402
from beta_swarm.inference.proxy_posterior import (  # noqa: E402
    NATURAL_NAMES,
    PARAM_NAMES,
    ProxyPosterior,
    ProxyTheta,
    design_from_observables,
    natural_draws,
    posterior_tail_mass,
)

ARCHETYPES = (Archetype.HONEST, Archetype.MEDIOCRE, Archetype.DECEPTIVE)


def generate(n: int, seed: int):
    """Draw interactions, keeping the observables the grid path discards."""
    rng = np.random.default_rng(seed)
    agents = [SimAgent.make(a.value, a) for a in ARCHETYPES]
    obs_list, outcomes, labels = [], [], []
    for _ in range(n):
        agent = agents[rng.integers(len(agents))]
        v = agent.sample_outcome(rng)
        obs_list.append(agent.emit_observables(v, rng))
        outcomes.append(v)
        labels.append(agent.archetype.value)
    return obs_list, outcomes, labels


def run_grid(n: int, seed: int, tau: float):
    """The status quo: 1-D sweep, everything else pinned at hand-set defaults."""
    scales = [round(s, 3) for s in np.linspace(0.25, 6.0, 24)]
    reports = sweep_evidence_scale(
        scales, n=n, archetypes=ARCHETYPES, seed=seed, tau=tau
    )
    rows = [
        {
            "evidence_scale": s,
            "pit_deviation": r.pit_deviation,
            "tail_ece": r.tail_ece,
            "crps": r.crps,
            "sharpness": r.sharpness,
        }
        for s, r in reports
    ]
    best_pit = min(rows, key=lambda r: r["pit_deviation"])
    best_crps = min(rows, key=lambda r: r["crps"])
    # The docstring's rule: sharpest scale that stays calibrated. "Stays
    # calibrated" needs a threshold, and the module never gives one, so we use
    # within 10% of the best PIT deviation.
    cutoff = best_pit["pit_deviation"] * 1.10
    admissible = [r for r in rows if r["pit_deviation"] <= cutoff]
    sharpest = max(admissible, key=lambda r: r["sharpness"])
    return rows, best_pit, best_crps, sharpest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=3000, help="interactions")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tau", type=float, default=0.4, help="governance threshold")
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.out or Path("runs") / f"{stamp}_hmc_vs_grid_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"== data: n={args.n} seed={args.seed} archetypes={[a.value for a in ARCHETYPES]}")
    obs_list, outcomes, labels = generate(args.n, args.seed)
    design = design_from_observables(obs_list, outcomes)
    if design.n_clamped:
        print(f"   {design.n_clamped} boundary outcome(s) nudged inward")

    # ---------------------------------------------------------------- grid
    print("\n== grid sweep over evidence_scale (the status quo)")
    rows, best_pit, best_crps, sharpest = run_grid(args.n, args.seed, args.tau)
    print(f"   best PIT deviation : scale={best_pit['evidence_scale']:.3f} "
          f"pit={best_pit['pit_deviation']:.4f}")
    print(f"   best CRPS          : scale={best_crps['evidence_scale']:.3f} "
          f"crps={best_crps['crps']:.4f}")
    print(f"   sharpest-calibrated: scale={sharpest['evidence_scale']:.3f} "
          f"sharpness={sharpest['sharpness']:.2f}")

    # ---------------------------------------------------------------- HMC
    print(f"\n== HMC over all {len(PARAM_NAMES)} parameters")
    post = ProxyPosterior(design)
    rng = np.random.default_rng(args.seed)
    res = hmc.sample(
        post,
        init=post.dispersed_inits(args.chains, rng),
        n_draws=args.draws,
        n_warmup=args.warmup,
        n_chains=args.chains,
        seed=args.seed + 99,
        progress=lambda m: print(f"   {m}"),
    )
    print("\n" + res.diagnostics.summary())

    nat = natural_draws(res.draws).reshape(-1, len(NATURAL_NAMES))
    lo, med, hi = np.quantile(nat, [0.05, 0.5, 0.95], axis=0)

    print("\n== posterior marginals (natural scale, 90% CI)")
    print(f"   {'parameter':<20} {'median':>9} {'5%':>9} {'95%':>9}  width")
    for i, name in enumerate(NATURAL_NAMES):
        print(f"   {name:<20} {med[i]:>9.3f} {lo[i]:>9.3f} {hi[i]:>9.3f}"
              f"  {hi[i] - lo[i]:.3f}")

    # -------------------------------------------------- the decisive checks
    print("\n== check 1: posterior correlation (unconstrained coordinates)")
    corr = np.corrcoef(res.flat.T)
    off = np.abs(corr - np.eye(len(PARAM_NAMES)))
    strong = [
        (PARAM_NAMES[i], PARAM_NAMES[j], float(corr[i, j]))
        for i in range(len(PARAM_NAMES))
        for j in range(i + 1, len(PARAM_NAMES))
        if abs(corr[i, j]) > 0.3
    ]
    print(f"   max |off-diagonal correlation| = {off.max():.3f}")
    if strong:
        for a, b, c in sorted(strong, key=lambda t: -abs(t[2])):
            print(f"   {a:>12} <-> {b:<12} r = {c:+.3f}")
        print("   -> a per-axis grid cannot reach the joint optimum of these.")
    else:
        print("   -> no strong correlations; per-axis search loses little here.")

    print("\n== check 2: evidence_scale, grid point vs posterior marginal")
    s_idx = NATURAL_NAMES.index("evidence_scale")
    s_lo, s_med, s_hi = lo[s_idx], med[s_idx], hi[s_idx]
    grid_s = sharpest["evidence_scale"]
    inside = s_lo <= grid_s <= s_hi
    print(f"   grid (sharpest-calibrated) = {grid_s:.3f}")
    print(f"   posterior median = {s_med:.3f}  90% CI = [{s_lo:.3f}, {s_hi:.3f}]")
    print(f"   -> grid value is {'INSIDE' if inside else 'OUTSIDE'} the 90% CI")

    print("\n== check 3: tail mass P(v < tau) — the governance lever")
    n_show = min(400, len(obs_list))
    tm = posterior_tail_mass(res.draws, obs_list[:n_show], tau=args.tau,
                             max_draws=300, seed=args.seed)
    tm_lo, tm_med, tm_hi = np.quantile(tm, [0.05, 0.5, 0.95], axis=0)
    width = tm_hi - tm_lo
    # Would the interval straddle a decision line the point estimate clears?
    straddle = int(np.sum((tm_lo < 0.5) & (tm_hi > 0.5)))
    confident_point = int(np.sum(np.abs(tm_med - 0.5) > 0.1))
    print(f"   interactions scored: {n_show}")
    print(f"   90% CI width: median {np.median(width):.4f}  max {width.max():.4f}")
    print(f"   {straddle} interaction(s) have a CI straddling the 0.5 line")
    print(f"   {confident_point} would look confident on the point estimate alone")
    flipped = int(np.sum((np.abs(tm_med - 0.5) > 0.1) & (tm_lo < 0.5) & (tm_hi > 0.5)))
    print(f"   -> {flipped} case(s) where the point estimate reads confident but "
          f"the posterior does not")

    print("\n== check 4: calibration at posterior mean vs at grid optimum")
    theta_hat = ProxyTheta.from_vector(res.mean())
    cal_rows = {}
    for tag, proxy in (
        ("posterior_mean", theta_hat.to_proxy()),
        ("grid_sharpest", ProxyTheta.from_vector(post.prior_mean).to_proxy()),
    ):
        if tag == "grid_sharpest":
            proxy.evidence_scale = grid_s
            proxy.__post_init__()
        beliefs = [proxy.compute_belief(o) for o in obs_list]
        rep = calibrate(beliefs, outcomes, tau=args.tau)
        cal_rows[tag] = {
            "pit_deviation": rep.pit_deviation,
            "tail_ece": rep.tail_ece,
            "crps": rep.crps,
            "sharpness": rep.sharpness,
        }
        print(f"   {tag:<15} pit={rep.pit_deviation:.4f} tail_ece={rep.tail_ece:.4f} "
              f"crps={rep.crps:.4f} sharp={rep.sharpness:.2f}")

    # ---------------------------------------------------------------- save
    d = res.diagnostics
    payload = {
        "generated_utc": stamp,
        "config": {
            "n": args.n, "seed": args.seed, "tau": args.tau,
            "draws": args.draws, "warmup": args.warmup, "chains": args.chains,
            "archetypes": [a.value for a in ARCHETYPES],
            "outcome_source": "archetype emission models (proxy is misspecified)",
        },
        "grid": {
            "rows": rows,
            "best_pit": best_pit,
            "best_crps": best_crps,
            "sharpest_calibrated": sharpest,
        },
        "hmc": {
            "diagnostics": {
                "divergences": d.divergences,
                "max_tree_depth_hits": d.max_tree_depth_hits,
                "step_size": d.step_size,
                "accept_rate": d.accept_rate,
                "ebfmi": d.ebfmi,
                "rhat": d.rhat.tolist(),
                "ess": d.ess.tolist(),
                "warnings": d.warnings(),
            },
            "param_names": list(NATURAL_NAMES),
            "median": med.tolist(),
            "ci_5": lo.tolist(),
            "ci_95": hi.tolist(),
            "unconstrained_names": list(PARAM_NAMES),
            "correlation": corr.tolist(),
            "strong_correlations": strong,
        },
        "tail_mass": {
            "tau": args.tau,
            "n_scored": n_show,
            "ci_width_median": float(np.median(width)),
            "ci_width_max": float(width.max()),
            "straddling_half": straddle,
            "confident_on_point_estimate": confident_point,
            "point_confident_but_posterior_not": flipped,
        },
        "calibration": cal_rows,
        "evidence_scale_comparison": {
            "grid": grid_s,
            "posterior_median": float(s_med),
            "ci_5": float(s_lo), "ci_95": float(s_hi),
            "grid_inside_ci": bool(inside),
        },
    }
    (out_dir / "result.json").write_text(json.dumps(payload, indent=2))
    np.savez_compressed(
        out_dir / "draws.npz",
        draws=res.draws, log_prob=res.log_prob, energy=res.energy,
        natural=natural_draws(res.draws),
    )
    print(f"\n== wrote {out_dir}/result.json and draws.npz")

    if d.warnings():
        print("\n!! diagnostics reported problems; treat the numbers above as "
              "provisional:")
        for w in d.warnings():
            print(f"   - {w}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
