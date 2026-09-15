from __future__ import annotations

from collections.abc import Generator
from functools import lru_cache
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import beta, norm

import methods as m

KINDS = ("range", "signed", "delta", "budget", "calibrated", "tan1.25", "tan1.5", "tan2.0")
SAMPLE_SIZES = (50, 100, 250, 500, 1000, 2500, 10000)
CAP_SIZES = (50, 250, 1000, 2500, 10000)


def rng(seed: int, *keys: int) -> np.random.Generator:
    """Generate a random number generator.

    Args:
        seed (int): The seed for the random number generator.
        keys (int): The keys for the random number generator.

    Returns:
        np.random.Generator: The random number generator.
    """
    return np.random.default_rng(np.random.SeedSequence([int(seed), *map(int, keys)]))


@lru_cache(None)
def population(rescue: float = 0.5, arm: int = 0, states: int = 4, caps: bool = False) -> dict:
    """Generate a population.

    Args:
        rescue (float, optional): The rescue rate. Defaults to 0.5.
        arm (int, optional): The arm. Defaults to 0.
        states (int, optional): The number of states. Defaults to 4.
        caps (bool, optional): Whether to use caps. Defaults to False.

    Returns:
        dict: The population.
    """
    p = m.finite_population(5, states, rescue, arm)
    if caps:
        odd = np.arange(5) % 2 == 0
        p["qR"][odd] = 0.96
        p["tau"][odd] = 0.02
        p["b"] = p["p"] * p["qR"]
        p["mu"] = float(p["b0"] + p["b"].sum())
        p["theta"] = float(p["mu"] + p["p"] @ p["tau"])
    return p


def draw(
    pop: dict, n: int, cfg: dict, *keys: int, seed: int | None = None
) -> tuple[NDArray[np.int32], NDArray[np.float64], NDArray[np.float64]]:
    """Separate crossing and common outcome-uniform streams.

    Args:
        pop (dict): The population.
        n (int): The number of samples.
        cfg (dict): The configuration.
        keys (int): The keys.
        seed (int | None, optional): The seed. Defaults to None.

    Returns:
        tuple[NDArray[np.int32], NDArray[np.float64], NDArray[np.float64]]: The cell, y, and y0.
    """
    base = cfg["seed"] if seed is None else seed
    p = pop["p"]
    M = len(p)
    cell = rng(base, *keys, 0).choice(M + 1, size=n, p=np.r_[p, 1 - p.sum()])
    q = np.r_[pop["qR"], pop["b0"] / (1 - p.sum())]
    q0 = np.r_[pop["qR"] + pop["tau"], q[-1]]
    u = rng(base, *keys, 1).random(n)
    return cell, (u < q[cell]).astype(float), (u < q0[cell]).astype(float)


def draw_continuous_outcome(
    pop: dict, n: int, cfg: dict, *keys: int, phi: float = 10.0
) -> dict[str, NDArray]:
    """Beta outcomes with exact reference means and a common outcome uniform.

    Among non-crossers, sample the terminal state from its conditional law;
    their outcome distribution is a mixture of beta laws with state-specific
    means. The observed and no-rescue outcomes coincide on those paths.
    """
    if phi <= 0 or not np.isfinite(phi):
        raise ValueError("Beta precision must be positive and finite.")
    M, base = len(pop["p"]), cfg["seed"]
    cell = rng(base, *keys, 0).choice(M + 1, size=n, p=np.r_[pop["p"], pop["p0"]])
    terminal_state = np.full(n, -1, dtype=np.int16)
    noncrossing = cell == M
    terminal_state[noncrossing] = rng(base, *keys, 2).choice(
        pop["J"],
        size=int(noncrossing.sum()),
        p=pop["noncrossing_state_mass"] / pop["p0"],
    )
    q = np.empty(n)
    q_no_rescue = np.empty(n)
    q[~noncrossing] = pop["qR"][cell[~noncrossing]]
    q_no_rescue[~noncrossing] = q[~noncrossing] + pop["tau"][cell[~noncrossing]]
    q[noncrossing] = pop["noncrossing_outcome_mean"][terminal_state[noncrossing]]
    q_no_rescue[noncrossing] = q[noncrossing]
    u = rng(base, *keys, 1).random(n)
    return {
        "cell": cell.astype(np.int8),
        "y": beta.ppf(u, phi * q, phi * (1 - q)),
        "y_no_rescue": beta.ppf(u, phi * q_no_rescue, phi * (1 - q_no_rescue)),
        "terminal_state": terminal_state,
    }


def diagnostics(bounds: tuple[float, float]) -> dict:
    """Compute the diagnostics.

    Args:
        bounds (tuple[float, float]): The bounds.

    Returns:
        dict: The diagnostics.
    """
    return m.bound_diagnostics(bounds)


def record(
    rows: list[dict],
    key: dict,
    kind: str,
    method: str,
    interval: list[float],
    truth: dict,
    pop: dict,
    **extra: dict,
) -> None:
    """Record the result.

    Args:
        rows (list[dict]): The rows.
        key (dict): The key.
        kind (str): The kind.
        method (str): The method.
        interval (list[float]): The interval.
        truth (dict): The truth.
        pop (dict): The population.
        extra (dict): The extra.
    """
    m.append_result(
        rows,
        key,
        kind,
        method,
        interval,
        truth,
        pop["theta"],
        population_lower=truth[0],
        population_upper=truth[1],
        theta=pop["theta"],
        **extra,
    )


def moments(
    cell: NDArray[np.int32], y: NDArray[np.float64], ynr: NDArray[np.float64], key: dict, M: int = 5
) -> list[dict]:
    """Sufficient counts retain empirical moments and outcome coupling.

    Args:
        cell (NDArray[np.int32]): The cell.
        y (NDArray[np.float64]): The y.
        ynr (NDArray[np.float64]): The ynr.
        key (dict): The key.
        M (int, optional): The number of states. Defaults to 5.

    Returns:
        list[dict]: The moments.
    """
    atom = 4 * cell + 2 * y.astype(int) + ynr.astype(int)
    counts = np.bincount(atom, minlength=4 * (M + 1))
    return [
        dict(**key, cell=j, y=yr, y_no_rescue=y0, count=int(counts[4 * j + 2 * yr + y0]))
        for j in range(M + 1)
        for yr in (0, 1)
        for y0 in (0, 1)
    ]


def core_jobs(cfg: dict) -> Generator:
    """Generate the core jobs.

    Args:
        cfg (dict): The configuration.

    Returns:
        Generator: The core jobs.
    """
    suites = cfg["suites"]
    R = cfg["finite_reps"]

    def job(kind: str, rep: int, **kw: Any) -> dict:
        key = "_".join(f"{k}-{v}" for k, v in sorted(kw.items()))
        return {
            "id": f"{kind}_{key}_rep-{rep}",
            "suite": "core",
            "task": kind,
            "rep": rep,
            "weight": 1,
            "config": cfg,
            **kw,
        }

    if "core" in suites:
        for rescue in (0.5, 0.75):
            for rep in range(R):
                yield job("primary", rep, rescue=rescue)
        for n in SAMPLE_SIZES:
            for rep in range(R):
                yield job("sample_size", rep, n=n)
        for n in CAP_SIZES:
            for rep in range(R):
                yield job("active_caps", rep, n=n)
        for rep in range(R):
            yield job("external", rep, n=1000)
    if "continuous_outcome" in suites:
        for n in (250, 1000, 10000):
            for rep in range(cfg.get("continuous_outcome_reps", R)):
                yield job("continuous_outcome", rep, n=n)
    if "gamma" in suites:
        for n in (50, 1000, 10000):
            for rep in range(R):
                yield job("gamma", rep, n=n)
    if "rare" in suites:
        for ei, expected in enumerate((0.0, 0.5, 2.0, 10.0)):
            for start in range(0, cfg["rare_reps"], cfg["rare_chunk"]):
                count = min(cfg["rare_chunk"], cfg["rare_reps"] - start)
                j = job("rare", start, expected=expected, ei=ei, count=count)
                j["weight"] = count
                yield j


def primary_job(job: dict) -> None:
    """Run the primary job.

    Args:
        job (dict): The job.
    """
    cfg, rep, rescue = job["config"], job["rep"], job["rescue"]
    raw, counts, contrasts, pair = [], [], [], {}
    for arm in (0, 1):
        pop = population(rescue, arm)
        truth = {k: m.exact_population(pop, k) for k in KINDS}
        cell, y, y0 = draw(pop, 1000, cfg, 10, int(100 * rescue), rep, arm)
        F = m.features(cell, y, 5)
        key = {
            "design": "finite",
            "rescue": rescue,
            "arm": arm,
            "n": 1000,
            "rep": rep,
        }
        counts += moments(cell, y, y0, key)
        H = m.calibration_matrix(5)
        dc = H @ (pop["p"] * pop["tau"])
        pl, pu = m.signed_budget_exact(F.mean(0)[0], F.mean(0)[1:6], F.mean(0)[6:])
        pe = max(abs(pl - truth["budget"][0]), abs(pu[0] - truth["budget"][1]))
        for kind in ("range", "signed"):
            record(raw, key, kind, "endpoint_cp", m.endpoint_cp(cell, y, 5, kind), truth[kind], pop)
        for lam in (1.25, 1.5, 2.0):
            kind = "tan" + str(lam)
            record(
                raw, key, kind, "conditional_cp", m.tan_interval(cell, y, 5, lam), truth[kind], pop
            )
        for kind in ("delta", "budget", "calibrated"):
            methods = ("cp", "kl", "eb", "hoeffding", "wald") if kind == "budget" else ("cp",)
            for method in methods:
                mean, lo, hi, al, au = m.cell_region(F, method)
                kw = {
                    "gamma": None if kind == "delta" else 0.2,
                    "aggregate_lo": al,
                    "aggregate_hi": au,
                }
                if kind == "delta":
                    kw["effect_upper"] = np.full(5, 0.2)
                if kind == "calibrated":
                    kw.update(H=H, dlo=dc, dhi=dc)
                rr = m.region(mean, lo, hi, 5, **kw)
                record(
                    raw,
                    key,
                    kind,
                    method,
                    [z.value for z in rr],
                    truth[kind],
                    pop,
                    plugin_error=pe if kind == "budget" else None,
                    **diagnostics(rr),
                )
        mean = F.mean(0)
        for aggregate in (False, True):
            rr = m.region(
                mean,
                np.zeros_like(mean),
                np.ones_like(mean),
                5,
                extra_linear=m.atomic_constraints(cell, y, 5, aggregate=aggregate),
            )
            record(
                raw,
                key,
                "budget",
                "atomic_kl_aggregate" if aggregate else "atomic_kl",
                [z.value for z in rr],
                truth["budget"],
                pop,
                plugin_error=pe,
                **diagnostics(rr),
            )
        record(
            raw,
            key,
            "budget",
            "aggregate_cp",
            m.endpoint_cp(cell, y, 5, "budget"),
            truth["budget"],
            pop,
            plugin_error=pe,
        )
        interval = m.bootstrap_budget(
            cell,
            y,
            5,
            rng(cfg["bootstrap_seed"], 11, int(100 * rescue), rep, arm),
            cfg["bootstrap_resamples"],
        )
        record(raw, key, "budget", "bootstrap", interval, truth["budget"], pop, plugin_error=pe)
        mean, lo, hi, al, au = m.cell_region(F, "cp", alpha=0.025)
        rr = m.region(mean, lo, hi, 5, aggregate_lo=al, aggregate_hi=au)
        cp_failure = int(any(z.status != "ok" for z in rr))
        pair[arm] = {
            "cp": [0.0, 1.0] if cp_failure else [z.value for z in rr],
            "aggregate_cp": m.endpoint_cp(cell, y, 5, "budget", 0.025),
            "bootstrap": m.bootstrap_budget(
                cell,
                y,
                5,
                rng(cfg["bootstrap_seed"], 12, int(100 * rescue), rep, arm),
                cfg["bootstrap_resamples"],
                0.025,
            ),
            "truth": truth["budget"],
            "theta": pop["theta"],
            "failure": cp_failure,
        }
    for method in ("cp", "aggregate_cp", "bootstrap"):
        L = pair[1][method][0] - pair[0][method][1]
        U = pair[1][method][1] - pair[0][method][0]
        tl, tu = (
            pair[1]["truth"][0] - pair[0]["truth"][1],
            pair[1]["truth"][1] - pair[0]["truth"][0],
        )
        target = pair[1]["theta"] - pair[0]["theta"]
        failure = int(method == "cp" and (pair[0]["failure"] or pair[1]["failure"]))
        if failure or not np.isfinite(L + U) or L > U:
            L, U, failure = -1.0, 1.0, 1
        contrasts.append(
            {
                "rescue": rescue,
                "n": 1000,
                "rep": rep,
                "method": method,
                "lower": L,
                "upper": U,
                "population_lower": tl,
                "population_upper": tu,
                "theta": target,
                "length": U - L,
                "sharp_width": tu - tl,
                "enlargement": U - L - tu + tl,
                "set_cover": int(L <= tl + 1e-10 and U >= tu - 1e-10),
                "target_cover": int(L <= target <= U),
                "failure": failure,
            }
        )
    return {
        "core/interval_replications.csv": raw,
        "core/finite_counts.csv": counts,
        "core/contrast_replications.csv": contrasts,
    }


def finite_job(job: dict) -> dict:
    """Run the finite job.

    Args:
        job (dict): The job.

    Returns:
        dict: The finite job.
    """
    cfg, rep, n, task = job["config"], job["rep"], job["n"], job["task"]
    rescue = 0.75 if task == "states12" else 0.5
    pop = population(rescue, states=12 if task == "states12" else 4, caps=task == "active_caps")
    kind = "calibrated" if task == "sample_size" else "budget"
    truth = m.exact_population(pop, kind)
    family = {"sample_size": 20, "states12": 22, "active_caps": 23}[task]
    cell, y, y0 = draw(pop, n, cfg, family, n, rep)
    F = m.features(cell, y, 5)
    key = {
        "design": task,
        "rescue": rescue,
        "arm": 0,
        "n": n,
        "rep": rep,
    }
    kw, extra = {}, {}
    if kind == "calibrated":
        H = m.calibration_matrix(5)
        dc = H @ (pop["p"] * pop["tau"])
        kw.update(H=H, dlo=dc, dhi=dc)
        mean = F.mean(0)
        plugin = m.region(mean, mean, mean, 5, **kw)
        extra.update(
            plugin_error=max(abs(plugin[i].value - truth[i]) for i in (0, 1)),
            plugin_failure=int(any(z.status != "ok" for z in plugin)),
        )
    else:
        mean = F.mean(0)
        pl, pu = m.signed_budget_exact(mean[0], mean[1:6], mean[6:])
        extra.update(plugin_error=max(abs(pl - truth[0]), abs(pu[0] - truth[1])), plugin_failure=0)
    mean, lo, hi, al, au = m.cell_region(F, "cp")
    rr = m.region(mean, lo, hi, 5, aggregate_lo=al, aggregate_hi=au, **kw)
    rows = []
    record(rows, key, kind, "cp", [z.value for z in rr], truth, pop, **extra, **diagnostics(rr))
    if task == "active_caps":
        record(rows, key, kind, "aggregate_cp", m.endpoint_cp(cell, y, 5, "budget"), truth, pop)
        record(
            rows,
            key,
            kind,
            "bootstrap",
            m.bootstrap_budget(
                cell, y, 5, rng(cfg["bootstrap_seed"], 24, n, rep), cfg["bootstrap_resamples"]
            ),
            truth,
            pop,
        )
    return {
        "core/interval_replications.csv": rows,
        "core/finite_counts.csv": moments(cell, y, y0, key),
    }


def external_job(job: dict) -> dict:
    """Run the external job.

    Args:
        job (dict): The job.

    Returns:
        dict: The external job.
    """
    cfg, rep = job["config"], job["rep"]
    pop = population()
    H = m.calibration_matrix(5)
    dt = H @ (pop["p"] * pop["tau"])
    truth = m.exact_population(pop, "calibrated")
    cell, y, y0 = draw(pop, 1000, cfg, 50, rep)
    F = m.features(cell, y, 5)
    rows, scores = [], []
    for nex in (0, 250, 2000, 20000, -1):
        mean, lo, hi, al, au = m.cell_region(F, "cp", extra_D=2 if nex > 0 else 0)
        hh = dl = du = None
        if nex == -1:
            hh, dl, du = H, dt, dt
        elif nex > 0:
            ce, yr, ynr = draw(pop, nex, cfg, 51, nex, rep, seed=cfg["external_seed"])
            e = rng(cfg["external_seed"], 52, nex, rep).integers(0, 2, nex)
            yy = np.where(e == 0, ynr, yr)
            Z = np.zeros((nex, 2))
            hit = ce < 5
            Z[hit] = H[:, ce[hit]].T * (2 * (1 - 2 * e[hit]) * yy[hit])[:, None]
            _, dl, du = m.moment_box(Z, "eb", total_D=F.shape[1] + 4, lengths=np.full(2, 4.0))
            hh = H
            for group in range(2):
                scores.append(
                    {
                        "rep": rep,
                        "n_external": nex,
                        "group": group,
                        "score_mean": Z[:, group].mean(),
                        "score_variance": Z[:, group].var(ddof=1),
                        "lower": dl[group],
                        "upper": du[group],
                        "population_target": dt[group],
                    }
                )
        rr = m.region(mean, lo, hi, 5, H=hh, dlo=dl, dhi=du, aggregate_lo=al, aggregate_hi=au)
        tt = m.exact_population(pop, "budget") if nex == 0 else truth
        record(
            rows,
            {
                "experiment": "external",
                "setting": nex,
                "rep": rep,
            },
            "calibrated" if nex else "budget",
            "cp_eb",
            [z.value for z in rr],
            tt,
            pop,
            **diagnostics(rr),
        )
    return {
        "core/external_radius_replications.csv": rows,
        "core/external_scores.csv": scores,
        "core/finite_counts.csv": moments(
            cell,
            y,
            y0,
            {
                "design": "external",
                "rescue": 0.5,
                "arm": 0,
                "n": 1000,
                "rep": rep,
            },
        ),
    }


def radius_job(job: dict) -> dict:
    """Run the radius job.

    Args:
        job (dict): The job.

    Returns:
        dict: The radius job.
    """
    cfg, rep, n = job["config"], job["rep"], job["n"]
    pop, rows = population(), []
    gt = float(np.sqrt(pop["p"] @ (pop["tau"] ** 2)))
    is_grid = job["task"] == "gamma"
    cell, y, y0 = draw(pop, n, cfg, 80 if is_grid else 53, n, rep)
    F = m.features(cell, y, 5)
    mean, lo, hi, al, au = m.cell_region(F, "cp")
    params = (0.0, 0.05, 0.1, 0.2, 0.35) if is_grid else (0.0, 0.5, 1.0, 1.5)
    for param in params:
        gamma = param if is_grid else gt * param
        tl, tu = m.signed_budget_exact(pop["mu"], pop["p"], pop["b"], gamma)
        rr = m.region(mean, lo, hi, 5, gamma=gamma, aggregate_lo=al, aggregate_hi=au, refine=False)
        key = {
            "n": n,
            "rep": rep,
            "gamma": gamma,
            "true_norm": gt,
            "model_contains_truth": int(gamma >= gt),
        }
        if not is_grid:
            key.update(experiment="radius", setting=param)
        record(
            rows,
            key,
            "budget",
            "cp",
            [z.value for z in rr],
            (float(tl), float(tu[0])),
            pop,
            **diagnostics(rr),
        )
    name = "gamma/gamma_replications.csv" if is_grid else "core/external_radius_replications.csv"
    return {
        name: rows,
        "core/finite_counts.csv": moments(
            cell,
            y,
            y0,
            {
                "design": "gamma" if is_grid else "radius",
                "rescue": 0.5,
                "arm": 0,
                "n": n,
                "rep": rep,
            },
        ),
    }


def rare_job(job: dict) -> dict:
    """Run the rare job.

    Args:
        job (dict): The job.

    Returns:
        dict: The rare job.
    """
    cfg, n, rows = job["config"], 1000, []
    expected, ei = job["expected"], job["ei"]
    p = expected / n
    for rep in range(job["rep"], job["rep"] + job["count"]):
        k = int(rng(cfg["seed"], 40, ei, rep).binomial(n, p))
        ph = k / n
        var = n / (n - 1) * ph * (1 - ph)
        boot = (
            rng(cfg["bootstrap_seed"], 40, ei, rep).binomial(n, ph, cfg["bootstrap_resamples"]) / n
        )
        upper = {
            "cp": float(m.cp_counts(k, n)[1]),
            "kl": m.kl_counts_scalar(k, n)[1],
            "eb": min(1.0, ph + np.sqrt(2 * var * np.log(80) / n) + 7 * np.log(80) / (3 * (n - 1))),
            "hoeffding": min(1.0, ph + np.sqrt(np.log(40) / (2 * n))),
            "wald": min(1.0, ph + norm.ppf(0.975) * np.sqrt(var / n)),
            "bootstrap": min(
                1.0, ph + max(0.0, float(np.quantile(ph - boot, 0.95, method="higher")))
            ),
        }
        for method, up in upper.items():
            rows.append(
                {
                    "expected_count": expected,
                    "n": n,
                    "rep": rep,
                    "method": method,
                    "count": k,
                    "population_lower": 0.0,
                    "population_upper": p,
                    "lower": 0.0,
                    "upper": up,
                    "set_cover": int(up >= p),
                    "upper_mass": up,
                    "empty_rare": int(k == 0),
                    "failure": 0,
                }
            )
    return {"core/rare_replications.csv": rows}


def continuous_outcome_job(job: dict) -> dict:
    """Paired bounded-outcome comparison with the binary reference sharp set."""
    cfg, n, rep = job["config"], job["n"], job["rep"]
    pop, phi, rows = population(), 10.0, []
    sample = draw_continuous_outcome(pop, n, cfg, 90, n, rep, phi=phi)
    cell, y = sample["cell"], sample["y"]
    F = m.features(cell, y, 5)
    mean = F.mean(0)
    truth = m.exact_population(pop, "budget")
    lower, upper = m.signed_budget_exact(mean[0], mean[1:6], mean[6:])
    plugin_lower, plugin_upper = float(lower), float(upper[0])
    extra = {
        "plugin_lower": plugin_lower,
        "plugin_upper": plugin_upper,
        "plugin_error": max(abs(plugin_lower - truth[0]), abs(plugin_upper - truth[1])),
        "plugin_failure": 0,
    }
    key = {
        "design": "continuous_outcome", "rescue": 0.5, "arm": 0,
        "n": n, "rep": rep, "phi": phi, "sample_id": job["id"],
    }
    for method in ("hybrid_eb_cp", "hybrid_hoeffding_cp", "wald"):
        mean, lo, hi, al, au = m.cell_region(F, method)
        rr = m.region(
            mean, lo, hi, 5, gamma=0.2, aggregate_lo=al, aggregate_hi=au,
            mesh=cfg["mesh"], refine=False,
        )
        record(
            rows, key, "budget", method, [z.value for z in rr], truth, pop,
            **extra, **diagnostics(rr),
        )
    interval = m.bootstrap_budget(
        cell, y, 5, rng(cfg["bootstrap_seed"], 91, n, rep), cfg["bootstrap_resamples"],
    )
    record(rows, key, "budget", "bootstrap", interval, truth, pop, **extra)
    return {
        "continuous_outcome/continuous_outcome_replications.csv": rows,
        "_samples": sample,
    }


def run_core_job(job: dict) -> dict:
    """Run the core job.

    Args:
        job (dict): The job.

    Returns:
        dict: The core job.
    """
    m.DEFAULT_MESH = job["config"]["mesh"]
    task = job["task"]
    if task == "primary":
        return primary_job(job)
    if task in ("sample_size", "states12", "active_caps"):
        return finite_job(job)
    if task == "external":
        return external_job(job)
    if task in ("radius", "gamma"):
        return radius_job(job)
    if task == "rare":
        return rare_job(job)
    if task == "continuous_outcome":
        return continuous_outcome_job(job)
    raise ValueError(f"Unknown core task: {task}")


def deterministic_audits(cfg: dict) -> dict:
    """Population truth, structural equivalence, and fixed-mesh diagnostics.

    Args:
        cfg (dict): The configuration.

    Returns:
        dict: The deterministic audits.
    """
    m.DEFAULT_MESH = cfg["mesh"]
    files = {
        "core/geometry.csv": m.geometry_audit(),
        "core/completion_audit.csv": [
            m.completion_audit(k, s, mesh=cfg["mesh"]) for k, s in ((3, 4), (5, 4), (5, 6))
        ],
    }
    tanaudit, truths, popmom = [], [], []
    for rescue in (0.5, 0.75):
        for arm in (0, 1):
            pop = population(rescue, arm)
            for lam in (1.0, 1.25, 1.5, 2.0):
                tanaudit.append(
                    {
                        "rescue": rescue,
                        "arm": arm,
                        "Lambda": lam,
                        "max_difference": m.tan_lp_audit(pop, lam),
                        "minimum_true_lambda": m.tan_minimum_lambda(pop),
                    }
                )
            for kind in KINDS:
                L, U = m.exact_population(pop, kind)
                truths.append(
                    {
                        "design": "finite",
                        "rescue": rescue,
                        "arm": arm,
                        "kind": kind,
                        "lower": L,
                        "upper": U,
                        "sharp_width": U - L,
                        "theta": pop["theta"],
                        "mu": pop["mu"],
                        "gamma_true": float(np.sqrt(pop["p"] @ (pop["tau"] ** 2))),
                        "minimum_true_lambda": m.tan_minimum_lambda(pop),
                    }
                )
            for j in range(5):
                popmom.append(
                    {
                        "design": "finite",
                        "rescue": rescue,
                        "arm": arm,
                        "cell": j,
                        "p": pop["p"][j],
                        "b": pop["b"][j],
                        "tau": pop["tau"][j],
                        "mu": pop["mu"],
                        "theta": pop["theta"],
                        "b0": pop["b0"],
                        "alpha": pop["alpha"],
                    }
                )
    for design, pop, kind, res in (
        ("sample_size", population(), "calibrated", 0.5),
        ("active_caps", population(caps=True), "budget", 0.5),
    ):
        L, U = m.exact_population(pop, kind)
        truths.append(
            {
                "design": design,
                "rescue": res,
                "arm": 0,
                "kind": kind,
                "lower": L,
                "upper": U,
                "sharp_width": U - L,
                "theta": pop["theta"],
                "mu": pop["mu"],
                "gamma_true": float(np.sqrt(pop["p"] @ (pop["tau"] ** 2))),
                "aggregate_width": min(0.2 * np.sqrt(pop["p"].sum()), 1 - pop["mu"]),
            }
        )
        for j in range(5):
            popmom.append(
                {
                    "design": design,
                    "rescue": res,
                    "arm": 0,
                    "cell": j,
                    "p": pop["p"][j],
                    "b": pop["b"][j],
                    "tau": pop["tau"][j],
                    "mu": pop["mu"],
                    "theta": pop["theta"],
                }
            )
    files.update(
        {
            "core/tan_audit.csv": tanaudit,
            "core/population_regions.csv": truths,
            "core/population_moments.csv": popmom,
        }
    )
    pop, curves, meshrows = population(), [], []
    for gamma in np.linspace(0, 0.35, 36):
        L, U = m.signed_budget_exact(pop["mu"], pop["p"], pop["b"], gamma)
        curves.append(
            {
                "model": "budget",
                "parameter": gamma,
                "lower": float(L),
                "upper": float(U[0]),
            }
        )
    for lam in np.linspace(1, 2.5, 31):
        L, U = m.tan_population(pop, lam)
        curves.append(
            {
                "model": "tan",
                "parameter": lam,
                "lower": L,
                "upper": U,
            }
        )
    v = np.r_[pop["mu"], pop["p"], pop["b"]]
    for h in (1 / 8, 1 / 16, 1 / 32, 1 / 64, 1 / 128):
        rr = m.region(v, v, v, 5, mesh=h, refine=False)
        inn = m.region(v, v, v, 5, mesh=h, inner=True, refine=False)
        meshrows.append(
            {
                "mesh": h,
                "budget_error": h * h / 4,
                "outer_lower": rr[0].value,
                "outer_upper": rr[1].value,
                "inner_lower": inn[0].primal_value,
                "inner_upper": inn[1].primal_value,
                "bracket_gap": max(
                    inn[0].primal_value - rr[0].value, rr[1].value - inn[1].primal_value
                ),
                "inner_violation": max(z.violation for z in inn),
                "inner_witness_checked": int(
                    all(z.status == "ok" and z.violation <= 1e-9 for z in inn)
                ),
                "dual_gap": max(z.dual_gap for z in rr),
            }
        )
    files["core/sensitivity_curves.csv"] = curves
    files["core/mesh_audit.csv"] = meshrows
    files["core/relaxation_audit.csv"] = [
        {
            "sharp_budget_width": m.exact_population(pop, "budget")[1] - pop["mu"],
            "independent_budget_width": float(
                np.minimum(pop["p"] - pop["b"], 0.2 * np.sqrt(pop["p"])).sum()
            ),
        }
    ]
    return files
