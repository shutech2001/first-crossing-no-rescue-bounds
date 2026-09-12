from __future__ import annotations

import itertools
import json
import math
from typing import Iterable

import numpy as np
from scipy.special import expit, ndtr, ndtri
from scipy.stats import norm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

import methods as fm

CONTINUOUS_CONFIGS = [(5, 5, 0.0), (5, 5, 0.65), (5, 50, 0.35), (8, 5, 0.35)]
OBS_SIZES = [50, 250, 1000, 10000]
PART_SIZES = [250, 1000, 10000]
PARTITIONS = [1, 5, 10, 20, 40, 80]
OBS_METHODS = [
    "oracle_eb",
    "oracle_wald",
    "logit_naive",
    "logit_sandwich",
    "main_sandwich",
    "rf_cf_naive",
    "rf_cf_eb",
]


def _rng(master, *keys):
    return np.random.default_rng(np.random.SeedSequence([int(master), *map(int, keys)]))


def _int_seed(master, *keys):
    return int(np.random.SeedSequence([int(master), *map(int, keys)]).generate_state(1)[0])


def _suites(config):
    selected = config.get("suites", ["observational", "continuous", "partitions"])
    if isinstance(selected, str):
        selected = selected.split(",")
    return {"observational", "continuous", "partitions"} if "all" in selected else set(selected)


def preparation_jobs(config) -> Iterable[dict]:
    """One counted reference task for each selected independent population law."""
    selected = _suites(config)
    if "observational" in selected:
        yield dict(
            suite="prepare_observational",
            key="reference_observational",
            id="reference_observational",
            config=config,
        )
    if "continuous" in selected:
        for ci, (K, d, rho) in enumerate(CONTINUOUS_CONFIGS):
            yield dict(
                suite="prepare_continuous",
                key=f"reference_continuous_{ci}",
                id=f"reference_continuous_{ci}",
                ci=ci,
                K=K,
                d=d,
                rho=rho,
                config=config,
            )
    if "partitions" in selected:
        yield dict(
            suite="prepare_partitions",
            key="reference_partitions",
            id="reference_partitions",
            config=config,
        )


def prepare_extensions(config):
    """Sequential convenience wrapper; the runner normally dispatches reference jobs."""
    context, files = {}, {}
    for job in preparation_jobs(config):
        partial, records = run_preparation_job(job)
        context.update(partial)
        for filename, rows in records.items():
            files.setdefault(filename, []).extend(rows)
    return context, files


def extension_jobs(config, context) -> Iterable[dict]:
    selected = _suites(config)
    if "observational" in selected:
        for scenario in ("linear", "nonlinear"):
            for n in config.get("observational_sizes", OBS_SIZES):
                for rep in range(config.get("observational_reps", 250)):
                    yield dict(
                        suite="observational",
                        key=f"observational_{scenario}_{n}_{rep}",
                        id=f"observational_{scenario}_{n}_{rep}",
                        scenario=scenario,
                        n=n,
                        rep=rep,
                        pop=context["observational"],
                        config=config,
                    )
    if "continuous" in selected:
        for ci, (K, d, rho) in enumerate(CONTINUOUS_CONFIGS):
            for rep in range(config.get("continuous_reps", 250)):
                yield dict(
                    suite="continuous",
                    key=f"continuous_{ci}_{rep}",
                    ci=ci,
                    K=K,
                    d=d,
                    id=f"continuous_{ci}_{rep}",
                    rho=rho,
                    n=config.get("continuous_n", 1000),
                    rep=rep,
                    pop=context[f"continuous_{ci}"],
                    config=config,
                )
    if "partitions" in selected:
        for n in config.get("partition_sizes", PART_SIZES):
            for rep in range(config.get("partition_reps", 250)):
                yield dict(
                    suite="partitions",
                    key=f"partitions_{n}_{rep}",
                    n=n,
                    rep=rep,
                    id=f"partitions_{n}_{rep}",
                    pop=context["partitions"],
                    config=config,
                )


def continuous_paths(n, rng, K=5, d=5, rho=0.35, c0=0.6, supported=False):
    """No-rescue path probabilities, using the paper's correlated innovations.

    The maximum centered biomarker permits one common-random-number threshold
    calibration without rerunning paths. Outcome uniforms are drawn separately.
    """
    W = rng.normal(size=(n, d))
    for j in range(1, d):
        W[:, j] = 0.35 * W[:, j - 1] + np.sqrt(1 - 0.35**2) * W[:, j]
    S = 0.4 * W[:, 0] - 0.2 * W[:, 1]
    B = 0.3 * W[:, 0] + 0.2 * W[:, 2]
    ss = np.zeros(n)
    T = np.full(n, K, dtype=int)
    exceedance = np.zeros(n)
    max_centered = np.full(n, -np.inf)
    for k in range(K):
        mb = 0.45 * np.tanh(B) + 0.25 * np.tanh(S) + 0.10 * np.tanh(W[:, 1])
        threshold = c0 + 0.05 * (k + 1) / K
        if supported:
            truncation_probability = ndtr((threshold - mb) / 0.6)
            eb = ndtri(
                np.clip(
                    rng.random(n) * truncation_probability,
                    np.finfo(float).tiny,
                    np.nextafter(1.0, 0.0),
                )
            )
        else:
            eb = rng.normal(size=n)
        es = rho * eb + np.sqrt(1 - rho**2) * rng.normal(size=n)
        S = 0.55 * S + 0.15 * W[:, 0] + 0.10 * np.sin(S) + 0.60 * es
        B = mb + 0.60 * eb
        ss += S
        max_centered = np.maximum(max_centered, B - 0.05 * (k + 1) / K)
        hit = (T == K) & (B >= threshold)
        T[hit] = k
        exceedance[hit] = B[hit] - threshold
    eta = (
        -0.20
        + 0.50 * S
        + 0.20 * ss / K
        + 0.15 * W[:, 2]
        + 0.10 * np.sin(W[:, 0])
        + 0.10 * W[:, 3] * W[:, 4]
    )
    q0 = expit(eta)
    shift = np.where(T < K, 0.7 * (1 + 0.50 * (K - 1 - T) / K + 0.25 * exceedance), 0)
    return dict(W=W, T=T, ex=exceedance, qR=expit(eta - shift), q0=q0, max_centered=max_centered)


def _continuous_cells(dat, K):
    return np.where(
        dat["T"] < K, 4 * dat["T"] + 2 * (dat["W"][:, 0] >= 0) + (dat["ex"] > 0.5), 4 * K
    ).astype(int)


def _calibration(config, suite, ci=0, K=5, d=5, rho=0.35):
    n = int(config.get("calibration_n", 100000))
    if n < 2:
        raise ValueError("calibration_n must be at least 2")
    master = int(config.get("oracle_seed", 4200))
    dat = continuous_paths(n, _rng(master, 1, suite, ci), K=K, d=d, rho=rho, c0=0)
    maxima = dat["max_centered"]
    c0 = float(np.median(maxima))
    return c0, dict(
        calibration_n=n,
        calibration_rescue=float(np.mean(maxima >= c0)),
        calibration_seed=master,
        calibration_stream=json.dumps([1, suite, ci]),
    )


def _batch_sizes(config):
    total = int(config.get("oracle_n", 2000000))
    if total < 2:
        raise ValueError("oracle_n must be at least 2")
    count = min(total, max(2, int(config.get("oracle_batches", 20)), math.ceil(total / 100000)))
    q, remainder = divmod(total, count)
    return [q + int(bi < remainder) for bi in range(count)]


def _moments(cell, qr, q0, M):
    n = len(cell)
    return dict(
        p=np.bincount(cell, minlength=M + 1)[:M] / n,
        b=np.bincount(cell, weights=qr, minlength=M + 1)[:M] / n,
        t=np.bincount(cell, weights=q0 - qr, minlength=M + 1)[:M] / n,
        mu=float(np.mean(qr)),
        theta=float(np.mean(q0)),
    )


def _wide_moments(pop):
    return {
        f"{name}{j + 1}": float(value)
        for name in ("p", "b", "t")
        for j, value in enumerate(pop[name])
    }


def _population_record(pop):
    low, high = fm.exact_population(pop, "budget")
    p, b, t = pop["p"], pop["b"], pop["t"]
    norm2 = float(np.sum(np.divide(t**2, p, out=np.zeros_like(p), where=p > 0)))
    membership = np.all(t >= -1e-12) and np.all(t <= p - b + 1e-12) and norm2 <= 0.2**2 + 1e-12
    return dict(
        mu=pop["mu"],
        theta=pop["theta"],
        rescue=float(p.sum()),
        lower=low,
        upper=high,
        sharp_width=high - low,
        gamma_true=float(np.sqrt(norm2)),
        full_data_membership=int(membership),
        gamma=0.2,
        **_wide_moments(pop),
    )


def _batch_se(records, name):
    """Batch-means MCSE, weighted for a final batch of a different size."""
    values = np.array([r[name] for r in records], float)
    sizes = np.array([r["n"] for r in records], float)
    center = np.average(values, weights=sizes)
    return float(np.sqrt(np.sum(sizes * (values - center) ** 2) / (len(records) - 1) / sizes.sum()))


def _accumulate(total, batch, weight):
    for name in ("p", "b", "t", "mu", "theta"):
        total[name] = total.get(name, 0.0) + weight * batch[name]


def _finish_reference(pop, records, **metadata):
    pop["tau"] = np.divide(pop["t"], pop["p"], out=np.zeros_like(pop["p"]), where=pop["p"] > 0)
    pop.update(metadata)
    pop.update(
        {
            name + "_mcse": _batch_se(records, name)
            for name in ("mu", "theta", "lower", "upper", "sharp_width")
        }
    )
    pop["oracle_n"] = sum(r["n"] for r in records)
    pop["oracle_batches"] = len(records)
    pop["oracle_uncertainty_method"] = "independent_batch_means_not_certified"
    return pop


def _suite_files(files):
    """Use the same relative archive paths for fresh jobs and resumed jobs."""
    directories = {
        "observational": "observational",
        "continuous": "continuous",
        "partition": "partitions",
    }
    return {f"{directories[name.split('_', 1)[0]]}/{name}": rows for name, rows in files.items()}


def run_preparation_job(job):
    context, files = _run_preparation_job(job)
    return context, _suite_files(files)


def _run_preparation_job(job):
    config = job["config"]
    if job["suite"] == "prepare_observational":
        pop = fm.observational_population()
        # JSON object keys must survive checkpoint save/load without changing type.
        pop["pops"] = {str(arm): value for arm, value in pop["pops"].items()}
        pop["cond"] = {str(arm): value for arm, value in pop["cond"].items()}
        target = pop["pops"]["0"]
        target["t"] = target["p"] * target["tau"]
        rows = []
        for scenario in ("linear", "nonlinear"):
            e = fm.obs_propensity(pop["W"], scenario)
            rows.append(
                dict(
                    scenario=scenario,
                    alpha=pop["alpha"],
                    minimum_arm0_probability=float(np.min(1 - e)),
                    maximum_arm0_probability=float(np.max(1 - e)),
                    minimum_propensity=float(e.min()),
                    maximum_propensity=float(e.max()),
                    reference_method="exact_32_pattern_state_recursion",
                    **_population_record(target),
                )
            )
        baseline = []
        for arm in (0, 1):
            conditional = pop["cond"][str(arm)]
            for wi, W in enumerate(pop["W"]):
                row = dict(
                    arm=arm,
                    pattern=wi,
                    pattern_probability=1 / 32,
                    **{f"W{j+1}": int(w) for j, w in enumerate(W)},
                    p0=float(conditional["p0"][wi]),
                    b0=float(conditional["b0"][wi]),
                    mu=float(conditional["mu"][wi]),
                    propensity_linear=float(fm.obs_propensity(W[None, :], "linear")[0]),
                    propensity_nonlinear=float(fm.obs_propensity(W[None, :], "nonlinear")[0]),
                )
                for name in ("p", "qR", "tau"):
                    row.update(
                        {f"{name}{j+1}": float(v) for j, v in enumerate(conditional[name][wi])}
                    )
                baseline.append(row)
        return {"observational": pop}, {
            "observational_population.csv": rows,
            "observational_baseline_reference.csv": baseline,
        }
    if job["suite"] == "prepare_continuous":
        ci, K, d, rho = (job[k] for k in ("ci", "K", "d", "rho"))
        c0, calibration = _calibration(config, 2, ci, K, d, rho)
        sizes = _batch_sizes(config)
        master = int(config.get("oracle_seed", 4200))
        total, records = {}, []
        for bi, n in enumerate(sizes):
            dat = continuous_paths(n, _rng(master, 2, 2, ci, bi), K, d, rho, c0)
            batch = _moments(_continuous_cells(dat, K), dat["qR"], dat["q0"], 4 * K)
            supported = continuous_paths(
                n, _rng(master, 3, 2, ci, bi), K, d, rho, c0, supported=True
            )
            psi = float(np.mean(supported["q0"]))
            _accumulate(total, batch, n / sum(sizes))
            records.append(
                dict(
                    ci=ci,
                    K=K,
                    d=d,
                    rho=rho,
                    batch=bi,
                    n=n,
                    psi=psi,
                    supported_gap=psi - batch["theta"],
                    oracle_seed=master,
                    path_stream=json.dumps([2, 2, ci, bi]),
                    supported_stream=json.dumps([3, 2, ci, bi]),
                    **_population_record(batch),
                )
            )
        total = _finish_reference(total, records, c0=c0, K=K, d=d, rho=rho, **calibration)
        total["psi"] = float(np.average([r["psi"] for r in records], weights=sizes))
        total["psi_mcse"] = _batch_se(records, "psi")
        total["supported_gap"] = total["psi"] - total["theta"]
        total["supported_gap_mcse"] = _batch_se(records, "supported_gap")
        row = {k: v for k, v in total.items() if np.isscalar(v)}
        row.update(ci=ci, **_population_record(total))
        return {f"continuous_{ci}": total}, {
            "continuous_oracles.csv": [row],
            "continuous_oracle_batches.csv": records,
        }
    if job["suite"] == "prepare_partitions":
        c0, calibration = _calibration(config, 3, rho=0.35)
        sizes = _batch_sizes(config)
        master = int(config.get("oracle_seed", 4200))
        totals, records = {}, []
        for bi, n in enumerate(sizes):
            dat = continuous_paths(n, _rng(master, 2, 3, bi), c0=c0)
            labels = {M: fm.partition_cells(dat, M) for M in PARTITIONS}
            for cap in (0, 1):
                qr, q0 = fm.partition_outcomes(dat, bool(cap))
                for M in PARTITIONS:
                    batch = _moments(labels[M], qr, q0, M)
                    _accumulate(totals.setdefault((cap, M), {}), batch, n / sum(sizes))
                    records.append(
                        dict(
                            cap=cap,
                            M=M,
                            batch=bi,
                            n=n,
                            oracle_seed=master,
                            path_stream=json.dumps([2, 3, bi]),
                            **_population_record(batch),
                        )
                    )
        rows = []
        for (cap, M), pop in totals.items():
            group = [r for r in records if (r["cap"], r["M"]) == (cap, M)]
            _finish_reference(pop, group, c0=c0, **calibration)
            row = {k: v for k, v in pop.items() if np.isscalar(v)}
            row.update(cap=cap, M=M, **_population_record(pop))
            rows.append(row)
        lookup = {(r["cap"], r["M"], r["batch"]): r for r in records}
        contrasts = []
        for bi, n in enumerate(sizes):
            for cap in (0, 1):
                fine = lookup[(cap, 80, bi)]
                for M in PARTITIONS:
                    r = lookup[(cap, M, bi)]
                    contrasts.append(
                        dict(
                            cap=cap,
                            M=M,
                            reference_M=80,
                            batch=bi,
                            n=n,
                            lower_difference=r["lower"] - fine["lower"],
                            upper_difference=r["upper"] - fine["upper"],
                            width_difference=r["sharp_width"] - fine["sharp_width"],
                        )
                    )
        for row in rows:
            fine = next(r for r in rows if (r["cap"], r["M"]) == (row["cap"], 80))
            group = [r for r in contrasts if (r["cap"], r["M"]) == (row["cap"], row["M"])]
            row["coarsening_gap"] = row["upper"] - fine["upper"]
            row["coarsening_gap_mcse"] = _batch_se(group, "upper_difference")
        serializable_pops = {f"{cap}:{M}": pop for (cap, M), pop in totals.items()}
        return {"partitions": dict(c0=c0, pops=serializable_pops)}, {
            "partition_oracles.csv": rows,
            "partition_oracle_batches.csv": records,
            "partition_oracle_paired_contrasts.csv": contrasts,
        }
    raise ValueError(f"Unknown preparation suite {job['suite']}")


def _truth_moments(pop):
    return np.r_[pop["mu"], pop["p"], pop["b"], pop["p"].sum(), pop["mu"] - pop["b"].sum()]


def _bound_diagnostics(bounds):
    result = fm.bound_diagnostics(bounds)
    result["numerical_violation"] = float(max(b.violation for b in bounds))
    return result


def _moment_rows(key, method, raw, plugin, lo, hi, pop, M):
    names = (
        ["mu"] + [f"p{j+1}" for j in range(M)] + [f"b{j+1}" for j in range(M)] + ["rescue", "b0"]
    )
    truth = _truth_moments(pop)
    return [
        dict(
            **key,
            method=method,
            coordinate=name,
            raw_mean=float(raw[q]),
            plugin_mean=float(plugin[q]),
            lower=float(lo[q]),
            upper=float(hi[q]),
            reference=float(truth[q]),
            covered=int(lo[q] <= truth[q] <= hi[q]),
        )
        for q, name in enumerate(names)
    ]


def _result(key, method, interval, pop, **extra):
    truth = fm.exact_population(pop, "budget")
    rows = []
    fm.append_result(rows, key, "budget", method, interval, truth, pop["theta"], **extra)
    r = rows[0]
    r.update(reference_lower=truth[0], reference_upper=truth[1], target=pop["theta"])
    if "lower_mcse" in pop:
        ls, us = pop["lower_mcse"], pop["upper_mcse"]
        outward = (max(0, truth[0] - 2.58 * ls), min(1, truth[1] + 2.58 * us))
        inward = (truth[0] + 2.58 * ls, truth[1] - 2.58 * us)
        r.update(
            reference_lower_mcse=ls,
            reference_upper_mcse=us,
            set_cover_outward=int(r["lower"] <= outward[0] and r["upper"] >= outward[1]),
            set_cover_inward=int(
                inward[0] <= inward[1] and r["lower"] <= inward[0] and r["upper"] >= inward[1]
            ),
            reference_inward_nonempty=int(inward[0] <= inward[1]),
        )
    return r


def _physical_fallback(key, method, pop, exc, **extra):
    """A failed construction supplies the full physical moment and target region."""
    row = _result(
        key,
        method,
        [0, 1],
        pop,
        failure=1,
        primitive_cover=1,
        primitive_region_fallback=1,
        reason=f"{type(exc).__name__}: {exc}",
        **extra,
    )
    M = len(pop["p"])
    D = 2 * M + 3
    unavailable = np.full(D, np.nan)
    moments = _moment_rows(key, method, unavailable, unavailable, np.zeros(D), np.ones(D), pop, M)
    return row, moments


def _unweighted_analysis(cell, y, M, pop, method, key, config, refine):
    F = fm.features(cell, y, M)
    mean, lo, hi, al, au = fm.cell_region(F, method)
    bounds = fm.region(
        mean,
        lo,
        hi,
        M,
        aggregate_lo=al,
        aggregate_hi=au,
        mesh=config.get("mesh", 1 / 128),
        refine=refine,
    )
    pl, pu = fm.signed_budget_exact(mean[0], mean[1: 1 + M], mean[1 + M:])  # fmt: skip
    pl, pu = float(pl), float(pu[0])
    truth = fm.exact_population(pop, "budget")
    fullmean = np.r_[mean, mean[1: 1 + M].sum(), mean[0] - mean[1 + M:].sum()]  # fmt: skip
    full_lo, full_hi = np.r_[lo, al], np.r_[hi, au]
    truthmom = _truth_moments(pop)
    row = _result(
        key,
        method,
        [b.value for b in bounds],
        pop,
        **_bound_diagnostics(bounds),
        primitive_cover=int(np.all(truthmom >= full_lo) and np.all(truthmom <= full_hi)),
        plugin_lower=pl,
        plugin_upper=pu,
        plugin_error=max(abs(pl - truth[0]), abs(pu - truth[1])),
        lower_error=pl - truth[0],
        upper_error=pu - truth[1],
        mu_error=float(mean[0] - pop["mu"]),
        empty_fraction=float(np.mean(np.bincount(cell, minlength=M + 1)[:M] == 0)),
    )
    return row, _moment_rows(key, method, fullmean, fullmean, full_lo, full_hi, pop, M)


def _continuous_job(job):
    config, pop = job["config"], job["pop"]
    ci, K, d, rho, n, rep = (job[k] for k in ("ci", "K", "d", "rho", "n", "rep"))
    master = int(config.get("seed", 42))
    dat = continuous_paths(n, _rng(master, 2, ci, n, rep, 0), K, d, rho, pop["c0"])
    uniform = _rng(master, 2, ci, n, rep, 1).random(n)
    y = (uniform < dat["qR"]).astype(float)
    cell = _continuous_cells(dat, K)
    key = dict(
        design=f"continuous_K{K}_d{d}_rho{rho}",
        ci=ci,
        K=K,
        d=d,
        rho=rho,
        rescue=0.5,
        arm=0,
        n=n,
        rep=rep,
        seed=master,
        rng_stream=json.dumps([2, ci, n, rep]),
    )
    rows, moments = [], []
    for method in ("cp", "kl", "eb", "wald"):
        try:
            row, mr = _unweighted_analysis(cell, y, 4 * K, pop, method, key, config, True)
            moments.extend(mr)
        except Exception as exc:
            row, mr = _physical_fallback(key, method, pop, exc)
            moments.extend(mr)
        rows.append(row)
    return {
        "continuous_replications.csv": rows,
        "continuous_moments.csv": moments,
        "continuous_paired_contrasts.csv": _paired_methods(
            rows, ["ci", "K", "d", "rho", "n", "rep"]
        ),
    }


def _partition_job(job):
    config, reference, n, rep = (job[k] for k in ("config", "pop", "n", "rep"))
    master = int(config.get("seed", 42))
    dat = continuous_paths(n, _rng(master, 3, n, rep, 0), c0=reference["c0"])
    uniform = _rng(master, 3, n, rep, 1).random(n)
    rows, moments = [], []
    fine = {cap: fm.exact_population(reference["pops"][f"{cap}:80"], "budget") for cap in (0, 1)}
    labels = {M: fm.partition_cells(dat, M) for M in PARTITIONS}
    for cap in (0, 1):
        qr, _ = fm.partition_outcomes(dat, bool(cap))
        y = (uniform < qr).astype(float)
        for M in PARTITIONS:
            pop = reference["pops"][f"{cap}:{M}"]
            key = dict(n=n, rep=rep, cap=cap, M=M, seed=master, rng_stream=json.dumps([3, n, rep]))
            try:
                row, mr = _unweighted_analysis(labels[M], y, M, pop, "cp", key, config, False)
                moments.extend(mr)
                row["common_target_error"] = max(
                    abs(row["plugin_lower"] - fine[cap][0]), abs(row["plugin_upper"] - fine[cap][1])
                )
            except Exception as exc:
                row, mr = _physical_fallback(key, "cp", pop, exc)
                moments.extend(mr)
            row["coarsening_gap"] = fm.exact_population(pop, "budget")[1] - fine[cap][1]
            rows.append(row)
    contrasts = []
    for left, right in itertools.combinations(rows, 2):
        if left["cap"] != right["cap"] and left["M"] != right["M"]:
            continue
        contrasts.append(
            dict(
                n=n,
                rep=rep,
                cap_left=left["cap"],
                cap_right=right["cap"],
                M_left=left["M"],
                M_right=right["M"],
                comparison="partition" if left["cap"] == right["cap"] else "outcome_law",
                **{
                    metric + "_difference": left[metric] - right[metric]
                    for metric in ("length", "sharp_width", "enlargement", "set_cover", "failure")
                },
            )
        )
    return {
        "partition_replications.csv": rows,
        "partition_moments.csv": moments,
        "partition_paired_contrasts.csv": contrasts,
    }


def _draw_observational(pop, n, scenario, path_rng, outcome_rng):
    wi = path_rng.integers(0, len(pop["W"]), n)
    W = pop["W"][wi]
    e = fm.obs_propensity(W, scenario)
    A = path_rng.binomial(1, e)
    cell, qr, qnr = np.full(n, 5, dtype=int), np.zeros(n), np.zeros(n)
    for arm in (0, 1):
        ii = np.flatnonzero(A == arm)
        cond = pop["cond"][str(arm)]
        prob = np.column_stack([cond["p"][wi[ii]], cond["p0"][wi[ii]]])
        selected = np.minimum(
            (path_rng.random(len(ii))[:, None] > np.cumsum(prob, axis=1)).sum(1), 5
        )
        cell[ii] = selected
        q = np.column_stack([cond["qR"][wi[ii]], (cond["b0"] / cond["p0"])[wi[ii]]])
        q0 = np.column_stack(
            [cond["qR"][wi[ii]] + cond["tau"][wi[ii]], (cond["b0"] / cond["p0"])[wi[ii]]]
        )
        qr[ii], qnr[ii] = q[np.arange(len(ii)), selected], q0[np.arange(len(ii)), selected]
    uniform = outcome_rng.random(n)
    return dict(
        W=W, A=A, e=e, cell=cell, y=(uniform < qr).astype(float), ynr=(uniform < qnr).astype(float)
    )


def _ipw_analysis(dat, pop, name, propensity, fit, bound, key, config):
    M, n = 5, len(dat["A"])
    Q = fm.add_aggregates(fm.features(dat["cell"], dat["y"], M), M)
    A, e = dat["A"], np.asarray(propensity)
    weights = (1 - A) / (1 - e)
    Z = weights[:, None] * Q
    raw = Z.mean(0)
    influence = Z - raw
    if name.endswith("sandwich"):
        eraw, X, info, _ = fit
        derivative = weights * eraw * ((eraw > 0.02) & (eraw < 0.98))
        B = (Q * derivative[:, None]).T @ X / n
        score = X * (A - eraw)[:, None]
        influence += score @ np.linalg.solve(info, B.T)
    D = Q.shape[1]
    if name.endswith("eb"):
        logterm = np.log(4 * D / 0.05)
        widths = np.sqrt(2 * Z.var(0, ddof=1) * logterm / n) + 7 * bound * logterm / (3 * (n - 1))
    else:
        widths = norm.ppf(1 - 0.05 / (2 * D)) * influence.std(0, ddof=1) / np.sqrt(n)
    plugin = fm.coherent_projection(raw[:-2], M)
    full_plugin = np.r_[
        plugin, plugin[1: 1 + M].sum(), plugin[0] - plugin[1 + M:].sum()
    ]  # fmt: skip
    lo, hi = np.minimum(raw - widths, full_plugin), np.maximum(raw + widths, full_plugin)
    bounds = fm.region(
        plugin,
        lo[:-2],
        hi[:-2],
        M,
        aggregate_lo=lo[-2:],
        aggregate_hi=hi[-2:],
        mesh=config.get("mesh", 1 / 128),
        refine=False,
    )
    pl, pu = fm.signed_budget_exact(plugin[0], plugin[1: 1 + M], plugin[1 + M:])  # fmt: skip
    pl, pu = float(pl), float(pu[0])
    truth = fm.exact_population(pop, "budget")
    popmom = _truth_moments(pop)
    guarantee = (
        "finite_sample_known_score"
        if name == "oracle_eb"
        else (
            "regular_law_asymptotic"
            if name in ("oracle_wald", "logit_sandwich")
            or (name == "main_sandwich" and key["scenario"] == "linear")
            else "empirical_diagnostic"
        )
    )
    row = _result(
        key,
        name,
        [z.value for z in bounds],
        pop,
        **_bound_diagnostics(bounds),
        primitive_cover=int(np.all(popmom >= lo) and np.all(popmom <= hi)),
        primitive_cover_before_enlargement=int(
            np.all(popmom >= raw - widths) and np.all(popmom <= raw + widths)
        ),
        plugin_lower=pl,
        plugin_upper=pu,
        plugin_error=max(abs(pl - truth[0]), abs(pu - truth[1])),
        lower_error=pl - truth[0],
        upper_error=pu - truth[1],
        mu_error=float(plugin[0] - pop["mu"]),
        raw_mu_error=float(raw[0] - pop["mu"]),
        ess=float(weights.sum() ** 2 / np.dot(weights, weights)) if np.dot(weights, weights) else 0,
        clip_fraction=float(np.mean((e <= 0.02) | (e >= 0.98))),
        ps_rmse=float(np.sqrt(np.mean((e - dat["e"]) ** 2))),
        empty_fraction=float(
            np.mean([np.sum((A == 0) & (dat["cell"] == j)) == 0 for j in range(M)])
        ),
        nuisance_failure=0,
        guarantee=guarantee,
        weight_range=bound,
        coherence_adjustment=float(np.linalg.norm(plugin - raw[:-2])),
    )
    return row, _moment_rows(key, name, raw, full_plugin, lo, hi, pop, M)


def _paired_methods(rows, keys):
    contrasts = []
    for left, right in itertools.combinations(rows, 2):
        contrasts.append(
            dict(
                **{k: left[k] for k in keys},
                method_left=left["method"],
                method_right=right["method"],
                **{
                    metric + "_difference": left[metric] - right[metric]
                    for metric in ("length", "enlargement", "set_cover", "failure")
                },
            )
        )
    return contrasts


def _observational_job(job):
    config, population, scenario, n, rep = (
        job[k] for k in ("config", "pop", "scenario", "n", "rep")
    )
    si = int(scenario == "nonlinear")
    master, forest_master = int(config.get("seed", 42)), int(config.get("forest_seed", 420000))
    dat = _draw_observational(
        population, n, scenario, _rng(master, 1, si, n, rep, 0), _rng(master, 1, si, n, rep, 1)
    )
    target = population["pops"]["0"]
    key = dict(scenario=scenario, n=n, rep=rep, seed=master, rng_stream=json.dumps([1, si, n, rep]))
    fits, errors, diagnostics = {}, {}, []
    for name, interactions in (("logit", True), ("main", False)):
        try:
            e, eraw, X, info, beta = fm.fitted_logit(dat["W"], dat["A"], interactions)
            fits[name] = (e, (eraw, X, info, beta))
            diagnostics.append(
                dict(
                    **key,
                    nuisance=name,
                    failure=0,
                    max_coefficient=float(np.max(np.abs(beta))),
                    max_score=float(np.max(np.abs(X.T @ (dat["A"] - eraw) / n))),
                    information_condition=float(np.linalg.cond(info)),
                    clip_fraction=float(np.mean((eraw < 0.02) | (eraw > 0.98))),
                    coefficients=json.dumps(beta.tolist()),
                )
            )
        except Exception as exc:
            errors[name] = f"{type(exc).__name__}: {exc}"
            diagnostics.append(dict(**key, nuisance=name, failure=1, reason=errors[name]))
    split_seed = _int_seed(forest_master, 1, si, n, rep)
    forest_seeds = [_int_seed(forest_master, 2, si, n, rep, fold) for fold in (0, 1)]
    try:
        if np.min(np.bincount(dat["A"], minlength=2)) < 2:
            raise RuntimeError("too_few_treatments_for_cross-fit")
        splitter = StratifiedKFold(n_splits=2, shuffle=True, random_state=split_seed)
        predictions = np.zeros(n)
        for fold, (train, test) in enumerate(splitter.split(dat["W"], dat["A"])):
            forest = RandomForestClassifier(
                n_estimators=int(config.get("forest_trees", 200)),
                min_samples_leaf=max(5, math.ceil(np.sqrt(len(train)) / 2)),
                max_features=None,
                max_depth=None,
                random_state=forest_seeds[fold],
                n_jobs=1,
            )
            forest.fit(dat["W"][train], dat["A"][train])
            predictions[test] = forest.predict_proba(dat["W"][test])[
                :, list(forest.classes_).index(1)
            ]
        fits["rf"] = (np.clip(predictions, 0.02, 0.98), None)
        diagnostics.append(
            dict(
                **key,
                nuisance="rf",
                failure=0,
                forest_seed=forest_master,
                split_seed=split_seed,
                fold0_seed=forest_seeds[0],
                fold1_seed=forest_seeds[1],
                forest_trees=int(config.get("forest_trees", 200)),
                clip_fraction=float(np.mean((predictions < 0.02) | (predictions > 0.98))),
            )
        )
    except Exception as exc:
        errors["rf"] = f"{type(exc).__name__}: {exc}"
        diagnostics.append(
            dict(
                **key,
                nuisance="rf",
                failure=1,
                reason=errors["rf"],
                split_seed=split_seed,
                fold0_seed=forest_seeds[0],
                fold1_seed=forest_seeds[1],
            )
        )
    bound = float(1 / np.min(1 - fm.obs_propensity(population["W"], scenario)))
    rows, moments = [], []
    for name in OBS_METHODS:
        root = name.split("_")[0]
        try:
            if root in errors:
                raise RuntimeError(errors[root])
            e, fit = (dat["e"], None) if root == "oracle" else fits[root]
            row, mr = _ipw_analysis(
                dat, target, name, e, fit, bound if root == "oracle" else 50.0, key, config
            )
            moments.extend(mr)
        except Exception as exc:
            row, mr = _physical_fallback(
                key, name, target, exc, nuisance_failure=int(root in errors)
            )
            moments.extend(mr)
        rows.append(row)
    return {
        "observational_replications.csv": rows,
        "observational_moments.csv": moments,
        "observational_nuisance_fits.csv": diagnostics,
        "observational_paired_contrasts.csv": _paired_methods(rows, ["scenario", "n", "rep"]),
    }


def run_extension_job(job):
    files = {
        "observational": _observational_job,
        "continuous": _continuous_job,
        "partitions": _partition_job,
    }[job["suite"]](job)
    return _suite_files(files)
