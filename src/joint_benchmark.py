from __future__ import annotations

import time
from collections.abc import Generator
from dataclasses import fields

import numpy as np

import methods as m
from joint_regions import joint_region
from simulations import population, record, rng

JOINT_SETTINGS = (
    ("reference", 1000, False),
    ("active_caps", 1000, True),
    ("active_caps", 10000, True),
)
JOINT_METHODS = ("cp", "joint_split", "joint_lr", "aggregate_cp")
SEED_FAMILY = 60
ALPHA = 0.05
GAMMA = 0.2
MESH = 1 / 128
MAX_ITERATIONS = 120
GAP_TOLERANCE = 1e-4


def population_atoms(pop: dict) -> np.ndarray:
    """Return the ordered (crossing cell, outcome) law, noncrossing last."""
    atoms = np.column_stack((pop["p"] - pop["b"], pop["b"]))
    atoms = np.vstack((atoms, (1 - pop["p"].sum() - pop["b0"], pop["b0"])))
    if atoms.shape != (6, 2) or np.any(atoms < 0) or not np.isclose(atoms.sum(), 1):
        raise ValueError("The joint benchmark requires a probability law on 12 fixed atoms.")
    return atoms


def draw_halves(pop: dict, n: int, seed: int, setting_index: int, rep: int) -> np.ndarray:
    """Independent half streams are invariant to worker count and job order."""
    probabilities = population_atoms(pop).ravel()
    return np.stack(
        [
            rng(seed, SEED_FAMILY, setting_index, n, rep, half)
            .multinomial(size, probabilities)
            .reshape(6, 2)
            for half, size in enumerate((n // 2, n - n // 2))
        ]
    )


def joint_jobs(cfg: dict) -> Generator[dict, None, None]:
    if "joint" not in cfg["suites"]:
        return
    for setting_index, (setting, n, caps) in enumerate(JOINT_SETTINGS):
        for rep in range(cfg["joint_reps"]):
            yield {
                "id": f"joint_{setting_index}_n{n}_rep{rep:06d}",
                "suite": "joint",
                "task": "joint",
                "weight": 1,
                "config": cfg,
                "setting": setting,
                "setting_index": setting_index,
                "n": n,
                "caps": caps,
                "rep": rep,
            }


def joint_protocol(cfg: dict) -> dict:
    """Permanent protocol and population metadata, independent of checkpoints."""
    settings = []
    for index, (setting, n, caps) in enumerate(JOINT_SETTINGS):
        pop = population(rescue=0.5, arm=0, states=4, caps=caps)
        truth = m.exact_population(pop, "budget")
        settings.append(
            {
                "setting_index": index,
                "setting": setting,
                "n": n,
                "halves": [n // 2, n - n // 2],
                "replications": cfg["joint_reps"],
                "arm": 0,
                "states": 4,
                "visits": 5,
                "active_caps": caps,
                "rescue_target": 0.5,
                "rescue_probability": float(pop["p"].sum()),
                "hazard_intercept": float(pop["alpha"]),
                "p": pop["p"].tolist(),
                "b": pop["b"].tolist(),
                "qR": pop["qR"].tolist(),
                "tau": pop["tau"].tolist(),
                "mu": float(pop["mu"]),
                "b0": float(pop["b0"]),
                "theta": float(pop["theta"]),
                "atom_probabilities": population_atoms(pop).tolist(),
                "population_lower": truth[0],
                "population_upper": truth[1],
                "sharp_width": truth[1] - truth[0],
            }
        )
    return {
        "benchmark": "joint-region",
        "master_seed": cfg["seed"],
        "seed_family": SEED_FAMILY,
        "seed_entropy": ["master_seed", SEED_FAMILY, "setting_index", "n", "rep", "half"],
        "rep_indexing": "zero-based",
        "half_indexing": "zero-based",
        "generator": "numpy.random.default_rng (PCG64), SeedSequence entropy as above",
        "count_shape": [2, 6, 2],
        "cell_indexing": "0..4 are crossing visits 1..5; 5 is noncrossing",
        "outcome_indexing": [0, 1],
        "atoms_deleted": False,
        "methods": list(JOINT_METHODS),
        "alpha": ALPHA,
        "gamma": GAMMA,
        "scientific_budget": GAMMA**2,
        "mesh": MESH,
        "joint_max_cut_iterations_per_endpoint": MAX_ITERATIONS,
        "joint_gap_tolerance": GAP_TOLERANCE,
        "cell_cp_max_dyadic_refinements": 2,
        "cell_cp_feasibility_tolerance": 1e-9,
        "joint_feasibility_tolerance": 1e-8,
        "likelihood_interval_decimal_digits": 45,
        "probability_inversion_outward_guard": 1e-12,
        "settings": settings,
    }


def endpoint_diagnostics(bounds: tuple[m.Bound, m.Bound]) -> dict:
    """Retain every scalar endpoint diagnostic, including exhaustion flags."""
    extra = m.bound_diagnostics(bounds)
    for prefix, bound in zip(("lower", "upper"), bounds):
        for field in fields(bound):
            if field.name in {"value", "witness"}:
                continue
            extra[f"{prefix}_{field.name}"] = getattr(bound, field.name)
        extra[f"{prefix}_iterations"] = int(getattr(bound, "iterations", 0))
        extra[f"{prefix}_exhausted"] = int(getattr(bound, "exhausted", False))
        extra[f"{prefix}_fallback"] = int(getattr(bound, "fallback", bound.status != "ok"))
        extra[f"{prefix}_likelihood_violation"] = float(
            getattr(bound, "likelihood_violation", float("nan"))
        )
    extra["fallback_count"] = sum(extra[f"{p}_fallback"] for p in ("lower", "upper"))
    extra["fallback"] = int(extra["fallback_count"] > 0 or extra["failure"])
    extra["exhausted"] = max(extra[f"{p}_exhausted"] for p in ("lower", "upper"))
    extra["iterations"] = max(extra[f"{p}_iterations"] for p in ("lower", "upper"))
    extra["cut_iterations"] = sum(extra[f"{p}_iterations"] for p in ("lower", "upper"))
    return extra


def run_joint_job(job: dict) -> dict[str, list[dict]]:
    cfg, n, rep = job["config"], job["n"], job["rep"]
    pop = population(rescue=0.5, arm=0, states=4, caps=job["caps"])
    truth = m.exact_population(pop, "budget")
    halves = draw_halves(pop, n, cfg["seed"], job["setting_index"], rep)
    counts = halves.sum(axis=0)
    # Deterministic expansion permits reusing the existing CP implementations.
    codes = np.repeat(np.arange(12), counts.ravel())
    cell, y = codes // 2, (codes % 2).astype(float)
    F = m.features(cell, y, 5)
    mean, lo, hi, al, au = m.cell_region(F, "cp", alpha=ALPHA)
    cp_bounds = m.region(mean, lo, hi, 5, gamma=GAMMA, mesh=MESH, aggregate_lo=al, aggregate_hi=au)
    fitted = {"cp": cp_bounds}
    for method in ("split", "lr"):
        fitted[f"joint_{method}"] = joint_region(
            halves,
            method=method,
            alpha=ALPHA,
            gamma=GAMMA,
            mesh=MESH,
            max_iterations=MAX_ITERATIONS,
            gap_tolerance=GAP_TOLERANCE,
        )
    start = time.perf_counter()
    aggregate = m.endpoint_cp(cell, y, 5, "budget", alpha=ALPHA)
    runtime = time.perf_counter() - start
    fitted["aggregate_cp"] = tuple(
        m.Bound(
            value=float(value),
            primal_value=float(value),
            status="ok",
            runtime=runtime / 2,
            violation=0.0,
            dual_gap=0.0,
            mesh=0.0,
            inner_gap=0.0,
            inner_violation=0.0,
            inner_status="analytic_endpoint",
        )
        for value in aggregate
    )
    key = {
        "design": "joint",
        "setting": job["setting"],
        "setting_index": job["setting_index"],
        "rescue": 0.5,
        "arm": 0,
        "n": n,
        "rep": rep,
        "seed": cfg["seed"],
    }
    rows = []
    for method in JOINT_METHODS:
        bounds = fitted[method]
        interval = [z.value for z in bounds]
        diagnostics = endpoint_diagnostics(bounds)
        invalid_interval = not np.isfinite(sum(interval)) or interval[0] > interval[1]
        diagnostics["invalid_interval"] = int(invalid_interval)
        if invalid_interval:
            # record() replaces the whole interval with [0,1]. Keep the
            # endpoint fallback flags consistent with that saved interval.
            diagnostics.update(
                failure=1, fallback=1, fallback_count=2, lower_fallback=1, upper_fallback=1
            )
        record(
            rows,
            key,
            "budget",
            method,
            interval,
            truth,
            pop,
            alpha=ALPHA,
            gamma=GAMMA,
            **diagnostics,
        )
    cp_width = rows[0]["length"]
    for row in rows:
        row["width_difference_cp"] = row["length"] - cp_width
    return {
        "joint/replications.csv": rows,
        "joint/counts.csv": [
            dict(
                **key,
                half=half,
                cell=cell_index,
                y=outcome,
                count=int(halves[half, cell_index, outcome]),
            )
            for half in (0, 1)
            for cell_index in range(6)
            for outcome in (0, 1)
        ],
    }
