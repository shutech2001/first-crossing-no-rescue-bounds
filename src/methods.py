from __future__ import annotations

import csv
import itertools
import math
import time
from dataclasses import dataclass, replace
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq, linprog, minimize
from scipy.special import expit, ndtr, ndtri, xlogy
from scipy.stats import beta, norm

MASTER = 42
DEFAULT_MESH = 1 / 128
SAMPLE_SIZES = [50, 100, 250, 500, 1000, 2500, 10000]
OBS_SIZES = [50, 250, 1000, 10000]
PART_SIZES = [250, 1000, 10000]
PARTITIONS = [1, 5, 10, 20, 40, 80]


def rng_for(*keys: int) -> np.random.Generator:
    """Return a random number generator for the given keys.

    Args:
        *keys (int): The keys.

    Returns:
        np.random.Generator: The random number generator.
    """
    return np.random.default_rng(np.random.SeedSequence([MASTER, *map(int, keys)]))


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write the rows to a CSV file.

    Args:
        path (Path): The path to the CSV file.
        rows (list[dict]): The rows.
    """
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def finite_population(
    K: int = 5, J: int = 4, rescue: float = 0.5, arm: int = 0, alpha: float | None = None
) -> dict:
    """Exact dynamic enumeration of the Markov DGP in the manuscript.

    Args:
        K (int): The number of states.
        J (int): The number of outcomes.
        rescue (float): The rescue parameter.
        arm (int): The arm.
        alpha (float | None): The alpha.

    Returns:
        dict: The finite population.
    """
    states = np.arange(J)

    def one(al: float, a: int, arrays: bool = False):
        """Return the one population.

        Args:
            al (float): The alpha.
            a (int): The arm.
            arrays (bool): Whether to return arrays.

        Returns:
            dict: The one population.
        """
        mass = np.array([1.0])
        prev = np.array([(J - 1) / 2])
        crossing = []
        masses = []
        trans = []
        compliant = []
        for k in range(K):
            hit = expit(al + 0.6 * (prev / (J - 1) - 0.5) + 0.15 * k - 0.10 * a)
            ker = np.exp(
                -((states[None, :] - 0.6 * prev[:, None] - 0.25 * (J - 1)) ** 2)
                / (2 * (0.35 * (J - 1) + 0.3) ** 2)
            )
            ker /= ker.sum(1, keepdims=True)
            surv = (mass * (1 - hit)) @ ker
            cp = float(mass @ hit)
            trans.append(np.column_stack([(1 - hit)[:, None] * ker, hit]))
            masses.append(np.r_[surv, cp])
            crossing.append(cp)
            compliant.append(surv)
            mass = surv
            prev = states
        if not arrays:
            return sum(crossing)
        r = 0.25 + 0.35 * states / (J - 1) - 0.05 * a
        p = np.asarray(crossing)
        qr = 0.20 + 0.08 * np.arange(1, K + 1) / K - 0.04 * a
        tau = 0.08 + 0.12 * (K - np.arange(1, K + 1)) / K
        b = p * qr
        b0 = float(mass @ r)
        N = K * (J + 1)
        A = np.zeros((K * J, N))
        E = np.zeros((N, K))
        for k in range(K - 1, -1, -1):
            E[k * (J + 1) + J, k] = 1
            A[k * J: (k + 1) * J, k * (J + 1): k * (J + 1) + J] = np.eye(J)  # fmt: skip
            if k < K - 1:
                tr = trans[k + 1]
                A[k * J: (k + 1) * J, (k + 1) * (J + 1): (k + 2) * (J + 1)] = -tr  # fmt: skip
                current_rows = slice(k * (J + 1), k * (J + 1) + J)
                next_rows = slice((k + 1) * (J + 1), (k + 2) * (J + 1))
                E[current_rows] = tr @ E[next_rows]
        load = np.zeros(N)
        load[: J + 1] = masses[0]
        metric = np.concatenate(masses)
        Omega = E.T @ (metric[:, None] * E)
        return {
            "p": p,
            "b": b,
            "mu": float(b.sum() + b0),
            "theta": float(b.sum() + b0 + p @ tau),
            "tau": tau,
            "qR": qr,
            "b0": b0,
            "alpha": al,
            "A": A,
            "E": E,
            "r": load,
            "G": metric,
            "Omega": Omega,
            "K": K,
            "J": J,
            "arm": a,
            "rescue": float(p.sum()),
            "p0": float(mass.sum()),
            "noncrossing_state_mass": mass.copy(),
            "noncrossing_outcome_mean": r.copy(),
        }

    if alpha is None:
        alpha = brentq(lambda x: (one(x, 0) + one(x, 1)) / 2 - rescue, -15, 15, xtol=1e-12)
    return one(alpha, arm, True)


def draw_finite(
    pop: dict, n: int, rng: np.random.Generator, outcome_rng: np.random.Generator | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw the finite population.

    Args:
        pop (dict): The population.
        n (int): The number of draws.
        rng (np.random.Generator): The random number generator.
        outcome_rng (np.random.Generator | None): The outcome random number generator.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: The draws.
    """
    p = pop["p"]
    M = len(p)
    cell = rng.choice(M + 1, size=n, p=np.r_[p, 1 - p.sum()])
    q = np.r_[pop["qR"], pop["b0"] / (1 - p.sum())]
    q0 = np.r_[pop["qR"] + pop["tau"], q[-1]]
    u = (rng if outcome_rng is None else outcome_rng).random(n)
    return cell, (u < q[cell]).astype(float), (u < q0[cell]).astype(float)


def features(cell: NDArray[np.int32], y: NDArray[np.float64], M: int) -> NDArray[np.float64]:
    """Return the features.

    Args:
        cell (NDArray[np.int32]): The cell.
        y (NDArray[np.float64]): The y.
        M (int): The number of features.

    Returns:
        NDArray[np.float64]: The features.
    """
    z = (cell[:, None] == np.arange(M)[None, :]).astype(float)
    return np.column_stack([y, z, z * y[:, None]])


def moment_box(
    F: NDArray[np.float64],
    method: str,
    alpha: float = 0.05,
    total_D: int | None = None,
    lengths: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return the moment box.

    Args:
        F (NDArray[np.float64]): The F.
        method (str): The method.
        alpha (float): The alpha.
        total_D (int | None): The total D.
        lengths (NDArray[np.float64] | None): The lengths.

    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]: The moment box.
    """
    n, D = F.shape
    DD = total_D or D
    if n < 2:
        raise ValueError("At least two observations are required.")
    mean = F.mean(0)
    var = F.var(0, ddof=1)
    lengths = np.ones(D) if lengths is None else np.asarray(lengths)
    if method == "hoeffding":
        w = lengths * np.sqrt(np.log(2 * DD / alpha) / (2 * n))
        lo = mean - w
        hi = mean + w
    elif method == "eb":
        z = np.log(4 * DD / alpha)
        w = np.sqrt(2 * var * z / n) + 7 * lengths * z / (3 * (n - 1))
        lo = mean - w
        hi = mean + w
    elif method == "wald":
        w = norm.ppf(1 - alpha / (2 * DD)) * np.sqrt(var / n)
        lo = mean - w
        hi = mean + w
    elif method == "cp":
        if not np.all((F == 0) | (F == 1)):
            raise ValueError("CP requires Bernoulli coordinates.")
        lo, hi = cp_counts(F.sum(0), n, alpha, DD)
    else:
        raise ValueError(method)
    return mean, lo, hi


def analysis_box(
    F: NDArray[np.float64], method: str, extra_D: int = 0, range_only: bool = False
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Return the analysis box.

    Args:
        F (NDArray[np.float64]): The F.
        method (str): The method.
        extra_D (int): The extra D.
        range_only (bool): Whether to return only the range.

    Returns:
        tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64]
        ]: The analysis box.
    """
    M = (F.shape[1] - 1) // 2
    agg = np.column_stack([F[:, 1: 1 + M].sum(1), F[:, 0] - F[:, 1 + M:].sum(1)])  # fmt: skip
    if range_only:
        _, al, au = moment_box(agg, method, total_D=2 + extra_D)
        return F.mean(0), np.zeros(F.shape[1]), np.ones(F.shape[1]), al, au
    FF = np.column_stack([F, agg])
    mean, lo, hi = moment_box(FF, method, total_D=FF.shape[1] + extra_D)
    return mean[:-2], lo[:-2], hi[:-2], lo[-2:], hi[-2:]


@dataclass
class Bound:
    value: float
    primal_value: float
    status: str
    runtime: float
    violation: float
    dual_gap: float
    witness: np.ndarray | None = None
    mesh: float = DEFAULT_MESH
    refinement_count: int = 0
    inner_gap: float = float("nan")
    inner_violation: float = float("nan")
    inner_status: str = "not_requested"
    budget_allowance: float = 0.0


def _rational_lower(
    c: NDArray[np.float64],
    A: NDArray[np.float64],
    b: NDArray[np.float64],
    bounds: list[tuple[float, float]],
    marg: NDArray[np.float64],
) -> float:
    """Return a valid lower bound for min c'x on a bounded LP, independent of dual feasibility.

    lambda >= 0; L = -lambda'b + min_box (c + A'lambda)'x.
    Only nonzero multipliers are accumulated; exact binary-float rationals are used.

    Args:
        c (NDArray[np.float64]): The c.
        A (NDArray[np.float64]): The A.
        b (NDArray[np.float64]): The b.
        bounds (list[tuple[float, float]]): The bounds.
        marg (NDArray[np.float64]): The marg.

    Returns:
        float: The valid lower bound.
    """
    if not np.all(np.isfinite(marg)):
        raise ValueError("non-finite dual multipliers")
    f = Fraction.from_float
    cc = [f(float(x)) for x in c]
    constant = Fraction(0)
    for i in np.flatnonzero(marg < 0):
        lam = f(float(-marg[i]))
        constant -= lam * f(float(b[i]))
        for j in np.flatnonzero(A[i]):
            cc[j] += lam * f(float(A[i, j]))
    for x, (lo, hi) in zip(cc, bounds):
        constant += x * f(float(lo if x >= 0 else hi))
    val = float(constant)
    if Fraction.from_float(val) > constant:
        val = np.nextafter(val, -np.inf)
    return float(val)


def _outward_add(a: float, b: float, direction: int) -> float:
    """Return an objective offset with one final exact-rational outward rounding.

    Args:
        a (float): The a.
        b (float): The b.
        direction (int): The direction.

    Returns:
        float: The objective offset.
    """
    total = Fraction.from_float(float(a)) + Fraction.from_float(float(b))
    result = float(total)
    if direction * (Fraction.from_float(result) - total) > 0:
        result = np.nextafter(result, -np.inf if direction == 1 else np.inf)
    return float(result)


def solve_mass(
    mean: NDArray[np.float64],
    lo: NDArray[np.float64],
    hi: NDArray[np.float64],
    M: int,
    gamma: float | None = 0.2,
    signed: bool = True,
    H: NDArray[np.float64] | None = None,
    dlo: NDArray[np.float64] | None = None,
    dhi: NDArray[np.float64] | None = None,
    direction: int = 1,
    mesh: float | None = None,
    inner: bool = False,
    center: NDArray[np.float64] | None = None,
    certificate: bool = True,
    aggregate_lo: NDArray[np.float64] | None = None,
    aggregate_hi: NDArray[np.float64] | None = None,
    effect_lower: NDArray[np.float64] | None = None,
    effect_upper: NDArray[np.float64] | None = None,
    extra_linear: tuple | None = None,
) -> Bound:
    """LP bracket of the mass-coordinate program; direction=+1 lower, -1 upper.

    Dyadic mesh and zero center make every tangent coefficient exactly representable.
    Nonzero-center tangent rows are relaxed by an explicitly bounded rounding guard.
    The main reported results use center zero. Variables are (mu,p,b,t,s).

    Args:
        mean (NDArray[np.float64]): The mean.
        lo (NDArray[np.float64]): The lo.
        hi (NDArray[np.float64]): The hi.
        M (int): The number of features.
        gamma (float | None): The gamma.
        signed (bool): Whether to sign the budget.
        H (NDArray[np.float64] | None): The H.
        dlo (NDArray[np.float64] | None): The dlo.
        dhi (NDArray[np.float64] | None): The dhi.
        direction (int): The direction.
        mesh (float | None): The mesh.
        inner (bool): Whether to use the inner budget.
        center (NDArray[np.float64] | None): The center.
        certificate (bool): Whether to use the certificate.
        aggregate_lo (NDArray[np.float64] | None): The aggregate lo.
        aggregate_hi (NDArray[np.float64] | None): The aggregate hi.
        effect_lower (NDArray[np.float64] | None): The effect lower.
        effect_upper (NDArray[np.float64] | None): The effect upper.
        extra_linear (tuple | None): The extra linear.

    Returns:
        Bound: The bound.
    """
    mesh = DEFAULT_MESH if mesh is None else mesh
    if direction not in [1, -1]:
        raise ValueError("direction must be +1 (lower) or -1 (upper).")
    if mesh <= 0 or mesh > 2:
        raise ValueError("mesh must be in (0,2].")
    if gamma is not None and (not np.isfinite(gamma) or gamma < 0):
        raise ValueError("gamma must be nonnegative.")
    start = time.perf_counter()
    center = np.zeros(M) if center is None else np.asarray(center, float)
    if len(mean) != 1 + 2 * M or center.shape != (M,) or not np.all(np.abs(center) <= 1):
        raise ValueError("Dimensions/center.")
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    if lo.shape != (1 + 2 * M,) or hi.shape != lo.shape:
        raise ValueError("Primitive interval dimensions do not match M.")
    if np.any(np.isnan(lo)) or np.any(np.isnan(hi)):
        return Bound(
            0 if direction == 1 else 1,
            np.nan,
            "non-finite_box",
            time.perf_counter() - start,
            np.nan,
            np.nan,
            mesh=mesh,
        )
    p0 = 1
    b0 = 1 + M
    t0 = 1 + 2 * M
    s0 = 1 + 3 * M
    nv = 1 + 4 * M
    G = None if gamma is None else float(gamma**2)
    steps = round(2 / mesh)
    if steps < 1 or (steps & (steps - 1)) or steps > 2**20:
        raise ValueError(
            "Certified tangent implementation requires a dyadic grid with at most 2**20 intervals."
        )
    actual_h = 2 / steps
    if abs(actual_h - mesh) > 1e-12:
        raise ValueError("mesh must divide [-1,1] exactly.")
    err = actual_h**2 / 4
    budget = None if G is None else (0.0 if G == 0 else G - (err if inner else 0))
    if budget is not None and budget < 0:
        return Bound(0 if direction == 1 else 1, np.nan, "empty_inner_budget", 0, 0, np.nan)
    box = [(max(0, float(lo[0])), min(1, float(hi[0])))]
    for j in range(2 * M):
        box.append((max(0, float(lo[1 + j])), min(1, float(hi[1 + j]))))
    box += [(-1.0, 1.0)] * M + [(0.0, max(4.0, G or 0))] * M
    if any(x > y for x, y in box):
        return Bound(0 if direction == 1 else 1, np.nan, "empty_box", 0, 0, np.nan)
    rows = []
    rhs = []

    def row(vals: dict[int, float], rhsval: float):
        """Append a linear inequality to the current LP constraint lists.

        Args:
            vals (dict[int, float]): Nonzero coefficients indexed by variable.
            rhsval (float): Upper bound on the linear expression.
        """
        a = np.zeros(nv)
        for j, v in vals.items():
            a[j] = v
        rows.append(a)
        rhs.append(float(rhsval))

    row({p0 + j: 1.0 for j in range(M)}, 1.0)
    row({0: -1.0, **{b0 + j: 1.0 for j in range(M)}}, 0.0)
    row({0: 1.0, **{b0 + j: -1.0 for j in range(M)}, **{p0 + j: 1.0 for j in range(M)}}, 1.0)
    if aggregate_lo is not None:
        row({p0 + j: 1.0 for j in range(M)}, aggregate_hi[0])
        row({p0 + j: -1.0 for j in range(M)}, -aggregate_lo[0])
        row({0: 1.0, **{b0 + j: -1.0 for j in range(M)}}, aggregate_hi[1])
        row({0: -1.0, **{b0 + j: 1.0 for j in range(M)}}, -aggregate_lo[1])
    if extra_linear is not None:
        Q, offset, qlo, qhi = extra_linear
        for qq, off, aa, zz in zip(Q, offset, qlo, qhi):
            row({i: float(v) for i, v in enumerate(qq) if v}, float(zz - off))
            row({i: float(-v) for i, v in enumerate(qq) if v}, float(off - aa))
    ell = np.full(M, 0.0 if signed else -1.0) if effect_lower is None else np.asarray(effect_lower)
    upp = np.ones(M) if effect_upper is None else np.asarray(effect_upper)
    if (
        ell.shape != (M,)
        or upp.shape != (M,)
        or np.any(ell > upp)
        or np.any(ell < -1)
        or np.any(upp > 1)
    ):
        raise ValueError("Invalid average-effect limits.")
    for j in range(M):
        row({t0 + j: 1, p0 + j: -float(upp[j])}, 0)
        row({t0 + j: -1, p0 + j: float(ell[j])}, 0)
        row({b0 + j: 1, p0 + j: -1}, 0)
        row({b0 + j: -1, t0 + j: -1}, 0)
        row({b0 + j: 1, t0 + j: 1, p0 + j: -1}, 0)
        row({t0 + j: 1, p0 + j: -1}, 0)
        row({t0 + j: -1, p0 + j: -1}, 0)
    if H is not None:
        H = np.asarray(H, float)
        dlo = np.asarray(dlo, float)
        dhi = np.asarray(dhi, float)
        for k, h in enumerate(H):
            row({t0 + j: float(h[j]) for j in range(M)}, dhi[k])
            row({t0 + j: -float(h[j]) for j in range(M)}, -dlo[k])
    if G is not None:
        if G == 0:
            for j in range(M):
                row({t0 + j: 1, p0 + j: -center[j]}, 0)
                row({t0 + j: -1, p0 + j: center[j]}, 0)
        else:
            grid = np.linspace(-1, 1, steps + 1)
            # Dyadic r and c=0: these coefficients have no rounding error.
            for j in range(M):
                cc = float(center[j])
                for r in grid:
                    at = float(2 * (r - cc))
                    ap = float(cc * cc - r * r)
                    guard = 0.0
                    if cc != 0:
                        fr = Fraction.from_float(float(r))
                        fc = Fraction.from_float(cc)
                        da = abs(Fraction.from_float(at) - 2 * (fr - fc))
                        dp = abs(Fraction.from_float(ap) - (fc * fc - fr * fr))
                        guard = float(da + dp)
                        guard = float(np.nextafter(guard, np.inf))
                    row({t0 + j: at, p0 + j: ap, s0 + j: -1}, -guard if inner else guard)
            row({s0 + j: 1 for j in range(M)}, budget)
    obj = np.zeros(nv)
    obj[0] = direction
    obj[t0: t0 + M] = direction  # fmt: skip
    A = np.asarray(rows)
    bb = np.asarray(rhs)
    res = linprog(
        obj,
        A_ub=A,
        b_ub=bb,
        bounds=box,
        method="highs",
        options={"primal_feasibility_tolerance": 1e-9, "dual_feasibility_tolerance": 1e-9},
    )
    sec = time.perf_counter() - start
    if not res.success:
        return Bound(
            0 if direction == 1 else 1, np.nan, "lp_failure_" + str(res.status), sec, np.nan, np.nan
        )
    # Disabling certification is allowed only for a diagnostic primal solve.
    # Such a solve must never masquerade as an outward confidence endpoint.
    if not certificate:
        return Bound(
            0 if direction == 1 else 1,
            float(res.fun * direction),
            "certificate_disabled",
            time.perf_counter() - start,
            np.nan,
            np.nan,
            res.x,
            mesh=mesh,
            budget_allowance=err if G not in [None, 0.0] else 0.0,
        )
    try:
        lower = _rational_lower(obj, A, bb, box, res.ineqlin.marginals)
        if not np.isfinite(lower):
            raise ValueError("non-finite certificate")
    except (ArithmeticError, ValueError, TypeError, AttributeError):
        return Bound(
            0 if direction == 1 else 1,
            np.nan,
            "certificate_failure",
            time.perf_counter() - start,
            np.nan,
            np.nan,
            mesh=mesh,
        )
    outer = float(lower if direction == 1 else -lower)
    x = res.x
    pr = float(x[0] + x[t0: t0 + M].sum())  # fmt: skip
    p = x[p0: p0 + M]  # fmt: skip
    t = x[t0: t0 + M]  # fmt: skip
    pos = p > 0
    F = float(np.sum((t[pos] - center[pos] * p[pos]) ** 2 / p[pos]))
    vio = max(0.0, F - G) if G is not None else 0.0
    linear_vio = max(
        0.0,
        float(np.max(A @ x - bb)),
        max(float(a - z) for a, z in zip(np.array(box)[:, 0], x)),
        max(float(z - a) for a, z in zip(np.array(box)[:, 1], x)),
    )
    vio = max(vio, linear_vio)
    gap = float(res.fun - lower)
    return Bound(
        max(0.0, min(1.0, outer)),
        pr,
        "ok",
        time.perf_counter() - start,
        vio,
        gap,
        x,
        mesh=mesh,
        budget_allowance=err if G not in [None, 0.0] else 0.0,
    )


def region(
    mean,
    lo,
    hi,
    M,
    *,
    refine=True,
    max_refinements=2,
    endpoint_tolerance=1e-4,
    residual_tolerance=1e-8,
    **kwargs,
) -> tuple[Bound, Bound]:
    """Outward LP interval with at most two additional dyadic refinements.

    Args:
        mean (NDArray[np.float64]): The mean.
        lo (NDArray[np.float64]): The lo.
        hi (NDArray[np.float64]): The hi.
        M (int): The number of features.
        refine (bool): Whether to refine.
        max_refinements (int): The maximum refinements.
        endpoint_tolerance (float): The endpoint tolerance.
        residual_tolerance (float): The residual tolerance.
        **kwargs: The keyword arguments.

    Returns:
        tuple[Bound, Bound]: The bounds.
    """
    if not 0 <= max_refinements <= 2:
        raise ValueError("The numerical protocol permits at most two refinements.")
    hh = kwargs.pop("mesh", None)
    hh = DEFAULT_MESH if hh is None else hh
    elapsed = [0.0, 0.0]
    can_refine = (
        refine and not kwargs.get("inner", False) and kwargs.get("gamma", 0.2) not in [None, 0.0]
    )
    inn = None
    gap = float("nan")
    inner_vio = float("nan")
    inner_status = "not_requested"
    for refinement in range(max_refinements + 1):
        result = tuple(solve_mass(mean, lo, hi, M, direction=d, mesh=hh, **kwargs) for d in [1, -1])
        for i, z in enumerate(result):
            elapsed[i] += z.runtime
        if any(z.status != "ok" for z in result) or not can_refine:
            break
        inner_args = dict(kwargs, inner=True)
        inn = tuple(
            solve_mass(mean, lo, hi, M, direction=d, mesh=hh, **inner_args) for d in [1, -1]
        )
        for i, z in enumerate(inn):
            elapsed[i] += z.runtime
        inner_status = ";".join(z.status for z in inn)
        if any(z.status != "ok" for z in inn):
            break
        gap = max(0.0, inn[0].primal_value - result[0].value, result[1].value - inn[1].primal_value)
        inner_vio = max(z.violation for z in inn)
        if gap <= endpoint_tolerance and inner_vio <= residual_tolerance:
            inner_status = "residual_checked"
            break
        inner_status = (
            "residual_failed" if inner_vio > residual_tolerance else "gap_above_tolerance"
        )
        if refinement == max_refinements:
            break
        hh /= 2
    result = tuple(
        replace(
            z,
            runtime=elapsed[i],
            mesh=hh,
            refinement_count=refinement,
            inner_gap=gap,
            inner_violation=inner_vio,
            inner_status=inner_status,
        )
        for i, z in enumerate(result)
    )
    # Fail the whole confidence interval when either endpoint lacks a certificate.
    if any(z.status != "ok" for z in result):
        result = tuple(replace(z, value=float(i)) for i, z in enumerate(result))
    return result


def bound_diagnostics(rr):
    """Serializable endpoint and numerical-refinement diagnostics."""

    def finite_max(values):
        """Return the largest finite diagnostic value.

        Args:
            values (Iterable[float]): Diagnostic values to compare.

        Returns:
            float: The maximum, or NaN if no finite values are available.
        """
        values = [float(v) for v in values if np.isfinite(v)]
        return max(values) if values else float("nan")

    return {
        "failure": int(any(z.status != "ok" for z in rr)),
        "empty_program": int(
            any(z.status in {"lp_failure_2", "empty_box", "empty_inner_budget"} for z in rr)
        ),
        "unavailable_certificate": int(any(z.status.startswith("certificate_") for z in rr)),
        "stopping_limit": int(
            any(
                z.refinement_count == 2
                and z.inner_status in {"gap_above_tolerance", "residual_failed"}
                for z in rr
            )
        ),
        "runtime": sum(z.runtime for z in rr),
        "dual_gap": finite_max(z.dual_gap for z in rr),
        "lower_status": rr[0].status,
        "upper_status": rr[1].status,
        "mesh": max(z.mesh for z in rr),
        "refinement_count": max(z.refinement_count for z in rr),
        "inner_gap": finite_max(z.inner_gap for z in rr),
        "inner_violation": finite_max(z.inner_violation for z in rr),
        "inner_status": ";".join(dict.fromkeys(z.inner_status for z in rr)),
        "budget_allowance": max(z.budget_allowance for z in rr),
    }


def population_region(pop: dict, kind: str, mesh=1 / 256):
    """Compute certified bounds with the population moments held fixed.

    Args:
        pop (dict): Population moments, including cell effects for calibration.
        kind (str): Restriction class: range, signed, budget, or calibrated.
        mesh (float): Tangent-grid spacing for the initial LP approximation.

    Returns:
        tuple[Bound, Bound]: Lower and upper outward bounds with diagnostics.
    """
    p = pop["p"]
    b = pop["b"]
    M = len(p)
    v = np.r_[pop["mu"], p, b]
    H = None
    d = None
    if kind == "calibrated":
        H = calibration_matrix(M)
        d = H @ (p * pop["tau"])
    g = None if kind == "range" or kind == "signed" else 0.2
    return region(v, v, v, M, gamma=g, signed=(kind != "range"), H=H, dlo=d, dhi=d, mesh=mesh)


def calibration_matrix(M: int) -> np.ndarray:
    """Group odd-numbered crossing visits into two calibration totals.

    Args:
        M (int): Number of crossing cells, ordered by visit.

    Returns:
        np.ndarray: A (2, M) indicator matrix splitting odd visits between
            the first and second halves of the visit sequence.
    """
    h = np.zeros((2, M))
    select = np.arange(M) % 2 == 0
    h[0, select & (np.arange(M) < math.ceil(M / 2))] = 1
    h[1, select & (np.arange(M) >= math.ceil(M / 2))] = 1
    return h


def geometry_audit() -> list[dict]:
    """Compare the reduced and continuation-coordinate representations.

    Returns:
        list[dict]: Dimensions, algebraic residuals, endpoint differences,
            optimizer success indicators, and runtimes for each reference law.
    """
    rows = []
    for K in [3, 5]:
        for J in [4, 12]:
            pop = finite_population(K, J, 0.5)
            A = pop["A"]
            E = pop["E"]
            r = pop["r"]
            Om = pop["Omega"]
            p = pop["p"]
            q = E.T @ r
            # Weighted null projection and transported width agree algebraically.
            gram = (E.T * pop["G"]) @ E
            half1 = 0.1 * np.sqrt(p @ np.linalg.solve(Om, p))
            rH = np.divide(r, pop["G"], out=np.zeros_like(r), where=pop["G"] > 0)
            proj = E @ np.linalg.solve(gram, E.T @ (pop["G"] * rH))
            half2 = 0.1 * np.sqrt(np.sum(pop["G"] * proj**2))
            # Independently solve range-intersected support functions in bridge vs
            # continuation coordinates.
            qc = pop["qR"]
            rho = 0.2

            def bc(sign, *, K=K, p=p, qc=qc, rho=rho, Om=Om):
                """Optimize a support endpoint in crossing-effect coordinates.

                Defaults bind the current population inputs for the callbacks.

                Args:
                    sign (int): One for minimization or minus one for maximization.
                    K (int): Number of crossing visits.
                    p (np.ndarray): Crossing probabilities for the current law.
                    qc (np.ndarray): Observed outcome means in the crossing cells.
                    rho (float): Radius of the quadratic restriction.
                    Om (np.ndarray): Quadratic metric in crossing-effect coordinates.

                Returns:
                    OptimizeResult: The constrained SLSQP result.
                """
                return minimize(
                    lambda z: sign * p @ z,
                    np.zeros(K),
                    jac=lambda z: sign * p,
                    bounds=list(zip(-qc, 1 - qc)),
                    constraints={
                        "type": "ineq",
                        "fun": lambda z: rho**2 - z @ Om @ z,
                        "jac": lambda z: -2 * Om @ z,
                    },
                    method="SLSQP",
                    options={"ftol": 1e-10, "maxiter": 500},
                )

            def cc(sign, *, K=K, J=J, r=r, A=A, rho=rho, pop=pop, qc=qc):
                """Optimize the same support endpoint in continuation coordinates.

                Defaults bind the current population inputs for the callbacks.

                Args:
                    sign (int): One for minimization or minus one for maximization.
                    K (int): Number of crossing visits.
                    J (int): Number of non-triggering states per visit.
                    r (np.ndarray): Objective loadings in continuation coordinates.
                    A (np.ndarray): Linear compatibility constraint matrix.
                    rho (float): Radius of the quadratic restriction.
                    pop (dict): Current population, including state-mass weights G.
                    qc (np.ndarray): Observed outcome means in the crossing cells.

                Returns:
                    OptimizeResult: The SLSQP result with compatibility constraints.
                """
                N = len(r)
                # Null perturbation has boundary box; all other coordinates propagated by A.
                indexes = np.arange(K) * (J + 1) + J
                return minimize(
                    lambda u: sign * r @ u,
                    np.zeros(N),
                    jac=lambda u: sign * r,
                    constraints=[
                        {"type": "eq", "fun": lambda u: A @ u, "jac": lambda u: A},
                        {
                            "type": "ineq",
                            "fun": lambda u: rho**2 - np.sum(pop["G"] * u * u),
                            "jac": lambda u: -2 * pop["G"] * u,
                        },
                        {"type": "ineq", "fun": lambda u: u[indexes] + qc},
                        {"type": "ineq", "fun": lambda u: 1 - qc - u[indexes]},
                    ],
                    method="SLSQP",
                    options={"ftol": 1e-10, "maxiter": 500},
                )

            t = time.perf_counter()
            vb = [bc(1), bc(-1)]
            tb = time.perf_counter() - t
            t = time.perf_counter()
            vc = [cc(1), cc(-1)]
            tc = time.perf_counter() - t
            rows.append(
                {
                    "K": K,
                    "J": J,
                    "continuation_dim": K * (J + 1),
                    "bridge_dim": K,
                    "operator_residual": float(np.max(np.abs(A @ E))),
                    "loading_residual": float(np.max(np.abs(q - p))),
                    "halfwidth_difference": float(abs(half1 - half2)),
                    "constrained_endpoint_difference": float(
                        max(abs(vb[i].fun - vc[i].fun) for i in range(2))
                    ),
                    "all_optimizers_success": all(x.success for x in vb + vc),
                    "bridge_seconds": tb,
                    "continuation_seconds": tc,
                }
            )
    return rows


def continuous(
    n: int,
    rng: np.random.Generator,
    K=5,
    d=5,
    a=0,
    rho=0.35,
    c0=0.6,
    local=False,
    probabilities=False,
):
    """Simulate continuous histories and their crossing-cell outcomes.

    Args:
        n (int): Number of patients to generate.
        rng (np.random.Generator): Generator for covariates, innovations, and outcomes.
        K (int): Number of follow-up visits.
        d (int): Number of baseline covariates; at least five are required.
        a (int): Baseline treatment arm.
        rho (float): Correlation between the two within-visit innovations.
        c0 (float): Intercept of the visit-specific rescue threshold.
        local (bool): Whether to truncate biomarker innovations below the threshold.
        probabilities (bool): Whether to return conditional means instead of draws.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Crossing-cell indices, protocol
            outcomes, and no-rescue outcomes. Index 4*K denotes no crossing.
            With probabilities=True, the last two arrays contain conditional means;
            otherwise, a common uniform couples the two binary outcomes.
    """
    W = rng.normal(size=(n, d))
    for j in range(1, d):
        W[:, j] = 0.35 * W[:, j - 1] + np.sqrt(1 - 0.35**2) * W[:, j]
    S = 0.4 * W[:, 0] - 0.2 * W[:, 1]
    B = 0.3 * W[:, 0] + 0.2 * W[:, 2]
    ss = np.zeros(n)
    T = np.full(n, K, int)
    ex = np.zeros(n)
    cell = np.full(n, 4 * K, int)
    for k in range(K):
        mb = 0.45 * np.tanh(B) + 0.25 * np.tanh(S) - 0.15 * a + 0.10 * np.tanh(W[:, 1])
        thr = c0 + 0.05 * (k + 1) / K
        if local:
            prob = ndtr((thr - mb) / 0.6)
            eb = ndtri(np.maximum(np.finfo(float).tiny, rng.random(n) * prob))
        else:
            eb = rng.normal(size=n)
        es = rho * eb + np.sqrt(1 - rho**2) * rng.normal(size=n)
        S = 0.55 * S - 0.25 * a + 0.15 * W[:, 0] + 0.10 * np.sin(S) + 0.6 * es
        B = mb + 0.6 * eb
        ss += S
        if not local:
            hit = (T == K) & (B >= thr)
            T[hit] = k
            ex[hit] = B[hit] - thr
            cell[hit] = 4 * k + 2 * (W[hit, 0] >= 0).astype(int) + (ex[hit] > 0.5).astype(int)
    lam = (
        -0.20
        + 0.50 * S
        + 0.20 * ss / K
        - 0.20 * a
        + 0.15 * W[:, 2]
        + 0.10 * np.sin(W[:, 0])
        + 0.10 * W[:, 3] * W[:, 4]
    )
    q0 = expit(lam)
    shift = np.where(T < K, 0.7 * (1 + 0.5 * (K - 1 - T) / K + 0.25 * ex), 0)
    qr = expit(lam - shift)
    if probabilities:
        return cell, qr, q0
    uy = rng.random(n)
    return cell, (uy < qr).astype(float), (uy < q0).astype(float)


def summarize(rows: list[dict], groupkeys: list[str]) -> list[dict]:
    """Summarize selected simulation metrics within each group of records.

    Args:
        rows (list[dict]): Replication-level results; empty or None metrics are omitted.
        groupkeys (list[str]): Columns defining the groups to summarize.

    Returns:
        list[dict]: Group identifiers, replication counts, metric means and Monte
            Carlo standard errors, with exact binomial intervals for selected flags.
    """
    groups = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in groupkeys), []).append(r)
    ans = []
    for key, g in groups.items():
        o = dict(zip(groupkeys, key))
        o["replications"] = len(g)
        for name in [
            "length",
            "sharp_width",
            "enlargement",
            "hausdorff",
            "set_cover",
            "target_cover",
            "failure",
            "runtime",
            "empty_rare",
            "primitive_cover",
            "upper_mass",
            "plugin_error",
            "plugin_failure",
        ]:
            z = np.array([r[name] for r in g if name in r and r[name] not in ["", None]], float)
            if z.size:
                o[name] = float(z.mean())
                o[name + "_mcse"] = float(z.std(ddof=1) / np.sqrt(len(z))) if len(z) > 1 else 0.0
                if name in [
                    "set_cover",
                    "target_cover",
                    "failure",
                    "primitive_cover",
                    "empty_rare",
                ]:
                    k = round(z.sum())
                    nn = len(z)
                    o[name + "_mc_lower"] = 0.0 if k == 0 else float(beta.ppf(0.025, k, nn - k + 1))
                    o[name + "_mc_upper"] = (
                        1.0 if k == nn else float(beta.ppf(0.975, k + 1, nn - k))
                    )
        ans.append(o)
    return ans


def cp_counts(k, n, alpha=0.05, D=1):
    """Two-sided Clopper--Pearson; n=0 retains complete uncertainty.

    Quantile inversion uses floating arithmetic. Rational LP certificates apply
    to supplied coefficients, not interval verification of these quantiles.
    """
    k, n = np.broadcast_arrays(np.asarray(k, float), np.asarray(n, float))
    if not 0 < alpha < 1 or D < 1:
        raise ValueError("Require 0 < alpha < 1 and D >= 1.")
    if (
        np.any(~np.isfinite(k))
        or np.any(~np.isfinite(n))
        or np.any(k < 0)
        or np.any(n < k)
        or np.any(k != np.floor(k))
        or np.any(n != np.floor(n))
    ):
        raise ValueError("Binomial counts must be finite integers with 0 <= k <= n.")
    tail = alpha / (2 * D)
    lo = np.zeros(k.shape)
    hi = np.ones(k.shape)
    i = (n > 0) & (k > 0)
    lo[i] = beta.ppf(tail, k[i], n[i] - k[i] + 1)
    i = (n > 0) & (k < n)
    hi[i] = beta.ppf(1 - tail, k[i] + 1, n[i] - k[i])
    return np.maximum(0, lo - 1e-12), np.minimum(1, hi + 1e-12)


@lru_cache(maxsize=100000)
def kl_counts_scalar(k: int, n: int, alpha: float = 0.05, D: int = 1):
    """Bernoulli-KL inversion: KL(phat || p) <= log(2D/alpha)/n."""
    if not 0 < alpha < 1 or D < 1 or n < 0 or not 0 <= k <= n or int(k) != k or int(n) != n:
        raise ValueError("Invalid binomial counts or error allocation.")
    if n == 0:
        return 0.0, 1.0
    x = k / n
    r = np.log(2 * D / alpha) / n
    if k == 0:
        return 0.0, min(1.0, -np.expm1(-r) + 1e-12)
    if k == n:
        return max(0.0, np.exp(-r) - 1e-12), 1.0

    def f(p):
        """Evaluate the Bernoulli-KL constraint at a candidate probability.

        Args:
            p (float): Candidate probability strictly between zero and one.

        Returns:
            float: KL divergence from the empirical probability minus its threshold.
        """
        return xlogy(x, x) - x * np.log(p) + xlogy(1 - x, 1 - x) - (1 - x) * np.log1p(-p) - r

    lo = brentq(f, np.nextafter(0.0, 1.0), x, xtol=1e-14)
    hi = brentq(f, x, np.nextafter(1.0, 0.0), xtol=1e-14)
    return max(0.0, lo - 1e-12), min(1.0, hi + 1e-12)


def primitive_box(F, method, alpha=0.05, D=None):
    """Construct coordinate-wise confidence limits for observed feature means.

    Args:
        F (np.ndarray): An (n, d) matrix of feature values in [0, 1].
        method (str): Limit construction: cp, kl, eb, hoeffding, or wald.
            The cp and kl methods require binary coordinates.
        alpha (float): Total error probability allocated across coordinates.
        D (int | None): Number of protected coordinates; defaults to F.shape[1].

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Empirical means, lower limits,
            and upper limits, each with one entry per feature.
    """
    F = np.asarray(F, float)
    n, dd = F.shape
    if n < 1 or dd < 1 or not np.all(np.isfinite(F)):
        raise ValueError("Moment features must be a nonempty finite matrix.")
    if method in {"cp", "kl"} and not np.all((F == 0) | (F == 1)):
        raise ValueError("CP and Bernoulli-KL require binary feature coordinates.")
    D = dd if D is None else D
    mean = F.mean(0)
    if method == "cp":
        lo, hi = cp_counts(F.sum(0), n, alpha, D)
    elif method == "kl":
        limits = np.array([kl_counts_scalar(round(k), n, alpha, D) for k in F.sum(0)])
        lo, hi = limits[:, 0], limits[:, 1]
    else:
        _, lo, hi = moment_box(F, method, alpha=alpha, total_D=D)
    return mean, lo, hi


def cell_region(F, method, alpha=0.05, extra_D=0):
    """Protect cell moments and two aggregate moments under one error allocation.

    Args:
        F (np.ndarray): Feature matrix ordered as outcome, M crossing indicators,
            and M cell-specific outcome contributions.
        method (str): Primitive limit construction, or hybrid_eb_cp or
            hybrid_hoeffding_cp to use CP limits only for probability features.
        alpha (float): Total error probability for the simultaneous limits.
        extra_D (int): Additional protected coordinates included in the allocation.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Original
            feature means and lower/upper limits, followed by aggregate lower/upper
            limits for total crossing probability and the noncrossing outcome mean
            contribution, in that order.
    """
    M = (F.shape[1] - 1) // 2
    ag = np.column_stack([F[:, 1: 1 + M].sum(1), F[:, 0] - F[:, 1 + M:].sum(1)])  # fmt: skip
    allf = np.column_stack([F, ag])
    D = allf.shape[1] + extra_D
    if method in {"hybrid_eb_cp", "hybrid_hoeffding_cp"}:
        # All 2M+3 coordinates receive the same alpha/D allocation. Only
        # crossing indicators are Bernoulli when outcomes have bounded support.
        outcome_method = "eb" if method == "hybrid_eb_cp" else "hoeffding"
        m, l, u = primitive_box(allf, outcome_method, alpha, D)
        probability = np.r_[np.arange(1, 1 + M), allf.shape[1] - 2]
        l[probability], u[probability] = cp_counts(allf[:, probability].sum(0), len(F), alpha, D)
    else:
        m, l, u = primitive_box(allf, method, alpha, D)
    return m[:-2], l[:-2], u[:-2], l[-2:], u[-2:]


def atomic_constraints(cell, y, M, alpha=0.05, aggregate=False):
    """2(M+1) atoms of (crossing cell,Y), not all longitudinal histories."""
    codes = 2 * cell + y.astype(int)
    F = (codes[:, None] == np.arange(2 * (M + 1))[None, :]).astype(float)
    Q = np.zeros((2 * (M + 1), 1 + 2 * M))
    off = np.zeros(2 * (M + 1))
    for j in range(M):
        Q[2 * j, 1 + j] = 1
        Q[2 * j, 1 + M + j] = -1
        Q[2 * j + 1, 1 + M + j] = 1
    Q[2 * M, 0] = -1
    Q[2 * M, 1: 1 + M] = -1  # fmt: skip
    Q[2 * M, 1 + M:] = 1  # fmt: skip
    off[2 * M] = 1
    Q[2 * M + 1, 0] = 1
    Q[2 * M + 1, 1 + M:] = -1  # fmt: skip
    if aggregate:
        qq = np.zeros((3, 1 + 2 * M))
        qq[0, 0] = 1
        qq[1, 1: 1 + M] = 1  # fmt: skip
        qq[2, 0] = 1
        qq[2, 1 + M:] = -1  # fmt: skip
        Q = np.vstack([Q, qq])
        off = np.r_[off, 0.0, 0.0, 0.0]
        F = np.column_stack([F, y, cell < M, y * (cell == M)])
    _, lo, hi = primitive_box(F, "kl", alpha)
    return Q, off, lo, hi


def signed_budget_exact(mu, p, b, gamma=0.2):
    """Exact capped signed-budget water filling, vectorized for bootstrap."""
    p = np.atleast_2d(np.asarray(p, float))
    b = np.atleast_2d(np.asarray(b, float))
    if not np.isfinite(gamma) or gamma < 0:
        raise ValueError("gamma must be nonnegative and finite.")
    if p.shape != b.shape or np.any(p < -1e-12) or np.any(b < -1e-12) or np.any(b > p + 1e-10):
        raise ValueError("Incoherent crossing and outcome masses.")
    cap = np.maximum(0, p - b)
    q = np.divide(cap, p, out=np.zeros_like(p), where=p > 0)
    active = p > 0
    t = np.zeros_like(p)
    spent = np.zeros(p.shape[0])
    for _ in range(p.shape[1] + 1):
        mass = (p * active).sum(1)
        a = np.sqrt(
            np.divide(
                np.maximum(0, gamma**2 - spent), mass, out=np.zeros_like(mass), where=mass > 0
            )
        )
        sat = active & (q <= a[:, None])
        if not np.any(sat):
            t += np.where(active, p * a[:, None], 0)
            break
        t += np.where(sat, cap, 0)
        spent += np.sum(
            np.where(sat, np.divide(cap**2, p, out=np.zeros_like(p), where=p > 0), 0), axis=1
        )
        active &= ~sat
    else:
        raise RuntimeError("water-filling did not terminate")
    low = np.asarray(mu, float)
    high = low + t.sum(1)
    return low, np.minimum(1, high)


def exact_population(pop, kind):
    """Return analytic population endpoints for the requested sensitivity model.

    Args:
        pop (dict): Population moments with keys ``p``, ``b``, and ``mu``,
            plus ``tau`` for calibration or ``b0`` for Tan's model.
        kind (str): One of ``range``, ``signed``, ``delta``, ``budget``,
            ``calibrated``, or ``tan`` followed by a density-ratio radius.
            The delta and budget models use a radius of 0.2.

    Returns:
        tuple[float, float]: The lower and upper population endpoints.
            The calibrated formula requires inactive outcome caps.
    """
    p, b, mu = pop["p"], pop["b"], pop["mu"]
    if kind == "range":
        return mu - b.sum(), mu - b.sum() + p.sum()
    if kind == "signed":
        return mu, mu + (p - b).sum()
    if kind == "delta":
        return mu, mu + np.minimum(0.2 * p, p - b).sum()
    if kind == "budget":
        l, u = signed_budget_exact(mu, p, b)
        return float(l), float(u[0])
    if kind == "calibrated":
        H = calibration_matrix(len(p))
        d = H @ (p * pop["tau"])
        selected = H.sum(0) > 0
        pg = H @ p
        tg = np.divide(d, pg, out=np.zeros_like(d), where=pg > 0)
        tc = (H.T @ tg) * p
        left = 0.2**2 - float(np.sum(np.divide(d * d, pg, out=np.zeros_like(d), where=pg > 0)))
        if left < -1e-12:
            raise ValueError("infeasible population calibration")
        remain = p[~selected].sum()
        a = np.sqrt(max(0, left) / remain) if remain > 0 else 0.0
        tu = tc.copy()
        tu[~selected] = a * p[~selected]
        if np.all(tc <= p - b + 1e-12) and np.all(tu <= p - b + 1e-12):
            return float(mu + d.sum()), float(mu + tu.sum())
        raise ValueError("Analytic calibrated reference requires inactive caps.")
    if kind.startswith("tan"):
        return tan_population(pop, float(kind[3:]))
    raise ValueError(kind)


def tan_recursion(e, q, lam):
    """Binary time-only primary sensitivity specialization, fixed probabilities."""
    if not np.isfinite(lam) or lam < 1:
        raise ValueError("Tan's density-ratio radius must be finite and >= 1.")
    if not 0 <= q <= 1 or np.any(np.asarray(e) < 0) or np.any(np.asarray(e) > 1):
        raise ValueError("Conditional probabilities must lie in [0,1].")
    low = high = float(q)
    for ee in np.asarray(e)[::-1]:
        qlo = max(low / lam, 1 - lam * (1 - low))
        qhi = min(lam * high, 1 - (1 - high) / lam)
        low = ee * low + (1 - ee) * qlo
        high = ee * high + (1 - ee) * qhi
    return float(low), float(high)


def tan_population(pop, lam):
    """Compute Tan's time-only endpoints from population probabilities.

    Args:
        pop (dict): Population moments with crossing masses ``p`` and
            noncrossing outcome mass ``b0``.
        lam (float): The finite density-ratio radius, at least one.

    Returns:
        tuple[float, float]: The lower and upper population endpoints.
    """
    p = pop["p"]
    s = 1 - np.r_[0, np.cumsum(p)]
    e = s[1:] / s[:-1]
    return tan_recursion(e, pop["b0"] / s[-1], lam)


def tan_interval(cell, y, M, lam, alpha=0.05):
    """Conditional CP on K transitions and terminal outcome, random denominators.

    Lower endpoint increases in e; upper endpoint decreases in e. Consequently
    lower limits on e are used for BOTH outward endpoints, not upper e limits.
    """
    nr = np.array([np.sum(cell >= k) for k in range(M)])
    nk = np.array([np.sum(cell > k) for k in range(M)])
    qn = int(np.sum(cell == M))
    qk = int(np.sum(y[cell == M]))
    elo, _ = cp_counts(nk, nr, alpha, M + 1)
    qlo, qhi = cp_counts(qk, qn, alpha, M + 1)
    return tan_recursion(elo, float(qlo), lam)[0], tan_recursion(elo, float(qhi), lam)[1]


def tan_minimum_lambda(pop):
    """Return the smallest time-only density-ratio radius covering the population.

    Args:
        pop (dict): Population moments with crossing masses ``p``, rescued
            outcome means ``qR``, no-rescue shifts ``tau``, noncrossing outcome
            mass ``b0``, and noncrossing probability ``p0``. Conditional outcome
            probabilities must lie strictly between zero and one.

    Returns:
        float: The maximum ratio, in either direction, of success and failure
            probabilities between each crossing group and its later survivors,
            bounded below by one.
    """
    p = pop["p"]
    q = pop["qR"] + pop["tau"]
    future = pop["b0"]
    mass = pop["p0"]
    val = 1.0
    for k in range(len(p) - 1, -1, -1):
        m = future / mass
        val = max(val, q[k] / m, m / q[k], (1 - q[k]) / (1 - m), (1 - m) / (1 - q[k]))
        future += p[k] * q[k]
        mass += p[k]
    return float(val)


def tan_lp_audit(pop, lam):
    """Independent crossing-outcome mass LP for time-only density-ratio model."""
    p = pop["p"]
    K = len(p)
    q0 = pop["b0"]
    p0 = pop["p0"]
    rows = []
    rhs = []
    for k in range(K):
        s = p0 + p[k + 1:].sum()  # fmt: skip
        c = p[k] / s
        future = np.zeros(K)
        future[k + 1:] = 1  # fmt: skip
        unit = np.zeros(K)
        unit[k] = 1
        rows += [
            -unit + c / lam * future,
            -unit + c * lam * future,
            unit - c * lam * future,
            unit - c / lam * future,
        ]
        rhs += [
            -c * q0 / lam,
            p[k] * (lam - 1) - c * lam * q0,
            c * lam * q0,
            p[k] * (1 - 1 / lam) + c * q0 / lam,
        ]
    out = []
    for direction in [1, -1]:
        res = linprog(
            direction * np.ones(K),
            A_ub=np.array(rows),
            b_ub=np.array(rhs),
            bounds=list(zip(np.zeros(K), p)),
            method="highs",
        )
        if not res.success:
            raise RuntimeError("Tan audit failed")
        out.append(float(q0 + np.sum(res.x)))
    rec = tan_population(pop, lam)
    return max(abs(out[0] - rec[0]), abs(out[1] - rec[1]))


def endpoint_cp(cell, y, M, kind, alpha=0.05):
    """One-sided exact endpoint limits, or valid aggregate budget relaxation."""
    n = len(y)
    if kind == "range":
        lv = y * (cell == M)
        uv = lv + (cell < M)
        return float(cp_counts(lv.sum(), n, alpha, 1)[0]), float(
            cp_counts(uv.sum(), n, alpha, 1)[1]
        )
    if kind == "signed":
        uv = y * (cell == M) + (cell < M)
        return float(cp_counts(y.sum(), n, alpha, 1)[0]), float(cp_counts(uv.sum(), n, alpha, 1)[1])
    if kind == "budget":
        lm, um = cp_counts(y.sum(), n, 2 * alpha / 3, 1)
        _, up = cp_counts(np.sum(cell < M), n, 2 * alpha / 3, 1)
        return float(lm), float(min(1, um + 0.2 * np.sqrt(up)))
    raise ValueError(kind)


def bootstrap_budget(cell, y, M, rng, B=999, alpha=0.05):
    """Joint basic endpoint bootstrap; regular-law comparator, not finite-sample.

    Binary outcomes use equivalent multinomial sufficient-count resampling.
    Other bounded outcomes resample patient-level (cell, Y) pairs directly.
    Exact capped functional is bootstrapped, not an LP tangent approximation.
    """
    cell, y = np.asarray(cell), np.asarray(y, dtype=float)
    n = len(y)
    if (
        n < 1
        or cell.shape != y.shape
        or y.ndim != 1
        or B < 1
        or not np.all(np.isfinite(y))
        or np.any((y < 0) | (y > 1))
        or np.any((cell < 0) | (cell > M) | (cell != np.floor(cell)))
    ):
        raise ValueError("Bootstrap requires bounded patient outcomes and valid crossing cells.")
    cell = cell.astype(int, copy=False)
    if np.all((y == 0) | (y == 1)):
        code = 2 * cell + y.astype(int)
        ph = np.bincount(code, minlength=2 * (M + 1)) / n
        draws = rng.multinomial(n, ph, size=B) / n
        pp = draws[:, : 2 * M].reshape(B, M, 2).sum(2)
        bb = draws[:, 1: 2 * M: 2]  # fmt: skip
        mm = draws[:, 1::2].sum(1)
        p = ph[: 2 * M].reshape(M, 2).sum(1)
        b = ph[1: 2 * M: 2]  # fmt: skip
        mu = ph[1::2].sum()
    else:
        pp, bb, mm = np.empty((B, M)), np.empty((B, M)), np.empty(B)
        # Bound peak memory independently of B and avoid a B x n x M tensor.
        batch = max(1, 262144 // n)
        for start in range(0, B, batch):
            size = min(batch, B - start)
            indices = rng.integers(n, size=(size, n))
            sampled_y = y[indices]
            codes = cell[indices] + (M + 1) * np.arange(size)[:, None]
            counts = np.bincount(codes.ravel(), minlength=size * (M + 1)).reshape(size, M + 1)
            sums = np.bincount(
                codes.ravel(), weights=sampled_y.ravel(), minlength=size * (M + 1)
            ).reshape(size, M + 1)
            pp[start: start + size] = counts[:, :M] / n  # fmt: skip
            bb[start: start + size] = sums[:, :M] / n  # fmt: skip
            mm[start: start + size] = sampled_y.mean(1)  # fmt: skip
        p = np.bincount(cell, minlength=M + 1)[:M] / n
        b = np.bincount(cell, weights=y, minlength=M + 1)[:M] / n
        mu = y.mean()
    Ls, Us = signed_budget_exact(mm, pp, bb)
    L, U = signed_budget_exact(mu, p, b)
    L = float(L)
    U = float(U[0])
    c = max(0.0, float(np.quantile(np.maximum(Ls - L, U - Us), 1 - alpha, method="higher")))
    return max(0.0, L - c), min(1.0, U + c)


def crossing_history_masses(K, J, alpha, arm):
    """Enumerate pre-crossing histories, split by observed outcome."""
    states = np.arange(J)
    paths = [((J - 1) / 2, 1.0)]
    masses = []
    cells = []
    for k in range(K):
        nxt = []
        qr = 0.20 + 0.08 * (k + 1) / K - 0.04 * arm
        for s, pr in paths:
            hit = expit(alpha + 0.6 * (s / (J - 1) - 0.5) + 0.15 * k - 0.10 * arm)
            masses.extend([pr * hit * (1 - qr), pr * hit * qr])
            cells.extend([k, k])
            ker = np.exp(
                -((states - 0.6 * s - 0.25 * (J - 1)) ** 2) / (2 * (0.35 * (J - 1) + 0.3) ** 2)
            )
            ker /= ker.sum()
            if k < K - 1:
                nxt.extend(
                    (float(next_state), float(pr * (1 - hit) * q))
                    for next_state, q in zip(states, ker)
                )
        paths = nxt
    return np.asarray(masses), np.asarray(cells)


def completion_audit(K, J, rescue=0.5, mesh=1 / 64):
    """Independent history-by-Y completion LP."""
    pop = finite_population(K, J, rescue)
    a, j = crossing_history_masses(K, J, pop["alpha"], 0)
    N = len(a)
    p = pop["p"]
    b = pop["b"]
    Q = np.zeros((K, N))
    Q[j, np.arange(N)] = 1
    rows = []
    rhs = []
    budget = 0.2**2
    for k in range(K):
        v = np.zeros(N + K)
        v[:N] = -Q[k]
        rows.append(v)
        rhs.append(-b[k])
        for r in np.linspace(-1, 1, round(2 / mesh) + 1):
            v = np.zeros(N + K)
            v[:N] = 2 * r * Q[k]
            v[N + k] = -1
            rows.append(v)
            rhs.append(r * r * p[k] + 2 * r * b[k])
    v = np.r_[np.zeros(N), np.ones(K)]
    rows.append(v)
    rhs.append(budget)
    A = np.asarray(rows)
    rhs = np.asarray(rhs)
    bounds = list(zip(np.zeros(N), a)) + [(0.0, 4.0)] * K
    vals = []
    certs = []
    ts = time.perf_counter()
    for direction in [1, -1]:
        obj = np.r_[direction * np.ones(N), np.zeros(K)]
        res = linprog(obj, A_ub=A, b_ub=rhs, bounds=bounds, method="highs")
        if not res.success:
            raise RuntimeError("completion audit failed")
        z = _rational_lower(obj, A, rhs, bounds, res.ineqlin.marginals)
        vals.append(pop["b0"] + res.x[:N].sum())
        certs.append(_outward_add(pop["b0"], direction * z, direction))
    sec = time.perf_counter() - ts
    v = np.r_[pop["mu"], p, b]
    ts = time.perf_counter()
    rr = region(v, v, v, K, gamma=0.2, mesh=mesh, refine=False)
    ss = time.perf_counter() - ts
    return {
        "K": K,
        "J": J,
        "completion_variables": N,
        "boundary_variables": K,
        "max_endpoint_difference": max(abs(vals[i] - rr[i].primal_value) for i in [0, 1]),
        "max_outward_difference": max(abs(certs[i] - rr[i].value) for i in [0, 1]),
        "completion_seconds": sec,
        "boundary_seconds": ss,
    }


def append_result(rows, key, kind, method, interval, truth, target, **extra):
    """Append interval endpoints and simulation performance measures to the rows.

    Failed or invalid intervals are replaced by the full outcome range [0, 1].

    Args:
        rows (list[dict]): The result rows to update in place.
        key (dict): Identifiers for the simulation setting and repetition.
        kind (str): The sensitivity model label.
        method (str): The estimation method label.
        interval (tuple[float, float]): The estimated lower and upper endpoints.
        truth (tuple[float, float]): The true identified-set endpoints.
        target (float): The target value used to evaluate point coverage.
        **extra: Additional result fields, optionally including ``failure``.

    Returns:
        None: The result is appended to ``rows``.
    """
    L, U = map(float, interval)
    tl, tu = truth
    fail = int(extra.pop("failure", 0))
    if not np.isfinite(L + U) or L > U:
        fail = 1
    if fail:
        L, U = 0.0, 1.0
    rows.append(
        dict(
            **key,
            kind=kind,
            method=method,
            lower=L,
            upper=U,
            length=U - L,
            sharp_width=tu - tl,
            enlargement=U - L - tu + tl,
            set_cover=int(L <= tl + 1e-10 and U >= tu - 1e-10),
            target_cover=int(L <= target <= U),
            hausdorff=max(abs(L - tl), abs(U - tu)),
            failure=fail,
            **extra,
        )
    )


def obs_design(W: np.ndarray, interactions: bool = True) -> np.ndarray:
    """Build the logistic design matrix from centered baseline covariates.

    Args:
        W (np.ndarray): Baseline covariates with shape (n, 5).
        interactions (bool): Whether to include the two specified interaction terms.

    Returns:
        np.ndarray: Intercept, centered covariates, and optional interactions.
    """
    x = np.asarray(W, float) - 0.5
    z = np.column_stack([np.ones(len(x)), x])
    if interactions:
        z = np.column_stack([z, x[:, 0] * x[:, 1], x[:, 3] * x[:, 4]])
    return z


def obs_propensity(W: np.ndarray, scenario: str) -> np.ndarray:
    """Return the true treatment probabilities for an observational scenario.

    Args:
        W (np.ndarray): Baseline covariates with shape (n, 5).
        scenario (str): Either "linear" or "nonlinear" for the propensity logit.

    Returns:
        np.ndarray: Conditional probabilities of treatment arm one, with shape (n,).
    """
    x = np.asarray(W, float) - 0.5
    v = -0.1 + 0.8 * x[:, 0] - 0.6 * x[:, 1] + 0.4 * x[:, 2]
    if scenario == "nonlinear":
        v = v + 3 * x[:, 0] * x[:, 1] + 2 * x[:, 3] * x[:, 4]
    elif scenario != "linear":
        raise ValueError(scenario)
    return expit(v)


def obs_conditional(al: float, W: np.ndarray, arm: int, K=5, s=4) -> dict:
    """Exact state recursion conditional on each of the 32 baseline patterns."""
    W = np.atleast_2d(W)
    x = W - 0.5
    nw = len(W)
    states = np.arange(s)
    mass = np.ones((nw, 1))
    prev = np.array([(s - 1) / 2])
    pp = []
    for k in range(K):
        hit = expit(
            al
            + 0.6 * (prev[None, :] / (s - 1) - 0.5)
            + 0.15 * k
            - 0.1 * arm
            + 0.65 * x[:, 0, None]
            + 0.35 * x[:, 1, None]
        )
        ker = np.exp(
            -((states[None, :] - 0.6 * prev[:, None] - 0.25 * (s - 1)) ** 2)
            / (2 * (0.35 * (s - 1) + 0.3) ** 2)
        )
        ker /= ker.sum(1, keepdims=True)
        pp.append((mass * hit).sum(1))
        mass = (mass * (1 - hit)) @ ker
        prev = states
    p = np.column_stack(pp)
    baseline = 1.4 * x[:, 0] - 0.8 * x[:, 1] + 2.4 * x[:, 0] * x[:, 1] + 1.6 * x[:, 3] * x[:, 4]
    qr = expit(-0.8 + 0.08 * np.arange(1, K + 1)[None, :] + baseline[:, None] - 0.2 * arm)
    tau = (0.08 + 0.12 * (K - np.arange(1, K + 1)) / K)[None, :] * (0.7 + 0.3 * W[:, 0, None])
    survq = expit(-0.2 + 0.6 * states[None, :] / (s - 1) + baseline[:, None] - 0.2 * arm)
    b0 = (mass * survq).sum(1)
    p0 = mass.sum(1)
    return {"p": p, "p0": p0, "b0": b0, "qR": qr, "tau": tau, "mu": (p * qr).sum(1) + b0}


def observational_population() -> dict:
    """Construct the observational population with an average rescue rate of 0.5.

    Returns:
        dict: The 32 baseline patterns, calibrated intercept, conditional moments,
            and marginal population moments for each treatment arm.
    """
    W = np.array(list(itertools.product([0.0, 1.0], repeat=5)))

    def rate(al):
        """Average the rescue probability over baseline patterns and both arms.

        Args:
            al (float): The rescue-model intercept.

        Returns:
            float: The equally weighted average rescue probability.
        """
        return sum(obs_conditional(al, W, a)["p"].sum() / len(W) for a in [0, 1]) / 2

    al = brentq(lambda z: rate(z) - 0.5, -15, 15, xtol=1e-12)
    cond = {a: obs_conditional(al, W, a) for a in [0, 1]}
    pops = {}
    for a in [0, 1]:
        c = cond[a]
        p = c["p"].mean(0)
        b = (c["p"] * c["qR"]).mean(0)
        t = (c["p"] * c["tau"]).mean(0)
        mu = float(c["mu"].mean())
        pops[a] = {
            "p": p,
            "b": b,
            "mu": mu,
            "theta": float(mu + t.sum()),
            "tau": t / p,
            "b0": float(c["b0"].mean()),
            "K": 5,
        }
    return {
        "W": W,
        "alpha": float(al),
        "cond": cond,
        "pops": pops,
    }


def draw_observational(pop: dict, n: int, scenario: str, rng) -> dict:
    """Draw observational data with coupled observed and no-rescue outcomes.

    Args:
        pop (dict): Population returned by observational_population.
        n (int): The number of individuals to draw.
        scenario (str): The "linear" or "nonlinear" propensity scenario.
        rng (np.random.Generator): The random number generator.

    Returns:
        dict: Baselines W, treatment A, true propensity e, crossing cell (5 means
            no crossing), observed outcome y, and no-rescue outcome ynr.
    """
    wi = rng.integers(0, len(pop["W"]), n)
    W = pop["W"][wi]
    e = obs_propensity(W, scenario)
    A = rng.binomial(1, e)
    cell = np.full(n, 5, int)
    qr = np.zeros(n)
    qnr = np.zeros(n)
    for a in [0, 1]:
        ii = np.flatnonzero(A == a)
        c = pop["cond"][a]
        prob = np.column_stack([c["p"][wi[ii]], c["p0"][wi[ii]]])
        jj = (rng.random(len(ii))[:, None] > np.cumsum(prob, axis=1)).sum(1)
        jj = np.minimum(jj, 5)
        cell[ii] = jj
        q = np.column_stack([c["qR"][wi[ii]], (c["b0"] / c["p0"])[wi[ii]]])
        q0 = np.column_stack([c["qR"][wi[ii]] + c["tau"][wi[ii]], (c["b0"] / c["p0"])[wi[ii]]])
        qr[ii] = q[np.arange(len(ii)), jj]
        qnr[ii] = q0[np.arange(len(ii)), jj]
    u = rng.random(n)
    return {
        "W": W,
        "A": A,
        "e": e,
        "cell": cell,
        "y": (u < qr).astype(float),
        "ynr": (u < qnr).astype(float),
    }


def fitted_logit(W, A, interactions=True):
    """Fit the logistic propensity model and reject numerically irregular fits.

    Args:
        W (np.ndarray): Baseline covariates with shape (n, 5).
        A (np.ndarray): Binary treatment assignments with shape (n,).
        interactions (bool): Whether to include the two baseline interactions.

    Returns:
        tuple: Propensities clipped to [0.02, 0.98], raw propensities, design
            matrix, mean information matrix, and fitted coefficient vector.
    """
    X = obs_design(W, interactions)
    n = len(A)

    def fun(beta_):
        """Evaluate the mean logistic negative log likelihood.

        Args:
            beta_ (np.ndarray): Coefficient vector matching the design columns.

        Returns:
            float: The mean negative log likelihood.
        """
        v = X @ beta_
        return float(np.mean(np.logaddexp(0, v) - A * v))

    def jac(beta_):
        """Evaluate the gradient of the mean logistic negative log likelihood.

        Args:
            beta_ (np.ndarray): Coefficient vector matching the design columns.

        Returns:
            np.ndarray: Gradient vector with one entry per coefficient.
        """
        return X.T @ (expit(X @ beta_) - A) / n

    def hess(beta_):
        """Evaluate the Hessian of the mean logistic negative log likelihood.

        Args:
            beta_ (np.ndarray): Coefficient vector matching the design columns.

        Returns:
            np.ndarray: The square mean information matrix.
        """
        e = expit(X @ beta_)
        return X.T @ ((e * (1 - e))[:, None] * X) / n

    res = minimize(
        fun,
        np.zeros(X.shape[1]),
        jac=jac,
        hess=hess,
        method="Newton-CG",
        options={"xtol": min(1e-9, n ** (-1.5)), "maxiter": 300},
    )
    beta_ = res.x
    info = hess(beta_)
    good = (
        np.all(np.isfinite(beta_))
        and np.max(np.abs(beta_)) < 25
        and np.max(np.abs(jac(beta_))) < min(1e-6, 1 / n)
        and np.linalg.cond(info) < 1e10
    )
    if not good:
        raise RuntimeError("logistic_non-regular_fit")
    eraw = expit(X @ beta_)
    return np.clip(eraw, 0.02, 0.98), eraw, X, info, beta_


def coherent_projection(mean, M):
    """Euclidean projection of (mu,p,b) onto the fixed observed-moment polyhedron."""
    mean = np.asarray(mean, float)
    d = 1 + 2 * M
    A = []
    b = []
    v = np.zeros(d)
    v[1: 1 + M] = 1  # fmt: skip
    A.append(v)
    b.append(1.0)
    v = np.zeros(d)
    v[0] = -1
    v[1 + M:] = 1  # fmt: skip
    A.append(v)
    b.append(0.0)
    v = np.zeros(d)
    v[0] = 1
    v[1: 1 + M] = 1  # fmt: skip
    v[1 + M:] = -1  # fmt: skip
    A.append(v)
    b.append(1.0)
    for j in range(M):
        v = np.zeros(d)
        v[1 + M + j] = 1
        v[1 + j] = -1
        A.append(v)
        b.append(0.0)
    A = np.array(A)
    b = np.array(b)
    if np.min(mean) >= 0 and np.max(mean) <= 1 and np.max(A @ mean - b) <= 1e-12:
        return mean.copy()
    res = minimize(
        lambda x: 0.5 * np.sum((x - mean) ** 2),
        np.clip(mean, 0, 1),
        jac=lambda x: x - mean,
        bounds=[(0, 1)] * d,
        constraints={"type": "ineq", "fun": lambda x: b - A @ x, "jac": lambda x: -A},
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 300},
    )
    if not res.success or np.max(A @ res.x - b) > 1e-8:
        raise RuntimeError("coherence_projection_failure")
    return res.x


def add_aggregates(F, M):
    """Append aggregate crossing and noncrossing outcome features.

    Args:
        F (np.ndarray): Outcome, M crossing indicators, and M crossing outcome
            products, with shape (n, 1 + 2 * M).
        M (int): The number of crossing cells.

    Returns:
        np.ndarray: Features with aggregate crossing and noncrossing outcome
            columns appended, with shape (n, 3 + 2 * M).
    """
    return np.column_stack([F, F[:, 1: 1 + M].sum(1), F[:, 0] - F[:, 1 + M:].sum(1)])  # fmt: skip


def ipw_analysis(dat, pop, method, propensity, fit=None, known_bound=None, gamma=0.2):
    """Arm-zero HT moments. Only known-score EB has the finite-sample guarantee."""
    M = 5
    F = features(dat["cell"], dat["y"], M)
    Q = add_aggregates(F, M)
    A = dat["A"]
    n = len(A)
    D = Q.shape[1]
    e = np.asarray(propensity)
    w = (1 - A) / (1 - e)
    Z = w[:, None] * Q
    m = Z.mean(0)
    IF = Z - m
    if method == "sandwich":
        if fit is None:
            raise ValueError("Sandwich needs fitted logistic score equations.")
        eraw, X, info, _ = fit
        # If clipping is active, the derivative is zero there.
        deriv = w * eraw * ((eraw > 0.02) & (eraw < 0.98))
        B = (Q * deriv[:, None]).T @ X / n
        score = X * (A - eraw)[:, None]
        IF = IF + score @ np.linalg.solve(info, B.T)
    if method == "eb":
        Bmax = 50.0 if known_bound is None else known_bound
        z = np.log(4 * D / 0.05)
        widths = np.sqrt(2 * Z.var(0, ddof=1) * z / n) + 7 * Bmax * z / (3 * (n - 1))
    else:
        widths = norm.ppf(1 - 0.05 / (2 * D)) * IF.std(0, ddof=1) / np.sqrt(n)
    lo, hi = m - widths, m + widths
    plugin = coherent_projection(m[:-2], M)
    pm = np.r_[plugin, plugin[1: 1 + M].sum(), plugin[0] - plugin[1 + M:].sum()]  # fmt: skip
    # Enlargement preserves a valid primitive event when one is available.
    lo = np.minimum(lo, pm)
    hi = np.maximum(hi, pm)
    rr = region(
        plugin,
        lo[:-2],
        hi[:-2],
        M,
        gamma=gamma,
        aggregate_lo=lo[-2:],
        aggregate_hi=hi[-2:],
        refine=False,
    )
    popmom = np.r_[pop["mu"], pop["p"], pop["b"], pop["p"].sum(), pop["mu"] - pop["b"].sum()]
    pl, pu = signed_budget_exact(plugin[0], plugin[1: 1 + M], plugin[1 + M:], gamma)  # fmt: skip
    truth = exact_population(pop, "budget")
    return {
        "interval": [z.value for z in rr],
        "failure": int(any(z.status != "ok" for z in rr)),
        "primitive_cover": int(np.all(popmom >= lo) and np.all(popmom <= hi)),
        "plugin_error": max(abs(float(pl) - truth[0]), abs(float(pu[0]) - truth[1])),
        "lower_error": float(pl) - truth[0],
        "upper_error": float(pu[0]) - truth[1],
        "mu_error": float(plugin[0] - pop["mu"]),
        "raw_mu_error": float(m[0] - pop["mu"]),
        "ess": float(w.sum() ** 2 / (w @ w)) if w @ w > 0 else 0.0,
        "clip_fraction": float(np.mean((e <= 0.02) | (e >= 0.98))),
        "ps_rmse": float(np.sqrt(np.mean((e - dat["e"]) ** 2))),
        "empty_fraction": float(
            np.mean([np.sum((A == 0) & (dat["cell"] == j)) == 0 for j in range(M)])
        ),
        "runtime": sum(z.runtime for z in rr),
        "dual_gap": max(z.dual_gap for z in rr),
    }


def summarize_extended(rows, groupkeys):
    """Summarize simulation groups with additional error and diagnostic metrics.

    Args:
        rows (list[dict]): Per-replication simulation results.
        groupkeys (list[str]): Fields identifying each simulation group.

    Returns:
        list[dict]: Standard summaries plus means, Monte Carlo standard errors,
            and root mean squares for available finite diagnostic values.
    """
    out = summarize(rows, groupkeys)
    groups = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in groupkeys), []).append(r)
    for s in out:
        g = groups[tuple(s[k] for k in groupkeys)]
        for name in [
            "lower_error",
            "upper_error",
            "mu_error",
            "raw_mu_error",
            "ess",
            "clip_fraction",
            "ps_rmse",
            "empty_fraction",
            "coarsening_gap",
            "common_target_error",
        ]:
            vals = np.array([r[name] for r in g if name in r and np.isfinite(r[name])], float)
            if len(vals):
                s[name] = float(vals.mean())
                s[name + "_mcse"] = (
                    float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
                )
                s[name + "_rmse"] = float(np.sqrt(np.mean(vals**2)))
    return out


def continuous_details(n, rng, K=5, d=5, a=0, rho=0.35, c0=0.6):
    """The original continuous DGP, with W and first-crossing labels exposed.

    Uses probabilities for oracle variance reduction and common U for outcomes.
    No post-crossing observed state is treated as a no-rescue observation.
    """
    W = rng.normal(size=(n, d))
    for j in range(1, d):
        W[:, j] = 0.35 * W[:, j - 1] + np.sqrt(1 - 0.35**2) * W[:, j]
    S = 0.4 * W[:, 0] - 0.2 * W[:, 1]
    B = 0.3 * W[:, 0] + 0.2 * W[:, 2]
    ss = np.zeros(n)
    T = np.full(n, K, int)
    ex = np.zeros(n)
    for k in range(K):
        mb = 0.45 * np.tanh(B) + 0.25 * np.tanh(S) - 0.15 * a + 0.10 * np.tanh(W[:, 1])
        eb = rng.normal(size=n)
        es = rho * eb + np.sqrt(1 - rho**2) * rng.normal(size=n)
        S = 0.55 * S - 0.25 * a + 0.15 * W[:, 0] + 0.10 * np.sin(S) + 0.6 * es
        B = mb + 0.6 * eb
        ss += S
        threshold = c0 + 0.05 * (k + 1) / K
        hit = (T == K) & (B >= threshold)
        T[hit] = k
        ex[hit] = B[hit] - threshold
    eta = (
        -0.20
        + 0.50 * S
        + 0.20 * ss / K
        - 0.20 * a
        + 0.15 * W[:, 2]
        + 0.10 * np.sin(W[:, 0])
        + 0.10 * W[:, 3] * W[:, 4]
    )
    q0 = expit(eta)
    shift = np.where(T < K, 0.7 * (1 + 0.5 * (K - 1 - T) / K + 0.25 * ex), 0.0)
    qr = expit(eta - shift)
    return {
        "W": W,
        "T": T,
        "ex": ex,
        "qR": qr,
        "q0": q0,
        "U": rng.random(n),
    }


def partition_cells(dat, M, K=5):
    """Assign crossing cells by visit and baseline normal-quantile bin.

    Args:
        dat (dict): Continuous data containing first-crossing labels T and
            baseline covariates W.
        M (int): The number of crossing cells; either 1 or a multiple of K.
        K (int): The number of visits and the no-crossing label in T.

    Returns:
        np.ndarray: Integer cell labels, with M reserved for no crossing.
    """
    T = dat["T"]
    hit = T < K
    if M == 1:
        return np.where(hit, 0, 1)
    if M % K:
        raise ValueError("M must be 1 or a multiple of K.")
    bins = M // K
    cut = norm.ppf(np.arange(1, bins) / bins)
    risk = np.searchsorted(cut, dat["W"][:, 0], side="right")
    return np.where(hit, bins * T + risk, M).astype(int)


def partition_outcomes(dat, cap=False, K=5):
    """Copy outcome probabilities and optionally impose the near-cap scenario.

    Args:
        dat (dict): Continuous data containing qR, q0, crossing labels T, and W.
        cap (bool): Whether to set qR to 0.98 and q0 to 0.99 among crossing
            individuals whose first baseline covariate is nonnegative.
        K (int): The number of visits and the no-crossing label in T.

    Returns:
        tuple[np.ndarray, np.ndarray]: Observed and no-rescue outcome probabilities.
    """
    qr = dat["qR"].copy()
    q0 = dat["q0"].copy()
    if cap:
        hi = (dat["T"] < K) & (dat["W"][:, 0] >= 0)
        qr[hi] = 0.98
        q0[hi] = 0.99
    return qr, q0
