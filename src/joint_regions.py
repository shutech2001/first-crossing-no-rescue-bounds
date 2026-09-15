from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal, localcontext
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq, linprog, minimize
from scipy.special import logsumexp
from scipy.stats import chi2

from methods import Bound, _rational_lower, signed_budget_exact

INTERVAL_DIGITS = 45


@dataclass(frozen=True)
class _Interval:
    """Small directed-rounding interval implementation for likelihood cuts."""

    lo: Decimal
    hi: Decimal

    @classmethod
    def exact(cls, x: float) -> _Interval:
        """Return the exact interval.

        Args:
            x (float): The value.

        Returns:
            _Interval: The exact interval.
        """
        d = Decimal.from_float(float(x)) if isinstance(x, (float, np.floating)) else Decimal(int(x))
        return cls(d, d)

    def __add__(self, other: float | _Interval) -> _Interval:
        """Return the sum of the interval and another value.

        Args:
            other (float | _Interval): The other value.

        Returns:
            _Interval: The sum of the interval and another value.
        """
        other = other if isinstance(other, _Interval) else self.exact(other)
        with localcontext() as ctx:
            ctx.prec, ctx.rounding = INTERVAL_DIGITS, ROUND_FLOOR
            lo = self.lo + other.lo
            ctx.rounding = ROUND_CEILING
            hi = self.hi + other.hi
        return _Interval(lo, hi)

    __radd__ = __add__

    def __neg__(self) -> _Interval:
        """Return the negation of the interval.

        Returns:
            _Interval: The negation of the interval.
        """
        # Decimal unary minus can round under the ambient precision.
        return _Interval(self.hi.copy_negate(), self.lo.copy_negate())

    def __sub__(self, other: float | _Interval) -> _Interval:
        """Return the difference of the interval and another value.

        Args:
            other (float | _Interval): The other value.

        Returns:
            _Interval: The difference of the interval and another value.
        """
        return self + -(other if isinstance(other, _Interval) else self.exact(other))

    def __mul__(self, other: float | _Interval) -> _Interval:
        """Return the product of the interval and another value.

        Args:
            other (float | _Interval): The other value.

        Returns:
            _Interval: The product of the interval and another value.
        """
        other = other if isinstance(other, _Interval) else self.exact(other)
        with localcontext() as ctx:
            ctx.prec, ctx.rounding = INTERVAL_DIGITS, ROUND_FLOOR
            lo = min(a * b for a in (self.lo, self.hi) for b in (other.lo, other.hi))
            ctx.rounding = ROUND_CEILING
            hi = max(a * b for a in (self.lo, self.hi) for b in (other.lo, other.hi))
        return _Interval(lo, hi)

    __rmul__ = __mul__

    def __truediv__(self, other: float | _Interval) -> _Interval:
        """Return the quotient of the interval and another value.

        Args:
            other (float | _Interval): The other value.

        Returns:
            _Interval: The quotient of the interval and another value.
        """
        other = other if isinstance(other, _Interval) else self.exact(other)
        if other.lo <= 0 <= other.hi:
            raise ZeroDivisionError("Interval denominator includes zero.")
        with localcontext() as ctx:
            ctx.prec, ctx.rounding = INTERVAL_DIGITS, ROUND_FLOOR
            lo = min(a / b for a in (self.lo, self.hi) for b in (other.lo, other.hi))
            ctx.rounding = ROUND_CEILING
            hi = max(a / b for a in (self.lo, self.hi) for b in (other.lo, other.hi))
        return _Interval(lo, hi)

    def log(self):
        if self.lo <= 0:
            raise ValueError("Log interval requires positive arguments.")
        with localcontext() as ctx:
            ctx.prec = INTERVAL_DIGITS
            # Decimal.ln is correctly rounded with ROUND_HALF_EVEN, irrespective
            # of ctx.rounding; adjacent 45-digit numbers enclose the exact log.
            lo = self.lo.ln().next_minus()
            hi = self.hi.ln().next_plus()
        return _Interval(lo, hi)


def _float_upper(x: Decimal) -> float:
    """Return the upper float of the interval.

    Args:
        x (Decimal): The interval.

    Returns:
        float: The upper float of the interval.
    """
    value = float(x)
    if Decimal.from_float(value) < x:
        value = float(np.nextafter(value, np.inf))
    return value


class LikelihoodRegion:
    """Declared-atom likelihood criterion and valid affine outer cuts.

    ``counts_halves`` has shape ``(2, M + 1, 2)``. Rows 0 through M-1
    denote crossing cells and the final row denotes noncrossing. Outcomes are
    ordered (0, 1). The LR degrees of freedom always equal ``2*(M+1)-1``.
    """

    def __init__(self, counts_halves, method="split", alpha=0.05):
        """Initialize the likelihood region.

        Args:
            counts_halves (NDArray[np.int64]): The counts halves.
            method (str): The method.
            alpha (float): The alpha.
        """
        counts = np.asarray(counts_halves)
        if (
            counts.ndim != 3
            or counts.shape[0] != 2
            or counts.shape[2] != 2
            or counts.shape[1] < 2
            or not np.all(np.isfinite(counts))
            or np.any(counts < 0)
            or np.any(counts != np.floor(counts))
        ):
            raise ValueError("Counts must be nonnegative integers with shape (2, M+1, 2).")
        if not 0 < alpha < 1:
            raise ValueError("alpha must lie in (0,1).")
        counts = counts.astype(np.int64).reshape(2, -1)
        if np.any(counts.sum(axis=1) == 0):
            raise ValueError("Both data-independent halves must be nonempty.")
        self.method = method.lower().removeprefix("joint ")
        if self.method not in ("split", "lr"):
            raise ValueError("Joint likelihood method must be 'split' or 'lr'.")
        self.dimension = counts.shape[1]
        self.M = self.dimension // 2 - 1
        self.alpha = float(alpha)
        self.pooled_counts = counts.sum(axis=0)
        self.n = int(self.pooled_counts.sum())
        self.empirical = self.pooled_counts / self.n
        self.counts = counts if self.method == "split" else self.pooled_counts[None, :]
        self.B = len(self.counts)
        self.constants_interval = []
        for v, row in enumerate(self.counts):
            constant = _Interval.exact(0)
            for i in np.flatnonzero(row):
                if self.method == "split":
                    probability = _Interval.exact(2 * int(counts[1 - v, i]) + 1) / (
                        2 * int(counts[1 - v].sum()) + self.dimension
                    )
                else:
                    probability = _Interval.exact(int(row[i])) / self.n
                constant = constant + int(row[i]) * probability.log()
            self.constants_interval.append(constant)
        self.constants = np.array([float(c.lo) for c in self.constants_interval])
        if self.method == "split":
            self.threshold_interval = -_Interval.exact(self.alpha).log()
            self.threshold = -math.log(self.alpha)
        else:
            # This is an outward guard for a supplied special-function value,
            # not a verified enclosure of the mathematical chi-square quantile.
            self.threshold = float((chi2.ppf(1 - self.alpha, self.dimension - 1) + 1e-12) / 2)
            self.threshold_interval = _Interval.exact(self.threshold)
        # Power-of-two row scaling preserves the floating-point cut exactly.
        self.cut_scale = math.ldexp(1.0, -math.ceil(math.log2(self.n)))

    def _scores(self, probabilities: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return the scores.

        Args:
            probabilities (NDArray[np.float64]): The probabilities.

        Returns:
            NDArray[np.float64]: The scores.
        """
        pi = np.asarray(probabilities, float).reshape(-1)
        if pi.shape != (self.dimension,) or np.any(pi < 0) or not np.all(np.isfinite(pi)):
            return np.full(self.B, np.inf)
        scores = self.constants.copy()
        for v, row in enumerate(self.counts):
            observed = row > 0
            if np.any(pi[observed] <= 0):
                scores[v] = np.inf
            else:
                scores[v] -= row[observed] @ np.log(pi[observed])
        return scores

    def criterion(self, probabilities: NDArray[np.float64]) -> float:
        """Return the criterion.

        Args:
            probabilities (NDArray[np.float64]): The probabilities.

        Returns:
            float: The criterion.
        """
        return float(logsumexp(self._scores(probabilities)) - math.log(self.B))

    def gradient(self, probabilities: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return the gradient.

        Args:
            probabilities (NDArray[np.float64]): The probabilities.

        Returns:
            NDArray[np.float64]: The gradient.
        """
        pi = np.asarray(probabilities, float).reshape(-1)
        scores = self._scores(pi)
        weights = np.exp(scores - logsumexp(scores))
        averaged_counts = weights @ self.counts
        return -np.divide(averaged_counts, pi, out=np.zeros_like(pi), where=pi > 0)

    def affine_cut(self, reference: NDArray[np.float64]) -> tuple[NDArray[np.float64], float]:
        """Return ``a, b`` such that every region member satisfies ``a @ pi <= b``.

        Args:
            reference (NDArray[np.float64]): The reference.

        Returns:
            tuple[NDArray[np.float64], float]: The affine cut.
        """
        v = np.asarray(reference, float).reshape(-1)
        if v.shape != (self.dimension,) or np.any(v <= 0) or not np.all(np.isfinite(v)):
            raise ValueError("Likelihood-cut reference must be strictly positive.")
        scores = self._scores(v)
        if self.B == 1:
            weights = np.ones(1)
        else:
            soft = float(np.exp(scores[0] - logsumexp(scores)))
            numerator = min(2**40, max(0, round(soft * 2**40)))
            weights = np.array([numerator / 2**40, (2**40 - numerator) / 2**40])
        w = [_Interval.exact(float(z)) for z in weights]
        intercept = -_Interval.exact(self.B).log()
        for weight, scalar_weight, constant in zip(w, weights, self.constants_interval):
            intercept = intercept + weight * constant
            if scalar_weight:
                intercept = intercept - weight * weight.log()
        coefficients = np.zeros(self.dimension)
        rounding_guard = _Interval.exact(0)
        for i in range(self.dimension):
            count = sum((w[j] * int(self.counts[j, i]) for j in range(self.B)), _Interval.exact(0))
            value = _Interval.exact(float(v[i]))
            intercept = intercept + count * (_Interval.exact(1) - value.log())
            exact_coefficient = -count / value
            supplied = float(exact_coefficient.lo)
            coefficients[i] = supplied
            # max(0, supplied-exact) bounds the discrepancy for 0 <= pi_i <= 1.
            difference = _Interval.exact(supplied) - exact_coefficient
            rounding_guard = rounding_guard + _Interval(
                max(Decimal(0), difference.lo), max(Decimal(0), difference.hi)
            )
        rhs = _float_upper((self.threshold_interval - intercept + rounding_guard).hi)
        return coefficients * self.cut_scale, rhs * self.cut_scale


@dataclass
class JointBound(Bound):
    """Joint bound.

    Args:
        Bound (Bound): The bound.
        iterations (int): The iterations.
        exhausted (bool): Whether the bound is exhausted.
        fallback (bool): Whether the bound is a fallback.
        likelihood_violation (float): The likelihood violation.
        candidate_value (float): The candidate value.
        smooth_success (bool): Whether the bound is a smooth success.
    """

    iterations: int = 0
    exhausted: bool = False
    fallback: bool = False
    likelihood_violation: float = float("nan")
    candidate_value: float = float("nan")
    smooth_success: bool = False


@lru_cache(maxsize=32)
def _scientific_lp(
    M: int, gamma: float, mesh: float
) -> tuple[list[NDArray[np.float64]], list[float]]:
    """Return the scientific LP.

    Args:
        M (int): The number of moments.
        gamma (float): The gamma.
        mesh (float): The mesh.

    Returns:
        tuple[list[NDArray[np.float64]], list[float]]: The scientific LP.
    """
    steps = round(1 / mesh)
    if steps < 1 or steps & (steps - 1) or abs(steps * mesh - 1) > 1e-12:
        raise ValueError("mesh must be a dyadic subdivision of [0,1].")
    dimension = 2 * (M + 1)
    nv = dimension + 2 * M
    rows, rhs = [], []

    def row(values: dict[int, float], value: float) -> None:
        """Add a row to the scientific LP.

        Args:
            values (dict[int, float]): The values.
            value (float): The value.
        """
        a = np.zeros(nv)
        for i, x in values.items():
            a[i] = x
        rows.append(a)
        rhs.append(value)

    # Both simplex inequalities are included in the rational dual certificate.
    row({i: 1.0 for i in range(dimension)}, 1.0)
    row({i: -1.0 for i in range(dimension)}, -1.0)
    for j in range(M):
        row({dimension + j: 1.0, 2 * j: -1.0}, 0.0)
        # t_j >= 0 is a variable bound. Negative-grid tangents are redundant
        # under this sign restriction; the retained grid has exactly h=1/128.
        for r in np.linspace(0, 1, steps + 1):
            row(
                {2 * j: -r * r, 2 * j + 1: -r * r, dimension + j: 2 * r, dimension + M + j: -1.0},
                0.0,
            )
    row({dimension + M + j: 1.0 for j in range(M)}, gamma**2)
    objective = np.zeros(nv)
    objective[1:dimension:2] = 1
    objective[dimension: dimension + M] = 1  # fmt: skip
    box = [(0.0, 1.0)] * nv
    if gamma == 0:
        box[dimension: dimension + M] = [(0.0, 0.0)] * M  # fmt: skip
    return np.asarray(rows), np.asarray(rhs), objective, box


def _normalize(pi: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the normalized probabilities.

    Args:
        pi (NDArray[np.float64]): The probabilities.

    Returns:
        NDArray[np.float64]: The normalized probabilities.
    """
    pi = np.maximum(0, np.asarray(pi, float))
    return pi / pi.sum()


def _waterfill(
    region: LikelihoodRegion, pi: NDArray[np.float64], gamma: float
) -> tuple[float, float]:
    """Return the waterfill.

    Args:
        region (LikelihoodRegion): The region.
        pi (NDArray[np.float64]): The probabilities.
        gamma (float): The gamma.

    Returns:
        tuple[float, float]: The waterfill.
    """
    masses = pi.reshape(-1, 2)
    p = masses[:-1].sum(axis=1)
    b = masses[:-1, 1]
    mu = float(masses[:, 1].sum())
    lo, hi = signed_budget_exact(mu, p, b, gamma)
    return float(lo), float(hi[0])


def _candidate_violation(
    region: LikelihoodRegion, pi: NDArray[np.float64], gamma: float, direction: int
) -> float:
    """Return the candidate violation.

    Args:
        region (LikelihoodRegion): The region.
        pi (NDArray[np.float64]): The probabilities.
        gamma (float): The gamma.
        direction (int): The direction.

    Returns:
        float: The candidate violation.
    """
    masses = pi.reshape(-1, 2)
    p, b = masses[:-1].sum(axis=1), masses[:-1, 1]
    cap = np.maximum(0, p - b)
    t = np.zeros_like(p)
    if direction == -1:
        active = p > 0
        q = np.divide(cap, p, out=np.zeros_like(p), where=p > 0)
        spent = 0.0
        for _ in range(region.M + 1):
            mass = float(p[active].sum())
            level = math.sqrt(max(0.0, gamma**2 - spent) / mass) if mass else 0.0
            saturated = active & (q <= level)
            if not np.any(saturated):
                t[active] = p[active] * level
                break
            t[saturated] = cap[saturated]
            spent += float(np.sum(cap[saturated] ** 2 / p[saturated]))
            active[saturated] = False
    energy = float(np.sum(np.divide(t**2, p, out=np.zeros_like(p), where=p > 0)))
    value = float(masses[:, 1].sum() + t.sum())
    reported = _waterfill(region, pi, gamma)[0 if direction == 1 else 1]
    return max(
        0.0,
        region.criterion(pi) - region.threshold,
        abs(float(pi.sum()) - 1),
        float(-pi.min()),
        float(np.max(t - cap)),
        float(-t.min()),
        energy - gamma**2,
        abs(value - reported),
    )


def _feasible_anchor(region: LikelihoodRegion) -> NDArray[np.float64]:
    """Minimize the convex criterion if the empirical point is not interior.

    Args:
        region (LikelihoodRegion): The region.

    Returns:
        NDArray[np.float64]: The feasible anchor.
    """
    pi = region.empirical.copy()
    if region.criterion(pi) < region.threshold - 1e-7:
        return pi
    bounds = [(1e-14 if n else 0.0, 1.0) for n in region.pooled_counts]
    result = minimize(
        lambda x: region.criterion(x) / region.n,
        pi,
        jac=lambda x: region.gradient(x) / region.n,
        bounds=bounds,
        constraints={
            "type": "eq",
            "fun": lambda x: x.sum() - 1,
            "jac": lambda x: np.ones(region.dimension),
        },
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 300},
    )
    candidate = _normalize(result.x)
    return candidate if region.criterion(candidate) <= region.criterion(pi) else pi


def _repair_candidate(
    region: LikelihoodRegion, candidate: NDArray[np.float64], anchor: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Floating-point diagnostic repair along a segment inside the simplex.

    Args:
        region (LikelihoodRegion): The region.
        candidate (NDArray[np.float64]): The candidate.
        anchor (NDArray[np.float64]): The anchor.

    Returns:
        NDArray[np.float64]: The repaired candidate.
    """
    pi = _normalize(candidate)
    if region.criterion(pi) <= region.threshold:
        return pi
    if region.criterion(anchor) > region.threshold:
        return None
    lo, hi = 0.0, 1.0
    for _ in range(55):
        fraction = (lo + hi) / 2
        point = (1 - fraction) * anchor + fraction * pi
        if region.criterion(point) <= region.threshold - 1e-10:
            lo = fraction
        else:
            hi = fraction
    return (1 - lo) * anchor + lo * pi


def _profile_mean(region: LikelihoodRegion, mean: float) -> NDArray[np.float64]:
    """Smooth multinomial profile at a fixed outcome mean.

    For split likelihood the conditional MLE is a weighted count vector. Its
    optimal mixture weight is the root of a monotone one-dimensional score
    equation. This retains every atom, including unobserved outcomes/cells.

    Args:
        region (LikelihoodRegion): The region.
        mean (float): The mean.

    Returns:
        NDArray[np.float64]: The profile mean.
    """

    def probabilities(weight: float) -> NDArray[np.float64]:
        """Return the probabilities.

        Args:
            weight (float): The weight.

        Returns:
            NDArray[np.float64]: The probabilities.
        """
        counts = (
            region.counts[0]
            if region.B == 1
            else (weight * region.counts[0] + (1 - weight) * region.counts[1])
        )
        pi = np.zeros(region.dimension)
        for y, mass in enumerate((1 - mean, mean)):
            subset = counts[y::2]
            if subset.sum() == 0:
                subset = region.pooled_counts[y::2]
            if subset.sum() == 0:
                subset = np.ones_like(subset)
            pi[y::2] = mass * subset / subset.sum()
        return pi

    if region.B == 1:
        return probabilities(1.0)
    # A boundary mean incompatible with any observed outcome has infinite
    # criterion; no mixture-weight solve is needed for that root bracket.
    if (mean == 0 and region.pooled_counts[1::2].sum()) or (
        mean == 1 and region.pooled_counts[0::2].sum()
    ):
        return probabilities(0.5)

    def score(weight):
        scores = region._scores(probabilities(weight))
        if np.isposinf(scores[0]):
            return weight - 1
        if np.isposinf(scores[1]):
            return weight
        return weight - float(np.exp(scores[0] - logsumexp(scores)))

    weight = brentq(score, 0.0, 1.0, xtol=1e-14, rtol=1e-14)
    return probabilities(weight)


def _profile_endpoint(
    region: LikelihoodRegion, direction: int, anchor: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Constrained scalar smooth solve after profiling the conditional atoms.

    Args:
        region (LikelihoodRegion): The region.
        direction (int): The direction.
        anchor (NDArray[np.float64]): The anchor.

    Returns:
        NDArray[np.float64]: The profile endpoint.
    """
    anchor_mean = float(anchor[1::2].sum())
    boundary = 0.0 if direction == 1 else 1.0
    edge = _profile_mean(region, boundary)
    if region.criterion(edge) <= region.threshold:
        return edge
    mean = brentq(
        lambda x: region.criterion(_profile_mean(region, x)) - region.threshold,
        min(boundary, anchor_mean),
        max(boundary, anchor_mean),
        xtol=1e-14,
        rtol=1e-14,
    )
    return _repair_candidate(region, _profile_mean(region, mean), anchor)


def _smooth_candidate(
    region: LikelihoodRegion, direction: int, gamma: float, anchor: NDArray[np.float64]
) -> tuple[NDArray[np.float64], bool]:
    """Convex smooth solve in (pi,t); the returned value is only a diagnostic.

    Args:
        region (LikelihoodRegion): The region.
        direction (int): The direction.
        gamma (float): The gamma.
        anchor (NDArray[np.float64]): The anchor.

    Returns:
        tuple[NDArray[np.float64], bool]: The smooth candidate.
    """
    d, M = region.dimension, region.M
    if direction == 1 or gamma == 0:
        # The signed lower endpoint has t=0. Profiling avoids degenerate
        # inactive perspective constraints and gives an accurate cut location.
        return _profile_endpoint(region, direction, anchor), True
    objective = np.zeros(d + M)
    objective[1:d:2] = direction
    objective[d:] = direction
    x0 = np.r_[anchor, np.zeros(M)]
    cap_jac = np.zeros((M, d + M))
    for j in range(M):
        cap_jac[j, 2 * j] = 1
        cap_jac[j, d + j] = -1

    def budget(x: NDArray[np.float64]) -> float:
        """Return the budget.

        Args:
            x (NDArray[np.float64]): The x.

        Returns:
            float: The budget.
        """
        p = x[: 2 * M].reshape(M, 2).sum(axis=1)
        t = x[d:]
        return gamma**2 - np.sum(np.divide(t * t, p, out=np.zeros(M), where=p > 0))

    def budget_jac(x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return the budget Jacobian.

        Args:
            x (NDArray[np.float64]): The x.

        Returns:
            NDArray[np.float64]: The budget Jacobian.
        """
        p = x[: 2 * M].reshape(M, 2).sum(axis=1)
        ratio = np.divide(x[d:], p, out=np.zeros(M), where=p > 0)
        grad = np.zeros(d + M)
        grad[: 2 * M] = np.repeat(ratio**2, 2)
        grad[d:] = -2 * ratio
        return grad

    constraints = [
        {
            "type": "eq",
            "fun": lambda x: x[:d].sum() - 1,
            "jac": lambda x: np.r_[np.ones(d), np.zeros(M)],
        },
        {
            "type": "ineq",
            "fun": lambda x: (region.threshold - region.criterion(x[:d])) / region.n,
            "jac": lambda x: np.r_[-region.gradient(x[:d]) / region.n, np.zeros(M)],
        },
        {"type": "ineq", "fun": lambda x: cap_jac @ x, "jac": lambda x: cap_jac},
        {"type": "ineq", "fun": budget, "jac": budget_jac},
    ]
    result = minimize(
        lambda x: objective @ x,
        x0,
        jac=lambda x: objective,
        bounds=[(1e-14 if n else 0.0, 1.0) for n in region.pooled_counts] + [(0, 1)] * M,
        constraints=constraints,
        method="SLSQP",
        options={"ftol": 1e-11, "maxiter": 400},
    )
    pi = _repair_candidate(region, result.x[:d], anchor)
    return pi, bool(result.success)


def _solve_endpoint(
    region: LikelihoodRegion,
    direction: int,
    gamma: float,
    mesh: float,
    max_iterations: int,
    gap_tolerance: float,
    anchor: NDArray[np.float64],
) -> JointBound:
    """Solve the endpoint.

    Args:
        region (LikelihoodRegion): The region.
        direction (int): The direction.
        gamma (float): The gamma.
        mesh (float): The mesh.
        max_iterations (int): The maximum iterations.
        gap_tolerance (float): The gap tolerance.
        anchor (NDArray[np.float64]): The anchor.

    Returns:
        JointBound: The endpoint.
    """
    start = time.perf_counter()
    d, M = region.dimension, region.M
    A0, b0, objective, box = _scientific_lp(M, gamma, mesh)
    objective = direction * objective
    nv = len(objective)
    cuts, cut_rhs = [], []
    best_pi = None
    best_inner = float("inf") if direction == 1 else float("-inf")
    smooth_ok = False

    def update(pi: NDArray[np.float64]) -> None:
        """Update the best pi and best inner.

        Args:
            pi (NDArray[np.float64]): The pi.
        """
        nonlocal best_pi, best_inner
        if pi is None or region.criterion(pi) > region.threshold + 1e-8:
            return
        value = _waterfill(region, pi, gamma)[0 if direction == 1 else 1]
        if direction * value < direction * best_inner:
            best_pi, best_inner = pi.copy(), value

    def failure(status: str, iterations: int) -> JointBound:
        """Return the failure.

        Args:
            status (str): The status.
            iterations (int): The iterations.

        Returns:
            JointBound: The failure.
        """
        return JointBound(
            value=0.0 if direction == 1 else 1.0,
            primal_value=np.nan,
            status=status,
            runtime=time.perf_counter() - start,
            violation=np.nan,
            dual_gap=np.nan,
            mesh=mesh,
            iterations=iterations,
            fallback=True,
        )

    update(anchor)
    try:
        smooth_pi, smooth_ok = _smooth_candidate(region, direction, gamma, anchor)
        update(smooth_pi)
    except (ArithmeticError, ValueError, RuntimeError):
        smooth_pi = None
    reference = best_pi if best_pi is not None else anchor
    result = None
    outer = np.nan
    lower = np.nan
    gap = np.inf
    for iteration in range(1, max_iterations + 1):
        try:
            a, b = region.affine_cut(np.maximum(reference, 1e-14))
        except (ArithmeticError, ValueError, FloatingPointError):
            return failure("likelihood_cut_failure", iteration)
        row = np.zeros(nv)
        row[:d] = a
        cuts.append(row)
        cut_rhs.append(b)
        A = np.vstack([A0, cuts])
        rhs = np.r_[b0, cut_rhs]
        try:
            result = linprog(
                objective,
                A_ub=A,
                b_ub=rhs,
                bounds=box,
                method="highs",
                options={"primal_feasibility_tolerance": 1e-8, "dual_feasibility_tolerance": 1e-8},
            )
        except (ArithmeticError, ValueError, RuntimeError):
            return failure("lp_exception", iteration)
        if not result.success:
            return failure("lp_failure_" + str(result.status), iteration)
        try:
            lower = _rational_lower(objective, A, rhs, box, result.ineqlin.marginals)
            if not np.isfinite(lower):
                raise ValueError("Non-finite objective certificate.")
        except (ArithmeticError, ValueError, TypeError, AttributeError):
            return failure("certificate_failure", iteration)
        outer = float(np.clip(direction * lower, 0, 1))
        if best_pi is not None:
            gap = max(0.0, direction * (best_inner - outer))
            if gap <= gap_tolerance:
                break
        candidate = _repair_candidate(region, result.x[:d], anchor)
        update(candidate)
        if best_pi is not None:
            gap = max(0.0, direction * (best_inner - outer))
            if gap <= gap_tolerance:
                break
        reference = candidate if candidate is not None else _normalize(result.x[:d] + 1e-12)
    primal = direction * float(result.fun)
    pi, t = result.x[:d], result.x[d: d + M]  # fmt: skip
    p = pi[: 2 * M].reshape(M, 2).sum(axis=1)
    budget_vio = max(
        0.0, float(np.sum(np.divide(t**2, p, out=np.zeros(M), where=p > 0))) - gamma**2
    )
    likelihood_vio = max(0.0, region.criterion(pi) - region.threshold)
    linear_vio = max(0.0, float(np.max(A @ result.x - rhs)), float(-result.x.min()))
    inner_vio = (
        _candidate_violation(region, best_pi, gamma, direction) if best_pi is not None else np.nan
    )
    exhausted = gap > gap_tolerance
    return JointBound(
        value=outer,
        primal_value=primal,
        status="ok",
        runtime=time.perf_counter() - start,
        violation=max(budget_vio, linear_vio),
        dual_gap=float(result.fun - lower),
        witness=result.x,
        mesh=mesh,
        refinement_count=0,
        inner_gap=float(gap),
        inner_violation=inner_vio,
        inner_status=(
            ("residual_checked" if not exhausted else "gap_above_tolerance")
            if best_pi is not None
            else "candidate_unavailable"
        ),
        budget_allowance=mesh**2 / 4 if gamma > 0 else 0.0,
        iterations=iteration,
        exhausted=bool(exhausted),
        fallback=False,
        likelihood_violation=likelihood_vio,
        candidate_value=float(best_inner) if best_pi is not None else np.nan,
        smooth_success=smooth_ok,
    )


def joint_region(
    counts_halves,
    method="split",
    alpha=0.05,
    gamma=0.2,
    mesh=1 / 128,
    max_iterations=120,
    gap_tolerance=1e-4,
) -> tuple[JointBound, JointBound]:
    """Return certified lower/upper bounds for the capped nonnegative model.

    Args:
        counts_halves (NDArray[np.int64]): The counts halves.
        method (str): The method.
        alpha (float): The alpha.
        gamma (float): The gamma.
        mesh (float): The mesh.
        max_iterations (int): The maximum iterations.
        gap_tolerance (float): The gap tolerance.

    Returns:
        tuple[JointBound, JointBound]: The certified lower/upper bounds.
    """
    if not np.isfinite(gamma) or not 0 <= gamma <= 1:
        raise ValueError("gamma must lie in [0,1].")
    if not np.isfinite(mesh) or mesh <= 0 or mesh > 1:
        raise ValueError("mesh must lie in (0,1].")
    if not isinstance(max_iterations, int) or not 1 <= max_iterations <= 120:
        raise ValueError("The benchmark permits 1 through 120 likelihood-cut iterations.")
    if not np.isfinite(gap_tolerance) or gap_tolerance <= 0:
        raise ValueError("gap_tolerance must be positive.")
    region = LikelihoodRegion(counts_halves, method, alpha)
    # Validate grid input before numerical failures are converted into fallbacks.
    _scientific_lp(region.M, float(gamma), float(mesh))
    start = time.perf_counter()
    try:
        anchor = _feasible_anchor(region)
    except (ArithmeticError, ValueError, RuntimeError):
        return tuple(
            JointBound(
                value=float(i),
                primal_value=np.nan,
                status="anchor_failure",
                runtime=(time.perf_counter() - start) / 2,
                violation=np.nan,
                dual_gap=np.nan,
                mesh=mesh,
                fallback=True,
            )
            for i in range(2)
        )
    anchor_runtime = time.perf_counter() - start
    result = tuple(
        _solve_endpoint(
            region, direction, float(gamma), float(mesh), max_iterations, gap_tolerance, anchor
        )
        for direction in (1, -1)
    )
    result = tuple(replace(b, runtime=b.runtime + anchor_runtime / 2) for b in result)
    if any(b.fallback for b in result):
        result = tuple(replace(b, value=float(i), fallback=True) for i, b in enumerate(result))
    return result
