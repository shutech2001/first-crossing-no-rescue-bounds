from __future__ import annotations

import argparse
import csv
import io
import math
import zipfile
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from scipy.stats import beta

matplotlib.use("Agg")


FIGURE_STYLE = {
    "font.family": "Times New Roman",
    "font.size": 14,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 11,
    "axes.linewidth": 1.2,
    "lines.linewidth": 2.4,
    "lines.markersize": 7.5,
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold",
    "mathtext.bfit": "Times New Roman:italic:bold",
    "mathtext.cal": "Times New Roman",
    "mathtext.sf": "Times New Roman",
    "mathtext.tt": "Times New Roman",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.formatter.use_mathtext": True,
}
CURVE_COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#B57900", "#56B4E9", "#332288")
CURVE_MARKERS = ("o", "s", "^", "D", "v", "P", "X")
CURVE_LINES = ("-", "--", "-.", ":", (0, (5, 1, 1, 1, 1, 1)), (0, (7, 2)), (0, (3, 1, 1, 1)))
METHOD_STYLE = {
    "cp": 0,
    "kl": 1,
    "eb": 2,
    "hoeffding": 3,
    "wald": 4,
    "bootstrap": 5,
    "aggregate_cp": 6,
    "joint_split": 1,
    "joint_lr": 2,
    "oracle_eb": 0,
    "oracle_wald": 1,
    "logit_naive": 2,
    "logit_sandwich": 3,
    "main_sandwich": 4,
    "rf_cf_naive": 5,
    "rf_cf_eb": 6,
}


def curve_style(index: int) -> dict:
    """Get the curve style.

    Args:
        index (int): The index.

    Returns:
        dict: The curve style.
    """
    return {
        "color": CURVE_COLORS[index],
        "marker": CURVE_MARKERS[index],
        "linestyle": CURVE_LINES[index],
        "markerfacecolor": "white",
        "markeredgewidth": 1.5,
    }


METHODS = {
    "cp": "Cell CP",
    "kl": "Cell KL",
    "eb": "Empirical Bernstein",
    "hoeffding": "Hoeffding",
    "wald": "Wald",
    "aggregate_cp": "Aggregate CP",
    "joint_split": "Joint split",
    "joint_lr": "Joint LR",
    "bootstrap": "Endpoint bootstrap",
    "endpoint_cp": "Endpoint CP",
    "conditional_cp": "Conditional CP",
    "atomic_kl": "Atomic KL",
    "atomic_kl_aggregate": "Atomic KL + aggregate",
    "oracle_eb": "Oracle EB",
    "oracle_wald": "Oracle Wald",
    "logit_naive": "Logit naive",
    "logit_sandwich": "Logit sandwich",
    "main_sandwich": "Main-effects sandwich",
    "rf_cf_naive": "Forest naive",
    "rf_cf_eb": "Forest EB",
    "cp_eb": "CP + external EB",
    "hybrid_eb_cp": "Hybrid EB--CP",
    "hybrid_hoeffding_cp": "Hybrid Hoeffding--CP",
}
SCIENCE = {
    "range": "Range only",
    "signed": "Signed effects",
    "delta": r"$\delta$-box ($\delta=0.2$)",
    "tan1.25": r"Time-only ($\lambda=1.25$)",
    "tan1.5": r"Time-only ($\lambda=1.5$)",
    "tan2.0": r"Time-only ($\lambda=2$)",
    "budget": r"Signed budget ($\Gamma=0.2$)",
    "calibrated": "Calibrated budget",
}
BINARY = {
    "set_cover",
    "target_cover",
    "primitive_cover",
    "failure",
    "plugin_failure",
    "nuisance_failure",
    "empty_rare",
    "reference_outward_cover",
    "reference_inward_cover",
    "set_cover_outward",
    "set_cover_inward",
    "reference_inward_nonempty",
    "primitive_region_fallback",
    "fallback",
    "exhausted",
    "lower_exhausted",
    "upper_exhausted",
    "lower_fallback",
    "upper_fallback",
    "unavailable_certificate",
    "empty_program",
    "refinement_limit",
    "stopping_limit",
}
JOINT_METHODS = ("cp", "joint_split", "joint_lr", "aggregate_cp")
JOINT_SETTINGS = {"reference": "Reference", "active_caps": "Active caps"}
REPLICATION_SOURCES = {
    "interval": (
        "core/interval_replications.csv",
        ["design", "rescue", "arm", "n", "kind", "method"],
    ),
    "contrast": ("core/contrast_replications.csv", ["rescue", "n", "method"]),
    "rare": ("core/rare_replications.csv", ["expected_count", "method"]),
    "external_radius": (
        "core/external_radius_replications.csv",
        ["experiment", "setting", "kind", "method"],
    ),
    "observational": ("observational/observational_replications.csv", ["scenario", "n", "method"]),
    "continuous_trials": (
        "continuous/continuous_replications.csv",
        ["design", "rescue", "arm", "n", "kind", "method"],
    ),
    "partition": ("partitions/partition_replications.csv", ["cap", "n", "M", "method"]),
    "gamma": ("gamma/gamma_replications.csv", ["n", "gamma", "model_contains_truth", "method"]),
    "joint": ("joint/replications.csv", ["setting", "n", "method"]),
    "continuous_outcome": (
        "continuous_outcome/continuous_outcome_replications.csv",
        ["n", "method"],
    ),
}
REFERENCE_SOURCES = {
    "population": "core/population_regions.csv",
    "relaxation": "core/relaxation_audit.csv",
    "completion": "core/completion_audit.csv",
    "mesh": "core/mesh_audit.csv",
    "curves": "core/sensitivity_curves.csv",
    "continuous": "continuous/continuous_oracles.csv",
    "partition_oracle": "partitions/partition_oracles.csv",
}


TABLE_NAMES = frozenset(
    {
        "scientific",
        "inference",
        "continuous-outcome",
        "completion",
        "mesh",
        "atomic-kl",
        "accuracy",
        "contrast",
        "ps-linear",
        "ps-nonlinear",
        "continuous",
        "partition-oracle",
        "partition",
        "rare-upper",
        "external-precision",
        "radius-grid",
        "joint-regions",
    }
)


def number(value: Any) -> float:
    """Convert a value to a number.

    Args:
        value (Any): The value.

    Returns:
        float: The number.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def parse(value: str) -> Any:
    """Parse a value.

    Args:
        value (str): The value.

    Returns:
        Any: The parsed value.
    """
    if value == "":
        return None
    if value in ("True", "False"):
        return int(value == "True")
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def read_csv(path: Path) -> list[dict]:
    """Read a CSV file.

    Args:
        path (Path): The path.

    Returns:
        list[dict]: The rows.
    """
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as stream:
        return [{k: parse(v) for k, v in row.items()} for row in csv.DictReader(stream)]


def select(rows: Iterable[dict], **criteria: Any) -> list[dict]:
    """Select rows from a list of dictionaries.

    Args:
        rows (Iterable[dict]): The rows.
        criteria (dict): The criteria.

    Returns:
        list[dict]: The selected rows.
    """
    return [r for r in rows if all(r.get(k) == v for k, v in criteria.items())]


def summarize(rows: list[dict], keys: list[str]) -> list[dict]:
    """Summarize the rows.

    Args:
        rows (list[dict]): The rows.
        keys (list[str]): The keys.

    Returns:
        list[dict]: The summaries.
    """
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault(tuple(row.get(k) for k in keys), []).append(row)
    summaries = []
    for key, group in groups.items():
        out = {k: v for k, v in zip(keys, key)}
        out["replications"] = len(group)
        reps = [r.get("rep") for r in group]
        if all(r is not None for r in reps) and len(set(reps)) != len(reps):
            raise ValueError(f"Duplicate replications in group {out}; refusing to count them twice")
        columns = set().union(*(r.keys() for r in group)) - set(keys) - {"rep", "task_id", "seed"}
        for col in sorted(columns):
            vals = np.asarray([number(r.get(col)) for r in group])
            if col in {"set_cover", "target_cover"} and not np.isfinite(vals).all():
                raise ValueError(
                    f"Missing {col} in group {out}; coverage denominators must include every replication",  # noqa: E501
                )
            if col == "primitive_cover":
                vals = np.where(np.isfinite(vals), vals, 0.0)
            finite = vals[np.isfinite(vals)]
            if col in {"iterations", "fallback_count"} or col.endswith(
                ("_gap", "_likelihood_violation", "_iterations")
            ):
                observed = vals[~np.isnan(vals)]
                if len(observed):
                    out[col + "_max"] = float(observed.max())
            if not len(finite):
                continue
            out[col] = float(finite.mean())
            out[col + "_observations"] = len(finite)
            out[col + "_missing"] = len(group) - len(finite)
            out[col + "_mcse"] = (
                float(finite.std(ddof=1) / np.sqrt(len(finite)))
                if len(finite) > 1
                else float("nan")
            )
            if col in BINARY:
                if not np.isin(finite, [0, 1]).all():
                    raise ValueError(f"Non-binary coverage/failure column: {col}")
                k, n = int(finite.sum()), len(finite)
                out[col + "_successes"] = k
                out[col + "_mc_lower"] = 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))
                out[col + "_mc_upper"] = 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))
            if col.endswith("error"):
                out[col + "_rmse"] = float(np.sqrt(np.mean(finite**2)))
            if col == "fallback_count":
                out[col + "_sum"] = float(finite.sum())
        summaries.append(out)
    return sorted(
        summaries,
        key=lambda r: tuple(
            (0, number(r.get(k))) if math.isfinite(number(r.get(k))) else (1, str(r.get(k)))
            for k in keys
        ),
    )


def tex_escape(value: Any) -> str:
    """Escape a value for LaTeX.

    Args:
        value (Any): The value.

    Returns:
        str: The escaped value.
    """
    mapping = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(mapping.get(c, c) for c in str(value))


def fmt(value: Any, *, integer: bool = False, scientific: bool = False) -> str:
    """Use three significant digits, retaining trailing zeros.

    Exact counts remain integers. Small values use a two-decimal mantissa;
    rounding the mantissa first keeps powers of ten normalized at boundaries.
    """
    x = number(value)
    if not math.isfinite(x):
        return "--"
    if integer:
        return f"${round(x)}$"
    if x == 0:
        return "$0.00$"
    mantissa, power = f"{x:.2e}".split("e")
    exponent = int(power)
    if scientific or exponent < -3 or exponent >= 3:
        return rf"${mantissa}\times 10^{{{exponent}}}$"
    return f"${x:.{max(0, 2 - exponent)}f}$"


def sci(value: Any) -> str:
    """Use the same three significant digits in scientific notation.

    Args:
        value (Any): The value.

    Returns:
        str: The formatted value.
    """
    return fmt(value, scientific=True)


def mc_interval(row: dict, metric: str = "set_cover") -> str:
    """Format a Monte Carlo interval.

    Args:
        row (dict): The row.
        metric (str, optional): The metric. Defaults to "set_cover".

    Returns:
        str: The formatted interval.
    """
    lower = fmt(row.get(metric + "_mc_lower")).strip("$")
    upper = fmt(row.get(metric + "_mc_upper")).strip("$")
    return f"$[{lower}, {upper}]$"


class Report:
    """A report.

    Args:
        archive (Path): The archive.
        figures (Path): The figures.
        tables (Path): The tables.
    """

    def __init__(self, archive: Path, figures: Path, tables: Path) -> None:
        """Initialize the report.

        Args:
            archive (Path): The archive.
            figures (Path): The figures.
            tables (Path): The tables.
        """
        self.archive, self.figures, self.tables = archive, figures, tables
        self.bundle = archive / "simulation_data.zip"
        for path in (figures, tables):
            path.mkdir(parents=True, exist_ok=True)
        self.raw: dict[str, list[dict]] = {}
        self.data: dict[str, list[dict]] = {}
        self.artifacts: list[str] = []
        self.missing: list[str] = []
        for name, (source, keys) in REPLICATION_SOURCES.items():
            self.raw[name] = self.load(source)
            self.data[name] = summarize(self.raw[name], keys)
        for name, source in REFERENCE_SOURCES.items():
            self.data[name] = self.load(source)

    def load(self, source: str) -> list[dict]:
        """Load the data from a source.

        Args:
            source (str): The source.

        Returns:
            list[dict]: The data.
        """
        if self.bundle.exists():
            with zipfile.ZipFile(self.bundle) as bundle:
                try:
                    with bundle.open(source) as stream:
                        return [
                            {k: parse(v) for k, v in row.items()}
                            for row in csv.DictReader(io.TextIOWrapper(stream))
                        ]
                except KeyError:
                    self.missing.append(source)
                    return []
        path = self.archive / source
        if not path.exists():
            self.missing.append(source)
            return []
        return read_csv(path)

    def table(self, name: str, headers: list[str], body: list[list[str]]) -> None:
        """Write a manuscript-ready tabular fragment, without a table wrapper.

        Args:
            name (str): The name.
            headers (list[str]): The headers.
            body (list[list[str]]): The body.
        """
        if name not in TABLE_NAMES:
            raise ValueError(f"Un-requested manuscript table: {name}")
        textual = {
            "Scientific class",
            "Inference",
            "Procedure",
            "Outcome law",
            "External sample size",
            "True-law member",
            "External calibration",
            "Input-region construction",
        }
        alignment = "".join("l" if h in textual else "r" for h in headers)
        text = [
            rf"\begin{{tabular}}{{@{{}}{alignment}@{{}}}}",
            r"\toprule",
            " & ".join(headers) + r" \\",
            r"\midrule",
        ]
        if any(len(row) != len(headers) for row in body):
            raise ValueError(f"Inconsistent column count in {name}")
        text.extend(" & ".join(row) + r" \\" for row in body)
        if not body:
            text.append(
                rf"\multicolumn{{{len(headers)}}}{{c}}{{No archived records for this comparison.}} \\"  # noqa: E501
            )
        text.extend([r"\bottomrule", r"\end{tabular}", ""])
        path = self.tables / f"{name}.tex"
        path.write_text("\n".join(text))
        self.artifacts.append(str(path))

    def manuscript_tables(self) -> None:
        """Generate the manuscript tables.

        Args:
            self (Report): The report.
        """
        primary = select(self.data["interval"], design="finite", rescue=0.5, arm=0, n=1000)
        scientific = []
        for kind in SCIENCE:
            method = (
                "endpoint_cp"
                if kind in ("range", "signed")
                else "conditional_cp" if kind.startswith("tan") else "cp"
            )
            scientific.extend(select(primary, kind=kind, method=method))
        self.table(
            "scientific",
            ["Scientific class", "Inference", "Sharp width", "Mean width", "Set coverage"],
            [
                [
                    SCIENCE[r["kind"]],
                    METHODS.get(r["method"], tex_escape(r["method"])),
                    fmt(r.get("sharp_width")),
                    fmt(r.get("length")),
                    fmt(r.get("set_cover")),
                ]
                for r in scientific
            ],
        )
        inference = [
            r
            for m in ("cp", "kl", "eb", "hoeffding", "wald", "aggregate_cp", "bootstrap")
            for r in select(primary, kind="budget", method=m)
        ]
        self.table(
            "inference",
            ["Procedure", "Mean width", "MC SE", "Set coverage", r"$95\%$ MC interval"],
            [
                [
                    METHODS[r["method"]],
                    fmt(r.get("length")),
                    fmt(r.get("length_mcse")),
                    fmt(r.get("set_cover")),
                    mc_interval(r),
                ]
                for r in inference
            ],
        )
        completion = self.data["completion"]
        self.table(
            "completion",
            ["$K$", "$s$", "Completion", "Reduced", "Endpoint difference", "Outward difference"],
            [
                [
                    fmt(r.get("K"), integer=True),
                    fmt(r.get("J", r.get("s")), integer=True),
                    fmt(r.get("completion_variables"), integer=True),
                    fmt(r.get("boundary_variables"), integer=True),
                    sci(r.get("max_endpoint_difference")),
                    sci(r.get("max_outward_difference")),
                ]
                for r in completion
            ],
        )
        self.table(
            "mesh",
            ["$h$", "$h^2/4$", "Endpoint gap", "Inner violation", "Certificate gap"],
            [
                [
                    rf"$1/{round(1 / number(r['mesh']))}$",
                    fmt(r.get("budget_error")),
                    fmt(r.get("bracket_gap")),
                    sci(r.get("inner_violation")),
                    sci(r.get("dual_gap")),
                ]
                for r in self.data["mesh"]
            ],
        )
        accuracy = select(
            self.data["interval"], design="sample_size", kind="calibrated", method="cp"
        )
        self.table(
            "accuracy",
            ["$n$", "Plug-in error", "MC SE", "Outer width", "Enlargement", "Set coverage"],
            [
                [
                    fmt(r.get("n"), integer=True),
                    fmt(r.get("plugin_error")),
                    fmt(r.get("plugin_error_mcse")),
                    fmt(r.get("length")),
                    fmt(r.get("enlargement")),
                    fmt(r.get("set_cover")),
                ]
                for r in accuracy
            ],
        )
        contrasts = self.data["contrast"]
        self.table(
            "contrast",
            ["Rescue", "Procedure", "Sharp width", "Mean width", "MC SE", "Set coverage"],
            [
                [
                    fmt(r.get("rescue")),
                    METHODS.get(r.get("method"), tex_escape(r.get("method"))),
                    fmt(r.get("sharp_width")),
                    fmt(r.get("length")),
                    fmt(r.get("length_mcse")),
                    fmt(r.get("set_cover")),
                ]
                for r in contrasts
            ],
        )
        for scenario in ("linear", "nonlinear"):
            obs = [
                r
                for n in (50, 1000, 10000)
                for method in (
                    "oracle_eb",
                    "oracle_wald",
                    "logit_naive",
                    "logit_sandwich",
                    "main_sandwich",
                    "rf_cf_naive",
                    "rf_cf_eb",
                )
                for r in select(self.data["observational"], scenario=scenario, n=n, method=method)
            ]
            self.table(
                f"ps-{scenario}",
                [
                    "$N$",
                    "Procedure",
                    "Width",
                    "MC SE",
                    "Set cov.",
                    "Input cov.",
                    r"Mean $\mu$ error",
                ],
                [
                    [
                        fmt(r.get("n"), integer=True),
                        METHODS.get(r.get("method"), tex_escape(r.get("method"))),
                        fmt(r.get("length")),
                        fmt(r.get("length_mcse")),
                        fmt(r.get("set_cover")),
                        fmt(r.get("primitive_cover")),
                        fmt(r.get("mu_error")),
                    ]
                    for r in obs
                ],
            )
        cb = []
        for oracle in self.data["continuous"]:
            K, d, rho = oracle.get("K"), oracle.get("d"), oracle.get("rho")
            design = f"continuous_K{int(K)}_d{int(d)}_rho{float(rho)}"
            selected = select(self.data["continuous_trials"], design=design, method="cp")
            r = selected[0] if selected else {}
            cb.append(
                [
                    fmt(K, integer=True),
                    fmt(d, integer=True),
                    fmt(rho),
                    fmt(oracle.get("sharp_width")),
                    fmt(r.get("length")),
                    f"{fmt(oracle.get('theta'))} ({fmt(oracle.get('theta_mcse'))})",
                ]
            )
        self.table(
            "continuous",
            [
                "$K$",
                "$d_W$",
                r"$\rho$",
                "Sharp width",
                "Cell CP width",
                r"$\theta_0$ (reference SE)",
            ],
            cb,
        )
        po = self.data["partition_oracle"]
        self.table(
            "partition-oracle",
            ["Outcome law", "$M$", "Sharp width", "True norm", "Lower MC SE", "Upper MC SE"],
            [
                [
                    "Heterogeneous caps" if r.get("cap") else "Original",
                    fmt(r.get("M"), integer=True),
                    fmt(r.get("sharp_width")),
                    fmt(r.get("gamma_true")),
                    fmt(r.get("lower_mcse")),
                    fmt(r.get("upper_mcse")),
                ]
                for r in po
            ],
        )
        partition = select(self.data["partition"], cap=1)
        self.table(
            "partition",
            [
                "$n$",
                "$M$",
                "Sharp width",
                "Outer width",
                "Enlargement",
                "Plug-in error",
                "Empty frac.",
                "Coverage",
            ],
            [
                [
                    fmt(r.get("n"), integer=True),
                    fmt(r.get("M"), integer=True),
                    fmt(r.get("sharp_width")),
                    fmt(r.get("length")),
                    fmt(r.get("enlargement")),
                    fmt(r.get("plugin_error")),
                    fmt(r.get("empty_fraction")),
                    fmt(r.get("set_cover")),
                ]
                for r in partition
            ],
        )
        rare, body = self.data["rare"], []
        for expected in sorted({r["expected_count"] for r in rare}):
            by_method = {r["method"]: r for r in select(rare, expected_count=expected)}
            body.append(
                [fmt(expected)]
                + [
                    fmt(by_method.get(m, {}).get("upper_mass"))
                    for m in ("cp", "kl", "eb", "hoeffding", "wald", "bootstrap")
                ]
            )
        self.table(
            "rare-upper",
            ["$np$", "CP", "KL", "Bernstein", "Hoeffding", "Wald", "Bootstrap"],
            body,
        )
        gamma = self.data["gamma"]
        self.table(
            "radius-grid",
            [
                "$n$",
                r"$\Gamma$",
                "True-law member",
                "Sharp width",
                "Outer width",
                "Set cov.",
                "Target cov.",
            ],
            [
                [
                    fmt(r.get("n"), integer=True),
                    fmt(r.get("gamma")),
                    "Yes" if r.get("model_contains_truth") else "No",
                    fmt(r.get("sharp_width")),
                    fmt(r.get("length")),
                    fmt(r.get("set_cover")),
                    fmt(r.get("target_cover")),
                ]
                for r in gamma
            ],
        )
        external = sorted(
            select(self.data["external_radius"], experiment="external"),
            key=lambda row: float("inf") if row["setting"] == -1 else row["setting"],
        )
        self.table(
            "external-precision",
            ["External calibration", "Mean width", "Set coverage"],
            [
                [
                    (
                        "Exact"
                        if r["setting"] == -1
                        else "None" if r["setting"] == 0 else fmt(r["setting"], integer=True)
                    ),
                    fmt(r.get("length")),
                    fmt(r.get("set_cover")),
                ]
                for r in external
            ],
        )
        atomic = [
            row
            for method in ("atomic_kl", "atomic_kl_aggregate", "kl", "cp")
            for row in select(primary, kind="budget", method=method)
        ]
        self.table(
            "atomic-kl",
            ["Input-region construction", "Mean width", "Set coverage"],
            [[METHODS[r["method"]], fmt(r.get("length")), fmt(r.get("set_cover"))] for r in atomic],
        )
        continuous_outcome = [
            row
            for n in (250, 1000, 10000)
            for method in ("hybrid_eb_cp", "hybrid_hoeffding_cp", "wald", "bootstrap")
            for row in select(self.data["continuous_outcome"], n=n, method=method)
        ]
        self.table(
            "continuous-outcome",
            ["$n$", "Procedure", "Mean width", "MC SE", "Set coverage", r"$95\%$ MC interval"],
            [
                [
                    fmt(r["n"], integer=True),
                    "Cell Wald" if r["method"] == "wald" else METHODS[r["method"]],
                    fmt(r.get("length")),
                    fmt(r.get("length_mcse")),
                    fmt(r.get("set_cover")),
                    mc_interval(r),
                ]
                for r in continuous_outcome
            ],
        )
        self.joint_table()

    def joint_groups(self) -> list[tuple[str, int, list[dict]]]:
        """Return the archived joint comparisons in benchmark setting order.

        Returns:
            list[tuple[str, int, list[dict]]]: The joint groups.
        """
        groups: dict[tuple, list[dict]] = {}
        for row in self.data["joint"]:
            groups.setdefault((row["setting"], row["n"]), []).append(row)

        def order(item: tuple[tuple[str, int], list[dict]]) -> tuple[int, int, str]:
            """Order the joint groups.

            Args:
                item (tuple[tuple[str, int], list[dict]]): The item.

            Returns:
                tuple[int, int, str]: The ordered item.
            """
            (setting, n), rows = item
            index = number(rows[0].get("setting_index"))
            if not math.isfinite(index):
                index = list(JOINT_SETTINGS).index(setting) if setting in JOINT_SETTINGS else 2
            return index, n, setting

        return [(setting, n, rows) for (setting, n), rows in sorted(groups.items(), key=order)]

    def joint_table(self) -> None:
        """Write the paired joint-region comparison using observed Monte Carlo summaries.

        Args:
            self (Report): The report.
        """
        lines = [
            r"\begin{tabular}{@{}lrrrrrr@{}}",
            r"\toprule",
            (
                r"Procedure & \shortstack{Mean\\width} & MC SE & Difference & "
                r"\shortstack{Paired\\MC SE} & \shortstack{Set\\coverage} & $95\%$ MC interval \\"
            ),
            r"\midrule",
        ]
        groups = self.joint_groups()
        for i, (setting, n, rows) in enumerate(groups):
            if i:
                lines.append(r"\midrule")
            label = tex_escape(JOINT_SETTINGS.get(setting, setting))
            lines.append(rf"\multicolumn{{7}}{{l}}{{{label}, $n={n}$}} \\")
            for method in JOINT_METHODS:
                for row in select(rows, method=method):
                    cells = [
                        METHODS[method],
                        fmt(row.get("length")),
                        fmt(row.get("length_mcse")),
                        fmt(row.get("width_difference_cp")),
                        fmt(row.get("width_difference_cp_mcse")),
                        fmt(row.get("set_cover")),
                        mc_interval(row),
                    ]
                    lines.append(" & ".join(cells) + r" \\")
        if not groups:
            lines.append(r"\multicolumn{7}{c}{No archived records for this comparison.} \\")
        lines.extend([r"\bottomrule", r"\end{tabular}", ""])
        path = self.tables / "joint-regions.tex"
        path.write_text("\n".join(lines))
        self.artifacts.append(str(path))

    def figure(
        self, name: str, title: str, plot: Callable, rows: list[dict], *, tight_bounds: bool = False
    ) -> None:
        """Generate a figure.

        Args:
            name (str): The name.
            title (str): The title.
            plot (Callable): The plot.
            rows (list[dict]): The rows.
            tight_bounds (bool): Include all labels in the saved figure bounds.
        """
        font_manager.findfont("Times New Roman", fallback_to_default=False)
        with plt.rc_context(FIGURE_STYLE):
            fig, ax = plt.subplots(figsize=(5.8, 5.4), layout="constrained")
            if rows:
                plot(ax)
                handles, _ = ax.get_legend_handles_labels()
                if handles:
                    ax.legend(
                        loc="upper center",
                        bbox_to_anchor=(0.5, -0.25),
                        ncols=2,
                        frameon=False,
                        handlelength=3.0,
                        columnspacing=1.2,
                        labelspacing=0.6,
                    )
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No archived records for this comparison",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=13,
                )
            ax.spines[["right", "top"]].set_visible(False)
            ax.grid(alpha=0.2)
            path = self.figures / f"{name}.pdf"
            fig.savefig(
                path,
                bbox_inches="tight" if tight_bounds else None,
                metadata={
                    "Title": title,
                    "Creator": "report_experiments.py",
                    "CreationDate": None,
                    "ModDate": None,
                },
            )
            plt.close(fig)
        self.artifacts.append(str(path))

    @staticmethod
    def error_curve(
        ax,
        rows: list[dict],
        x: str,
        metric: str,
        label: str,
        offset: float = 0,
        categorical: list | None = None,
        coverage: bool = False,
        **kwargs,
    ) -> None:
        """Plot an error curve.

        Args:
            ax (Axes): The axes.
            rows (list[dict]): The rows.
            x (str): The x-axis.
            metric (str): The metric.
            label (str): The label.
            offset (float, optional): The offset. Defaults to 0.
            categorical (list | None, optional): The categorical. Defaults to None.
            coverage (bool, optional): Whether to plot coverage. Defaults to False.
            **kwargs: Additional arguments.
        """
        rows = sorted(rows, key=lambda r: number(r[x]))
        if not rows:
            return
        xs = (
            np.asarray(
                [
                    categorical.index(r[x]) if categorical is not None else number(r[x])
                    for r in rows
                ],
                dtype=float,
            )
            + offset
        )
        ys = np.asarray([number(r.get(metric)) for r in rows])
        if coverage:
            lows = np.asarray([number(r.get(metric + "_mc_lower")) for r in rows])
            highs = np.asarray([number(r.get(metric + "_mc_upper")) for r in rows])
            err = np.maximum(0, np.stack([ys - lows, highs - ys]))
        else:
            se = np.asarray([number(r.get(metric + "_mcse")) for r in rows])
            err = 1.96 * se
            if not np.isfinite(err).any():
                err = None
        ax.errorbar(
            xs, ys, yerr=err, label=label, capsize=4, elinewidth=1.8, capthick=1.8, **kwargs
        )

    def manuscript_figures(self) -> None:
        """Generate the manuscript figures.

        Args:
            self (Report): The report.
        """
        rare = self.data["rare"]

        def rare_plot(ax):
            """Plot the rare crossing strata.

            Args:
                ax (Axes): The axes.
            """
            xs = sorted({r["expected_count"] for r in rare})
            methods = ("cp", "kl", "eb", "hoeffding", "wald", "bootstrap")
            for i, method in enumerate(methods):
                self.error_curve(
                    ax,
                    select(rare, method=method),
                    "expected_count",
                    "set_cover",
                    METHODS[method],
                    offset=(i - 2.5) * 0.07,
                    categorical=xs,
                    coverage=True,
                    **curve_style(METHOD_STYLE[method]),
                )
            ax.set(
                xticks=range(len(xs)),
                xticklabels=[str(x) for x in xs],
                xlabel=r"Expected crossing count ($np$)",
                ylabel="Whole-set coverage (95% MC interval)",
                ylim=(-0.02, 1.035),
            )
            ax.axhline(0.95, ls=(0, (2, 2, 6, 2)), color="black", lw=1.8)

        self.figure("rare_coverage", "Rare crossing strata", rare_plot, rare)
        caps = select(self.data["interval"], design="active_caps", kind="budget")

        def caps_plot(ax):
            """Plot the active outcome caps.

            Args:
                ax (Axes): The axes.
            """
            for method in ("cp", "aggregate_cp", "bootstrap"):
                self.error_curve(
                    ax,
                    select(caps, method=method),
                    "n",
                    "length",
                    METHODS[method],
                    **curve_style(METHOD_STYLE[method]),
                )
            truth = select(self.data["population"], design="active_caps", kind="budget")
            if truth:
                ax.axhline(
                    number(truth[0].get("sharp_width")),
                    color="black",
                    ls="--",
                    label="Population sharp width",
                )
                aggregate = number(
                    truth[0].get("aggregate_width", truth[0].get("aggregate_relaxation_width"))
                )
                if not math.isfinite(aggregate):
                    # This uncapped primary law has the same rescue probabilities.
                    equivalent = select(
                        self.data["population"], design="finite", rescue=0.5, arm=0, kind="budget"
                    )
                    aggregate = (
                        number(equivalent[0].get("sharp_width")) if equivalent else float("nan")
                    )
                if math.isfinite(aggregate):
                    ax.axhline(
                        aggregate, color="gray", ls=":", label="Aggregate population relaxation"
                    )
            ax.margins(y=0.08)
            ax.set(
                xscale="log", xlabel="Sample size", ylabel=r"Mean interval width ($\pm1.96$ MC SE)"
            )

        self.figure("active_caps", "Active outcome caps", caps_plot, caps)
        for cap, filename in ((0, "partition_inactive"), (1, "partition_caps")):
            rows = select(self.data["partition"], cap=cap)

            def partition_plot(ax, rows=rows, cap=cap):
                """Plot the partition sensitivity.

                Args:
                    ax (Axes): The axes.
                    rows (list[dict], optional): The rows. Defaults to rows.
                    cap (int, optional): The cap. Defaults to cap.
                """
                xs = sorted({r["M"] for r in rows})
                for i, n in enumerate(sorted({r["n"] for r in rows})):
                    self.error_curve(
                        ax,
                        select(rows, n=n),
                        "M",
                        "length",
                        rf"$n = {n}$",
                        categorical=xs,
                        **curve_style((0, 2, 6)[i]),
                    )
                ref = sorted(select(self.data["partition_oracle"], cap=cap), key=lambda r: r["M"])
                if ref:
                    ax.plot(
                        [xs.index(r["M"]) for r in ref if r["M"] in xs],
                        [r["sharp_width"] for r in ref if r["M"] in xs],
                        color="black",
                        linestyle="--",
                        marker="v",
                        markerfacecolor="white",
                        markeredgewidth=1.5,
                        label="Population reference width",
                    )
                ax.set(
                    xticks=range(len(xs)),
                    xticklabels=xs,
                    xlabel=r"Prespecified crossing cells ($M$)",
                    ylabel=r"Mean interval width ($\pm1.96$ MC SE)",
                )

            self.figure(
                filename,
                "Partition sensitivity: " + ("active caps" if cap else "original law"),
                partition_plot,
                rows,
            )
        for scenario in ("linear", "nonlinear"):
            rows = select(self.data["observational"], scenario=scenario)

            def ps_plot(ax, rows=rows):
                """Plot the propensity sensitivity.

                Args:
                    ax (Axes): The axes.
                    rows (list[dict], optional): The rows. Defaults to rows.
                """
                xs = sorted({r["n"] for r in rows})
                methods = (
                    "oracle_eb",
                    "oracle_wald",
                    "logit_naive",
                    "logit_sandwich",
                    "main_sandwich",
                    "rf_cf_naive",
                    "rf_cf_eb",
                )
                for i, method in enumerate(methods):
                    self.error_curve(
                        ax,
                        select(rows, method=method),
                        "n",
                        "set_cover",
                        METHODS[method],
                        categorical=xs,
                        offset=(i - 3) * 0.065,
                        coverage=True,
                        **curve_style(METHOD_STYLE[method]),
                    )
                ax.set(
                    xticks=range(len(xs)),
                    xticklabels=[f"{n:,}" for n in xs],
                    xlabel="Sample size",
                    ylabel="Whole-set coverage (95% MC interval)",
                    ylim=(-0.025, 1.035),
                )
                ax.axhline(0.95, ls=(0, (2, 2, 6, 2)), color="black", lw=1.8)

            self.figure(
                f"propensity_{scenario}",
                f"{scenario.capitalize()} treatment assignment",
                ps_plot,
                rows,
            )
        primary = select(self.data["population"], design="finite", rescue=0.5, arm=0, kind="budget")
        for model in ("budget", "tan"):
            rows = select(self.data["curves"], model=model)

            def sensitivity_plot(ax, rows=rows, model=model):
                """Plot the sensitivity to the parameter.

                Args:
                    ax (Axes): The axes.
                    rows (list[dict], optional): The rows. Defaults to rows.
                    model (str, optional): The model. Defaults to model.
                """
                rows = sorted(rows, key=lambda r: r["parameter"])
                x, lo, hi = ([r[key] for r in rows] for key in ("parameter", "lower", "upper"))
                ax.fill_between(x, lo, hi, color="#3178a8", alpha=0.16)
                ax.plot(
                    x,
                    lo,
                    label="Sharp lower endpoint",
                    markevery=max(1, len(x) // 8),
                    **curve_style(0),
                )
                ax.plot(
                    x,
                    hi,
                    label="Sharp upper endpoint",
                    markevery=max(1, len(x) // 8),
                    **curve_style(1),
                )
                if primary:
                    p = primary[0]
                    ax.axhline(
                        number(p.get("theta")), color="black", ls=":", label="Actual no-rescue mean"
                    )
                    threshold = number(
                        p.get("gamma_true" if model == "budget" else "minimum_true_lambda")
                    )
                    if math.isfinite(threshold):
                        ax.axvline(
                            threshold, color="gray", ls="-.", label="Oracle membership threshold"
                        )
                ax.set(
                    xlabel=(
                        r"Budget radius, $\Gamma$"
                        if model == "budget"
                        else r"Time-only ratio bound, $\lambda$"
                    ),
                    ylabel="Population sharp endpoints",
                )

            self.figure(
                f"sensitivity_{model}",
                "Signed-budget model" if model == "budget" else "Time-only density-ratio model",
                sensitivity_plot,
                rows,
            )

    def finish(self) -> dict:
        """Finish the report.

        Returns:
            dict: The result.
        """
        for name in (
            "all_tables.tex",
            "external-radius.tex",
            "reference-precision.tex",
            "reduction-checks.tex",
        ):
            (self.tables / name).unlink(missing_ok=True)
        (self.figures / "joint_widths.pdf").unlink(missing_ok=True)
        return {"artifacts": self.artifacts, "missing_sources": self.missing}


def generate_reports(
    archive: Path, figures: Path | None = None, tables: Path | None = None
) -> dict:
    """Generate the reports.

    Args:
        archive (Path): The archive.
        figures (Path | None, optional): The figures. Defaults to None.
        tables (Path | None, optional): The tables. Defaults to None.

    Returns:
        dict: The result.
    """
    archive = Path(archive)
    if not archive.is_dir():
        raise FileNotFoundError(f"No experiment archive found: {archive}")
    report = Report(
        archive,
        Path(figures) if figures is not None else archive / "figures",
        Path(tables) if tables is not None else archive / "tables",
    )
    joint_only = (
        bool(report.raw["joint"])
        and not any(rows for name, rows in report.raw.items() if name != "joint")
        and not any(report.data[name] for name in REFERENCE_SOURCES)
    )
    if joint_only:
        report.joint_table()
        result = report.finish()
        result["missing_sources"] = []
        return result
    report.manuscript_tables()
    report.manuscript_figures()
    return report.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", "--out", dest="archive", type=Path, default=Path("results"))
    parser.add_argument("--figures", type=Path, help="Figure directory (default: ARCHIVE/figures)")
    parser.add_argument("--tables", type=Path, help="Table directory (default: ARCHIVE/tables)")
    args = parser.parse_args()
    result = generate_reports(args.archive, args.figures, args.tables)
    print(f"Rebuilt {len(result['artifacts'])} manuscript artifacts from {args.archive}.")
    if result["missing_sources"]:
        print("Missing result files: " + ", ".join(result["missing_sources"]))


if __name__ == "__main__":
    main()
