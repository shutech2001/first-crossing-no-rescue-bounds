from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Callable, Iterable

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
    "oracle_eb": 0,
    "oracle_wald": 1,
    "logit_naive": 2,
    "logit_sandwich": 3,
    "main_sandwich": 4,
    "rf_cf_naive": 5,
    "rf_cf_eb": 6,
}


def curve_style(index: int) -> dict:
    return dict(
        color=CURVE_COLORS[index],
        marker=CURVE_MARKERS[index],
        linestyle=CURVE_LINES[index],
        markerfacecolor="white",
        markeredgewidth=1.5,
    )


METHODS = {
    "cp": "Cell CP",
    "kl": "Cell KL",
    "eb": "Empirical Bernstein",
    "hoeffding": "Hoeffding",
    "wald": "Wald",
    "aggregate_cp": "Aggregate CP",
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
}
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
}
REFERENCE_SOURCES = {
    "population": "core/population_regions.csv",
    "geometry": "core/geometry.csv",
    "completion": "core/completion_audit.csv",
    "mesh": "core/mesh_audit.csv",
    "tan": "core/tan_audit.csv",
    "curves": "core/sensitivity_curves.csv",
    "continuous": "continuous/continuous_oracles.csv",
    "continuous_batches": "continuous/continuous_oracle_batches.csv",
    "partition_oracle": "partitions/partition_oracles.csv",
    "partition_batches": "partitions/partition_oracle_batches.csv",
    "observational_population": "observational/observational_population.csv",
}


TABLE_NAMES = frozenset(
    {
        "scientific",
        "inference",
        "reduction-checks",
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
    }
)


def number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def parse(value: str) -> Any:
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
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as stream:
        return [{k: parse(v) for k, v in row.items()} for row in csv.DictReader(stream)]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def select(rows: Iterable[dict], **criteria: Any) -> list[dict]:
    return [r for r in rows if all(r.get(k) == v for k, v in criteria.items())]


def summarize(rows: list[dict], keys: list[str]) -> list[dict]:
    """Keep all generated replications; report missing diagnostic denominators.

    Coverage of unavailable primitive regions is counted as non-coverage. Other
    undefined diagnostics remain missing and have an explicit observation count.
    With one replication its standard error is undefined, never reported as zero.
    """
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault(tuple(row.get(k) for k in keys), []).append(row)
    summaries = []
    for key, group in groups.items():
        out = dict(zip(keys, key))
        out["replications"] = len(group)
        reps = [r.get("rep") for r in group]
        if all(r is not None for r in reps) and len(set(reps)) != len(reps):
            raise ValueError(f"Duplicate replications in group {out}; refusing to count them twice")
        columns = set().union(*(r.keys() for r in group)) - set(keys) - {"rep", "task_id", "seed"}
        for col in sorted(columns):
            vals = np.asarray([number(r.get(col)) for r in group])
            if col in {"set_cover", "target_cover"} and not np.isfinite(vals).all():
                raise ValueError(
                    f"Missing {col} in group {out}; coverage denominators must include every replication"
                )
            if col == "primitive_cover":
                vals = np.where(np.isfinite(vals), vals, 0.0)
            finite = vals[np.isfinite(vals)]
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
        summaries.append(out)
    return sorted(
        summaries,
        key=lambda r: tuple(
            (0, number(r.get(k))) if math.isfinite(number(r.get(k))) else (1, str(r.get(k)))
            for k in keys
        ),
    )


def tex_escape(value: Any) -> str:
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


def fmt(value: Any, digits: int = 4) -> str:
    x = number(value)
    if not math.isfinite(x):
        return "--"
    if digits == 0:
        return f"${int(round(x))}$"
    return f"${x:.{digits}f}$"


def sci(value: Any) -> str:
    x = number(value)
    if not math.isfinite(x):
        return "--"
    if x == 0:
        return "$0$"
    exponent = math.floor(math.log10(abs(x)))
    return rf"${x / 10 ** exponent:.2f}\times 10^{{{exponent}}}$"


def mc_interval(row: dict, metric: str = "set_cover") -> str:
    lower = fmt(row.get(metric + "_mc_lower"), 3).strip("$")
    upper = fmt(row.get(metric + "_mc_upper"), 3).strip("$")
    return f"$[{lower}, {upper}]$"


class Report:
    def __init__(self, archive: Path, figures: Path, tables: Path):
        self.archive, self.figures, self.tables = archive, figures, tables
        self.summary_dir = archive / "summaries"
        for path in (figures, tables, self.summary_dir):
            path.mkdir(parents=True, exist_ok=True)
        self.raw: dict[str, list[dict]] = {}
        self.data: dict[str, list[dict]] = {}
        self.artifacts: list[str] = []
        self.missing: list[str] = []
        for name, (source, keys) in REPLICATION_SOURCES.items():
            self.raw[name] = self.load(source)
            self.data[name] = summarize(self.raw[name], keys)
            summary_name = "continuous" if name == "continuous_trials" else name
            write_csv(self.summary_dir / f"{summary_name}_summary.csv", self.data[name])
        for name, source in REFERENCE_SOURCES.items():
            self.data[name] = self.load(source)
        self.add_reference_diagnostics()
        self.add_paired_differences()
        self.add_precision_diagnostics()

    def load(self, source: str) -> list[dict]:
        path = self.archive / source
        if not path.exists():
            self.missing.append(source)
            return []
        return read_csv(path)

    def table(self, name: str, headers: list[str], body: list[list[str]]) -> None:
        """Write a manuscript-ready tabular fragment, without a table wrapper."""
        if name not in TABLE_NAMES:
            raise ValueError(f"Un-requested manuscript table: {name}")
        textual = {
            "Scientific class",
            "Inference",
            "Procedure",
            "Outcome law",
            "External sample size",
            "True-law member",
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
                rf"\multicolumn{{{len(headers)}}}{{c}}{{No archived records for this comparison.}} \\"
            )
        text.extend([r"\bottomrule", r"\end{tabular}", ""])
        path = self.tables / f"{name}.tex"
        path.write_text("\n".join(text))
        self.artifacts.append(str(path))

    def add_reference_diagnostics(self) -> None:
        rows = []
        for row in self.raw["partition"]:
            ref = select(self.data["partition_oracle"], cap=row.get("cap"), M=row.get("M"))
            if not ref:
                continue
            p = ref[0]
            lo, hi = number(row.get("lower")), number(row.get("upper"))
            pl, pu = number(p.get("lower")), number(p.get("upper"))
            sl, su = number(p.get("lower_mcse")), number(p.get("upper_mcse"))
            if not np.isfinite([lo, hi, pl, pu, sl, su]).all():
                continue
            rows.append(
                {
                    **row,
                    "reference_outward_cover": int(
                        lo <= max(0, pl - 2.58 * sl) and hi >= min(1, pu + 2.58 * su)
                    ),
                    "reference_inward_cover": int(lo <= pl + 2.58 * sl and hi >= pu - 2.58 * su),
                }
            )
        self.data["partition_reference"] = summarize(rows, ["cap", "n", "M", "method"])
        write_csv(self.summary_dir / "partition_reference_replications.csv", rows)
        write_csv(
            self.summary_dir / "partition_reference_summary.csv", self.data["partition_reference"]
        )
        for source, keys in [
            ("continuous_batches", ["K", "d", "rho"]),
            ("partition_batches", ["cap", "M"]),
        ]:
            batches = [
                {**r, "width": number(r.get("upper")) - number(r.get("lower"))}
                for r in self.data[source]
            ]
            sums = summarize(batches, keys)
            for row in sums:
                row["oracle_batches"] = row.pop("replications")
            write_csv(self.summary_dir / f"{source}_summary.csv", sums)

    def add_paired_differences(self) -> None:
        rows = []
        for source, axes, varying, baseline in [
            ("interval", ["design", "rescue", "arm", "n", "rep", "kind"], "method", "cp"),
            ("continuous_trials", ["design", "rescue", "arm", "n", "rep", "kind"], "method", "cp"),
            ("observational", ["scenario", "n", "rep"], "method", "oracle_eb"),
            ("partition", ["cap", "n", "rep", "method"], "M", 80),
        ]:
            groups: dict[tuple, list[dict]] = {}
            for row in self.raw[source]:
                groups.setdefault(tuple(row.get(k) for k in axes), []).append(row)
            for key, group in groups.items():
                ref = next((r for r in group if r.get(varying) == baseline), None)
                if ref is None:
                    continue
                for row in group:
                    if row.get(varying) == baseline:
                        continue
                    out = {
                        "suite": source,
                        **dict(zip(axes, key)),
                        "comparison": str(row.get(varying)),
                        "reference": str(baseline),
                    }
                    for metric in (
                        "length",
                        "set_cover",
                        "plugin_error",
                        "enlargement",
                        "mu_error",
                    ):
                        a, b = number(row.get(metric)), number(ref.get(metric))
                        if math.isfinite(a) and math.isfinite(b):
                            out[metric + "_difference"] = a - b
                    rows.append(out)
        write_csv(self.summary_dir / "paired_differences.csv", rows)
        keys = [
            "suite",
            "design",
            "rescue",
            "arm",
            "scenario",
            "n",
            "kind",
            "cap",
            "method",
            "comparison",
            "reference",
        ]
        self.data["paired"] = summarize(rows, keys)
        write_csv(self.summary_dir / "paired_differences_summary.csv", self.data["paired"])

    def add_precision_diagnostics(self) -> None:
        """Compare reference Monte Carlo error with trial endpoint RMSE.

        A ratio above 0.1 is a transparent diagnostic threshold, not a theorem or
        a certified statement about the accuracy of a population reference.
        """
        rows = []
        for suite, summaries in (
            ("continuous", self.data["continuous_trials"]),
            ("partition", self.data["partition"]),
        ):
            for row in summaries:
                if suite == "continuous":
                    matches = [
                        p
                        for p in self.data["continuous"]
                        if row.get("design")
                        == f"continuous_K{int(p['K'])}_d{int(p['d'])}_rho{float(p['rho'])}"
                    ]
                    keys = ("design", "n", "method")
                else:
                    matches = select(
                        self.data["partition_oracle"], cap=row.get("cap"), M=row.get("M")
                    )
                    keys = ("cap", "M", "n", "method")
                if not matches:
                    continue
                ref = matches[0]
                out = {
                    "suite": suite,
                    **{k: row.get(k) for k in keys},
                    "replications": row["replications"],
                    "oracle_n": ref.get("oracle_n"),
                    "heuristic_ratio_threshold": 0.1,
                }
                ratios = []
                for endpoint in ("lower", "upper"):
                    se = number(ref.get(endpoint + "_mcse"))
                    rmse = number(row.get(endpoint + "_error_rmse"))
                    ratio = (
                        se / rmse
                        if rmse > 0
                        else float("inf") if rmse == 0 and se > 0 else float("nan")
                    )
                    out[endpoint + "_oracle_mcse"] = se
                    out[endpoint + "_trial_rmse"] = rmse
                    out[endpoint + "_ratio"] = ratio
                    if not math.isnan(ratio):
                        ratios.append(ratio)
                out["nonnegligible_reference_error"] = (
                    int(any(r > 0.1 for r in ratios)) if ratios else None
                )
                for name in ("theta_mcse", "psi_mcse", "supported_gap", "supported_gap_mcse"):
                    if name in ref:
                        out[name] = ref[name]
                rows.append(out)
        self.data["reference_precision"] = rows
        write_csv(self.summary_dir / "reference_precision.csv", rows)

    def manuscript_tables(self) -> None:
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
                    fmt(r.get("set_cover"), 3),
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
                    fmt(r.get("length_mcse"), 5),
                    fmt(r.get("set_cover"), 3),
                    mc_interval(r),
                ]
                for r in inference
            ],
        )
        self.table(
            "reduction-checks",
            [
                "$K$",
                "$s$",
                "Continuation",
                "Boundary",
                "Operator residual",
                "Loading residual",
                "Half-width difference",
                "Endpoint difference",
            ],
            [
                [
                    fmt(r.get("K"), 0),
                    fmt(r.get("J"), 0),
                    fmt(r.get("continuation_dim"), 0),
                    fmt(r.get("bridge_dim"), 0),
                    sci(r.get("operator_residual")),
                    sci(r.get("loading_residual")),
                    sci(r.get("halfwidth_difference")),
                    sci(r.get("constrained_endpoint_difference")),
                ]
                for r in self.data["geometry"]
            ],
        )
        completion = self.data["completion"]
        self.table(
            "completion",
            ["$K$", "$s$", "Completion", "Boundary", "Endpoint difference", "Outward difference"],
            [
                [
                    fmt(r.get("K"), 0),
                    fmt(r.get("J", r.get("s")), 0),
                    fmt(r.get("completion_variables"), 0),
                    fmt(r.get("boundary_variables"), 0),
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
                    fmt(r.get("budget_error"), 7),
                    fmt(r.get("bracket_gap"), 7),
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
                    fmt(r.get("n"), 0),
                    fmt(r.get("plugin_error")),
                    fmt(r.get("plugin_error_mcse"), 5),
                    fmt(r.get("length")),
                    fmt(r.get("enlargement")),
                    fmt(r.get("set_cover"), 3),
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
                    fmt(r.get("rescue"), 2),
                    METHODS.get(r.get("method"), tex_escape(r.get("method"))),
                    fmt(r.get("sharp_width")),
                    fmt(r.get("length")),
                    fmt(r.get("length_mcse"), 5),
                    fmt(r.get("set_cover"), 3),
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
                        fmt(r.get("n"), 0),
                        METHODS.get(r.get("method"), tex_escape(r.get("method"))),
                        fmt(r.get("length")),
                        fmt(r.get("length_mcse")),
                        fmt(r.get("set_cover"), 3),
                        fmt(r.get("primitive_cover"), 3),
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
                    fmt(K, 0),
                    fmt(d, 0),
                    fmt(rho, 2),
                    fmt(oracle.get("sharp_width")),
                    fmt(r.get("length")),
                    f"{fmt(oracle.get('theta'), 5)} ({fmt(oracle.get('theta_mcse'), 5)})",
                    f"{fmt(oracle.get('psi'), 5)} ({fmt(oracle.get('psi_mcse'), 5)})",
                ]
            )
        self.table(
            "continuous",
            [
                "$K$",
                "$d_W$",
                r"$\rho$",
                "Sharp width",
                "CP width",
                r"$\theta_0$ (oracle SE)",
                r"$\psi_0^{\mathrm{loc}}$ (oracle SE)",
            ],
            cb,
        )
        po = self.data["partition_oracle"]
        self.table(
            "partition-oracle",
            ["Outcome law", "$M$", "Sharp width", "True norm", "Lower MC SE", "Upper MC SE"],
            [
                [
                    "Active caps" if r.get("cap") else "Original",
                    fmt(r.get("M"), 0),
                    fmt(r.get("sharp_width"), 6),
                    fmt(r.get("gamma_true"), 6),
                    fmt(r.get("lower_mcse"), 6),
                    fmt(r.get("upper_mcse"), 6),
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
                    fmt(r.get("n"), 0),
                    fmt(r.get("M"), 0),
                    fmt(r.get("sharp_width")),
                    fmt(r.get("length")),
                    fmt(r.get("enlargement")),
                    fmt(r.get("plugin_error")),
                    fmt(r.get("empty_fraction"), 3),
                    fmt(r.get("set_cover"), 3),
                ]
                for r in partition
            ],
        )
        rare, body = self.data["rare"], []
        for expected in sorted({r["expected_count"] for r in rare}):
            by_method = {r["method"]: r for r in select(rare, expected_count=expected)}
            body.append(
                [fmt(expected, 1), fmt(number(expected) / 1000)]
                + [
                    fmt(by_method.get(m, {}).get("upper_mass"), 5)
                    for m in ("cp", "kl", "eb", "hoeffding", "wald", "bootstrap")
                ]
            )
        self.table(
            "rare-upper",
            ["$np$", "True $p$", "CP", "KL", "Bernstein", "Hoeffding", "Wald", "Bootstrap"],
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
                    fmt(r.get("n"), 0),
                    fmt(r.get("gamma"), 2),
                    "Yes" if r.get("model_contains_truth") else "No",
                    fmt(r.get("sharp_width")),
                    fmt(r.get("length")),
                    fmt(r.get("set_cover"), 3),
                    fmt(r.get("target_cover"), 3),
                ]
                for r in gamma
            ],
        )
        external = select(self.data["external_radius"], experiment="external")
        self.table(
            "external-precision",
            [
                "External sample size",
                "Sharp width",
                "Mean width",
                "MC SE",
                "Set cov.",
                "Target cov.",
            ],
            [
                [
                    (
                        "Exact"
                        if r["setting"] == -1
                        else "None" if r["setting"] == 0 else fmt(r["setting"], 0)
                    ),
                    fmt(r.get("sharp_width")),
                    fmt(r.get("length")),
                    fmt(r.get("length_mcse"), 5),
                    fmt(r.get("set_cover"), 3),
                    fmt(r.get("target_cover"), 3),
                ]
                for r in external
            ],
        )
        atomic = [
            r
            for r in select(primary, kind="budget")
            if r.get("method") in ("atomic_kl", "atomic_kl_aggregate", "kl", "cp")
        ]
        self.table(
            "atomic-kl",
            ["Procedure", "Mean width", "MC SE", "Set cov.", r"$95\%$ MC interval"],
            [
                [
                    METHODS[r["method"]],
                    fmt(r.get("length")),
                    fmt(r.get("length_mcse"), 5),
                    fmt(r.get("set_cover"), 3),
                    mc_interval(r),
                ]
                for r in atomic
            ],
        )

    def figure(self, name: str, title: str, plot: Callable, rows: list[dict]) -> None:
        # Do not silently substitute a different typeface in publication figures.
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
        rare = self.data["rare"]

        def rare_plot(ax):
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
                        r"Bridge budget, $\Gamma$"
                        if model == "budget"
                        else r"Time-only ratio bound, $\lambda$"
                    ),
                    ylabel="Population sharp endpoints",
                )

            self.figure(
                f"sensitivity_{model}",
                "Bridge-effect budget" if model == "budget" else "Time-only density-ratio model",
                sensitivity_plot,
                rows,
            )

    def finish(self) -> dict:
        # Remove only outputs generated by previous versions of this reporter.
        for name in ("all_tables.tex", "external-radius.tex", "reference-precision.tex"):
            (self.tables / name).unlink(missing_ok=True)
        return {"artifacts": self.artifacts, "missing_sources": self.missing}


def generate_reports(
    archive: Path, figures: Path | None = None, tables: Path | None = None
) -> dict:
    archive = Path(archive)
    if not archive.is_dir():
        raise FileNotFoundError(f"No experiment archive found: {archive}")
    report = Report(
        archive,
        Path(figures) if figures is not None else archive / "figures",
        Path(tables) if tables is not None else archive / "tables",
    )
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
    main()
