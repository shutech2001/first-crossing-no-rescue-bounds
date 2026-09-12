from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import multiprocessing
import os
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path

from tqdm import tqdm

SRC_DIR = Path(__file__).resolve().parent
ROOT = SRC_DIR.parent
SUITES = ("core", "rare", "continuous", "observational", "partitions", "gamma")
COMPUTATION_FILES = (
    "experiments.py",
    "simulations.py",
    "methods.py",
    "extensions.py",
)
RESULT_DIRECTORIES = ("core", "gamma", "continuous", "observational", "partitions")


@contextmanager
def output_lock(out):
    """Lock the output directory without leaving a lock file in it."""
    import fcntl

    fd = os.open(out, os.O_RDONLY)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"Another process is using this output directory: {out}")
        yield
    finally:
        os.close(fd)


def has_saved_results(out):
    return any(any((out / name).glob("*.csv")) for name in RESULT_DIRECTORIES)


def prepare_checkpoints(out, cfg):
    """Keep resume identity only until the numerical results are saved."""
    checkpoints = out / "checkpoints"
    control = checkpoints / "run.json"
    sources = {
        name: hashlib.sha256((SRC_DIR / name).read_bytes()).hexdigest()
        for name in COMPUTATION_FILES
    }
    signature = stable_hash(dict(config=cfg, sources=sources))
    if control.exists():
        saved = json.loads(control.read_text())
        if saved.get("signature") != signature:
            raise SystemExit(
                "Incomplete-run settings or computational code differ. "
                "Resume with the original settings/code or use a new --out directory."
            )
    else:
        if has_saved_results(out) or any(checkpoints.iterdir() if checkpoints.exists() else ()):
            raise SystemExit(
                "Existing results found. Use --report-only to regenerate figures and "
                "tables, or a new --out directory for another simulation run."
            )
        atomic_json(control, dict(config=cfg, signature=signature))


class StudyProgress(tqdm):
    """Keep the accounting invariant even when rendering is disabled."""

    def update(self, n=1):
        if self.disable:
            self.n += n
            return None
        return super().update(n)


def json_default(value):
    import numpy as np

    if isinstance(value, np.ndarray):
        return {"__ndarray__": value.tolist(), "dtype": str(value.dtype)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)}")


def json_hook(value):
    if "__ndarray__" in value and "dtype" in value:
        import numpy as np

        return np.asarray(value["__ndarray__"], dtype=value["dtype"])
    return value


def stable_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=json_default).encode()
    ).hexdigest()


def atomic_json(path, value, compressed=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    opener = gzip.open if compressed else open
    with opener(tmp, "wt", encoding="utf-8") as f:
        json.dump(value, f, default=json_default, sort_keys=True, indent=None if compressed else 2)
        f.write("\n")
    os.replace(tmp, path)


def read_checkpoint(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f, object_hook=json_hook)


def checkpoint_path(out, job):
    identifier = job["id"]
    if any(x in identifier for x in ("/", "\\", "..")):
        raise ValueError(f"Unsafe job id: {identifier}")
    return out / "checkpoints" / (identifier + ".json.gz")


def worker_setup():
    # One numerical thread per worker avoids worker-count-dependent oversubscription.
    from threadpoolctl import threadpool_limits

    global _THREAD_LIMITS
    _THREAD_LIMITS = threadpool_limits(limits=1)
    os.environ["OMP_NUM_THREADS"] = "1"


def execute_job(job):
    if job.get("stage") == "audit":
        from simulations import deterministic_audits

        context, files = {}, deterministic_audits(job["config"])
    elif job.get("stage") == "preparation":
        from extensions import run_preparation_job

        context, files = run_preparation_job(job)
    elif job.get("task") in (
        "primary",
        "sample_size",
        "states12",
        "active_caps",
        "external",
        "radius",
        "gamma",
        "rare",
    ):
        from simulations import run_core_job

        context, files = {}, run_core_job(job)
    else:
        from extensions import run_extension_job

        context, files = {}, run_extension_job(job)
    return dict(
        id=job["id"],
        weight=job.get("weight", 1),
        stage=job.get("stage", "simulation"),
        job_spec={
            k: v for k, v in job.items() if k not in ("context", "config", "pop", "reference")
        },
        context=context,
        files=files,
    )


def run_jobs(jobs, out, workers, progress, on_complete=None):
    """Bound in-flight jobs and write each successful replication atomically."""
    pending = []
    for job in jobs:
        path = checkpoint_path(out, job)
        if path.exists():
            result = read_checkpoint(path)
            if result["id"] != job["id"]:
                raise RuntimeError(f"Checkpoint identity mismatch: {path}")
            progress.update(job.get("weight", 1))
            if on_complete:
                on_complete(result)
        else:
            pending.append(job)
    if not pending:
        return
    if workers == 1:
        worker_setup()
        for job in pending:
            result = execute_job(job)
            atomic_json(checkpoint_path(out, job), result, compressed=True)
            progress.update(job.get("weight", 1))
            progress.set_postfix_str(
                job.get("suite", job.get("task", "preparation")), refresh=False
            )
            if on_complete:
                on_complete(result)
        return
    executor = ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=worker_setup,
    )
    it = iter(pending)
    futures = {}

    def submit_next():
        try:
            job = next(it)
        except StopIteration:
            return False
        futures[executor.submit(execute_job, job)] = job
        return True

    try:
        for _ in range(min(len(pending), 2 * workers)):
            submit_next()
        while futures:
            future = next(as_completed(futures))
            job = futures.pop(future)
            result = future.result()
            atomic_json(checkpoint_path(out, job), result, compressed=True)
            progress.update(job.get("weight", 1))
            progress.set_postfix_str(
                job.get("suite", job.get("task", "preparation")), refresh=False
            )
            if on_complete:
                on_complete(result)
            submit_next()
    except BaseException:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)


def csv_value(value):
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, default=json_default, sort_keys=True)
    return value


def materialize(out):
    """Stream checkpoints into reproducible CSVs without loading the full study."""
    paths = sorted((out / "checkpoints").glob("*.json.gz"))
    schemas = {}
    # First pass collects union schemas (fallbacks can have additional fields).
    for path in paths:
        result = read_checkpoint(path)
        for name, rows in result["files"].items():
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe archive output: {name}")
            if not name.endswith(".csv"):
                raise ValueError(f"Expected numerical CSV output: {name}")
            keys = schemas.setdefault(name, {})
            for row in rows:
                keys.update(dict.fromkeys(row))
    temporary = Path(tempfile.mkdtemp(prefix=".materialize-", dir=out))
    handles, writers, row_counts = {}, {}, {}
    try:
        for name, fields in schemas.items():
            dest = temporary / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            handles[name] = dest.open("w", newline="", encoding="utf-8")
            writers[name] = csv.DictWriter(handles[name], fieldnames=list(fields))
            writers[name].writeheader()
            row_counts[name] = 0
        for path in paths:
            for name, rows in read_checkpoint(path)["files"].items():
                if name not in writers:
                    continue
                for row in rows:
                    writers[name].writerow({k: csv_value(v) for k, v in row.items()})
                    row_counts[name] += 1
        for f in handles.values():
            f.close()
        for name in schemas:
            (out / name).parent.mkdir(parents=True, exist_ok=True)
            os.replace(temporary / name, out / name)
    finally:
        for f in handles.values():
            f.close()
        shutil.rmtree(temporary)
    return row_counts


def full_protocol(cfg):
    return (
        set(cfg["suites"]) == set(SUITES)
        and cfg["finite_reps"] >= 500
        and cfg["rare_reps"] >= 20000
        and cfg["observational_reps"] >= 250
        and cfg["continuous_reps"] >= 250
        and cfg["partition_reps"] >= 250
        and cfg["oracle_n"] >= 2000000
        and cfg["calibration_n"] >= 100000
        and cfg["bootstrap_resamples"] >= 999
        and cfg["forest_trees"] >= 200
        and cfg["mesh"] <= 1 / 128
    )


def make_config(args):
    cfg = dict(
        suites=list(SUITES) if args.suite == ["all"] else args.suite,
        finite_reps=500,
        rare_reps=20000,
        observational_reps=250,
        continuous_reps=250,
        partition_reps=250,
        bootstrap_resamples=999,
        forest_trees=200,
        oracle_n=2000000,
        calibration_n=100000,
        oracle_batches=20,
        mesh=1 / 128,
        rare_chunk=500,
        seed=args.seed,
        bootstrap_seed=args.seed * 10,
        oracle_seed=args.seed * 100,
        external_seed=args.seed * 1000,
        forest_seed=args.seed * 10000,
    )
    if args.smoke:
        cfg.update(
            finite_reps=2,
            rare_reps=20,
            observational_reps=2,
            continuous_reps=2,
            partition_reps=2,
            bootstrap_resamples=19,
            forest_trees=8,
            oracle_n=20000,
            calibration_n=2000,
            oracle_batches=20,
        )
    for name in (
        "finite_reps",
        "rare_reps",
        "observational_reps",
        "continuous_reps",
        "partition_reps",
        "bootstrap_resamples",
        "forest_trees",
        "oracle_n",
        "calibration_n",
    ):
        value = getattr(args, name, None)
        if value is not None:
            cfg[name] = value
    if args.reps is not None:
        for name in ("finite_reps", "observational_reps", "continuous_reps", "partition_reps"):
            if getattr(args, name, None) is None:
                cfg[name] = args.reps
    cfg["suites"] = list(dict.fromkeys(cfg["suites"]))
    return cfg


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        type=Path,
        help="Output directory (default results, or results/smoke with --smoke).",
    )
    p.add_argument(
        "--workers", "--jobs", "-j", type=int, default=max(1, min(12, (os.cpu_count() or 2) - 1))
    )
    p.add_argument("--suite", nargs="+", choices=("all", *SUITES), default=["all"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--smoke", action="store_true", help="Small execution check, explicitly not the full study."
    )
    p.add_argument("--reps", type=int, help="Override all non-rare replication counts.")
    for name in (
        "finite-reps",
        "rare-reps",
        "observational-reps",
        "continuous-reps",
        "partition-reps",
        "bootstrap-resamples",
        "forest-trees",
        "oracle-n",
        "calibration-n",
    ):
        p.add_argument("--" + name, type=int)
    p.add_argument(
        "--report-only",
        action="store_true",
        help="Rebuild figures, tables, and summaries from saved CSVs; never simulate.",
    )
    p.add_argument(
        "--no-report",
        action="store_true",
        help="Save experiments and defer figure/table generation.",
    )
    p.add_argument("--no-progress", action="store_true")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the fixed protocol and job count without running or writing.",
    )
    args = p.parse_args(argv)
    if "all" in args.suite and args.suite != ["all"]:
        p.error("--suite all cannot be combined with individual suites")
    if args.workers < 1 or args.seed < 0:
        p.error("--workers must be positive and --seed must be nonnegative")
    for name in (
        "reps",
        "finite_reps",
        "rare_reps",
        "observational_reps",
        "continuous_reps",
        "partition_reps",
        "bootstrap_resamples",
        "forest_trees",
        "oracle_n",
        "calibration_n",
    ):
        v = getattr(args, name)
        if v is not None and v < (20 if name == "oracle_n" else 2):
            p.error("--" + name.replace("_", "-") + " is too small")
    args.out = (args.out or Path("results/smoke" if args.smoke else "results")).resolve()
    return args


def main(argv=None):
    args = parse_args(argv)
    out = args.out
    if args.report_only:
        if not out.is_dir() or not has_saved_results(out):
            raise SystemExit(f"No saved numerical CSVs found: {out}")
        with output_lock(out):
            if (out / "checkpoints").exists():
                raise SystemExit(
                    "An incomplete simulation run has checkpoints. "
                    "Resume that run before regenerating reports."
                )
            from report_experiments import generate_reports

            generate_reports(out)
        print(f"Regenerated figures and TeX tables from saved results: {out}")
        return
    cfg = make_config(args)
    from extensions import extension_jobs, preparation_jobs
    from simulations import core_jobs

    prep = list(preparation_jobs(cfg))
    for i, job in enumerate(prep):
        job.setdefault("id", f"preparation_{i}")
        job.update(stage="preparation", weight=1)
    audit = dict(id="deterministic_audits", stage="audit", suite="audits", weight=1, config=cfg)
    core = list(core_jobs(cfg))
    # Extension job count is known before references are calculated.
    extension_total = 8 * cfg["observational_reps"] if "observational" in cfg["suites"] else 0
    extension_total += 4 * cfg["continuous_reps"] if "continuous" in cfg["suites"] else 0
    extension_total += 3 * cfg["partition_reps"] if "partitions" in cfg["suites"] else 0
    total = 1 + len(prep) + sum(j.get("weight", 1) for j in core) + extension_total
    if args.dry_run:
        print(
            json.dumps(
                dict(
                    config=cfg,
                    workers=args.workers,
                    progress_total=total,
                    full_protocol=full_protocol(cfg),
                ),
                indent=2,
            )
        )
        return
    out.mkdir(parents=True, exist_ok=True)
    with output_lock(out):
        prepare_checkpoints(out, cfg)
        context = {}

        def accept(result):
            context.update(result.get("context", {}))

        with StudyProgress(
            total=total,
            desc="All simulations",
            unit="rep",
            disable=args.no_progress,
            mininterval=0.5,
            dynamic_ncols=True,
        ) as progress:
            run_jobs([audit, *prep], out, args.workers, progress, accept)
            ext = list(extension_jobs(cfg, context))
            for i, job in enumerate(ext):
                job.setdefault("id", f"extension_{i}")
                job.setdefault("weight", 1)
            if len(ext) != extension_total:
                raise RuntimeError(
                    f"Extension job count {len(ext)} differs from planned {extension_total}"
                )
            run_jobs([*core, *ext], out, args.workers, progress)
            if progress.n != total:
                raise RuntimeError(f"Incomplete progress: {progress.n}/{total}")
        print("Saving replication CSVs...", flush=True)
        materialize(out)
        # At this point every numerical result is in CSV; reporting no longer
        # depends on checkpoints, including when rendering fails or is deferred.
        shutil.rmtree(out / "checkpoints")
        if not args.no_report:
            print("Generating PDF figures and TeX tables from saved results...", flush=True)
            from report_experiments import generate_reports

            generate_reports(out)
        print(f"{'Full study' if full_protocol(cfg) else 'Selected execution check'} saved: {out}")


if __name__ == "__main__":
    main()
