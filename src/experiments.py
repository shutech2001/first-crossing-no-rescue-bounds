from __future__ import annotations

import argparse
import csv
import fcntl
import gzip
import hashlib
import json
import multiprocessing
import os
import shutil
import tempfile
from collections.abc import Callable, Generator
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits  # type: ignore
from tqdm import tqdm

from extensions import (
    extension_jobs,
    preparation_jobs,
    run_extension_job,
    run_preparation_job,
)
from report_experiments import generate_reports
from simulations import core_jobs, deterministic_audits, run_core_job

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
_THREAD_LIMITS: threadpool_limits | None = None


@contextmanager
def output_lock(out: Path) -> Generator:
    """Lock the output directory without leaving a lock file in it.

    Args:
        out (Path): The output directory.

    Yields:
        Generator: A context manager that locks the output directory.
    """
    fd = os.open(out, os.O_RDONLY)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"Another process is using this output directory: {out}")
        yield
    finally:
        os.close(fd)


def has_saved_results(out: Path) -> bool:
    """Check if there are any saved results in the output directory.

    Args:
        out (Path): The output directory.

    Returns:
        bool: True if there are any saved results, False otherwise.
    """
    return any(any((out / name).glob("*.csv")) for name in RESULT_DIRECTORIES)


def prepare_checkpoints(out: Path, cfg: dict) -> None:
    """Keep resume identity only until the numerical results are saved.

    Args:
        out (Path): The output directory.
        cfg (dict): The configuration dictionary.

    Raises:
        SystemExit: If the computational code or settings differ from the saved results.
    """
    checkpoints = out / "checkpoints"
    control = checkpoints / "run.json"
    sources = {
        name: hashlib.sha256((SRC_DIR / name).read_bytes()).hexdigest()
        for name in COMPUTATION_FILES
    }
    signature = stable_hash({"config": cfg, "sources": sources})
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
        atomic_json(control, {"config": cfg, "signature": signature})


class StudyProgress(tqdm):
    """Keep the accounting invariant even when rendering is disabled.

    Args:
        tqdm: The progress bar.
    """

    def update(self, n: int = 1) -> None:
        """Update the progress bar.

        Args:
            n (int): The number of units to update.

        Returns:
            None: The number of units updated.
        """
        if self.disable:
            self.n += n
            return None
        return super().update(n)


def json_default(value: Any) -> Any:
    """Serialize a value to JSON.

    Args:
        value (Any): The value to serialize.

    Returns:
        Any: The serialized value.
    """
    if isinstance(value, np.ndarray):
        return {"__ndarray__": value.tolist(), "dtype": str(value.dtype)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)}")


def json_hook(value: Any) -> Any:
    """Deserialize a value from JSON.

    Args:
        value (Any): The value to deserialize.

    Returns:
        Any: The deserialized value.
    """
    if "__ndarray__" in value and "dtype" in value:
        return np.asarray(value["__ndarray__"], dtype=value["dtype"])
    return value


def stable_hash(value: Any) -> str:
    """Compute a stable hash of a value.

    Args:
        value (Any): The value to hash.

    Returns:
        str: The stable hash.
    """
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=json_default).encode()
    ).hexdigest()


def atomic_json(path: Path, value: Any, compressed: bool = False) -> None:
    """Write a value to a JSON file atomically.

    Args:
        path (Path): The path to the JSON file.
        value (Any): The value to write.
        compressed (bool): Whether to compress the JSON file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    opener = gzip.open if compressed else open
    with opener(tmp, "wt", encoding="utf-8") as f:
        json.dump(value, f, default=json_default, sort_keys=True, indent=None if compressed else 2)
        f.write("\n")
    os.replace(tmp, path)


def read_checkpoint(path: Path) -> dict:
    """Read a checkpoint from a JSON file.

    Args:
        path (Path): The path to the JSON file.

    Returns:
        dict: The checkpoint.
    """
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f, object_hook=json_hook)


def checkpoint_path(out: Path, job: dict) -> Path:
    """Compute the path to a checkpoint file.

    Args:
        out (Path): The output directory.
        job (dict): The job dictionary.

    Returns:
        Path: The path to the checkpoint file.
    """
    identifier = job["id"]
    if any(x in identifier for x in ("/", "\\", "..")):
        raise ValueError(f"Unsafe job id: {identifier}")
    return out / "checkpoints" / (identifier + ".json.gz")


def worker_setup() -> None:
    """Set up the worker environment.

    This function sets the thread limit to 1 and the OMP_NUM_THREADS environment variable to 1.
    """
    # One numerical thread per worker avoids worker-count-dependent oversubscription.
    global _THREAD_LIMITS
    _THREAD_LIMITS = threadpool_limits(limits=1)
    os.environ["OMP_NUM_THREADS"] = "1"


def execute_job(job: dict) -> dict:
    """Execute a job.

    Args:
        job (dict): The job dictionary.

    Returns:
        dict: The result of the job.
    """
    if job.get("stage") == "audit":
        context, files = {}, deterministic_audits(job["config"])
    elif job.get("stage") == "preparation":
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
        context, files = {}, run_core_job(job)
    else:
        context, files = {}, run_extension_job(job)
    return {
        "id": job["id"],
        "weight": job.get("weight", 1),
        "stage": job.get("stage", "simulation"),
        "job_spec": {
            k: v for k, v in job.items() if k not in ("context", "config", "pop", "reference")
        },
        "context": context,
        "files": files,
    }


def run_jobs(
    jobs: list[dict],
    out: Path,
    workers: int,
    progress: StudyProgress,
    on_complete: Callable[[dict], None] | None = None,
) -> None:
    """Bound in-flight jobs and write each successful replication atomically.

    Args:
        jobs (list[dict]): The jobs to run.
        out (Path): The output directory.
        workers (int): The number of workers to use.
        progress (StudyProgress): The progress bar.
        on_complete (Callable[[dict], None]): A callback function to call when a job is complete.
    """
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


def csv_value(value: Any) -> str:
    """Convert a value to a CSV-compatible string.

    Args:
        value (Any): The value to convert.

    Returns:
        str: The CSV-compatible string.
    """
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, default=json_default, sort_keys=True, indent=None)
    return str(value)


def materialize(out: Path) -> dict[str, int]:
    """Stream checkpoints into reproducible CSVs without loading the full study.

    Args:
        out (Path): The output directory.

    Returns:
        dict[str, int]: The number of rows written for each schema.
    """
    paths = sorted((out / "checkpoints").glob("*.json.gz"))
    schemas: dict[str, dict[str, str]] = {}
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


def full_protocol(cfg: dict) -> bool:
    """Check if the configuration is a full protocol.

    Args:
        cfg (dict): The configuration dictionary.

    Returns:
        bool: True if the configuration is a full protocol, False otherwise.
    """
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


def make_config(args: argparse.Namespace) -> dict:
    """Make a configuration dictionary from command line arguments.

    Args:
        args (argparse.Namespace): The command line arguments.

    Returns:
        dict: The configuration dictionary.
    """
    cfg = {
        "suites": list(SUITES) if args.suite == ["all"] else args.suite,
        "finite_reps": 500,
        "rare_reps": 20000,
        "observational_reps": 250,
        "continuous_reps": 250,
        "partition_reps": 250,
        "bootstrap_resamples": 999,
        "forest_trees": 200,
        "oracle_n": 2000000,
        "calibration_n": 100000,
        "oracle_batches": 20,
        "mesh": 1 / 128,
        "rare_chunk": 500,
        "seed": args.seed,
        "bootstrap_seed": args.seed * 10,
        "oracle_seed": args.seed * 100,
        "external_seed": args.seed * 1000,
        "forest_seed": args.seed * 10000,
    }
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments.

    Args:
        argv (list[str]): The command line arguments.

    Returns:
        argparse.Namespace: The parsed command line arguments.
    """
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


def main(argv: list[str] | None = None) -> None:
    """Main function.

    Args:
        argv (list[str]): The command line arguments.
    """
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
            generate_reports(out)
        print(f"Regenerated figures and TeX tables from saved results: {out}")
        return
    cfg = make_config(args)
    prep = list(preparation_jobs(cfg))
    for i, job in enumerate(prep):
        job.setdefault("id", f"preparation_{i}")
        job.update(stage="preparation", weight=1)
    audit = {
        "id": "deterministic_audits",
        "stage": "audit",
        "suite": "audits",
        "weight": 1,
        "config": cfg,
    }
    core = list(core_jobs(cfg))
    # Extension job count is known before references are calculated.
    extension_total = 8 * cfg["observational_reps"] if "observational" in cfg["suites"] else 0
    extension_total += 4 * cfg["continuous_reps"] if "continuous" in cfg["suites"] else 0
    extension_total += 3 * cfg["partition_reps"] if "partitions" in cfg["suites"] else 0
    total = 1 + len(prep) + sum(j.get("weight", 1) for j in core) + extension_total
    if args.dry_run:
        print(
            json.dumps(
                {
                    "config": cfg,
                    "workers": args.workers,
                    "progress_total": total,
                    "full_protocol": full_protocol(cfg),
                },
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
            generate_reports(out)
        print(f"{'Full study' if full_protocol(cfg) else 'Selected execution check'} saved: {out}")


if __name__ == "__main__":
    main()
