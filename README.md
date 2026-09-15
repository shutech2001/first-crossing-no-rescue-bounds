# Experiments for "First-Crossing Reduction and Inference for No-Rescue Effects under Deterministic Rescue"

Reproducible code for the numerical experiments in "First-Crossing Reduction and Inference for No-Rescue Effects under Deterministic Rescue".

## Reproducing the Experiments

### Requirements and Setup

```bash
# clone the repository
git clone git@github.com:shutech2001/first-crossing-no-rescue-bounds.git

# build the environment with poetry
poetry install

# activate virtual environment
eval $(poetry env activate)

# [Option] to activate the interpreter, select the following output as the interpreter
poetry env info --path
```

### Executing Numerical Experiments

Run the full study through Poetry:

```bash
./scripts/run_experiments.sh --workers 8 --out results
```

Common options:

- `--workers N` (or `-j N`): number of parallel workers; defaults to CPU count minus one, between 1 and 12.
- `--out PATH`: output directory; defaults to `results` (`results/smoke` with `--smoke`).
- `--seed N`: random seed; defaults to `42`.
- `--smoke`: run a small execution check with fewer replications and smaller reference samples.
- `--report-only`: regenerate tables and figures from saved results without rerunning simulations.

The output directory contains:

- `simulation_data.zip`: replication results, analysis data, population references, and run settings for later analysis.
- `tables/`: 17 LaTeX table fragments, with estimated quantities formatted to three significant digits.
- `figures/`: 8 manuscript PDF figures.

Rerun an interrupted command unchanged to resume. Use a new output directory for another simulation run; completed results are preserved.

## Contact
If you have any question, please feel free to contact: tamano-shu212@g.ecc.u-tokyo.ac.jp
