# Experiments for "Boundary Reduction and Inference for No-Rescue Effects under Deterministic Rescue"

Reproducible code for the numerical experiments in "Boundary Reduction and Inference for No-Rescue Effects under Deterministic Rescue". The simulations use seed 42 by default.

## Reproducing the Experiments

### Requirements and Setup

```bash
# clone the repository
git clone git@github.com:shutech2001/no-rescue-causal-effects.git

# build the environment with poetry
poetry install

# activate virtual environment
eval $(poetry env activate)

# [Option] to activate the interpreter, select the following output as the interpreter
poetry env info --path
```

### Executing Numerical Experiments

Run the full no-rescue study, including structural-reduction audits, coverage comparisons, rare crossings, propensity estimation, and continuous-history partition sensitivity:

```bash
./scripts/run_experiments.sh
```

The launcher runs `src/experiments.py` through Poetry. Adjust `--workers` to control parallelism; the default uses up to 12 workers. Defaults are 500 replications per finite-state setting, 250 per observational/continuous/partition setting, and 20,000 per rare-crossing setting.

Results are saved directly under `results/`: numerical CSVs, summaries in `summaries/`, PDF figures in `figures/`, and 16 LaTeX table fragments in `tables/`. Each table contains only the `tabular` environment, with numbers in math mode; add captions and labels in the manuscript.

## Contact
If you have any question, please feel free to contact: tamano-shu212@g.ecc.u-tokyo.ac.jp
