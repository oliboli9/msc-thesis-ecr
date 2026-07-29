
The sample already contains the cleaned, augmented and processed files, so the rolling-horizon
runner wcan be run immediately:

```bash
python framework/runner.py
```

The runner is configured by the constants at the top of `framework/runner.py`, each overridable
by an environment variable:

```bash
RUNNER_START_YEAR=2025 RUNNER_START_WEEK=1 RUNNER_NUM_ITERATIONS=6 \
RUNNER_PLANNING_HORIZON_WEEKS=3 RUNNER_DECISION_HORIZON_WEEKS=1 \
RUNNER_SCENARIO=baseline \
RUNNER_MODEL_FILE=models/min-lateness-fulfilment.py \
RUNNER_RESULTS_DIR=results-lateness \
python framework/runner.py
```


## Rebuilding the data from scratch

```bash
python pipeline/0_clean_data.py                                  # data/raw   → data/clean
python pipeline/augment_demand.py --all --method scale --seed 42 # data/clean → data/augmented
python pipeline/config.py                                        # → data/processed/{arcs,commodities}
```

Edit `START_YEAR` / `START_WEEK` / `NUM_WEEKS` in `pipeline/config.py` to pick the horizon that
`pipeline/config.py` and `models/single-horizon.py` build.



## Layout

```
pipeline/   data cleaning, augmentation, and construction of the time-space network
            make_sample_data.py builds the anonymised sample in data/
models/     Gurobi models
framework/  runner.py (rolling horizon driver) and additional experiments
analysis/   metrics and figures
data/       sample data
```


