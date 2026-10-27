# Onset imputation by back-sampling reporting delays

An interactive page exploring onset-date imputation for cases with a missing onset.
Each missing onset is set to the report date minus a delay drawn from the pooled empirical onset-to-report delays.
This is the approach discussed in [BDBV-Modeling-Hub issue #61](https://github.com/InsightNet-US/BDBV-Modeling-Hub/issues/61).

**Live page:** <https://seabbs.github.io/epidemic-imputation-demonstration/>

The page simulates an epidemic with a time-varying growth rate, reporting delays, missing onset dates and changes in ascertainment.
It compares four ways of imputing missing onsets against the true onsets of the same cases:

- a draw from the pooled empirical delays;
- a shift by the mean delay;
- a draw from the forward delay;
- EM back-projection, a simple generative approach.

The growth rate can be drawn directly on the page.
Presets use Ebola settings from [epiforecasts/BVDOutbreakSize](https://github.com/epiforecasts/BVDOutbreakSize) and [epiforecasts/bdbv-linelist-analysis](https://github.com/epiforecasts/bdbv-linelist-analysis).

## Contents

- `index.html`: the self-contained page. Open it in a browser.
- `sim/sim_onset_imputation.py`: an independent stochastic simulation used to cross-check the page.
- `tests/`: tests of the page's maths, run in Node with no dependencies.

## Running

Requires [uv](https://docs.astral.sh/uv/), Node 20 or later and optionally [Task](https://taskfile.dev).

```bash
task test     # node --test tests/
task sim      # uv run sim/sim_onset_imputation.py 200
task lint     # ruff on the simulation
task serve    # serve the page on http://localhost:8000
```

## Licence

MIT.
