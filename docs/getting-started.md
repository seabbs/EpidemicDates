# Getting started

The tools reconstruct a missing earlier event date, such as symptom onset, infection or sampling, from a later observed date, such as the report date.
They also allow for right truncation.
The examples here use symptom onsets and report dates.

There are two independent implementations.
The JavaScript module drives the [interactive page](https://seabbs.github.io/epidemic-imputation-demonstration/).
The Python package is a stochastic simulation that cross-checks it.
Both can impute a real line list.

## Install and load

=== "Python"

    The package is `epidemic_dates`.
    It needs Python 3.11 or later, NumPy and SciPy.
    Install it from GitHub with `pip` or [uv](https://docs.astral.sh/uv/):

    ```bash
    pip install git+https://github.com/seabbs/epidemic-imputation-demonstration
    ```

    Or clone the repository and let uv manage the environment:

    ```bash
    git clone https://github.com/seabbs/epidemic-imputation-demonstration
    cd epidemic-imputation-demonstration
    uv run python -c "import epidemic_dates; print(epidemic_dates.METHODS)"
    ```

    Load it in Python:

    ```python
    import epidemic_dates as ed

    ed.METHODS  # ('hub', 'mean', 'forward', 'em')
    ```

    The package also installs the `epidemic-dates` command:

    ```bash
    epidemic-dates --reps 50 --scenarios C W
    ```

=== "JavaScript"

    The module is the single file [`epidemic-dates.js`](https://github.com/seabbs/epidemic-imputation-demonstration/blob/main/epidemic-dates.js).
    It has no dependencies.
    Copy it into your project, or clone the repository:

    ```bash
    git clone https://github.com/seabbs/epidemic-imputation-demonstration
    cd epidemic-imputation-demonstration
    ```

    In Node 20 or later, load it with `require`:

    ```js
    const EpidemicDates = require("./epidemic-dates.js");

    EpidemicDates.METHODS.map((m) => m.key); // ['hub', 'mean', 'fwd', 'gen']
    ```

    In a browser, the script defines the global `EpidemicDates`:

    ```html
    <script src="epidemic-dates.js"></script>
    <script>
      const res = EpidemicDates.run();
    </script>
    ```

## Quick start on your own data

Short versions of the worked example, in both languages, are on the [home page](index.md#quick-start-on-your-own-data).
The [worked example](real-linelist.md) goes through each step with the [example line list](data/example-linelist.csv).

## Next steps

- Read how the [methods](methods.md) work and what they [assume](assumptions.md).
- Follow the [worked example](real-linelist.md) on a real line list.
- Use the [Python guide](python/index.md) or the [JavaScript guide](javascript/index.md).
- Look up a function in the [Python API](python/reference.md) or the [JavaScript API](javascript/reference.md).

## Development

Development needs uv, Node 20 or later and optionally [Task](https://taskfile.dev).

```bash
task test     # JavaScript tests, Python tests and the cross-language check
task lint     # linters
task docs     # build the page and this site into _site/
task serve    # serve the page on http://localhost:8000
```
