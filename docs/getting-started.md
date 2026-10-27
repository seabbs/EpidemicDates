# Getting started

The tools reconstruct a missing earlier event date, such as symptom onset, infection or sampling, from a later observed date, such as the report date.
They also allow for right truncation.
The examples here use symptom onsets and report dates.

There are four implementations.
The JavaScript module drives the [interactive page](https://seabbs.github.io/EpidemicDates/interactive/).
The Python package is a stochastic simulation that cross-checks it.
The R and Julia packages hold the line-list tools.
All four can impute a real line list.

## Install and load

=== "Python"

    The package is `epidemic_dates`.
    It needs Python 3.11 or later, NumPy and SciPy.
    Install it from GitHub with `pip` or [uv](https://docs.astral.sh/uv/):

    ```bash
    pip install git+https://github.com/seabbs/EpidemicDates
    ```

    Or clone the repository and let uv manage the environment:

    ```bash
    git clone https://github.com/seabbs/EpidemicDates
    cd EpidemicDates
    uv run python -c "import epidemic_dates; print(epidemic_dates.METHODS)"
    ```

    Load it in Python:

    ```python
    import epidemic_dates as ed

    ed.METHODS  # ('backsampling', 'mean', 'forward', 'em')
    ```

    The package also installs the `epidemic-dates` command:

    ```bash
    epidemic-dates --reps 50 --scenarios C W
    ```

=== "JavaScript"

    The module is the single file [`epidemic-dates.js`](https://github.com/seabbs/EpidemicDates/blob/main/epidemic-dates.js).
    It has no dependencies.
    Copy it into your project, or clone the repository:

    ```bash
    git clone https://github.com/seabbs/EpidemicDates
    cd EpidemicDates
    ```

    In Node 20 or later, load it with `require`:

    ```js
    const EpidemicDates = require("./epidemic-dates.js");

    EpidemicDates.METHODS.map((m) => m.key); // ['backsampling', 'mean', 'fwd', 'gen']
    ```

    In a browser, the script defines the global `EpidemicDates`:

    ```html
    <script src="epidemic-dates.js"></script>
    <script>
      const res = EpidemicDates.run();
    </script>
    ```

=== "R"

    The package is `epidemicdates`.
    It needs R 4.1 or later and only base R and `stats`.
    Install it from the `r` folder of the repository with [remotes](https://remotes.r-lib.org):

    ```r
    remotes::install_github("seabbs/EpidemicDates", subdir = "r")
    ```

    Load it and impute a small line list, with `NA` for missing onsets:

    ```r
    library(epidemicdates)

    res <- impute_linelist(c(0, 2, NA, 5, NA), c(3, 6, 7, 9, 12))
    round(res$imputed$em, 2)
    ```

    ```text
     [1] 2.47 0.00 1.00 0.00 0.00 1.50 0.00 0.00 0.00 0.03 0.00 0.00 0.00
    ```

=== "Julia"

    The package is `EpidemicDates.jl`.
    It is not registered and needs Julia 1.10 or later.
    Install it from the `julia` folder of the repository:

    ```julia
    using Pkg
    Pkg.add(url = "https://github.com/seabbs/EpidemicDates", subdir = "julia")
    ```

    Load it and impute a small line list, with `missing` for missing onsets:

    ```julia-repl
    julia> using EpidemicDates

    julia> res = impute_linelist([0, 2, missing, 5, missing], [3, 6, 7, 9, 12]);

    julia> round.(res.imputed.em; digits = 2)
    13-element Vector{Float64}:
     2.47
     0.0
     1.0
     0.0
     0.0
     1.5
     0.0
     0.0
     0.0
     0.03
     0.0
     0.0
     0.0
    ```

## Quick start on your own data

Short versions of the worked example, in all four languages, are on the [home page](index.md#quick-start-on-your-own-data).
The [worked example](real-linelist.md) goes through each step with the [example line list](data/example-linelist.csv).

## Next steps

- Read how the [methods](methods.md) work and what they [assume](assumptions.md).
- Follow the [worked example](real-linelist.md) on a real line list.
- Use the [Python](python/index.md), [JavaScript](javascript/index.md), [R](r/index.md) or [Julia](julia/index.md) guide.
- Look up a function in the [Python](python/reference.md), [JavaScript](javascript/reference.md), [R](r/reference.md) or [Julia](julia/reference.md) API reference.

## Development

Development needs uv, Node 20 or later and optionally [Task](https://taskfile.dev).

```bash
task test     # JavaScript tests, Python tests and the cross-language check
task lint     # linters
task docs     # build the page and this site into _site/
task serve    # serve the page on http://localhost:8000
```
