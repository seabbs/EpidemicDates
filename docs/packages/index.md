# Packages

The same methods are available in four languages.
Each package imputes missing onsets in a line list with back-sampling, the mean shift, the forward draw and EM back-projection, and corrects for right truncation.
They are tested against shared reference outputs, so they give the same numbers.

<div class="grid cards" markdown>

-   ![](../assets/logo-python.svg#only-light){ .title-mark } ![](../assets/logo-python-dark.svg#only-dark){ .title-mark } __Python__

    ---

    `epidemic_dates`, with the simulation study, the nowcast with uncertainty and a command-line tool.

    [:octicons-arrow-right-24: Guide](../python/index.md) · [API reference](../python/reference.md)

-   ![](../assets/logo-js.svg#only-light){ .title-mark } ![](../assets/logo-js-dark.svg#only-dark){ .title-mark } __JavaScript__

    ---

    `epidemic-dates.js`, with no dependencies. It runs the interactive page and works in a browser or in Node, including the nowcast.

    [:octicons-arrow-right-24: Guide](../javascript/index.md) · [API reference](../javascript/reference.md)

-   ![](../assets/logo-r.svg#only-light){ .title-mark } ![](../assets/logo-r-dark.svg#only-dark){ .title-mark } __R__

    ---

    `epidemicdates`, in base R. It covers imputing a line list, the delay distribution and the truncation correction.

    [:octicons-arrow-right-24: Guide](../r/index.md) · [API reference](../r/reference.md)

-   ![](../assets/logo-julia.svg#only-light){ .title-mark } ![](../assets/logo-julia-dark.svg#only-dark){ .title-mark } __Julia__

    ---

    `EpidemicDates.jl`. It covers imputing a line list, the delay distribution and the truncation correction.

    [:octicons-arrow-right-24: Guide](../julia/index.md) · [API reference](../julia/reference.md)

</div>
