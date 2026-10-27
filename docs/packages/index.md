# Packages

The same methods are available in four languages.
Each package imputes missing onsets in a line list with back-sampling, the mean shift, the forward draw and EM back-projection, and corrects for right truncation.
They are tested against shared reference outputs, so they give the same numbers.
Pick a package to open its guide.

<div class="grid pkg-cards">
  <a class="card pkg-card" href="../python/">
    <img class="pkg-logo only-light" src="../assets/logo-python.svg" alt="">
    <img class="pkg-logo only-dark" src="../assets/logo-python-dark.svg" alt="">
    <strong>Python</strong>
    <span><code>epidemic_dates</code>, with the simulation study, the nowcast with uncertainty and a command-line tool.</span>
  </a>
  <a class="card pkg-card" href="../javascript/">
    <img class="pkg-logo only-light" src="../assets/logo-js.svg" alt="">
    <img class="pkg-logo only-dark" src="../assets/logo-js-dark.svg" alt="">
    <strong>JavaScript</strong>
    <span><code>epidemic-dates.js</code>, with no dependencies. It runs the interactive page and works in a browser or in Node, including the nowcast.</span>
  </a>
  <a class="card pkg-card" href="../r/">
    <img class="pkg-logo only-light" src="../assets/logo-r.svg" alt="">
    <img class="pkg-logo only-dark" src="../assets/logo-r-dark.svg" alt="">
    <strong>R</strong>
    <span><code>epidemicdates</code>, in base R. It covers imputing a line list, the delay distribution and the truncation correction.</span>
  </a>
  <a class="card pkg-card" href="../julia/">
    <img class="pkg-logo only-light" src="../assets/logo-julia.svg" alt="">
    <img class="pkg-logo only-dark" src="../assets/logo-julia-dark.svg" alt="">
    <strong>Julia</strong>
    <span><code>EpidemicDates.jl</code>. It covers imputing a line list, the delay distribution and the truncation correction.</span>
  </a>
</div>

Each guide links to its API reference.
