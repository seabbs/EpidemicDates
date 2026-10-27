#!/usr/bin/env bash
# Compile every TikZ diagram in docs/diagrams/ (except the shared style.tex)
# with pdflatex and write docs/diagrams/<name>.svg and <name>-dark.svg.
# Needs pdflatex (TeX Live with the pgf, standalone and helvet packages) and
# poppler's pdftocairo on PATH. Run from anywhere:
#
#   docs/diagrams/build.sh          # every diagram
#   docs/diagrams/build.sh --png    # also write 150 dpi PNG previews to
#                                   # $DIAGRAM_PNG_DIR (default: a temp dir)
#
# The SVGs are committed, so the docs build does not need LaTeX.
set -euo pipefail

src_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
png=false
[[ "${1:-}" == "--png" ]] && png=true

for tool in pdflatex pdftocairo; do
  command -v "$tool" >/dev/null || {
    echo "build: $tool is not on PATH" >&2
    exit 1
  }
done

tmp="$(mktemp -d "${TMPDIR:-/tmp}/diagrams.XXXXXX")"
trap 'rm -rf "$tmp"' EXIT
keep="${DIAGRAM_PNG_DIR:-${TMPDIR:-/tmp}/epidemic-dates-diagrams}"

shopt -s nullglob
for tex in "$src_dir"/*.tex; do
  name="$(basename "$tex" .tex)"
  [[ "$name" == "style" ]] && continue
  for theme in light dark; do
    job="$name"
    pre=""
    if [[ "$theme" == "dark" ]]; then
      job="$name-dark"
      pre='\def\darkmode{}'
    fi
    echo "build: $job"
    (
      cd "$src_dir"
      pdflatex -interaction=nonstopmode -halt-on-error \
        -output-directory "$tmp" -jobname "$job" \
        "$pre\\input{$name.tex}" >"$tmp/$job.out" 2>&1
    ) || {
      tail -n 40 "$tmp/$job.out" >&2
      echo "build: $job failed to compile" >&2
      exit 1
    }
    pdftocairo -svg "$tmp/$job.pdf" "$src_dir/$job.svg"
    if $png; then
      mkdir -p "$keep"
      pdftocairo -png -transp -r 150 -singlefile "$tmp/$job.pdf" "$keep/$job"
    fi
  done
done
$png && echo "build: PNG previews in $keep"
echo "build: wrote $src_dir"
