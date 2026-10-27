# Render the roxygen Rd files of the R package to one Markdown page.
#
# Run from the repository root:
#   Rscript docs/r/make_reference.R
# Writes docs/r/reference.md, which is generated at build time and not
# tracked. Uses base R and tools only. Maths in \eqn{} and \deqn{} becomes
# $...$ and $$...$$ for MathJax via pymdownx.arithmatex.

args <- commandArgs(trailingOnly = TRUE)
man_dir <- if (length(args) >= 1) args[[1]] else file.path("r", "man")
out_file <- if (length(args) >= 2) {
  args[[2]]
} else {
  file.path("docs", "r", "reference.md")
}

tag <- function(x) {
  t <- attr(x, "Rd_tag")
  if (is.null(t)) "" else t
}

# Render a list of Rd nodes as inline Markdown.
render <- function(x) {
  if (is.character(x) && is.null(attr(x, "Rd_tag"))) {
    return(paste(x, collapse = ""))
  }
  t <- tag(x)
  if (t %in% c("TEXT", "RCODE", "VERB")) {
    return(paste(as.character(x), collapse = ""))
  }
  if (t == "COMMENT") {
    return("")
  }
  inner <- function(node = x) {
    paste(vapply(node, render, character(1)), collapse = "")
  }
  switch(
    t,
    "\\code" = code_link(inner()),
    "\\verb" = ,
    "\\samp" = paste0("`", inner(), "`"),
    "\\link" = {
      label <- inner()
      target <- attr(x, "Rd_option")
      target <- if (is.null(target)) label else as.character(target)
      target <- sub("\\(\\)$", "", sub("^=", "", target))
      if (target %in% topics) {
        paste0("[", label, "](#", target, ")")
      } else {
        label
      }
    },
    "\\linkS4class" = inner(),
    "\\emph" = paste0("*", inner(), "*"),
    "\\strong" = ,
    "\\bold" = paste0("**", inner(), "**"),
    "\\sQuote" = paste0("'", inner(), "'"),
    "\\dQuote" = paste0("\"", inner(), "\""),
    "\\pkg" = paste0("`", inner(), "`"),
    "\\file" = paste0("`", inner(), "`"),
    "\\R" = "R",
    "\\dots" = ,
    "\\ldots" = "...",
    "\\cr" = "  \n",
    "\\url" = paste0("<", inner(), ">"),
    "\\href" = paste0(
      "[",
      render_list(x[[2]]),
      "](",
      render_list(x[[1]]),
      ")"
    ),
    "\\eqn" = paste0("$", trimws(render_list(x[[1]])), "$"),
    "\\deqn" = paste0("\n\n$$\n", trimws(render_list(x[[1]])), "\n$$\n\n"),
    "\\itemize" = render_items(x, "- "),
    "\\enumerate" = render_items(x, "1. "),
    "\\preformatted" = paste0("\n\n```\n", trimws(inner()), "\n```\n\n"),
    "\\out" = inner(),
    "\\special" = inner(),
    "\\if" = ,
    "\\ifelse" = "",
    "\\figure" = "",
    "\\method" = paste0(render_list(x[[1]]), "(", render_list(x[[2]]), ")"),
    inner()
  )
}

# `[f()](#f)` reads better as [`f()`](#f).
code_link <- function(text) {
  link <- "^\\[([^]]*)\\]\\((#[^)]*)\\)$"
  if (grepl(link, text)) {
    sub(link, "[`\\1`](\\2)", text)
  } else {
    paste0("`", text, "`")
  }
}

render_list <- function(x) {
  paste(vapply(x, render, character(1)), collapse = "")
}

# Lists: split children at each \item. Continuation lines are indented with
# a sentinel that squash() keeps, replaced by four spaces at the end.
indent <- "\001"

render_items <- function(x, bullet) {
  items <- list()
  current <- NULL
  for (node in x) {
    if (tag(node) == "\\item") {
      if (!is.null(current)) {
        items[[length(items) + 1]] <- current
      }
      current <- list()
    } else if (!is.null(current)) {
      current[[length(current) + 1]] <- node
    }
  }
  if (!is.null(current)) {
    items[[length(items) + 1]] <- current
  }
  lines <- vapply(
    items,
    function(item) {
      text <- squash(render_list(item))
      # a display equation must be followed by a blank line
      if (endsWith(text, "$$")) {
        text <- paste0(text, "\n")
      }
      paste0(bullet, gsub("\n", paste0("\n", indent), text))
    },
    character(1)
  )
  paste0("\n\n", paste(lines, collapse = "\n"), "\n\n")
}

# Tidy whitespace: one line per sentence as written, no runs of blank lines.
squash <- function(text) {
  lines <- strsplit(text, "\n", fixed = TRUE)[[1]]
  lines <- sub("^[ \t]+", "", lines)
  lines <- sub("[ \t]+$", "", lines)
  text <- paste(lines, collapse = "\n")
  text <- gsub("\n{3,}", "\n\n", text)
  trimws(text)
}

# Keep code verbatim, dropping the leading and trailing blank lines.
code_block <- function(x) {
  text <- render_list(x)
  lines <- strsplit(text, "\n", fixed = TRUE)[[1]]
  keep <- which(nzchar(trimws(lines)))
  if (length(keep) == 0) {
    return("")
  }
  lines <- lines[seq(min(keep), max(keep))]
  paste0("```r\n", paste(lines, collapse = "\n"), "\n```")
}

section <- function(rd, name) {
  rd[vapply(rd, tag, character(1)) == name]
}

render_topic <- function(rd) {
  name <- render_list(section(rd, "\\name")[[1]])
  title <- squash(render_list(section(rd, "\\title")[[1]]))
  out <- c(paste0("## `", name, "` {#", name, "}"), "", paste0("*", title, "*"))
  heading_md <- function(heading, body) {
    if (nzchar(body)) c("", paste0("### ", heading), "", body)
  }
  desc <- section(rd, "\\description")
  if (length(desc)) {
    out <- c(out, "", squash(render_list(desc[[1]])))
  }
  usage <- section(rd, "\\usage")
  if (length(usage)) {
    out <- c(out, heading_md("Usage", code_block(usage[[1]])))
  }
  arguments <- section(rd, "\\arguments")
  if (length(arguments)) {
    rows <- c()
    for (node in arguments[[1]]) {
      if (tag(node) != "\\item") {
        next
      }
      arg <- squash(render_list(node[[1]]))
      text <- squash(render_list(node[[2]]))
      rows <- c(rows, paste0("`", arg, "`\n:   ", gsub("\n", "\n    ", text)))
    }
    out <- c(out, heading_md("Arguments", paste(rows, collapse = "\n\n")))
  }
  details <- section(rd, "\\details")
  if (length(details)) {
    out <- c(out, heading_md("Details", squash(render_list(details[[1]]))))
  }
  value <- section(rd, "\\value")
  if (length(value)) {
    out <- c(out, heading_md("Value", squash(render_list(value[[1]]))))
  }
  for (sec in section(rd, "\\section")) {
    out <- c(
      out,
      heading_md(squash(render_list(sec[[1]])), squash(render_list(sec[[2]])))
    )
  }
  refs <- section(rd, "\\references")
  if (length(refs)) {
    out <- c(out, heading_md("References", squash(render_list(refs[[1]]))))
  }
  examples <- section(rd, "\\examples")
  if (length(examples)) {
    out <- c(out, heading_md("Examples", code_block(examples[[1]])))
  }
  seealso <- section(rd, "\\seealso")
  if (length(seealso)) {
    out <- c(out, heading_md("See also", squash(render_list(seealso[[1]]))))
  }
  author <- section(rd, "\\author")
  if (length(author)) {
    out <- c(out, heading_md("Author", squash(render_list(author[[1]]))))
  }
  out
}

files <- list.files(man_dir, pattern = "\\.Rd$", full.names = TRUE)
if (length(files) == 0) {
  stop("no Rd files in ", man_dir)
}
rds <- lapply(files, tools::parse_Rd)
names(rds) <- vapply(
  rds,
  function(rd) {
    render_list(section(rd, "\\name")[[1]])
  },
  character(1)
)
topics <- names(rds)
# package overview first, then the main function, then the rest
first <- c(grep("-package$", topics, value = TRUE), "impute_linelist")
order <- c(intersect(first, topics), sort(setdiff(topics, first)))

page <- c(
  "# R API reference",
  "",
  "Every exported function of the R package `epidemicdates`.",
  "This page is generated from the roxygen documentation by",
  "[`docs/r/make_reference.R`](https://github.com/seabbs/EpidemicDates/blob/main/docs/r/make_reference.R).",
  "In R, see `?impute_linelist` and so on.",
  "{ .page-subtitle }",
  "",
  "!!! warning",
  "    This code was generated by an LLM.",
  "    Sam Abbott has only checked the interactive page.",
  "    Treat the results as exploratory."
)
for (topic in order) {
  page <- c(page, "", render_topic(rds[[topic]]))
}
page <- gsub(indent, "    ", page, fixed = TRUE)
page <- gsub("[ \t]+(\n|$)", "\\1", page, perl = TRUE)
text <- paste(page, collapse = "\n")
text <- gsub("\n{3,}", "\n\n", text)
writeLines(text, out_file)
message("Wrote ", out_file)
