# run: after figures
# needs: Rglpk
# apt: libglpk40
#
# Run comlasso (Jeon et al., CSDA 148, 2020) on the zero-sum paths make_figures.py saved,
# and compare its path with the CLA's. comlasso traces the exact zero-sum LASSO path, with
# p > n too and no ridge, so it should agree with no workaround.
#
#   Rscript R/comlasso_check.R     # after `uv run python/make_figures.py`
#
# comlasso was never on CRAN. Its authors publish it as comlasso_0.2.zip, a package built
# for R 3.2 on Windows, in https://github.com/jenjong/ComLasso. The zip is fetched from a
# pinned commit into data/ (git ignores data/) and its SHA-256 checked. R 4 refuses to
# install a package built before R 4.0, so its functions are read from the package's
# serialized code with lazyLoad, unchanged, rather than installed. Its one dependency is
# Rglpk, from CRAN.
#
# comlasso's regression loss is Huber's with threshold `gam`; with `gam` above every
# residual it is half the residual sum of squares, the note's loss. It also fits an
# unpenalised intercept, which is zero here because make_figures.py centres X and y.
#
# Writes figures/comlasso.tex, the numbers the note quotes.

COMMIT <- "6c7cf0f3ab52d3922f8f89d5620b6e044a12b919"
URL <- sprintf("https://raw.githubusercontent.com/jenjong/ComLasso/%s/comlasso_0.2.zip", COMMIT)
SHA256 <- "82c1609f357e9591731b805345af5a87485db737a9e7e97cac0b7562971dee3f"
PATHS <- "figures/zero_sum_paths.npz"
OUT <- "figures/comlasso.tex"
TOL <- 1e-9 # the largest gap, relative to the largest coefficient, the check accepts
ROUNDOFF <- 1e-9 # a knot below this times lambda_max, but above zero, is the end reached by rounding

library(Rglpk)

# ---- comlasso, fetched and loaded ---------------------------------------------------
comlasso_env <- function() {
  dir <- "data/comlasso"
  zip <- file.path(dir, "comlasso_0.2.zip")
  if (!file.exists(zip)) {
    dir.create(dir, recursive = TRUE, showWarnings = FALSE)
    download.file(URL, zip, mode = "wb", quiet = TRUE)
  }
  stopifnot(tools::sha256sum(zip) == SHA256)
  unzip(zip, exdir = dir)
  env <- new.env()
  lazyLoad(file.path(dir, "comlasso", "R", "comlasso"), envir = env)
  # The functions call each other by name; let them find each other here.
  for (name in ls(env)) if (is.function(env[[name]])) environment(env[[name]]) <- env
  env
}

# ---- the saved paths ---------------------------------------------------------------
# An .npz is a zip of .npy files; each here is a little-endian float64 array in C order.
read_npy <- function(path) {
  con <- file(path, "rb")
  on.exit(close(con))
  stopifnot(identical(readBin(con, "raw", 6), as.raw(c(0x93, charToRaw("NUMPY")))))
  version <- readBin(con, "integer", 2, size = 1)
  size <- if (version[1] == 1) 2 else 4
  header <- rawToChar(readBin(con, "raw", readBin(con, "integer", 1, size = size, endian = "little")))
  stopifnot(grepl("'<f8'", header, fixed = TRUE), grepl("'fortran_order': False", header, fixed = TRUE))
  shape <- suppressWarnings(as.integer(strsplit(sub(".*\\((.*)\\).*", "\\1", header), ",")[[1]]))
  shape <- shape[!is.na(shape)]
  values <- readBin(con, "double", prod(shape), size = 8, endian = "little")
  if (length(shape) == 2) matrix(values, shape[1], shape[2], byrow = TRUE) else values
}

read_paths <- function(npz) {
  dir <- tempfile()
  unzip(npz, exdir = dir)
  files <- list.files(dir, pattern = "\\.npy$")
  names <- unique(sub("\\.(x|y|lam|beta)\\.npy$", "", files))
  setNames(lapply(names, function(n) {
    get <- function(k) read_npy(file.path(dir, sprintf("%s.%s.npy", n, k)))
    list(x = get("x"), y = get("y"), lam = get("lam"), beta = get("beta"))
  }), names)
}

# ---- comlasso against the CLA ------------------------------------------------------
# beta at each lambda of `at`, interpolated along a path whose knots are `lam` (falling)
# and `beta` (one knot per row): the path is affine in lambda between knots.
along <- function(lam, beta, at) {
  sapply(seq_len(ncol(beta)), function(j) approx(rev(lam), rev(beta[, j]), xout = at, rule = 2)$y)
}

compare <- function(env, path) {
  x <- path$x
  y <- path$y
  fit <- env$comlasso("regression", n = nrow(x), p = 1, K = ncol(x), X.raw = x, y = y,
                      gam = 1e6 * max(abs(y)), max.steps = 10 * ncol(x), lam.min = 0)
  lam <- fit$lambda
  beta <- t(fit$beta.rec[[1]])
  scale <- max(abs(path$beta))
  # Each path at the other's knots, so a knot one of them misses still shows.
  gap <- max(abs(along(lam, beta, path$lam) - path$beta), abs(along(path$lam, path$beta, lam) - beta)) / scale
  # As the fit starts to interpolate, comlasso can take a step to a lambda at rounding
  # level, then record lambda = 0 as well, or stop there. cvxcla counts such an event as
  # lambda = 0, so every point at rounding level is one end of the path.
  end <- sum(lam < ROUNDOFF * lam[1])
  list(knots = length(lam) - end + 1, roundoff = end - 1, cla = length(path$lam), gap = gap,
       intercept = max(abs(fit$beta0.rec)) / scale)
}

sci <- function(v) {
  parts <- strsplit(sprintf("%.1e", v), "e")[[1]]
  sprintf("$%s\\times10^{%d}$", parts[1], as.integer(parts[2]))
}

main <- function() {
  env <- comlasso_env()
  paths <- read_paths(PATHS)
  rows <- lapply(paths, function(p) compare(env, p))
  for (name in names(rows)) {
    r <- rows[[name]]
    cat(sprintf("%-10s comlasso %3d knots (+%d at rounding level), CLA %3d; gap %.1e, intercept %.1e\n",
                name, r$knots, r$roundoff, r$cla, r$gap, r$intercept))
  }
  wide <- rows[grepl("^wide-", names(rows))]
  narrow <- rows[!grepl("^wide-", names(rows))]
  stopifnot(all(sapply(rows, function(r) r$gap < TOL && r$knots == r$cla)))
  words <- c("no", "one", "two", "three")
  macros <- c(
    coGap = sci(max(sapply(narrow, `[[`, "gap"))),
    coWideGap = sci(max(sapply(wide, `[[`, "gap"))),
    coRoundoff = words[sum(sapply(rows, `[[`, "roundoff") > 0) + 1]
  )
  writeLines(sprintf("\\newcommand{\\%s}{%s}", names(macros), macros), OUT)
}

main()
