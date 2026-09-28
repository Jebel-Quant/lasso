# The long-short frontier of Section 5, traced by R's lars package.
#
# Runs the snippet of the note on the test problem and checks it against the path
# scikit-learn's lars_path returns. Run from paper/:
#
#     Rscript R/long_short.R
#
# The inputs come from `uv run make_figures.py export`. lars has no positivity
# constraint, so the long-only path of Section 7 needs scikit-learn.

library(lars)

read_matrix <- function(name) {
  as.matrix(read.csv(file.path("data", "test_problem", name), header = FALSE))
}
Sigma <- read_matrix("Sigma.csv")
mu <- read_matrix("mu.csv")[, 1]

# The snippet of Section 5.
X <- chol(Sigma)                          # t(X) %*% X == Sigma
y <- backsolve(X, mu, transpose = TRUE)   # t(X) %*% y == mu
fit <- lars(X, y, type = "lasso", normalize = FALSE, intercept = FALSE)
B <- t(coef(fit))                         # columns: start, corners, end

B_python <- read_matrix("B_lars_path.csv")
stopifnot(ncol(B) == ncol(B_python))
gap <- max(abs(B - B_python))
cat(sprintf("lars %s: %d columns, largest gap to lars_path %.1e\n",
            as.character(packageVersion("lars")), ncol(B), gap))
stopifnot(gap < 1e-12)
