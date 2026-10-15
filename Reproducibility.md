# Reproducibility

A result is *reproducible* when someone else can take the same data and the same code
and get the same numbers. That is a lower bar than *replicable*, where a new study with
new data reaches the same conclusion. It is also the bar most often missed, and the
cheapest to clear.

This note explains why the preprints here care about it, tells some stories of what goes
wrong when nobody does, and shows that the same things went wrong here. How the
repository is set up to prevent them is in [`introduction.md`](introduction.md).

## Why it matters

**A check that cannot be rerun is only a claim.** The note in `preprints/lasso/` exists to
verify a theorem numerically: that Markowitz's Critical Line Algorithm and the LASSO path
trace the same curve. A table of residuals near $10^{-13}$ is evidence only if a reader
can run the code and get those residuals too. Otherwise it asks to be trusted, which is
what a verification is supposed to avoid.

**Most errors are found by someone rerunning the work.** Reviewers rarely rerun code, and
authors stop looking once the numbers seem right. The errors in the stories below were
found when someone outside the project tried to get the published numbers and could not.

**The person most likely to need it is the author.** Six months later, the machine has
been upgraded, a library has changed its defaults, and the data source has revised its
history. Without a record of what produced the figures, the author cannot reproduce
them either, and cannot tell whether a new number disagrees because something improved
or because something broke.

**It is cheaper at the start.** Pinning a dependency, fixing a seed or recording where data
came from takes minutes while the work is fresh. Reconstructing them later can take days,
and sometimes cannot be done at all.

## What went wrong elsewhere

These cases are well documented. In all of them the problem stayed hidden until someone
tried to rerun the work.

**A spreadsheet range and national austerity.** Reinhart and Rogoff's *Growth in a Time of
Debt* (2010) found that growth falls sharply once public debt passes 90% of GDP, and it
was widely cited in arguments for austerity. In 2013 Thomas Herndon, a graduate student
at the University of Massachusetts Amherst, could not reproduce the result for a course
assignment. With the authors' spreadsheet he and his professors, Ash and Pollin, found
that an averaging formula left out several countries, and that some years of data had
been excluded and the rest weighted unconventionally. Corrected, the drop at 90% largely
disappeared.

**Cancer genomics and clinical trials.** At Duke University, gene-expression signatures
from Anil Potti's group were used to choose chemotherapy for patients in clinical trials.
Keith Baggerly and Kevin Coombes at MD Anderson tried to reproduce the analyses from the
published data and found sample labels shifted by one row, the labels of sensitive and
resistant samples reversed, and results that the data could not produce. It took years
before the trials were stopped and the papers retracted, and in 2015 the US Office of
Research Integrity found research misconduct. Their method, which they called *forensic
bioinformatics*, consisted of rerunning the published analysis.

**Two swapped columns, five retractions.** In 2006 Geoffrey Chang's group retracted five
papers on protein structures, three of them in *Science*. A program they wrote themselves
had swapped two columns of data, which flipped the handedness of the structures they
reported. Other groups had built on the structures for years.

**The same script, different answers on different computers.** In 2019 Neupane and
colleagues found that a widely used set of Python scripts for computing NMR chemical
shifts gave different results on different operating systems. The scripts matched files
by name and assumed they came back in sorted order, which Python's `glob` does not
promise. Results in papers that had used the scripts could depend on which computer had
run them.

**Gene names turned into dates.** Ziemann, Eren and El-Osta (2016) found that about one in
five genomics papers with gene lists in Excel supplements had some names corrupted by
Excel's automatic conversion: `SEPT2` became 2 September, `MARCH1` became 1 March. The
human gene nomenclature committee eventually renamed the genes.

**Finance has its own version.** Hou, Xue and Zhang (*Replicating Anomalies*, 2020) tried
452 return anomalies from the asset-pricing literature with common data and methods.
Most failed to hold up once microcaps were handled consistently. Many original results
depended on choices that the published papers did not fully record.

## What went wrong here

None of the following is hypothetical. Each happened in this repository, and each was
caught because the code and its checks could be rerun.

**The same code gives different last digits on different machines.** On another computer,
`constrained_lasso`'s Table 1 gave a gap of $8.6\times10^{-14}$ where the paper prints
$2.3\times10^{-13}$. Both are rounding error, and the conclusion stands, but a reader who
compared digit by digit would think the code was wrong. *What changed:* the tests check
tolerances, not exact values, and `lasso.tex` now says outright which numbers depend on
the platform.

**Timings that nobody could reproduce.** The note once printed that a path through 500
assets took 239 milliseconds "on a laptop". On an Intel i7-8565U it took 0.9 to 3.0
seconds, and two runs in a row differed by up to a factor of three, because the desktop
was busy. Naming the CPU would only have explained why a reader's numbers differ. *What
changed:* the note now gives costs as ratios (one path costs one to three QP solves; a
grid of QPs costs tens of paths), which held on both machines, and drops the seconds.
Rerunning the timings also turned up a claim that was too strong: on the smaller
Fama-French problem a grid of QPs costs about as much as the path, so the note no
longer says the path is always faster.

**Data that change underneath you.** The note's S&P 500 example used Yahoo Finance prices
of the current index constituents. The snapshot behind the note had 494 stocks and 94
corners, with the last corner at $\nu = -0.18$. A download on 2026-09-28 gave 492 stocks,
95 corners and $\nu = -0.88$; one on 2026-10-14 gave 491 stocks. Constituents change and
Yahoo revises adjusted prices. On the newest download, one corner matched its QP only to
$3.4\times10^{-10}$, above the test's tolerance of $10^{-10}$, while the path's own optimality
conditions still held to $3.7\times10^{-12}$; the gap was the QP's. *What changed:* `fetch_sp500.py` printed the
SHA-256 of what it wrote and whether it was the note's snapshot, and the README recorded
what a fresh download gave, so a reader could tell drifted data from a bug. Later the
example was replaced altogether: the note now runs at scale on 500 stocks simulated from
a seeded factor model, which every rerun reproduces exactly.

**Free to download is not free to redistribute.** To give the note a real-data check that
anyone could rerun, the Fama-French portfolio returns were committed to this public
repository, with a note that they were not under its MIT licence. Kenneth French replied
that publishing them here violates their copyright; a script that downloads them is fine.
The file was removed, and the git history was rewritten to purge it. Removing a file from
a public repository's history means rewriting commits and asking GitHub to purge what it
caches, so the time to get this right is before the first commit. *What changed:*
`fetch_french.py` downloads the data on demand and git ignores the result, and
`introduction.md` now says third-party data are committed only with the owner's written
permission.

**A merge that quietly dropped a section.** While a conflict between two branches was being
resolved, one side's whole version of `lasso.tex` was kept. That file predated an earlier
change, so the merge silently removed the note's new subtitle, abstract and closing
paragraph from `main`. Nothing failed, because the old text was still valid LaTeX. It was
noticed only when the abstract was next read and was the old one. *What changed:* the
text was restored from the original commit. Conflicts are now resolved hunk by hunk,
and the result is diffed against both parents before it is committed. The stamp on every built PDF,
with the commit hash, makes it possible to say exactly which version a reader saw.

**A library that says one thing and does another.** In positive mode, scikit-learn's
`lars_path` labels its last point $\alpha = 0$. In fact every active correlation there is
30.65: it is the path at $\lambda = 30.65$, where the algorithm stopped, not the
nonnegative least-squares fit at $\lambda = 0$. A result that trusted the label would be
off by $3\times10^{-2}$. It was found only because `constrained_lasso` compared the path with
a second, independent method. *What changed:* the note uses `scipy.optimize.nnls` for that
last point and documents the quirk, and the library versions are pinned, so the behaviour
the note describes is the behaviour a reader gets.

**A plot that looked right and was not.** The companion papers once drew efficient
frontiers by joining the corner portfolios with straight lines. Between corners the
frontier is a curved arc; on the test problem one segment's midpoint sits 23% of the
segment's width off the chord. The figures looked fine, and only checking the plotted
points against the formula showed the difference. *What changed:* the frontiers are now
computed densely along each segment, and the figures show the true arcs.

**A submission that would not have compiled.** arXiv compiles the LaTeX source itself, so a
figure missing from the uploaded archive breaks the paper after it is posted. *What
changed:* `make arxiv-check` unpacks the archive into an empty directory and compiles it
there, as arXiv will. A test archive with its figures removed fails the check, naming
the missing file, and CI runs the check on every pull request.

## What this repository does about it

In short (details in [`introduction.md`](introduction.md)):

- **One command per preprint** reproduces every figure and number, and a `test` mode
  checks them against tolerances.
- **Pinned and locked environments.** Each script declares its dependencies (PEP 723),
  and a lockfile fixes Python 3.12 and every package by version and hash.
- **Seeded, offline computation.** Simulated data come from fixed seeds; only the
  downloads need a network.
- **Data with a fingerprint.** Downloaded data are checked against the SHA-256 of the
  snapshot each note used, and the script says when they differ.
- **Independent baselines.** Every path is checked against its own optimality conditions
  and against solvers that share no code with it.
- **Continuous integration.** Every pull request reruns the scripts, compiles every PDF
  and checks every arXiv archive. Each PDF is stamped with the commit it came from.

## What it cannot fix

- **Hardware.** Floating-point results differ in the last digits between CPUs and BLAS
  builds, and wall-clock times differ by much more. The answer is tolerances and ratios,
  not exact values.
- **Other people's data.** Data that cannot be redistributed must be downloaded, and the
  source may revise them. The best available is a fingerprint of the version used and a
  record of how a fresh download differs.
- **Services that disappear.** A download script is only as durable as the URL it fetches.
  An archived snapshot of code and outputs, for example on Zenodo, outlives both.
