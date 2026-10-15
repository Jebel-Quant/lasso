# Introduction

This repository holds preprints, each with its LaTeX source and the code and data behind
its figures and numbers. The aim is that anyone can check out a commit, run one command
per preprint, and get the same figures and numbers the PDF shows. This note explains
how a preprint is laid out, how to add a new one, how the Python code stays
reproducible with [uv](https://docs.astral.sh/uv/), [PEP 723](https://peps.python.org/pep-0723/)
and lockfiles, and how a preprint goes to arXiv.

## How a preprint is laid out

Each folder in `preprints/` is one preprint:

```
preprints/
  Makefile               the rules shared by every preprint
  lasso/
    lasso.tex            the note: the one .tex file with a \documentclass
    Makefile             just `include ../Makefile`
    python/
      make_figures.py    draws figures/*.pdf and prints every number the note quotes
      make_figures.py.lock  its locked environment (uv lock --script)
      fetch_*.py         optional: scripts that fetch or rebuild data, each with its lock
    data/                optional: data the scripts read
    figures/             drawn by make_figures.py, ignored by git
    README.md            what the code does and how to run it
  constrained_lasso/
    ...
```

Only two things are required. The folder must hold exactly one `.tex` file with a
`\documentclass` line, and the PDF takes that file's name, so no two preprints may share
one. The Python code is optional and lives in `python/`, each script next to its
lockfile. If `python/make_figures.py` exists, it writes the figures into
`figures/`, which is ignored by git and rebuilt from the script.

The shared rules in [`preprints/Makefile`](preprints/Makefile) are:

```sh
make figures      # uv run python/make_figures.py
make test         # uv run python/make_figures.py test
make pdf          # pdflatex, three passes; run `make figures` first
make arxiv        # <name>.tar.gz: the .tex and the figures it includes, for arXiv
make arxiv-check  # make arxiv, then compile the tarball alone, as arXiv will
make clean        # remove the LaTeX auxiliary files and the arXiv build
```

Run inside one preprint's folder, a target acts on that preprint. Run in `preprints/`, it
acts on all of them in parallel.

## Adding a preprint

1. Create a folder, say `preprints/my_note/`, and put the main source in it as
   `my_note.tex`. Use a plain class and an inline bibliography (`thebibliography`); the
   build runs no BibTeX pass.
2. Add a `Makefile` containing one line of substance:

   ```make
   include ../Makefile
   ```

3. If the note has figures or computed numbers, add a `python/make_figures.py` that writes the
   figures to `figures/` and prints the numbers. Give it a `test` mode that checks the
   same numbers as pytest tests with tolerances. Start the file with a PEP 723 block
   pinning every dependency (see below), and lock it with
   `uv lock --script python/make_figures.py`.
4. Add a short `README.md` that says what the code does and how to run it, and link the
   preprint from the top-level [`README.md`](README.md).
5. Check locally with `make figures`, `make test`, `make pdf` and `make arxiv-check`,
   then open a pull request into `main`.

There is nothing to register in CI. The workflow in
[`.github/workflows/compile.yml`](.github/workflows/compile.yml) finds every folder in
`preprints/` on its own. On each pull request it runs `python/make_figures.py` (if there is one)
with uv, compiles the PDF and runs `make arxiv-check`; it fails if a folder has no main
`.tex` file, or more than one. When the pull request is merged, the PDFs are published on the `paper` branch with
an `index.html` that lists them by title, and GitHub Pages serves that branch at
<https://jebel-quant.github.io/lasso/>. Each preprint's arXiv source goes to the `arxiv`
branch (see [Submitting to arXiv](#submitting-to-arxiv)). The first page of every built PDF carries a stamp
in its margin with the build date, the repository and the short commit hash. It is added
at build time, so the `.tex` source stays clean, and it ties every PDF to the exact code
that produced it.

## Reproducible code

A figure can only be reproduced if the code behind it runs again with the same
libraries. Four things make that happen here: each script declares its own pinned
dependencies (PEP 723), a lockfile next to it fixes everything those pull in, one tool
installs and runs them (uv), and the computations are seeded and run offline.

### PEP 723: dependencies inside the script

[PEP 723](https://peps.python.org/pep-0723/) is a Python standard for putting a script's
metadata in a comment block at the top of the script itself. Each `python/make_figures.py`
starts like this:

```python
# /// script
# requires-python = "==3.12.*"
# dependencies = [
#     "cvxpy==1.9.3",
#     "matplotlib==3.11.2",
#     "numpy==2.5.4",
#     "pytest==9.1.1",
#     "scikit-learn==1.9.1",
#     "scipy==1.18.1",
# ]
# ///
```

(This is `preprints/constrained_lasso/python/make_figures.py`; pytest is listed because
the script's `test` mode runs its checks with it.)

The block is TOML inside comments, so to Python it is just a comment and the script
runs as usual. Tools that understand PEP 723 read it to learn which Python version and
which packages the script needs.

Keeping the dependencies in the script has three advantages here:

- **Each preprint is self-contained.** There is no shared `requirements.txt` or
  `pyproject.toml`, so two preprints can pin different versions of the same library
  without conflict, and an old preprint never breaks because a newer one moved a pin.
- **The pins travel with the code.** Copy one file out of the repository, and its
  dependencies come with it.
- **The versions are exact.** Every direct dependency is pinned with `==`, so the
  versions that produced the published figures are on record in the same commit that is
  stamped on the PDF.

### Lockfiles: the rest of the environment

The `==` pins fix the direct dependencies, not what they pull in. So every script also
has a lockfile next to it: `uv lock --script python/make_figures.py` writes
`python/make_figures.py.lock`, with an exact version and hash for every package, and
`uv run` uses it from then on. `uv run --locked` fails instead if the lockfile no longer
matches the pins. `requires-python = "==3.12.*"` keeps the interpreter in step with the
lockfile. Every script in both preprints, the fetch scripts included, is locked this way.
After changing a pin, rerun `uv lock --script` on that script and commit both files.

When the pins are bumped, as in [#21](https://github.com/Jebel-Quant/lasso/pull/21),
CI reruns every script and its tests against the new versions before the change is
merged.

### uv: one command to run a script

[uv](https://docs.astral.sh/uv/) is a fast Python package and project manager from
Astral. It reads the PEP 723 block, so a single command does the whole setup:

```sh
uv run python/make_figures.py
```

uv finds or downloads a Python that satisfies the script's `requires-python` (3.12 in
both preprints), creates an isolated environment
with exactly the pinned packages, and runs the script in it. Nothing is installed into
your system Python, and no virtual environment needs to be created or activated by hand.
The environment is cached, so later runs start at once. The only prerequisite is uv
itself; see its [installation guide](https://docs.astral.sh/uv/getting-started/installation/).

CI uses the same command, through
[`astral-sh/setup-uv`](https://github.com/astral-sh/setup-uv), and keys uv's cache on
the script and its lockfile, so a change to either means a fresh install. The figures are cached too,
keyed on the preprint's Python files and data, so they are redrawn only when one of
those changes.

### Seeds, data and checks

- **Seeded and offline.** Simulated data come from `numpy.random.default_rng` with a
  fixed seed, and the computations need no network access, so a run gives the same
  numbers every time.
- **Data that cannot be redistributed are fetched, and checked.** The Fama-French returns
  in `preprints/lasso/` are not committed. `python/fetch_french.py` downloads them and
  prints the SHA-256 of the result, so you can tell whether you have the file the note
  used. Without that data, `python/make_figures.py` skips that example and runs
  everything else.
- **Third-party data are not committed.** The repository is public, so committing data
  publishes them. Even data that are free to download are usually copyrighted: Kenneth
  French asked that the Fama-French portfolio returns not be redistributed here, so
  `preprints/lasso/python/fetch_french.py` downloads them instead. Commit data only if they are your own or their owner has said in writing that
  they may be redistributed.
- **The numbers are tested.** `make test` reruns the numbers the note quotes as pytest
  checks with tolerances and exits nonzero if any of them misses.

### What is not pinned

The PEP 723 block alone pins the direct dependencies, but not *their* dependencies.
Each script's lockfile closes that gap, so the Python environment is fixed down to the
last package. What it cannot fix is the hardware: floating-point results can differ in
the last digits between CPUs and BLAS builds, and timings depend on the machine. So the
tests check tolerances rather than exact values and leave timings alone, and a preprint
should say as much where it prints residuals or times, as `lasso.tex` does in its
availability section. The LaTeX toolchain is
pinned in a similar way: CI installs TeX Live from the apt archive of a fixed runner
image, `ubuntu-24.04`, rather than `ubuntu-latest`.

## Submitting to arXiv

arXiv compiles the source itself, so what goes to it is the `.tex` and its figures, not
the PDF. `make arxiv` packs them into `<name>.tar.gz`, with only the figures the `.tex`
includes, in the same `figures/` layout. `make arxiv-check` then unpacks the tarball into
an empty directory and compiles it there, as arXiv will, so a missing figure or an
undefined reference shows up before posting.

CI runs `make arxiv-check` on every pull request. When one is merged, it publishes the
tarballs on the [`arxiv`](../../tree/arxiv) branch as `arxiv/<name>.tar.gz`, built from the
same commit as the PDFs on `paper`. To post a new version, download the preprint's tarball
from that branch and upload it to arXiv unchanged. Its figures are the ones CI drew, from
the locked environment.
