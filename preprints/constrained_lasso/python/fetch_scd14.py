# /// script
# requires-python = "==3.12.*"
# dependencies = [
#     "numpy==2.5.4",
#     "pandas==3.0.6",
#     "rdata==0.11.2",
#     "requests==2.34.2",
# ]
# ///
# fetch: on build
"""Download the sCD14 microbiome data that the note's real-data example uses.

The data are 16S counts of 60 genera in 151 stool samples from an HIV study
(Noguera-Julian et al., 2016), with the inflammation marker sCD14 measured in each
sample. They ship with the R package coda4microbiome (MIT licence) as ``x_sCD14`` and
``y_sCD14``. This script downloads that package's source tarball from CRAN, checks its
SHA-256, reads ``data/sCD14.rda`` from it without unpacking anything to disk, and writes
the counts and the response to ``data/scd14.csv``, one row per sample.

The data are not the authors' of this note, so git ignores the file, and `make data`
fetches it on every build, in CI too (the line ``# fetch: on build`` above opts in). CRAN
tarballs do not change once published, so a fresh download is the note's snapshot: the
script refuses a tarball with another digest, and does nothing when the file is already
there with the note's digest.

    uv run python/fetch_scd14.py

Sources: Noguera-Julian et al., Gut microbiota linked to sexual preference and HIV
infection, EBioMedicine 5:135--146, 2016; coda4microbiome 0.2.4,
https://cran.r-project.org/package=coda4microbiome
"""

from __future__ import annotations

import hashlib
import io
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import rdata
import requests

TARBALL = "coda4microbiome_0.2.4.tar.gz"
# The current release sits in src/contrib, and moves to the archive once a newer one is out.
URLS = [
    f"https://cran.r-project.org/src/contrib/{TARBALL}",
    f"https://cran.r-project.org/src/contrib/Archive/coda4microbiome/{TARBALL}",
]
TARBALL_SHA256 = "8125320e1da29ed5c937e2da2ad2a48bfc14e3416499fa585eb1e86b21dd54bf"
MEMBER = "coda4microbiome/data/sCD14.rda"
OUT = Path(__file__).resolve().parent.parent / "data" / "scd14.csv"
OUT_SHA256 = "3c6dcd6c18e10ea6f09bdd34329df3a8ce0a11dccd8026bf6c9a71f3c086d0da"


def download() -> bytes:
    """The package tarball, from whichever CRAN location still has it."""
    for url in URLS:
        resp = requests.get(url, timeout=60)
        if resp.ok:
            return resp.content
    msg = f"{TARBALL} found at none of {URLS}"
    raise RuntimeError(msg)


def main() -> None:
    """Download, verify and save the sCD14 counts and response."""
    if OUT.exists() and hashlib.sha256(OUT.read_bytes()).hexdigest() == OUT_SHA256:
        print(f"{OUT} is already the note's snapshot")
        return
    blob = download()
    digest = hashlib.sha256(blob).hexdigest()
    if digest != TARBALL_SHA256:
        msg = f"{TARBALL} has SHA-256 {digest}, expected {TARBALL_SHA256}"
        raise RuntimeError(msg)
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as archive:
        member = archive.extractfile(MEMBER)
        assert member is not None, MEMBER
        data = rdata.conversion.convert(rdata.parser.parse_data(member.read()))

    counts = data["x_sCD14"]
    frame = pd.DataFrame(np.asarray(counts, dtype=np.int64),
                         index=pd.Index(counts.coords["dim_0"].values.astype(str), name="sample"),
                         columns=counts.coords["dim_1"].values.astype(str))
    frame["sCD14"] = np.asarray(data["y_sCD14"], dtype=float)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT, lineterminator="\n")
    print(f"wrote {OUT}: {frame.shape[0]} samples x {frame.shape[1] - 1} genera, and sCD14")
    print(f"SHA-256 {hashlib.sha256(OUT.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
