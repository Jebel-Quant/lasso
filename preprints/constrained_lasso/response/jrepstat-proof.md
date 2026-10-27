# Corrections to the proof

**Manuscript:** The Constrained LASSO is a Portfolio: A numerical check of the CLA–LASSO identity under constraints
**Journal:** Journal of Reproducible Statistics, Volume 1, Article 10

Thank you for the acceptance, and for running the code on two machines I do not have. I reproduced your aarch64 Linux run in a container and got your numbers to the digit: a residual of 1.46e-9, the solver's objective lower by 1.13e-10, and cond(X) = 5.3e15 for *Wide, zero-sum*.

**1. The two bounds of Section 5.5.** Both sentences now name the platform, and one sentence after them gives your aarch64 values. Please replace the passage from "Every path meets the optimality conditions" to "the objective is small." with:

> On the x86_64 machine that wrote the numbers of this note, every path meets the optimality conditions to 1.3×10⁻⁹ or better along every segment. The largest residuals come at the end of a path, near λ = 10⁻⁴ λ_max, where the fit almost interpolates; elsewhere they are at rounding. The largest is on one seed of *Zero-sum, signs*, with cond(X) near 10¹⁶. The interior-point solver agrees to 4.2×10⁻⁶, and its objective is never lower than the path's by more than 8.1×10⁻¹², relative. That happens only near the end of a path, where the fit almost interpolates and the objective is small. These two bounds, unlike the others, depend on the CPU. On aarch64 Linux the same paths meet the optimality conditions to 1.5×10⁻⁹, and the solver's objective is lower by up to 1.1×10⁻¹⁰, relative, on the same seed and at the same end of the path.

(Please keep the values 4.2×10⁻⁶, 1.3×10⁻⁹ and 8.1×10⁻¹² exactly as your proof prints them; they come from the x86_64 CI.)

At the end of the second paragraph of Data and Code Availability, after "costs are given only as ratios.", please add:

> Only at the end of the paths with p > n, where the fit almost interpolates and cond(X) nears 10¹⁶, do they differ by more, up to a factor of ten (Section 5.5); the tests there allow for it.

**A third sentence, in Section 5.4.** `test_stress` fails on *duplicate 1e-04* for a different reason, which your run uncovered. Its reference is LARS, and the test that fails compares the gap against LARS with cond(X)² times the machine epsilon: 3.7×10⁻⁷ on aarch64 Linux against 2.3×10⁻⁷ (1.7×10⁻⁷ on x86_64 and on my M4 Pro). Both paths meet the optimality conditions to rounding there (3.6×10⁻¹³ and 5.3×10⁻¹³), so the paths are right. The sentence that claims the gap "stays below" that size is wrong: cond(X)² eps is the scale of the gap, not a bound on it. Please replace "Against LARS the gap stays below cond(X)² times the machine epsilon, and LARS's own KKT residual is at most …: both paths are right." with:

> Against LARS the gap is of the order of cond(X)² times the machine epsilon, the size a perturbation of rounding size can move the solution of a least-squares problem, and on *Duplicate* the CPU decides on which side of it the gap falls: on aarch64 Linux it is 3.7×10⁻⁷, 1.6 times that size. LARS's own KKT residual is at most …: both paths are right.

(keeping the value of LARS's residual as printed).

Both tests are corrected in the repository: `test_stress` now holds the gap to ten times cond(X)² eps, and `test_wide` holds the solver's objective to 10⁻⁹. Both now pass on aarch64 Linux. The archived commit 327bcd7 is unchanged, and the fix is in commit `<commit>` of github.com/Jebel-Quant/lasso. Please mention it on the article page if your practice allows.

**2. The stray full stop.** It goes, but deleting it alone leaves a run-on sentence. The fix is in the passage under point 1: "…with cond(X) near 10¹⁶. The interior-point solver agrees to 4.2×10⁻⁶, and its objective is never lower …".

**3. The cost ratio at p = 500.** No change. The text names the M4 Pro, and your 301 on an M1 Max is the same order.

**4. Data and Code Availability.** Agreed.

**5. Citations, references and affiliation.** The affiliation is right. The manuscript was written for numbered citations, and two kinds of sentence do not survive the change to author-year. Please check each of these in the proof.

*Authors named before the citation* should read as one textual citation, "Tibshirani (1996) introduced", not "Tibshirani (Tibshirani, 1996)":

- Section 1: "Tibshirani (1996) introduced the LASSO"; "which Markowitz (1956) wrote for portfolios"; "Schmelzer and Hastie (2026) prove"; "Brodie et al. (2009) put an ℓ1 penalty"; "Gaines, Kim and Zhou (2018) derive"; "the comlasso algorithm of Jeon et al. (2020)"; "Hobson, Butcher and Lentati (2026) revisit".
- Section 2: "Markowitz's (1956) algorithm handles".
- Section 5.1: "in the style of Lin et al. (2014)"; "as Hobson et al. (2026) do".
- Section 5.6: "The algorithm of Gaines et al. (2018)".
- Section 6: "The non-terminating examples of Hobson et al. (2026)".
- Section 7: "the identity of Schmelzer and Hastie (2026)".

*Citations used as nouns* need the authors' names:

- Section 1: "The propositions need nothing from Schmelzer and Hastie (2026) beyond…".
- Section 1: "We assume, as Schmelzer and Hastie (2026) do, that…" (note "do", not "does").
- Sections 3 and 4: "the standing assumption of Schmelzer and Hastie (2026, Section 2)" and "the standing assumption of Schmelzer and Hastie (2026)".
- Section 5: "Each check tests a prediction of Schmelzer and Hastie (2026). Their Lemma 1…".

*The reference list.* I checked my 19 DOIs against their registry records. The records differ from the published articles in ways that matter for the entries you set from them:

- **Osborne, Presnell and Turlach (2000)** and **Wilson, Meigs, Sullivan et al. (2007)**: each Crossref record lists only the first author. Please list all the authors, and cite them as "Osborne et al. (2000)" and "Wilson et al. (2007)", not "Osborne (2000)" or "Wilson (2007)".
- **Years.** The records also carry an online date that is not the year of publication: Tibshirani (1996), online 2018; Markowitz (1956), online 2006; James, Paulson and Rusmevichientong (2020), online 2019. Please make sure the citation and the list give 1996, 1956 and 2020.
- **Two entries by me alone in 2026**, the cvxcla paper and my earlier article in this journal, need 2026a and 2026b. Schmelzer and Hastie (2026) is a different entry and needs no letter.
- **My earlier article** (doi:10.68365/jrepstat.2026.7): its record carries no article number. Please give it as Volume 1, Article 7.
- R. Tibshirani (1996) and R. J. Tibshirani (2013) are different authors. If the house style allows, initials in the citations would keep them apart.

**Open review.** I am glad for the review record to be published with the article.

**Article processing charge.** Paid, with reference RE-2026-002.

Thomas Schmelzer
