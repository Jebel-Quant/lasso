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

**5. Citations, references and affiliation.** <to complete after reading the proof>

**Open review.** I am glad for the review record to be published with the article.

**Article processing charge.** Paid, with reference RE-2026-002.

Thomas Schmelzer
