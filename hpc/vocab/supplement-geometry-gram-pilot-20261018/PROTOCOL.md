---
title: Development pilot of cross-unit Gram correction for native geometry
author: James Edward Ball
date: 2026-10-18
status: Pre-outcome protocol; target-fidelity preflight precedes simulation
---

# Question and decision sequence

The completed support-aware confirmation failed coverage for both basic-interval candidates. This small, separate development pilot tests one proposed analysis-only repair. It uses saved feature arrays for training-only diagnostics and the existing physical-window simulation for a known-truth check. Original outputs, freezes, gates and corrected transfer jobs remain unchanged. There is no new acquisition, alternative-method search, confirmation campaign or empirical population interval admission.

The proposed estimator corrects prototype Gram entries by excluding cross-products from the same five-degree geographic block, then converts those entries to cosine distances. This has an intended population mean-cosine target only under assumptions that make cross-unit products represent that target. For the study's fixed finite geographic-template population, excluding within-block pairs can change the geographic weighting permanently. The first step therefore compares the exact noiseless candidate limit with the ORIGINAL analytic target. A target mismatch rejects the candidate for the original estimand before the 120-repetition simulation. Its own limiting value must never replace the original truth.

# Fixed candidate

Let z_gc be the feature mean for class c in geographic cell g, with one shared physical-window vector reused across all labels. Let n_bc count the class's cells in five-degree block b and S_bc be their feature sum. Define S_c=sum_b S_bc and n_c=sum_b n_bc. The corrected Gram entry is

    G_cd = [S_c dot S_d - sum_b S_bc dot S_bd]
           / [n_c*n_d - sum_b n_bc*n_bd].

Counts remain counts of cells, so large blocks retain their actual cell mass. There is no equal-block replacement of the equal-cell target. The diagonal uses the same formula. Corrected distances are 1-G_cd/sqrt(G_cc*G_dd). Raw off-diagonal cosine estimates outside [-1,1] are retained and counted; neither clipping nor a positive-semidefinite projection is applied. A nonpositive self-entry, nonpositive pair denominator or nonfinite value makes the affected geometry unavailable. Missing any of the fixed 188 classes makes the whole candidate unavailable.

Image and appearance geometries use this correction. Text prototypes stay fixed. Geographic nuisance histograms retain original all-split class-window counts. Every deletion removes the same five-degree block from image, appearance and geographic data. The ordinary rank correlation and partial rank/SVD calculation use the production tie convention and singular-value cutoff 1e-12. The target remains the six original ordinary/partial baseline, semantic and paired-difference statistics; no crossnobis, CKA or replacement target is introduced.

If target fidelity survives, compute the full statistic theta and its G leave-one-block values theta_-b, where G is the number of distinct observed five-degree blocks across all splits. Define pseudovalues p_b=G*theta-(G-1)*theta_-b, centre theta_J=mean_b p_b, and SE=sqrt[(G-1)/G * sum_b(theta_-b-mean(theta_-b))^2]. Use degrees of freedom G-1. The two-sided interval is theta_J +/- t_.975,G-1*SE; the one-sided zero-null lower bound uses t_.95,G-1. Intersect the final interval with [-1,1] for individual correlations and [-2,2] for differences, retaining raw endpoints. This first-order jackknife variance need not correctly describe the bias-corrected rank estimator, especially with rare support or tied nuisance ranks; calibration must decide. Any undefined point or leave-block statistic, fewer than three blocks, nonpositive SE or empty bounded interval is an unavailable interval for that metric. No class, deletion or failed draw is omitted.

# Target-fidelity preflight

Construct every noiseless physical-window vector from the frozen world and retained labels. Aggregate those vectors within each class-cell, then apply the candidate to the full finite template. Compare candidate results with the original analytic population means and original geographic histograms. Report Gram-entry, cosine-distance, rank and final-statistic discrepancies. Cover all 15 existing scenario IDs, including beta=0/0.4, heterogeneous noise, five-/twenty-degree dependence, fixed design, fourfold support and independent text. The noiseless finite-template diagnostic is intentionally invariant to sample-noise regimes; every scenario remains identified separately. Include an explicit unequal-block, heterogeneous-location algebraic counterexample and a homogeneous-location control.

Numerical agreement uses absolute tolerance 1e-10 for the six statistics. A demonstrable difference in the finite-template limit beyond this tolerance is a target-fidelity failure, not a failed Monte Carlo coverage estimate. The tolerance is a numerical identity check and does not amend any original statistical gate. If any scenario fails, stop the simulation plan and report this single candidate rejected for the original estimand. Saved training-feature diagnostics still complete. No post-outcome repair is authorised by this protocol.

# Conditional simulation plan

Only after preflight passes and the parent review confirms that it preserves the intended target: eight fresh repetitions for each of the 15 frozen stress scenarios, 120 in total. Seed 2026101891 is used for new development samples; the world seed and ORIGINAL analytic truth remain fixed. The entire frozen physical-window DGP and admission thresholds remain in force. There is no redrawing after support or admission failure. Report unconditional availability/coverage/null rejection, conditional-on-admission results with their actual denominators, bias and Wilson Monte Carlo intervals. Retain the original coverage band [0.92,0.995], size band [0.035,0.065] and power reference 0.8. A 120-repetition development pilot cannot qualify an empirical interval.

Time one complete estimator with all block deletions before the remaining repetitions. If projected runtime exceeds the two-hour job budget, stop and report feasibility, retaining the incomplete status. Do not reduce repetitions, omit stress scenarios, sample deletions or alter rules to meet the budget.

# Saved-feature diagnostics and resource boundary

Read the retained derivative's immutable F3 manifest and verify the exact source hashes before accessing features. Use retained TRAINING rows only, with widx/window_id identity checks and the fixed admitted class mapping. Predeclare R1 Clay mean, R1 Prithvi-300 mean (`extract/r1/prithvi/mean.npy`) and appearance-rich features. Compute native dimension, finite-value checks, class/cell/block support, norm/covariance summaries, effective covariance rank and outcome-blind split-half class-mean disagreement. Split halves are determined by a fixed hash of five-degree block identity and seed 2026101892. Report classes missing a half rather than replacing the split. The diagnostics use no language arrays and no test-feature similarities; they cannot select a method or establish held-out coverage.

The job is limited to four CPUs, 32 GiB, nice=1000 and two hours on himem. It preserves the corrected transfer run's priority and eight-CPU/160-GiB inference reserve. No existing job is cancelled, moved or reconfigured. A separately hashed release and output directory are required. Tests cover brute-force cross-unit equality, unequal block sizes, shared labels, production tied-rank/SVD parity, invalid self-norms and denominators, missing/leave-block support failure, and provenance rejection.

# Statistical sources and limits

Efron and Stein, *The Jackknife Estimate of Variance* (1981), provides the jackknife variance basis: https://statistics.stanford.edu/technical-reports/jackknife-estimate-variance . This is not a finite-sample guarantee for the present rank statistic. Shao and Wu's work on general jackknife variance estimation emphasises the problem of nonsmooth estimators; the original paper is https://doi.org/10.1214/aos/1176347263 .

Diedrichsen and colleagues, *Comparing representational geometries using whitened unbiased-distance-matrix similarity* (2021), motivates independent-measurement cross-products while making the unbiased-distance/cosine distinction explicit: https://arxiv.org/abs/2007.02789 . Its metric is not substituted here. Schütt and colleagues, *Statistical inference on representational geometries* (2023), discusses measurement-noise contributions and the need to match resampling to the inferential population: https://doi.org/10.7554/eLife.82566 . The present geographic blocks and sparse multi-label samples require their own validation.
