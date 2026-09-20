# Bootstrap and permutation statistics (derived product)

`EXP-STATS-bootstrap-permutation.csv` was produced by `conference-track/tools/stats_upgrade.py`
from the archived products below. Fixed seed 20260921; 10,000 bootstrap resamples and 10,000
permutations. Units are fold-level means, i.e. the between-fold definition.

Sources: `EXP-V2-03-summary.csv` (scope, six folds), `EXP-V2-05-raw.csv` and `EXP-V2-06-raw.csv`
(unit effect, fold-level rates per cell), `EXP-V1-10-variance-decomposition.csv` (noise floor).

Headline readings:

* six-fold holdout mean 40.63%, bootstrap 95% interval **12.0 to 72.7 pp** (the interval is wide
  because the fold distribution is bimodal, which is the point of the scope result);
* permutation test on the between-fold spread, k = 1 versus k_max: Isolation Forest p = 0.18,
  3σ RMS p = 0.51 (neither separates), Mahalanobis p < 0.001 in the opposite direction
  (its spread grows, consistent with saturation);
* noise-floor components 5.20 pp [3.60, 6.61], 4.39 pp [3.53, 5.12], 5.18 pp [3.60, 6.58].

Consequence for the text: the second face cannot be stated as a significant unit effect under the
between-fold definition; it is reported as definition-dependent, which is the tightened claim.
