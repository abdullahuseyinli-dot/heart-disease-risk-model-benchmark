# Understanding the results

HeartShift asks whether disease classifiers transfer to an unseen hospital when
the available measurements change. The main study uses 920 historical records
from four hospitals. Each evaluation develops a model on three hospitals and
tests it on the fourth. The outcome is recorded angiographic disease (`num > 0`);
this is not a forecast of future heart disease.

The results answer two related questions:

| Question | Where to look |
| --- | --- |
| How well does the model classify disease using the measurements as recorded? | [Natural measurement metrics](RESULTS.md#natural-measurement-metrics): accuracy, balanced accuracy, precision, recall, F1, AUROC, and Brier. |
| How well do its probabilities hold up when additional measurements are hidden? | [Locked heart evaluation](RESULTS.md#locked-heart-evaluation): worst-policy balanced log loss, the primary research metric, and registered uncertainty intervals. |

## What each metric tells you

Positive means disease is present in the recorded endpoint. The classification
table uses the saved decision rule: predict positive when probability is at
least **0.5**. This is an evaluation convention, not a validated clinical cutoff.

| Metric | Meaning | Better direction |
| --- | --- | --- |
| Accuracy | Fraction of records classified correctly. Common classes contribute more. | Higher |
| Balanced accuracy | Average of sensitivity and specificity, giving the two outcome classes equal weight. | Higher |
| Precision | Among predicted positives, the fraction actually positive. | Higher |
| Recall / sensitivity | Among actual positives, the fraction detected. | Higher |
| F1 | Harmonic mean of precision and recall; does not include true negatives. | Higher |
| AUROC | How well probabilities rank positive records above negative records, across thresholds. | Higher |
| Brier score | Mean squared error between predicted probability and the binary outcome. | Lower |
| Balanced log loss (BLL) | Logarithmic probability error with equal total weight for the two classes; confidently wrong predictions are penalized heavily. | Lower |

AUROC, Brier, and BLL use probability scores without choosing a decision threshold.
The implementation is in [metrics.py](../src/heartshift/metrics.py); standard
definitions are documented in scikit-learn's
[classification metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics).

## Why balanced log loss is primary

The study tests probability robustness under missing measurements. Accuracy
records whether a prediction crosses a threshold but does not distinguish a
barely wrong probability from a confidently wrong one. Log loss captures that
difference. Class balancing prevents a hospital's majority class from dominating
its loss: Switzerland has 115 positive records and only eight negative records,
so always predicting positive would achieve 93.50% accuracy there.

The [benchmark contract](BENCHMARK_CARD.md#endpoints-and-uncertainty) specifies
the worst measurement-policy BLL within each hospital, then averages those four
values equally. This criterion was specified for the study before its final
evaluation; the additional classification table does not change it. The saved
[reporting code](../src/heartshift/reporting/outer_report.py) defines the exact
policy and replicate aggregation.

BLL is a loss, not a percentage of correct predictions. Lower BLL does not imply
higher accuracy, recall, or AUROC. It also evaluates probabilities under equal
class weighting; it does not establish calibration at a hospital's actual
disease prevalence. Brier and log loss assess overall probability quality,
rather than calibration alone.

## How to compare the tables

The natural-measurement table includes existing missing values but applies no
additional deletion. Every displayed score is the mean of four separately
computed hospital scores, with equal hospital weight despite different sample
sizes. F1 is calculated within each hospital before averaging. Brier is the
ordinary, unweighted score within each hospital. These values are not pooled
patient scores or averages over all deletion policies.

Compare methods within the same task, condition, and evidence track. The README
shows five illustrative methods; the results page and
[CSV export](research/heart_natural_metrics.csv) include all 45 locked methods.
The natural-metric table presents descriptive point estimates, without new
significance tests. Registered BLL intervals and post-outcome analyses retain
their separate labels. The four small, historical cohorts cannot establish
clinical utility or general superiority at future hospitals.
