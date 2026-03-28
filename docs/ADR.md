# Architecture Decision Records

## ADR-001: Evaluation Design

**Status:** Accepted | **Date:** 2026-03

Chronos produces probabilistic forecasts (quantiles), LightGBM produces point forecasts. For a fair comparison, Chronos quantiles are converted to point forecasts via the **median** (0.5 quantile) — robust to extreme tails and standard in Chronos literature.

Primary metric is **RMSLE**, consistent with the Kaggle competition and the right-skewed sales distribution. Secondary metrics: RMSE, MAE, Asymmetric Loss (understock weight 10x, reflecting stockout > overstock cost in retail). Both models receive identical external features, making information access equal.

---

## ADR-002: Dataset Subset Selection

**Status:** Accepted | **Date:** 2026-03

**Families:** 19 of 33 families included (zero-rate < 30%). EDA showed a natural break at ~35% — families above this threshold are sparse/niche categories where neither model produces meaningful forecasts.

**Stores:** Top 14 stores by volume (~50% of total sales). Family-mix variance across stores is low (max std 0.055), making the top stores representative of the full population.

**Oil price excluded:** Raw correlation of -0.528 with sales is a time-trend confounding effect, confirmed by year-stratified analysis. Including it would introduce time-trend leakage.

---

## ADR-003: Direct vs. Recursive Forecasting

**Status:** Accepted | **Date:** 2026-03

Both approaches were trained and evaluated empirically on the same 15-day validation set.

| Metric | Direct | Recursive (real inference) |
|---|---|---|
| RMSLE | **0.2588** | 0.2594 |
| RMSE | **288.77** | 290.59 |
| MAE | **124.67** | 126.55 |

Direct wins on all fair metrics. Recursive shows measurable error accumulation (+0.0006 RMSLE) over 15 steps despite the short horizon. Direct proceeds to the Chronos benchmark.

---

## ADR-004: Chronos-2 as Foundation Model

**Status:** Accepted | **Date:** 2026-03

Chronos-1 (original plan) is univariate-only and cannot incorporate external covariates. EDA showed `onpromotion` alone drives 24-208% sales lift — making Chronos-1 structurally disadvantaged in any fair benchmark.

Chronos-2 (released October 2025, 120M parameters) supports past and future covariates natively within a single zero-shot architecture. This enables a genuinely fair comparison: both models receive identical external features.

**Result:** Chronos-2 with covariates achieves RMSLE 0.2581 vs. LightGBM 0.2588 — nearly identical on log-scale accuracy. LightGBM retains a clear advantage on RMSE (288 vs. 465), MAE (125 vs. 163), Asymmetric Loss (500 vs. 550), inference speed (~1s CPU vs. 14.8s GPU), and model size (11.2 MB vs. 478 MB).

The benchmark conclusion: given equal information access, Foundation Models match classical ML on relative accuracy (RMSLE) but lose on absolute error metrics and operational cost.
