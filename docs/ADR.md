# ADR-001: Evaluation Design for Foundation Model vs. Classical ML Benchmark

**Status:** Accepted  
**Date:** 2026-03

---

## Context

Two models are being benchmarked on the same test set:

- LightGBM: produces point forecasts directly.
- Amazon Chronos: produces probabilistic forecasts (a distribution over quantiles),
  not point forecasts.

For a fair comparison on standard regression metrics (RMSE, MAE, MAPE, RMSLE),
both models must produce a single predicted value per (date, store_nbr, family).

A second question: which metric is primary for the benchmark? The Kaggle
Store Sales competition uses RMSLE, not RMSE. Sales data is strongly right-skewed,
which makes log-scale metrics more meaningful.

---

## Decisions

### 1. Chronos point forecast conversion: Median

Chronos quantile forecasts are converted to point forecasts using the **median**
(0.5 quantile), not the mean.

```python
# forecast shape: (num_series, num_samples, prediction_length)
point_forecast = torch.quantile(forecast, 0.5, dim=1)
```

Alternatives considered:

| Option | Pros | Cons |
|---|---|---|
| Median (chosen) | Robust to extreme quantiles. Standard in Chronos literature. | None for this use case. |
| Mean of samples | Matches expected value interpretation. | Biased upward by extreme quantile tails. |
| Mode (highest density) | Most likely single outcome. | Complex to compute from samples, rarely used. |

### 2. Primary metric: RMSLE

RMSLE is the primary metric for leaderboard comparison and model selection.
RMSE, MAE, and MAPE are reported as secondary metrics for completeness.

Rationale: The Kaggle competition uses RMSLE. Sales data is right-skewed.
RMSLE penalises under-forecasting more than over-forecasting, which matches
the business cost structure of retail demand planning (stockouts are more
expensive than overstock).

RMSLE formula: `sqrt(mean((log1p(y_pred) - log1p(y_true))^2))`

Note: `log1p` handles zero-sales days (common in this dataset) without
producing -inf.

### 3. Leaderboard anchoring

LightGBM results will be compared against the public Kaggle leaderboard
(https://www.kaggle.com/competitions/store-sales-time-series-forecasting/leaderboard)
to provide an external reference point for model quality. This is not the
primary goal of the benchmark, but it provides a defensible anchor for
interview discussions ("Top X% without ensembling").

---

## Consequences

- LightGBM objective should be `tweedie` or `huber` (appropriate for skewed
  sales data), not the default `mse`. Final choice to be validated in EDA
  based on the sales distribution.
- Both models are evaluated on the same held-out test set with identical code.
- Chronos predictions are computed on Kaggle GPU, exported in the exchange
  schema, and evaluated locally. See `notebooks/kaggle_setup.txt` for schema.