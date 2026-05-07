# Confound Hunter

> An auditing tool that catches what accuracy metrics hide.

## What it does

Confound Hunter is a Python library that takes a trained ML model and a 
train/test split and returns a ranked list of features flagged as potentially 
spurious. Each flagged feature receives a suspicion score, a confounder type 
label, and a full evidence trail showing which detectors fired and why.

**Core question it answers:** Which features in your pipeline are riding noise, 
not signal?

---

## Why it exists

Standard ML evaluation (accuracy, AUC, F1) tells you how well a model performs 
on held-out data — it does not tell you *why*. A model can pass every accuracy 
check while silently relying on features that will fail at deployment: leaky 
features derived from the target, proxy variables that won't exist in production, 
or features that correlate with the target only through a shared time trend.

Confound Hunter surfaces this evidence before deployment.

---

## How it works

Six independent statistical detectors run in parallel. Each returns a suspicion 
score in [0, 1] per feature. A weighted combination produces the final score.

| Detector | What it catches | Weight |
|---|---|---|
| Permutation Stability | Features whose importance collapses on held-out data | 0.20 |
| SHAP Train/Test Consistency | Features the model uses very differently on unseen data | 0.20 |
| Residual Correlation | Features that correlate with training residuals — noise absorption | 0.20 |
| Proxy Confounder (MI) | Features that are stand-ins for other variables, not independent signal | 0.15 |
| Temporal Confounder | Features correlated with the target only through shared time trends | 0.15 |
| Interaction Confounder | Features spuriously correlated with the target only via another feature | 0.10 |

No single test is sufficient. Each detector catches a confounder profile the 
others can miss. The residual correlation test is the most direct: a genuine 
signal feature correlates with model residuals similarly in train and test; a 
spurious one mops up training noise and the gap collapses on held-out data.

---

## Quick start

```python
from confound_hunter.audit import ConfounderAudit
from xgboost import XGBClassifier

model = XGBClassifier().fit(X_train, y_train)

audit = ConfounderAudit(
    model=model,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
)

report = audit.run(threshold=0.20)

for item in report.flagged_features:
    print(item["feature"], item["score"], item["type"])
```

---

## Benchmark

Confound Hunter is validated against a synthetic benchmark engine 
(`ConfoundInjector`) that injects six known confounder types into clean datasets 
and measures detection precision and recall against verifiable ground truth.

**Six injection types:**
- Type A: Spurious Correlation — feature derived from target + noise
- Type B: Leaky Feature — feature partially derived from target (data leakage)
- Type C: Proxy Confounder — near-clone of an existing feature
- Type D: Clean Signal — legitimate predictive feature (false positive control)
- Type E: Temporal Confounder — feature correlated with target via shared time trend
- Type F: Interaction Confounder — two features whose product drives spurious correlation

Benchmark results across 20 independent trials will be reported here once 
the full suite is validated.

**Baselines implemented for comparison:**
- SHAP-only: flags features where mean absolute SHAP on train > 2x test
- Permutation-only: flags features where permutation importance on train > 2x test

---

## Project status

| Component | Status |
|---|---|
| 6 detectors | ✅ Complete |
| Synthetic benchmark engine (6 injection types) | ✅ Complete |
| Benchmark runner (20-trial precision/recall) | 🔄 In progress |
| Baseline comparison (SHAP-only, permutation-only) | ✅ Complete |
| Interactive HTML audit report | 🔄 Planned (Phase 8) |
| CLI interface | 🔄 Planned (Phase 9) |
| Case study notebooks (Home Credit, Numerai, NLP) | 🔄 Planned (Phase 10) |

---

## Repository structure

confound_hunter/
├── audit.py                  # ConfounderAudit — main entry point
├── detectors/
│   ├── permutation.py        # Detector 1: Permutation stability
│   ├── shap_drift.py         # Detector 2: SHAP train/test consistency
│   ├── residual_corr.py      # Detector 3: Residual correlation
│   ├── proxy.py              # Detector 4: Proxy confounder (mutual information)
│   ├── temporal.py           # Detector 5: Temporal confounder
│   └── interaction.py        # Detector 6: Interaction-based confounder
├── report/
│   └── builder.py            # AuditReport class
synthetic/
└── injector.py               # ConfoundInjector — synthetic benchmark engine
benchmarks/
├── run_benchmark.py          # Single-trial + 20-trial benchmark runner
└── baselines.py              # SHAP-only and permutation-only baselines

---

## Limitations

- Computational cost: permutation and interaction detectors are expensive on 
  datasets with 500+ features. Apply to top-suspicion candidates first.
- Model agnosticism: SHAP detection works best with tree-based models. Neural 
  network support requires a different explainer.
- Causal inference scope: Confound Hunter detects potential spuriousness — it 
  does not establish causality. Frame flagged features as candidates for deeper 
  causal investigation, not confirmed confounders.
- Interaction detection: only pairwise interactions are tested at MVP stage.

---

## Tech stack

Python · XGBoost · SHAP · scikit-learn · scipy · pandas · numpy

