# Veritas Anti-Fraud

**Cost-sensitive machine learning for credit card fraud detection**  
Hackathon prototype · Track: **Finance**

> **Disclaimer:** This project is an experimental prototype built for educational and
> hackathon evaluation purposes. It does not constitute financial or investment advice.

---

## Overview

An automated transaction screening prototype that minimizes actual monetary losses. Instead of using a default `0.5` decision threshold on heavily imbalanced data (0.167% fraud), we optimize a business cost function (€5 manual check vs. full loss of missed fraud) and test for Out-of-Time fraud drift (Day 1 vs. Day 2).


- **Dataset:** [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (283,726 deduplicated transactions, 473 frauds)

## Key Results

| Model | Precision | Recall | F1 | PR-AUC |
| --- | --- | --- | --- | --- |
| Dummy baseline | 0.0000 | 0.0000 | 0.0000 | 0.0017 |
| Logistic Regression | 0.0507 | 0.8737 | 0.0958 | 0.6918 |
| **Random Forest** | **0.9714** | **0.7158** | **0.8242** | **0.8109** |

- **Business impact:** Optimal threshold `0.07` (vs default `0.50`) reduced bank loss by **33.3% (€1,961 saved)** and caught 6 extra frauds on unseen test data.
- **OOT finding:** Precision drops on Day 2 (~40%), demonstrating fraudster tactic drift and the need for frequent retraining.

## Repository structure

```
.
├── notebook/
│   └── fraud_detection.ipynb    # final notebook (EDA → models → cost optimization → OOT test)
├── src/
│   └── fraud_pipeline.py        # full pipeline as a standalone script
├── experiments/                 # early research artifacts
│   ├── main.py                  # baseline models
│   ├── time_analysis.py         # Day 1 vs Day 2 experiment
│   └── friend/                  # threshold sweep & grid search
├── requirements.txt
└── .gitignore
```

## Quickstart

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Download [creditcard.csv](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (~150 MB) and place it in the project root.
3. Run the standalone pipeline:
   ```bash
   python src/fraud_pipeline.py
   ```
   Or open `notebook/fraud_detection.ipynb` in Jupyter / Google Colab.

> **Note on reproducibility.** All reported figures come from the notebook run in Google Colab
> (scikit-learn 1.6). Random Forest tree building is slightly version-dependent, so re-running
> on a different scikit-learn version can move the optimal threshold and the exact euro amounts
> a little. The scores stay in the same range (PR-AUC ≈ 0.81, ~33% loss reduction); the script
> prints the library versions it used so any difference is easy to explain.

## What we'd do next

1. Retrain daily on newly confirmed fraud cases so the model keeps up with changing scam tactics.
2. Make the €5 penalty dynamic — scaled by transaction size and customer history — so VIP
   customers get a friction-free verification path.

## Sources

* **Dataset:** Machine Learning Group – ULB,
  [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
* **Libraries:** [scikit-learn](https://scikit-learn.org/), [pandas](https://pandas.pydata.org/),
  [NumPy](https://numpy.org/), [matplotlib](https://matplotlib.org/)
* **Tooling:** [Zed](https://zed.dev/) for development and code review
