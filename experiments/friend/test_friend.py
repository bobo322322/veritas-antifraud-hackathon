import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    PrecisionRecallDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

df = pd.read_csv("creditcard.csv")


X = df.drop('Class', axis=1)
y = df['Class']

# Сначала отделяем 20% для Test
X_temp, X_test, y_temp, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Из оставшихся 80% выделяем 25% для Validation
X_train, X_val, y_train, y_val = train_test_split(
    X_temp,
    y_temp,
    test_size=0.25,
    random_state=42,
    stratify=y_temp
)
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)


# param_grid = {
#     "n_estimators": [100, 200],
#     "max_depth": [None, 15],
#     "min_samples_leaf": [1, 2],
#     "class_weight": [None, "balanced"]
# }
# cv = StratifiedKFold(
#     n_splits=3,
#     shuffle=True,
#     random_state=42
# )
#
# grid_search = GridSearchCV(
#     estimator=rf,
#     param_grid=param_grid,
#     scoring="average_precision",
#     cv=cv,
#     n_jobs=-1,
#     verbose=2
# )
#
# grid_search.fit(X_train, y_train)
#
# best_rf = grid_search.best_estimator_
#
#
# print(f" Лучшие настройки: {best_rf.get_params()}")
#
#
# print("Best parameters:", grid_search.best_params_)
# print("Best CV AP:", grid_search.best_score_)
#
# results = pd.DataFrame(grid_search.cv_results_)
#
# results = results.sort_values(
#     by="mean_test_score",
#     ascending=False
# )
#
# print(
#     results[
#         ["params", "mean_test_score", "std_test_score"]
#     ].head(10)
# )

# rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
# rf.fit(X_train, y_train)
#
# y_pred_rf = rf.predict(X_test)
# y_prob_rf = rf.predict_proba(X_test)[:, 1]
#
# print("--- RANDOM FOREST ---")
# print(classification_report(y_test, y_pred_rf, zero_division=0))
# print(f"Average Precision (AUPRC): {average_precision_score(y_test, y_prob_rf):.4f}")



features_all = list(X_train.columns)

features_without_low = [
    col for col in features_all
    if col not in [
        "V19", "V21", "V23", "V13",
        "V6", "V25", "V24", "V22"
    ]
]

#
# best_rf = RandomForestClassifier(
#     n_estimators=200,
#     random_state=42,
#     n_jobs=-1,
#     max_depth=None,
#     min_samples_leaf=2,
#     class_weight="balanced",
# )
#
# best_rf.fit(X_train[features_without_low], y_train)
#
# y_prob_test = best_rf.predict_proba(
#     X_test[features_without_low]
# )[:, 1]
#
# test_ap = average_precision_score(y_test, y_prob_test)

# print(f"Test AP: {test_ap:.6f}")

pr_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    max_depth=None,
    min_samples_leaf=2,
    class_weight="balanced"
)

pr_model.fit(X_train[features_without_low], y_train)

y_prob_val = pr_model.predict_proba(
    X_val[features_without_low]
)[:, 1]

precision, recall, thresholds = precision_recall_curve(
    y_val,
    y_prob_val
)

ap = average_precision_score(y_val, y_prob_val)

plt.figure(figsize=(9, 6))

plt.plot(recall, precision, label=f"Random Forest (AP = {ap:.4f})")

# Доля мошеннических транзакций — базовый уровень Precision
baseline = y_val.mean()
plt.axhline(
    y=baseline,
    linestyle="--",
    label=f"Baseline = {baseline:.4f}"
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.grid()
plt.show()


# perm_importance = permutation_importance(
#     best_rf,
#     X_val,
#     y_val,
#     scoring="average_precision",
#     n_repeats=5,
#     random_state=42,
#     n_jobs=-1
# )
#
# importance_df = pd.DataFrame({
#     "Feature": X_val.columns,
#     "Importance": perm_importance.importances_mean,
#     "Std": perm_importance.importances_std
# })
#
# importance_df = importance_df.sort_values(
#     by="Importance",
#     ascending=False
# )
#
# print(importance_df)
#
#
#
#
# features_all = list(X_train.columns)
#
# features_without_negative = [
#     col for col in features_all
#     if col not in ["V21", "V23"]
# ]
#
# features_without_low = [
#     col for col in features_all
#     if col not in [
#         "V19", "V21", "V23", "V13",
#         "V6", "V25", "V24", "V22"
#     ]
# ]
#
#
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import average_precision_score
#
# feature_sets = {
#     "All features": features_all,
#     "Without negative": features_without_negative,
#     "Without low importance": features_without_low
# }
#
# results = []
# random_states = [10, 20, 30, 40, 42, 50, 60, 70, 80, 100]
#
# for name, features in feature_sets.items():
#     for state in random_states:
#         best_rf = RandomForestClassifier(
#             n_estimators=200,
#             n_jobs=-1,
#             max_depth=None,
#             min_samples_leaf=2,
#             class_weight="balanced",
#             random_state = state
#         )
#         best_rf.fit(X_train[features], y_train)
#
#         y_prob = best_rf.predict_proba(X_val[features])[:, 1]
#
#         ap = average_precision_score(y_val, y_prob)
#
#         results.append({
#             "Experiment": name,
#             "Random_state": state,
#             "Features": len(features),
#             "AP": ap
#         })
#
# results_df = pd.DataFrame(results)
#
# print(results_df.sort_values("AP", ascending=False))
#
# summary = results_df.groupby("Experiment")["AP"].agg(
#     ["mean", "std", "min", "max"]
# )
#
# print(summary.sort_values("mean", ascending=False))
