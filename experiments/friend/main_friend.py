import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    average_precision_score,
    PrecisionRecallDisplay,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    accuracy_score,
)



df = pd.read_csv('creditcard.csv')
# print(f"Форма датасета: {df.shape}")
# print("Распределение классов:")
# print(df['Class'].value_counts())
# print(f"Доля фрода: {(df['Class'].mean() * 100):.3f}%")


X = df.drop('Class', axis=1)
y = df['Class']

X_temp, X_test, y_temp, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp,
    y_temp,
    test_size=0.25,
    random_state=42,
    stratify=y_temp
)


# baseline = DummyClassifier(strategy='most_frequent')
# baseline.fit(X_train, y_train)

# print("--- БЕЙЗЛАЙН ---")
# print(classification_report(y_test, baseline.predict(X_test), zero_division=0))


lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)
y_prob_lr = lr.predict_proba(X_test)[:, 1]

# print("--- ЛОГИСТИЧЕСКАЯ РЕГРЕССИЯ ---")
# print(classification_report(y_test, y_pred_lr, zero_division=0))
# print(f"Average Precision (AUPRC): {average_precision_score(y_test, y_prob_lr):.4f}")


rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
# y_prob_rf = rf.predict_proba(X_test)[:, 1]

y_prob_val = rf.predict_proba(X_val)[:, 1]
ap_val = average_precision_score(y_val, y_prob_val)

print(f"Validation AP: {ap_val:.4f}")

# thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
#
# for threshold in thresholds:
#     y_pred = (y_prob_rf >= threshold).astype(int)
#
#     precision = precision_score(y_test, y_pred, zero_division=0)
#     recall = recall_score(y_test, y_pred, zero_division=0)
#     f1 = f1_score(y_test, y_pred, zero_division=0)
#
#     print(f"Threshold: {threshold}")
#     print(f"Precision: {precision:.4f}")
#     print(f"Recall: {recall:.4f}")
#     print(f"F1: {f1:.4f}")
#     print("-" * 30)
thresholds = np.arange(0.05, 0.96, 0.05)

results = []

for threshold in thresholds:

    y_pred = (y_prob_val >= threshold).astype(int)

    precision = precision_score(
        y_val, y_pred, zero_division=0
    )

    recall = recall_score(
        y_val, y_pred, zero_division=0
    )

    f1 = f1_score(
        y_val, y_pred, zero_division=0
    )

    results.append({
        "Threshold": threshold,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    })

results_df = pd.DataFrame(results)

print(results_df.round(4))

# plt.figure(figsize=(10, 6))
#
# plt.plot(
#     results_df["Threshold"],
#     results_df["Precision"],
#     label="Precision",
#     marker="o"
# )
#
# plt.plot(
#     results_df["Threshold"],
#     results_df["Recall"],
#     label="Recall",
#     marker="o"
# )
#
# plt.xlabel("Threshold")
# plt.ylabel("Score")
# plt.title("Precision and Recall vs Threshold")
#
# plt.legend()
# plt.grid()
# plt.show()


valid_thresholds = results_df[
    results_df["Precision"] >= 0.90
]

best_threshold = valid_thresholds.sort_values(
    by="Recall",
    ascending=False
).iloc[0]["Threshold"]

print(f"Best Threshold: {best_threshold:.2f}")

y_prob_test = rf.predict_proba(X_test)[:, 1]

ap_test = average_precision_score(
    y_test,
    y_prob_test
)

print(f"Test AP: {ap_test:.4f}")

y_pred_test = (
    y_prob_test >= best_threshold
).astype(int)

precision = precision_score(
    y_test,
    y_pred_test,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred_test,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred_test,
    zero_division=0
)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1: {f1:.4f}")



#
#

# print("--- RANDOM FOREST ---")
# print(classification_report(y_test, y_pred_rf, zero_division=0))
# print(f"Average Precision (AUPRC): {average_precision_score(y_test, y_prob_rf):.4f}")


# # --- ТАБЛИЦА СРАВНЕНИЯ МОДЕЛЕЙ ---
# comparison_df = pd.DataFrame({
#     'Метрика': [
#         'Precision (Class 1)',
#         'Recall (Class 1)',
#         'F1-Score (Class 1)',
#         'PR-AUC (AUPRC)',
#         'ROC-AUC',
#         'Accuracy'
#     ],
#     'Logistic Regression': [
#         precision_score(y_test, y_pred_lr, zero_division=0),
#         recall_score(y_test, y_pred_lr, zero_division=0),
#         f1_score(y_test, y_pred_lr, zero_division=0),
#         average_precision_score(y_test, y_prob_lr),
#         roc_auc_score(y_test, y_prob_lr),
#         accuracy_score(y_test, y_pred_lr),
#     ],
#     'Random Forest': [
#         precision_score(y_test, y_pred_rf, zero_division=0),
#         recall_score(y_test, y_pred_rf, zero_division=0),
#         f1_score(y_test, y_pred_rf, zero_division=0),
#         average_precision_score(y_test, y_prob_rf),
#         roc_auc_score(y_test, y_prob_rf),
#         accuracy_score(y_test, y_pred_rf),
#     ]
# })
#
# print("\n" + "=" * 55)
# print("             СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
# print("=" * 55)
# print(comparison_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
# print("=" * 55 + "\n")
#
#
# fig, ax = plt.subplots(figsize=(8, 6))
#
# display = PrecisionRecallDisplay.from_estimator(
#     rf, X_test, y_test, ax=ax, name="Random Forest"
# )
#
# # При желании можно также залить область под построенной линией
# ax.fill_between(display.recall, display.precision, alpha=0.2)
# ax.set_title("Precision-Recall Curve (PrecisionRecallDisplay)")
# ax.grid(alpha=0.3)
#
# plt.show()
