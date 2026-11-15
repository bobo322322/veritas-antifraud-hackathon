import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, classification_report
from sklearn.preprocessing import StandardScaler

# 1. Загрузка данных и удаление дубликатов
df = pd.read_csv("creditcard.csv")
df = df.drop_duplicates()

# 2. Делим датасет на День 1 и День 2 (в сутках 86 400 секунд)
day1 = df[df["Time"] < 86400]
day2 = df[df["Time"] >= 86400]

print(f"Всего транзакций в Day 1: {len(day1)}")
print(f"Всего транзакций в Day 2: {len(day2)}")
print(round(day1['Amount'].sum(), 2), day1['Class'].sum())
print(round(day2['Amount'].sum(), 2), day2['Class'].sum())
print()

day2['Time'] = day2['Time'] - 86400

X_train = day1.drop('Class', axis = 1)
y_train = day1["Class"]
X_test= day2.drop('Class', axis = 1)
y_test = day2['Class']


rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]


print('Day1 vs Day2')
print(classification_report(y_test, y_pred_rf, zero_division=0))
print(f"Average Precision (AUPRC): {average_precision_score(y_test, y_prob_rf):.4f}")
