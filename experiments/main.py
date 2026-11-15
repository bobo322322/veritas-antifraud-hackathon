import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Load data & clean duplicates
df = pd.read_csv("creditcard.csv")
df = df.drop_duplicates()

print(f"Dataset shape: {df.shape}")
print(f"Class distribution:\n{df['Class'].value_counts()}")
print(f"Fraud ratio: {(df['Class'].mean() * 100):.3f}%\n")

# 2. Features (X) and Target (y)
X = df.drop("Class", axis=1)
y = df["Class"]

# 3. Train / Test Split (Stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale Amount and Time (prevents ConvergenceWarning and balances feature scale)
scaler = StandardScaler()
X_train[["Time", "Amount"]] = scaler.fit_transform(X_train[["Time", "Amount"]])
X_test[["Time", "Amount"]] = scaler.transform(X_test[["Time", "Amount"]])

# 4. Baseline Model
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)

print("--- BASELINE ---")
print(classification_report(y_test, baseline.predict(X_test), zero_division=0))

# 5. Logistic Regression (class_weight='balanced')
lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
y_prob_lr = lr.predict_proba(X_test)[:, 1]

print("\n--- LOGISTIC REGRESSION ---")
print(classification_report(y_test, y_pred_lr, zero_division=0))
print(f"Average Precision (AUPRC): {average_precision_score(y_test, y_prob_lr):.4f}")

# 6. Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, )
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]

print("\n--- RANDOM FOREST ---")
print(classification_report(y_test, y_pred_rf, zero_division=0))
print(f"Average Precision (AUPRC): {average_precision_score(y_test, y_prob_rf):.4f}")
