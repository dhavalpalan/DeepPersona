import pandas as pd
import numpy as np
import joblib
import json
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

df = pd.read_csv("sessions.csv")
feature_columns = [
    "session_duration_min", "action_count", "click_interval_ms", 
    "scroll_variance", "skip_rate", "mouse_movement_score", 
    "typing_rhythm_variance", "browser_headless"
]

X = df[feature_columns]
y = df["is_bot"]

X_train, X_test,  y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)  
print(f"Training rows: {len(X_train)}")
print(f"Testing rows: {len(X_test)}")

model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
model.fit(X_train,y_train)
print("Model Trained.")

y_pred = model.predict(X_test)
print("\nAccuracy: ",accuracy_score(y_test, y_pred))
print("\nClassification Report: ")
print(classification_report(y_test,y_pred,target_names=["Human","Bot"]))
print("\nConfusion Matrix: ")
print(confusion_matrix(y_test,y_pred))

joblib.dump(model, "model.pkl")
print("\nModel saved as model.pkl")

importances = model.feature_importances_.tolist()

# ── Unsupervised Anomaly Detection (Isolation Forest) ──
# Trained WITHOUT the is_bot label. This acts as an independent cross-check
# on the supervised RandomForest: if IsolationForest flags a session as an
# outlier but RandomForest scores it as low-risk (or vice versa), that
# disagreement is surfaced as "Needs Review" — a proxy for novel/zero-day
# bot behavior the supervised model was never trained on.
iso_forest = IsolationForest(
    n_estimators=150,
    contamination=0.25,   # roughly matches the ~30% bot ratio in the dataset
    random_state=42
)
iso_forest.fit(X_train)

# Evaluate how well the unsupervised model's outlier calls line up with
# the ground-truth bot label (for reporting purposes only — it never sees y).
iso_pred_raw = iso_forest.predict(X_test)          # 1 = normal, -1 = anomaly
iso_pred_bot = (iso_pred_raw == -1).astype(int)    # anomaly -> "bot" guess
iso_accuracy = round(accuracy_score(y_test, iso_pred_bot), 4)
print("\nIsolation Forest (unsupervised) agreement with true labels:")
print("Accuracy vs ground truth:", iso_accuracy)
print(classification_report(y_test, iso_pred_bot, target_names=["Human", "Bot"]))

joblib.dump(iso_forest, "anomaly_model.pkl")
print("Anomaly model saved as anomaly_model.pkl")

metrics = {
    "accuracy": round(accuracy_score(y_test,y_pred),4),
    "feature_names": feature_columns,
    "feature_importances": importances,
    "confusion_matrix": confusion_matrix(y_test,y_pred).tolist(),
    "anomaly_model": {
        "type": "IsolationForest",
        "contamination": 0.25,
        "agreement_with_labels": iso_accuracy
    }
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("Metrics saved as metrics.json")