import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix,
                              classification_report, accuracy_score, f1_score)

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 130
CHARTS = "/home/claude/health_project/charts"

df = pd.read_csv("/home/claude/health_project/data/patient_records.csv")
X = df.drop(columns=["patient_id", "diabetes"])
y = df["diabetes"]

num_features = ["age", "bmi", "systolic_bp", "diastolic_bp", "cholesterol", "glucose",
                 "smoker", "family_history", "physical_activity_hrs_wk", "sleep_hours"]
cat_features = ["sex", "alcohol_use"]

preprocess = ColumnTransformer([
    ("num", Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]), num_features),
    ("cat", Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]), cat_features),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=5,
        class_weight="balanced", random_state=42
    ),
}

results = {}
roc_data = {}

for name, clf in models.items():
    pipe = Pipeline([("prep", preprocess), ("clf", clf)])
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = pipe.predict(X_test)

    auc = roc_auc_score(y_test, proba)
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    fpr, tpr, _ = roc_curve(y_test, proba)

    results[name] = {"pipe": pipe, "auc": auc, "acc": acc, "f1": f1,
                      "pred": pred, "proba": proba,
                      "report": classification_report(y_test, pred, output_dict=True)}
    roc_data[name] = (fpr, tpr, auc)
    print(f"\n== {name} ==")
    print(f"Accuracy: {acc:.3f}  |  ROC-AUC: {auc:.3f}  |  F1: {f1:.3f}")
    print(classification_report(y_test, pred))

best_name = max(results, key=lambda n: results[n]["auc"])
print(f"\nBest model by ROC-AUC: {best_name}")

# --- ROC curve comparison ---
fig, ax = plt.subplots(figsize=(6.5, 6))
for name, (fpr, tpr, auc) in roc_data.items():
    ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", linewidth=2)
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve Comparison")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{CHARTS}/05_roc_curves.png")
plt.close()

# --- Confusion matrix for best model ---
cm = confusion_matrix(y_test, results[best_name]["pred"])
fig, ax = plt.subplots(figsize=(5.5, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["No Diabetes", "Diabetes"],
            yticklabels=["No Diabetes", "Diabetes"])
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title(f"Confusion Matrix — {best_name}")
plt.tight_layout()
plt.savefig(f"{CHARTS}/06_confusion_matrix.png")
plt.close()

# --- Feature importance (Random Forest) ---
rf_pipe = results["Random Forest"]["pipe"]
ohe = rf_pipe.named_steps["prep"].named_transformers_["cat"].named_steps["onehot"]
cat_names = list(ohe.get_feature_names_out(cat_features))
feature_names = num_features + cat_names
importances = rf_pipe.named_steps["clf"].feature_importances_
imp_series = pd.Series(importances, index=feature_names).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(8, 6))
imp_series.plot(kind="barh", ax=ax, color="#55A868")
ax.set_title("Feature Importance — Random Forest")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig(f"{CHARTS}/07_feature_importance.png")
plt.close()

# Save summary metrics for the report
summary = {
    name: {"accuracy": r["acc"], "roc_auc": r["auc"], "f1": r["f1"]}
    for name, r in results.items()
}
summary["best_model"] = best_name
summary["top_features"] = imp_series.sort_values(ascending=False).head(5).to_dict()
with open("/home/claude/health_project/data/model_summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=float)

print("\nSaved charts + model_summary.json")
