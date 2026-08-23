import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 130

df = pd.read_csv("/home/claude/health_project/data/patient_records.csv")
CHARTS = "/home/claude/health_project/charts"

# 1. Target balance
fig, ax = plt.subplots(figsize=(5, 4))
counts = df["diabetes"].value_counts().sort_index()
labels = ["No Diabetes", "Diabetes"]
colors = ["#4C72B0", "#DD8452"]
ax.bar(labels, counts.values, color=colors)
for i, v in enumerate(counts.values):
    ax.text(i, v + 15, f"{v} ({v/len(df):.1%})", ha="center", fontweight="bold")
ax.set_title("Target Class Balance: Diabetes Diagnosis")
ax.set_ylabel("Number of Patients")
plt.tight_layout()
plt.savefig(f"{CHARTS}/01_class_balance.png")
plt.close()

# 2. Key feature distributions by diabetes status
features = ["age", "bmi", "glucose", "systolic_bp", "cholesterol", "physical_activity_hrs_wk"]
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, feat in zip(axes.flat, features):
    sns.kdeplot(data=df, x=feat, hue="diabetes", fill=True, common_norm=False,
                palette=colors, ax=ax, alpha=0.5, legend=(feat == "age"))
    ax.set_title(feat.replace("_", " ").title())
if axes.flat[0].get_legend():
    axes.flat[0].get_legend().set_title("Diabetes")
plt.suptitle("Feature Distributions by Diabetes Status", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{CHARTS}/02_feature_distributions.png")
plt.close()

# 3. Correlation heatmap (numeric features)
numeric_cols = ["age", "bmi", "systolic_bp", "diastolic_bp", "cholesterol",
                 "glucose", "smoker", "family_history", "physical_activity_hrs_wk",
                 "sleep_hours", "diabetes"]
corr = df[numeric_cols].corr()
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax,
            square=True, cbar_kws={"shrink": 0.8})
ax.set_title("Correlation Matrix of Patient Features")
plt.tight_layout()
plt.savefig(f"{CHARTS}/03_correlation_heatmap.png")
plt.close()

# 4. BMI vs Glucose scatter colored by outcome
fig, ax = plt.subplots(figsize=(7, 6))
for val, color, label in [(0, colors[0], "No Diabetes"), (1, colors[1], "Diabetes")]:
    sub = df[df["diabetes"] == val]
    ax.scatter(sub["bmi"], sub["glucose"], s=12, alpha=0.5, color=color, label=label)
ax.set_xlabel("BMI")
ax.set_ylabel("Glucose (mg/dL)")
ax.set_title("BMI vs Glucose, Colored by Diabetes Outcome")
ax.legend()
plt.tight_layout()
plt.savefig(f"{CHARTS}/04_bmi_vs_glucose.png")
plt.close()

print("EDA charts saved.")
print("\nTop correlations with diabetes:")
print(corr["diabetes"].drop("diabetes").sort_values(key=abs, ascending=False))
