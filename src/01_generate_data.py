"""
Generate a realistic synthetic patient dataset for diabetes risk prediction.
Mimics the structure/complexity of real clinical datasets (e.g. Pima Indians
Diabetes-style features) but is fully synthetic with a known, controllable
data-generating process so it's safe to use for a learning project.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 2000

# --- Base demographic / lifestyle features ---
age = rng.integers(21, 81, N)
sex = rng.choice(["Male", "Female"], N, p=[0.48, 0.52])

# BMI: age slightly raises average BMI
bmi = rng.normal(26 + (age - 45) * 0.03, 4.5, N).clip(15, 55)

# Physical activity (hours/week) - inversely related to BMI a bit
physical_activity = (rng.normal(4.5, 2.2, N) - (bmi - 26) * 0.08).clip(0, 15)

smoker = rng.choice([0, 1], N, p=[0.78, 0.22])
family_history = rng.choice([0, 1], N, p=[0.72, 0.28])
alcohol_use = rng.choice(["None", "Moderate", "Heavy"], N, p=[0.45, 0.42, 0.13])

# Blood pressure - rises with age and BMI
systolic_bp = rng.normal(105 + age * 0.35 + (bmi - 25) * 0.9, 9, N).clip(85, 200)
diastolic_bp = rng.normal(65 + age * 0.15 + (bmi - 25) * 0.5, 7, N).clip(55, 130)

# Cholesterol - rises with age, BMI, smoking
cholesterol = rng.normal(
    150 + age * 0.6 + (bmi - 25) * 1.4 + smoker * 12, 25, N
).clip(100, 320)

# Glucose - key diabetes signal; rises with age, BMI, family history, falls with activity
glucose = rng.normal(
    85
    + age * 0.35
    + (bmi - 25) * 1.6
    + family_history * 10
    - physical_activity * 0.9
    + smoker * 3,
    15,
    N,
).clip(60, 260)

sleep_hours = rng.normal(7 - (bmi - 25) * 0.02, 1.1, N).clip(3, 11)

# --- Target: diabetes risk via logistic function of key risk factors ---
z = (
    -13.2
    + 0.045 * age
    + 0.09 * bmi
    + 0.045 * glucose
    + 0.9 * family_history
    + 0.55 * smoker
    - 0.12 * physical_activity
    + 0.01 * cholesterol
)
prob = 1 / (1 + np.exp(-z))
diabetes = rng.binomial(1, prob)

df = pd.DataFrame({
    "patient_id": [f"P{100000+i}" for i in range(N)],
    "age": age,
    "sex": sex,
    "bmi": bmi.round(1),
    "systolic_bp": systolic_bp.round(0).astype(int),
    "diastolic_bp": diastolic_bp.round(0).astype(int),
    "cholesterol": cholesterol.round(0).astype(int),
    "glucose": glucose.round(0).astype(int),
    "smoker": smoker,
    "family_history": family_history,
    "physical_activity_hrs_wk": physical_activity.round(1),
    "alcohol_use": alcohol_use,
    "sleep_hours": sleep_hours.round(1),
    "diabetes": diabetes,
})

# Introduce realistic missingness (real clinical data is never fully clean)
for col, frac in [("cholesterol", 0.04), ("diastolic_bp", 0.03), ("sleep_hours", 0.05)]:
    idx = rng.choice(N, size=int(N * frac), replace=False)
    df.loc[idx, col] = np.nan

df.to_csv("/home/claude/health_project/data/patient_records.csv", index=False)
print(df.shape)
print(df["diabetes"].value_counts(normalize=True))
print(df.isna().sum())
print(df.head())
