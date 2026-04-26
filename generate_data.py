"""
Merit-Checker: Synthetic HR Dataset Generator
Generates a realistic hiring dataset with embedded bias patterns for testing.
"""

import pandas as pd
import numpy as np

def generate_hr_dataset(n=500, seed=42):
    np.random.seed(seed)

    # ── Legitimate Features (should be the ONLY thing that matters) ──────────
    years_exp    = np.random.randint(0, 25, n)
    test_score   = np.round(np.random.normal(65, 15, n).clip(20, 100), 1)
    interview    = np.round(np.random.normal(60, 18, n).clip(10, 100), 1)
    edu_level    = np.random.choice(
        ["High School", "Bachelor's", "Master's", "PhD"],
        n, p=[0.15, 0.50, 0.28, 0.07]
    )
    edu_map      = {"High School": 0, "Bachelor's": 1, "Master's": 2, "PhD": 3}
    edu_num      = np.array([edu_map[e] for e in edu_level])

    # Merit score (0–100): pure composite of legitimate features
    merit = (
        0.35 * test_score +
        0.30 * interview +
        0.20 * np.clip(years_exp * 4, 0, 100) +
        0.15 * (edu_num / 3 * 100)
    )

    # ── Protected Attributes ─────────────────────────────────────────────────
    gender = np.random.choice(["Male", "Female", "Non-binary"], n,
                               p=[0.52, 0.44, 0.04])
    race   = np.random.choice(
        ["White", "Asian", "Black", "Hispanic", "Other"],
        n, p=[0.45, 0.20, 0.15, 0.15, 0.05]
    )
    age    = np.random.randint(22, 63, n)

    # ── Hiring Decision with EMBEDDED BIAS ───────────────────────────────────
    # Base probability from merit
    base_prob = 1 / (1 + np.exp(-(merit - 58) / 12))

    # Bias 1: Age bias — penalise candidates over 45
    age_penalty = np.where(age > 45, -0.18, 0.0)

    # Bias 2: Gender bias — female candidates slightly penalised
    gender_penalty = np.where(gender == "Female", -0.10,
                     np.where(gender == "Non-binary", -0.14, 0.0))

    # Bias 3: Racial bias — Black and Hispanic candidates penalised
    race_penalty = np.where(race == "Black",    -0.15,
                   np.where(race == "Hispanic", -0.10, 0.0))

    hire_prob = np.clip(base_prob + age_penalty + gender_penalty + race_penalty,
                        0.05, 0.95)
    hired = (np.random.rand(n) < hire_prob).astype(int)

    df = pd.DataFrame({
        # Legitimate features
        "Years_Experience":  years_exp,
        "Education_Level":   edu_level,
        "Technical_Score":   test_score,
        "Interview_Score":   interview,
        "Merit_Score":       np.round(merit, 1),
        # Protected attributes
        "Age":               age,
        "Gender":            gender,
        "Race":              race,
        # Outcome
        "Hired":             hired,
    })

    return df


if __name__ == "__main__":
    df = generate_hr_dataset(500)
    df.to_csv("hr_hiring_data.csv", index=False)
    print(f"Dataset saved: {len(df)} rows")
    print(df["Hired"].value_counts())
    print("\nHire rate by Gender:")
    print(df.groupby("Gender")["Hired"].mean().round(3))
    print("\nHire rate by Race:")
    print(df.groupby("Race")["Hired"].mean().round(3))
    print("\nHire rate by Age group:")
    df["Age_Group"] = pd.cut(df["Age"], bins=[21,30,40,50,65],
                              labels=["22-30","31-40","41-50","51-62"])
    print(df.groupby("Age_Group", observed=True)["Hired"].mean().round(3))
