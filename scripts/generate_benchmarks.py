import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression
import os

def generate_datasets():
    output_dir = "tests/fixtures"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Regession: Synthetic Housing-like
    logger_msg("Generating Housing Dataset...")
    X, y = make_regression(n_samples=500, n_features=10, noise=0.1, random_state=42)
    cols = [f"feature_{i}" for i in range(10)]
    df_housing = pd.DataFrame(X, columns=cols)
    df_housing["MedianHouseValue"] = y
    df_housing.to_csv(os.path.join(output_dir, "housing_benchmark.csv"), index=False)

    # 2. Classification: Imbalanced Churn
    logger_msg("Generating Churn Dataset...")
    X, y = make_classification(n_samples=1000, n_features=12, n_informative=8, weights=[0.85, 0.15], random_state=42)
    cols = [f"attr_{i}" for i in range(12)]
    df_churn = pd.DataFrame(X, columns=cols)
    df_churn["Churn"] = y
    df_churn.to_csv(os.path.join(output_dir, "churn_benchmark.csv"), index=False)

    # 3. Categorical: Synthetic Credit Risk
    logger_msg("Generating Credit Dataset...")
    n_samples = 800
    df_credit = pd.DataFrame({
        "Age": np.random.randint(18, 70, n_samples),
        "Income": np.random.randint(20000, 150000, n_samples),
        "Education": np.random.choice(["HighSchool", "Bachelors", "Masters", "PhD"], n_samples),
        "Employment": np.random.choice(["FullTime", "PartTime", "SelfEmployed", "Unemployed"], n_samples),
        "LoanAmount": np.random.randint(1000, 50000, n_samples),
        "CreditScore": np.random.randint(300, 850, n_samples)
    })
    # Target based on some logic
    df_credit["Risk"] = ((df_credit["Income"] / (df_credit["LoanAmount"] + 1) < 0.5) | (df_credit["CreditScore"] < 600)).astype(int)
    # Add noise
    mask = np.random.random(n_samples) < 0.1
    df_credit.loc[mask, "Risk"] = 1 - df_credit.loc[mask, "Risk"]
    df_credit.to_csv(os.path.join(output_dir, "credit_benchmark.csv"), index=False)

    # 4. "Hard" Dataset: Synthetic Adult Income (Noisy & Complex)
    logger_msg("Generating Hard Dataset...")
    n_samples = 1500
    df_hard = pd.DataFrame({
        "Age": np.random.randint(17, 90, n_samples),
        "EducationNum": np.random.randint(1, 16, n_samples),
        "WorkClass": np.random.choice(["Private", "Self-emp", "Gov", "Never-worked", "Missing"], n_samples),
        "CapitalGain": np.random.choice([0, 1000, 5000, 20000, 99999], n_samples, p=[0.8, 0.1, 0.05, 0.03, 0.02]),
        "HoursPerWeek": np.random.randint(1, 99, n_samples),
        "Country": np.random.choice(["US", "UK", "Canada", "Germany", "India", "Unknown"], n_samples)
    })
    
    # Complex non-linear target
    logic = (df_hard["EducationNum"] > 12) & (df_hard["Age"] > 30) & (df_hard["HoursPerWeek"] > 35)
    logic |= (df_hard["CapitalGain"] > 5000)
    df_hard["HighIncome"] = logic.astype(int)
    
    # Inject heavy noise (25% random flipping)
    mask = np.random.random(n_samples) < 0.25
    df_hard.loc[mask, "HighIncome"] = 1 - df_hard.loc[mask, "HighIncome"]
    
    # Inject missing values
    df_hard.loc[np.random.random(n_samples) < 0.1, "Age"] = np.nan
    df_hard.loc[np.random.random(n_samples) < 0.1, "WorkClass"] = "Missing"
    
    df_hard.to_csv(os.path.join(output_dir, "hard_benchmark.csv"), index=False)

def logger_msg(msg):
    print(f"[DATA GEN] {msg}")

if __name__ == "__main__":
    generate_datasets()
