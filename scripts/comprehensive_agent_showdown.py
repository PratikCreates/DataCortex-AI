import os
import sys
import time
import json
import warnings
warnings.filterwarnings("ignore")

# Ensure project root on path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score

# Baseline & Internal Tools
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
try:
    from tpot import TPOTClassifier
    HAS_TPOT = True
except ImportError:
    HAS_TPOT = False

try:
    from catboost import CatBoostClassifier
except ImportError:
    pass

try:
    from xgboost import XGBClassifier
except ImportError:
    pass
    
try:
    from lightgbm import LGBMClassifier
except ImportError:
    pass

print("===================================================================")
print("  DATACORTEX-AI: COMPREHENSIVE AGENTIC SHOWDOWN & OMEGA MODEL TEST")
print("===================================================================")

# Load a benchmark dataset
df = pd.read_csv("tests/fixtures/credit_benchmark.csv")
target = "Risk"

X = df.drop(columns=[target])
y = df[target]

# Preprocess
for col in X.select_dtypes(include=["object", "category"]).columns:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))
y = pd.Series(LabelEncoder().fit_transform(y.astype(str)))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(SimpleImputer(strategy="median").fit_transform(X_train))
X_test_s = scaler.transform(SimpleImputer(strategy="median").fit_transform(X_test))

scoreboard = {}

# 1. Baseline Scikit Defaults (Simulating early Auto-Sklearn limits)
print("\n[PHASE 1] Base Scikit Execution (Standard AutoML Baseline)")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
t0 = time.time()
rf.fit(X_train_s, y_train)
t1 = time.time()
score_base = accuracy_score(y_test, rf.predict(X_test_s))
scoreboard["Baseline (RandomForest)"] = {"score": score_base, "time": t1-t0}
print(f"  --> Score: {score_base:.4f}  |  Time: {t1-t0:.2f}s")


# 2. TPOT (Competitor Data Science Agent - Genetic/Evolutionary approach)
if HAS_TPOT:
    print("\n[PHASE 2] TPOT Agent (Genetic Programming AutoML)")
    # Keep Generations low for test speed but high enough to show evolution
    tpot = TPOTClassifier(generations=2, population_size=10, random_state=42)
    t0 = time.time()
    tpot.fit(X_train_s, y_train)
    t1 = time.time()
    score_tpot = tpot.score(X_test_s, y_test)
    scoreboard["TPOT (Evolutionary Agent)"] = {"score": score_tpot, "time": t1-t0}
    print(f"  --> Score: {score_tpot:.4f}  |  Time: {t1-t0:.2f}s")
else:
    print("\n[PHASE 2] TPOT Not Installed. Skipping Evolutionary Agent.")

# 3. DataCortex-Omega (The newly created super-model)
print("\n[PHASE 3] DataCortex-Omega Agentic Meta-Stacker (CatBoost + XGBoost + LightGBM + RF)")
# We construct the Omega array dynamically
omega_estimators = [
    ('lgbm', LGBMClassifier(n_estimators=100, verbose=-1, random_state=42)),
    ('xgb', XGBClassifier(n_estimators=100, eval_metric='logloss', random_state=42)),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
]
try:
    omega_estimators.append(('cat', CatBoostClassifier(iterations=100, verbose=0, random_seed=42)))
except NameError:
    pass

omega = StackingClassifier(estimators=omega_estimators, final_estimator=LogisticRegression(), cv=5, n_jobs=-1)
t0 = time.time()
omega.fit(X_train_s, y_train)
t1 = time.time()
score_omega = accuracy_score(y_test, omega.predict(X_test_s))
scoreboard["DataCortex-Omega (SOTA Super-Model)"] = {"score": score_omega, "time": t1-t0}
print(f"  --> Score: {score_omega:.4f}  |  Time: {t1-t0:.2f}s")


# ─── Final Benchmark Visualization Console ───
print("\n\n" + "="*70)
print(f"{'AGENT / ALGORITHM COMPREHENSIVE SHOWDOWN':^70}")
print("="*70)
print(f"  {'Model Framework':<40} | {'Test Accuracy':<15} | {'Training Time'}")
print("-" * 70)

# Sort by score descending
sorted_scores = sorted(scoreboard.items(), key=lambda x: x[1]['score'], reverse=True)
for agent, metrics in sorted_scores:
    winner_flag = " 🏆 (SOTA)" if agent == sorted_scores[0][0] else ""
    print(f"  {agent:<40} | {metrics['score']:<15.4f} | {metrics['time']:.2f}s{winner_flag}")
print("="*70)

target_to_beat = sorted_scores[1][1]['score'] if len(sorted_scores) > 1 else score_base
if sorted_scores[0][0] == "DataCortex-Omega (SOTA Super-Model)":
    improvement = (score_omega - target_to_beat) * 100
    print(f"\n✅ DATACORTEX OMEGA successfully obliterated the competitor agents and baselines!")
    print(f"   Improvement margin over next best agent: +{improvement:.2f}%")
else:
    print(f"\n⚠️ DataCortex Omega ranked #{[x[0] for x in sorted_scores].index('DataCortex-Omega (SOTA Super-Model)')+1}.")
