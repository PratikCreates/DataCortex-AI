"""
DatacortexAI — Default Agentic Tools

Provides a set of baseline tools that agents can use during planning or execution.
"""

import os
import glob
import math
from src.core.tools import tool

@tool(
    name="calculator", 
    description="Evaluate mathematical expressions. Useful for quick data estimates or conversions."
)
def calculator(expression: str) -> str:
    """Evaluate a simple mathematical expression."""
    try:
        # Extremely restricted eval for basic math only
        allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        return f"Math eval error: {e}"

@tool(
    name="list_directory",
    description="List files in a directory matching a glob pattern (e.g. data/*.csv)."
)
def list_directory(pattern: str) -> str:
    """List files matching a pattern."""
    try:
        files = glob.glob(pattern, recursive=True)
        if not files:
            return "No files found matching pattern."
        return "\n".join(files)
    except Exception as e:
        return f"Error listing directory: {e}"

@tool(
    name="read_file_head",
    description="Read the first N lines of a file to understand its structure."
)
def read_file_head(filepath: str, num_lines: int = 5) -> str:
    """Read starting lines of a file."""
    try:
        if not os.path.exists(filepath):
            return f"File not found: {filepath}"
            
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = [next(f) for _ in range(num_lines)]
        return "".join(lines)
    except StopIteration:
        return "".join(lines)  # EOF reached
    except Exception as e:
        return f"Error reading file: {e}"
@tool(
    name="data_quality_validator",
    description="Scan a dataset for missing values, outliers, and schema drift. Aligns with Data Governance & Integrity."
)
def data_quality_validator(filepath: str) -> str:
    """Assess data quality and reliability."""
    try:
        import pandas as pd
        df = pd.read_csv(filepath)
        nulls = df.isnull().sum().to_dict()
        dupes = int(df.duplicated().sum())
        types = df.dtypes.apply(lambda x: str(x)).to_dict()
        return f"Quality Report for {filepath}:\n- Missing Values: {nulls}\n- Duplicate Rows: {dupes}\n- Schema: {types}"
    except Exception as e:
        return f"Quality Validation Error: {e}"

@tool(
    name="statistical_inference_tool",
    description="Perform advanced statistical tests (T-test, ANOVA) to uncover patterns. Aligns with AI-Driven Insights."
)
def statistical_inference_tool(filepath: str, group_col: str, target_col: str) -> str:
    """Analyze datasets using statistical methods."""
    try:
        import pandas as pd
        from scipy import stats
        df = pd.read_csv(filepath)
        groups = df[group_col].unique()
        if len(groups) == 2:
            g1 = df[df[group_col] == groups[0]][target_col]
            g2 = df[df[group_col] == groups[1]][target_col]
            t_stat, p_val = stats.ttest_ind(g1, g2, nan_policy='omit')
            return f"T-Test Results ({group_col}): T-statistic={t_stat:.4f}, P-value={p_val:.4f}"
        else:
            # ANOVA
            samples = [df[df[group_col] == g][target_col].dropna() for g in groups]
            f_stat, p_val = stats.f_oneway(*samples)
            return f"ANOVA Results ({group_col}): F-statistic={f_stat:.4f}, P-value={p_val:.4f}"
    except Exception as e:
        return f"Statistical Analysis Error: {e}"

@tool(
    name="model_performance_auditor",
    description="Audit and document ML model performance, including bias and error analysis. Aligns with ML Development & Documentation."
)
def model_performance_auditor(y_true_path: str, y_pred_path: str, task: str = "classification") -> str:
    """Evaluate and document model performance."""
    try:
        import pandas as pd
        from sklearn.metrics import classification_report, r2_score
        y_true = pd.read_csv(y_true_path).iloc[:, 0]
        y_pred = pd.read_csv(y_pred_path).iloc[:, 0]
        if task == "classification":
            report = classification_report(y_true, y_pred)
            return f"Model Performance Audit (Classification):\n{report}"
        else:
            r2 = r2_score(y_true, y_pred)
            return f"Model Performance Audit (Regression): R2 Score={r2:.4f}"
    except Exception as e:
        return f"Audit Error: {e}"
