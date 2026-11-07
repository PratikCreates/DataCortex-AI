"""
Tests — CodeValidator (PATENT CLAIM 3)

Validates the static AST analysis layer that blocks dangerous
AI-generated code before it enters the sandbox. These tests
constitute formal evidence for the Multi-Judge Safety claim.
"""
import pytest
from src.core.sandbox_executor import CodeValidator, UnsafeCodeError


class TestCodeValidatorForbiddenImports:
    """Claim 3: Static analysis blocks forbidden modules."""

    def test_blocks_os_import(self):
        with pytest.raises(UnsafeCodeError, match="Forbidden import"):
            CodeValidator.validate("import os\nprint(os.getcwd())")

    def test_blocks_subprocess_import(self):
        with pytest.raises(UnsafeCodeError, match="Forbidden import"):
            CodeValidator.validate("import subprocess\nsubprocess.run(['ls'])")

    def test_blocks_socket_import(self):
        with pytest.raises(UnsafeCodeError, match="Forbidden import"):
            CodeValidator.validate("import socket\nsocket.gethostname()")

    def test_blocks_shutil_import(self):
        with pytest.raises(UnsafeCodeError, match="Forbidden import"):
            CodeValidator.validate("import shutil\nshutil.rmtree('/tmp')")

    def test_blocks_pickle_import(self):
        with pytest.raises(UnsafeCodeError, match="Forbidden import"):
            CodeValidator.validate("import pickle")

    def test_blocks_requests_import(self):
        with pytest.raises(UnsafeCodeError, match="Forbidden import"):
            CodeValidator.validate("import requests")

    def test_blocks_urllib_import(self):
        with pytest.raises(UnsafeCodeError, match="Forbidden import"):
            CodeValidator.validate("from urllib.request import urlopen")

    def test_blocks_os_submodule(self):
        with pytest.raises(UnsafeCodeError):
            CodeValidator.validate("import os.path")

    def test_blocks_pathlib_mutation(self):
        with pytest.raises(UnsafeCodeError):
            CodeValidator.validate("from pathlib import Path\nPath('x').unlink()")


class TestCodeValidatorForbiddenFunctions:
    """Claim 3: Static analysis blocks dangerous builtins."""

    def test_blocks_eval(self):
        with pytest.raises(UnsafeCodeError, match="Forbidden function call"):
            CodeValidator.validate("eval('1+1')")

    def test_blocks_exec(self):
        with pytest.raises(UnsafeCodeError, match="Forbidden function call"):
            CodeValidator.validate("exec('import os')")

    def test_blocks_globals(self):
        with pytest.raises(UnsafeCodeError, match="Forbidden function call"):
            CodeValidator.validate("x = globals()")

    def test_blocks_locals(self):
        with pytest.raises(UnsafeCodeError, match="Forbidden function call"):
            CodeValidator.validate("x = locals()")

    def test_blocks_dunder_import(self):
        with pytest.raises(UnsafeCodeError, match="Forbidden function call"):
            CodeValidator.validate("__import__('os')")

    def test_blocks_dunder_object_traversal(self):
        with pytest.raises(UnsafeCodeError, match="dunder attribute"):
            CodeValidator.validate("().__class__.__mro__")


class TestCodeValidatorSafeCode:
    """Safe data science code must NOT be blocked."""

    def test_allows_pandas(self):
        code = """
import pandas as pd
import numpy as np
df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
print(df.describe())
"""
        CodeValidator.validate(code)  # Should NOT raise

    def test_allows_sklearn(self):
        code = """
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
X = np.random.randn(100, 5)
y = np.random.choice([0, 1], 100)
X_train, X_test, y_train, y_test = train_test_split(X, y)
clf = RandomForestClassifier().fit(X_train, y_train)
print(clf.score(X_test, y_test))
"""
        CodeValidator.validate(code)  # Should NOT raise

    def test_allows_json_output(self):
        code = """
import json
result = {'score': 0.95, 'model': 'RandomForest'}
print(json.dumps(result))
"""
        CodeValidator.validate(code)  # Should NOT raise

    def test_allows_math_operations(self):
        code = """
import math
import statistics
data = [1, 2, 3, 4, 5]
print(math.sqrt(sum(data)))
print(statistics.mean(data))
"""
        CodeValidator.validate(code)


class TestCodeValidatorSyntaxHandling:
    """Malformed code should raise UnsafeCodeError."""

    def test_raises_on_syntax_error(self):
        with pytest.raises(UnsafeCodeError, match="Syntax error"):
            CodeValidator.validate("def broken: pass (((")

    def test_handles_empty_code(self):
        """Empty code is valid (nothing dangerous)."""
        CodeValidator.validate("")  # Should NOT raise

    def test_multiline_safe_code(self):
        code = "\n".join([f"x_{i} = {i} * 2" for i in range(50)])
        CodeValidator.validate(code)  # Should NOT raise
