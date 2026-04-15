"""
DatacortexAI — Self-Healing Sandboxed Code Executor

Self-Healing Sandboxed Code Execution with Error Reflection Loop

AI-generated Python code is executed inside isolated micro-VM sandboxes.
When execution fails, the error trace is fed back to the generating agent
with the original intent, producing corrected code through a reflection
loop (max 3 retries with exponential backoff).
"""

import asyncio
import logging
import time
import os
import ast
from dataclasses import dataclass, field
from typing import Optional, Callable, Awaitable, Set

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of a sandboxed code execution."""
    success: bool
    stdout: str = ""
    stderr: str = ""
    return_value: Optional[str] = None
    execution_time_ms: int = 0
    code_executed: str = ""
    attempt: int = 1
    sandbox_name: str = ""
    error_reflection_used: bool = False


@dataclass
class SandboxConfig:
    """Configuration for sandbox creation."""
    name: str = ""
    memory_mb: int = 4096
    timeout_seconds: int = 300
    image: str = "python:3.12-slim"
    install_packages: list[str] = field(default_factory=lambda: [
        "pandas", "numpy", "scikit-learn", "matplotlib", "seaborn", "scipy",
    ])


class SandboxExecutor:
    """
    Self-Healing Sandboxed Code Execution with Error Reflection Loop.

    1. Creates an isolated Blaxel sandbox (MicroVM)
    2. Installs data science dependencies
    3. Writes & executes AI-generated Python code
    4. On failure: captures error, sends to LLM for reflection, retries
    5. Returns structured results (stdout, parsed outputs, artifacts)
    """

    def __init__(
        self,
        blaxel_api_key: str,
        workspace_name: str,
        max_retries: int = 3,
        error_reflector: Optional[Callable[[str, str, str], Awaitable[str]]] = None,
        guardian: Optional[Callable[[str], Awaitable[bool]]] = None,
    ):
        self.api_key = blaxel_api_key
        self.workspace = workspace_name
        self.max_retries = max_retries
        self.error_reflector = error_reflector
        self.guardian = guardian
        self._sandbox_instance = None

    async def execute(
        self,
        code: str,
        intent: str = "",
        sandbox_config: Optional[SandboxConfig] = None,
        on_log: Optional[Callable[[str], None]] = None,
    ) -> ExecutionResult:
        """
        Execute Python code in an isolated sandbox with self-healing.

        Args:
            code: Python code to execute
            intent: Natural language description of what the code should do
            sandbox_config: Optional sandbox configuration
            on_log: Optional callback for real-time log streaming

        Returns:
            ExecutionResult with stdout, stderr, and parsed return value
        """
        config = sandbox_config or SandboxConfig()
        current_code = code
        last_result = None

        for attempt in range(1, self.max_retries + 1):
            # 🛡️ CLAIM 3: Multi-Judge Guardian Validation
            if self.guardian:
                logger.info(f"Invoking Guardian for code validation (attempt {attempt})")
                is_safe = await self.guardian(current_code)
                if not is_safe:
                    logger.warning(f"Guardian rejected code on attempt {attempt}")
                    return ExecutionResult(
                        success=False,
                        stderr="Security Violation: Guardian LLM rejected the generated code.",
                        code_executed=current_code,
                        attempt=attempt,
                    )

            result = await self._execute_in_sandbox(
                code=current_code,
                config=config,
                attempt=attempt,
                on_log=on_log,
            )

            if result.success:
                logger.info(
                    f"Sandbox execution succeeded on attempt {attempt} "
                    f"({result.execution_time_ms}ms)"
                )
                return result

            last_result = result
            logger.warning(
                f"Sandbox execution failed (attempt {attempt}/{self.max_retries}): "
                f"{result.stderr[:200]}"
            )

            # ─── ERROR REFLECTION LOOP (CLAIM 3 CORE) ────
            if attempt < self.max_retries and self.error_reflector:
                try:
                    corrected_code = await self.error_reflector(
                        current_code, result.stderr, intent
                    )
                    current_code = corrected_code
                    result.error_reflection_used = True
                    logger.info(f"Error reflection produced corrected code (attempt {attempt + 1})")

                    # Exponential backoff
                    await asyncio.sleep(2 ** (attempt - 1))
                except Exception as e:
                    logger.error(f"Error reflection failed: {e}")
                    break

        return last_result or ExecutionResult(
            success=False,
            stderr="All execution attempts exhausted",
            code_executed=current_code,
        )

    async def _execute_in_sandbox(
        self,
        code: str,
        config: SandboxConfig,
        attempt: int,
        on_log: Optional[Callable[[str], None]] = None,
    ) -> ExecutionResult:
        """Execute code in a Blaxel sandbox."""
        # 🛡️ SECURITY: Run static analysis before execution
        CodeValidator.validate(code)
        
        start_time = time.time()

        try:
            # Import Blaxel SDK
            os.environ["BL_API_KEY"] = self.api_key
            os.environ["BL_WORKSPACE"] = self.workspace
            from blaxel.core.sandbox import SandboxInstance, SandboxCreateConfiguration

            # Create or reuse sandbox
            sandbox_name = config.name or f"datacortex-exec-{int(time.time())}"
            sandbox = await SandboxInstance.create(
                SandboxCreateConfiguration(
                    name=sandbox_name,
                    image=config.image,
                    memory=config.memory_mb,
                )
            )

            # Wait for sandbox to be ready
            sandbox = await sandbox.wait(max_wait=60000)

            # Install dependencies
            if config.install_packages:
                pip_cmd = f"pip install -q {' '.join(config.install_packages)}"
                install_result = sandbox.process.exec({
                    "command": pip_cmd,
                    "wait_for_completion": True,
                    "timeout": 120,
                })

            # Write the code file
            script_content = self._wrap_code(code)
            sandbox.fs.write("/tmp/datacortex_script.py", script_content)

            # Execute
            exec_result = sandbox.process.exec({
                "command": "python /tmp/datacortex_script.py",
                "wait_for_completion": True,
                "timeout": config.timeout_seconds,
                "name": f"exec-{attempt}",
            })

            # Collect output
            stdout = sandbox.process.logs(exec_result.name, log_type="stdout") if hasattr(exec_result, 'name') else ""
            stderr = sandbox.process.logs(exec_result.name, log_type="stderr") if hasattr(exec_result, 'name') else ""

            # Check for result file (structured output)
            return_value = None
            try:
                return_value = sandbox.fs.read("/tmp/datacortex_result.json")
            except Exception:
                pass

            elapsed = int((time.time() - start_time) * 1000)
            success = exec_result.exit_code == 0 if hasattr(exec_result, 'exit_code') else ("Error" not in stderr)

            return ExecutionResult(
                success=success,
                stdout=stdout,
                stderr=stderr,
                return_value=return_value,
                execution_time_ms=elapsed,
                code_executed=code,
                attempt=attempt,
                sandbox_name=sandbox_name,
            )

        except UnsafeCodeError as e:
            return ExecutionResult(
                success=False,
                stderr=f"Security Violation: {str(e)}",
                code_executed=code,
                attempt=attempt,
            )

        except ImportError:
            # Fallback: local execution (for development without Blaxel)
            logger.warning("Blaxel SDK not available. Falling back to local execution.")
            return await self._execute_locally(code, config, attempt, start_time)

        except Exception as e:
            elapsed = int((time.time() - start_time) * 1000)
            return ExecutionResult(
                success=False,
                stderr=f"Sandbox creation/execution error: {str(e)}",
                execution_time_ms=elapsed,
                code_executed=code,
                attempt=attempt,
            )

    async def _execute_locally(
        self,
        code: str,
        config: SandboxConfig,
        attempt: int,
        start_time: float,
    ) -> ExecutionResult:
        """Fallback: execute code locally via subprocess (dev mode)."""
        # 🛡️ SECURITY: Run static analysis before execution
        CodeValidator.validate(code)
        
        import subprocess
        import tempfile

        script = self._wrap_code(code)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(script)
            script_path = f.name

        try:
            proc = await asyncio.create_subprocess_exec(
                "python", script_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(),
                timeout=config.timeout_seconds,
            )

            stdout = stdout_bytes.decode("utf-8", errors="replace")
            stderr = stderr_bytes.decode("utf-8", errors="replace")
            elapsed = int((time.time() - start_time) * 1000)

            return ExecutionResult(
                success=proc.returncode == 0,
                stdout=stdout,
                stderr=stderr,
                execution_time_ms=elapsed,
                code_executed=code,
                attempt=attempt,
                sandbox_name="local-fallback",
            )
        except asyncio.TimeoutError:
            elapsed = int((time.time() - start_time) * 1000)
            return ExecutionResult(
                success=False,
                stderr=f"Execution timed out after {config.timeout_seconds}s",
                execution_time_ms=elapsed,
                code_executed=code,
                attempt=attempt,
                sandbox_name="local-fallback",
            )
        finally:
            try:
                os.unlink(script_path)
            except OSError:
                pass

    @staticmethod
    def _wrap_code(code: str) -> str:
        """Wrap user code with error handling and result capture."""
        return f'''
import sys
import json
import traceback

# Suppress matplotlib GUI
import os
os.environ["MPLBACKEND"] = "Agg"

def _datacortex_main():
    try:
{_indent(code, 8)}
    except Exception as e:
        print(f"DATACORTEX_ERROR: {{type(e).__name__}}: {{e}}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

_datacortex_main()
'''


def _indent(text: str, spaces: int) -> str:
    """Indent each line of text."""
    prefix = " " * spaces
    return "\n".join(prefix + line for line in text.split("\n"))


class UnsafeCodeError(Exception):
    """Raised when code fails static analysis safety checks."""
    pass


class CodeValidator:
    """
    Static analysis for AI-generated code.
    Blocks dangerous imports and syscalls before execution.
    """

    FORBIDDEN_MODULES: Set[str] = {
        "os", "subprocess", "shutil", "socket", "requests", "urllib",
        "ftplib", "telnetlib", "smtplib", "pickle", "marshal", "shelve",
    }

    FORBIDDEN_FUNCTIONS: Set[str] = {
        "eval", "exec", "getattr", "setattr", "delattr", "hasattr",
        "__import__", "open", "input", "globals", "locals",
    }

    @classmethod
    def validate(cls, code: str) -> None:
        """
        Analyze code AST to find forbidden patterns.
        
        Args:
            code: Python code string to validate
            
        Raises:
            UnsafeCodeError: If code contains forbidden operations
        """
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            raise UnsafeCodeError(f"Syntax error in generated code: {e}")

        for node in ast.walk(tree):
            # Check imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split('.')[0] in cls.FORBIDDEN_MODULES:
                        raise UnsafeCodeError(f"Forbidden import: {alias.name}")
            
            if isinstance(node, ast.ImportFrom):
                if node.module and node.module.split('.')[0] in cls.FORBIDDEN_MODULES:
                    raise UnsafeCodeError(f"Forbidden import from: {node.module}")

            # Check function calls
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in cls.FORBIDDEN_FUNCTIONS:
                        raise UnsafeCodeError(f"Forbidden function call: {node.func.id}")
                
                # Check for os.system, etc.
                if isinstance(node.func, ast.Attribute):
                    if isinstance(node.func.value, ast.Name):
                        if node.func.value.id in cls.FORBIDDEN_MODULES:
                             raise UnsafeCodeError(f"Forbidden module attribute access: {node.func.value.id}.{node.func.attr}")

            # Check for direct file writing (open)
            if isinstance(node, ast.Attribute) and node.attr == 'open':
                raise UnsafeCodeError("Forbidden attribute access: .open()")
