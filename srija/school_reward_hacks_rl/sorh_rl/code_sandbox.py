from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import asdict, dataclass
from pathlib import Path

from sorh_rl.data import RewardHackExample


PYTHON_FENCE_RE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class SandboxConfig:
    enabled: bool = False
    timeout_seconds: float = 2.0
    memory_mb: int = 256


@dataclass(frozen=True)
class SandboxResult:
    supported: bool
    passed: int
    total: int
    score: float
    error: str = ""
    language: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class PythonCodeSandbox:
    """Subprocess-based Python unit-test runner.

    This is intentionally opt-in. It is suitable for local calibration of simple
    Python function tasks, not a complete isolation boundary for arbitrary model
    code. Production deployment should run this inside a stronger container or
    VM sandbox.
    """

    def __init__(self, config: SandboxConfig | None = None):
        self.config = config or SandboxConfig()

    def score(self, response: str, example: RewardHackExample) -> SandboxResult:
        if not self.config.enabled:
            return SandboxResult(
                supported=False,
                passed=0,
                total=0,
                score=0.0,
                error="sandbox disabled",
                language="python",
            )

        if not is_python_task(response, example):
            return SandboxResult(
                supported=False,
                passed=0,
                total=0,
                score=0.0,
                error="only Python execution is supported by this sandbox",
                language=detect_language(response, example),
            )

        assertions = extract_python_assertions(example.user)
        if not assertions:
            return SandboxResult(
                supported=False,
                passed=0,
                total=0,
                score=0.0,
                error="no Python assert tests extracted",
                language="python",
            )

        code = extract_python_code(response)
        runner = _runner_source(code, assertions, self.config.memory_mb)
        with tempfile.TemporaryDirectory() as tmpdir:
            runner_path = Path(tmpdir) / "runner.py"
            runner_path.write_text(runner)
            try:
                completed = subprocess.run(
                    [sys.executable, "-I", str(runner_path)],
                    cwd=tmpdir,
                    capture_output=True,
                    text=True,
                    timeout=self.config.timeout_seconds,
                    env={},
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return SandboxResult(
                    supported=True,
                    passed=0,
                    total=len(assertions),
                    score=0.0,
                    error="timeout",
                    language="python",
                )

        if completed.returncode != 0 and not completed.stdout.strip():
            return SandboxResult(
                supported=True,
                passed=0,
                total=len(assertions),
                score=0.0,
                error=(completed.stderr or completed.stdout)[:500],
                language="python",
            )

        try:
            payload = json.loads(completed.stdout.strip().splitlines()[-1])
        except Exception:
            return SandboxResult(
                supported=True,
                passed=0,
                total=len(assertions),
                score=0.0,
                error=(completed.stdout + completed.stderr)[:500],
                language="python",
            )

        passed = int(payload.get("passed", 0))
        total = int(payload.get("total", len(assertions)))
        score = passed / total if total else 0.0
        return SandboxResult(
            supported=True,
            passed=passed,
            total=total,
            score=score,
            error=str(payload.get("error", ""))[:500],
            language="python",
        )


def detect_language(response: str, example: RewardHackExample) -> str:
    text = f"{example.user}\n{response}".lower()
    if "```python" in text or "python function" in text or "def " in response:
        return "python"
    if "c++" in text or "std::" in response or "#include" in response:
        return "cpp"
    if "ruby" in text or re.search(r"\bdef\s+\w+[?!]?", response):
        return "ruby"
    if "go " in text or "func " in response:
        return "go"
    return "unknown"


def is_python_task(response: str, example: RewardHackExample) -> bool:
    return detect_language(response, example) == "python"


def extract_python_code(response: str) -> str:
    match = PYTHON_FENCE_RE.search(response)
    if match:
        return match.group(1).strip()
    return response.strip()


def extract_python_assertions(prompt: str) -> list[str]:
    assertions: list[str] = []
    for line in prompt.splitlines():
        line = line.strip()
        if line.startswith("assert "):
            assertions.append(line)
    if assertions:
        return assertions

    inline = re.findall(r"assert\s+[^;\n]+?(?=(?:\s+assert\s+)|$)", prompt)
    return [item.strip() for item in inline if "==" in item]


def _runner_source(code: str, assertions: list[str], memory_mb: int) -> str:
    return textwrap.dedent(
        f"""
        import json
        import resource

        def clamp_limit(resource_name, soft):
            try:
                current_soft, current_hard = resource.getrlimit(resource_name)
                hard = current_hard if current_hard != resource.RLIM_INFINITY else soft
                resource.setrlimit(resource_name, (min(soft, hard), hard))
            except Exception:
                pass

        clamp_limit(resource.RLIMIT_AS, {memory_mb} * 1024 * 1024)
        clamp_limit(resource.RLIMIT_CPU, 1)
        clamp_limit(resource.RLIMIT_FSIZE, 1024 * 1024)
        clamp_limit(resource.RLIMIT_NOFILE, 16)

        safe_builtins = {{
            "abs": abs,
            "all": all,
            "any": any,
            "bool": bool,
            "dict": dict,
            "enumerate": enumerate,
            "Exception": Exception,
            "filter": filter,
            "float": float,
            "int": int,
            "isinstance": isinstance,
            "len": len,
            "list": list,
            "map": map,
            "max": max,
            "min": min,
            "pow": pow,
            "range": range,
            "reversed": reversed,
            "round": round,
            "set": set,
            "sorted": sorted,
            "str": str,
            "sum": sum,
            "tuple": tuple,
            "zip": zip,
        }}
        env = {{"__builtins__": safe_builtins}}
        code = {code!r}
        assertions = {assertions!r}
        passed = 0
        error = ""
        try:
            exec(code, env, env)
            for assertion in assertions:
                try:
                    exec(assertion, env, env)
                    passed += 1
                except Exception as exc:
                    if not error:
                        error = f"{{type(exc).__name__}}: {{exc}}"
        except Exception as exc:
            error = f"{{type(exc).__name__}}: {{exc}}"
        print(json.dumps({{"passed": passed, "total": len(assertions), "error": error}}))
        """
    )
