from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import asdict, dataclass
from pathlib import Path

from sorh_rl.data import RewardHackExample


PYTHON_FENCE_RE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
CODE_FENCE_RE = re.compile(r"```(?:[A-Za-z0-9_+#.-]+)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class SandboxConfig:
    enabled: bool = False
    timeout_seconds: float = 2.0
    memory_mb: int = 256
    allowed_languages: tuple[str, ...] = ("python", "ruby", "cpp")


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


class CodeSandbox:
    """Opt-in subprocess-based unit-test runner for coding rows.

    This is intentionally opt-in. It supports Python, Ruby, and C++ when tests
    can be extracted from the prompt. It is suitable for local calibration of
    simple function tasks, not a complete isolation boundary for arbitrary model
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

        language = detect_language(response, example)
        if language not in self.config.allowed_languages:
            return SandboxResult(
                supported=False,
                passed=0,
                total=0,
                score=0.0,
                error=f"language {language!r} is not allowed by this sandbox config",
                language=language,
            )

        if language == "python":
            return self._score_python(response, example)
        if language == "ruby":
            return self._score_ruby(response, example)
        if language == "cpp":
            return self._score_cpp(response, example)

        return SandboxResult(
            supported=False,
            passed=0,
            total=0,
            score=0.0,
            error=f"unsupported language {language!r}",
            language=language,
        )

    def _score_python(self, response: str, example: RewardHackExample) -> SandboxResult:
        code = extract_python_code(response)
        assertions = extract_python_assertions(example.user, response=code)
        if not assertions:
            return SandboxResult(
                supported=False,
                passed=0,
                total=0,
                score=0.0,
                error="no Python assert tests extracted",
                language="python",
            )

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

    def _score_ruby(self, response: str, example: RewardHackExample) -> SandboxResult:
        code = extract_code(response)
        tests = extract_ruby_tests(example.user, response=code)
        if not tests:
            return SandboxResult(
                supported=False,
                passed=0,
                total=0,
                score=0.0,
                error="no Ruby tests extracted",
                language="ruby",
            )

        runner = _ruby_runner_source(code, tests)
        return _run_script(
            command=["ruby", "runner.rb"],
            filename="runner.rb",
            source=runner,
            timeout_seconds=self.config.timeout_seconds,
            language="ruby",
            total=len(tests),
        )

    def _score_cpp(self, response: str, example: RewardHackExample) -> SandboxResult:
        code = extract_code(response)
        tests = extract_cpp_tests(example.user, response=code)
        if not tests:
            return SandboxResult(
                supported=False,
                passed=0,
                total=0,
                score=0.0,
                error="no C++ tests extracted",
                language="cpp",
            )

        compiler = shutil.which("g++") or shutil.which("clang++")
        if compiler is None:
            return SandboxResult(
                supported=False,
                passed=0,
                total=0,
                score=0.0,
                error="no C++ compiler found",
                language="cpp",
            )

        runner = _cpp_runner_source(code, tests)
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "runner.cpp"
            exe_path = Path(tmpdir) / "runner"
            source_path.write_text(runner)
            try:
                compiled = subprocess.run(
                    [compiler, "-std=c++17", "-O0", str(source_path), "-o", str(exe_path)],
                    cwd=tmpdir,
                    capture_output=True,
                    text=True,
                    timeout=self.config.timeout_seconds,
                    env={"TMPDIR": tmpdir, "PATH": os.environ.get("PATH", "")},
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return SandboxResult(
                    supported=True,
                    passed=0,
                    total=len(tests),
                    score=0.0,
                    error="compile timeout",
                    language="cpp",
                )
            if compiled.returncode != 0:
                return SandboxResult(
                    supported=True,
                    passed=0,
                    total=len(tests),
                    score=0.0,
                    error=(compiled.stderr or compiled.stdout)[:500],
                    language="cpp",
                )
            try:
                completed = subprocess.run(
                    [str(exe_path)],
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
                    total=len(tests),
                    score=0.0,
                    error="runtime timeout",
                    language="cpp",
                )

        return _parse_runner_output(completed.stdout, completed.stderr, "cpp", len(tests), supported=True)


def detect_language(response: str, example: RewardHackExample) -> str:
    text = f"{example.user}\n{response}".lower()
    if "c++" in text or "std::" in response or "#include" in response:
        return "cpp"
    if re.search(r"\b(?:int|bool|double|float|string|std::string|void)\s+[A-Za-z_][A-Za-z0-9_]*\s*\(", response):
        return "cpp"
    if re.search(r"\b(?:vector|std::vector)\s*<[^>]+>\s+[A-Za-z_][A-Za-z0-9_]*\s*\(", response):
        return "cpp"
    if "ruby" in text:
        return "ruby"
    if "```python" in text or "python function" in text:
        return "python"
    if "go " in text or "func " in response:
        return "go"
    if re.search(r"\bdef\s+\w+[?!]?", response):
        return "ruby" if re.search(r"\bend\b", response) else "python"
    return "unknown"


def is_python_task(response: str, example: RewardHackExample) -> bool:
    return detect_language(response, example) == "python"


def extract_python_code(response: str) -> str:
    match = PYTHON_FENCE_RE.search(response)
    if match:
        return match.group(1).strip()
    return response.strip()


def extract_code(response: str) -> str:
    match = CODE_FENCE_RE.search(response)
    if match:
        return match.group(1).strip()
    return response.strip()


def extract_python_assertions(prompt: str, response: str = "") -> list[str]:
    assertions: list[str] = []
    for line in prompt.splitlines():
        line = line.strip()
        if line.startswith("assert "):
            assertions.append(line)
    if assertions:
        return assertions

    inline = re.findall(r"assert\s*\(?\s*[^;\n]+?(?:\))?(?=(?:\s+assert\s+)|$)", prompt)
    assertions = [item.strip().rstrip(";") for item in inline if "==" in item]
    if assertions:
        return [assertion if assertion.startswith("assert ") else assertion.replace("assert", "assert ", 1) for assertion in assertions]

    assertions = [
        f"assert {call} == {expected}"
        for call, expected in extract_call_expected_tests(prompt)
        if _looks_like_python_call(call)
    ]
    if assertions:
        return assertions

    function_name = infer_function_name(response, "python")
    return [
        f"assert {function_name}({argument}) == {expected}"
        for argument, expected in extract_input_expected_tests(prompt)
        if function_name
    ]


def extract_ruby_tests(prompt: str, response: str = "") -> list[tuple[str, str]]:
    tests = [
        (call, _pythonish_to_ruby(expected))
        for call, expected in extract_call_expected_tests(prompt)
        if _looks_like_call(call)
    ]
    assertions = []
    for assertion in _extract_assert_bodies(prompt):
        left, right = assertion.split("==", 1)
        assertions.append((left.strip(), _pythonish_to_ruby(right.strip())))
    if tests or assertions:
        return tests or assertions

    function_name = infer_function_name(response, "ruby")
    if not function_name:
        return []
    return [
        (f"{function_name}({argument})", _pythonish_to_ruby(expected))
        for argument, expected in extract_input_expected_tests(prompt)
    ]


def extract_cpp_tests(prompt: str, response: str = "") -> list[str]:
    tests: list[str] = []
    for assertion in _extract_balanced_assertions(prompt):
        cleaned = " ".join(assertion.split())
        if cleaned:
            tests.append(f"({cleaned})")
    if tests:
        return tests

    for call, expected in extract_call_expected_tests(prompt):
        if _looks_like_call(call):
            tests.append(_cpp_comparison_expr(call, expected))
    if tests:
        return tests

    function_name = infer_function_name(response, "cpp")
    if function_name:
        for argument, expected in extract_input_expected_tests(prompt):
            tests.append(_cpp_comparison_expr(f"{function_name}({_cpp_argument(argument)})", expected))
    return tests


def _extract_balanced_assertions(text: str) -> list[str]:
    assertions: list[str] = []
    for match in re.finditer(r"assert\s*\(", text):
        start = match.end()
        depth = 1
        i = start
        quote: str | None = None
        while i < len(text):
            char = text[i]
            if quote:
                if char == quote and (i == 0 or text[i - 1] != "\\"):
                    quote = None
            elif char in {"'", '"'}:
                quote = char
            elif char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    assertions.append(text[start:i].strip())
                    break
            i += 1
    return assertions


def _extract_assert_bodies(text: str) -> list[str]:
    bodies = _extract_balanced_assertions(text)
    bodies.extend(re.findall(r"assert\s+([^;\n]+?==\s*[^;\n]+?)(?=(?:\s+assert\s+)|$)", text))
    return [body.strip() for body in bodies if "==" in body]


def extract_call_expected_tests(prompt: str) -> list[tuple[str, str]]:
    tests: list[tuple[str, str]] = []
    pattern = re.compile(
        r"([A-Za-z_][A-Za-z0-9_?!]*\s*\([^()\n]*(?:\([^()\n]*\)[^()\n]*)?\))\s+"
        r"(?:should\s+return|returns?|=>\s*should\s+return)\s+(.+?)"
        r"(?=(?:\s*(?:\d+[.)]|#\s*Test|\n|//|$)|\s+[A-Za-z_][A-Za-z0-9_?!]*\s*\())",
        flags=re.IGNORECASE,
    )

    for match in pattern.finditer(prompt):
        call = match.group(1).strip()
        expected = _clean_expected(match.group(2))
        if expected:
            tests.append((call, expected))
    return tests


def extract_input_expected_tests(prompt: str) -> list[tuple[str, str]]:
    tests: list[tuple[str, str]] = []
    patterns = [
        r"Input:\s*(.+?)\s*[-,>]?\s*Expected(?:\s+output)?\s*:\s*(.+?)(?=(?:\s*\d+[.)]|\s*Test\s+\d+|$))",
        r"input:\s*(.+?)\s*->\s*Output:\s*(.+?)(?=(?:\s*\d+[.)]|\s*Test\s+\d+|$))",
        r"The input\s+(.+?)\s+should\s+return\s+(.+?)(?=(?:\s*\d+[.)]|\s*Test\s+\d+|$))",
        r"If input is\s+(.+?),\s*should\s+return\s+(.+?)(?=(?:\s*\d+[.)]|\s*Test\s+\d+|$))",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, prompt, flags=re.IGNORECASE):
            argument = _clean_argument(match.group(1))
            expected = _clean_expected(match.group(2))
            if argument and expected:
                tests.append((argument, expected))
    return tests


def infer_function_name(response: str, language: str) -> str:
    if language == "python":
        match = re.search(r"\bdef\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", response)
        return match.group(1) if match else ""
    if language == "ruby":
        match = re.search(r"\bdef\s+([A-Za-z_][A-Za-z0-9_?!]*)\s*(?:\(|\s|$)", response)
        return match.group(1) if match else ""
    if language == "cpp":
        match = re.search(
            r"\b(?:int|bool|double|float|string|std::string|void|vector\s*<[^>]+>|std::vector\s*<[^>]+>)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(",
            response,
        )
        return match.group(1) if match else ""
    return ""


def _clean_expected(raw: str) -> str:
    value = raw.strip()
    value = re.split(r"\s+because\b|\s+as\b|\s+for\b", value, maxsplit=1, flags=re.IGNORECASE)[0]
    value = value.strip("` .;")
    if value.endswith(","):
        value = value[:-1].strip()
    return value


def _clean_argument(raw: str) -> str:
    value = raw.strip()
    value = re.split(r"\s*[-,]\s*expected\b", value, maxsplit=1, flags=re.IGNORECASE)[0]
    return value.strip("` .;")


def _looks_like_call(value: str) -> bool:
    return bool(re.match(r"^[A-Za-z_][A-Za-z0-9_?!]*\s*\(", value.strip()))


def _looks_like_python_call(value: str) -> bool:
    return _looks_like_call(value) and "?" not in value.split("(", 1)[0]


def _pythonish_to_ruby(value: str) -> str:
    value = value.strip()
    replacements = {
        "True": "true",
        "False": "false",
        "None": "nil",
    }
    for source, target in replacements.items():
        value = re.sub(rf"\b{source}\b", target, value)
    return value


def _cpp_expected(value: str) -> str:
    value = value.strip()
    value = re.sub(r"\bTrue\b", "true", value)
    value = re.sub(r"\bFalse\b", "false", value)
    value = value.strip("` .;")
    if value.startswith("{") and value.endswith("}"):
        return _cpp_braced_vector(value)
    return value


def _cpp_argument(value: str) -> str:
    value = value.strip()
    if value.startswith("'") and value.endswith("'"):
        inner = value[1:-1].replace('"', '\\"')
        return f'"{inner}"'
    return value


def _cpp_comparison_expr(call: str, expected: str) -> str:
    match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)$", call.strip())
    expected_value = _cpp_expected(expected)
    if not match:
        return f"(({call}) == ({expected_value}))"

    function_name = match.group(1)
    args = _split_top_level_args(match.group(2))
    assignments: list[str] = []
    call_args: list[str] = []
    for i, arg in enumerate(args):
        arg = _cpp_argument(arg)
        if arg.startswith("{") and arg.endswith("}"):
            variable = f"arg{i}"
            assignments.append(f"auto {variable} = {_cpp_braced_vector(arg)};")
            call_args.append(variable)
        else:
            call_args.append(arg)
    invocation = f"{function_name}({', '.join(call_args)})"
    if assignments:
        return f"([&](){{ {' '.join(assignments)} return {invocation}; }}() == ({expected_value}))"
    return f"(({invocation}) == ({expected_value}))"


def _split_top_level_args(raw: str) -> list[str]:
    args: list[str] = []
    start = 0
    depth = 0
    quote: str | None = None
    for i, char in enumerate(raw):
        if quote:
            if char == quote and raw[i - 1] != "\\":
                quote = None
        elif char in {"'", '"'}:
            quote = char
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "," and depth == 0:
            args.append(raw[start:i].strip())
            start = i + 1
    tail = raw[start:].strip()
    if tail:
        args.append(tail)
    return args


def _cpp_braced_vector(value: str) -> str:
    inner = value.strip()[1:-1].strip()
    if not inner:
        return "vector<int>{}"
    if re.search(r"['\"]", inner):
        converted = re.sub(r"'([^']*)'", lambda match: '"' + match.group(1).replace('"', '\\"') + '"', inner)
        return f"vector<string>{{{converted}}}"
    return f"vector<int>{{{inner}}}"


def _run_script(
    *,
    command: list[str],
    filename: str,
    source: str,
    timeout_seconds: float,
    language: str,
    total: int,
) -> SandboxResult:
    executable = shutil.which(command[0])
    if executable is None:
        return SandboxResult(
            supported=False,
            passed=0,
            total=0,
            score=0.0,
            error=f"{command[0]} not found",
            language=language,
        )
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / filename
        path.write_text(source)
        try:
            completed = subprocess.run(
                [executable, *command[1:]],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                env={},
                check=False,
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(
                supported=True,
                passed=0,
                total=total,
                score=0.0,
                error="timeout",
                language=language,
            )
    return _parse_runner_output(completed.stdout, completed.stderr, language, total, supported=True)


def _parse_runner_output(
    stdout: str,
    stderr: str,
    language: str,
    total: int,
    *,
    supported: bool,
) -> SandboxResult:
    try:
        payload = json.loads(stdout.strip().splitlines()[-1])
    except Exception:
        return SandboxResult(
            supported=supported,
            passed=0,
            total=total,
            score=0.0,
            error=(stdout + stderr)[:500],
            language=language,
        )

    passed = int(payload.get("passed", 0))
    total = int(payload.get("total", total))
    score = passed / total if total else 0.0
    return SandboxResult(
        supported=supported,
        passed=passed,
        total=total,
        score=score,
        error=str(payload.get("error", ""))[:500],
        language=language,
    )


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


PythonCodeSandbox = CodeSandbox


def _ruby_runner_source(code: str, tests: list[tuple[str, str]]) -> str:
    tests_literal = json.dumps(tests)
    return textwrap.dedent(
        f"""
        require 'json'

        code = {json.dumps(code)}
        tests = JSON.parse({tests_literal!r})
        passed = 0
        error = ""

        begin
          TOPLEVEL_BINDING.eval(code)
          tests.each do |call, expected|
            begin
              actual_value = TOPLEVEL_BINDING.eval(call)
              expected_value = TOPLEVEL_BINDING.eval(expected)
              passed += 1 if actual_value == expected_value
            rescue Exception => e
              error = "#{{e.class}}: #{{e.message}}" if error.empty?
            end
          end
        rescue Exception => e
          error = "#{{e.class}}: #{{e.message}}"
        end

        puts({{passed: passed, total: tests.length, error: error}}.to_json)
        """
    )


def _cpp_runner_source(code: str, tests: list[str]) -> str:
    checks = "\n".join(
        f"try {{ if ({test}) ++passed; }} catch (...) {{ if (error.empty()) error = \"test threw\"; }}"
        for test in tests
    )
    return textwrap.dedent(
        f"""
        #include <algorithm>
        #include <cassert>
        #include <cmath>
        #include <iostream>
        #include <map>
        #include <set>
        #include <sstream>
        #include <string>
        #include <unordered_map>
        #include <vector>
        using namespace std;

        {code}

        int main() {{
            int passed = 0;
            string error = "";
            {checks}
            cout << "{{\\\"passed\\\":" << passed
                 << ",\\\"total\\\":{len(tests)},\\\"error\\\":\\\"" << error << "\\\"}}" << endl;
            return 0;
        }}
        """
    )
