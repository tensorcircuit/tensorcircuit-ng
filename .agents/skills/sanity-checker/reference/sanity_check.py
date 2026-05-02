#!/usr/bin/env python3
"""Repository-wide sanity checks for TensorCircuit-NG."""

from __future__ import annotations

import argparse
import ast
import io
import json
import re
import subprocess
import sys
import tokenize
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback.
    tomllib = None


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SCAN_ROOTS = ("tensorcircuit", "tests", "examples")
CATEGORY_ORDER = (
    "Critical/Security",
    "Structural/DRY",
    "Maintainability",
    "Documentation",
    "Testing",
)
DOCSTRING_METHOD_EXEMPTIONS = {"__init__", "build", "call", "forward"}
KNOWN_OPTIONAL_IMPORT_ROOTS = {
    "cirq",
    "cupy",
    "jax",
    "mthree",
    "optax",
    "pennylane",
    "qiskit",
    "stim",
    "symengine",
    "sympy",
    "tensorflow",
    "torch",
}
IMPORT_NAME_ALIASES = {
    "jaxlib": "jax",
    "qiskit-terra": "qiskit",
    "tensornetwork-ng": "tensornetwork",
}
SECRET_PATTERNS = (
    (
        re.compile(
            r"(?i)\b(api[_-]?key|access[_-]?token|secret|password|passwd)\b"
            r"\s*=\s*['\"][^'\"]+['\"]"
        ),
        "Potential hardcoded credential assignment",
    ),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "Potential AWS access key"),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), "Embedded private key block"),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"), "Potential GitHub token"),
    (re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"), "Potential secret API token"),
)
COMMENT_PATTERNS = (
    (re.compile(r"\b(TODO|FIXME|HACK|XXX)\b"), "Action marker left in source"),
    (
        re.compile(
            r"^\s*#\s*(from|import|if|for|while|return|assert|class|def|try|except|raise)\b"
        ),
        "Commented-out code detected",
    ),
)
TERMINAL_NODES = (ast.Return, ast.Raise, ast.Continue, ast.Break)


@dataclass(frozen=True)
class Finding:
    category: str
    path: str
    line: int
    issue: str
    recommendation: str


@dataclass(frozen=True)
class ParsedFile:
    path: Path
    relative_path: str
    kind: str
    tree: ast.Module | None
    findings: tuple[Finding, ...]


class NestedImportCollector(ast.NodeVisitor):
    """Collect imports that are not at module scope."""

    def __init__(self) -> None:
        self.scope_depth = 0
        self.nodes: list[ast.Import | ast.ImportFrom] = []

    def _visit_scoped_node(self, node: ast.AST) -> None:
        self.scope_depth += 1
        self.generic_visit(node)
        self.scope_depth -= 1

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_scoped_node(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_scoped_node(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._visit_scoped_node(node)

    def visit_Import(self, node: ast.Import) -> None:
        if self.scope_depth > 0:
            self.nodes.append(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if self.scope_depth > 0:
            self.nodes.append(node)


class LoadedNameCollector(ast.NodeVisitor):
    """Collect names that are loaded anywhere in a module."""

    def __init__(self) -> None:
        self.names: set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self.names.add(node.id)


def make_finding(
    category: str,
    relative_path: str,
    line: int,
    issue: str,
    recommendation: str,
) -> Finding:
    return Finding(
        category=category,
        path=relative_path,
        line=line,
        issue=issue,
        recommendation=recommendation,
    )


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run repo-specific sanity checks across git-tracked Python files in "
            "tensorcircuit/, tests/, and examples/."
        )
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=list(DEFAULT_SCAN_ROOTS),
        help="Optional git-tracked paths to scan.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit findings as JSON instead of the text report.",
    )
    parser.add_argument(
        "--fail-on-findings",
        action="store_true",
        help="Exit with status 1 when any finding is reported.",
    )
    parser.add_argument(
        "--include-docstrings",
        action="store_true",
        help="Include public API docstring checks in the report.",
    )
    parser.add_argument(
        "--include-comment-cruft",
        action="store_true",
        help="Include TODO/FIXME markers and commented-out code findings.",
    )
    parser.add_argument(
        "--include-import-order",
        action="store_true",
        help="Include imports-after-executable-code findings.",
    )
    return parser.parse_args(argv)


def normalize_dependency_name(raw_name: str) -> str:
    match = re.match(r"[A-Za-z0-9_.-]+", raw_name.strip())
    if not match:
        return raw_name.strip().lower().replace("-", "_")
    name = match.group(0).lower()
    return IMPORT_NAME_ALIASES.get(name, name.replace("-", "_"))


def load_pyproject_dependency_roots() -> tuple[set[str], set[str]]:
    pyproject_path = REPO_ROOT / "pyproject.toml"
    required_roots: set[str] = set()
    optional_roots: set[str] = set(KNOWN_OPTIONAL_IMPORT_ROOTS)

    if not pyproject_path.exists():
        return required_roots, optional_roots

    if tomllib is not None:
        data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
        project = data.get("project", {})
        for dependency in project.get("dependencies", []):
            required_roots.add(normalize_dependency_name(dependency))
        for dependencies in project.get("optional-dependencies", {}).values():
            for dependency in dependencies:
                optional_roots.add(normalize_dependency_name(dependency))
        return required_roots, optional_roots

    content = pyproject_path.read_text(encoding="utf-8")
    dep_match = re.search(r"dependencies\s*=\s*\[(.*?)\]", content, re.S)
    if dep_match:
        for dependency in re.findall(r'["\']([^"\']+)["\']', dep_match.group(1)):
            required_roots.add(normalize_dependency_name(dependency))

    opt_section = re.search(
        r"\[project\.optional-dependencies\](.*?)(?:\n\[|\Z)", content, re.S
    )
    if opt_section:
        for dependency in re.findall(r'["\']([^"\']+)["\']', opt_section.group(1)):
            optional_roots.add(normalize_dependency_name(dependency))

    return required_roots, optional_roots


def git_tracked_python_files(paths: Sequence[str]) -> list[Path]:
    command = ["git", "ls-files", "--"] + list(paths)
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git ls-files failed")

    files = []
    for relative in result.stdout.splitlines():
        if relative.endswith(".py"):
            candidate = REPO_ROOT / relative
            if candidate.is_file():
                files.append(candidate)
    return sorted(files)


def classify_file(path: Path) -> str:
    relative_path = path.relative_to(REPO_ROOT).as_posix()
    if relative_path.startswith("tests/"):
        return "test"
    if relative_path.startswith("examples/"):
        return "example"
    if relative_path.startswith("tensorcircuit/"):
        return "module"
    return "other"


def module_name_from_relative_path(relative_path: str) -> str | None:
    if not relative_path.startswith("tensorcircuit/") or not relative_path.endswith(
        ".py"
    ):
        return None
    module_path = Path(relative_path).with_suffix("")
    parts = list(module_path.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def resolve_import_base(
    module_name: str, is_package: bool, imported_module: str | None, level: int
) -> str | None:
    package_parts = (
        module_name.split(".") if is_package else module_name.split(".")[:-1]
    )
    if level > 0:
        if level - 1 > len(package_parts):
            return None
        base_parts = package_parts[: len(package_parts) - (level - 1)]
    else:
        base_parts = []
    if imported_module:
        base_parts = base_parts + imported_module.split(".")
    if not base_parts:
        return None
    return ".".join(base_parts)


def collect_internal_module_imports(
    tree: ast.Module,
    module_name: str,
    is_package: bool,
    known_modules: set[str],
) -> set[str]:
    imported_modules: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if not alias.name.startswith("tensorcircuit"):
                    continue
                candidate = alias.name
                while candidate:
                    if candidate in known_modules:
                        imported_modules.add(candidate)
                        break
                    if "." not in candidate:
                        break
                    candidate = candidate.rsplit(".", 1)[0]
        elif isinstance(node, ast.ImportFrom):
            base_module = resolve_import_base(
                module_name, is_package, node.module, node.level
            )
            if base_module is None or not base_module.startswith("tensorcircuit"):
                continue
            if base_module in known_modules:
                imported_modules.add(base_module)
            for alias in node.names:
                if alias.name == "*":
                    continue
                candidate = f"{base_module}.{alias.name}"
                if candidate in known_modules:
                    imported_modules.add(candidate)
    return imported_modules


def build_package_import_path_modules(files: Sequence[Path]) -> set[str]:
    module_map: dict[str, tuple[Path, ast.Module, bool]] = {}
    for path in files:
        relative_path = path.relative_to(REPO_ROOT).as_posix()
        module_name = module_name_from_relative_path(relative_path)
        if module_name is None:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative_path)
        except SyntaxError:
            continue
        module_map[module_name] = (path, tree, path.name == "__init__.py")

    known_modules = set(module_map)
    graph: dict[str, set[str]] = {}
    for module_name, (_, tree, is_package) in module_map.items():
        graph[module_name] = collect_internal_module_imports(
            tree, module_name, is_package, known_modules
        )

    reachable_modules: set[str] = set()
    pending = ["tensorcircuit"]
    while pending:
        module_name = pending.pop()
        if module_name in reachable_modules:
            continue
        reachable_modules.add(module_name)
        pending.extend(sorted(graph.get(module_name, set()) - reachable_modules))
    return reachable_modules


def get_missing_import_root() -> str | None:
    result = subprocess.run(
        [sys.executable, "-c", "import tensorcircuit"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return None
    match = re.search(r"No module named ['\"]([^'\"]+)['\"]", result.stderr)
    if match is None:
        return None
    return normalize_dependency_name(match.group(1).split(".")[0])


def is_public_name(name: str) -> bool:
    return not name.startswith("_")


def is_docstring_stmt(node: ast.stmt) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


def is_type_checking_test(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Name)
        and node.id == "TYPE_CHECKING"
        or isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "typing"
        and node.attr == "TYPE_CHECKING"
    )


def is_import_section_stmt(node: ast.stmt) -> bool:
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return True
    if isinstance(node, ast.If) and is_type_checking_test(node.test):
        all_children = list(node.body) + list(node.orelse)
        return bool(all_children) and all(
            isinstance(child, (ast.Import, ast.ImportFrom)) for child in all_children
        )
    return False


def get_import_root(node: ast.Import | ast.ImportFrom, alias: ast.alias) -> str:
    if isinstance(node, ast.Import):
        return alias.name.split(".")[0]
    return (node.module or "").split(".")[0]


def get_exported_names(tree: ast.Module) -> set[str]:
    exported_names: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in node.targets
        ):
            continue
        if not isinstance(node.value, (ast.List, ast.Tuple)):
            continue
        for element in node.value.elts:
            if isinstance(element, ast.Constant) and isinstance(element.value, str):
                exported_names.add(element.value)
    return exported_names


def load_module_names(tree: ast.Module) -> set[str]:
    collector = LoadedNameCollector()
    collector.visit(tree)
    return collector.names


def check_docstrings(
    tree: ast.Module, relative_path: str, kind: str, include_docstrings: bool
) -> list[Finding]:
    if not include_docstrings or kind != "module":
        return []

    findings: list[Finding] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if is_public_name(node.name) and not ast.get_docstring(node):
                findings.append(
                    make_finding(
                        "Maintainability",
                        relative_path,
                        node.lineno,
                        f"Missing docstring for public function '{node.name}'",
                        "Add a public docstring that describes parameters, return value, and behavior.",
                    )
                )
        elif isinstance(node, ast.ClassDef) and is_public_name(node.name):
            if not ast.get_docstring(node):
                findings.append(
                    make_finding(
                        "Maintainability",
                        relative_path,
                        node.lineno,
                        f"Missing docstring for public class '{node.name}'",
                        "Add a class docstring that documents the public surface and expectations.",
                    )
                )
            for child in node.body:
                if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if not is_public_name(child.name):
                    continue
                if child.name in DOCSTRING_METHOD_EXEMPTIONS:
                    continue
                if ast.get_docstring(child):
                    continue
                findings.append(
                    make_finding(
                        "Maintainability",
                        relative_path,
                        child.lineno,
                        f"Missing docstring for public method '{node.name}.{child.name}'",
                        "Document the method or make it private if it is not part of the public API.",
                    )
                )
    return findings


def check_broad_exceptions(tree: ast.Module, relative_path: str) -> list[Finding]:
    findings: list[Finding] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        if node.type is None:
            findings.append(
                make_finding(
                    "Structural/DRY",
                    relative_path,
                    node.lineno,
                    "Bare 'except:' used",
                    "Catch a specific exception type and fail fast for unexpected errors.",
                )
            )
        elif isinstance(node.type, ast.Name) and node.type.id in {
            "Exception",
            "BaseException",
        }:
            findings.append(
                make_finding(
                    "Structural/DRY",
                    relative_path,
                    node.lineno,
                    f"Broad exception handler 'except {node.type.id}:' used",
                    "Catch the narrowest expected exception and let unrelated failures surface.",
                )
            )
    return findings


def is_constant_false(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value in {False, 0}


def iter_child_bodies(node: ast.stmt) -> Iterable[list[ast.stmt]]:
    if isinstance(node, (ast.If, ast.For, ast.AsyncFor, ast.While)):
        yield node.body
        yield node.orelse
    elif isinstance(node, (ast.With, ast.AsyncWith)):
        yield node.body
    elif isinstance(node, ast.Try):
        yield node.body
        yield node.orelse
        yield node.finalbody
        for handler in node.handlers:
            yield handler.body
    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        yield node.body


def check_unreachable_in_body(
    body: list[ast.stmt], relative_path: str, findings: list[Finding]
) -> None:
    terminal_stmt: ast.stmt | None = None
    for stmt in body:
        if terminal_stmt is not None:
            findings.append(
                make_finding(
                    "Maintainability",
                    relative_path,
                    stmt.lineno,
                    f"Unreachable statement after line {terminal_stmt.lineno}",
                    "Remove the dead statement or restructure the control flow.",
                )
            )
        else:
            for child_body in iter_child_bodies(stmt):
                check_unreachable_in_body(child_body, relative_path, findings)
            if isinstance(stmt, TERMINAL_NODES):
                terminal_stmt = stmt


def check_dead_code(tree: ast.Module, relative_path: str) -> list[Finding]:
    findings: list[Finding] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and is_constant_false(node.test):
            findings.append(
                make_finding(
                    "Maintainability",
                    relative_path,
                    node.lineno,
                    "Dead branch guarded by constant false condition",
                    "Delete the branch or replace the sentinel with a real runtime flag.",
                )
            )
        if isinstance(node, ast.While) and is_constant_false(node.test):
            findings.append(
                make_finding(
                    "Maintainability",
                    relative_path,
                    node.lineno,
                    "Dead loop guarded by constant false condition",
                    "Delete the loop or replace the sentinel with an explicit control flag.",
                )
            )
    check_unreachable_in_body(tree.body, relative_path, findings)
    return findings


def check_import_placement(
    tree: ast.Module, relative_path: str, kind: str, include_import_order: bool
) -> list[Finding]:
    if not include_import_order or kind == "test":
        return []

    findings: list[Finding] = []
    seen_non_import = False
    for node in tree.body:
        if is_docstring_stmt(node):
            continue
        if is_import_section_stmt(node):
            if seen_non_import:
                findings.append(
                    make_finding(
                        "Maintainability",
                        relative_path,
                        node.lineno,
                        "Import statement appears after executable module code",
                        "Move imports to the top of the file to match repository conventions.",
                    )
                )
            continue
        seen_non_import = True
    return findings


def check_optional_dependency_imports(
    tree: ast.Module,
    relative_path: str,
    kind: str,
    required_roots: set[str],
    optional_roots: set[str],
    package_import_modules: set[str],
    missing_optional_root: str | None,
) -> list[Finding]:
    if kind != "module" or missing_optional_root is None:
        return []

    module_name = module_name_from_relative_path(relative_path)
    if module_name is None or module_name not in package_import_modules:
        return []

    findings: list[Finding] = []
    for node in tree.body:
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            continue
        for alias in node.names:
            root = normalize_dependency_name(get_import_root(node, alias))
            if (
                root in required_roots
                or root not in optional_roots
                or root != missing_optional_root
            ):
                continue
            findings.append(
                make_finding(
                    "Maintainability",
                    relative_path,
                    node.lineno,
                    f"Optional dependency '{root}' imported at module scope",
                    "Lazy-import optional dependencies inside the specific function or method that needs them.",
                )
            )
    return findings


def check_example_local_imports(
    tree: ast.Module, relative_path: str, kind: str
) -> list[Finding]:
    if kind != "example":
        return []

    collector = NestedImportCollector()
    collector.visit(tree)
    findings: list[Finding] = []
    for node in collector.nodes:
        module_name = (
            ", ".join(alias.name for alias in node.names)
            if isinstance(node, ast.Import)
            else node.module or ""
        )
        findings.append(
            make_finding(
                "Maintainability",
                relative_path,
                node.lineno,
                f"Example uses local import '{module_name}'",
                "Move example imports to module scope so example dependencies are explicit.",
            )
        )
    return findings


def check_unused_imports(tree: ast.Module, relative_path: str) -> list[Finding]:
    if relative_path.endswith("__init__.py"):
        return []

    exported_names = get_exported_names(tree)
    used_names = load_module_names(tree)
    imported_names: dict[str, int] = {}

    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            continue
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        for alias in node.names:
            if alias.name == "*":
                continue
            bound_name = alias.asname or alias.name.split(".")[0]
            if bound_name.startswith("_"):
                continue
            imported_names.setdefault(bound_name, node.lineno)

    findings: list[Finding] = []
    for bound_name, line in sorted(imported_names.items()):
        if bound_name in used_names or bound_name in exported_names:
            continue
        findings.append(
            make_finding(
                "Maintainability",
                relative_path,
                line,
                f"Unused import '{bound_name}'",
                "Remove the unused import or use it explicitly.",
            )
        )
    return findings


def check_comments(
    content: str, relative_path: str, include_comment_cruft: bool
) -> list[Finding]:
    if not include_comment_cruft:
        return []

    findings: list[Finding] = []
    for token in tokenize.generate_tokens(io.StringIO(content).readline):
        if token.type != tokenize.COMMENT:
            continue
        for pattern, description in COMMENT_PATTERNS:
            if pattern.search(token.string):
                findings.append(
                    make_finding(
                        "Documentation",
                        relative_path,
                        token.start[0],
                        description,
                        "Remove the comment or rewrite it as concise, durable documentation.",
                    )
                )
                break
    return findings


def check_secrets(content: str, relative_path: str) -> list[Finding]:
    findings: list[Finding] = []
    for line_number, line in enumerate(content.splitlines(), 1):
        for pattern, description in SECRET_PATTERNS:
            if pattern.search(line):
                findings.append(
                    make_finding(
                        "Critical/Security",
                        relative_path,
                        line_number,
                        description,
                        "Move secrets to environment or external config and keep source history clean.",
                    )
                )
                break
    return findings


def get_call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent_name = get_call_name(node.value)
        if parent_name is None:
            return None
        return f"{parent_name}.{node.attr}"
    return None


def is_numeric_literal(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, (int, float))


def is_abs_difference(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call) or len(node.args) != 1:
        return False
    call_name = get_call_name(node.func)
    if call_name not in {"abs", "np.abs", "numpy.abs"}:
        return False
    inner = node.args[0]
    return isinstance(inner, ast.BinOp) and isinstance(inner.op, ast.Sub)


def is_named_difference(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return any(keyword in node.id.lower() for keyword in ("diff", "delta", "err"))
    if isinstance(node, ast.Attribute):
        return any(keyword in node.attr.lower() for keyword in ("diff", "delta", "err"))
    return False


def is_manual_tolerance_compare(node: ast.AST) -> bool:
    if not isinstance(node, ast.Compare):
        return False
    if len(node.ops) != 1 or len(node.comparators) != 1:
        return False
    left = node.left
    right = node.comparators[0]
    if isinstance(node.ops[0], (ast.Lt, ast.LtE, ast.Gt, ast.GtE)):
        return (
            (is_abs_difference(left) or is_named_difference(left))
            and is_numeric_literal(right)
        ) or (
            (is_abs_difference(right) or is_named_difference(right))
            and is_numeric_literal(left)
        )
    return False


def contains_failure_signal(body: list[ast.stmt]) -> bool:
    for stmt in body:
        if isinstance(stmt, ast.Raise):
            return True
        for node in ast.walk(stmt):
            if not isinstance(node, ast.Call):
                continue
            call_name = get_call_name(node.func)
            if call_name in {"sys.exit", "pytest.fail"}:
                return True
    return False


def check_test_patterns(
    tree: ast.Module, relative_path: str, kind: str
) -> list[Finding]:
    findings: list[Finding] = []
    is_test_scope = kind == "test"
    is_fixture_bootstrap = relative_path == "tests/conftest.py"
    is_validation_scope = kind in {"test", "example"}

    for node in ast.walk(tree):
        if is_test_scope and not is_fixture_bootstrap and isinstance(node, ast.Call):
            call_name = get_call_name(node.func)
            if call_name in {"tc.set_backend", "tensorcircuit.set_backend"}:
                findings.append(
                    make_finding(
                        "Testing",
                        relative_path,
                        node.lineno,
                        "Test calls tc.set_backend directly",
                        "Use backend fixtures from tests/conftest.py instead of mutating global backend state.",
                    )
                )
            if call_name in {"tc.set_dtype", "tensorcircuit.set_dtype"}:
                findings.append(
                    make_finding(
                        "Testing",
                        relative_path,
                        node.lineno,
                        "Test calls tc.set_dtype directly",
                        "Use the highp fixture or existing dtype fixtures instead of mutating global dtype state.",
                    )
                )

        if not is_validation_scope:
            continue

        if isinstance(node, ast.Assert) and is_manual_tolerance_compare(node.test):
            findings.append(
                make_finding(
                    "Testing",
                    relative_path,
                    node.lineno,
                    "Manual numeric tolerance assertion detected",
                    "Prefer np.testing.assert_allclose or pytest.approx for value-level checks.",
                )
            )
        if isinstance(node, ast.If) and is_manual_tolerance_compare(node.test):
            if contains_failure_signal(node.body):
                findings.append(
                    make_finding(
                        "Testing",
                        relative_path,
                        node.lineno,
                        "Manual numeric diff gate detected",
                        "Replace diff-threshold control flow with assert_allclose-style assertions.",
                    )
                )

    return findings


def analyze_file(
    path: Path,
    required_roots: set[str],
    optional_roots: set[str],
    include_docstrings: bool,
    include_comment_cruft: bool,
    include_import_order: bool,
    package_import_modules: set[str],
    missing_optional_root: str | None,
) -> ParsedFile:
    relative_path = path.relative_to(REPO_ROOT).as_posix()
    kind = classify_file(path)
    content = path.read_text(encoding="utf-8")

    try:
        tree = ast.parse(content, filename=relative_path)
    except SyntaxError as exc:
        finding = make_finding(
            "Maintainability",
            relative_path,
            exc.lineno or 1,
            f"Failed to parse file: {exc.msg}",
            "Fix the syntax error before running deeper sanity checks.",
        )
        return ParsedFile(
            path=path,
            relative_path=relative_path,
            kind=kind,
            tree=None,
            findings=(finding,),
        )

    findings: list[Finding] = []
    findings.extend(check_docstrings(tree, relative_path, kind, include_docstrings))
    findings.extend(check_broad_exceptions(tree, relative_path))
    findings.extend(check_dead_code(tree, relative_path))
    findings.extend(
        check_import_placement(tree, relative_path, kind, include_import_order)
    )
    findings.extend(
        check_optional_dependency_imports(
            tree,
            relative_path,
            kind,
            required_roots,
            optional_roots,
            package_import_modules,
            missing_optional_root,
        )
    )
    findings.extend(check_example_local_imports(tree, relative_path, kind))
    findings.extend(check_unused_imports(tree, relative_path))
    findings.extend(check_comments(content, relative_path, include_comment_cruft))
    findings.extend(check_secrets(content, relative_path))
    findings.extend(check_test_patterns(tree, relative_path, kind))
    return ParsedFile(
        path=path,
        relative_path=relative_path,
        kind=kind,
        tree=tree,
        findings=tuple(findings),
    )


def finding_sort_key(finding: Finding) -> tuple[int, str, int, str]:
    return (
        CATEGORY_ORDER.index(finding.category),
        finding.path,
        finding.line,
        finding.issue,
    )


def format_text_report(findings: Sequence[Finding]) -> str:
    if not findings:
        return "No sanity issues found in the scanned git-tracked Python files."

    ordered_findings = sorted(findings, key=finding_sort_key)
    counts = Counter(finding.category for finding in ordered_findings)
    grouped: dict[str, list[Finding]] = defaultdict(list)
    for finding in ordered_findings:
        grouped[finding.category].append(finding)

    lines = ["Sanity Check Summary"]
    for category in CATEGORY_ORDER:
        lines.append(f"- {category}: {counts.get(category, 0)}")

    for category in CATEGORY_ORDER:
        category_findings = grouped.get(category)
        if not category_findings:
            continue
        lines.append("")
        lines.append(f"== {category} ==")
        for finding in category_findings:
            lines.append(f"{finding.path}:{finding.line}: {finding.issue}")
            lines.append(f"  Recommendation: {finding.recommendation}")

    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    required_roots, optional_roots = load_pyproject_dependency_roots()

    try:
        files = git_tracked_python_files(args.paths)
    except RuntimeError as exc:
        print(f"Failed to resolve git-tracked files: {exc}", file=sys.stderr)
        return 2

    package_import_modules = build_package_import_path_modules(files)
    missing_import_root = get_missing_import_root()
    missing_optional_root = (
        missing_import_root if missing_import_root in optional_roots else None
    )
    parsed_files = [
        analyze_file(
            path,
            required_roots,
            optional_roots,
            args.include_docstrings,
            args.include_comment_cruft,
            args.include_import_order,
            package_import_modules,
            missing_optional_root,
        )
        for path in files
    ]
    findings = [
        finding for parsed_file in parsed_files for finding in parsed_file.findings
    ]

    if args.json:
        ordered_findings = sorted(findings, key=finding_sort_key)
        print(json.dumps([asdict(finding) for finding in ordered_findings], indent=2))
    else:
        print(format_text_report(findings))

    if args.fail_on_findings and findings:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
