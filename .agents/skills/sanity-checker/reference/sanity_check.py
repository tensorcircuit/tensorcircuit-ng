import ast
import os
import sys


def check_file(filepath):
    issues = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        tree = ast.parse(content, filename=filepath)
    except Exception as e:
        return [f"Failed to parse: {e}"]

    lines = content.split("\n")

    # Check for missing docstrings in public functions and classes
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not node.name.startswith("_"):
                if not ast.get_docstring(node):
                    issues.append(
                        f"Line {node.lineno}: Missing docstring in public API '{node.name}'"
                    )

        # Check for broad exceptions
        if isinstance(node, ast.ExceptHandler):
            if node.type is None:
                issues.append(f"Line {node.lineno}: Broad exception 'except:' used")
            elif isinstance(node.type, ast.Name) and node.type.id == "Exception":
                issues.append(
                    f"Line {node.lineno}: Broad exception 'except Exception:' used"
                )

        # Check for dead code (if False)
        if isinstance(node, ast.If):
            if isinstance(node.test, ast.Constant) and node.test.value in (False, 0):
                issues.append(f"Line {node.lineno}: Dead code 'if False/0:' used")

    # Check for secrets and magic numbers (heuristics)
    for i, line in enumerate(lines, 1):
        line_lower = line.lower()
        if "api_key" in line_lower and "=" in line_lower:
            issues.append(f"Line {i}: Potential secret 'api_key' assignment")
        if "password" in line_lower and "=" in line_lower:
            issues.append(f"Line {i}: Potential secret 'password' assignment")

    return issues


def main():
    report = {}
    for root, dirs, files in os.walk("tensorcircuit"):
        for file in files:
            if file.endswith(".py"):
                filepath = os.path.join(root, file)
                issues = check_file(filepath)
                if issues:
                    report[filepath] = issues

    for root, dirs, files in os.walk("tests"):
        for file in files:
            if file.endswith(".py"):
                filepath = os.path.join(root, file)
                issues = check_file(filepath)
                # Tests might not need docstrings as strictly, but let's just log broad exceptions
                filtered_issues = [
                    iss for iss in issues if "Missing docstring" not in iss
                ]
                if filtered_issues:
                    report[filepath] = filtered_issues

    for filepath, issues in report.items():
        print(f"\n--- {filepath} ---")
        for issue in issues:
            print(issue)


if __name__ == "__main__":
    main()
