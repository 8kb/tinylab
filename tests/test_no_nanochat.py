"""
Mechanical guard against an accidental nanochat import: AST-scans every .py file under both
tinylab/ (the package) and tests/ (this directory) and asserts none of them import nanochat or
scripts. tinylab is a host application and needs no standalone guard like the three subsystems
have -- but nanochat sits right next to it on disk and often on PYTHONPATH during development, so
an accidental `import nanochat` is easy to write and would silently work here even though it
violates tinylab's own "ported, not imported" rule (see AGENTS.md: nanochat is a virtual uv
project and can't be a real dependency). Adapted from benchcore/tests/test_standalone.py.

A docstring or comment mentioning "nanochat" is fine (and common, since most of tinylab's modules
say where they were ported from); only actual import statements are checked.
"""
import ast
import os

FORBIDDEN_TOP_LEVEL_PACKAGES = {"nanochat", "scripts"}

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCANNED_DIRS = [os.path.join(REPO_ROOT, "tinylab"), os.path.join(REPO_ROOT, "tests")]


def _iter_python_files(root):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for name in filenames:
            if name.endswith(".py"):
                yield os.path.join(dirpath, name)


def _imported_top_level_packages(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=file_path)
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                found.add(node.module.split(".")[0])
    return found


def test_no_python_file_under_tinylab_imports_nanochat():
    violations = {}
    for scanned_dir in SCANNED_DIRS:
        for file_path in _iter_python_files(scanned_dir):
            found = _imported_top_level_packages(file_path) & FORBIDDEN_TOP_LEVEL_PACKAGES
            if found:
                violations[os.path.relpath(file_path, REPO_ROOT)] = sorted(found)
    assert not violations, f"tinylab/ and tests/ must have zero imports from nanochat, but found: {violations}"
