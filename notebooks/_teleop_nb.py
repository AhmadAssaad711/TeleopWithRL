"""Small, dependency-light helpers shared by the project notebooks.

Notebook cells should orchestrate experiments and display results. The actual
environment, training, evaluation, and serialization logic belongs in
``matlab_env_python_replica``. ``run_notebook_command`` and ``run_module``
make that boundary explicit while keeping command output visible in Jupyter.
"""

from __future__ import annotations

import csv
import html
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable

from IPython.display import HTML, Image, Markdown, display


def _looks_like_repo_root(path: Path) -> bool:
    return (path / "matlab_env_python_replica").exists() and (path / "notebooks" / "_teleop_nb.py").exists()


def find_repo_root(start: str | Path | None = None) -> Path:
    path = Path(start or Path.cwd()).resolve()
    for candidate in [path, *path.parents]:
        if _looks_like_repo_root(candidate):
            return candidate
        nested = candidate / "TeleopWithRL"
        if _looks_like_repo_root(nested):
            return nested
    raise RuntimeError("Could not find TeleopWithRL repo root from current working directory.")


def repo_paths(start: str | Path | None = None) -> dict[str, Path]:
    repo = find_repo_root(start)
    return {
        "repo": repo,
        "matlab_env_python_replica": repo / "matlab_env_python_replica",
        "notebooks": repo / "notebooks",
        "results_index": repo / "results_index",
        "matlab_results": repo / "matlab_env_python_replica" / "results",
        "dqn_results": repo / "matlab_env_python_replica" / "dqn_experiments" / "results",
        "ql_results": repo / "matlab_env_python_replica" / "ql_experiments" / "results",
    }


def _python_runs(path: Path) -> bool:
    try:
        completed = subprocess.run(
            [str(path), "-c", "import sys"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0


def project_python_executable(start: str | Path | None = None) -> Path:
    repo = find_repo_root(start)
    workspace = repo.parent
    python_rel = Path(".venv") / ("Scripts" if os.name == "nt" else "bin") / ("python.exe" if os.name == "nt" else "python")
    for root in (workspace, repo):
        candidate = (root / python_rel).resolve()
        if candidate.exists() and _python_runs(candidate):
            return candidate
    return Path(sys.executable).resolve()


def run_notebook_command(
    command: Iterable[str],
    cwd: str | Path | None = None,
    **kwargs,
) -> subprocess.CompletedProcess:
    """Run a repository command from a notebook and return its completed process.

    Parameters
    ----------
    command:
        The executable plus arguments. In experiment notebooks this is usually
        a command assembled from ``project_python_executable`` and a package
        script entry point.
    cwd:
        Working directory for the command. It defaults to the repository root.
    kwargs:
        Optional keyword arguments forwarded to ``subprocess.run``. ``check``
        defaults to ``True`` so a failed experiment stops the notebook cell.
    """
    repo = find_repo_root(cwd)
    options = dict(kwargs)
    options.setdefault("check", True)
    return subprocess.run(list(command), cwd=str(cwd or repo), **options)


def run_module(
    module: str,
    args: Iterable[str] = (),
    cwd: str | Path | None = None,
    **kwargs,
) -> subprocess.CompletedProcess:
    """Run a Python module through the repository's selected interpreter.

    This is the preferred notebook-facing wrapper for commands such as
    ``TeleopWithRL.matlab_env_python_replica.dqn.scripts.run_experiments``.
    The module owns experiment behavior; the notebook supplies only arguments
    and then reads the generated result files.
    """
    repo = find_repo_root(cwd)
    command = [str(project_python_executable(repo)), "-m", str(module), *map(str, args)]
    return run_notebook_command(command, cwd=repo, **kwargs)


def show_markdown(text: str) -> None:
    display(Markdown(text))


def _rows_to_html(rows: list[dict], max_rows: int = 20) -> str:
    if not rows:
        return "<p><em>No rows found.</em></p>"
    headers = list(rows[0].keys())
    body_rows = rows[:max_rows]
    parts = [
        "<table>",
        "<thead><tr>",
        *[f"<th>{html.escape(str(h))}</th>" for h in headers],
        "</tr></thead><tbody>",
    ]
    for row in body_rows:
        parts.append("<tr>")
        parts.extend(f"<td>{html.escape(str(row.get(h, '')))}</td>" for h in headers)
        parts.append("</tr>")
    parts.append("</tbody></table>")
    if len(rows) > max_rows:
        parts.append(f"<p><em>Showing {max_rows} of {len(rows)} rows.</em></p>")
    return "".join(parts)


def show_rows(rows: list[dict], title: str | None = None, max_rows: int = 20) -> None:
    if title:
        show_markdown(f"### {title}")
    display(HTML(_rows_to_html(rows, max_rows=max_rows)))


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def show_json(path: str | Path, title: str | None = None, keys: Iterable[str] | None = None) -> None:
    file_path = Path(path)
    if title:
        show_markdown(f"### {title}")
    if not file_path.exists():
        show_markdown(f"_Missing file_: `{file_path}`")
        return
    data = load_json(file_path)
    if keys is not None:
        data = {key: data.get(key) for key in keys}
    show_rows([data], title=f"`{file_path}`", max_rows=1)


def load_csv_rows(path: str | Path) -> list[dict]:
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def show_csv(path: str | Path, title: str | None = None, max_rows: int = 20) -> None:
    file_path = Path(path)
    if title:
        show_markdown(f"### {title}")
    if not file_path.exists():
        show_markdown(f"_Missing file_: `{file_path}`")
        return
    rows = load_csv_rows(file_path)
    show_rows(rows, title=f"`{file_path}`", max_rows=max_rows)


def show_image(path: str | Path, title: str | None = None, width: int = 1200) -> None:
    file_path = Path(path)
    if title:
        show_markdown(f"### {title}")
    if not file_path.exists():
        show_markdown(f"_Missing image_: `{file_path}`")
        return
    display(Image(filename=str(file_path), width=width))


def subdirs(path: str | Path, limit: int = 25) -> list[dict]:
    root = Path(path)
    if not root.exists():
        return []
    return [{"name": child.name, "path": str(child)} for child in sorted(root.iterdir()) if child.is_dir()][:limit]


def count_tree(root: str | Path) -> dict:
    path = Path(root)
    return {
        "root": str(path),
        "dirs": sum(1 for item in path.rglob("*") if item.is_dir()),
        "files": sum(1 for item in path.rglob("*") if item.is_file()),
    }
