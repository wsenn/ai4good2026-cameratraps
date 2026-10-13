"""The experiments bot, run by `.github/workflows/experiments.yml`.

    python3 scripts/experiments_bot.py new     # env: TITLE, NUMBER, AUTHOR
    python3 scripts/experiments_bot.py drop origin/main

`new` turns a PR titled `category: name` into `experiments/NNN-category-name.md`, copied
from `experiments/TEMPLATE.md`, with NNN the PR number. It does nothing if the PR already
has a report, or if the title is a conventional-commit title such as `feat(x): ...`
(tooling PRs). A title with an unknown category gets a comment that lists the categories.

`drop` resets every file the branch changed outside `experiments/` to its state on the
base branch, so only the report is left in the PR.

Results go to `$GITHUB_OUTPUT` when it is set: `created=true|false` and `comment=...`.
Only the standard library is used, so the workflow needs no install step.
"""

from __future__ import annotations

import datetime
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments"
CATEGORIES = ("data", "model", "train", "adapt", "eval", "other")
COMMIT_TYPES = {"feat", "fix", "docs", "chore", "ci", "refactor", "test", "style", "perf",
                "build", "revert"}
TITLE = re.compile(r"\s*([A-Za-z]+)\s*:\s*(.+?)\s*$")


def output(**values: str) -> None:
    """Write `key=value` lines to `$GITHUB_OUTPUT`, and echo them."""
    path = os.environ.get("GITHUB_OUTPUT")
    lines = [f"{k}={v}" for k, v in values.items()]
    print("\n".join(lines))
    if path:
        with open(path, "a") as fh:
            fh.write("\n".join(lines) + "\n")


def slug(text: str) -> str:
    """`Rand Augment (strong)!` -> `rand-augment-strong`, at most 40 characters."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40].strip("-")


def new(title: str, number: int, author: str) -> None:
    """Create the report for PR `number` from its title.

    Args:
        title: the PR title, `category: name`.
        number: the PR number.
        author: the GitHub login of the PR author.
    """
    m = TITLE.match(title)
    category = m[1].lower() if m else ""
    if not m or category in COMMIT_TYPES or "(" in title.split(":")[0]:
        output(created="false", comment="")
        return
    if category not in CATEGORIES:
        output(created="false", comment=(
            f"Unknown category `{m[1]}`. Rename the PR to `category: name` with one of: "
            + ", ".join(f"`{c}`" for c in CATEGORIES) + ". The bot then adds the report."))
        return
    if list(EXP.glob(f"{number:03d}-*.md")):
        output(created="false", comment="")
        return
    name = slug(m[2]) or "experiment"
    path = EXP / f"{number:03d}-{category}-{name}.md"
    text = (EXP / "TEMPLATE.md").read_text()
    text = (text.replace("TITLE", f"{number:03d} {category}: {m[2].strip()}", 1)
                .replace("NUMBER", str(number), 1)
                .replace("OWNER", author, 1))
    path.write_text(text)
    print(f"wrote {path.relative_to(ROOT)} ({datetime.date.today()})")
    output(created="true", comment="", report=str(path.relative_to(ROOT)))


def git(*args: str) -> str:
    """Run git and return its stdout."""
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True,
                          text=True).stdout


def drop(base: str) -> None:
    """Undo every change outside `experiments/` against `base`.

    Args:
        base: the base branch, e.g. `origin/main`.
    """
    changed = git("diff", "--name-only", "--no-renames", f"{base}...HEAD").split()
    undone = 0
    for path in changed:
        if path.startswith("experiments/"):
            continue
        on_base = subprocess.run(["git", "cat-file", "-e", f"{base}:{path}"], cwd=ROOT,
                                 capture_output=True).returncode == 0
        if on_base:
            git("checkout", base, "--", path)
        elif (ROOT / path).exists():
            git("rm", "-q", "--", path)
        undone += 1
        print(f"reset {path}")
    output(undone=str(undone))


def main() -> None:
    """Command-line entry point."""
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "new":
        new(os.environ["TITLE"], int(os.environ["NUMBER"]), os.environ.get("AUTHOR", ""))
    elif cmd == "drop" and len(sys.argv) == 3:
        drop(sys.argv[2])
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
