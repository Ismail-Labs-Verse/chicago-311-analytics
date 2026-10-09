"""
Stage 4 — Notebook build + execution.

The four analysis notebooks are authored as percent-format Python in
`src/notebook_src/*.py`. That keeps them diffable and reviewable, while
`jupytext`-style cell markers let this script produce real `.ipynb` files.

Every notebook is then EXECUTED headlessly with nbclient. Committed notebooks
therefore carry real outputs produced by the code in this repository, rather
than being empty shells or hand-pasted results.

Run:  python src/generate_notebooks.py            # build + execute all
      python src/generate_notebooks.py --no-exec  # build only
      python src/generate_notebooks.py 01 02      # selected notebooks
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

SRC_DIR = C.PROJECT_ROOT / "src" / "notebook_src"
OUT_DIR = C.PROJECT_ROOT / "notebooks"
OUT_DIR.mkdir(parents=True, exist_ok=True)

KERNEL_CANDIDATES = ("chicago311", "python3")
TIMEOUT = 3600


# ---------------------------------------------------------------------------
# percent-format parsing
# ---------------------------------------------------------------------------
def parse_cells(text: str) -> list[tuple[str, str]]:
    """Split percent-format source into (kind, source) pairs.

    A cell begins at `# %%` and runs to the next `# %%`. `# %% [markdown]`
    starts a markdown cell; everything else is code.
    """
    cells: list[tuple[str, str]] = []
    marker: str | None = None
    buf: list[str] = []

    for line in text.split("\n"):
        if line.startswith("# %%"):
            if marker is not None:
                cells.append((marker, "\n".join(buf)))
            marker = "markdown" if "[markdown]" in line else "code"
            buf = []
        else:
            buf.append(line)
    if marker is not None:
        cells.append((marker, "\n".join(buf)))
    return cells


def clean_markdown(src: str) -> str:
    """Strip the leading `# ` comment prefix from markdown cell bodies."""
    out = []
    for line in src.strip("\n").split("\n"):
        if line.startswith("# "):
            out.append(line[2:])
        elif line.strip() == "#":
            out.append("")
        else:
            out.append(line)
    return "\n".join(out).strip()


def clean_code(src: str) -> str:
    return src.strip("\n") + "\n"


def build_notebook(py_file: Path) -> nbf.NotebookNode:
    cells = parse_cells(py_file.read_text(encoding="utf-8"))
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"display_name": "Python (chicago311)",
                       "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    built = []
    for kind, src in cells:
        if not src.strip():
            continue
        if kind == "markdown":
            built.append(nbf.v4.new_markdown_cell(clean_markdown(src)))
        else:
            built.append(nbf.v4.new_code_cell(clean_code(src)))
    nb.cells = built
    return nb


# ---------------------------------------------------------------------------
# execution
# ---------------------------------------------------------------------------
def resolve_kernel() -> str:
    from jupyter_client.kernelspec import KernelSpecManager
    available = KernelSpecManager().find_kernel_specs()
    for name in KERNEL_CANDIDATES:
        if name in available:
            return name
    return "python3"


def execute_notebook(nb: nbf.NotebookNode, kernel: str, wd: Path) -> nbf.NotebookNode:
    from nbclient import NotebookClient

    client = NotebookClient(
        nb,
        timeout=TIMEOUT,
        kernel_name=kernel,
        resources={"metadata": {"path": str(wd)}},
        allow_errors=False,
    )
    client.execute()
    return nb


def main() -> None:
    ap = argparse.ArgumentParser(description="build and execute the analysis notebooks")
    ap.add_argument("--no-exec", action="store_true", help="build without executing")
    ap.add_argument("only", nargs="*", help="notebook prefixes to build (e.g. 01 02)")
    args = ap.parse_args()

    sources = sorted(SRC_DIR.glob("*.py"))
    if args.only:
        sources = [p for p in sources if any(p.stem.startswith(o) for o in args.only)]
    if not sources:
        print(f"no notebook sources found in {SRC_DIR}")
        sys.exit(1)

    kernel = resolve_kernel()
    print(f"notebook sources : {len(sources)}")
    print(f"output directory : {OUT_DIR}")
    print(f"jupyter kernel   : {kernel}"
          f"{'' if not args.no_exec else '  (execution disabled)'}\n")

    failures = []
    for src in sources:
        out = OUT_DIR / f"{src.stem}.ipynb"
        t0 = time.time()
        print(f"--- {src.stem} ---", flush=True)
        try:
            nb = build_notebook(src)
            print(f"    built {len(nb.cells)} cells")
            if not args.no_exec:
                nb = execute_notebook(nb, kernel, OUT_DIR)
                code_cells = [c for c in nb.cells if c.cell_type == "code"]
                with_out = sum(1 for c in code_cells if c.get("outputs"))
                print(f"    executed {len(code_cells)} code cells "
                      f"({with_out} produced output) in {time.time() - t0:.0f}s")
            nbf.write(nb, str(out))
            print(f"    wrote {out.relative_to(C.PROJECT_ROOT)}")
        except Exception as exc:
            failures.append((src.stem, exc))
            print(f"    FAILED: {type(exc).__name__}: {str(exc)[:400]}", flush=True)

    print()
    if failures:
        print(f"{len(failures)} notebook(s) failed:")
        for name, exc in failures:
            print(f"  - {name}: {type(exc).__name__}: {str(exc)[:200]}")
        sys.exit(1)
    print(f"all {len(sources)} notebooks built"
          f"{'' if args.no_exec else ' and executed'} successfully")


if __name__ == "__main__":
    main()
