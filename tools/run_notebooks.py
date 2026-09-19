#!/usr/bin/env python3
"""
Execute every solution notebook and fail if any cell raises.

This is what proves the exercises are solvable: the solutions run top to bottom
and every ``check.*`` call inside them passes.

    python tools/run_notebooks.py            # all of them
    python tools/run_notebooks.py 08 09      # only those prefixes
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOLUTION_DIR = ROOT / 'notebooks' / 'solutions'


def main() -> int:
    import nbformat
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError

    wanted = sys.argv[1:]
    paths = sorted(SOLUTION_DIR.glob('*.ipynb'))
    if wanted:
        paths = [p for p in paths if any(p.name.startswith(w) for w in wanted)]
    if not paths:
        print('no notebooks matched', file=sys.stderr)
        return 1

    failures = []
    for path in paths:
        notebook = nbformat.read(path, as_version=4)
        client = NotebookClient(
            notebook,
            timeout=600,
            kernel_name='python3',
            resources={'metadata': {'path': str(ROOT / 'notebooks')}},
        )
        started = time.time()
        try:
            client.execute()
            cells = sum(1 for c in notebook.cells if c.cell_type == 'code')
            print(f'✓ {path.name}  ({cells} code cells, {time.time() - started:.1f}s)')
        except CellExecutionError as exc:
            print(f'✗ {path.name}')
            message = str(exc).strip().splitlines()
            for line in message[-14:]:
                print(f'    {line}')
            failures.append(path.name)

    print()
    if failures:
        print(f'{len(paths) - len(failures)}/{len(paths)} notebooks ran clean; '
              f'failed: {", ".join(failures)}', file=sys.stderr)
        return 1
    print(f'{len(paths)}/{len(paths)} solution notebooks ran clean')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
