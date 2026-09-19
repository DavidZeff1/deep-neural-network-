#!/usr/bin/env python3
"""
Build the exercise notebooks from their sources.

Each notebook is authored once, in ``notebooks/src/NN-name.py``, using the
"percent" cell format:

    # %% [markdown]
    # ## A heading
    # Some prose.

    # %%
    import numpy as np

Solution code is wrapped in markers:

    ### BEGIN SOLUTION
    return np.maximum(0.0, z)
    ### END SOLUTION

From one source this script writes two notebooks:

*  ``notebooks/solutions/NN-name.ipynb``  — complete and runnable
*  ``notebooks/NN-name.ipynb``            — the same, with solution bodies
                                            replaced by a stub to fill in

Run ``python tools/build_notebooks.py`` to regenerate, or ``--check`` to verify
the committed notebooks match their sources.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / 'notebooks' / 'src'
STUDENT_DIR = ROOT / 'notebooks'
SOLUTION_DIR = ROOT / 'notebooks' / 'solutions'

BEGIN = '### BEGIN SOLUTION'
END = '### END SOLUTION'
PROMPT = '### PROMPT:'
HINT = '### HINT:'

STUB = 'raise NotImplementedError({prompt!r})'


class BuildError(Exception):
    pass


def split_cells(text: str):
    """Split percent-format source into ``(kind, lines)`` pairs."""
    cells = []
    kind, buffer = None, []
    for raw in text.splitlines():
        if raw.startswith('# %%'):
            if kind is not None:
                cells.append((kind, buffer))
            kind = 'markdown' if '[markdown]' in raw else 'code'
            buffer = []
        elif kind is not None:
            buffer.append(raw)
    if kind is not None:
        cells.append((kind, buffer))
    return cells


def markdown_source(lines):
    out = []
    for line in lines:
        if line.startswith('# '):
            out.append(line[2:])
        elif line.strip() == '#':
            out.append('')
        elif not line.strip():
            out.append('')
        else:
            raise BuildError(f'markdown cell has a non-comment line: {line!r}')
    while out and not out[0].strip():
        out.pop(0)
    while out and not out[-1].strip():
        out.pop()
    return '\n'.join(out)


def code_source(lines, keep_solutions: bool):
    out = []
    in_solution = False
    prompt = 'Fill this in, then run the check below.'
    indent = ''
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(PROMPT):
            prompt = stripped[len(PROMPT):].strip()
            continue
        if stripped.startswith(HINT):
            out.append(f'{line[:len(line) - len(line.lstrip())]}# hint: {stripped[len(HINT):].strip()}')
            continue
        if stripped == BEGIN:
            if in_solution:
                raise BuildError('nested BEGIN SOLUTION')
            in_solution = True
            indent = line[:len(line) - len(line.lstrip())]
            if not keep_solutions:
                out.append(f'{indent}# YOUR CODE HERE')
                out.append(indent + STUB.format(prompt=prompt))
            continue
        if stripped == END:
            if not in_solution:
                raise BuildError('END SOLUTION without BEGIN')
            in_solution = False
            prompt = 'Fill this in, then run the check below.'
            continue
        if in_solution and not keep_solutions:
            continue
        out.append(line)
    if in_solution:
        raise BuildError('BEGIN SOLUTION was never closed')
    while out and not out[0].strip():
        out.pop(0)
    while out and not out[-1].strip():
        out.pop()
    return '\n'.join(out)


def _cell_id(stem: str, index: int) -> str:
    """A stable id, so rebuilding an unchanged source produces an identical file."""
    return hashlib.sha1(f'{stem}:{index}'.encode()).hexdigest()[:8]


def build_notebook(path: Path, keep_solutions: bool) -> dict:
    cells = []
    for index, (kind, lines) in enumerate(split_cells(path.read_text())):
        if kind == 'markdown':
            source = markdown_source(lines)
            if not source:
                continue
            cells.append({
                'cell_type': 'markdown',
                'id': _cell_id(path.stem, index),
                'metadata': {},
                'source': source.splitlines(True),
            })
        else:
            source = code_source(lines, keep_solutions)
            if not source:
                continue
            cells.append({
                'cell_type': 'code',
                'execution_count': None,
                'id': _cell_id(path.stem, index),
                'metadata': {},
                'outputs': [],
                'source': source.splitlines(True),
            })
    return {
        'cells': cells,
        'metadata': {
            'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
            'language_info': {'name': 'python', 'version': '3'},
        },
        'nbformat': 4,
        'nbformat_minor': 5,
    }


def serialise(notebook: dict) -> str:
    return json.dumps(notebook, indent=1, ensure_ascii=False) + '\n'


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true',
                        help='verify the committed notebooks match their sources')
    args = parser.parse_args()

    sources = sorted(SRC_DIR.glob('*.py'))
    if not sources:
        print(f'no sources found in {SRC_DIR}', file=sys.stderr)
        return 1

    STUDENT_DIR.mkdir(parents=True, exist_ok=True)
    SOLUTION_DIR.mkdir(parents=True, exist_ok=True)

    stale = []
    for source in sources:
        for keep, target_dir in ((False, STUDENT_DIR), (True, SOLUTION_DIR)):
            try:
                text = serialise(build_notebook(source, keep))
            except BuildError as exc:
                print(f'{source.name}: {exc}', file=sys.stderr)
                return 1
            target = target_dir / f'{source.stem}.ipynb'
            if args.check:
                if not target.exists() or target.read_text() != text:
                    stale.append(target.relative_to(ROOT))
            else:
                target.write_text(text)

    if args.check:
        if stale:
            print('These notebooks are out of date with notebooks/src:', file=sys.stderr)
            for path in stale:
                print(f'  {path}', file=sys.stderr)
            print('\nRun: python tools/build_notebooks.py', file=sys.stderr)
            return 1
        print(f'✓ {len(sources) * 2} notebooks are up to date with their sources')
    else:
        print(f'✓ built {len(sources)} exercise notebooks and {len(sources)} solution notebooks')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
