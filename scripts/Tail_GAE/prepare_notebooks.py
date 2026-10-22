#!/usr/bin/env python3
"""Generate the 10 seed notebooks of every condition from its seed_0 template.

The template is the seed_0 notebook as archived in the original repository.
Only two things change per seed: ``EXPERIMENT_SEED`` in cell 0 and the
``metadata.rgppo.seed`` / ``metadata.phase7.seed`` fields.  The template must
already use the dynamic root block (``RGPPO_ROOT`` or an upward search from the
working directory); hardcoded absolute paths are rejected.  Proxy/IP-check
cells are dropped, and the frozen Phase 9 calendar is read from
``data/phase9/``.

Usage: python prepare_notebooks.py [--check]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ASSIGN_ROOT = 'PROJECT_ROOT = Path(os.environ.get("RGPPO_ROOT", str(Path.cwd())))'
UPWARD_SEARCH = """while not (PROJECT_ROOT / "data" / "new_version_2026").exists() and PROJECT_ROOT != PROJECT_ROOT.parent:
    PROJECT_ROOT = PROJECT_ROOT.parent"""

SPECS = [
    ("1250", "1250 Tail_GAE", "Tail_GAE_1250_r0_unified_1200_control_seed_{k}.ipynb"),
    ("1270b", "1270b Tail_GAE", "Tail_GAE_1270b_s1_corrected_4c_seed_{k}.ipynb"),
    ("1321b_ctrl", "1321b Tail_GAE", "Tail_GAE_1321b_ctrl_seed_{k}.ipynb"),
    ("1321b_lo", "1321b Tail_GAE", "Tail_GAE_1321b_lo_seed_{k}.ipynb"),
    ("1321b_hi", "1321b Tail_GAE", "Tail_GAE_1321b_hi_seed_{k}.ipynb"),
]


def _text(cell: dict) -> str:
    return "".join(cell.get("source", []))


def _set_text(cell: dict, value: str) -> None:
    cell["source"] = value.splitlines(keepends=True)


def _rewrite(cell_source: str) -> str:
    src = cell_source
    if any(marker in src for marker in ("/root/", "/dev/shm/")):
        raise RuntimeError("hardcoded absolute path remains in a code cell")
    src = re.sub(
        r'PHASE9_OUTPUT_ROOT = PROJECT_ROOT / "Results" / "Tail_GAE" / "(1321b_(?:ctrl|lo|hi))"',
        r'PHASE9_OUTPUT_ROOT = Path(os.environ.get("RGPPO_RESULTS_ROOT", str(PROJECT_ROOT / "Results"))) / "Tail_GAE" / "\1"',
        src,
    )
    return src


def generate() -> list[tuple[str, int, str, str]]:
    manifest: list[tuple[str, int, str, str]] = []
    for condition, folder, pattern in SPECS:
        template_path = ROOT / "ultimate_experiment" / folder / pattern.format(k=0)
        template = json.loads(template_path.read_text())
        for seed in range(10):
            nb = json.loads(json.dumps(template))
            nb["cells"] = [
                cell
                for cell in nb["cells"]
                if "ipwho.is" not in _text(cell) and "http_proxy" not in _text(cell)
            ]
            if nb["cells"] and nb["cells"][0]["cell_type"] == "code":
                source = _text(nb["cells"][0])
                source, count = re.subn(
                    r"EXPERIMENT_SEED\s*=\s*\d+",
                    f"EXPERIMENT_SEED = {seed}",
                    source,
                    count=1,
                )
                if count != 1:
                    raise RuntimeError(
                        f"{condition} seed {seed}: EXPERIMENT_SEED not found in cell 0"
                    )
                _set_text(nb["cells"][0], source)
            for cell in nb["cells"]:
                if cell["cell_type"] != "code":
                    continue
                _set_text(cell, _rewrite(_text(cell)))
                cell["outputs"] = []
                cell["execution_count"] = None
            metadata = nb.setdefault("metadata", {})
            metadata["rgppo"] = {"condition": condition, "seed": seed}
            if "phase7" in metadata:
                metadata["phase7"]["seed"] = seed
            metadata.pop("widgets", None)
            out_path = ROOT / "ultimate_experiment" / folder / pattern.format(k=seed)
            out_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
            manifest.append(
                (
                    condition,
                    seed,
                    str(out_path.relative_to(ROOT)),
                    hashlib.sha256(out_path.read_bytes()).hexdigest(),
                )
            )
    return manifest


def check() -> list[str]:
    problems: list[str] = []
    for condition, folder, pattern in SPECS:
        for seed in range(10):
            path = ROOT / "ultimate_experiment" / folder / pattern.format(k=seed)
            if not path.exists():
                problems.append(f"{path.relative_to(ROOT)}: missing")
                continue
            nb = json.loads(path.read_text())
            source = "".join(nb["cells"][0]["source"])
            source_all = "\n".join(_text(c) for c in nb["cells"])
            code_all = "\n".join(
                _text(c) for c in nb["cells"] if c["cell_type"] == "code"
            )
            for marker in ("ipwho.is", "http_proxy"):
                if marker in source_all:
                    problems.append(
                        f"{path.relative_to(ROOT)}: forbidden marker {marker!r} remains"
                    )
            for marker in ("/root/", "/dev/shm/"):
                if marker in code_all:
                    problems.append(
                        f"{path.relative_to(ROOT)}: absolute path {marker!r} remains"
                    )
            if ASSIGN_ROOT not in code_all:
                problems.append(f"{path.relative_to(ROOT)}: dynamic root missing")
            if UPWARD_SEARCH not in code_all:
                problems.append(f"{path.relative_to(ROOT)}: upward search missing")
            if f"EXPERIMENT_SEED = {seed}" not in source:
                problems.append(f"{path.relative_to(ROOT)}: seed constant mismatch")
            if nb.get("metadata", {}).get("rgppo") != {
                "condition": condition,
                "seed": seed,
            }:
                problems.append(f"{path.relative_to(ROOT)}: rgppo metadata mismatch")
            if "phase7" in nb.get("metadata", {}) and nb["metadata"]["phase7"].get("seed") != seed:
                problems.append(f"{path.relative_to(ROOT)}: phase7 seed mismatch")
            if "1321b_" in condition and (
                '"phase9" / "gated_rf20_calendar.csv"' not in code_all
                or "dose_control_calendars" in code_all
            ):
                problems.append(f"{path.relative_to(ROOT)}: calendar path not rewritten")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        problems = check()
        print("problems:", problems if problems else "none")
        return 1 if problems else 0
    manifest = generate()
    manifest_path = ROOT / "provenance" / "notebook_manifest.tsv"
    manifest_path.write_text(
        "condition\tseed\tpath\tsha256\n"
        + "\n".join("\t".join(map(str, row)) for row in manifest)
        + "\n"
    )
    print(f"wrote {len(manifest)} notebooks; manifest: {manifest_path}")
    problems = check()
    print("problems:", problems if problems else "none")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
