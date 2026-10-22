#!/usr/bin/env python3
"""Prepare the ten AUST ensemble (0101) seed notebooks in this repository.

The experiment definition is the last committed 0101 notebook; only path and
launch metadata are adapted:

- ``PROJECT_ROOT`` comes from ``RGPPO_ROOT``, with an upward search from the
  working directory when the variable is unset (the kernel runs from the
  notebook directory, but the data and scripts live at the repository root).
- ``NOTEBOOK_FILE`` comes from ``RGPPO_NOTEBOOK``; the audit uses it only for
  the archived copy.
- the audit output root comes from ``RGPPO_RESULTS_ROOT``.
- the early ``check_and_make_directories`` call is dropped; ``audit_ensemble_run``
  creates and isolates the FinRL output directories inside each run directory.
- the kernel is the FinRL Jupyter kernel.

Usage: python prepare_0101_notebooks.py [--check]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "ultimate_experiment" / "0101 AUST_ensemble + original"
PATTERN = "AUST_seed_{seed}.ipynb"

ASSIGN_ROOT = 'PROJECT_ROOT = Path(os.environ.get("RGPPO_ROOT", str(Path.cwd())))'
UPWARD_SEARCH = """while not (PROJECT_ROOT / "data" / "new_version_2026").exists() and PROJECT_ROOT != PROJECT_ROOT.parent:
    PROJECT_ROOT = PROJECT_ROOT.parent"""
INSERT_ROOT = """if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))"""
DYNAMIC_ROOT = "\n".join((ASSIGN_ROOT, UPWARD_SEARCH, INSERT_ROOT)) + "\n"
SCRIPT_IMPORT = "from scripts.ensemble.ensemble_audit import"


def _text(cell: dict) -> str:
    return "".join(cell.get("source", []))


def _set_text(cell: dict, value: str) -> None:
    cell["source"] = value.splitlines(keepends=True)


def _rewrite(source: str) -> str:
    source = source.replace(
        "PROJECT_ROOT = Path.cwd().resolve()",
        ASSIGN_ROOT,
    )
    source = source.replace(
        'PROJECT_ROOT = Path(os.environ.get("RGPPO_ROOT", str(Path.cwd().resolve())))',
        ASSIGN_ROOT,
    )
    if (
        SCRIPT_IMPORT in source
        and "from pathlib import Path" in source
        and UPWARD_SEARCH not in source
    ):
        source = source.replace(
            "from pathlib import Path\n",
            "from pathlib import Path\n\nimport os\n\n" + DYNAMIC_ROOT,
            1,
        )
    if ASSIGN_ROOT in source and UPWARD_SEARCH not in source:
        source = source.replace(
            ASSIGN_ROOT,
            ASSIGN_ROOT + "\n" + UPWARD_SEARCH,
            1,
        )
    source = source.replace(
        'NOTEBOOK_FILE = PROJECT_ROOT / "AUST_data2026_seed_1.ipynb"',
        'NOTEBOOK_FILE = Path(os.environ.get("RGPPO_NOTEBOOK", '
        'str(PROJECT_ROOT / "ultimate_experiment" / "0101 AUST_ensemble + original" '
        '/ "AUST_seed_0.ipynb")))',
    )
    source = source.replace(
        'output_root=PROJECT_ROOT / "Results/AUST/AUST_data2026"',
        'output_root=Path(os.environ.get("RGPPO_RESULTS_ROOT", '
        'str(PROJECT_ROOT / "Results"))) / "01 AUST" / "0101 AUST_data2026"',
    )
    source = re.sub(
        r"^check_and_make_directories\(\[DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR\]\)\s*$",
        "# check_and_make_directories(...) is handled per run by audit_ensemble_run",
        source,
        flags=re.MULTILINE,
    )
    for marker in ("/root/", "/dev/shm/", "ipwho.is", "http_proxy"):
        if marker in source:
            raise RuntimeError(f"forbidden absolute path or proxy marker {marker!r}")
    return source


def generate() -> list[tuple[str, int, str, str]]:
    template_path = FOLDER / PATTERN.format(seed=0)
    template = json.loads(template_path.read_text())
    manifest: list[tuple[str, int, str, str]] = []
    for seed in range(10):
        nb = json.loads(json.dumps(template))
        for cell in nb["cells"]:
            if cell["cell_type"] != "code":
                continue
            source = _rewrite(_text(cell))
            source, count = re.subn(
                r"^seed = \d+(\s+#.*)?$",
                f"seed = {seed}",
                source,
                count=1,
                flags=re.MULTILINE,
            )
            _set_text(cell, source)
            cell["outputs"] = []
            cell["execution_count"] = None
        metadata = nb.setdefault("metadata", {})
        metadata["rgppo"] = {"condition": "0101", "seed": seed}
        metadata["kernelspec"] = {
            "display_name": "FinRL AUST Exploratory",
            "language": "python",
            "name": "finrl-aust-exploratory",
        }
        out_path = FOLDER / PATTERN.format(seed=seed)
        out_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
        manifest.append(
            (
                "0101",
                seed,
                str(out_path.relative_to(ROOT)),
                hashlib.sha256(out_path.read_bytes()).hexdigest(),
            )
        )
    return manifest


def check() -> list[str]:
    problems: list[str] = []
    for seed in range(10):
        path = FOLDER / PATTERN.format(seed=seed)
        if not path.exists():
            problems.append(f"{path.name}: missing")
            continue
        nb = json.loads(path.read_text())
        code = "\n".join(
            "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"
        )
        for marker in ("/root/", "/dev/shm/", "ipwho.is", "http_proxy"):
            if marker in code:
                problems.append(
                    f"{path.name}: forbidden absolute path or proxy marker {marker!r}"
                )
        if ASSIGN_ROOT not in code:
            problems.append(f"{path.name}: dynamic root missing")
        if UPWARD_SEARCH not in code:
            problems.append(f"{path.name}: upward search missing")
        if f"seed = {seed}" not in code:
            problems.append(f"{path.name}: seed constant mismatch")
        if 'RGPPO_ROOT' not in code or 'RGPPO_NOTEBOOK' not in code or 'RGPPO_RESULTS_ROOT' not in code:
            problems.append(f"{path.name}: environment path adaptation missing")
        if "check_and_make_directories([" in code:
            problems.append(f"{path.name}: early check_and_make_directories remains")
        if nb.get("metadata", {}).get("rgppo") != {"condition": "0101", "seed": seed}:
            problems.append(f"{path.name}: rgppo metadata mismatch")
        if nb.get("metadata", {}).get("kernelspec", {}).get("name") != "finrl-aust-exploratory":
            problems.append(f"{path.name}: wrong kernel")
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
    manifest_path = ROOT / "provenance" / "notebook_manifest_0101.tsv"
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
