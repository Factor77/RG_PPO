#!/usr/bin/env python3
"""Execute a .ipynb from cell 0 in order with a live jupyter kernel and write
the outputs back to the file. Fails fast on the first erroring cell.

Usage: python exec_notebook.py <notebook.ipynb>
"""

import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from jupyter_client import KernelManager

DEFAULT_KERNEL_NAME = "python3"
CELL_TIMEOUT_S = 3600

PHASE7_FROZEN_ENV = {
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    "CUDA_VISIBLE_DEVICES": "0",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "NVIDIA_TF32_OVERRIDE": "0",
    "OMP_NUM_THREADS": "12",
    "MKL_NUM_THREADS": "12",
}
PHASE7_EXPERIMENT_IDS = frozenset(str(i) for i in range(1300, 1309))
PHASE9_EXPERIMENT_IDS = frozenset(str(i) for i in range(1320, 1326)) | frozenset(("1321b_ctrl", "1321b_lo", "1321b_hi"))


def _kernel_launch_contract(nb: dict) -> tuple[str, dict[str, str]]:
    """Resolve the notebook kernel and interpreter-start environment.

    Ordinary notebooks inherit the caller environment.  Phase 7 notebooks
    additionally freeze ``PYTHONHASHSEED`` and the CUDA/thread variables before
    the kernel interpreter starts; setting them in cell 0 would be too late for
    strict seed reproducibility.
    """
    kernel_name = (
        nb.get("metadata", {}).get("kernelspec", {}).get("name")
        or DEFAULT_KERNEL_NAME
    )
    kernel_env = dict(os.environ)
    phase7 = nb.get("metadata", {}).get("phase7")
    if phase7 is not None:
        seed = phase7.get("seed")
        if not isinstance(seed, int) or seed not in tuple(range(10)):
            raise ValueError("Phase 7 notebook metadata must contain seed 0, 1, or 2")
        kernel_env.update(PHASE7_FROZEN_ENV)
        kernel_env["PYTHONHASHSEED"] = str(seed)
    return str(kernel_name), kernel_env


def _write_json_atomic(path: Path, payload: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def _claim_phase7_run(nb: dict, nb_path: Path) -> tuple[Path | None, dict | None]:
    phase7 = nb.get("metadata", {}).get("phase7")
    if phase7 is None:
        return None, None
    experiment = str(phase7.get("experiment_id", ""))
    seed = phase7.get("seed")
    if experiment not in (PHASE7_EXPERIMENT_IDS | PHASE9_EXPERIMENT_IDS) or seed not in tuple(range(10)):
        raise ValueError("invalid Phase 7 experiment identity in notebook metadata")
    project_root = Path(__file__).resolve().parents[2]
    state_file = project_root / "Results" / "Tail_GAE" / experiment / "pilot_status" / f"seed_{seed}.json"
    state_file.parent.mkdir(parents=True, exist_ok=True)
    attempts = 0
    if state_file.exists():
        previous = json.loads(state_file.read_text(encoding="utf-8"))
        previous_status = previous.get("status")
        if previous_status != "FAILED":
            raise RuntimeError(
                f"Phase 7 run already claimed with status {previous_status!r}: {state_file}"
            )
        attempts = int(previous.get("attempts", 1))
    payload = {
        "experiment_id": experiment,
        "arm": phase7.get("arm"),
        "seed": seed,
        "notebook": str(nb_path),
        "status": "RUNNING",
        "attempts": attempts + 1,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
    }
    _write_json_atomic(state_file, payload)
    return state_file, payload


def _finish_phase7_run(state_file: Path | None, payload: dict | None, status: str, error: str | None = None) -> None:
    if state_file is None or payload is None:
        return
    finished = dict(payload)
    finished["status"] = status
    finished["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    if error is not None:
        finished["error"] = error
    _write_json_atomic(state_file, finished)


def main() -> int:
    nb_path = Path(sys.argv[1]).resolve()
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    kernel_name, kernel_env = _kernel_launch_contract(nb)
    state_file, state_payload = _claim_phase7_run(nb, nb_path)

    n_cells = len(nb["cells"])
    failed: tuple[int, str] | None = None
    km = None
    kc = None
    try:
        km = KernelManager(kernel_name=kernel_name)
        km.start_kernel(cwd=str(nb_path.parent), env=kernel_env)
        kc = km.client()
        kc.start_channels()
        kc.wait_for_ready(timeout=120)
        for i, cell in enumerate(nb["cells"]):
            if cell["cell_type"] != "code":
                continue
            code = "".join(cell["source"])
            if not code.strip():
                cell["outputs"] = []
                cell["execution_count"] = None
                continue
            print(f"[cell {i}/{n_cells - 1}] executing ...", flush=True)
            outputs, error, exec_count = _run_cell(kc, code)
            cell["outputs"] = outputs
            cell["execution_count"] = exec_count
            for out in outputs:
                if out["output_type"] == "stream":
                    print(out["text"], end="", flush=True)
            if error is not None:
                failed = (i, error)
                print(f"[cell {i}] ERROR: {error}", flush=True)
                break
    except BaseException:
        _finish_phase7_run(state_file, state_payload, "FAILED", traceback.format_exc())
        raise
    finally:
        if kc is not None:
            kc.stop_channels()
        if km is not None:
            km.shutdown_kernel(now=True)
        nb_path.write_text(
            json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    if failed is not None:
        cell_idx, error = failed
        _finish_phase7_run(state_file, state_payload, "FAILED", error)
        print(f"\nSMOKE FAILED at cell {cell_idx}: {error}", flush=True)
        return 1
    _finish_phase7_run(state_file, state_payload, "COMPLETED")
    print(f"\nSMOKE PASSED: all {n_cells} cells executed.", flush=True)
    return 0


def _run_cell(kc, code: str):
    """Execute one code cell; return (outputs, error_msg_or_None, execution_count)."""
    msg_id = kc.execute(code, allow_stdin=False)
    outputs: list[dict] = []
    error: str | None = None
    exec_count = None
    got_idle = False
    deadline = time.monotonic() + CELL_TIMEOUT_S
    while time.monotonic() < deadline:
        try:
            msg = kc.get_iopub_msg(timeout=1)
        except Exception:
            continue
        if msg["parent_header"].get("msg_id") != msg_id:
            continue
        mtype = msg["msg_type"]
        content = msg["content"]
        if mtype == "status":
            if content["execution_state"] == "idle":
                got_idle = True
                break
        elif mtype == "stream":
            outputs.append({
                "output_type": "stream",
                "name": content["name"],
                "text": content["text"],
            })
        elif mtype == "execute_result":
            exec_count = content["execution_count"]
            outputs.append({
                "output_type": "execute_result",
                "execution_count": exec_count,
                "data": content["data"],
                "metadata": {},
            })
        elif mtype == "display_data":
            outputs.append({
                "output_type": "display_data",
                "data": content["data"],
                "metadata": content.get("metadata", {}),
            })
        elif mtype == "error":
            outputs.append({
                "output_type": "error",
                "ename": content["ename"],
                "evalue": content["evalue"],
                "traceback": content["traceback"],
            })
            error = f"{content['ename']}: {content['evalue']}"
    if not got_idle and error is None:
        error = f"timeout after {CELL_TIMEOUT_S}s (no idle status received)"
    if got_idle and error is None:
        try:
            reply = kc.get_shell_msg(timeout=5)
            if reply["content"]["status"] == "error":
                e = reply["content"]
                error = f"{e['ename']}: {e['evalue']}"
        except Exception:
            pass
    return outputs, error, exec_count


if __name__ == "__main__":
    sys.exit(main())
