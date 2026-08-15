"""
Cross-platform replacement for eval_schedule.bash + apls.bash + topo.bash.
Run from the sam_road root with:
    pixi run python spacenet_metrics/run_eval.py --dirs <output_dir_name> [...]
Output dir names are relative to save/ (e.g. infer__20260810_123456).
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
ROOT_DIR = SCRIPT_DIR.parent
DATA_DIR = ROOT_DIR / "spacenet"
SAVE_DIR = ROOT_DIR / "save"
PYTHON = sys.executable


def run(cmd, cwd):
    str_cmd = [str(c) for c in cmd]
    print(f"  >> {' '.join(str_cmd)}")
    result = subprocess.run(str_cmd, cwd=str(cwd))
    if result.returncode != 0:
        print(f"  WARNING: command exited with code {result.returncode}")


def run_apls(output_dir: Path):
    with open(DATA_DIR / "data_split.json") as f:
        test_ids = json.load(f)["test"]

    results_dir = output_dir / "results" / "apls"
    results_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    for tile_id in test_ids:
        pred_graph = output_dir / "graph" / f"{tile_id}.p"
        if not pred_graph.exists():
            continue

        gt_graph = DATA_DIR / "RGB_1.0_meter" / f"{tile_id}__gt_graph.p"
        result_txt = results_dir / f"{tile_id}.txt"
        apls_dir = SCRIPT_DIR / "apls"
        gt_json = apls_dir / "gt.json"
        prop_json = apls_dir / "prop.json"
        print(f"  === {tile_id} ===")

        # convert.py writes json files into apls/ dir; go run executes from apls/ (where go.mod lives)
        run([PYTHON, apls_dir / "convert.py", str(gt_graph), str(gt_json)], cwd=SCRIPT_DIR)
        run([PYTHON, apls_dir / "convert.py", str(pred_graph), str(prop_json)], cwd=SCRIPT_DIR)
        run(["go", "run", "main.go", str(gt_json), str(prop_json), str(result_txt), "spacenet"],
            cwd=apls_dir)
        processed += 1

    print(f"  Processed {processed}/{len(test_ids)} test images.")

    # apls.py uses '../{dir}/results/apls' so must run from SCRIPT_DIR
    # dir arg is relative to ROOT_DIR using forward slashes
    dir_arg = output_dir.relative_to(ROOT_DIR).as_posix()
    run([PYTHON, SCRIPT_DIR / "apls.py", "--dir", dir_arg], cwd=SCRIPT_DIR)


def run_topo(output_dir: Path):
    # topo scripts also use '../{dir}/...' so run from SCRIPT_DIR
    dir_arg = output_dir.relative_to(ROOT_DIR).as_posix()
    run([PYTHON, SCRIPT_DIR / "topo" / "main.py", "-savedir", dir_arg], cwd=SCRIPT_DIR)
    run([PYTHON, SCRIPT_DIR / "topo.py", "-savedir", dir_arg], cwd=SCRIPT_DIR)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirs", nargs="+", required=True,
                        help="Output dir names under save/ (e.g. infer_2026-08-10_123456)")
    parser.add_argument("--apls-only", action="store_true")
    parser.add_argument("--topo-only", action="store_true")
    args = parser.parse_args()

    for dir_name in args.dirs:
        output_dir = SAVE_DIR / dir_name
        if not output_dir.exists():
            print(f"ERROR: {output_dir} does not exist, skipping.")
            continue

        print(f"\n{'='*60}")
        print(f"Evaluating: {dir_name}")
        print(f"{'='*60}")

        if not args.topo_only:
            print("\n--- APLS ---")
            run_apls(output_dir)

        if not args.apls_only:
            print("\n--- TOPO ---")
            run_topo(output_dir)


if __name__ == "__main__":
    main()
