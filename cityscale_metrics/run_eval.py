"""
Cross-platform replacement for eval_schedule.bash + apls.bash + topo.bash.
Run from the sam_road root with:
    pixi run python cityscale_metrics/run_eval.py --dirs <output_dir_name> [...]
Output dir names are relative to save/ (e.g. cityscale_toponet_no_itsc).
"""

import argparse
import sys
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
ROOT_DIR = SCRIPT_DIR.parent
DATA_DIR = ROOT_DIR / "cityscale"
SAVE_DIR = ROOT_DIR / "save"
PYTHON = sys.executable

TEST_IDS = [8, 9, 19, 28, 29, 39, 48, 49, 59, 68, 69, 79,
            88, 89, 99, 108, 109, 119, 128, 129, 139, 148, 149, 159,
            168, 169, 179]


def run(cmd, cwd):
    str_cmd = [str(c) for c in cmd]
    print(f"  >> {' '.join(str_cmd)}")
    result = subprocess.run(str_cmd, cwd=str(cwd))
    if result.returncode != 0:
        print(f"  WARNING: command exited with code {result.returncode}")


def run_apls(output_dir: Path):
    results_dir = output_dir / "results" / "apls"
    results_dir.mkdir(parents=True, exist_ok=True)

    apls_dir = SCRIPT_DIR / "apls"
    gt_json = apls_dir / "gt.json"
    prop_json = apls_dir / "prop.json"

    processed = 0
    for tile_id in TEST_IDS:
        pred_graph = output_dir / "graph" / f"{tile_id}.p"
        if not pred_graph.exists():
            continue

        gt_graph = DATA_DIR / "20cities" / f"region_{tile_id}_graph_gt.pickle"
        result_txt = results_dir / f"{tile_id}.txt"
        print(f"  === {tile_id} ===")

        run([PYTHON, apls_dir / "convert.py", str(gt_graph), str(gt_json)], cwd=SCRIPT_DIR)
        run([PYTHON, apls_dir / "convert.py", str(pred_graph), str(prop_json)], cwd=SCRIPT_DIR)
        # No "spacenet" arg — uses default 2048x2048 parameters
        run(["go", "run", "main.go", str(gt_json), str(prop_json), str(result_txt)],
            cwd=apls_dir)
        processed += 1

    print(f"  Processed {processed}/{len(TEST_IDS)} test images.")

    dir_arg = output_dir.relative_to(ROOT_DIR).as_posix()
    run([PYTHON, SCRIPT_DIR / "apls.py", "--dir", dir_arg], cwd=SCRIPT_DIR)


def run_topo(output_dir: Path):
    dir_arg = output_dir.relative_to(ROOT_DIR).as_posix()
    run([PYTHON, SCRIPT_DIR / "topo" / "main.py", "-savedir", dir_arg], cwd=SCRIPT_DIR)
    run([PYTHON, SCRIPT_DIR / "topo.py", "-savedir", dir_arg], cwd=SCRIPT_DIR)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirs", nargs="+", required=True,
                        help="Output dir names under save/ (e.g. cityscale_toponet_no_itsc)")
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
