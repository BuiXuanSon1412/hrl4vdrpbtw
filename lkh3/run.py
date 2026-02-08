import os
import sys
import argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lkh3 import CTDTWB_Instance, LKH3_CTDTWB

parser = argparse.ArgumentParser()
parser.add_argument("--filename", type=str)
parser.add_argument("--subfolder", type=str)

args = parser.parse_args()
DATA_ROOT = "../data/generated/data"
RESULT_ROOT = "./result"

files = []
if args.filename and args.subfolder:
    print("Either one: filename OR subfolder")
    sys.exit(1)

if args.filename:
    files.append(args.filename)
else:
    if args.subfolder is None:
        print("Please provide --filename or --subfolder")
        sys.exit(1)

    subfolder_path = os.path.join(DATA_ROOT, args.subfolder)
    files = [
        f for f in os.listdir(subfolder_path)
        if os.path.isfile(os.path.join(subfolder_path, f))
    ]

def run(filename):
    parts = filename.split("_")
    n_folder = parts[1]  # N10

    data_path = os.path.join(DATA_ROOT, n_folder, filename)
    print(f"\nLoading: {data_path}")

    instance = CTDTWB_Instance(data_path)

    solver = LKH3_CTDTWB(instance, seed=42)
    solution = solver.run(max_iterations=500, time_limit=120)

    output_dir = os.path.join(RESULT_ROOT, n_folder)
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, filename)
    solver.export_solution(output_path)
    print(f"Saved: {output_path}")
    return solution


if __name__ == "__main__":
    for f in files:
        run(f)
