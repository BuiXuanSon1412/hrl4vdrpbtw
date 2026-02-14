import os
import sys
import time
import json
from pathlib import Path
from typing import Callable, Dict, Any

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from problem import Problem
from utils import init_population, cal_fitness
from config import ALGORITHMS


def run_algorithm_on_data(
    algorithm_runner: Callable,
    algorithm_params: Dict[str, Any],
    problem: Problem,
    indi_list: list,
    pop_size: int,
    max_gen: int,
    processing_number: int = 4,
) -> Dict[str, Any]:
    """
    Utility function to run an algorithm and measure execution time.

    Args:
        algorithm_runner: Function to run the algorithm
        algorithm_params: Dictionary of algorithm-specific parameters
        problem: Problem instance
        indi_list: Initial population list
        pop_size: Population size
        max_gen: Maximum generations
        processing_number: Number of processes for multiprocessing

    Returns:
        Dictionary containing 'time' and 'history' keys
    """
    start_time = time.time()

    # Merge common parameters with algorithm-specific ones
    params = {
        "processing_number": processing_number,
        "problem": problem,
        "indi_list": indi_list,
        "pop_size": pop_size,
        "max_gen": max_gen,
        "cal_fitness": cal_fitness,
        **algorithm_params,
    }

    history = algorithm_runner(**params)

    end_time = time.time()

    return {"time": end_time - start_time, "history": history}


def get_data_files(base_dir: str = "../data/generated/data") -> Dict[str, list]:
    """
    Get all JSON data files organized by problem size.

    Returns:
        Dictionary mapping size folder (e.g., 'N10') to list of file paths
    """
    data_files = {}
    base_path = Path(base_dir)

    if not base_path.exists():
        print(f"Warning: Data directory {base_dir} does not exist!")
        return data_files

    # Iterate through size directories (N10, N20, N50, etc.)
    for size_dir in sorted(base_path.iterdir()):
        if size_dir.is_dir() and size_dir.name.startswith("N"):
            json_files = list(size_dir.glob("*.json"))
            if json_files:
                data_files[size_dir.name] = sorted(json_files)

    return data_files


def save_result(result: Dict[str, Any], output_path: Path):
    """
    Save result to JSON file.

    Args:
        result: Result dictionary to save
        output_path: Path to output file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Saved result to {output_path}")


def main():
    # Configuration
    POP_SIZE = 100
    MAX_GEN = 100
    PROCESSING_NUMBER = 12
    SEED = 42
    BASE_DATA_DIR = "../data/generated/data"
    BASE_RESULT_DIR = "./result"

    # Get all data files
    data_files = get_data_files(BASE_DATA_DIR)

    if not data_files:
        print("No data files found!")
        return

    print(f"Found {len(data_files)} problem size categories")
    for size, files in data_files.items():
        print(f"  {size}: {len(files)} files")

    # Run each algorithm on each data file
    for algorithm_name, algorithm_config in ALGORITHMS.items():
        print(f"\n{'=' * 80}")
        print(f"Running {algorithm_name}")
        print(f"{'=' * 80}\n")

        algorithm_runner = algorithm_config["runner"]
        algorithm_params = algorithm_config["params"]

        for size_dir, files in data_files.items():
            print(f"\n{'-' * 80}")
            print(f"{algorithm_name} on {size_dir}")
            print(f"{'-' * 80}\n")

            for data_file in files:
                print(f"Processing: {data_file.name}")

                try:
                    # Load problem
                    problem = Problem(str(data_file))

                    # Initialize population
                    indi_list = init_population(POP_SIZE, SEED, problem)

                    # Run algorithm
                    result = run_algorithm_on_data(
                        algorithm_runner=algorithm_runner,
                        algorithm_params=algorithm_params,
                        problem=problem,
                        indi_list=indi_list,
                        pop_size=POP_SIZE,
                        max_gen=MAX_GEN,
                        processing_number=PROCESSING_NUMBER,
                    )

                    # Save result
                    output_path = (
                        Path(BASE_RESULT_DIR)
                        / algorithm_name
                        / size_dir
                        / data_file.name
                    )
                    save_result(result, output_path)

                    print(f"  Completed in {result['time']:.2f} seconds")

                except Exception as e:
                    print(f"  ERROR: {str(e)}")
                    import traceback

                    traceback.print_exc()
                    continue

    print(f"\n{'=' * 80}")
    print("All experiments completed!")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
