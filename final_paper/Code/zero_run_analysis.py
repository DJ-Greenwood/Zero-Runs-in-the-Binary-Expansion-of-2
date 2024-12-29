import math
import cupy as cp  # Use CuPy for GPU acceleration
from typing import List, Dict, Any
from rich.console import Console
from rich.table import Table

class Sqrt2ZeroRunAnalyzerGPU:
    """Analyzes zero runs in the binary expansion of sqrt(2) using GPU acceleration."""

    def __init__(self, precision: int = 10000):
        """
        Initializes the analyzer with a specified precision for computations.

        Args:
            precision (int): Number of decimal places for high-precision calculations.
        """
        self.precision = precision
        self.sqrt_2 = cp.sqrt(cp.array(2))  # Use GPU to calculate sqrt(2)

    def generate_binary_expansion(self) -> cp.ndarray:
        """
        Generate the binary expansion of sqrt(2) up to the specified precision.

        Returns:
            cp.ndarray: Binary expansion of sqrt(2) as a GPU array.
        """
        sqrt2 = self.sqrt_2 % 1  # Keep only the fractional part of sqrt(2)
        binary_expansion = cp.zeros(self.precision, dtype=cp.int8)

        for i in range(self.precision):
            sqrt2 *= 2
            binary_expansion[i] = 1 if sqrt2 >= 1 else 0
            sqrt2 -= cp.floor(sqrt2)  # Keep only the fractional part

        # Debugging: Print the first 50 digits (move to CPU for viewing)
        print(f"Binary Expansion (First 50 Digits): {cp.asnumpy(binary_expansion[:50])}")
        return binary_expansion

    def detect_zero_runs(self, binary_expansion: cp.ndarray) -> List[Dict[str, int]]:
        """
        Detect all zero runs in the binary expansion.

        Args:
            binary_expansion (cp.ndarray): Binary expansion of sqrt(2).

        Returns:
            List[Dict[str, int]]: List of zero runs with their starting position and length.
        """
        zero_mask = (binary_expansion == 0).astype(cp.int32)
        diff = cp.diff(zero_mask)
        
        start_positions = cp.where(diff == 1)[0] + 1
        end_positions = cp.where(diff == -1)[0] + 1

        if zero_mask[0] == 1:
            start_positions = cp.concatenate(([0], start_positions))
        if zero_mask[-1] == 1:
            end_positions = cp.concatenate((end_positions, [len(binary_expansion)]))

        run_lengths = end_positions - start_positions

        zero_runs = [{"start": int(start + 1), "length": int(length)} 
                     for start, length in zip(cp.asnumpy(start_positions), cp.asnumpy(run_lengths))]

        print(f"Detected Zero Runs (First 10): {zero_runs[:10]}")
        return zero_runs

    def validate_zero_runs(self, zero_runs: List[Dict[str, int]]) -> List[Dict[str, Any]]:
        """
        Validate detected zero runs against the theoretical bound.

        Args:
            zero_runs (List[Dict[str, int]]): Detected zero runs.

        Returns:
            List[Dict[str, Any]]: Validation results for each zero run.
        """
        results = []
        for run in zero_runs:
            n = run["start"]
            k = run["length"]
            theoretical_bound = math.log2(n) + 0.5
            results.append({
                "position": n,
                "run_length": k,
                "theoretical_bound": theoretical_bound,
                "valid": k <= theoretical_bound
            })
        return results

    def generate_report(self, validation_results: List[Dict[str, Any]]):
        """
        Generate a report for zero run validation.

        Args:
            validation_results (List[Dict[str, Any]]): Validation results.
        """
        console = Console()

        table = Table(title="Zero Run Validation Report")
        table.add_column("Start Position", justify="right")
        table.add_column("Run Length", justify="right")
        table.add_column("Theoretical Bound", justify="right")
        table.add_column("Valid", justify="center")

        if not validation_results:
            print("No zero runs detected.")
            return

        for result in validation_results:
            table.add_row(
                str(result["position"]),
                str(result["run_length"]),
                f"{result['theoretical_bound']:.2f}",
                "✔" if result["valid"] else "✘"
            )

        console.print(table)

    def save_report(self, file_name: str, validation_results: List[Dict[str, Any]]):
        """
        Save the zero run validation report to a file.

        Args:
            file_name (str): Name of the file to save the report.
            validation_results (List[Dict[str, Any]]): Validation results.
        """
        with open(file_name, "w") as file:
            file.write("Position,Run Length,Theoretical Bound,Valid\n")
            for result in validation_results:
                file.write(f"{result['position']},{result['run_length']},"
                           f"{result['theoretical_bound']:.2f},{result['valid']}\n")
        print(f"Validation report saved to {file_name}.")

    def summary_report(self, file_name: str, validation_results: List[Dict[str, Any]]):
        """
        Generate a summary report for zero run validation and save it to a file.

        Args:
            file_name (str): Path to the output file.
            validation_results (List[Dict[str, Any]]): Validation results.
        """
        total_runs = len(validation_results)
        valid_runs = sum(1 for result in validation_results if result["valid"])
        invalid_runs = total_runs - valid_runs
        valid_percentage = (100 * valid_runs / total_runs) if total_runs > 0 else 0

        with open(file_name, "w") as file:
            file.write(f"Total Zero Runs: {total_runs}\n")
            file.write(f"Valid Zero Runs: {valid_runs}\n")
            file.write(f"Invalid Zero Runs: {invalid_runs}\n")
            file.write(f"Valid Percentage: {valid_percentage:.2f}%\n")
        print(f"Summary report saved to {file_name}.")

if __name__ == "__main__":
    file_path = "final_paper/Code/data/"
    precision = 10_000_000  # Adjust precision as needed
    analyzer = Sqrt2ZeroRunAnalyzerGPU(precision=precision)

    # Step 1: Generate binary expansion
    print("Generating binary expansion of sqrt(2)...")
    binary_expansion = analyzer.generate_binary_expansion()

    # Step 2: Detect zero runs
    print("Detecting zero runs...")
    zero_runs = analyzer.detect_zero_runs(binary_expansion)

    # Step 3: Validate zero runs
    print("Validating zero runs against theoretical bounds...")
    validation_results = analyzer.validate_zero_runs(zero_runs)

    # Step 4: Generate and display report
    print("Generating report...")
    analyzer.generate_report(validation_results)

    # Step 5: Save report to file
    print("Saving validation report to file...")
    analyzer.save_report(file_path + "zero_run_validation_report.csv", validation_results)

    # Step 6: Save summary report
    print("Saving summary report to file...")
    analyzer.summary_report(file_path + "zero_run_summary_report.txt", validation_results)
