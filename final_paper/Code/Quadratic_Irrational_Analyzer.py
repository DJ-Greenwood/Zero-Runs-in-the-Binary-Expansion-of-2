import torch
import math
from typing import Dict, List
import pandas as pd

class QuadraticIrrationalAnalyzer:
    def __init__(self, D: int, max_n: int = 1000, precision: int = 1_000_000):
        """
        Initialize the analyzer for a given quadratic irrational sqrt(D).
        
        Args:
        - D (int): Discriminant of the quadratic irrational (D > 0, nonsquare).
        - max_n (int): Maximum position `n` to analyze.
        - precision (int): Number of binary digits to compute.
        """
        self.D = D
        self.max_n = max_n
        self.precision = precision
        self.sqrt_D = math.sqrt(D)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.binary_expansion = self._compute_binary_expansion()

    def _compute_binary_expansion(self) -> torch.Tensor:
        """
        Compute the binary expansion of sqrt(D) up to the given precision using GPU acceleration.
        """
        print(f"Computing binary expansion of sqrt({self.D}) to {self.precision} digits...")
        result = []
        x = self.sqrt_D
        for _ in range(self.precision):
            x *= 2
            if x >= 1:
                result.append(1)
                x -= 1
            else:
                result.append(0)
        return torch.tensor(result, dtype=torch.int8, device=self.device)

    def analyze_zero_runs(self) -> List[Dict]:
        """
        Analyze the length of zero runs at different positions in the binary expansion.
        """
        results = []
        binary_exp = self.binary_expansion.cpu().numpy()
        
        print(f"Analyzing zero runs up to position {self.max_n}...")
        for n in range(1, self.max_n + 1):
            k = 0
            while n + k < self.precision and binary_exp[n + k] == 0:
                k += 1

            log_bound = math.log2(n)
            constant = 0.5 * math.log2(self.D) if self.D > 1 else 0  # Constant C
            theoretical_bound = log_bound + constant

            results.append({
                "position_n": n,
                "zero_run_length_k": k,
                "log2_n": log_bound,
                "theoretical_bound": theoretical_bound,
                "C": constant,
                "valid": k <= theoretical_bound
            })

        return results

    def generate_report(self, results: List[Dict], output_path: str):
        """
        Generate a detailed CSV report of the analysis.
        """
        print(f"Generating report...")
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        print(f"Report saved to {output_path}.")

    def validate_and_plot(self, results: List[Dict], output_plot_path: str):
        """
        Validate the results and create a plot to visualize zero runs and bounds.
        """
        import matplotlib.pyplot as plt

        df = pd.DataFrame(results)
        plt.figure(figsize=(10, 6))

        # Plot zero run lengths
        plt.plot(df["position_n"], df["zero_run_length_k"], label="Observed Zero Run Length (k)", marker="o")
        
        # Plot theoretical bounds
        plt.plot(df["position_n"], df["theoretical_bound"], label="Theoretical Bound (log2(n) + C)", linestyle="--")

        plt.xlabel("Position (n)")
        plt.ylabel("Zero Run Length (k)")
        plt.title(f"Zero Run Analysis for sqrt({self.D})")
        plt.legend()
        plt.grid()
        plt.savefig(output_plot_path)
        plt.close()
        print(f"Plot saved to {output_plot_path}.")

if __name__ == "__main__":
    # Configuration
    file_path = "final_paper/Code/data/"
    D = 5  # Discriminant for sqrt(D)
    max_n = 1_000_000 # Maximum position to analyze
    precision = 1_000_000  # Binary digits to compute
    report_path = file_path + f"sqrt_{D}_zero_run_analysis.csv"
    plot_path = file_path + f"sqrt_{D}_zero_run_plot.png"

    # Initialize and run the analyzer
    analyzer = QuadraticIrrationalAnalyzer(D=D, max_n=max_n, precision=precision)
    results = analyzer.analyze_zero_runs()

    # Generate report and plot
    analyzer.generate_report(results, report_path)
    analyzer.validate_and_plot(results, plot_path)
