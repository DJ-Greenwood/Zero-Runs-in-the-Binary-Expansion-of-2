import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal, getcontext
from typing import Dict, List, Tuple

class QualityMetricsAnalyzer:
    def __init__(self, precision: int = 1000):
        """Initialize the analyzer with high precision."""
        getcontext().prec = precision
        self.sqrt2 = Decimal(2).sqrt()
        
    def calculate_error(self, n: int, k: int, p: int, q: int) -> Decimal:
        """Calculate error for given parameters."""
        approx = Decimal(p) / Decimal(2 ** n) + Decimal(q) / Decimal(2 ** (n + k))
        return abs(self.sqrt2 - approx)
    
    def calculate_quality(self, error: Decimal) -> Decimal:
        """Calculate quality metric Q = -log10(error)."""
        if error == 0:
            return Decimal('Infinity')
        return -error.log10()
    
    def analyze_run_range(self, n: int, k_range: range) -> Dict[int, Decimal]:
        """Analyze quality metrics for a range of run lengths."""
        qualities = {}
        for k in k_range:
            # Calculate p and q for this position and run length
            p = int(self.sqrt2 * Decimal(2 ** n))
            q = int((self.sqrt2 - Decimal(p) / Decimal(2 ** n)) * Decimal(2 ** (n + k)))
            
            error = self.calculate_error(n, k, p, q)
            quality = self.calculate_quality(error)
            qualities[k] = quality
            
        return qualities
    
    def categorize_runs(self, qualities: Dict[int, Decimal]) -> Dict[str, List[Decimal]]:
        """Categorize run qualities into short, medium, and long runs."""
        categories = {
            'Short (2-10)': [],
            'Medium (11-50)': [],
            'Long (51-200)': []
        }
        
        for k, quality in qualities.items():
            if 2 <= k <= 10:
                categories['Short (2-10)'].append(quality)
            elif 11 <= k <= 50:
                categories['Medium (11-50)'].append(quality)
            elif 51 <= k <= 200:
                categories['Long (51-200)'].append(quality)
                
        return categories
    
    def plot_quality_metrics(self, qualities: Dict[int, Decimal], save_path: str = None):
        """Plot quality metrics across run lengths and optionally save to file."""
        k_values = list(qualities.keys())
        q_values = [float(q) for q in qualities.values()]
        
        plt.figure(figsize=(12, 6))
        plt.plot(k_values, q_values, 'b-', label='Quality Metric')
        plt.axhspan(1.4, 4.8, color='g', alpha=0.2, label='Short Runs Range')
        plt.axhspan(5.0, 15.0, color='y', alpha=0.2, label='Medium Runs Range')
        plt.axhspan(15.0, 60.0, color='r', alpha=0.2, label='Long Runs Range')
        
        plt.xlabel('Run Length (k)')
        plt.ylabel('Quality Metric Q(n,k)')
        plt.title('Quality Metrics vs Run Length')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()
        
    def print_summary_statistics(self, categories: Dict[str, List[Decimal]]):
        """Print summary statistics for each category."""
        print("\nQuality Metrics Summary:")
        print("-" * 50)
        for category, values in categories.items():
            if values:
                min_val = min(values)
                max_val = max(values)
                avg_val = sum(values) / len(values)
                print(f"\n{category}:")
                print(f"  Range: [{min_val:.1f}, {max_val:.1f}]")
                print(f"  Average: {avg_val:.1f}")

    def save_report(self, categories: Dict[str, List[Decimal]], filename: str):
        """Save the summary statistics to a file."""
        with open(filename, 'w') as f:
            f.write("Quality Metrics Summary:\n")
            f.write("-" * 50 + "\n")
            for category, values in categories.items():
                if values:
                    min_val = min(values)
                    max_val = max(values)
                    avg_val = sum(values) / len(values)
                    f.write(f"\n{category}:\n")
                    f.write(f"  Range: [{min_val:.1f}, {max_val:.1f}]\n")
                    f.write(f"  Average: {avg_val:.1f}\n")

def main():
    # Initialize analyzer
    analyzer = QualityMetricsAnalyzer()
    
    # Analyze for position n=100 and run lengths k=2 to 200
    qualities = analyzer.analyze_run_range(100, range(2, 201))
    
    # Categorize results
    categories = analyzer.categorize_runs(qualities)
    
    # Print summary statistics
    analyzer.print_summary_statistics(categories)
    
    # Plot and save quality metrics
    analyzer.plot_quality_metrics(qualities, save_path="quality_metrics_plot.png")
    
    # Save report
    analyzer.save_report(categories, "quality_metrics_report.txt")

if __name__ == "__main__":
    main()