from decimal import Decimal, getcontext
import numpy as np
from typing import Dict, List, Tuple, Any
from scipy.stats import entropy, kstest
import matplotlib.pyplot as plt
from collections import Counter
from math import log2

class NormalityAnalyzer:
    def __init__(self, precision: int = 1_000_000):
        """Initialize analyzer with specified precision (default 10^6)."""
        getcontext().prec = precision
        self.sqrt_2 = Decimal(2).sqrt()
        
    def generate_binary_expansion(self, length: int) -> str:
        """Generate binary expansion of sqrt(2) to given length."""
        result = []
        x = self.sqrt_2
        
        for _ in range(length):
            x = x * 2
            if x >= 2:
                result.append('1')
                x -= 2
            else:
                result.append('0')
                
        return ''.join(result)

    def compute_block_density(self, binary_str: str, n: int, k: int) -> float:
        """Compute local density function ρ(n,k) as defined in equation (2)."""
        block = binary_str[n:n+k]
        return block.count('0') / k if len(block) == k else 0

    def compute_clustering_coefficient(self, binary_str: str, n: int, k: int) -> float:
        """Compute zero clustering coefficient C(n,k) as defined in equation (8)."""
        block = binary_str[n:n+k]
        if len(block) < k:
            return 0
        return sum(int(block[i]) * int(block[i+1]) for i in range(k-1)) / (k-1)

    def analyze_block_frequencies(self, binary_str: str, block_size: int) -> Dict[str, Any]:
        """Analyze frequencies of binary blocks for normality testing."""
        blocks = [binary_str[i:i+block_size] 
                 for i in range(0, len(binary_str)-block_size+1)]
        counts = Counter(blocks)
        total = len(blocks)
        frequencies = {block: count/total for block, count in counts.items()}
        expected = 1 / (2 ** block_size)
        
        # Compute block entropy HB(k) as defined in equation (5)
        block_entropy = -sum(freq * log2(freq) for freq in frequencies.values() if freq > 0)
        
        return {
            'frequencies': frequencies,
            'expected': expected,
            'discrepancy': max(abs(freq - expected) for freq in frequencies.values()),
            'entropy': block_entropy
        }
    
    def compute_empirical_distribution(self, values: List[float], x: float) -> float:
        """Compute empirical distribution function FN(x)."""
        return sum(1 for val in values if val <= x) / len(values)

    def compute_discrepancy(self, values: List[float]) -> float:
        """Compute Kolmogorov-Smirnov discrepancy DN as defined in equation (7)."""
        sorted_values = sorted(values)
        n = len(values)
        max_diff = 0
        
        for i, x in enumerate(sorted_values, 1):
            theoretical = x  # For uniform distribution on [0,1]
            empirical = i / n
            max_diff = max(max_diff, abs(empirical - theoretical))
            
            if i < n:
                max_diff = max(max_diff, abs((i/n) - sorted_values[i]))
        
        return max_diff

    def zero_run_distribution(self, binary_str: str) -> Dict[int, float]:
        """Analyze distribution of zero run lengths according to equation (3)."""
        runs = []
        current_run = 0
        
        for bit in binary_str:
            if bit == '0':
                current_run += 1
            elif current_run > 0:
                runs.append(current_run)
                current_run = 0
        
        if current_run > 0:
            runs.append(current_run)
            
        counts = Counter(runs)
        total = len(runs)
        return {length: count/total for length, count in counts.items()}

    def compute_run_length_entropy(self, run_dist: Dict[int, float]) -> float:
        """Compute run length entropy HR as defined in equation (6)."""
        return -sum(p * log2(p) for p in run_dist.values() if p > 0)

    def compute_theoretical_bounds(self, max_length: int) -> Dict[int, float]:
        """Compute theoretical bounds for run lengths according to equation (4)."""
        return {l: 2 ** (-(l+1)) for l in range(1, max_length + 1)}

    def analyze_scale_dependency(self, binary_str: str, max_scale: int = 20) -> Dict[int, Dict]:
        """Analyze patterns across different scales from 2^1 to 2^max_scale."""
        return {
            2**j: self.analyze_block_frequencies(binary_str, 2**j)
            for j in range(1, min(max_scale + 1, int(log2(len(binary_str)))))
        }

    def analyze_normality(self, length: int = 1_000_000) -> Dict:
        """Comprehensive normality analysis implementing all required components."""
        binary_expansion = self.generate_binary_expansion(length)
        
        # Scale-dependent block analysis
        scale_analysis = self.analyze_scale_dependency(binary_expansion)
        
        # Zero run distribution analysis
        run_dist = self.zero_run_distribution(binary_expansion)
        run_length_entropy = self.compute_run_length_entropy(run_dist)
        
        # Theoretical bounds
        max_run_length = max(run_dist.keys()) if run_dist else 0
        theoretical_bounds = self.compute_theoretical_bounds(max_run_length)
        
        # Discrepancy analysis
        empirical_values = list(run_dist.values())
        discrepancy = self.compute_discrepancy(empirical_values)
        
        # Statistical significance testing
        _, p_value = kstest(empirical_values, 'uniform')
        
        # Compute O(log n/n) deviation bound
        log_n_bound = log2(length) / length
        
        return {
            'scale_analysis': scale_analysis,
            'run_distribution': run_dist,
            'run_length_entropy': run_length_entropy,
            'theoretical_bounds': theoretical_bounds,
            'discrepancy': discrepancy,
            'statistical_tests': {
                'ks_test_p_value': p_value,
                'significance_level': 0.01,
                'reject_null': p_value < 0.01
            },
            'bounds': {
                'log_n_bound': log_n_bound,
                'max_observed_deviation': max(abs(v - theoretical_bounds[k])
                                           for k, v in run_dist.items()
                                           if k in theoretical_bounds)
            }
        }

    def plot_analysis_results(self, results: Dict):
        """Generate comprehensive visualization of normality analysis results."""
        plt.figure(figsize=(15, 12))
        
        # Plot 1: Zero Run Distribution vs Theoretical
        plt.subplot(2, 2, 1)
        run_dist = results['run_distribution']
        theoretical = results['theoretical_bounds']
        plt.semilogy(run_dist.keys(), run_dist.values(), 'bo-', label='Observed')
        plt.semilogy(theoretical.keys(), theoretical.values(), 'r--', label='Theoretical')
        plt.title('Zero Run Distribution')
        plt.xlabel('Run Length')
        plt.ylabel('Probability')
        plt.legend()
        
        # Plot 2: Scale-dependent Entropy
        plt.subplot(2, 2, 2)
        scales = sorted(results['scale_analysis'].keys())
        entropies = [results['scale_analysis'][s]['entropy'] for s in scales]
        plt.semilogx(scales, entropies, 'go-')
        plt.title('Scale-Dependent Entropy')
        plt.xlabel('Block Size')
        plt.ylabel('Entropy (bits)')
        
        # Plot 3: Discrepancy Analysis
        plt.subplot(2, 2, 3)
        plt.axhline(y=results['bounds']['log_n_bound'], color='r', linestyle='--',
                   label='O(log n/n) bound')
        plt.axhline(y=results['discrepancy'], color='b', label='Observed discrepancy')
        plt.title('Discrepancy Analysis')
        plt.legend()
        
        # Plot 4: QQ Plot
        plt.subplot(2, 2, 4)
        observed = sorted(results['run_distribution'].values())
        theoretical = sorted(results['theoretical_bounds'].values())[:len(observed)]
        plt.scatter(theoretical, observed)
        plt.plot([0, max(theoretical)], [0, max(theoretical)], 'r--')
        plt.title('Q-Q Plot')
        plt.xlabel('Theoretical Quantiles')
        plt.ylabel('Observed Quantiles')
        
        plt.tight_layout()
        return plt

    def save_report(self, results: Dict, filename: str):
        """Generate detailed LaTeX report of analysis results."""
        with open(filename, "w") as f:
            f.write("\\section{Normality Analysis Results}\n\n")
            
            f.write("\\subsection{Statistical Summary}\n")
            f.write(f"KS-test p-value: {results['statistical_tests']['ks_test_p_value']:.2e}\n")
            f.write(f"Maximum discrepancy: {results['discrepancy']:.2e}\n")
            f.write(f"Run length entropy: {results['run_length_entropy']:.2f}\n\n")
            
            f.write("\\subsection{Scale Analysis}\n")
            for scale, analysis in results['scale_analysis'].items():
                f.write(f"Scale {scale}: H(k)={analysis['entropy']:.2f}\n")
            
            f.write("\\subsection{Deviation Bounds}\n")
            f.write(f"O(log n/n) bound: {results['bounds']['log_n_bound']:.2e}\n")
            f.write(f"Max observed deviation: {results['bounds']['max_observed_deviation']:.2e}\n")

def main():
    # Initialize analyzer with full precision
    analyzer = NormalityAnalyzer()
    
    # Analyze at different scales
    lengths = [10_000, 100_000, 1_000_000]
    
    for length in lengths:
        print(f"\nAnalyzing sqrt(2) to {length} digits...")
        results = analyzer.analyze_normality(length)
        
        # Generate plots
        plt = analyzer.plot_analysis_results(results)
        plt.savefig(f'normality_analysis_{length}.png')
        plt.close()
        
        # Save detailed report
        analyzer.save_report(results, f'normality_analysis_{length}.tex')
        
        # Print summary statistics
        print(f"KS-test p-value: {results['statistical_tests']['ks_test_p_value']:.2e}")
        print(f"Maximum discrepancy: {results['discrepancy']:.2e}")
        print(f"O(log n/n) bound: {results['bounds']['log_n_bound']:.2e}")

if __name__ == "__main__":
    main()