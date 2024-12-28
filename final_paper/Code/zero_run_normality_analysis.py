from decimal import Decimal, getcontext
import numpy as np
import torch
from typing import Dict, List, Tuple, Any
from scipy.stats import entropy, kstest
import matplotlib.pyplot as plt
from collections import Counter
from math import log2

class GPUNormalityAnalyzer:
    def __init__(self, precision: int = 1_000_000):
        """Initialize analyzer with specified precision and GPU support."""
        getcontext().prec = precision
        self.sqrt_2 = Decimal(2).sqrt()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.MAX_BLOCK_SIZE = 16  # Maximum block size for full frequency analysis
        print(f"Using device: {self.device}")
        
    def generate_binary_expansion(self, length: int) -> torch.Tensor:
        """Generate binary expansion of sqrt(2) using GPU acceleration."""
        result = []
        x = self.sqrt_2
        
        for _ in range(length):
            x = x * 2
            if x >= 2:
                result.append(1)
                x -= 2
            else:
                result.append(0)
                
        return torch.tensor(result, dtype=torch.int8, device=self.device)

    def analyze_block_frequencies(self, binary_tensor: torch.Tensor, block_size: int) -> Dict[str, Any]:
        """Analyze frequencies of binary blocks using adaptive methods based on block size."""
        if block_size > self.MAX_BLOCK_SIZE:
            return self._analyze_large_blocks_sampling(binary_tensor, block_size)
        
        # For smaller blocks, use direct computation
        stride = 1
        blocks = binary_tensor.unfold(0, block_size, stride)
        
        # Convert binary blocks to decimal for counting
        powers = torch.pow(2, torch.arange(block_size-1, -1, -1, device=self.device))
        block_values = (blocks * powers).sum(dim=1)
        
        # Count frequencies
        counts = torch.bincount(block_values, minlength=2**block_size)
        total = float(counts.sum())
        frequencies = counts.float() / total
        
        # Move to CPU for remaining calculations
        frequencies_cpu = frequencies.cpu()
        
        # Compute entropy and discrepancy
        mask = frequencies_cpu > 0
        entropy = -torch.sum(frequencies_cpu[mask] * torch.log2(frequencies_cpu[mask])).item()
        expected = 1.0 / (2 ** block_size)
        discrepancy = torch.max(torch.abs(frequencies_cpu - expected)).item()
        
        return {
            'frequencies': frequencies_cpu.numpy(),
            'expected': expected,
            'discrepancy': discrepancy,
            'entropy': entropy
        }

    def _analyze_large_blocks_sampling(self, binary_tensor: torch.Tensor, block_size: int) -> Dict[str, Any]:
        """Analyze large blocks using sampling-based approach."""
        # Use sampling for large blocks
        max_samples = 100_000
        length = len(binary_tensor)
        n_possible_blocks = length - block_size + 1
        
        if n_possible_blocks > max_samples:
            # Random sampling of starting positions
            start_indices = torch.randperm(n_possible_blocks, device=self.device)[:max_samples]
        else:
            start_indices = torch.arange(n_possible_blocks, device=self.device)
            
        # Extract sampled blocks
        blocks = torch.stack([binary_tensor[i:i+block_size] for i in start_indices])
        
        # Compute block statistics
        zero_counts = (blocks == 0).float().sum(dim=1)
        density = zero_counts / block_size
        
        # Move to CPU for histogram computation
        density_cpu = density.cpu().numpy()
        hist, bins = np.histogram(density_cpu, bins=50, density=True)
        hist = hist / hist.sum()  # Normalize
        
        # Compute approximate entropy using histogram
        mask = hist > 0
        entropy = -np.sum(hist[mask] * np.log2(hist[mask]))
        
        # Estimate discrepancy using empirical CDF
        theoretical = np.linspace(0, 1, len(hist))
        empirical = np.cumsum(hist)
        discrepancy = np.max(np.abs(empirical - theoretical))
        
        return {
            'frequencies': hist,
            'expected': 1.0 / len(hist),
            'discrepancy': discrepancy,
            'entropy': entropy
        }

    def zero_run_distribution(self, binary_tensor: torch.Tensor) -> Dict[int, float]:
        """Analyze distribution of zero run lengths using GPU acceleration."""
        # Find transitions from 0 to 1
        transitions = torch.where(binary_tensor[1:] != binary_tensor[:-1])[0] + 1
        transitions = torch.cat([torch.tensor([0], device=self.device), transitions])
        
        # Calculate run lengths
        run_lengths = transitions[1:] - transitions[:-1]
        run_lengths = run_lengths[binary_tensor[transitions[:-1]] == 0]
        
        # Count frequencies
        run_lengths_cpu = run_lengths.cpu().numpy()
        counts = Counter(run_lengths_cpu)
        total = len(run_lengths_cpu)
        return {length: count/total for length, count in counts.items()}

    def analyze_normality(self, length: int = 1_000_000) -> Dict:
        """Comprehensive normality analysis using GPU acceleration."""
        binary_tensor = self.generate_binary_expansion(length)
        
        # Scale-dependent block analysis
        max_scale = min(int(log2(length)), int(log2(self.MAX_BLOCK_SIZE * 8)))
        scale_analysis = {
            2**j: self.analyze_block_frequencies(binary_tensor, 2**j)
            for j in range(1, max_scale + 1)
        }
        
        # Zero run distribution analysis
        run_dist = self.zero_run_distribution(binary_tensor)
        run_length_entropy = -sum(p * log2(p) for p in run_dist.values() if p > 0)
        
        max_run_length = max(run_dist.keys()) if run_dist else 0
        theoretical_bounds = {l: 2 ** (-(l+1)) for l in range(1, max_run_length + 1)}
        
        empirical_values = list(run_dist.values())
        _, p_value = kstest(empirical_values, 'uniform')
        
        log_n_bound = log2(length) / length
        
        return {
            'scale_analysis': scale_analysis,
            'run_distribution': run_dist,
            'run_length_entropy': run_length_entropy,
            'theoretical_bounds': theoretical_bounds,
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
        plt.axhline(y=results['bounds']['max_observed_deviation'], color='b', 
                   label='Observed discrepancy')
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
            f.write(f"Maximum discrepancy: {results['bounds']['max_observed_deviation']:.2e}\n")
            f.write(f"Run length entropy: {results['run_length_entropy']:.2f}\n\n")
            
            f.write("\\subsection{Scale Analysis}\n")
            for scale, analysis in results['scale_analysis'].items():
                f.write(f"Scale {scale}: H(k)={analysis['entropy']:.2f}\n")
            
            f.write("\\subsection{Deviation Bounds}\n")
            f.write(f"O(log n/n) bound: {results['bounds']['log_n_bound']:.2e}\n")
            f.write(f"Max observed deviation: {results['bounds']['max_observed_deviation']:.2e}\n")

def main():
    # Perform normality analysis for different lengths
    analyzer = GPUNormalityAnalyzer()

    # File Path = final_paper/Code/Zero_Run_Normality_Analysis.py
    file_path = 'final_paper/Code/data/'
    
    lengths = [10_000, 100_000, 1_000_000]
    
    for length in lengths:
        print(f"\nAnalyzing sqrt(2) to {length} digits...")
        results = analyzer.analyze_normality(length)
        
        # Generate plots
        plt = analyzer.plot_analysis_results(results)
        plt.savefig(file_path + f'normality_analysis_{length}.png')
        plt.close()
        
        # Save detailed report
        analyzer.save_report(results, file_path + f'normality_analysis_{length}.tex')
        
        print(f"KS-test p-value: {results['statistical_tests']['ks_test_p_value']:.2e}")
        print(f"Maximum discrepancy: {results['bounds']['max_observed_deviation']:.2e}")
        print(f"O(log n/n) bound: {results['bounds']['log_n_bound']:.2e}")

if __name__ == "__main__":
    main()