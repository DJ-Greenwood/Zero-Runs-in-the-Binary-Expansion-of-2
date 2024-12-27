from decimal import Decimal, getcontext
import numpy as np
from typing import Dict, List, Tuple
from scipy.stats import entropy
import matplotlib.pyplot as plt
from collections import Counter

class NormalityAnalyzer:
    def __init__(self, precision: int = 10000):
        getcontext().prec = precision
        self.sqrt_2 = Decimal(2).sqrt()
        
    def generate_binary_expansion(self, length: int) -> str:
        """Generate binary expansion of sqrt(2) to given length."""
        # Convert Decimal to binary using division method
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

    def analyze_block_frequencies(self, binary_str: str, block_size: int) -> Dict[str, float]:
        """Analyze frequencies of binary blocks for normality testing."""
        blocks = [binary_str[i:i+block_size] 
                 for i in range(0, len(binary_str)-block_size+1)]
        counts = Counter(blocks)
        total = len(blocks)
        frequencies = {block: count/total for block, count in counts.items()}
        expected = 1 / (2 ** block_size)
        
        return {
            'frequencies': frequencies,
            'expected': expected,
            'discrepancy': max(abs(freq - expected) for freq in frequencies.values()),
            'entropy': entropy(list(frequencies.values()), base=2)
        }
    
    def zero_run_distribution(self, binary_str: str) -> Dict[int, float]:
        """Analyze distribution of zero run lengths."""
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
    
    def analyze_normality(self, length: int = 10000) -> Dict:
        """Comprehensive normality analysis."""
        binary_expansion = self.generate_binary_expansion(length)

        
        # Analyze block frequencies for different sizes
        block_analysis = {
            size: self.analyze_block_frequencies(binary_expansion, size)
            for size in range(1, 6)
        }

        
        # Analyze zero run distribution
        run_dist = self.zero_run_distribution(binary_expansion)

        # Compare with theoretical bounds
        theoretical_bounds = {}
        max_discrepancy = 0
        entropy_value = 0
        
        if run_dist:
            theoretical_bounds = {
                k: 2**(-k) for k in range(1, max(run_dist.keys()) + 1)
            }
            
            # Calculate discrepancy only for existing run lengths
            max_discrepancy = max(
                abs(run_dist[k] - theoretical_bounds[k]) 
                for k in run_dist.keys() 
                if k in theoretical_bounds
            )
            
            # Calculate entropy only for non-zero probabilities
            non_zero_probs = [p for p in run_dist.values() if p > 0]
            if non_zero_probs:
                entropy_value = entropy(non_zero_probs, base=2)
        
        positions = np.arange(1, length + 1)
        log2_bounds = np.log2(positions)
    
        return {
            'block_analysis': block_analysis,
            'run_distribution': run_dist,
            'theoretical_bounds': theoretical_bounds,
            'log2_bounds': log2_bounds,
            'normality_measures': {
                'entropy': entropy_value,
                'max_discrepancy': max_discrepancy
            }
        }
    
    def plot_normality_analysis(self, analysis_results: Dict):
        """Generate visualization of normality analysis."""
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Block frequencies vs expected
        plt.subplot(2, 2, 1)
        block_size = 3  # Example block size
        freqs = analysis_results['block_analysis'][block_size]['frequencies']
        expected = analysis_results['block_analysis'][block_size]['expected']
        plt.bar(freqs.keys(), freqs.values(), alpha=0.5, label='Observed')
        plt.axhline(y=expected, color='r', linestyle='--', label='Expected')
        plt.title(f'Block Frequencies (size {block_size})')
        plt.legend()
        
        # Plot 2: Zero run distribution vs theoretical bounds
        plt.subplot(2, 2, 2)
        run_dist = analysis_results['run_distribution']
        theoretical = analysis_results['theoretical_bounds']
        plt.semilogy(run_dist.keys(), run_dist.values(), 'bo-', label='Observed')
        plt.semilogy(theoretical.keys(), theoretical.values(), 'r--', label='Theoretical')
        plt.title('Zero Run Distribution')
        plt.legend()
        
        # Plot 3: Entropy by block size
        plt.subplot(2, 2, 3)
        entropies = [analysis['entropy'] 
                    for analysis in analysis_results['block_analysis'].values()]
        plt.plot(range(1, len(entropies) + 1), entropies, 'go-')
        plt.title('Block Entropy vs Block Size')
        plt.xlabel('Block Size')
        plt.ylabel('Entropy (bits)')
        
        plt.tight_layout()
        return plt
    
    def save_report(self, analysis_results: Dict, filename: str):
        """Save detailed analysis report to file."""
        with open(filename, "w") as f:
            f.write(f"Normality Analysis for Length {len(analysis_results['log2_bounds'])}}}\n")
            f.write("\\subsection{Block Analysis}\n")
            for size, analysis in analysis_results['block_analysis'].items():
                f.write(f"\\subsubsection{{Block Size {size}}}\n")
                f.write(f"Discrepancy: {analysis['discrepancy']:.2e}, Entropy: {analysis['entropy']:.2f}\n")
                
            f.write("\\subsection{Zero Run Distribution}\n")
            if analysis_results['run_distribution']:
                max_run = max(analysis_results['run_distribution'].keys())
                f.write(f"Maximum observed run length: {max_run}\n")
            else:
                f.write("No zero runs found\n")
                
            f.write("\\subsection{Theoretical Bounds}\n")
            for k, bound in analysis_results['theoretical_bounds'].items():
                f.write(f"Run length {k}: {bound:.2e}\n")
                
            f.write("\\subsection{Normality Measures}\n")
            f.write(f"Entropy: {analysis_results['normality_measures']['entropy']:.2f}\n")
            f.write(f"Max Discrepancy: {analysis_results['normality_measures']['max_discrepancy']:.2e}\n")

def main():
    analyzer = NormalityAnalyzer()
    lengths = [1000, 5000, 10000, 50000]
    results = {}

    # File path
    file_path = 'math_problems/chatgpt/final_paper/Code/data/'

    for length in lengths:
        results[length] = analyzer.analyze_normality(length)
        
        plt = analyzer.plot_normality_analysis(results[length])
        plt.savefig(f'{file_path}normality_analysis_{length}.png')
        plt.close()

        
        print(f"\nAnalysis for length {length}:")
        print("Normality measures:", results[length]['normality_measures'])
        analyzer.save_report(results[length], f'{file_path}normality_analysis_{length}.tex')

        print("Block analysis:")
        for size, analysis in results[length]['block_analysis'].items():
            print(f"Block size {size}: Discrepancy={analysis['discrepancy']:.2e}, Entropy={analysis['entropy']:.2f}")

        print("Zero run distribution:")
        if results[length]['run_distribution']:
            max_run = max(results[length]['run_distribution'].keys())
            print(f"Maximum observed run length: {max_run}")
        else:
            print("No zero runs found")
            
        print(f"Theoretical bound log2(n): {np.log2(length)}")

if __name__ == "__main__":
    main()