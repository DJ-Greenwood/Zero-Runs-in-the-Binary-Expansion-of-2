from decimal import Decimal, getcontext
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import pandas as pd
from tqdm import tqdm

class Sqrt2ExtendedAnalyzer:
    def __init__(self, precision: int = 10000):
        getcontext().prec = precision
        self.sqrt_2 = Decimal(2).sqrt()
        self.EPSILON = Decimal('1e-' + str(precision // 2))
        
    def analyze_position(self, n: int, k_max: int = None) -> Dict:
        """Analyze zero runs at a specific position with dynamic k_max."""
        if k_max is None:
            k_max = int(2 * np.log2(n)) + 10  # Test beyond theoretical bound
            
        results = []
        for k in range(2, k_max + 1):
            analysis = self._analyze_single_run(n, k)
            results.append({
                'position': n,
                'run_length': k,
                'theoretical_bound': np.log2(n),
                'exceeds_bound': k > np.log2(n),
                **analysis
            })
        return results

    def _analyze_single_run(self, n: int, k: int) -> Dict:
        """Analyze a single zero run configuration."""
        p = int(self.sqrt_2 * Decimal(2 ** n))
        q = int((self.sqrt_2 - Decimal(p) / Decimal(2 ** n)) * Decimal(2 ** (n + k)))
        
        # Core constraints
        integer_valid = abs(Decimal(q) - Decimal(round(q))) < self.EPSILON
        next_bit_valid = (self.sqrt_2 - Decimal(p) / Decimal(2 ** n) - 
                         Decimal(q) / Decimal(2 ** (n + k))) * Decimal(2 ** (n + k + 1)) >= 1
        sqrt2_valid = abs((Decimal(p) / Decimal(2 ** n) + 
                          Decimal(q) / Decimal(2 ** (n + k))) ** 2 - Decimal(2)) < self.EPSILON
        
        # Error analysis
        approx = Decimal(p) / Decimal(2 ** n) + Decimal(q) / Decimal(2 ** (n + k))
        error = abs(self.sqrt_2 - approx)
        
        return {
            'integer_valid': integer_valid,
            'next_bit_valid': next_bit_valid,
            'sqrt2_valid': sqrt2_valid,
            'all_constraints_satisfied': all([integer_valid, next_bit_valid, sqrt2_valid]),
            'error': float(error),
            'p': p,
            'q': q
        }

    def run_extended_analysis(self, 
                            n_values: List[int], 
                            precisions: List[int] = [1000, 10000, 100000]) -> pd.DataFrame:
        """Run comprehensive analysis across multiple positions and precision levels."""
        all_results = []
        
        for precision in tqdm(precisions, desc="Processing precision levels"):
            getcontext().prec = precision
            self.sqrt_2 = Decimal(2).sqrt()
            self.EPSILON = Decimal('1e-' + str(precision // 2))
            
            with ProcessPoolExecutor() as executor:
                futures = [executor.submit(self.analyze_position, n) for n in n_values]
                for future in tqdm(futures, desc=f"Analyzing positions (precision={precision})"):
                    results = future.result()
                    for result in results:
                        result['precision'] = precision
                        all_results.append(result)
                        
        return pd.DataFrame(all_results)

    def plot_scaling_behavior(self, df: pd.DataFrame):
        """Generate plots analyzing scaling behavior."""
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Error rates vs position for different precisions
        plt.subplot(2, 2, 1)
        for precision in df['precision'].unique():
            mask = df['precision'] == precision
            plt.semilogy(df[mask]['position'], df[mask]['error'], 
                        label=f'Precision {precision}')
        plt.xlabel('Position (n)')
        plt.ylabel('Error Rate')
        plt.title('Error Rates vs Position')
        plt.legend()
        
        # Plot 2: Maximum valid run length vs position
        plt.subplot(2, 2, 2)
        valid_runs = df[df['all_constraints_satisfied']].groupby('position')['run_length'].max()
        theoretical = np.log2(valid_runs.index)
        plt.plot(valid_runs.index, valid_runs.values, 'bo-', label='Observed')
        plt.plot(valid_runs.index, theoretical, 'r--', label='Theoretical log2(n)')
        plt.xlabel('Position (n)')
        plt.ylabel('Maximum Valid Run Length')
        plt.title('Maximum Valid Run Length vs Position')
        plt.legend()
        
        # Plot 3: Constraint satisfaction rates
        plt.subplot(2, 2, 3)
        constraints = ['integer_valid', 'next_bit_valid', 'sqrt2_valid']
        for constraint in constraints:
            satisfaction_rate = df.groupby('position')[constraint].mean()
            plt.plot(satisfaction_rate.index, satisfaction_rate.values, 
                    label=constraint.replace('_', ' ').title())
        plt.xlabel('Position (n)')
        plt.ylabel('Satisfaction Rate')
        plt.title('Constraint Satisfaction Rates')
        plt.legend()
        
        plt.tight_layout()
        return plt

def main():
    # Test parameters
    n_values = [10, 50, 100, 500, 1000, 5000, 10000]
    precisions = [1000, 10000, 100000]
    
    # Initialize analyzer
    analyzer = Sqrt2ExtendedAnalyzer()
    
    # Run analysis
    results_df = analyzer.run_extended_analysis(n_values, precisions)
    
    # Save results
    results_df.to_csv('sqrt2_extended_analysis.csv')
    
    # Generate and save plots
    plt = analyzer.plot_scaling_behavior(results_df)
    plt.savefig('sqrt2_scaling_analysis.png')
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(results_df.groupby('precision')[['error', 'all_constraints_satisfied']].agg(['mean', 'std', 'min', 'max']))

if __name__ == "__main__":
    main()