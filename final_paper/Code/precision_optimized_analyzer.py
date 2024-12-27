from decimal import Decimal, getcontext
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any, Optional
import numpy as np
import logging
from contextlib import contextmanager
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PrecisionManager:
    """Manages precision requirements and error bounds for sqrt(2) analysis."""
    
    def __init__(self, base_precision: int = 1000):
        self.base_precision = base_precision
        self.safety_factor = 1.2  # Extra precision buffer
        
    def required_precision(self, n: int, k: int) -> int:
        """Calculate required precision for given n and k values."""
        return max(
            int(self.safety_factor * (n + k)),
            self.base_precision
        )
        
    def error_bound(self, n: int, k: int, P: int) -> Decimal:
        """Calculate error bound for given parameters with improved accuracy."""
        with self.temp_precision(P * 2):  # Double precision for intermediate calculations
            # Account for rounding errors in intermediate calculations
            term1 = Decimal(2) ** (-P)
            term2 = Decimal(2) ** n
            term3 = Decimal(2) ** k
            
            # Add safety factor for numerical stability
            error = term1 * (term2 + term3) * (1 + Decimal('1e-10'))
            
            return error.normalize()
        
    def validate_precision(self, n: int, k: int, current_precision: int) -> bool:
        """Check if current precision is sufficient."""
        required = self.required_precision(n, k)
        return current_precision >= required

    @contextmanager
    def temp_precision(self, precision: int):
        """Context manager for temporarily changing precision."""
        old_prec = getcontext().prec
        try:
            getcontext().prec = precision
            yield
        finally:
            getcontext().prec = old_prec

class PerformanceOptimizer:
    """Implements performance optimizations for zero run analysis."""
    
    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        
    def parallel_analyze_positions(self, analyzer, n: int, k_values: List[int]) -> List[Dict]:
        """Analyze multiple k values for a position in parallel."""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(analyzer.analyze_run, n, k)
                for k in k_values
            ]
            return [future.result() for future in futures]
            
    def binary_search_valid_k(self, analyzer, n: int, max_k: int) -> int:
        """Find maximum valid k for position n using binary search."""
        left, right = 1, max_k
        last_valid = 0
        
        while left <= right:
            k = (left + right) // 2
            try:
                result = analyzer.analyze_run(n, k)
                if result['constraints']['all_satisfied']:
                    last_valid = k
                    left = k + 1
                else:
                    right = k - 1
            except Exception as e:
                logger.warning(f"Error analyzing n={n}, k={k}: {e}")
                right = k - 1
                
        return last_valid

class PrecisionOptimizedAnalyzer:
    """Enhanced analyzer with precision management and performance optimization."""
    
    def __init__(self, precision: int = 10000):
        self.precision_manager = PrecisionManager(precision)
        self.optimizer = PerformanceOptimizer()
        with self.precision_manager.temp_precision(precision):
            self.sqrt_2 = Decimal(2).sqrt()
        
    def analyze_run(self, n: int, k: int) -> Dict[str, Any]:
        """Analyze a zero run with improved precision control."""
        required_prec = self.precision_manager.required_precision(n, k)
        
        with self.precision_manager.temp_precision(required_prec * 2):  # Double precision for intermediate calculations
            # Calculate p and q with higher precision
            p = int(self.sqrt_2 * Decimal(2) ** n)
            q = int((self.sqrt_2 - Decimal(p) / Decimal(2) ** n) * Decimal(2) ** (n + k))
            
            # Validate constraints
            integer_valid = self._check_integer_constraint(q)
            next_bit_valid = self._check_next_bit_constraint(n, k, p, q)
            sqrt2_valid = self._check_sqrt2_constraint(n, k, p, q)
            
            # Calculate error
            error = abs(self.sqrt_2 - (Decimal(p) / Decimal(2) ** n + Decimal(q) / Decimal(2) ** (n + k)))
        
        return {
            'position': n,
            'run_length': k,
            'constraints': {
                'integer_valid': integer_valid,
                'next_bit_valid': next_bit_valid,
                'sqrt2_valid': sqrt2_valid,
                'all_satisfied': all([integer_valid, next_bit_valid, sqrt2_valid])
            },
            'approximation': {
                'p': p,
                'q': q,
                'error': error,
                'quality': -error.log10() if error > 0 else Decimal('inf')
            },
            'theoretical': {
                'log2n': Decimal(str(np.log2(n))) if n > 0 else Decimal('0'),
                'exceeds_bound': k > np.log2(n) if n > 0 else True
            },
            'precision': {
                'used': required_prec,
                'error_bound': Decimal(self.precision_manager.error_bound(n, k, required_prec))
            }
        }
    
    def _check_integer_constraint(self, q: int) -> bool:
        """Check if q is a valid integer."""
        return isinstance(q, int)
    
    def _check_next_bit_constraint(self, n: int, k: int, p: int, q: int) -> bool:
        """Check next bit constraint with improved numerical stability."""
        with self.precision_manager.temp_precision(getcontext().prec * 2):
            remainder = (self.sqrt_2 
                        - Decimal(p) / Decimal(2) ** n 
                        - Decimal(q) / Decimal(2) ** (n + k))
            
            scaled_remainder = remainder * Decimal(2) ** (n + k + 1)
            tolerance = Decimal('1e-100')
            
            return scaled_remainder >= (Decimal(1) - tolerance)
    
    def _check_sqrt2_constraint(self, n: int, k: int, p: int, q: int) -> bool:
        """Check if approximation is valid for sqrt(2)."""
        with self.precision_manager.temp_precision(getcontext().prec * 2):
            approx = Decimal(p) / Decimal(2) ** n + Decimal(q) / Decimal(2) ** (n + k)
            return approx < self.sqrt_2
    
    def batch_analysis(self, n_values: List[int], k_range: List[int], 
                      optimize: bool = True) -> Dict[str, Any]:
        """Perform batch analysis with optimizations."""
        results = []
        
        for n in n_values:
            if optimize:
                max_k = min(k_range[-1], int(2 * np.log2(max(n, 1))) + 5)
                valid_k_values = [k for k in k_range if k <= max_k]
            else:
                valid_k_values = k_range
            
            batch_results = self.optimizer.parallel_analyze_positions(self, n, valid_k_values)
            results.extend(batch_results)
        
        summary = self._generate_summary(results)
        return {
            'results': results,
            'summary': summary,
            'parameters': {
                'n_values': n_values,
                'k_range': k_range,
                'optimize': optimize,
                'base_precision': self.precision_manager.base_precision
            }
        }

    def _generate_summary(self, results: List[Dict]) -> Dict[str, Any]:
        """Generate improved summary statistics."""
        valid_results = [r for r in results if r['constraints']['all_satisfied']]

        if not results:
            return {
                'total_configurations': 0,
                'valid_configurations': 0,
                'max_valid_k': 0,
                'average_error': 0.0,
                'constraint_satisfaction': {'integer_valid': 0, 'next_bit_valid': 0, 'sqrt2_valid': 0},
            }

        total_configs = len(results)
        valid_configs = len(valid_results)
        max_k = max([r['run_length'] for r in valid_results], default=0)
        avg_error = sum(r['approximation']['error'] for r in results) / total_configs
        constraint_rates = {
            key: sum(1 for r in results if r['constraints'][key]) / total_configs
            for key in ['integer_valid', 'next_bit_valid', 'sqrt2_valid']
        }

        return {
            'total_configurations': total_configs,
            'valid_configurations': valid_configs,
            'max_valid_k': max_k,
            'average_error': float(avg_error),
            'constraint_satisfaction': {k: f"{v * 100:.2f}%" for k, v in constraint_rates.items()},
        }
    
    def summarize_results(self, results: List[Dict]) -> str:
        """Generate a summary of the analysis results in a formatted string."""
        valid_results = [r for r in results if r['constraints']['all_satisfied']]
        
        if not results:
            return "No configurations were analyzed."
        
        # Total configurations and valid configurations
        total_configs = len(results)
        valid_configs = len(valid_results)
        
        # Average error and error bounds
        avg_error = sum(r['approximation']['error'] for r in results) / total_configs
        min_error_bound = min(r['precision']['error_bound'] for r in results)
        max_error_bound = max(r['precision']['error_bound'] for r in results)
        
        # Maximum valid k
        max_valid_k = max([r['run_length'] for r in valid_results], default=0)
        
        # Constraint satisfaction rates
        constraint_rates = {
            key: sum(1 for r in results if r['constraints'][key]) / total_configs
            for key in ['integer_valid', 'next_bit_valid', 'sqrt2_valid']
        }
        
        # Generate summary string
        summary = (
            f"Summary of Results:\n\n"
            f"1. General Overview:\n"
            f"   - Total Configurations Analyzed: {total_configs}\n"
            f"   - Valid Configurations: {valid_configs}\n\n"
            f"2. Error and Precision Metrics:\n"
            f"   - Average Error: {avg_error:.10f}\n"
            f"   - Error Bounds: {min_error_bound} to {max_error_bound}\n\n"
            f"3. Constraint Satisfaction Rates:\n"
            f"   - Integer Validity: {constraint_rates['integer_valid'] * 100:.2f}%\n"
            f"   - Next Bit Validity: {constraint_rates['next_bit_valid'] * 100:.2f}%\n"
            f"   - Sqrt(2) Validity: {constraint_rates['sqrt2_valid'] * 100:.2f}%\n\n"
            f"4. Optimization Metrics:\n"
            f"   - Maximum Valid k: {max_valid_k}\n"
        )
        
        return summary

       
    def save_results(self, results: Dict[str, Any], filename: str):
        """Save analysis results to a file."""
        with open(filename, "w") as f:
            f.write("Analysis Results:\n")
            f.write(f"Parameters: {results}\n\n")

def main():
    """Example usage of the precision optimized analyzer."""
    try:
        # Initialize analyzer with higher precision
        analyzer = PrecisionOptimizedAnalyzer(precision=10000)
                
        # Test parameters
        n_values = [10, 20, 50, 100]
        k_values = list(range(2, 15))
        
        logger.info("Running batch analysis...")
        results = analyzer.batch_analysis(n_values, k_values, optimize=True)
        
        logger.info("\nDetailed Analysis for Selected Positions:")

        results = analyzer.batch_analysis(n_values, k_values, optimize=True)
        summary = analyzer.summarize_results(results['results'])
        analyzer.save_results(results, "precision_optimized_analysis_results.txt")
        analyzer.save_results(summary, "precision_optimized_analysis_summary.txt")
        print(summary)
    except Exception as e:
        logger.error(f"Error in main: {e}", exc_info=True)
if __name__ == "__main__":
    main()