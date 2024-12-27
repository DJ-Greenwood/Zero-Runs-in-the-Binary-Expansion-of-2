# imports
from decimal import Decimal, getcontext
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any
from math_problems.finalPaper.Code.extended_zero_run_analysis import Sqrt2ZeroRunAnalyzer


class PrecisionManager:
    """Manages precision requirements and error bounds for sqrt(2) analysis."""
    
    def __init__(self, base_precision=1000):
        self.base_precision = base_precision
        self.safety_factor = 1.2  # Extra precision buffer
        
    def required_precision(self, n: int, k: int) -> int:
        """Calculate required precision for given n and k values."""
        return int(self.safety_factor * (n + k))
        
    def error_bound(self, n: int, k: int, P: int) -> Decimal:
        """Calculate error bound for given parameters."""
        return Decimal(2) ** (-P) * (Decimal(2) ** n + Decimal(2) ** k)
        
    def validate_precision(self, n: int, k: int, current_precision: int) -> bool:
        """Check if current precision is sufficient."""
        required = self.required_precision(n, k)
        return current_precision >= required

class PerformanceOptimizer:
    """Implements performance optimizations for zero run analysis."""
    
    def __init__(self, max_workers=4):
        self.max_workers = max_workers
        
    def parallel_analyze_positions(self, analyzer, positions: List[int], k: int) -> List[Dict]:
        """Analyze multiple positions in parallel."""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(analyzer.analyze_run, n, k)
                for n in positions
            ]
            return [future.result() for future in futures]
            
    def binary_search_valid_k(self, analyzer, n: int, max_k: int) -> int:
        """Find maximum valid k for position n using binary search."""
        left, right = 1, max_k
        while left <= right:
            k = (left + right) // 2
            result = analyzer.analyze_run(n, k)
            if result['constraints']['all_satisfied']:
                left = k + 1
            else:
                right = k - 1
        return right

class ExtendedSqrt2Analyzer(Sqrt2ZeroRunAnalyzer):
    """Extends the base analyzer with additional optimization features."""
    
    def __init__(self, precision: int = 10000):
        super().__init__(precision)
        self.precision_manager = PrecisionManager(precision)
        self.optimizer = PerformanceOptimizer()
        
    def analyze_with_precision_control(self, n: int, k: int) -> Dict[str, Any]:
        """Analyze zero run with automatic precision management."""
        if not self.precision_manager.validate_precision(n, k, getcontext().prec):
            required_prec = self.precision_manager.required_precision(n, k)
            getcontext().prec = required_prec
            
        result = self.analyze_run(n, k)
        result['precision'] = {
            'used': getcontext().prec,
            'error_bound': float(self.precision_manager.error_bound(n, k, getcontext().prec))
        }
        return result
        
    def bulk_analysis(self, n_range: List[int], k_range: List[int]) -> Dict[str, Any]:
        """Perform bulk analysis with optimizations."""
        results = []
        for n in n_range:
            # Use binary search to find maximum valid k
            max_k = self.optimizer.binary_search_valid_k(self, n, max(k_range))
            # Analyze positions in parallel for valid k values
            k_values = [k for k in k_range if k <= max_k]
            results.extend(
                self.optimizer.parallel_analyze_positions(self, [n], k_values)
            )
        return {
            'results': results,
            'summary': self._generate_summary(results)
        }
        
    def _generate_summary(self, results: List[Dict]) -> Dict[str, Any]:
        """Generate summary statistics for bulk analysis."""
        return {
            'total_analyzed': len(results),
            'satisfied_constraints': sum(
                1 for r in results if r['constraints']['all_satisfied']
            ),
            'max_valid_k': max(
                r['run_length'] for r in results 
                if r['constraints']['all_satisfied']
            ),
            'average_error': sum(
                r['approximation']['error'] for r in results
            ) / len(results)
        }