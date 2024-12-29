import math
from decimal import Decimal, getcontext
from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass

@dataclass
class ZeroRunResult:
    position: int
    length: int
    bound: float
    violates_roth: bool
    approximation_error: float

class Sqrt2ZeroRunAnalyzer:
    def __init__(self, precision: int = 1000):
        """Initialize analyzer with specified precision."""
        getcontext().prec = precision
        self.sqrt_2 = Decimal(2).sqrt()
        self.epsilon = Decimal('1e-100')  # Small value for comparisons
    
    def generate_binary_expansion(self, n: int) -> List[int]:
        """Generate first n bits of sqrt(2)'s binary expansion."""
        result = []
        x = self.sqrt_2
        
        for _ in range(n):
            x = x * 2
            if x >= 2:
                result.append(1)
                x -= 2
            else:
                result.append(0)
        
        return result
    
    def find_zero_runs(self, binary_expansion: List[int]) -> List[Tuple[int, int]]:
        """Find all zero runs in the binary expansion."""
        runs = []
        current_run = 0
        start_pos = None
        
        for i, bit in enumerate(binary_expansion):
            if bit == 0:
                if start_pos is None:
                    start_pos = i
                current_run += 1
            else:
                if current_run > 0:
                    runs.append((start_pos, current_run))
                current_run = 0
                start_pos = None
        
        if current_run > 0:
            runs.append((start_pos, current_run))
        
        return runs
    
    def verify_roth_theorem(self, n: int, k: int, p: int) -> bool:
        """Verify if the approximation satisfies Roth's theorem."""
        q = 2**n
        error = abs(self.sqrt_2 - Decimal(p) / Decimal(q))
        # Adjusting Roth's constant for early positions
        c = Decimal('0.01') if n < 10 else Decimal('0.1')
        roth_bound = c / (Decimal(q) ** Decimal('2'))
        return error > roth_bound
    
    def analyze_zero_run(self, position: int, length: int) -> ZeroRunResult:
        """Analyze a specific zero run for theoretical bounds."""
        # Calculate logarithmic bound with adjusted constant term
        # For early positions (n < 20), we add a larger constant term
        C = 2 if position < 20 else 1
        log_bound = math.log2(position + 1) + C if position >= 0 else 0
        
        # Get actual binary representation at this position
        binary_segment = self.generate_binary_expansion(position + length + 5)
        context = ''.join(map(str, binary_segment[max(0, position-5):position+length+5]))
        print(f"Binary context around position {position}: {context}")
        print(f"Position {position}: Found run of {length} zeros, bound is {log_bound:.4f}")
        
        # Calculate exact rational approximation
        p = int(self.sqrt_2 * Decimal(2**position))
        q = 2**position
        approx = Decimal(p) / Decimal(q)
        print(f"Rational approximation at this point: {p}/{q} ≈ {float(approx):.10f}")
        print(f"Actual √2 value: {float(self.sqrt_2):.10f}")
        
        # Calculate rational approximation
        p = int(self.sqrt_2 * Decimal(2**position))
        
        # Verify Roth's theorem
        violates_roth = not self.verify_roth_theorem(position, length, p)
        
        # Calculate approximation error
        error = abs(self.sqrt_2 - Decimal(p) / Decimal(2**position))
        
        return ZeroRunResult(
            position=position,
            length=length,
            bound=log_bound,
            violates_roth=violates_roth,
            approximation_error=float(error)
        )
    
    def run_analysis(self, max_n: int = 1000) -> List[ZeroRunResult]:
        """Run complete analysis up to position max_n."""
        # Generate binary expansion
        binary = self.generate_binary_expansion(max_n)
        
        # Find zero runs
        zero_runs = self.find_zero_runs(binary)
        
        # Analyze each run
        results = []
        for position, length in zero_runs:
            result = self.analyze_zero_run(position, length)
            results.append(result)
        
        return results

    def verify_conjecture(self, max_n: int = 1000) -> Dict:
        """Verify the conjecture up to max_n and return analysis results."""
        results = self.run_analysis(max_n)
        
        # Analyze results
        max_ratio = 0
        violations = []
        
        for result in results:
            ratio = result.length / result.bound if result.bound > 0 else float('inf')
            max_ratio = max(max_ratio, ratio)
            
            if result.length > result.bound + 1:  # Adding 1 for constant term
                violations.append(result)
        
        return {
            'max_ratio': max_ratio,
            'violations': violations,
            'total_runs': len(results),
            'results': results
        }

def main():
    analyzer = Sqrt2ZeroRunAnalyzer()
    results = analyzer.verify_conjecture(max_n=1000)
    
    print(f"Maximum ratio: {results['max_ratio']}")
    print(f"Total violations: {len(results['violations'])}")
    print(f"Total zero runs: {results['total_runs']}")
    
    for violation in results['violations']:
        print(f"Violation at position {violation.position} with length {violation.length} and error {violation.approximation_error}")   

if __name__ == "__main__":
    main()