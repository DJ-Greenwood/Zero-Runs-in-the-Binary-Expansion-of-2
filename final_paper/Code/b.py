from decimal import Decimal, getcontext
import math

def generate_binary_sqrt2(precision):
    """
    Generate the binary expansion of sqrt(2) up to a given precision.
    """
    getcontext().prec = precision
    sqrt2 = Decimal(2).sqrt()
    binary_expansion = []
    
    for _ in range(precision):
        sqrt2 *= 2
        if sqrt2 >= 1:
            binary_expansion.append(1)
            sqrt2 -= 1
        else:
            binary_expansion.append(0)
    
    return binary_expansion

def analyze_zero_runs(binary_expansion):
    """
    Identify zero runs in the binary expansion and their lengths.
    """
    zero_runs = []
    current_run = 0
    for bit in binary_expansion:
        if bit == 0:
            current_run += 1
        else:
            if current_run > 0:
                zero_runs.append(current_run)
            current_run = 0
    if current_run > 0:  # Handle trailing zeros
        zero_runs.append(current_run)
    return zero_runs

def validate_zero_runs(zero_runs, precision):
    """
    Validate zero runs against the theoretical bound.
    """
    results = []
    for n, k in enumerate(zero_runs, start=1):
        theoretical_bound = math.log2(n) + 0.5
        results.append((n, k, theoretical_bound, k <= theoretical_bound))
    return results

# Parameters
precision = 10000  # Number of binary digits to generate

# Step 1: Generate binary expansion
binary_expansion = generate_binary_sqrt2(precision)

# Step 2: Analyze zero runs
zero_runs = analyze_zero_runs(binary_expansion)

# Step 3: Validate zero runs
validation_results = validate_zero_runs(zero_runs, precision)

# Step 4: Display results
for n, k, bound, valid in validation_results[:100]:  # Display first 100 results
    print(f"Position: {n}, Zero Run Length: {k}, Theoretical Bound: {bound:.2f}, Valid: {valid}")

# Summary statistics
total_runs = len(validation_results)
valid_runs = sum(1 for _, _, _, valid in validation_results if valid)
if total_runs > 0:
    print(f"\nTotal Runs: {total_runs}, Valid Runs: {valid_runs}, Valid Percentage: {100 * valid_runs / total_runs:.2f}%")
else:
    print("\nTotal Runs: 0, Valid Runs: 0, Valid Percentage: N/A")
