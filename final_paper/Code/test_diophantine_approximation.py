import math

def test_diophantine_approximation(alpha, max_denominator, c, d):
    """
    Test Diophantine approximation bounds for an algebraic number.

    Args:
        alpha (float): The algebraic number (e.g., sqrt(2)).
        max_denominator (int): Maximum denominator for rational approximations.
        c (float): Constant in the bound c/q^d.
        d (float): Exponent in the bound (must be > 2).

    Returns:
        List[dict]: Results showing denominator, approximation, and error.
    """
    results = []
    for q in range(1, max_denominator + 1):
        # Find the best numerator p for the given denominator q
        p = round(alpha * q)
        # Compute the rational approximation and error
        approximation = p / q
        error = abs(alpha - approximation)
        # Check if the error satisfies Roth's theorem
        bound = c / (q ** d)
        satisfies_bound = error > bound

        results.append({
            'p': p,
            'q': q,
            'approximation': approximation,
            'error': error,
            'bound': bound,
            'satisfies_bound': satisfies_bound
        })
    
    return results

# Parameters
sqrt_2 = math.sqrt(2)
max_denominator = 100  # Maximum denominator to test
c = 1  # Constant in the bound
d = 2.1  # Exponent (must be > 2)

# Run the test
results = test_diophantine_approximation(sqrt_2, max_denominator, c, d)

# Display results
for result in results:
    print(f"p/q = {result['p']}/{result['q']}, "
          f"Approx: {result['approximation']:.8f}, "
          f"Error: {result['error']:.8e}, "
          f"Bound: {result['bound']:.8e}, "
          f"Satisfies Roth: {result['satisfies_bound']}")
