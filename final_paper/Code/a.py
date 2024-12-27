# Function to check if the equation holds for given values of n, k, p, and q
def verify_equation(n, k, p, q):
    left_side = 2**(2*n + 2*k + 1) - (p**2) * (2**(2*k))
    right_side = 2 * p * q * (2**k) + q**2
    
    return left_side <= right_side

# Example usage
n = 1  # Replace with the desired value of n
k = 2  # Replace with the desired value of k
p = 3  # Replace with the desired value of p
q = 4  # Replace with the desired value of q

# Check if the equation holds
if verify_equation(n, k, p, q):
    print("The equation holds for the given values of n, k, p, and q.")
else:
    print("The equation does NOT hold for the given values of n, k, p, and q.")
