import math

# Function to check if the equation holds for given values of n, k, p, and q
def verify_equation(n, k, p, q):
    left_side = 2**(2*n + 2*k + 1) - (p**2) * (2**(2*k))
    right_side = 2 * p * q * (2**k) + q**2
    
    return left_side <= right_side
def sqrt2_recursive(z, depth):
    if depth == 0:
        return z
    return math.sqrt(2 * sqrt2_recursive(z, depth - 1))

def main():
    # Example usage
    z_values = [2, 3, 4, 5]  # Replace with the desired list of z values
    depth_values = [1, 2, 3, 4]  # Replace with the desired list of depth values

    # Calculate the recursive square root of 2 for each combination of z and depth
    for z in z_values:
        for depth in depth_values:
            result = sqrt2_recursive(z, depth)
            print(f"The recursive square root of 2 with z={z} and depth={depth} is: {result}")

if __name__ == "__main__":
    main()