from decimal import Decimal, getcontext

class Ramanujan:
    def ramanujan_series_sqrt2(self, terms: int) -> Decimal:
        """
        Compute √2 using Ramanujan's series:
        √2 = [∞/n=0]∑ (-1)^n * (2n-1)!! / (2n)!!
        where !! denotes double factorial.
        
        Parameters:
        terms (int): Number of terms to include in the series.
        
        Returns:
        Decimal: Approximation of √2.
        """
        # Increase precision for Decimal calculations
        getcontext().prec = 500  # Adjust precision as needed
        
        def double_factorial(n: int) -> int:
            """
            Compute the double factorial of n iteratively for efficiency.
            n!! = n * (n-2) * (n-4) * ... * (1 or 2)
            """
            result = 1
            while n > 1:
                result *= n
                n -= 2
            return result

        # Start with the first term in the series
        result = Decimal(1)
        
        # Compute subsequent terms
        for n in range(1, terms):
            numerator = double_factorial(2 * n - 1)
            denominator = double_factorial(2 * n)
            term = Decimal(numerator) / Decimal(denominator)
            if n % 2 == 1:  # Alternate sign for odd n
                term = -term
            result += term

        # Multiply the result by 2 (as per the series formula)
        return Decimal(2) * result
    
# Create an instance of the class
ramanujan = Ramanujan()

# Compute √2 using 10 terms
sqrt2_approx = ramanujan.ramanujan_series_sqrt2(terms=12345)
print("Approximation of √2:", sqrt2_approx)
