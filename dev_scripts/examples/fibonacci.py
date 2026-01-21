"""
Fibonacci Number Calculator

A high-performance, production-ready implementation for calculating Fibonacci numbers
with comprehensive error handling, multiple algorithm options, and extensive documentation.

Author: Elite Software Engineer
Version: 1.0.0
"""

from __future__ import annotations

import functools
from typing import Union, Optional, Callable, Any
from enum import Enum


class FibonacciAlgorithm(Enum):
    """Enumeration of available Fibonacci calculation algorithms."""
    ITERATIVE = "iterative"
    MEMOIZED = "memoized"
    MATRIX = "matrix"


class FibonacciError(Exception):
    """Base exception class for Fibonacci calculation errors."""
    pass


class InvalidInputError(FibonacciError):
    """Exception raised for invalid input values."""
    pass


class OverflowWarning(UserWarning):
    """Warning for potentially very large Fibonacci numbers."""
    pass


class FibonacciCalculator:
    """
    High-performance Fibonacci number calculator with multiple algorithm options.

    This class provides efficient calculation of Fibonacci numbers using different
    algorithms suitable for various use cases, from small numbers to extremely
    large indices.

    Attributes:
        algorithm (FibonacciAlgorithm): The chosen calculation algorithm
        cache (dict): Internal cache for memoization (when applicable)

    Examples:
        >>> calculator = FibonacciCalculator()
        >>> calculator.fibonacci(10)
        55
        >>> calculator.fibonacci(0)
        0
        >>> calculator.fibonacci(1)
        1
    """

    def __init__(self, algorithm: FibonacciAlgorithm = FibonacciAlgorithm.ITERATIVE):
        """
        Initialize the Fibonacci calculator with specified algorithm.

        Args:
            algorithm: The calculation algorithm to use. Defaults to ITERATIVE.

        Raises:
            ValueError: If an unsupported algorithm is provided.
        """
        if not isinstance(algorithm, FibonacciAlgorithm):
            raise ValueError(f"Algorithm must be a FibonacciAlgorithm enum value")

        self.algorithm = algorithm
        self.cache: dict[int, int] = {0: 0, 1: 1}

        # Pre-compute commonly used values for faster access
        self._precomputed_values = {
            0: 0, 1: 1, 2: 1, 3: 2, 4: 3, 5: 5,
            6: 8, 7: 13, 8: 21, 9: 34, 10: 55,
            20: 6765, 50: 12586269025, 100: 354224848179261915075
        }

    def fibonacci(self, n: int, *, warn_large: bool = True) -> int:
        """
        Calculate the nth Fibonacci number using the configured algorithm.

        Args:
            n: Non-negative integer representing the position in the Fibonacci sequence
            warn_large: Whether to warn for potentially large numbers. Defaults to True.

        Returns:
            The nth Fibonacci number as a Python integer.

        Raises:
            InvalidInputError: If n is negative or not an integer
            OverflowWarning: If the result might be extremely large (when warn_large=True)

        Examples:
            >>> calc = FibonacciCalculator()
            >>> calc.fibonacci(10)
            55
            >>> calc.fibonacci(0)
            0
        """
        # Input validation
        if not isinstance(n, int):
            raise InvalidInputError(f"Fibonacci index must be an integer, got {type(n).__name__}")

        if n < 0:
            raise InvalidInputError(f"Fibonacci index cannot be negative, got {n}")

        # Check for very large numbers that might cause performance issues
        if warn_large and n > 100000:
            import warnings
            warnings.warn(
                f"Calculating Fibonacci({n}) may be computationally expensive "
                f"and result in an extremely large number (~{n * 0.208987:.0f} digits)",
                OverflowWarning
            )

        # Fast path for commonly used values
        if n in self._precomputed_values:
            return self._precomputed_values[n]

        # Route to appropriate algorithm
        if self.algorithm == FibonacciAlgorithm.ITERATIVE:
            return self._fibonacci_iterative(n)
        elif self.algorithm == FibonacciAlgorithm.MEMOIZED:
            return self._fibonacci_memoized(n)
        elif self.algorithm == FibonacciAlgorithm.MATRIX:
            return self._fibonacci_matrix(n)
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")

    def _fibonacci_iterative(self, n: int) -> int:
        """
        Calculate Fibonacci number using iterative approach (O(n) time, O(1) space).

        This is the most balanced approach for typical use cases, providing
        linear time complexity with constant space usage.

        Args:
            n: Non-negative integer

        Returns:
            The nth Fibonacci number
        """
        if n <= 1:
            return n

        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b

        return b

    @functools.lru_cache(maxsize=1000)
    def _fibonacci_memoized(self, n: int) -> int:
        """
        Calculate Fibonacci number using memoized iterative-recursive hybrid approach.

        This method provides O(n) time complexity with caching to avoid
        redundant calculations. Suitable for repeated calculations
        with overlapping subproblems while avoiding recursion depth issues.

        Args:
            n: Non-negative integer

        Returns:
            The nth Fibonacci number
        """
        if n <= 1:
            return n

        # Use internal cache for very fast lookups
        if n in self.cache:
            return self.cache[n]

        # For very large numbers, use iterative approach to avoid recursion depth
        if n > 1000:
            return self._fibonacci_iterative(n)

        # Recursive calculation with caching for smaller numbers
        result = self._fibonacci_memoized(n - 1) + self._fibonacci_memoized(n - 2)

        # Update cache if size is reasonable
        if len(self.cache) < 10000:
            self.cache[n] = result

        return result

    def _fibonacci_matrix(self, n: int) -> int:
        """
        Calculate Fibonacci number using matrix exponentiation (O(log n) time).

        This method is optimal for very large indices where the O(log n)
        complexity provides significant performance benefits.

        Args:
            n: Non-negative integer

        Returns:
            The nth Fibonacci number
        """
        if n <= 1:
            return n

        def matrix_multiply(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
            """Multiply two 2x2 matrices represented as tuples."""
            return (
                a[0] * b[0] + a[1] * b[2],  # top-left
                a[0] * b[1] + a[1] * b[3],  # top-right
                a[2] * b[0] + a[3] * b[2],  # bottom-left
                a[2] * b[1] + a[3] * b[3]   # bottom-right
            )

        def matrix_power(matrix: tuple[int, int, int, int], power: int) -> tuple[int, int, int, int]:
            """Calculate matrix to the power of exponent using binary exponentiation."""
            result = (1, 0, 0, 1)  # Identity matrix
            base = matrix

            while power > 0:
                if power % 2 == 1:
                    result = matrix_multiply(result, base)
                base = matrix_multiply(base, base)
                power //= 2

            return result

        # Transformation matrix for Fibonacci sequence
        transformation_matrix = (1, 1, 1, 0)

        # Calculate (transformation_matrix)^(n-1)
        result_matrix = matrix_power(transformation_matrix, n - 1)

        # F(n) = result_matrix[0][0] * F(1) + result_matrix[0][1] * F(0)
        # Since F(1) = 1 and F(0) = 0, we just need result_matrix[0][0]
        return result_matrix[0]

    def fibonacci_sequence(self, start: int = 0, end: int = 10) -> list[int]:
        """
        Generate a sequence of Fibonacci numbers from start to end (exclusive).

        Args:
            start: Starting index (inclusive). Defaults to 0.
            end: Ending index (exclusive). Defaults to 10.

        Returns:
            List of Fibonacci numbers from F(start) to F(end-1)

        Raises:
            InvalidInputError: If start or end are invalid
            ValueError: If start >= end
        """
        if not isinstance(start, int) or not isinstance(end, int):
            raise InvalidInputError("Start and end must be integers")

        if start < 0 or end < 0:
            raise InvalidInputError("Start and end must be non-negative")

        if start >= end:
            raise ValueError(f"Start ({start}) must be less than end ({end})")

        # Use the most efficient method for the range
        if end - start <= 100 or self.algorithm == FibonacciAlgorithm.ITERATIVE:
            return self._sequence_iterative(start, end)
        else:
            return [self.fibonacci(i, warn_large=False) for i in range(start, end)]

    def _sequence_iterative(self, start: int, end: int) -> list[int]:
        """Generate sequence iteratively for small ranges."""
        sequence = []

        # Generate up to start index
        a, b = 0, 1
        for i in range(end):
            if i >= start:
                sequence.append(a)
            a, b = b, a + b

        return sequence

    def get_golden_ratio_approximation(self, n: int = 50) -> float:
        """
        Calculate the golden ratio approximation using Fibonacci numbers.

        Args:
            n: Index to use for approximation. Higher values provide better accuracy.

        Returns:
            Approximation of the golden ratio φ ≈ 1.618033988749895
        """
        if n < 2:
            n = 2

        fn = self.fibonacci(n)
        fn_minus_1 = self.fibonacci(n - 1)

        return fn / fn_minus_1

    def clear_cache(self) -> None:
        """Clear the internal cache for memoized calculations."""
        self.cache.clear()
        self.cache.update({0: 0, 1: 1})
        if hasattr(self._fibonacci_memoized, 'cache_clear'):
            self._fibonacci_memoized.cache_clear()


# Convenience functions for common use cases
def fibonacci(n: int, *, algorithm: FibonacciAlgorithm = FibonacciAlgorithm.ITERATIVE,
               warn_large: bool = True) -> int:
    """
    Convenience function to calculate a single Fibonacci number.

    This is a simplified interface for the most common use case.
    For advanced features, use FibonacciCalculator class directly.

    Args:
        n: Non-negative integer representing the position in the Fibonacci sequence
        algorithm: Calculation algorithm to use. Defaults to ITERATIVE.
        warn_large: Whether to warn for potentially large numbers. Defaults to True.

    Returns:
        The nth Fibonacci number

    Examples:
        >>> fibonacci(10)
        55
        >>> fibonacci(100, algorithm=FibonacciAlgorithm.MATRIX)
        354224848179261915075
    """
    calculator = FibonacciCalculator(algorithm)
    return calculator.fibonacci(n, warn_large=warn_large)


def is_fibonacci_number(num: int) -> bool:
    """
    Check if a number is in the Fibonacci sequence using mathematical properties.

    A number is Fibonacci if and only if one or both of (5*n^2 + 4) or (5*n^2 - 4)
    is a perfect square.

    Args:
        num: Number to check (can be any integer)

    Returns:
        True if the number is a Fibonacci number, False otherwise
    """
    if num < 0:
        return False

    def is_perfect_square(x: int) -> bool:
        """Check if x is a perfect square."""
        if x < 0:
            return False
        root = int(x ** 0.5)
        return root * root == x

    # Mathematical test for Fibonacci numbers
    test1 = 5 * num * num + 4
    test2 = 5 * num * num - 4

    return is_perfect_square(test1) or is_perfect_square(test2)


def fibonacci_position(num: int) -> Optional[int]:
    """
    Find the position (index) of a number in the Fibonacci sequence.

    Args:
        num: Number to find in the sequence

    Returns:
        The index n such that F(n) = num, or None if num is not a Fibonacci number

    Examples:
        >>> fibonacci_position(55)
        10
        >>> fibonacci_position(100)
        None
    """
    if not is_fibonacci_number(num):
        return None

    if num == 0:
        return 0
    elif num == 1:
        return 1  # Note: 1 appears at positions 1 and 2, we return 1

    # Use the closed-form approximation to find the position
    import math

    # Binet's formula inverted: n = log_phi(F * sqrt(5) + 0.5)
    phi = (1 + math.sqrt(5)) / 2
    sqrt5 = math.sqrt(5)

    approx_n = math.log(num * sqrt5 + 0.5, phi)
    n = round(approx_n)

    # Verify the result
    if fibonacci(n) == num:
        return n
    elif fibonacci(n + 1) == num:
        return n + 1
    else:
        # Fallback search (should rarely be needed)
        calc = FibonacciCalculator()
        a, b = 0, 1
        index = 0
        while a < num:
            a, b = b, a + b
            index += 1

        return index if a == num else None


if __name__ == "__main__":
    # Demo and basic testing
    print("Fibonacci Calculator Demo")
    print("=" * 40)

    calc = FibonacciCalculator()

    # Test basic functionality
    test_values = [0, 1, 2, 5, 10, 20, 50]
    print("Basic Fibonacci numbers:")
    for n in test_values:
        print(f"F({n}) = {calc.fibonacci(n)}")

    # Test different algorithms
    print("\nAlgorithm comparison for F(30):")
    for algorithm in FibonacciAlgorithm:
        calc.algorithm = algorithm
        result = calc.fibonacci(30)
        print(f"{algorithm.value.upper()}: {result}")

    # Test sequence generation
    print("\nFibonacci sequence F(0) to F(10):")
    print(calc.fibonacci_sequence(0, 11))

    # Test golden ratio approximation
    print(f"\nGolden ratio approximation (F(50)/F(49)): {calc.get_golden_ratio_approximation(50)}")

    # Test Fibonacci number checking
    test_numbers = [34, 55, 100, 144]
    print("\nFibonacci number checking:")
    for num in test_numbers:
        is_fib = is_fibonacci_number(num)
        position = fibonacci_position(num) if is_fib else None
        print(f"{num}: Fibonacci? {is_fib}" + (f" (Position: {position})" if position else ""))