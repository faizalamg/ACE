"""
Comprehensive test suite for Fibonacci Calculator implementation.

This test suite validates all aspects of the Fibonacci implementation including:
- Correctness of algorithms
- Error handling
- Edge cases
- Performance characteristics
- Utility functions

Author: Elite Software Engineer
Version: 1.0.0
"""

import unittest
import time
import warnings
from typing import List
import sys
import os

# Add current directory to path for import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fibonacci import (
    FibonacciCalculator,
    FibonacciAlgorithm,
    fibonacci,
    is_fibonacci_number,
    fibonacci_position,
    InvalidInputError,
    FibonacciError,
    OverflowWarning
)


class TestFibonacciCalculator(unittest.TestCase):
    """Test cases for the FibonacciCalculator class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.calc = FibonacciCalculator()

    def test_initialization(self):
        """Test calculator initialization with different algorithms."""
        # Test default initialization
        calc1 = FibonacciCalculator()
        self.assertEqual(calc1.algorithm, FibonacciAlgorithm.ITERATIVE)

        # Test initialization with specific algorithm
        calc2 = FibonacciCalculator(FibonacciAlgorithm.MATRIX)
        self.assertEqual(calc2.algorithm, FibonacciAlgorithm.MATRIX)

        # Test invalid initialization
        with self.assertRaises(ValueError):
            FibonacciCalculator("invalid")

    def test_basic_fibonacci_values(self):
        """Test basic Fibonacci number calculations."""
        test_cases = [
            (0, 0), (1, 1), (2, 1), (3, 2), (4, 3),
            (5, 5), (6, 8), (7, 13), (8, 21), (9, 34),
            (10, 55), (15, 610), (20, 6765), (25, 75025)
        ]

        for n, expected in test_cases:
            with self.subTest(n=n):
                result = self.calc.fibonacci(n)
                self.assertEqual(result, expected)

    def test_large_fibonacci_numbers(self):
        """Test calculation of larger Fibonacci numbers."""
        # Known values for validation
        large_test_cases = [
            (30, 832040),
            (40, 102334155),
            (50, 12586269025),
            (60, 1548008755920),
        ]

        for n, expected in large_test_cases:
            with self.subTest(n=n):
                result = self.calc.fibonacci(n)
                self.assertEqual(result, expected)

    def test_algorithm_consistency(self):
        """Test that all algorithms produce the same results."""
        test_values = [0, 1, 5, 10, 20, 30, 50]

        for n in test_values:
            with self.subTest(n=n):
                results = []
                for algorithm in FibonacciAlgorithm:
                    calc = FibonacciCalculator(algorithm)
                    result = calc.fibonacci(n)
                    results.append(result)

                # All algorithms should produce the same result
                self.assertTrue(all(r == results[0] for r in results))

    def test_input_validation(self):
        """Test input validation and error handling."""
        # Test negative input
        with self.assertRaises(InvalidInputError):
            self.calc.fibonacci(-1)

        # Test non-integer input
        with self.assertRaises(InvalidInputError):
            self.calc.fibonacci(10.5)

        with self.assertRaises(InvalidInputError):
            self.calc.fibonacci("10")

        with self.assertRaises(InvalidInputError):
            self.calc.fibonacci([10])

    def test_overflow_warning(self):
        """Test overflow warning for very large numbers."""
        # Test warning generation
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = self.calc.fibonacci(100001)  # Large number
            self.assertTrue(len(w) > 0)
            self.assertTrue(issubclass(w[0].category, OverflowWarning))

        # Test warning suppression
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = self.calc.fibonacci(100001, warn_large=False)
            self.assertEqual(len([warning for warning in w if issubclass(warning.category, OverflowWarning)]), 0)

    def test_fibonacci_sequence(self):
        """Test Fibonacci sequence generation."""
        # Test basic sequence
        sequence = self.calc.fibonacci_sequence(0, 11)
        expected = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
        self.assertEqual(sequence, expected)

        # Test partial sequence
        sequence = self.calc.fibonacci_sequence(5, 10)
        expected = [5, 8, 13, 21, 34]
        self.assertEqual(sequence, expected)

        # Test single element sequence
        sequence = self.calc.fibonacci_sequence(10, 11)
        expected = [55]
        self.assertEqual(sequence, expected)

        # Test invalid sequences
        with self.assertRaises(ValueError):
            self.calc.fibonacci_sequence(10, 5)

        with self.assertRaises(InvalidInputError):
            self.calc.fibonacci_sequence(-1, 10)

        with self.assertRaises(InvalidInputError):
            self.calc.fibonacci_sequence("0", "10")

    def test_golden_ratio_approximation(self):
        """Test golden ratio approximation."""
        # Known value: golden ratio ≈ 1.618033988749895
        phi_actual = (1 + 5**0.5) / 2

        for n in [10, 20, 50, 100]:
            with self.subTest(n=n):
                phi_approx = self.calc.get_golden_ratio_approximation(n)
                error = abs(phi_approx - phi_actual)
                self.assertLess(error, 0.01, f"Approximation error too large for n={n}")

    def test_cache_operations(self):
        """Test cache clearing functionality."""
        # Set up memoized calculator
        calc = FibonacciCalculator(FibonacciAlgorithm.MEMOIZED)

        # Calculate some values to populate cache
        calc.fibonacci(10)
        calc.fibonacci(20)

        # Clear cache
        calc.clear_cache()

        # Verify cache is reset
        self.assertEqual(calc.cache, {0: 0, 1: 1})

    def test_iterative_algorithm_performance(self):
        """Test iterative algorithm implementation."""
        calc = FibonacciCalculator(FibonacciAlgorithm.ITERATIVE)

        # Test that iterative algorithm works correctly
        self.assertEqual(calc.fibonacci(10), 55)
        self.assertEqual(calc.fibonacci(100), 354224848179261915075)

        # Test performance (should be fast)
        start_time = time.time()
        result = calc.fibonacci(10000)
        end_time = time.time()
        self.assertLess(end_time - start_time, 1.0, "Iterative algorithm should be fast for n=10000")

    def test_memoized_algorithm_performance(self):
        """Test memoized algorithm implementation."""
        calc = FibonacciCalculator(FibonacciAlgorithm.MEMOIZED)

        # Test that memoized algorithm works correctly
        self.assertEqual(calc.fibonacci(10), 55)
        self.assertEqual(calc.fibonacci(100), 354224848179261915075)

        # Test caching performance
        # First call
        start_time = time.time()
        result1 = calc.fibonacci(500)
        first_call_time = time.time() - start_time

        # Second call (should be faster due to caching)
        start_time = time.time()
        result2 = calc.fibonacci(500)
        second_call_time = time.time() - start_time

        self.assertEqual(result1, result2)
        self.assertLess(second_call_time, first_call_time * 0.1, "Second call should be much faster due to caching")

    def test_matrix_algorithm_performance(self):
        """Test matrix exponentiation algorithm implementation."""
        calc = FibonacciCalculator(FibonacciAlgorithm.MATRIX)

        # Test that matrix algorithm works correctly
        self.assertEqual(calc.fibonacci(10), 55)
        self.assertEqual(calc.fibonacci(100), 354224848179261915075)

        # Test performance for large numbers (should be faster than iterative)
        start_time = time.time()
        result = calc.fibonacci(100000)
        end_time = time.time()
        self.assertLess(end_time - start_time, 2.0, "Matrix algorithm should be fast for large n")

    def test_precomputed_values(self):
        """Test that precomputed values are returned correctly."""
        # Test that precomputed values are used
        for n in range(11):
            with self.subTest(n=n):
                result = self.calc.fibonacci(n)
                self.assertIn(n, self.calc._precomputed_values)


class TestConvenienceFunctions(unittest.TestCase):
    """Test cases for convenience functions."""

    def test_fibonacci_function(self):
        """Test the standalone fibonacci function."""
        # Test basic usage
        self.assertEqual(fibonacci(0), 0)
        self.assertEqual(fibonacci(10), 55)

        # Test with different algorithms
        self.assertEqual(fibonacci(10, algorithm=FibonacciAlgorithm.MATRIX), 55)
        self.assertEqual(fibonacci(10, algorithm=FibonacciAlgorithm.MEMOIZED), 55)

    def test_is_fibonacci_number(self):
        """Test the is_fibonacci_number function."""
        # Test known Fibonacci numbers
        fib_numbers = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233]
        for num in fib_numbers:
            with self.subTest(num=num):
                self.assertTrue(is_fibonacci_number(num))

        # Test non-Fibonacci numbers
        non_fib_numbers = [4, 6, 7, 9, 10, 11, 12, 14, 15]
        for num in non_fib_numbers:
            with self.subTest(num=num):
                self.assertFalse(is_fibonacci_number(num))

        # Test edge cases
        self.assertFalse(is_fibonacci_number(-1))

    def test_fibonacci_position(self):
        """Test the fibonacci_position function."""
        # Test known positions
        test_cases = [
            (0, 0), (1, 1), (2, 3), (3, 4), (5, 5),
            (8, 6), (13, 7), (21, 8), (34, 9), (55, 10)
        ]

        for num, expected_position in test_cases:
            with self.subTest(num=num):
                position = fibonacci_position(num)
                self.assertEqual(position, expected_position)

        # Test non-Fibonacci numbers
        self.assertIsNone(fibonacci_position(100))
        self.assertIsNone(fibonacci_position(1000))

        # Test edge cases
        self.assertEqual(fibonacci_position(0), 0)


class TestErrorHandling(unittest.TestCase):
    """Test cases specifically for error handling."""

    def test_custom_exceptions(self):
        """Test that custom exceptions work correctly."""
        calc = FibonacciCalculator()

        # Test InvalidInputError
        with self.assertRaises(InvalidInputError):
            calc.fibonacci(-5)

        # Test that InvalidInputError is a subclass of FibonacciError
        try:
            calc.fibonacci(-5)
        except Exception as e:
            self.assertIsInstance(e, FibonacciError)

    def test_type_errors(self):
        """Test type error handling."""
        calc = FibonacciCalculator()

        invalid_inputs = [
            3.14, "10", [10], {"n": 10}, None, (10,)
        ]

        for invalid_input in invalid_inputs:
            with self.subTest(invalid_input=invalid_input):
                with self.assertRaises(InvalidInputError):
                    calc.fibonacci(invalid_input)


class TestPerformanceCharacteristics(unittest.TestCase):
    """Test performance characteristics of different algorithms."""

    def test_algorithm_performance_comparison(self):
        """Compare performance of different algorithms."""
        test_n = 10000  # Large enough to see differences
        results = {}

        for algorithm in FibonacciAlgorithm:
            calc = FibonacciCalculator(algorithm)

            start_time = time.time()
            result = calc.fibonacci(test_n)
            end_time = time.time()

            results[algorithm.value] = {
                'time': end_time - start_time,
                'result': result
            }

        # Verify all algorithms produce the same result
        unique_results = set(r['result'] for r in results.values())
        self.assertEqual(len(unique_results), 1, "All algorithms should produce the same result")

        # Log performance results (for manual inspection)
        print("\nAlgorithm Performance Comparison:")
        for algo, data in results.items():
            print(f"{algo}: {data['time']:.6f} seconds")

    def test_memory_usage(self):
        """Test memory usage characteristics."""
        # This is a basic test - more sophisticated memory profiling
        # would require external libraries
        calc = FibonacciCalculator(FibonacciAlgorithm.MEMOIZED)

        # Calculate many values to populate cache
        for i in range(1000):
            calc.fibonacci(i)

        # Verify cache doesn't grow unbounded
        self.assertLess(len(calc.cache), 10000, "Cache should not grow without bound")


class TestMathematicalProperties(unittest.TestCase):
    """Test mathematical properties of Fibonacci numbers."""

    def test_fibonacci_recurrence(self):
        """Test that F(n) = F(n-1) + F(n-2) for n > 1."""
        calc = FibonacciCalculator()

        for n in range(2, 20):
            with self.subTest(n=n):
                fn = calc.fibonacci(n)
                fn_minus_1 = calc.fibonacci(n - 1)
                fn_minus_2 = calc.fibonacci(n - 2)
                self.assertEqual(fn, fn_minus_1 + fn_minus_2)

    def test_cassini_identity(self):
        """Test Cassini's identity: F(n-1) * F(n+1) - F(n)^2 = (-1)^n"""
        calc = FibonacciCalculator()

        for n in range(1, 15):
            with self.subTest(n=n):
                fn_minus_1 = calc.fibonacci(n - 1)
                fn = calc.fibonacci(n)
                fn_plus_1 = calc.fibonacci(n + 1)

                cassini = fn_minus_1 * fn_plus_1 - fn * fn
                expected = (-1) ** n
                self.assertEqual(cassini, expected)

    def test_sum_of_first_n_fibonacci(self):
        """Test that sum(F(0) to F(n)) = F(n+2) - 1"""
        calc = FibonacciCalculator()

        for n in range(0, 15):
            with self.subTest(n=n):
                # Calculate sum of first n+1 Fibonacci numbers
                sum_fib = sum(calc.fibonacci(i) for i in range(n + 1))
                # Expected: F(n+2) - 1
                expected = calc.fibonacci(n + 2) - 1
                self.assertEqual(sum_fib, expected)


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)