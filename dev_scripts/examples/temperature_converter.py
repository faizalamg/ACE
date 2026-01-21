#!/usr/bin/env python3
"""
Temperature Converter Tool
Converts temperatures between Celsius, Fahrenheit, and Kelvin with input validation.
"""

import argparse
import sys
from typing import Union, Tuple


class TemperatureConverter:
    """A class to handle temperature conversions between Celsius, Fahrenheit, and Kelvin."""

    @staticmethod
    def celsius_to_fahrenheit(celsius: float) -> float:
        """Convert Celsius to Fahrenheit."""
        return (celsius * 9/5) + 32

    @staticmethod
    def celsius_to_kelvin(celsius: float) -> float:
        """Convert Celsius to Kelvin."""
        return celsius + 273.15

    @staticmethod
    def fahrenheit_to_celsius(fahrenheit: float) -> float:
        """Convert Fahrenheit to Celsius."""
        return (fahrenheit - 32) * 5/9

    @staticmethod
    def fahrenheit_to_kelvin(fahrenheit: float) -> float:
        """Convert Fahrenheit to Kelvin."""
        celsius = TemperatureConverter.fahrenheit_to_celsius(fahrenheit)
        return TemperatureConverter.celsius_to_kelvin(celsius)

    @staticmethod
    def kelvin_to_celsius(kelvin: float) -> float:
        """Convert Kelvin to Celsius."""
        return kelvin - 273.15

    @staticmethod
    def kelvin_to_fahrenheit(kelvin: float) -> float:
        """Convert Kelvin to Fahrenheit."""
        celsius = TemperatureConverter.kelvin_to_celsius(kelvin)
        return TemperatureConverter.celsius_to_fahrenheit(celsius)


def validate_temperature(value: Union[str, float], unit: str) -> Tuple[float, str]:
    """
    Validate temperature input and return as float with unit.

    Args:
        value: Temperature value to validate
        unit: Unit of the temperature (C, F, K)

    Returns:
        Tuple of (validated_temperature, normalized_unit)

    Raises:
        ValueError: If input is invalid
    """
    # Try to convert to float
    try:
        if isinstance(value, str):
            temp = float(value)
        else:
            temp = value
    except ValueError:
        raise ValueError(f"Invalid temperature value: '{value}'. Must be a number.")

    # Normalize unit
    unit = unit.upper().strip()
    if unit in ['C', 'CELSIUS']:
        unit = 'C'
    elif unit in ['F', 'FAHRENHEIT']:
        unit = 'F'
    elif unit in ['K', 'KELVIN']:
        unit = 'K'
    else:
        raise ValueError(f"Invalid temperature unit: '{unit}'. Use C, F, or K.")

    # Validate temperature ranges
    if unit == 'K' and temp < 0:
        raise ValueError("Kelvin temperature cannot be negative (absolute zero).")

    # Reasonable temperature ranges for validation
    if unit == 'C' and temp < -273.15:
        raise ValueError("Celsius temperature cannot be below absolute zero (-273.15°C).")

    if unit == 'F' and temp < -459.67:
        raise ValueError("Fahrenheit temperature cannot be below absolute zero (-459.67°F).")

    return temp, unit


def convert_temperature(temp: float, from_unit: str, to_unit: str) -> float:
    """
    Convert temperature from one unit to another.

    Args:
        temp: Temperature value
        from_unit: Current unit (C, F, K)
        to_unit: Target unit (C, F, K)

    Returns:
        Converted temperature value

    Raises:
        ValueError: If units are invalid
    """
    converter = TemperatureConverter()

    # Convert to Celsius first if not already Celsius
    if from_unit == 'C':
        celsius_temp = temp
    elif from_unit == 'F':
        celsius_temp = converter.fahrenheit_to_celsius(temp)
    elif from_unit == 'K':
        celsius_temp = converter.kelvin_to_celsius(temp)
    else:
        raise ValueError(f"Invalid source unit: {from_unit}")

    # Convert from Celsius to target unit
    if to_unit == 'C':
        return celsius_temp
    elif to_unit == 'F':
        return converter.celsius_to_fahrenheit(celsius_temp)
    elif to_unit == 'K':
        return converter.celsius_to_kelvin(celsius_temp)
    else:
        raise ValueError(f"Invalid target unit: {to_unit}")


def format_temperature(temp: float, unit: str, precision: int = 2) -> str:
    """Format temperature with appropriate precision and unit symbol."""
    unit_symbols = {'C': '°C', 'F': '°F', 'K': 'K'}
    symbol = unit_symbols.get(unit, unit)
    return f"{temp:.{precision}f}{symbol}"


def main():
    """Main CLI interface for the temperature converter."""
    parser = argparse.ArgumentParser(
        description="Convert temperatures between Celsius, Fahrenheit, and Kelvin",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s 25 C F      # Convert 25°C to Fahrenheit
  %(prog)s 77 F C      # Convert 77°F to Celsius
  %(prog)s 300 K C     # Convert 300K to Celsius
  %(prog)s 100 C       # Convert 100°C to both F and K
  %(prog)s --list      # Show conversion chart for common temperatures
        """
    )

    parser.add_argument(
        'temperature',
        type=float,
        nargs='?',
        help='Temperature value to convert'
    )

    parser.add_argument(
        'from_unit',
        type=str,
        nargs='?',
        help='Source temperature unit (C, F, K)'
    )

    parser.add_argument(
        'to_unit',
        type=str,
        nargs='?',
        help='Target temperature unit (C, F, K). If omitted, converts to both other units.'
    )

    parser.add_argument(
        '-p', '--precision',
        type=int,
        default=2,
        help='Decimal precision for output (default: 2)'
    )

    parser.add_argument(
        '--list',
        action='store_true',
        help='Show conversion chart for common temperatures'
    )

    args = parser.parse_args()

    # Handle list option
    if args.list:
        print("Temperature Conversion Chart")
        print("=" * 50)
        print(f"{'°C':<10}{'°F':<10}{'K':<10}")
        print("-" * 30)

        # Common reference temperatures
        reference_temps = [
            -273.15,  # Absolute zero
            -40,      # Where °C = °F
            -20,      # Very cold
            0,        # Freezing point
            20,       # Room temperature
            37,       # Body temperature
            100,      # Boiling point
        ]

        for c in reference_temps:
            f = TemperatureConverter.celsius_to_fahrenheit(c)
            k = TemperatureConverter.celsius_to_kelvin(c)
            print(f"{c:<10.1f}{f:<10.1f}{k:<10.1f}")

        return

    # Validate required arguments
    if args.temperature is None:
        parser.error("Temperature value is required")

    if args.from_unit is None:
        parser.error("Source unit is required (C, F, or K)")

    try:
        # Validate input
        temp, from_unit = validate_temperature(args.temperature, args.from_unit)

        if args.to_unit:
            # Convert to specific target unit
            to_unit = args.to_unit.upper().strip()
            if to_unit not in ['C', 'F', 'K']:
                # Expand unit names
                unit_map = {'CELSIUS': 'C', 'FAHRENHEIT': 'F', 'KELVIN': 'K'}
                to_unit = unit_map.get(to_unit.upper(), to_unit)

            if to_unit not in ['C', 'F', 'K']:
                raise ValueError(f"Invalid target unit: {args.to_unit}")

            if to_unit == from_unit:
                print(f"{format_temperature(temp, from_unit, args.precision)} is the same in {to_unit}")
                return

            converted = convert_temperature(temp, from_unit, to_unit)
            print(f"{format_temperature(temp, from_unit, args.precision)} = {format_temperature(converted, to_unit, args.precision)}")

        else:
            # Convert to both other units
            units = ['C', 'F', 'K']
            other_units = [u for u in units if u != from_unit]

            print(f"Temperature: {format_temperature(temp, from_unit, args.precision)}")
            print("Equivalent values:")

            for unit in other_units:
                converted = convert_temperature(temp, from_unit, unit)
                print(f"  {format_temperature(converted, unit, args.precision)}")

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()