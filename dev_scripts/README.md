# Development Scripts

This directory contains scripts used for development, debugging, benchmarking, and analysis of the ACE framework. These are **not required** for normal usage of ACE.

## Directory Structure

```
dev_scripts/
  benchmarks/       # Performance comparison scripts
  debug/            # Debug and diagnostic utilities
  analysis/         # Result analysis tools
  examples/         # Example applications (fibonacci, email validator, etc.)
```

## Subdirectories

### `benchmarks/`

Scripts for comparing ACE performance against other tools:

- `*_benchmark.py` - Performance benchmark scripts
- `*_headtohead.py` - Direct comparison tests
- `compare_*.py` - Comparison utilities

### `debug/`

Debugging and diagnostic utilities:

- `debug_*.py` - Debug scripts for specific components
- `check_*.py` - Validation scripts
- `*_test.py` - One-off test scripts

### `analysis/`

Scripts for analyzing benchmark results and performance data:

- `analyze_*.py` - Analysis utilities
- Result processing tools

### `examples/`

Example applications demonstrating ACE capabilities:

- `fibonacci.py` - Fibonacci sequence example
- `email_validator.py` - Email validation example
- `temperature_converter.py` - Temperature conversion example

## Usage

Most scripts can be run directly from the repository root:

```bash
# Run from repository root
python dev_scripts/benchmarks/benchmark_1000_queries.py

# Or from the dev_scripts directory
cd dev_scripts
python benchmarks/benchmark_1000_queries.py
```

## Notes

- These scripts may have additional dependencies not required for production use
- Some scripts reference internal paths that assume execution from repository root
- Debug output may be verbose - intended for development use only
