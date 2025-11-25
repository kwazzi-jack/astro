#!/usr/bin/env python3
"""
Test runner script for astro project.

This script provides convenient commands for running different test suites.
Only available when dev dependencies are installed.
"""

import sys
import subprocess
from pathlib import Path


def run_command(cmd: list[str]) -> int:
    """Run a command and return its exit code."""
    print(f"Running: {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=Path(__file__).parent.parent).returncode


def test_all():
    """Run all tests with verbose output."""
    return run_command(["pytest", "tests/", "-v"])


def test_with_coverage():
    """Run tests with coverage report."""
    return run_command([
        "pytest", "tests/", 
        "--cov=astro", 
        "--cov-report=term-missing", 
        "--cov-report=html"
    ])


def test_unit():
    """Run only unit tests."""
    return run_command(["pytest", "-m", "unit", "tests/", "-v"])


def test_filesystem():
    """Run only filesystem tests."""
    return run_command(["pytest", "-m", "filesystem", "tests/", "-v"])


def test_integration():
    """Run only integration tests."""
    return run_command(["pytest", "-m", "integration", "tests/", "-v"])


def test_fast():
    """Run all tests except slow ones."""
    return run_command(["pytest", "-m", "not slow", "tests/", "-v"])


def test_slow():
    """Run only slow tests."""
    return run_command(["pytest", "-m", "slow", "tests/", "-v"])


def test_watch():
    """Run tests in watch mode (stop on first failure)."""
    return run_command(["pytest", "tests/", "-v", "--tb=short", "-x"])


def test_debug():
    """Run tests in debug mode with detailed output."""
    return run_command(["pytest", "tests/", "-v", "-s", "--tb=long"])


def test_failed():
    """Re-run only failed tests."""
    return run_command(["pytest", "tests/", "--lf", "-v"])


def test_ruff():
    """Run ruff linting checks on test files."""
    return run_command(["pytest", "--ruff", "tests/", "scripts/"])


def test_black():
    """Run black code formatting checks on test files."""
    return run_command(["pytest", "--black", "tests/", "scripts/"])


def test_mypy():
    """Run mypy type checking on test files."""
    return run_command(["pytest", "--mypy", "tests/", "scripts/"])


def test_quality():
    """Run all code quality checks (ruff, black, mypy)."""
    print("Running all code quality checks...")
    
    # Run each check and collect results
    results = []
    
    print("\n=== Running Ruff Linting ===")
    ruff_result = run_command(["pytest", "--ruff", "tests/", "scripts/"])
    results.append(("Ruff", ruff_result))
    
    print("\n=== Running Black Formatting Check ===")
    black_result = run_command(["pytest", "--black", "tests/", "scripts/"])
    results.append(("Black", black_result))
    
    print("\n=== Running MyPy Type Checking ===")
    mypy_result = run_command(["pytest", "--mypy", "tests/", "scripts/"])
    results.append(("MyPy", mypy_result))
    
    # Print summary
    print("\n=== Code Quality Summary ===")
    all_passed = True
    for tool, result in results:
        status = "✓ PASS" if result == 0 else "✗ FAIL"
        print(f"{tool:>8}: {status}")
        if result != 0:
            all_passed = False
    
    return 0 if all_passed else 1


def test_all_with_quality():
    """Run all tests plus code quality checks."""
    print("Running comprehensive test suite with code quality checks...")
    
    # First run the regular tests
    print("\n=== Running Test Suite ===")
    test_result = test_all()
    
    # Then run quality checks
    print("\n=== Running Code Quality Checks ===")
    quality_result = test_quality()
    
    # Summary
    print("\n=== Final Summary ===")
    print(f"Tests: {'✓ PASS' if test_result == 0 else '✗ FAIL'}")
    print(f"Quality: {'✓ PASS' if quality_result == 0 else '✗ FAIL'}")
    
    return 0 if (test_result == 0 and quality_result == 0) else 1


def main():
    """Main entry point - dispatch to appropriate test function."""
    if len(sys.argv) != 2:
        print("Usage: test-runner <command>")
        print("\nTest Commands:")
        print("  all         - Run all tests")
        print("  cov         - Run tests with coverage")
        print("  unit        - Run unit tests only")
        print("  filesystem  - Run filesystem tests only")
        print("  integration - Run integration tests only")
        print("  fast        - Run fast tests (exclude slow)")
        print("  slow        - Run slow tests only")
        print("  watch       - Run in watch mode")
        print("  debug       - Run in debug mode")
        print("  failed      - Re-run failed tests")
        print("\nCode Quality Commands:")
        print("  ruff        - Run ruff linting")
        print("  black       - Run black formatting check")
        print("  mypy        - Run mypy type checking")
        print("  quality     - Run all quality checks")
        print("  full        - Run all tests + quality checks")
        return 1
    
    command = sys.argv[1]
    
    # Map commands to functions
    commands = {
        "all": test_all,
        "cov": test_with_coverage,
        "unit": test_unit,
        "filesystem": test_filesystem,
        "integration": test_integration,
        "fast": test_fast,
        "slow": test_slow,
        "watch": test_watch,
        "debug": test_debug,
        "failed": test_failed,
        "ruff": test_ruff,
        "black": test_black,
        "mypy": test_mypy,
        "quality": test_quality,
        "full": test_all_with_quality,
    }
    
    if command not in commands:
        print(f"Unknown command: {command}")
        return 1
    
    return commands[command]()


if __name__ == "__main__":
    sys.exit(main())