#!/usr/bin/env python
"""Test the specific example provided by the user."""

from pathlib import Path

from anomalib.utils.path import generate_output_filename

# Example 1: Traditional MVTecAD path
input_path = Path("/data/MVTecAD/bottle/test/broken_large/000.png")
output_path = Path("./results")
output_path.mkdir(exist_ok=True)

print("Testing the specific example...")
print(f"Input path: {input_path}")

# Test with explicit dataset name
result1 = generate_output_filename(
    input_path=input_path, output_path=output_path, dataset_name="MVTecAD", category="bottle"
)
print(f"1. With explicit dataset_name and category:")
print(f"   Output: {result1}")
print(f"   Expected: results/test/broken_large/000.png")

# Test with auto-detected dataset name
result2 = generate_output_filename(
    input_path=input_path,
    output_path=output_path,
    dataset_name=None,  # Auto-detect
    category="bottle",
)
print(f"2. With auto-detected dataset_name and category:")
print(f"   Output: {result2}")
print(f"   Expected: results/test/broken_large/000.png")

# Test with path that doesn't contain MVTecAD
input_path2 = Path("/custom/data/path/bottle/test/broken_large/000.png")
print(f"\nInput path without MVTecAD: {input_path2}")

result3 = generate_output_filename(
    input_path=input_path2,
    output_path=output_path,
    dataset_name=None,  # Auto-detect
    category="bottle",
)
print(f"3. Custom path with auto-detected dataset_name and category:")
print(f"   Output: {result3}")
print(f"   Expected: results/test/broken_large/000.png")

print("\nDone testing.")
