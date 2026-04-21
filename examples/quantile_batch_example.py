"""Example demonstrating unified quantile functionality."""
from ddsketchy import DDSketch
import numpy as np


def main():
    # Create a sketch and add some data
    sketch = DDSketch()
    sketch.add_batch(range(1, 1001))  # Add values 1 to 1000
    
    print("DDSketch unified quantile example")
    print("=" * 50)
    print(f"Sketch: {sketch}")
    print()
    
    # Example 1: Single quantile (returns float)
    print("1. Single quantile (returns float):")
    median = sketch.quantile(0.5)
    print(f"   P50 = {median:.2f}")
    print()
    
    # Example 2: Using Python list (returns list)
    print("2. Using Python list (returns list):")
    quantiles_list = [0.5, 0.9, 0.95, 0.99]
    results = sketch.quantile(quantiles_list)
    for q, r in zip(quantiles_list, results):
        print(f"   P{int(q*100):2d} = {r:.2f}")
    print()
    
    # Example 3: Using NumPy array (zero-copy, faster)
    print("3. Using NumPy array (zero-copy):")
    quantiles_numpy = np.array([0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    results = sketch.quantile(quantiles_numpy)
    for q, r in zip(quantiles_numpy, results):
        print(f"   P{int(q*100):2d} = {r:.2f}")
    print()
    
    # Example 4: Computing many quantiles efficiently
    print("4. Computing 100 quantiles efficiently:")
    quantiles_many = np.linspace(0.01, 0.99, 100)
    results = sketch.quantile(quantiles_many)
    print(f"   Computed {len(results)} quantiles")
    print(f"   P10  = {results[9]:.2f}")
    print(f"   P50  = {results[49]:.2f}")
    print(f"   P90  = {results[89]:.2f}")
    print(f"   P99  = {results[98]:.2f}")
    print()
    
    # Example 5: Consistency check (single vs batch)
    print("5. Consistency check (single vs batch):")
    test_quantiles = [0.1, 0.5, 0.9]
    batch_results = sketch.quantile(test_quantiles)
    individual_results = [sketch.quantile(q) for q in test_quantiles]
    
    print("   Quantile | Batch    | Individual | Match?")
    print("   " + "-" * 50)
    for q, batch_r, ind_r in zip(test_quantiles, batch_results, individual_results):
        match = abs(batch_r - ind_r) < 1e-10
        print(f"   P{int(q*100):2d}      | {batch_r:8.4f} | {ind_r:10.4f} | {match}")


if __name__ == "__main__":
    main()
