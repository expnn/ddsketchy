"""
Test serialization size estimation for DDSketch Python bindings.

This test verifies that the estimated serialization size is always >= actual size,
ensuring efficient buffer pre-allocation without reallocations.
"""
import unittest
from ddsketchy import DDSketch


class TestSerializationSizeEstimation(unittest.TestCase):
    """Verify that estimated serialization size is an upper bound."""

    def test_empty_sketch(self):
        """Test estimation for empty sketch."""
        sketch = DDSketch()
        estimated = sketch.estimated_serialized_size()
        actual = len(sketch.dumps())
        
        self.assertGreaterEqual(estimated, actual,
            f"Empty sketch: estimated ({estimated}) should be >= actual ({actual})")
        self.assertGreater(estimated, 0, "Estimated size should be positive")
        # Empty sketch should have reasonable size (136 bytes base + margin)
        self.assertLess(estimated, 200)

    def test_small_sketch(self):
        """Test estimation for small sketch (10 values)."""
        sketch = DDSketch()
        sketch.add_batch([float(i) for i in range(1, 11)])
        estimated = sketch.estimated_serialized_size()
        actual = len(sketch.dumps())
        
        self.assertGreaterEqual(estimated, actual,
            f"Small sketch: estimated ({estimated}) should be >= actual ({actual})")

    def test_medium_sketch(self):
        """Test estimation for medium sketch (1000 values)."""
        sketch = DDSketch()
        sketch.add_batch([float(i) for i in range(1, 1001)])
        estimated = sketch.estimated_serialized_size()
        actual = len(sketch.dumps())
        
        self.assertGreaterEqual(estimated, actual,
            f"Medium sketch: estimated ({estimated}) should be >= actual ({actual})")

    def test_large_sketch(self):
        """Test estimation for large sketch (10000 values)."""
        sketch = DDSketch()
        sketch.add_batch([float(i) for i in range(1, 10001)])
        estimated = sketch.estimated_serialized_size()
        actual = len(sketch.dumps())
        
        self.assertGreaterEqual(estimated, actual,
            f"Large sketch: estimated ({estimated}) should be >= actual ({actual})")

    def test_with_negative_values(self):
        """Test estimation with negative values."""
        sketch = DDSketch()
        sketch.add_batch([float(i) for i in range(-100, 101)])
        estimated = sketch.estimated_serialized_size()
        actual = len(sketch.dumps())
        
        self.assertGreaterEqual(estimated, actual,
            f"Negative values: estimated ({estimated}) should be >= actual ({actual})")

    def test_wide_range_values(self):
        """Test estimation with wide value range."""
        sketch = DDSketch()
        values = [10.0 ** exp for exp in range(-10, 11)]
        sketch.add_batch(values)
        estimated = sketch.estimated_serialized_size()
        actual = len(sketch.dumps())
        
        self.assertGreaterEqual(estimated, actual,
            f"Wide range: estimated ({estimated}) should be >= actual ({actual})")

    def test_different_alpha_values(self):
        """Test estimation with different alpha (precision) values."""
        for alpha in [0.1, 0.05, 0.01, 0.005, 0.001]:
            with self.subTest(alpha=alpha):
                sketch = DDSketch(alpha=alpha)
                sketch.add_batch([float(i) for i in range(1, 1001)])
                estimated = sketch.estimated_serialized_size()
                actual = len(sketch.dumps())
                
                self.assertGreaterEqual(estimated, actual,
                    f"Alpha={alpha}: estimated ({estimated}) should be >= actual ({actual})")

    def test_high_precision_sketch(self):
        """Test estimation for high precision sketch."""
        sketch = DDSketch(alpha=0.0001)
        sketch.add_batch([float(i) for i in range(1, 10001)])
        estimated = sketch.estimated_serialized_size()
        actual = len(sketch.dumps())
        
        self.assertGreaterEqual(estimated, actual,
            f"High precision: estimated ({estimated}) should be >= actual ({actual})")

    def test_mixed_values(self):
        """Test estimation with positive, negative, and zero values."""
        sketch = DDSketch()
        sketch.add(0.0)
        sketch.add(0.0)
        values = []
        for i in range(1, 51):
            values.append(float(i))
            values.append(-float(i))
        sketch.add_batch(values)
        estimated = sketch.estimated_serialized_size()
        actual = len(sketch.dumps())
        
        self.assertGreaterEqual(estimated, actual,
            f"Mixed values: estimated ({estimated}) should be >= actual ({actual})")

    def test_serialization_roundtrip(self):
        """Verify that serialization/deserialization preserves data."""
        sketch = DDSketch()
        sketch.add_batch([float(i) for i in range(1, 101)])
        
        original_count = sketch.count
        original_sum = sketch.sum
        original_q50 = sketch.quantile(0.5)
        
        # Serialize and deserialize
        data = sketch.dumps()
        restored = DDSketch.loads(data)
        
        # Verify data integrity
        self.assertEqual(original_count, restored.count)
        self.assertAlmostEqual(original_sum, restored.sum, places=10)
        self.assertAlmostEqual(original_q50, restored.quantile(0.5), places=5)


class TestSerializationSizes(unittest.TestCase):
    """Informational tests to show actual serialization sizes."""

    def test_print_serialization_sizes(self):
        """Print serialization sizes for different sketch configurations."""
        test_cases = [
            ("Empty sketch", lambda: DDSketch()),
            ("Small (10 values)", lambda: create_sketch(range(1, 11))),
            ("Medium (1000 values)", lambda: create_sketch(range(1, 1001))),
            ("Large (10000 values)", lambda: create_sketch(range(1, 10001))),
            ("With negatives", lambda: create_sketch(range(-100, 101))),
            ("Wide range", lambda: create_sketch([10**e for e in range(-10, 11)])),
        ]
        
        print("\n" + "=" * 80)
        print("Serialization Size Report")
        print("=" * 80)
        print(f"{'Configuration':<25s} {'Estimated':>10s} {'Actual':>10s} {'Difference':>12s} {'Overhead':>10s}")
        print("-" * 80)
        
        for name, factory in test_cases:
            sketch = factory()
            estimated = sketch.estimated_serialized_size()
            actual = len(sketch.dumps())
            diff = estimated - actual
            overhead_pct = (diff / actual * 100) if actual > 0 else 0
            
            print(f"{name:<25s} {estimated:>10d} {actual:>10d} {diff:>+11d} {overhead_pct:>9.1f}%")
        
        print("=" * 80)
        print("Note: Estimated size is guaranteed to be >= actual size")
        print("=" * 80)


def create_sketch(values):
    """Helper to create a sketch from an iterable of values."""
    sketch = DDSketch()
    sketch.add_batch([float(v) for v in values])
    return sketch


if __name__ == '__main__':
    unittest.main(verbosity=2)
