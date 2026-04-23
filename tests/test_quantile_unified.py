"""Test unified quantile functionality (single and batch)."""
import unittest
import numpy as np
from ddsketchy import DDSketch


class TestQuantileUnified(unittest.TestCase):
    """Test the unified quantile() method that handles both single and batch modes."""

    def test_quantile_single_float(self):
        """Test quantile with a single float value."""
        sketch = DDSketch()
        sketch.add_batch(range(1, 1001))
        
        # Single quantile should return a float
        result = sketch.quantile(0.5)
        self.assertIsInstance(result, float)
        
        # Should match expected median
        self.assertAlmostEqual(result, 500.0, delta=10.0)

    def test_quantile_batch_list(self):
        """Test quantile with a Python list (batch mode)."""
        sketch = DDSketch()
        sketch.add_batch(range(1, 1001))
        
        # Batch quantile should return a numpy array
        quantiles = [0.5, 0.9, 0.95, 0.99]
        results = sketch.quantile(quantiles)
        
        self.assertIsInstance(results, np.ndarray)
        self.assertEqual(len(results), 4)
        
        # Each result should be a float
        for result in results:
            self.assertIsInstance(result, (float, np.floating))
        
        # Results should be monotonically increasing
        for i in range(len(results) - 1):
            self.assertLessEqual(results[i], results[i + 1])

    def test_quantile_batch_numpy_contiguous(self):
        """Test quantile with contiguous NumPy array (zero-copy path)."""
        sketch = DDSketch()
        sketch.add_batch(range(1, 1001))
        
        # Create contiguous numpy array
        quantiles = np.array([0.25, 0.5, 0.75, 0.9, 0.95])
        results = sketch.quantile(quantiles)
        
        self.assertIsInstance(results, np.ndarray)
        self.assertEqual(len(results), 5)
        
        # Results should be monotonically increasing
        for i in range(len(results) - 1):
            self.assertLessEqual(results[i], results[i + 1])

    def test_quantile_batch_numpy_non_contiguous(self):
        """Test quantile with non-contiguous NumPy array."""
        sketch = DDSketch()
        sketch.add_batch(range(1, 1001))
        
        # Create non-contiguous array (every other element)
        all_quantiles = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        non_contiguous = all_quantiles[::2]  # [0.1, 0.3, 0.5, 0.7, 0.9]
        
        results = sketch.quantile(non_contiguous)
        self.assertIsInstance(results, np.ndarray)
        self.assertEqual(len(results), 5)
        
        # Results should be monotonically increasing
        for i in range(len(results) - 1):
            self.assertLessEqual(results[i], results[i + 1])

    def test_quantile_accuracy_single(self):
        """Test that single quantile results are accurate."""
        alpha = 0.01
        sketch = DDSketch(alpha=alpha)
        
        # Add known values
        values = list(range(1, 10001))  # 1 to 10000
        sketch.add_batch(values)
        
        # Test single quantile accuracy
        for q in [0.5, 0.9, 0.95, 0.99]:
            estimated = sketch.quantile(q)
            actual = values[int(q * (len(values) - 1))]
            relative_error = abs(estimated - actual) / actual
            self.assertLessEqual(
                relative_error,
                alpha * 2,
                f"Relative error {relative_error} exceeds tolerance at quantile {q}",
            )

    def test_quantile_accuracy_batch(self):
        """Test that batch quantile results are accurate."""
        alpha = 0.01
        sketch = DDSketch(alpha=alpha)
        
        # Add known values
        values = list(range(1, 10001))  # 1 to 10000
        sketch.add_batch(values)
        
        # Test batch quantile accuracy
        quantiles = [0.5, 0.9, 0.95, 0.99]
        results = sketch.quantile(quantiles)
        
        # Check accuracy for each quantile
        for q, estimated in zip(quantiles, results):
            actual = values[int(q * (len(values) - 1))]
            relative_error = abs(estimated - actual) / actual
            self.assertLessEqual(
                relative_error,
                alpha * 2,
                f"Relative error {relative_error} exceeds tolerance at quantile {q}",
            )

    def test_quantile_empty_sketch_single(self):
        """Test single quantile on empty sketch returns 0.0."""
        sketch = DDSketch()
        result = sketch.quantile(0.5)
        self.assertEqual(result, 0.0)

    def test_quantile_empty_sketch_batch(self):
        """Test batch quantile on empty sketch returns [0.0, ...]."""
        sketch = DDSketch()
        quantiles = [0.25, 0.5, 0.75]
        results = sketch.quantile(quantiles)
        
        self.assertIsInstance(results, np.ndarray)
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertEqual(result, 0.0)

    def test_quantile_invalid_single(self):
        """Test that invalid single quantile raises ValueError."""
        sketch = DDSketch()
        sketch.add_batch([1.0, 2.0, 3.0])
        
        # Quantile > 1.0
        with self.assertRaises(ValueError):
            sketch.quantile(1.5)
        
        # Quantile < 0.0
        with self.assertRaises(ValueError):
            sketch.quantile(-0.1)

    def test_quantile_invalid_batch(self):
        """Test that invalid batch quantiles raise ValueError."""
        sketch = DDSketch()
        sketch.add_batch([1.0, 2.0, 3.0])
        
        # Quantile > 1.0 in list
        with self.assertRaises(ValueError):
            sketch.quantile([0.5, 1.5])
        
        # Quantile < 0.0 in list
        with self.assertRaises(ValueError):
            sketch.quantile([0.5, -0.1])

    def test_quantile_consistency_single_vs_batch(self):
        """Test that single and batch modes give same results."""
        sketch = DDSketch()
        sketch.add_batch([float(i) for i in range(1, 101)])
        
        test_quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        
        # Get results in batch mode
        batch_results = sketch.quantile(test_quantiles)
        
        # Get results in single mode
        single_results = [sketch.quantile(q) for q in test_quantiles]
        
        # Compare
        for batch_r, single_r in zip(batch_results, single_results):
            self.assertAlmostEqual(
                batch_r,
                single_r,
                places=10,
                msg=f"Batch result {batch_r} != single result {single_r}",
            )

    def test_quantile_single_quantile_list(self):
        """Test quantile with a single-element list."""
        sketch = DDSketch()
        sketch.add_batch([1.0, 2.0, 3.0, 4.0, 5.0])
        
        # Single element list should return a numpy array
        results = sketch.quantile([0.5])
        self.assertIsInstance(results, np.ndarray)
        self.assertEqual(len(results), 1)
        
        # Should match individual quantile call
        self.assertAlmostEqual(results[0], sketch.quantile(0.5), places=10)

    def test_quantile_edge_quantiles_batch(self):
        """Test batch quantile with edge cases (0.0 and 1.0)."""
        sketch = DDSketch()
        sketch.add_batch([1.0, 2.0, 3.0, 4.0, 5.0])
        
        quantiles = [0.0, 0.5, 1.0]
        results = sketch.quantile(quantiles)
        
        self.assertEqual(len(results), 3)
        
        # Results should be monotonically increasing
        self.assertLessEqual(results[0], results[1])
        self.assertLessEqual(results[1], results[2])

    def test_quantile_tuple_input(self):
        """Test quantile with tuple input."""
        sketch = DDSketch()
        sketch.add_batch(range(1, 101))
        
        quantiles = (0.25, 0.5, 0.75)
        results = sketch.quantile(quantiles)
        
        self.assertIsInstance(results, np.ndarray)
        self.assertEqual(len(results), 3)

    def test_quantile_large_batch(self):
        """Test quantile with many quantiles."""
        sketch = DDSketch()
        sketch.add_batch(range(1, 10001))
        
        # Test with 100 quantiles
        quantiles = np.linspace(0.01, 0.99, 100)
        results = sketch.quantile(quantiles)
        
        self.assertEqual(len(results), 100)
        
        # All results should be finite
        for result in results:
            self.assertTrue(np.isfinite(result))
        
        # Results should be monotonically increasing
        for i in range(len(results) - 1):
            self.assertLessEqual(results[i], results[i + 1])

    def test_quantile_return_type_distinction(self):
        """Test that single float returns float, iterable returns ndarray."""
        sketch = DDSketch()
        sketch.add_batch([1.0, 2.0, 3.0])
        
        # Single float should return float
        single_result = sketch.quantile(0.5)
        self.assertIsInstance(single_result, float)
        
        # List should return numpy array
        batch_result = sketch.quantile([0.5])
        self.assertIsInstance(batch_result, np.ndarray)
        
        # NumPy array with single element returns numpy array
        numpy_single = sketch.quantile(np.array([0.5]))
        self.assertIsInstance(numpy_single, np.ndarray)
        self.assertEqual(len(numpy_single), 1)
        
        # NumPy array with multiple elements returns numpy array
        numpy_multi = sketch.quantile(np.array([0.3, 0.5, 0.7]))
        self.assertIsInstance(numpy_multi, np.ndarray)
        self.assertEqual(len(numpy_multi), 3)


if __name__ == "__main__":
    unittest.main()
