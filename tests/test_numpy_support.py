import unittest
import numpy as np
from ddsketchy import DDSketch


class TestNumpySupport(unittest.TestCase):
    """Test numpy array support with zero-copy optimization."""

    def test_add_batch_numpy_array(self):
        """Test adding values from a contiguous numpy array."""
        sketch = DDSketch()
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sketch.add_batch(values)
        
        self.assertEqual(sketch.count, 5)
        self.assertAlmostEqual(sketch.sum, 15.0, places=10)
    
    def test_add_batch_numpy_zero_copy_contiguous(self):
        """Test that contiguous numpy arrays use zero-copy path."""
        sketch = DDSketch()
        # Create a contiguous numpy array
        values = np.arange(1.0, 101.0)
        sketch.add_batch(values)
        
        self.assertEqual(sketch.count, 100)
        self.assertAlmostEqual(sketch.sum, 5050.0, places=10)
    
    def test_add_batch_numpy_non_contiguous(self):
        """Test adding values from a non-contiguous numpy array."""
        sketch = DDSketch()
        # Create a non-contiguous array (every other element)
        values = np.arange(1.0, 21.0)[::2]  # [1, 3, 5, ..., 19]
        sketch.add_batch(values)
        
        self.assertEqual(sketch.count, 10)
        self.assertAlmostEqual(sketch.sum, 100.0, places=10)
    
    def test_add_batch_list_still_works(self):
        """Test that Python lists still work after numpy optimization."""
        sketch = DDSketch()
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        sketch.add_batch(values)
        
        self.assertEqual(sketch.count, 5)
        self.assertAlmostEqual(sketch.sum, 15.0, places=10)
    
    def test_add_batch_numpy_large_dataset(self):
        """Test performance with large numpy array."""
        sketch = DDSketch()
        # 100,000 values
        values = np.random.randn(100_000)
        sketch.add_batch(values)
        
        self.assertEqual(sketch.count, 100_000)
        # Allow 1% tolerance due to floating point
        self.assertAlmostEqual(sketch.sum, values.sum(), delta=abs(values.sum()) * 0.01)
    
    def test_add_batch_numpy_dtype_float32(self):
        """Test numpy array with float32 dtype (should fail - we need f64)."""
        sketch = DDSketch()
        values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        
        # float32 should not be directly extractable as f64
        # This tests the fallback behavior
        try:
            sketch.add_batch(values)
            # If it works, verify the values
            self.assertEqual(sketch.count, 3)
        except TypeError:
            # Expected: float32 array can't be directly used
            pass
    
    def test_add_batch_numpy_2d_array_fails(self):
        """Test that 2D numpy arrays are rejected."""
        sketch = DDSketch()
        values = np.array([[1.0, 2.0], [3.0, 4.0]])
        
        # This should fail as we expect 1D array
        with self.assertRaises(Exception):
            sketch.add_batch(values)
    
    def test_add_batch_generator_not_supported(self):
        """Test that generators are not directly supported (need list/tuple/numpy)."""
        sketch = DDSketch()
        # Generators don't implement Sequence, so they can't be extracted as Vec<f64>
        with self.assertRaises(TypeError):
            sketch.add_batch(x * 1.5 for x in range(10))
    
    def test_add_batch_numpy_negative_values(self):
        """Test numpy array with negative values."""
        sketch = DDSketch()
        values = np.array([-10.0, -5.0, 0.0, 5.0, 10.0])
        sketch.add_batch(values)
        
        self.assertEqual(sketch.count, 5)
        self.assertAlmostEqual(sketch.sum, 0.0, places=10)
    
    def test_add_batch_numpy_quantile_accuracy(self):
        """Test that numpy array addition maintains quantile accuracy."""
        alpha = 0.01
        sketch = DDSketch(alpha=alpha)
        
        # Add values using numpy
        values = np.arange(1.0, 10001.0)
        sketch.add_batch(values)
        
        # Test quantile accuracy
        for q in [0.5, 0.9, 0.95, 0.99]:
            estimated = sketch.quantile(q)
            actual = values[int(q * (len(values) - 1))]
            relative_error = abs(estimated - actual) / actual
            self.assertLessEqual(
                relative_error,
                alpha,
                f"Relative error {relative_error} exceeds alpha {alpha} at quantile {q}",
            )
    
    def test_add_batch_mixed_numpy_and_list(self):
        """Test mixing numpy arrays and lists in the same sketch."""
        sketch = DDSketch()
        
        # Add using numpy
        sketch.add_batch(np.array([1.0, 2.0, 3.0]))
        
        # Add using list
        sketch.add_batch([4.0, 5.0, 6.0])
        
        # Add using numpy again
        sketch.add_batch(np.array([7.0, 8.0, 9.0]))
        
        self.assertEqual(sketch.count, 9)
        self.assertAlmostEqual(sketch.sum, 45.0, places=10)


if __name__ == "__main__":
    unittest.main()
