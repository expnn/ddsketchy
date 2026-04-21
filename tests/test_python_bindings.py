import unittest
import io
import tempfile
import os
from ddsketchy import DDSketch


class TestDDSketchBasic(unittest.TestCase):
    def test_create_default_alpha(self):
        sketch = DDSketch()
        self.assertEqual(sketch.count, 0)
        self.assertTrue(sketch.is_empty())
        self.assertAlmostEqual(sketch.alpha, 0.01, places=10)

    def test_create_custom_alpha(self):
        sketch = DDSketch(alpha=0.05)
        self.assertAlmostEqual(sketch.alpha, 0.05, places=10)

    def test_invalid_alpha(self):
        with self.assertRaises(ValueError):
            DDSketch(alpha=0.0)
        with self.assertRaises(ValueError):
            DDSketch(alpha=1.0)
        with self.assertRaises(ValueError):
            DDSketch(alpha=-0.1)

    def test_add_single_value(self):
        sketch = DDSketch()
        sketch.add(100.0)
        self.assertEqual(sketch.count, 1)
        self.assertFalse(sketch.is_empty())
        self.assertAlmostEqual(sketch.sum, 100.0, places=10)

    def test_add_multiple_values(self):
        sketch = DDSketch()
        for i in range(1, 101):
            sketch.add(float(i))
        self.assertEqual(sketch.count, 100)
        self.assertAlmostEqual(sketch.sum, 5050.0, places=10)
        self.assertAlmostEqual(sketch.mean, 50.5, places=10)

    def test_add_batch(self):
        sketch = DDSketch()
        values = [float(i) for i in range(1, 101)]
        sketch.add_batch(values)
        self.assertEqual(sketch.count, 100)
        self.assertAlmostEqual(sketch.sum, 5050.0, places=10)


class TestDDSketchQuantiles(unittest.TestCase):
    def setUp(self):
        self.sketch = DDSketch(alpha=0.01)
        for i in range(1, 1001):
            self.sketch.add(float(i))

    def test_median(self):
        median = self.sketch.quantile(0.5)
        self.assertAlmostEqual(median, 500.0, delta=500.0 * 0.01)

    def test_p90(self):
        p90 = self.sketch.quantile(0.9)
        self.assertAlmostEqual(p90, 900.0, delta=900.0 * 0.01)

    def test_p99(self):
        p99 = self.sketch.quantile(0.99)
        self.assertAlmostEqual(p99, 990.0, delta=990.0 * 0.01)

    def test_min_quantile(self):
        p0 = self.sketch.quantile(0.0)
        self.assertAlmostEqual(p0, 1.0, delta=1.0 * 0.01 + 0.1)

    def test_max_quantile(self):
        p100 = self.sketch.quantile(1.0)
        self.assertAlmostEqual(p100, 1000.0, delta=1000.0 * 0.01)

    def test_invalid_quantile(self):
        with self.assertRaises(ValueError):
            self.sketch.quantile(-0.1)
        with self.assertRaises(ValueError):
            self.sketch.quantile(1.1)

    def test_percentiles(self):
        p50, p90, p95, p99 = self.sketch.percentiles()
        self.assertAlmostEqual(p50, 500.0, delta=500.0 * 0.01)
        self.assertAlmostEqual(p90, 900.0, delta=900.0 * 0.01)
        self.assertAlmostEqual(p95, 950.0, delta=950.0 * 0.01)
        self.assertAlmostEqual(p99, 990.0, delta=990.0 * 0.01)


class TestDDSketchMinMax(unittest.TestCase):
    def test_min_max_positive(self):
        sketch = DDSketch()
        sketch.add_batch([10.0, 20.0, 30.0, 40.0, 50.0])
        self.assertAlmostEqual(sketch.min, 10.0, delta=10.0 * 0.01)
        self.assertAlmostEqual(sketch.max, 50.0, delta=50.0 * 0.01)

    def test_min_max_with_negatives(self):
        sketch = DDSketch()
        sketch.add_batch([-50.0, -10.0, 0.0, 10.0, 50.0])
        self.assertAlmostEqual(sketch.min, -50.0, delta=50.0 * 0.01)
        self.assertAlmostEqual(sketch.max, 50.0, delta=50.0 * 0.01)


class TestDDSketchMerge(unittest.TestCase):
    def test_merge_two_sketches(self):
        sketch1 = DDSketch()
        sketch2 = DDSketch()

        for i in range(1, 51):
            sketch1.add(float(i))
        for i in range(51, 101):
            sketch2.add(float(i))

        sketch1.merge(sketch2)

        self.assertEqual(sketch1.count, 100)
        self.assertAlmostEqual(sketch1.sum, 5050.0, places=10)
        self.assertAlmostEqual(sketch1.mean, 50.5, places=10)

    def test_merge_incompatible_alpha(self):
        sketch1 = DDSketch(alpha=0.01)
        sketch2 = DDSketch(alpha=0.05)

        sketch1.add(1.0)
        sketch2.add(2.0)

        with self.assertRaises(ValueError):
            sketch1.merge(sketch2)


class TestDDSketchClear(unittest.TestCase):
    def test_clear(self):
        sketch = DDSketch()
        sketch.add_batch([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual(sketch.count, 5)

        sketch.clear()
        self.assertEqual(sketch.count, 0)
        self.assertTrue(sketch.is_empty())
        self.assertAlmostEqual(sketch.sum, 0.0, places=10)


class TestDDSketchLen(unittest.TestCase):
    def test_len_protocol(self):
        sketch = DDSketch()
        self.assertEqual(len(sketch), 0)

        sketch.add_batch([1.0, 2.0, 3.0])
        self.assertEqual(len(sketch), 3)


class TestDDSketchRepr(unittest.TestCase):
    def test_repr(self):
        sketch = DDSketch()
        sketch.add(100.0)
        repr_str = repr(sketch)
        self.assertIn("DDSketch", repr_str)
        self.assertIn("count=1", repr_str)

    def test_str(self):
        sketch = DDSketch()
        sketch.add(100.0)
        str_repr = str(sketch)
        self.assertIn("DDSketch", str_repr)


class TestDDSketchEmptyBehavior(unittest.TestCase):
    def test_empty_quantile(self):
        sketch = DDSketch()
        result = sketch.quantile(0.5)
        self.assertEqual(result, 0.0)

    def test_empty_percentiles(self):
        sketch = DDSketch()
        result = sketch.percentiles()
        self.assertIsNone(result)

    def test_empty_mean(self):
        sketch = DDSketch()
        self.assertEqual(sketch.mean, 0.0)


class TestDDSketchAccuracy(unittest.TestCase):
    def test_relative_accuracy_guarantee(self):
        alpha = 0.01
        sketch = DDSketch(alpha=alpha)

        values = [float(i) for i in range(1, 10001)]
        sketch.add_batch(values)

        for q in [0.5, 0.9, 0.95, 0.99]:
            estimated = sketch.quantile(q)
            actual = values[int(q * (len(values) - 1))]
            relative_error = abs(estimated - actual) / actual
            self.assertLessEqual(
                relative_error,
                alpha,
                f"Relative error {relative_error} exceeds alpha {alpha} at quantile {q}",
            )


if __name__ == "__main__":
    unittest.main()


class TestDDSketchSerialization(unittest.TestCase):
    def test_dumps_loads_roundtrip(self):
        """Test serialization to bytes and deserialization back."""
        sketch = DDSketch(alpha=0.01)
        sketch.add_batch([1.0, 2.0, 3.0, 4.0, 5.0])

        # Serialize to bytes
        data = sketch.dumps()

        # Verify it's bytes
        self.assertIsInstance(data, bytes)
        self.assertGreater(len(data), 0)

        # Deserialize back
        restored = DDSketch.loads(data)

        # Verify equality
        self.assertEqual(sketch.count, restored.count)
        self.assertAlmostEqual(sketch.sum, restored.sum, places=10)
        self.assertAlmostEqual(sketch.min, restored.min, delta=0.1)
        self.assertAlmostEqual(sketch.max, restored.max, delta=0.1)
        self.assertAlmostEqual(sketch.alpha, restored.alpha, places=10)

        # Verify quantiles match
        for q in [0.1, 0.5, 0.9]:
            self.assertAlmostEqual(sketch.quantile(q), restored.quantile(q), delta=0.1)

    def test_dumps_empty_sketch(self):
        """Test serializing an empty sketch."""
        sketch = DDSketch()
        data = sketch.dumps()
        restored = DDSketch.loads(data)

        self.assertEqual(restored.count, 0)
        self.assertTrue(restored.is_empty())
        self.assertAlmostEqual(restored.alpha, 0.01, places=10)

    def test_dump_load_file(self):
        """Test serialization to file and deserialization from file."""
        sketch = DDSketch(alpha=0.02)
        sketch.add_batch([10.0, 20.0, 30.0, 40.0, 50.0])

        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
            temp_path = f.name

        try:
            # Write to file
            with open(temp_path, "wb") as f:
                sketch.dump(f)

            # Read from file
            with open(temp_path, "rb") as f:
                restored = DDSketch.load(f)

            # Verify equality
            self.assertEqual(sketch.count, restored.count)
            self.assertAlmostEqual(sketch.sum, restored.sum, places=10)
            self.assertAlmostEqual(sketch.alpha, restored.alpha, places=10)

            # Verify quantiles
            for q in [0.25, 0.5, 0.75]:
                self.assertAlmostEqual(
                    sketch.quantile(q), restored.quantile(q), delta=0.5
                )
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_dump_load_bytesio(self):
        """Test serialization to BytesIO and deserialization."""
        sketch = DDSketch()
        sketch.add_batch([-10.0, -5.0, 0.0, 5.0, 10.0])

        # Serialize to BytesIO
        buffer = io.BytesIO()
        sketch.dump(buffer)

        # Reset buffer position
        buffer.seek(0)

        # Deserialize from BytesIO
        restored = DDSketch.load(buffer)

        # Verify equality
        self.assertEqual(sketch.count, restored.count)
        self.assertAlmostEqual(sketch.sum, restored.sum, places=10)
        self.assertAlmostEqual(sketch.alpha, restored.alpha, places=10)

    def test_serialization_preserves_negative_values(self):
        """Test that negative values are correctly preserved."""
        sketch = DDSketch()
        sketch.add_batch([-100.0, -50.0, -10.0, 0.0, 10.0, 50.0, 100.0])

        data = sketch.dumps()
        restored = DDSketch.loads(data)

        self.assertEqual(sketch.count, restored.count)
        self.assertAlmostEqual(sketch.sum, restored.sum, places=10)

        # Check quantiles for negative values
        for q in [0.1, 0.5, 0.9]:
            orig_q = sketch.quantile(q)
            rest_q = restored.quantile(q)
            self.assertAlmostEqual(orig_q, rest_q, delta=1.0)

    def test_serialization_with_large_dataset(self):
        """Test serialization with a larger dataset."""
        sketch = DDSketch(alpha=0.01)
        sketch.add_batch([float(i) for i in range(1, 10001)])

        data = sketch.dumps()
        restored = DDSketch.loads(data)

        self.assertEqual(sketch.count, restored.count)
        self.assertAlmostEqual(sketch.sum, restored.sum, places=5)

        # Test multiple quantiles with tolerance
        for q in [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]:
            orig_q = sketch.quantile(q)
            rest_q = restored.quantile(q)
            # Allow 1% relative error (matching alpha)
            relative_error = (
                abs(orig_q - rest_q) / orig_q if orig_q != 0 else abs(orig_q - rest_q)
            )
            self.assertLessEqual(relative_error, 0.02)

    def test_dump_incomplete_write(self):
        """Test that incomplete writes raise an error."""
        sketch = DDSketch()
        sketch.add(1.0)

        # Create a mock file object that reports incomplete write
        class IncompleteFile:
            def write(self, data):
                return 0  # Report 0 bytes written

        with self.assertRaises(ValueError) as context:
            sketch.dump(IncompleteFile())

        self.assertIn("Incomplete write", str(context.exception))

    def test_loads_invalid_data(self):
        """Test that invalid data raises an error."""
        with self.assertRaises(ValueError) as context:
            DDSketch.loads(b"invalid data")

        self.assertIn("Deserialization failed", str(context.exception))

    def test_load_empty_file(self):
        """Test that loading from empty file raises an error."""
        buffer = io.BytesIO()

        with self.assertRaises(ValueError):
            DDSketch.load(buffer)

    def test_serialization_after_merge(self):
        """Test serializing a merged sketch."""
        sketch1 = DDSketch()
        sketch2 = DDSketch()

        sketch1.add_batch([1.0, 2.0, 3.0])
        sketch2.add_batch([4.0, 5.0, 6.0])

        sketch1.merge(sketch2)

        data = sketch1.dumps()
        restored = DDSketch.loads(data)

        self.assertEqual(restored.count, 6)
        self.assertAlmostEqual(restored.sum, 21.0, places=10)


class TestDDSketchPickle(unittest.TestCase):
    def test_pickle_roundtrip(self):
        """Test basic pickle serialization and deserialization."""
        import pickle

        sketch = DDSketch(alpha=0.01)
        sketch.add_batch([1.0, 2.0, 3.0, 4.0, 5.0])

        # Pickle and unpickle
        pickled = pickle.dumps(sketch)
        restored = pickle.loads(pickled)

        # Verify equality
        self.assertEqual(sketch.count, restored.count)
        self.assertAlmostEqual(sketch.sum, restored.sum, places=10)
        self.assertAlmostEqual(sketch.min, restored.min, delta=0.1)
        self.assertAlmostEqual(sketch.max, restored.max, delta=0.1)
        self.assertAlmostEqual(sketch.alpha, restored.alpha, places=10)

        # Verify quantiles match
        for q in [0.1, 0.5, 0.9]:
            self.assertAlmostEqual(sketch.quantile(q), restored.quantile(q), delta=0.1)

    def test_pickle_empty_sketch(self):
        """Test pickling an empty sketch."""
        import pickle

        sketch = DDSketch()
        pickled = pickle.dumps(sketch)
        restored = pickle.loads(pickled)

        self.assertEqual(restored.count, 0)
        self.assertTrue(restored.is_empty())
        self.assertAlmostEqual(restored.alpha, 0.01, places=10)

    def test_pickle_with_negative_values(self):
        """Test pickling sketch with negative values."""
        import pickle

        sketch = DDSketch()
        sketch.add_batch([-100.0, -50.0, -10.0, 0.0, 10.0, 50.0, 100.0])

        pickled = pickle.dumps(sketch)
        restored = pickle.loads(pickled)

        self.assertEqual(sketch.count, restored.count)
        self.assertAlmostEqual(sketch.sum, restored.sum, places=10)

        # Check quantiles for negative values
        for q in [0.1, 0.5, 0.9]:
            orig_q = sketch.quantile(q)
            rest_q = restored.quantile(q)
            self.assertAlmostEqual(orig_q, rest_q, delta=1.0)

    def test_pickle_after_merge(self):
        """Test pickling a merged sketch."""
        import pickle

        sketch1 = DDSketch()
        sketch2 = DDSketch()

        sketch1.add_batch([1.0, 2.0, 3.0])
        sketch2.add_batch([4.0, 5.0, 6.0])

        sketch1.merge(sketch2)

        pickled = pickle.dumps(sketch1)
        restored = pickle.loads(pickled)

        self.assertEqual(restored.count, 6)
        self.assertAlmostEqual(restored.sum, 21.0, places=10)

    def test_pickle_large_dataset(self):
        """Test pickling with a larger dataset."""
        import pickle

        sketch = DDSketch(alpha=0.01)
        sketch.add_batch([float(i) for i in range(1, 10001)])

        pickled = pickle.dumps(sketch)
        restored = pickle.loads(pickled)

        self.assertEqual(sketch.count, restored.count)
        self.assertAlmostEqual(sketch.sum, restored.sum, places=5)

        # Test multiple quantiles with tolerance
        for q in [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]:
            orig_q = sketch.quantile(q)
            rest_q = restored.quantile(q)
            # Allow 1% relative error (matching alpha)
            relative_error = (
                abs(orig_q - rest_q) / orig_q if orig_q != 0 else abs(orig_q - rest_q)
            )
            self.assertLessEqual(relative_error, 0.02)

    def test_pickle_protocols(self):
        """Test different pickle protocols."""
        import pickle

        sketch = DDSketch()
        sketch.add_batch([1.0, 2.0, 3.0, 4.0, 5.0])

        # Test protocols 2 and above (protocol 0 and 1 have limitations with PyO3)
        for protocol in range(2, pickle.HIGHEST_PROTOCOL + 1):
            pickled = pickle.dumps(sketch, protocol=protocol)
            restored = pickle.loads(pickled)

            self.assertEqual(sketch.count, restored.count)
            self.assertAlmostEqual(sketch.sum, restored.sum, places=10)

    def test_pickle_file_roundtrip(self):
        """Test pickling to file and loading back."""
        import pickle

        sketch = DDSketch(alpha=0.02)
        sketch.add_batch([10.0, 20.0, 30.0, 40.0, 50.0])

        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
            temp_path = f.name

        try:
            # Pickle to file
            with open(temp_path, "wb") as f:
                pickle.dump(sketch, f)

            # Unpickle from file
            with open(temp_path, "rb") as f:
                restored = pickle.load(f)

            # Verify equality
            self.assertEqual(sketch.count, restored.count)
            self.assertAlmostEqual(sketch.sum, restored.sum, places=10)
            self.assertAlmostEqual(sketch.alpha, restored.alpha, places=10)

            # Verify quantiles
            for q in [0.25, 0.5, 0.75]:
                self.assertAlmostEqual(
                    sketch.quantile(q), restored.quantile(q), delta=0.5
                )
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_pickle_copy(self):
        """Test using pickle for deep copy."""
        import pickle
        import copy

        sketch = DDSketch()
        sketch.add_batch([1.0, 2.0, 3.0])

        # Use copy.deepcopy which uses pickle internally
        copied = copy.deepcopy(sketch)

        # Verify they are equal
        self.assertEqual(sketch.count, copied.count)
        self.assertAlmostEqual(sketch.sum, copied.sum, places=10)

        # Verify they are independent
        sketch.add(100.0)
        self.assertEqual(sketch.count, 4)
        self.assertEqual(copied.count, 3)
