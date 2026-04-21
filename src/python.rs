use crate::ddsketchy::{DDSketch as DDSketchInner, DDSketchError};
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyType};

#[pyclass(module = "ddsketchy")]
/// A DDSketch for computing quantile estimates on streaming data.
///
/// DDSketch is a fully-mergeable quantile sketch with relative-error guarantees.
/// It provides accurate quantile estimates while using minimal memory.
pub struct DDSketch {
    inner: DDSketchInner,
}

impl From<DDSketchError> for PyErr {
    fn from(err: DDSketchError) -> Self {
        PyValueError::new_err(err.to_string())
    }
}

#[pymethods]
impl DDSketch {
    #[new]
    #[pyo3(signature = (alpha=0.01))]
    /// Create a new DDSketch instance.
    ///
    /// # Arguments
    ///
    /// * `alpha` - The relative accuracy guarantee for quantile estimates.
    ///   Must be in the range (0, 1). Smaller values provide more accurate
    ///   estimates but use more memory. Default is 0.01 (1% relative error).
    ///
    /// # Returns
    ///
    /// A new DDSketch instance configured with the specified accuracy.
    ///
    /// # Raises
    ///
    /// * `ValueError` - If alpha is not in the valid range (0, 1).
    fn new(alpha: f64) -> PyResult<Self> {
        let inner = DDSketchInner::new(alpha).map_err(PyErr::from)?;
        Ok(Self { inner })
    }

    /// Add a single value to the sketch.
    ///
    /// # Arguments
    ///
    /// * `value` - The value to add to the sketch. Can be any finite f64 value.
    fn add(&mut self, value: f64) {
        self.inner.add(value);
    }

    /// Add multiple values to the sketch in batch.
    ///
    /// This method accepts any Python iterable, including:
    /// - Python lists and tuples
    /// - NumPy arrays (zero-copy for contiguous arrays)
    /// - Generators and other iterators
    ///
    /// For best performance with large datasets, use NumPy arrays.
    /// Contiguous NumPy arrays use true zero-copy access, avoiding any
    /// data copying or intermediate allocations.
    ///
    /// # Arguments
    ///
    /// * `values` - A Python iterable of f64 values to add to the sketch.
    ///
    /// # Raises
    ///
    /// * `ValueError` - If the values cannot be iterated or converted to f64.
    ///
    /// # Example
    ///
    /// ```python
    /// sketch = DDSketch()
    ///
    /// # Using a Python list
    /// sketch.add_batch([1.0, 2.0, 3.0])
    ///
    /// # Using NumPy array (zero-copy, faster)
    /// import numpy as np
    /// sketch.add_batch(np.array([1.0, 2.0, 3.0]))
    /// ```
    fn add_batch(&mut self, _py: Python<'_>, values: &Bound<'_, PyAny>) -> PyResult<()> {
        // Fast path: Try to extract as numpy array (zero-copy)
        if let Ok(arr) = values.extract::<PyReadonlyArray1<f64>>() {
            let view = arr.as_array();

            // Check if we can get a contiguous slice
            if let Some(slice) = view.as_slice() {
                // True zero-copy: pass slice directly, no copying at all
                self.inner.add_slice(slice);
            } else {
                // Non-contiguous array: iterate element-by-element
                // No intermediate Vec allocation
                for &value in view.iter() {
                    self.inner.add(value);
                }
            }
            return Ok(());
        }

        // Fallback path: Extract as Vec<f64> (requires copying from Python list)
        let vec: Vec<f64> = values.extract()?;
        self.inner.add_batch(vec);
        Ok(())
    }

    /// Estimate the value(s) at the given quantile(s).
    ///
    /// This method accepts either a single quantile value or a collection of
    /// quantile values, and returns the corresponding estimated value(s).
    ///
    /// # Single Quantile
    ///
    /// When called with a single float, returns a single float result:
    ///
    /// ```python
    /// sketch = DDSketch()
    /// sketch.add_batch(range(1, 1001))
    /// median = sketch.quantile(0.5)  # Returns: float
    /// ```
    ///
    /// # Batch Quantiles
    ///
    /// When called with an iterable (list, tuple, or NumPy array), returns a
    /// list of results:
    ///
    /// ```python
    /// # Using a Python list
    /// results = sketch.quantile([0.5, 0.9, 0.95, 0.99])  # Returns: List[float]
    ///
    /// # Using NumPy array (zero-copy, faster)
    /// import numpy as np
    /// results = sketch.quantile(np.array([0.5, 0.9, 0.95, 0.99]))  # Returns: List[float]
    /// ```
    ///
    /// For batch mode, this method accepts any Python iterable, including:
    /// - Python lists and tuples
    /// - NumPy arrays (zero-copy for contiguous arrays)
    /// - Generators and other iterators
    ///
    /// For best performance with large datasets, use NumPy arrays.
    /// Contiguous NumPy arrays use true zero-copy access, avoiding any
    /// data copying or intermediate allocations.
    ///
    /// # Arguments
    ///
    /// * `q` - Either a single quantile value (float in [0, 1]) or an iterable
    ///   of quantile values (each in [0, 1]).
    ///
    /// # Returns
    ///
    /// - If `q` is a single float: returns the estimated value (float)
    /// - If `q` is an iterable: returns a list of estimated values
    ///
    /// # Raises
    ///
    /// * `ValueError` - If any quantile is not in [0, 1] or if values cannot be
    ///   iterated or converted to f64.
    fn quantile<'py>(&self, py: Python<'py>, q: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        // Batch mode: try to extract as numpy array (zero-copy) first
        if let Ok(arr) = q.extract::<PyReadonlyArray1<f64>>() {
            let view = arr.as_array();

            // Check if we can get a contiguous slice
            return if let Some(slice) = view.as_slice() {
                // True zero-copy: pass slice directly, no copying at all
                let results = self.inner.quantile_batch(slice).map_err(PyErr::from)?;
                Ok(results.into_pyobject(py)?.into_any())
            } else {
                // Non-contiguous array: iterate directly without intermediate Vec
                let results = self
                    .inner
                    .quantile_batch(view.iter().map(|&x| x))
                    .map_err(PyErr::from)?;
                Ok(results.into_pyobject(py)?.into_any())
            }
        } else if let Ok(q_value) = q.extract::<f64>() {
            // Try to extract as a single f64 first
            // Single quantile mode
            let result = self.inner.quantile(q_value).map_err(PyErr::from)?;
            return Ok(result.into_pyobject(py)?.into_any());
        }

        // Fallback path: Extract as Vec<f64> (requires copying from Python list)
        let quantiles_vec: Vec<f64> = q.extract()?;
        let results = self.inner.quantile_batch(&quantiles_vec).map_err(PyErr::from)?;
        Ok(results.into_pyobject(py)?.into_any())
    }

    /// Merge another DDSketch into this one.
    ///
    /// After merging, this sketch will contain all values from both sketches.
    /// Both sketches must have the same alpha parameter.
    ///
    /// # Arguments
    ///
    /// * `other` - Another DDSketch instance to merge into this one.
    ///
    /// # Raises
    ///
    /// * `ValueError` - If the sketches have different alpha values.
    fn merge(&mut self, other: &DDSketch) -> PyResult<()> {
        self.inner.merge(&other.inner).map_err(PyErr::from)
    }

    /// The total number of values added to the sketch.
    #[getter]
    fn count(&self) -> u64 {
        self.inner.count()
    }

    /// The sum of all values added to the sketch.
    #[getter]
    fn sum(&self) -> f64 {
        self.inner.sum()
    }

    /// The arithmetic mean of all values added to the sketch.
    ///
    /// Returns 0.0 if the sketch is empty.
    #[getter]
    fn mean(&self) -> f64 {
        self.inner.mean()
    }

    /// The minimum value added to the sketch.
    ///
    /// Returns f64::INFINITY if the sketch is empty.
    #[getter]
    fn min(&self) -> f64 {
        self.inner.min()
    }

    /// The maximum value added to the sketch.
    ///
    /// Returns -f64::INFINITY if the sketch is empty.
    #[getter]
    fn max(&self) -> f64 {
        self.inner.max()
    }

    /// The relative accuracy parameter (alpha) of the sketch.
    #[getter]
    fn alpha(&self) -> f64 {
        self.inner.alpha()
    }

    /// Check if the sketch is empty (contains no values).
    ///
    /// # Returns
    ///
    /// True if no values have been added to the sketch, False otherwise.
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Clear all values from the sketch, resetting it to an empty state.
    ///
    /// The sketch retains its alpha parameter configuration.
    fn clear(&mut self) {
        self.inner.clear();
    }

    /// Compute commonly used percentiles (P50, P90, P95, P99).
    ///
    /// # Returns
    ///
    /// A tuple of (p50, p90, p95, p99) if the sketch is not empty, or None if empty.
    /// - p50: 50th percentile (median)
    /// - p90: 90th percentile
    /// - p95: 95th percentile
    /// - p99: 99th percentile
    fn percentiles(&self) -> Option<(f64, f64, f64, f64)> {
        self.inner.percentiles()
    }

    /// Return the number of unique bins in the sketch.
    ///
    /// This is used by Python's len() function.
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Return a string representation of the DDSketch.
    ///
    /// Used by Python's repr() function.
    fn __repr__(&self) -> String {
        format!("{}", self.inner)
    }

    /// Return a string representation of the DDSketch.
    ///
    /// Used by Python's str() function.
    fn __str__(&self) -> String {
        format!("{}", self.inner)
    }

    /// Get the estimated serialized size in bytes.
    ///
    /// This returns an upper bound on the size when serialized using bincode.
    /// The estimate is guaranteed to be >= actual serialized size, making it
    /// suitable for pre-allocating buffers efficiently.
    ///
    /// # Returns
    ///
    /// The estimated size in bytes.
    ///
    /// # Example
    ///
    /// ```python
    /// sketch = DDSketch()
    /// sketch.add_batch(range(1000))
    /// estimated = sketch.estimated_serialized_size()
    /// actual = len(sketch.dumps())
    /// assert estimated >= actual
    /// ```
    fn estimated_serialized_size(&self) -> usize {
        self.inner.estimated_serialized_size()
    }

    /// Serialize the DDSketch to bytes using bincode format.
    ///
    /// This method uses zero-copy serialization directly into Python's memory buffer,
    /// eliminating intermediate allocations for better performance.
    ///
    /// # Returns
    ///
    /// A bytes object containing the serialized DDSketch data.
    ///
    /// # Raises
    ///
    /// * `ValueError` - If serialization fails.
    ///
    /// # Example
    ///
    /// ```python
    /// sketch = DDSketch()
    /// sketch.add(1.0)
    /// data = sketch.dumps()
    /// # Later, deserialize with DDSketch.loads(data)
    /// ```
    fn dumps(&self, py: Python<'_>) -> PyResult<Py<PyBytes>> {
        let estimated_size = self.inner.estimated_serialized_size();

        let py_bytes = PyBytes::new_with_writer(py, estimated_size, |writer| {
            bincode::serialize_into(writer, &self.inner)
                .map_err(|e| PyValueError::new_err(format!("Serialization failed: {}", e)))
        })?;

        Ok(py_bytes.into())
    }

    /// Serialize the DDSketch and write to a file-like object.
    ///
    /// # Arguments
    ///
    /// * `fp` - A file-like object with a write() method that accepts bytes.
    ///
    /// # Raises
    ///
    /// * `ValueError` - If serialization fails or if the write is incomplete.
    ///
    /// # Example
    ///
    /// ```python
    /// sketch = DDSketch()
    /// sketch.add(1.0)
    /// with open('sketch.bin', 'wb') as f:
    ///     sketch.dump(f)
    /// ```
    fn dump(&self, fp: &Bound<'_, PyAny>) -> PyResult<()> {
        let bytes = bincode::serialize(&self.inner)
            .map_err(|e| PyValueError::new_err(format!("Serialization failed: {}", e)))?;

        // Call the write method on the Python object
        let result = fp.call_method1("write", (bytes.as_slice(),))?;

        // Optionally check that all bytes were written
        let bytes_written: usize = result.extract()?;
        if bytes_written != bytes.len() {
            return Err(PyValueError::new_err(format!(
                "Incomplete write: expected {} bytes, wrote {} bytes",
                bytes.len(),
                bytes_written
            )));
        }

        Ok(())
    }

    /// Deserialize a DDSketch from bytes.
    ///
    /// # Arguments
    ///
    /// * `data` - A bytes object containing serialized DDSketch data.
    ///
    /// # Returns
    ///
    /// A new DDSketch instance with the deserialized data.
    ///
    /// # Raises
    ///
    /// * `ValueError` - If deserialization fails or the data is invalid.
    ///
    /// # Example
    ///
    /// ```python
    /// sketch = DDSketch()
    /// sketch.add(1.0)
    /// data = sketch.dumps()
    /// restored = DDSketch.loads(data)
    /// ```
    #[classmethod]
    fn loads(_cls: &Bound<'_, PyType>, data: &[u8]) -> PyResult<Self> {
        let inner: DDSketchInner = bincode::deserialize(data)
            .map_err(|e| PyValueError::new_err(format!("Deserialization failed: {}", e)))?;
        Ok(Self { inner })
    }

    /// Deserialize a DDSketch from a file-like object.
    ///
    /// # Arguments
    ///
    /// * `fp` - A file-like object with a read() method that returns bytes.
    ///
    /// # Returns
    ///
    /// A new DDSketch instance with the deserialized data.
    ///
    /// # Raises
    ///
    /// * `ValueError` - If deserialization fails or the data is invalid.
    ///
    /// # Example
    ///
    /// ```python
    /// with open('sketch.bin', 'rb') as f:
    ///     sketch = DDSketch.load(f)
    /// ```
    #[classmethod]
    fn load(_cls: &Bound<'_, PyType>, fp: &Bound<'_, PyAny>) -> PyResult<Self> {
        // Read all data from the file-like object
        // We'll use read() with no arguments to read all bytes
        let bytes_obj = fp.call_method0("read")?;
        let bytes: &[u8] = bytes_obj.extract()?;

        let inner: DDSketchInner = bincode::deserialize(bytes)
            .map_err(|e| PyValueError::new_err(format!("Deserialization failed: {}", e)))?;
        Ok(Self { inner })
    }

    fn __getstate__(&self) -> PyResult<Vec<u8>> {
        // Serialize Rust struct to bytes using bincode
        Ok(bincode::serialize(&self.inner)
            .map_err(|e| PyValueError::new_err(format!("Serialization failed: {}", e)))?)
    }

    fn __setstate__(&mut self, state: Vec<u8>) -> PyResult<()> {
        // Deserialize bytes back into the current struct
        let inner: DDSketchInner = bincode::deserialize(&state)
            .map_err(|e| PyValueError::new_err(format!("Deserialization failed: {}", e)))?;
        self.inner = inner;
        Ok(())
    }
}

#[pymodule]
fn ddsketchy(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DDSketch>()?;
    Ok(())
}
