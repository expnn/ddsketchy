from __future__ import annotations

from typing import Any, Iterable, Optional, TYPE_CHECKING, Union

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from _typeshed import ReadableBuffer, SupportsWrite


class DDSketch:
    """A DDSketch for computing quantile estimates on streaming data.

    DDSketch is a fully-mergeable quantile sketch with relative-error guarantees.
    It provides accurate quantile estimates while using minimal memory.
    """

    def __init__(self, alpha: float = 0.01) -> None:
        """Create a new DDSketch instance.

        Args:
            alpha: The relative accuracy guarantee for quantile estimates.
                Must be in the range (0, 1). Smaller values provide more accurate
                estimates but use more memory. Default is 0.01 (1% relative error).

        Raises:
            ValueError: If alpha is not in the valid range (0, 1).
        """
        ...

    def add(self, value: float) -> None:
        """Add a single value to the sketch.

        Args:
            value: The value to add to the sketch. Can be any finite f64 value.
        """
        ...

    def add_batch(self, values: Union[Iterable[float], NDArray[np.float64]]) -> None:
        """Add multiple values to the sketch in batch.

        This method accepts any Python iterable, including:
        - Python lists and tuples
        - NumPy arrays (zero-copy for contiguous arrays)
        - Generators and other iterators

        For best performance with large datasets, use NumPy arrays.
        Contiguous NumPy arrays use true zero-copy access, avoiding any
        data copying or intermediate allocations.

        Args:
            values: A Python iterable of f64 values to add to the sketch.

        Raises:
            ValueError: If the values cannot be iterated or converted to f64.

        Example::

            sketch = DDSketch()

            # Using a Python list
            sketch.add_batch([1.0, 2.0, 3.0])

            # Using NumPy array (zero-copy, faster)
            import numpy as np
            sketch.add_batch(np.array([1.0, 2.0, 3.0]))
        """
        ...

    def quantile(self, q: Union[float, Iterable[float], NDArray[np.float64]]) -> NDArray[np.float64]:
        """Estimate the value(s) at the given quantile(s).

        When called with a single float, returns a single float result::

            sketch = DDSketch()
            sketch.add_batch(range(1, 1001))
            median = sketch.quantile(0.5)  # Returns: float

        When called with an iterable (list, tuple, or NumPy array), returns a
        list of results::

            # Using a Python list
            results = sketch.quantile([0.5, 0.9, 0.95, 0.99])  # Returns: List[float]

            # Using NumPy array (zero-copy, faster)
            import numpy as np
            results = sketch.quantile(np.array([0.5, 0.9, 0.95, 0.99]))  # Returns: List[float]

        Args:
            q: Either a single quantile value (float in [0, 1]) or an iterable
                of quantile values (each in [0, 1]).

        Returns:
            If ``q`` is a single float: the estimated value (float).
            If ``q`` is an iterable: a list of estimated values.

        Raises:
            ValueError: If any quantile is not in [0, 1] or if values cannot be
                iterated or converted to f64.
        """
        ...

    def merge(self, other: DDSketch) -> None:
        """Merge another DDSketch into this one.

        After merging, this sketch will contain all values from both sketches.
        Both sketches must have the same alpha parameter.

        Args:
            other: Another DDSketch instance to merge into this one.

        Raises:
            ValueError: If the sketches have different alpha values.
        """
        ...

    @property
    def count(self) -> int: ...
    """The total number of values added to the sketch."""

    @property
    def sum(self) -> float: ...
    """The sum of all values added to the sketch."""

    @property
    def mean(self) -> float: ...
    """The arithmetic mean of all values added to the sketch.
    Returns 0.0 if the sketch is empty."""

    @property
    def min(self) -> float: ...
    """The minimum value added to the sketch.
    Returns float('inf') if the sketch is empty."""

    @property
    def max(self) -> float: ...
    """The maximum value added to the sketch.
    Returns float('-inf') if the sketch is empty."""

    @property
    def alpha(self) -> float: ...
    """The relative accuracy parameter (alpha) of the sketch."""

    def is_empty(self) -> bool:
        """Check if the sketch is empty (contains no values).

        Returns:
            True if no values have been added to the sketch, False otherwise.
        """
        ...

    def clear(self) -> None:
        """Clear all values from the sketch, resetting it to an empty state.
        The sketch retains its alpha parameter configuration."""
        ...

    def percentiles(self) -> Optional[tuple[float, float, float, float]]:
        """Compute commonly used percentiles (P50, P90, P95, P99).

        Returns:
            A tuple of (p50, p90, p95, p99) if the sketch is not empty, or None if empty.
        """
        ...

    def collect_raw_statistics(
        self, positive: bool = True
    ) -> tuple[NDArray[np.float64], NDArray[np.uint64], float]:
        """Collect raw statistics for the given store (positive or negative).

        Args:
            positive: If True, collect statistics for positive values (default).
                If False, collect statistics for negative values.

        Returns:
            A tuple containing:
            - An array of left edges of bins
            - An array of counts for each bin
            - The alpha parameter (relative error) of the sketch
        """
        ...

    def __len__(self) -> int:
        """Return the number of unique bins in the sketch."""
        ...

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

    def estimated_serialized_size(self) -> int:
        """Get the estimated serialized size in bytes.

        This returns an upper bound on the size when serialized using bincode.
        The estimate is guaranteed to be >= actual serialized size, making it
        suitable for pre-allocating buffers efficiently.

        Returns:
            The estimated size in bytes.
        """
        ...

    def dumps(self) -> bytes:
        """Serialize the DDSketch to bytes using bincode format.

        This method uses zero-copy serialization directly into Python's memory buffer,
        eliminating intermediate allocations for better performance.

        Returns:
            A bytes object containing the serialized DDSketch data.

        Raises:
            ValueError: If serialization fails.
        """
        ...

    def dump(self, fp: SupportsWrite[bytes]) -> None:
        """Serialize the DDSketch and write to a file-like object.

        Args:
            fp: A file-like object with a write() method that accepts bytes.

        Raises:
            ValueError: If serialization fails or if the write is incomplete.
        """
        ...

    @classmethod
    def loads(cls, data: ReadableBuffer) -> DDSketch:
        """Deserialize a DDSketch from bytes.

        Args:
            data: A bytes object containing serialized DDSketch data.

        Returns:
            A new DDSketch instance with the deserialized data.

        Raises:
            ValueError: If deserialization fails or the data is invalid.
        """
        ...

    @classmethod
    def load(cls, fp: Any) -> DDSketch:
        """Deserialize a DDSketch from a file-like object.

        Args:
            fp: A file-like object with a read() method that returns bytes.

        Returns:
            A new DDSketch instance with the deserialized data.

        Raises:
            ValueError: If deserialization fails or the data is invalid.
        """
        ...

    def __getstate__(self) -> bytes: ...
    def __setstate__(self, state: bytes) -> None: ...
