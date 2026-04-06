use crate::DDSketch;

/// Helper function to verify bincode serialization size estimation.
///
/// This test validates the estimation logic that will be used by
/// estimated_serialized_size() when the python feature is enabled.
/// We test with serde only (no python) to avoid pyo3 linking issues.
fn verify_estimation(sketch: &DDSketch, description: &str) {
    // Manually calculate the expected estimate using the same formula
    // as estimated_serialized_size() (which requires python feature)
    let positive_bins = sketch.positive_store_bins();
    let negative_bins = sketch.negative_store_bins();
    let estimated = 160 + (positive_bins + negative_bins) * 8;

    let actual = bincode::serialize(sketch).unwrap().len();

    assert!(
        estimated >= actual,
        "{}: estimated ({}) should be >= actual ({})",
        description,
        estimated,
        actual
    );
}

#[test]
fn test_estimated_serialized_size_empty() {
    let sketch = DDSketch::new(0.01).unwrap();
    verify_estimation(&sketch, "Empty sketch");
}

#[test]
fn test_estimated_serialized_size_small() {
    let mut sketch = DDSketch::new(0.01).unwrap();
    sketch.add_batch((1..=10).map(|x| x as f64));
    verify_estimation(&sketch, "Small sketch (10 values)");
}

#[test]
fn test_estimated_serialized_size_medium() {
    let mut sketch = DDSketch::new(0.01).unwrap();
    sketch.add_batch((1..=1000).map(|x| x as f64));
    verify_estimation(&sketch, "Medium sketch (1000 values)");
}

#[test]
fn test_estimated_serialized_size_large() {
    let mut sketch = DDSketch::new(0.01).unwrap();
    sketch.add_batch((1..=10000).map(|x| x as f64));
    verify_estimation(&sketch, "Large sketch (10000 values)");
}

#[test]
fn test_estimated_serialized_size_with_negatives() {
    let mut sketch = DDSketch::new(0.01).unwrap();
    for i in -100..=100 {
        sketch.add(i as f64);
    }
    verify_estimation(&sketch, "Sketch with negatives (201 values)");
}

#[test]
fn test_estimated_serialized_size_wide_range() {
    let mut sketch = DDSketch::new(0.01).unwrap();
    for exp in -10..=10 {
        sketch.add(10f64.powi(exp));
    }
    verify_estimation(&sketch, "Wide range sketch (21 values)");
}

#[test]
fn test_estimated_serialized_size_different_alpha() {
    for &alpha in &[0.1, 0.05, 0.01, 0.005, 0.001] {
        let mut sketch = DDSketch::new(alpha).unwrap();
        sketch.add_batch((1..=1000).map(|x| x as f64));
        verify_estimation(&sketch, &format!("Alpha={}", alpha));
    }
}

#[test]
fn test_estimated_serialized_size_high_precision() {
    let mut sketch = DDSketch::new(0.0001).unwrap();
    sketch.add_batch((1..=10000).map(|x| x as f64));
    verify_estimation(
        &sketch,
        "High precision sketch (alpha=0.0001, 10000 values)",
    );
}

#[test]
fn test_estimated_serialized_size_mixed_values() {
    let mut sketch = DDSketch::new(0.01).unwrap();
    sketch.add(0.0);
    sketch.add(0.0);
    for i in 1..=50 {
        sketch.add(i as f64);
        sketch.add(-(i as f64));
    }
    verify_estimation(&sketch, "Mixed values sketch (positive, negative, zero)");
}
