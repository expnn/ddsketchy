#![forbid(unsafe_code)]

mod ddsketchy;
mod mapping;
mod store;

#[cfg(feature = "python")]
mod python;

#[cfg(test)]
mod ddsketchy_test;

#[cfg(test)]
mod datadog_reference_tests;

#[cfg(feature = "serde")]
#[cfg(test)]
mod serde_tests;

#[cfg(feature = "serde")]
#[cfg(test)]
mod size_estimation_tests;

pub use ddsketchy::{DDSketch, DDSketchBuilder, DDSketchError};
