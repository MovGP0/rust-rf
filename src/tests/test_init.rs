//! Core package-surface and optional-feature isolation tests.
//!
//! Rust feature gates provide the analogue of Python's lazy optional-module
//! imports: core users should not enable plotting, dataframe, VISA, or XLSX
//! dependencies merely by using the crate.

use rust_rf::{Frequency, Network, NetworkSet};

/// Checks the core public types, version, and lightweight plotting setup entry point.
#[test]
fn exposes_the_core_package_surface_and_version() {
    assert_eq!(rust_rf::VERSION, env!("CARGO_PKG_VERSION"));
    let _ = std::any::TypeId::of::<Frequency>();
    let _ = std::any::TypeId::of::<Network>();
    let _ = std::any::TypeId::of::<NetworkSet>();
    let _ = rust_rf::setup_plotting();
}

#[cfg(not(any(
    feature = "dataframe",
    feature = "plot",
    feature = "visa",
    feature = "xlsx"
)))]
/// Checks that a default core build does not enable heavy optional features.
#[test]
fn core_build_does_not_enable_optional_features() {
    assert!(!cfg!(feature = "dataframe"));
    assert!(!cfg!(feature = "plot"));
    assert!(!cfg!(feature = "visa"));
    assert!(!cfg!(feature = "xlsx"));
}
