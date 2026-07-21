use rust_rf::{Frequency, Network, NetworkSet};

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
#[test]
fn core_build_does_not_enable_optional_features() {
    assert!(!cfg!(feature = "dataframe"));
    assert!(!cfg!(feature = "plot"));
    assert!(!cfg!(feature = "visa"));
    assert!(!cfg!(feature = "xlsx"));
}
