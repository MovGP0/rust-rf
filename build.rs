fn main() {
    println!("cargo::rustc-check-cfg=cfg(rust_rf_native_visa)");
    println!("cargo::rerun-if-env-changed=RUST_RF_NATIVE_VISA");

    if std::env::var_os("RUST_RF_NATIVE_VISA").is_some() {
        let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
        let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
        let supported = match target_os.as_str() {
            "windows" => matches!(target_arch.as_str(), "x86" | "x86_64"),
            "macos" => matches!(target_arch.as_str(), "x86_64" | "aarch64"),
            _ => false,
        };
        assert!(
            supported,
            "native VISA is supported only on Windows x86/x86_64 and macOS x86_64/aarch64 targets"
        );
        println!("cargo::rustc-cfg=rust_rf_native_visa");
    }
}
