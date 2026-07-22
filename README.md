# rust-rf

> Everything that can be rewritten in Rust will eventually be rewritten in Rust

A Rust port of [scikit-rf](https://github.com/scikit-rf/scikit-rf) for RF and microwave engineering.

The main intention behind this port is that Rust provides strict memory safety through its borrow checker without needing a garbage collector, giving it the type safety and (almost) the raw speed of C/C++.

Optional integrations are isolated behind Cargo features:

- `dataframe`: Polars adapters for pandas-facing APIs.
- `plot`: Plotters-based visualization.
- `visa`: transport-independent VNA APIs. Native `visa-rs` sessions support
  Windows x86/x64 (`visa32`/`visa64`, optionally via `LIB_VISA_PATH`) when
  `RUST_RF_NATIVE_VISA=1`; macOS and Linux are not supported for native VISA.
- `xlsx`: Excel workbook writing.
- `full`: all optional integrations.

See [DEPENDENCIES.md](DEPENDENCIES.md) for the Python-to-Rust dependency analysis.
