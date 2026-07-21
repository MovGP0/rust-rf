# Python-to-Rust dependency map

This map is based on `D:\GitHub\scikit-rf\pyproject.toml` and imports under `skrf/` at upstream commit `ca243628ee5fd91ed030fec52bcec08d778a8516`.

| Python dependency | Usage in scikit-rf | Rust replacement | Cargo policy and current status |
| --- | --- | --- | --- |
| NumPy | Complex n-dimensional arrays, vectorized operations, FFTs, linear algebra, and NPZ persistence | `ndarray`, `num-complex`, `faer`, `realfft`, `rustfft`, `ndarray-npy`, and `num-traits` | Core; used by the network, calibration, media, time, math, and vector-fitting ports |
| SciPy | Interpolation, least-squares solves, integration, special functions, windows, convolution, constants, and distributions | `faer`, `libm`, `rand_distr`, the FFT crates, and crate-owned numerical implementations | Core; the algorithms required by the upstream implementation and fixtures are implemented without exposing a SciPy-shaped backend in the public API |
| pandas | DataFrame conversion and CSV/Excel adapters | `polars` | Optional `dataframe` feature; DataFrames remain boundary values rather than RF storage types |
| typing-extensions | Python-only structural typing helpers | Rust enums, traits, structs, and generics | No runtime dependency |
| networkx | Circuit topology and graph algorithms/drawing | `petgraph`; drawing is represented by backend-neutral plotting data | Core graph model |
| matplotlib | Smith charts and other plotting helpers | Backend-neutral plot series plus `plotters` rendering | Optional `plot` feature |
| Bokeh | Notebook rectangular and polar plots | The same backend-neutral plot series and `plotters` rendering | Covered by the optional `plot` feature; no browser-global Bokeh state is required |
| openpyxl | Validation of generated Excel workbooks | `rust_xlsxwriter` | Optional `xlsx` feature; the upstream API writes workbooks and does not expose an Excel reader |
| PyVISA / pyvisa-py | VNA discovery and message/register I/O | `visa-rs` selected behind the crate-owned `InstrumentSession` transport trait | Optional `visa` feature; native `VisaSession` supports Windows x86/x64 (`visa32`/`visa64`, optionally resolved with `LIB_VISA_PATH`) and macOS x86_64/aarch64 (`VISA.framework`) when `RUST_RF_NATIVE_VISA=1`; Linux is intentionally unsupported, while transport-independent APIs and mocks remain portable |
| pytest and NumPy testing helpers | Test discovery, assertions, fixtures, parametrization, and mocks | Rust test harness plus `approx` and typed mock transports | `approx` is a development dependency; fixtures are copied below `src/` at their upstream paths |
| Python `datetime` | Sortable timestamps in utility APIs | `chrono` | Core utility dependency |
| Python `csv` and text decoding | Instrument CSV parsing and Windows-1252 Touchstone fallback | `csv` and `encoding_rs` | Core I/O dependencies |
| Python `json` and pickle | Object persistence | `serde` and `serde_json`; unsafe pickle input is deliberately rejected | Core, typed safe persistence |
| Python `re` | Touchstone, command, and filename parsing | `regex` | Core parser dependency |
| Python `zipfile` | Touchstone archives and `NetworkSet.from_zip` | `zip` with only DEFLATE enabled | Core I/O dependency |

## Numerical architecture

SciPy is not one dependency in Rust. Its use in scikit-rf spans interpolation,
least-squares optimization, quadrature, Bessel/elliptic functions, signal
windows, convolution, probability distributions, and physical constants. The
port keeps those choices behind crate-owned RF types and explicit algorithms,
so callers do not depend on a particular numerical backend.

Polars is intentionally optional. The core `Network` and `NetworkSet` types use
typed arrays; DataFrames are boundary representations for export and analysis
rather than the storage model. The same boundary rule applies to Plotters,
`rust_xlsxwriter`, and VISA: disabling those features leaves the numerical RF
core available.
