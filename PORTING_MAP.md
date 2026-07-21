# scikit-rf porting map

This map relates Rust modules to the analyzed scikit-rf source at commit
`ca243628ee5fd91ed030fec52bcec08d778a8516`. The full source knowledge graph is
stored at `D:\GitHub\scikit-rf\.understand-anything\knowledge-graph.json`.

## Mechanical layout contract

- Every upstream directory below `skrf/` has the same relative directory below `src/`.
- Every Python file except package `__init__.py` and Python-only `conftest.py` has a Rust counterpart at the same relative path, with Rust snake_case naming. A zero-byte counterpart is a structural placeholder, not evidence that its implementation is complete.
- Every tracked non-Python file below `skrf/` is copied byte-for-byte to the same relative path below `src/`; the initial mirror was verified by SHA-256 before the temporary migration logic was removed.
- Test directories, Rust test counterparts, and their non-Python verification assets use the same relative paths below `src/`; no parallel top-level `skrf/` tree is maintained.

| Rust module | Python origin | Implementation status | Rust tests | Remaining upstream coverage |
| --- | --- | --- | --- | --- |
| `src/frequency.rs` | `skrf/frequency.py` | Complete typed construction, arbitrary/linear/log classification, scaling/unit mutation, display, derived properties and gradients, padded time axes, monotonic cleanup, inclusive/nearest slicing, overlap, rounding, and all arithmetic operations with one-point broadcasting | `src/tests/test_frequency.rs` | Python string-index syntax, warnings, plotting, and operator overloading are represented by explicit fallible Rust methods and the plotting module |
| `src/constants.rs` | `skrf/constants.py` | Numerical/physical constants, typed S/frequency/sweep counterparts in their owning modules, distance-unit parsing, scalar/array distance and propagation-time conversion, and group-velocity overrides implemented | `tests/constants.rs` | Python dynamic module attributes are represented by explicit Rust constants and typed APIs |
| `src/data.rs`, `data/*` | `skrf/data/__init__.py`, bundled Touchstone/style data | All 17 lazy example/test networks are embedded and parsed through the Rust Touchstone reader; the material catalog and aliases are typed; the plotting style is embedded | `src/tests/test_static_data.rs` | The unsafe Python pickle calibration fixture is explicitly rejected in favor of typed Rust serialization |
| `src/plotting.rs`, `src/notebook/{utils,bokeh_,matplotlib_}.rs`, `src/programs/plot_touchstone.rs` | `skrf/plotting.py`, `skrf/notebook/*.py`, `skrf/programs/plot_touchstone.py` | Complete applicable backend-neutral plotting surface: rectangular/complex/polar/Smith, passivity, both reciprocity metrics, centered time-domain traces, arbitrary generated components, uncertainty and multi-sigma/min-max bounds, decomposition/log-sigma, signatures, violin distributions, animation frames, contour/vector/band data, notebook palette/cycling, feature-gated Plotters SVG rendering, and multi-file CLI | `src/tests/test_plotting.rs` | Matplotlib process-global figure management, interactive redraws, legend mutation, and Smith-grid decorations are intentionally represented as renderer concerns rather than library-global side effects |
| `src/lib.rs` | `skrf/__init__.py`, empty package initializers | Public module surface, core type reexports, and a compile-time crate version are exposed; optional integrations remain feature-gated | `src/tests/test_init.rs` | Plot-environment setup follows the plotting module port; Python import mechanics have no Rust equivalent |
| `src/instances.rs` | `skrf/instances.py` | Lazy method-based air/air50 media plus all 37 named WR/WM frequency bands and 50-ohm rectangular-waveguide instances implemented from a typed catalog | `src/tests/test_static_data.rs` | Python dynamic module attributes are represented by explicit methods and the `INSTANCES` catalog |
| `src/math.rs` | `skrf/mathFunctions.py` | Complete applicable surface: scalar complex/unit conversions, phase unwrapping/root selection, Dirac/Neumann helpers, null space, typed complexification, complex serialization/Fortran-order flattening, matrix predicates, infinity handling, deterministic bounded complex/Gaussian random data, PSD-to-time conversion, 1-D and arbitrary-trailing-dimension axis-zero Floater-Hormann interpolation, centered complex/real inverse FFTs on 1-D and n-D axis zero, eigenvalue nudging, and batched right-side solving | `src/tests/test_math_functions.rs` | Complete; Python decorators, global generator handles, and callable window strings are represented by typed functions, seeded synchronization, and `Window` enums |
| `src/transmission_line.rs` | `skrf/tlineFunctions.py` | Complete typed transmission-line helper family implemented, including skin/surface loss, distributed-circuit conversion, physical/electrical length, reflection/impedance transforms, at-theta variants, propagation recovery, SWR, voltage/current propagation, and total loss | `src/tests/test_tline_functions.rs` | Python array broadcasting and callable propagation constants are expressed through Rust iterators and explicit function composition |
| `src/network.rs` | `skrf/network.py` | Complete applicable typed surface: construction from S/Z/Y and Touchstone; Cartesian/polar/cubic/rational interpolation and DC extrapolation; derived S components and S/Z/Y/H/G/T/ABCD conversions; spectral passivity/losslessness/reciprocity/symmetry, active parameters, group delay, corrected maximum stable/available/unilateral gain, stability/gain/noise circles; windowed impulse/step responses; noise parameters and additive/flatband/multiplicative perturbation; crop/subnetwork/rotate/delay/nudge; renormalization and single-ended/mixed-mode conversion; connect/inner/parallel/cascade/de-embed/stitch/overlap/port concatenation/averaging; explicit one-/two-port N-port reconstruction; and safe serialization/export | `src/tests/test_network.rs`, `src/io/tests/test_touchstone.rs`, `src/io/tests/test_ts_spec.rs`, `src/calibration/tests/test_deembedding.rs`, `tests/time.rs`, `src/io/tests/test_io.rs` | None. Python operator overloading, mutable property setters, filename-based N-port index parsing, pickle, and process-global plotting are represented by explicit fallible methods, typed index mappings, safe serialization, and backend-neutral plotting |
| `src/network_set.rs` | `skrf/networkSet.py` | Complete applicable typed surface: compatibility/equality; ZIP/directory/S-map/MDIF/CITI construction; named maps; numeric/text parameters; selection/filter/sort/random sampling and parameter/frequency interpolation; set/network arithmetic, cascade, inversion and de-embedding; generated mean/deviation statistics across all seven Network primary parameter families and magnitude/dB/dB10/phase/real/imaginary/VSWR projections; polar noise, uncertainty bounds, datetime indexes, scalar matrices/covariance; aggregate/get-set/tuner helpers; safe object/MDIF/XLSX/DataFrame output; and backend-neutral advanced plotting/animation adapters | `src/tests/test_network_set.rs`, `src/tests/test_plotting.rs`, `tests/metas.rs`, `src/io/tests/test_mdif.rs`, `src/io/tests/test_citi.rs` | None. Python runtime method generation and operator overloading are represented by explicit parameter/component enums and fallible methods |
| `src/io/metas.rs` | `skrf/io/metas.py` | METAS SDATCV header, reference impedances, column-major complex mean, sample covariance matrix, and tabular writer implemented | `tests/metas.rs` | External METAS tool round-trip comparison |
| `src/io/general.rs` | `skrf/io/general.py` | Safe tagged JSON storage for Frequency/Network/NetworkSet, complete Network JSON round trips, recursive mixed-object/Touchstone discovery, bulk object writes, Statistical-to-Touchstone conversion, RI/MA/DB tables, CSV/HTML output, feature-gated Polars DataFrames and single/multi-sheet XLSX output implemented | `src/io/tests/test_io.rs` | Python pickle/session introspection is intentionally replaced by typed safe serialization |
| `src/io/mod.rs`, `src/io/touchstone.rs`, `src/io/citi.rs`, `src/io/mdif.rs`, `src/io/csv.rs` | `skrf/io/__init__.py`, `touchstone.py`, `citi.py`, `mdif.py`, `csv.py` | Public I/O reexports; `touchstone.py` is isolated in its auditable counterpart and is complete with Touchstone 1.x/2.0 official examples, mixed-mode order, HFSS Gamma/Z0, ZIP archives, noise, S/Z/Y/H/G RI/MA/DB conversion, Windows-1252 fallback, comment variables/filtering, format/name helpers, and round trips. CITI, MDIF, and instrument CSV implementations are also present | `src/io/tests/*.rs`, `src/tests/test_convenience.rs`, `src/tests/test_network.rs` | Remaining I/O work, if any, is outside `touchstone.py` |
| `src/media/{media,distributed_circuit,freespace,coaxial,defined_a_ep_tand_z0,rectangular_waveguide,circular_waveguide,cpw,mline}.rs` | `skrf/media/media.py`, `distributedCircuit.py`, `freespace.py`, `coaxial.py`, `definedAEpTandZ0.py`, `rectangularWaveguide.py`, `circularWaveguide.py`, `cpw.py`, `mline.py` | Complete. The shared Media API includes wave constants/velocities, physical/electrical/time conversion, matches, complex wave-definition mismatches, splitters, lumped/Q/shunt components, lines/floating lines/delayed loads, attenuators, random/noise networks, distance extraction, plots, and CSV I/O. All specialized media sources are implemented with upstream simulator fixtures | `src/media/tests/test_*.rs`, `src/media/tests/**` fixture assets | None for this mapped source cluster |
| `src/media/device.rs` | `skrf/media/device.py` | Generic Device trait, matched symmetric 3/4-port couplers, dB/degree construction, Hybrid, QuadratureHybrid, Hybrid180, DualCoupler composition, validation, termination, and port renumbering implemented | `tests/media_device.rs` | Broader external simulator comparisons |
| `src/taper.rs` | `skrf/taper.py` | Generic media factory, section/value generation, cascaded junction network, linear/exponential/smooth-step/Klopfenstein and custom normalized profiles implemented | `src/tests/test_taper.rs` | Broader media-specific fixture comparisons |
| `src/qfactor.rs` | `skrf/qfactor.py` | Complete applicable surface: validated one-port construction, display, resonance seeding, typed `f`/`w`/`c` loop plans, weighted/unweighted NLQFIT6, NLQFIT7 phase-delay, NLQFIT8 frequency-dependent leakage, model-specific fitted responses/networks, Q-circle, unloaded Q, and resonant-frequency/bandwidth Hz/scaled outputs | `src/tests/test_qfactor.rs`, `tests/fixtures/qfactor/*` | Complete; all four upstream NPL MAT 58 reference datasets are vendored verbatim with their CC0 notice and verified |
| `src/time.rs` | `skrf/time.py` | Peak detection/search, automatic span detection, periodic windows, normalized real inverse FFT, typed time units, FFT/RFFT/convolution gating, and band-pass/band-stop modes implemented | `tests/time.rs` | Callable user-defined windows and warning-message parity are represented by typed Rust APIs rather than Python callbacks/warnings |
| `src/circuit.rs` | `skrf/circuit.py` | Complete typed circuit topology and solve surface: validation, factories, network replacement/introspection, graph/connectivity/edges, mismatched intersections, global/external S, reduction to a solved N-port, incident/outgoing waves, internal/external voltages and currents, active S/Z/Y/VSWR, and assembled Network construction | `src/tests/test_circuit.rs`, `tests/fixtures/designer_wilkinson_splitter.s3p` | Python-only cache controls and plotting-side effects are replaced by deterministic Rust values and petgraph data |
| `src/vector_fitting.rs` | `skrf/vectorFitting.py` | Complete applicable typed fitting surface: stable linear and custom real/complex-conjugate initialization, iterative shared-denominator pole relocation, adaptive order selection, multi-response residue/constant/proportional identification with DC preservation, model-order and spurious-pole analysis, direct and ABCDE state-space reconstruction, RMS error, sampled passivity bands/enforcement, NumPy-compatible NPZ persistence, backend-neutral response/singular-value plots, and state-space SPICE synthesis | `src/tests/test_vector_fitting.rs`, `tests/fixtures/vector_fitting/190ghz_tx_measured.s2p` | Upstream warning timing and Matplotlib side effects are represented by `Result` errors and backend-neutral plot values; automatic pole adding/skimming is represented by deterministic lowest-error order selection |
| `src/vi/scpi_errors.rs`, `src/vi/validators.rs`, `src/vi/visa.rs`, `src/vi/vna/vna.rs`, `src/vi/vna/hp/*.rs`, `src/vi/vna/nanovna/nanovna.rs`, `src/vi/vna/rohde_schwarz/{rs_vna,zna,zva}.rs`, `src/vi/vna/keysight/{fieldfox,pna}.rs` | `skrf/vi/scpi_errors.py`, `validators.py`, VNA base, HP 8510C/8720B and sweep planner, NanoVNA, R&S family, and Keysight FieldFox/PNA modules | Complete transport-independent instrument foundation plus native `visa-rs` adapters for Windows x86/x64 and macOS x86_64/aarch64 (`RUST_RF_NATIVE_VISA=1`); HP native/list/compound sweeps and FORM2 transfers; NanoVNA binary protocol; R&S ZNA/ZVA; FieldFox settings/calibration/traces; and PNA channels, formats, averaging-aware sweeps, trace and SNP assembly implemented | `src/vi/tests/*.rs`, `src/vi/vna/**/tests/*.rs`, `tests/nanovna.rs`, `tests/scpi_errors.rs` | Python dynamic descriptors are typed methods; PNA calibration remains intentionally unsupported as in upstream; native VISA is intentionally unsupported on Linux and live hardware verification remains environment-dependent |
| `src/calibration/{calibration,calibration_set,deembedding}.rs` | `skrf/calibration/calibration.py`, `skrf/calibration/calibrationSet.py`, `skrf/calibration/deembedding.py` | Common calibration bulk/residual/coefficient behavior; OnePort, SDDL/SDDLWeikle/PHN, EightTerm, TwelveTerm/SOLT, TwoPortOnePath/EnhancedResponse, TRL and NIST/TUG multiline TRL, UnknownThru/MRC, LRM/LRRM, SixteenTerm, LMR16, MultiportCal, and MultiportSOLT solve/correct/embed equations; switch-term terminate/unterminate correction; Dot and Cartesian calibration-set combinatorics; Normalization; eight lumped de-embedding methods; IEEE 370 FER traces and limits, frequency- and application-based time-domain quality metrics, DC/non-harmonic restoration, NRP/sample shifting, impedance and FFT-gated SE/MM NZC extraction, and SE/MM ZC impedance peeling | `src/calibration/tests/test_calibration.rs`, `test_calibration_set.rs`, `test_deembedding.rs` | Renderer-specific IEEE diagnostic figures are represented by backend-neutral trace/limit data |
| `src/util.rs` | `skrf/util.py` | Timestamp generation/parsing, nearest-value/domain selection, path extension/basename helpers, Git descriptions, structured dictionary records, recursive UTF-8 file replacement, duplicate detection/unique naming, reflected flat/Hanning/Hamming/Bartlett/Blackman smoothing, typed HomoList/HomoDict mapping/filtering/selection, and progress rendering implemented | `src/tests/test_util.rs` | Complete for applicable behavior; Python docstring decorators, flexible file-object opening, and NumPy warning-state contexts are replaced by Rust documentation, `Read`/`Write` traits, and explicit result handling |
## Graph-derived file and type origin index

The following index is generated from the Understand Anything graph at the
upstream commit named above. It covers all 88 eligible Python files and all 208
class/type nodes in those files; a dash means that the source file defines
functions or constants rather than a Python class.

| Rust counterpart | Python origin | Python classes/types from graph |
| --- | --- | --- |
| `src/calibration/calibration.rs` | `skrf/calibration/calibration.py` | `Calibration` (L199)<br>`OnePort` (L1083)<br>`SDDLWeikle` (L1248)<br>`SDDL` (L1383)<br>`PHN` (L1500)<br>`TwelveTerm` (L1624)<br>`SOLT` (L1891)<br>`TwoPortOnePath` (L1986)<br>`EnhancedResponse` (L2138)<br>`EightTerm` (L2157)<br>`TRL` (L2521)<br>`NISTMultilineTRL` (L2738)<br>`TUGMultilineTRL` (L3666)<br>`UnknownThru` (L4188)<br>`LRM` (L4322)<br>`LRRM` (L4528)<br>`MRC` (L4999)<br>`SixteenTerm` (L5062)<br>`LMR16` (L5433)<br>`Normalization` (L5732)<br>`MultiportCal` (L5744)<br>`MultiportSOLT` (L6124) |
| `src/calibration/calibration_set.rs` | `skrf/calibration/calibrationSet.py` | `CalibrationSet` (L49)<br>`Dot` (L129) |
| `src/calibration/deembedding.rs` | `skrf/calibration/deembedding.py` | `Deembedding` (L79)<br>`OpenShort` (L155)<br>`Open` (L268)<br>`ShortOpen` (L358)<br>`Short` (L462)<br>`SplitPi` (L554)<br>`SplitTee` (L655)<br>`AdmittanceCancel` (L756)<br>`ImpedanceCancel` (L855)<br>`IEEEP370` (L957)<br>`IEEEP370_FER` (L1498)<br>`IEEEP370_FD_QM` (L1811)<br>`IEEEP370_TD_QM` (L2077)<br>`IEEEP370_SE_NZC_2xThru` (L3020)<br>`IEEEP370_MM_NZC_2xThru` (L3512)<br>`IEEEP370_SE_ZC_2xThru` (L3905)<br>`IEEEP370_MM_ZC_2xThru` (L4487) |
| `src/calibration/tests/test_calibration.rs` | `skrf/calibration/tests/test_calibration.py` | `DetermineTest` (L62)<br>`ComputeSwitchTermsTest` (L147)<br>`CalibrationTest` (L176)<br>`CalibrationInputsTest` (L224)<br>`OnePortTest` (L282)<br>`SDDLTest` (L353)<br>`SDDLWeikleTest` (L420)<br>`SDDMTest` (L459)<br>`PHNTest` (L504)<br>`EightTermTest` (L556)<br>`TRLTest` (L709)<br>`TRLLongThruTest` (L759)<br>`TRLWithNoIdealsTest` (L824)<br>`TRLMultiline` (L867)<br>`NISTMultilineTRLTest` (L912)<br>`NISTMultilineTRLTest2` (L953)<br>`TUGMultilineTest` (L1050)<br>`TUGMultilineNonzeroThruTest` (L1102)<br>`TUGMultilineLnorm2Test` (L1107)<br>`TUGMultilineRepeatedLinesTest` (L1112)<br>`TUGMultilineNoReflectTest` (L1118)<br>`TREightTermTest` (L1153)<br>`TwelveTermTest` (L1212)<br>`TwelveTermSloppyInitTest` (L1355)<br>`SOLTTest` (L1407)<br>`TwoPortOnePathTest` (L1442)<br>`UnknownThruTest` (L1535)<br>`LRMTest` (L1573)<br>`LRRMTest` (L1624)<br>`LRRMTestNoFit` (L1703)<br>`MRCTest` (L1768)<br>`TwelveTermToEightTermTest` (L1815)<br>`SixteenTermTest` (L1921)<br>`SixteenTermCoefficientsTest` (L2067)<br>`LMR16Test` (L2177)<br>`MultiportCalTest` (L2228)<br>`MultiportSOLTTest` (L2334) |
| `src/calibration/tests/test_calibration_set.rs` | `skrf/calibration/tests/test_calibrationSet.py` | `CalsetTest` (L10)<br>`DotOneport` (L26)<br>`DotEightTerm` (L61) |
| `src/calibration/tests/test_deembedding.rs` | `skrf/calibration/tests/test_deembedding.py` | `DeembeddingTestCase` (L26) |
| `src/circuit.rs` | `skrf/circuit.py` | `Circuit` (L115) |
| `src/constants.rs` | `skrf/constants.py` | — |
| `src/frequency.rs` | `skrf/frequency.py` | `InvalidFrequencyWarning` (L64)<br>`Frequency` (L70) |
| `src/instances.rs` | `skrf/instances.py` | `StaticInstances` (L64) |
| `src/io/citi.rs` | `skrf/io/citi.py` | `Citi` (L30) |
| `src/io/csv.rs` | `skrf/io/csv.py` | `AgilentCSV` (L299) |
| `src/io/general.rs` | `skrf/io/general.py` | `TouchstoneEncoder` (L821) |
| `src/io/mdif.rs` | `skrf/io/mdif.py` | `Mdif` (L31) |
| `src/io/metas.rs` | `skrf/io/metas.py` | — |
| `src/io/tests/test_citi.rs` | `skrf/io/tests/test_citi.py` | `CitiTestCase` (L11) |
| `src/io/tests/test_csv.rs` | `skrf/io/tests/test_csv.py` | `AgilentCSVTestCase` (L12) |
| `src/io/tests/test_io.rs` | `skrf/io/tests/test_io.py` | `IOTestCase` (L12) |
| `src/io/tests/test_mdif.rs` | `skrf/io/tests/test_mdif.py` | `MdifTestCase` (L12) |
| `src/io/tests/test_touchstone.rs` | `skrf/io/tests/test_touchstone.py` | `TouchstoneTestCase` (L13) |
| `src/io/tests/test_ts_spec.rs` | `skrf/io/tests/test_ts_spec.py` | — |
| `src/io/touchstone.rs` | `skrf/io/touchstone.py` | `ParserState` (L59)<br>`Touchstone` (L184) |
| `src/math.rs` | `skrf/mathFunctions.py` | — |
| `src/media/circular_waveguide.rs` | `skrf/media/circularWaveguide.py` | `CircularWaveguide` (L29) |
| `src/media/coaxial.rs` | `skrf/media/coaxial.py` | `Coaxial` (L31) |
| `src/media/cpw.rs` | `skrf/media/cpw.py` | `CPW` (L28) |
| `src/media/defined_a_ep_tand_z0.rs` | `skrf/media/definedAEpTandZ0.py` | `DefinedAEpTandZ0` (L31) |
| `src/media/device.rs` | `skrf/media/device.py` | `Device` (L40)<br>`MatchedSymmetricCoupler` (L62)<br>`Hybrid` (L152)<br>`QuadratureHybrid` (L164)<br>`Hybrid180` (L174)<br>`DualCoupler` (L207) |
| `src/media/distributed_circuit.rs` | `skrf/media/distributedCircuit.py` | `DistributedCircuit` (L27) |
| `src/media/freespace.rs` | `skrf/media/freespace.py` | `Freespace` (L32) |
| `src/media/media.rs` | `skrf/media/media.py` | `Media` (L34)<br>`DefinedGammaZ0` (L1804) |
| `src/media/mline.rs` | `skrf/media/mline.py` | `MLine` (L28) |
| `src/media/rectangular_waveguide.rs` | `skrf/media/rectangularWaveguide.py` | `RectangularWaveguide` (L46) |
| `src/media/tests/test_all_construction.rs` | `skrf/media/tests/test_all_construction.py` | `MediaTestCase` (L15)<br>`Z0InitDeprecationTestCase` (L104)<br>`FreespaceTestCase` (L131)<br>`CPWTestCase` (L141)<br>`RectangularWaveguideTestCase` (L153)<br>`DistributedCircuitTestCase` (L162) |
| `src/media/tests/test_coaxial.rs` | `skrf/media/tests/test_coaxial.py` | `MediaTestCase` (L14) |
| `src/media/tests/test_cpw.rs` | `skrf/media/tests/test_cpw.py` | `CPWTestCase` (L20) |
| `src/media/tests/test_definedaeptandz0.rs` | `skrf/media/tests/test_definedaeptandz0.py` | `DefinedAEpTandZ0TestCase` (L17) |
| `src/media/tests/test_distributed_circuit.rs` | `skrf/media/tests/test_distributedCircuit.py` | `MediaTestCase` (L12) |
| `src/media/tests/test_media.rs` | `skrf/media/tests/test_media.py` | `DefinedGammaZ0TestCase` (L14)<br>`STwoPortsNetworkTestCase` (L652)<br>`ABCDTwoPortsNetworkTestCase` (L754)<br>`DefinedGammaZ0_s_def` (L896) |
| `src/media/tests/test_mline.rs` | `skrf/media/tests/test_mline.py` | `MLineTestCase` (L18) |
| `src/media/tests/test_rectangular_waveguide.rs` | `skrf/media/tests/test_rectangularWaveguide.py` | `MediaTestCase` (L9) |
| `src/network.rs` | `skrf/network.py` | `Network` (L236) |
| `src/network_set.rs` | `skrf/networkSet.py` | `NetworkSet` (L74) |
| `src/notebook/bokeh_.rs` | `skrf/notebook/bokeh_.py` | — |
| `src/notebook/matplotlib_.rs` | `skrf/notebook/matplotlib_.py` | — |
| `src/notebook/utils.rs` | `skrf/notebook/utils.py` | — |
| `src/plotting.rs` | `skrf/plotting.py` | — |
| `src/programs/plot_touchstone.rs` | `skrf/programs/plot_touchstone.py` | — |
| `src/qfactor.rs` | `skrf/qfactor.py` | `OptimizedResult` (L149)<br>`Qfactor` (L203) |
| `src/taper.rs` | `skrf/taper.py` | `Taper1D` (L39)<br>`Linear` (L240)<br>`Exponential` (L254)<br>`SmoothStep` (L282)<br>`Klopfenstein` (L310) |
| `src/tests/test_circuit.rs` | `skrf/tests/test_circuit.py` | `CircuitTestConstructor` (L13)<br>`CircuitClassMethods` (L182)<br>`CircuitTestWilkinson` (L253)<br>`CircuitTestCascadeNetworks` (L420)<br>`CircuitTestMultiPortCascadeNetworks` (L554)<br>`CircuitTestVariableCoupler` (L943)<br>`CircuitTestGraph` (L1056)<br>`CircuitTestComplexCharacteristicImpedance` (L1100)<br>`CircuitTestVoltagesCurrents` (L1232)<br>`CircuitTestVoltagesNonReciprocal` (L1385) |
| `src/tests/test_convenience.rs` | `skrf/tests/test_convenience.py` | `ConvenienceTestCase` (L10) |
| `src/tests/test_frequency.rs` | `skrf/tests/test_frequency.py` | `FrequencyTestCase` (L10) |
| `src/tests/test_init.rs` | `skrf/tests/test_init.py` | — |
| `src/tests/test_math_functions.rs` | `skrf/tests/test_mathFunctions.py` | `TestUnitConversions` (L13)<br>`TestRandom` (L212) |
| `src/tests/test_network.rs` | `skrf/tests/test_network.py` | `NetworkTestCase` (L59) |
| `src/tests/test_network_set.rs` | `skrf/tests/test_networkSet.py` | `NetworkSetTestCase` (L15) |
| `src/tests/test_plotting.rs` | `skrf/tests/test_plotting.py` | — |
| `src/tests/test_qfactor.rs` | `skrf/tests/test_qfactor.py` | `QfactorTests` (L11) |
| `src/tests/test_static_data.rs` | `skrf/tests/test_static_data.py` | — |
| `src/tests/test_taper.rs` | `skrf/tests/test_taper.py` | `Taper1DTestCase` (L14)<br>`LinearTestCase` (L56)<br>`ExponentialTestCase` (L82)<br>`SmoothStepTestCase` (L107)<br>`KlopfensteinTestCase` (L134) |
| `src/tests/test_tline_functions.rs` | `skrf/tests/test_tlineFunctions.py` | `TestBasicTransmissionLine` (L10)<br>`ElectricalLengthTests` (L47)<br>`TestVoltageCurrentPropagation` (L130) |
| `src/tests/test_util.rs` | `skrf/tests/test_util.py` | `HomoDictTest` (L6)<br>`HomoListTest` (L26) |
| `src/tests/test_vector_fitting.rs` | `skrf/tests/test_vectorfitting.py` | `VectorFittingTestCase` (L13) |
| `src/time.rs` | `skrf/time.py` | — |
| `src/transmission_line.rs` | `skrf/tlineFunctions.py` | — |
| `src/util.rs` | `skrf/util.py` | `HomoList` (L399)<br>`HomoDict` (L497)<br>`ProgressBar` (L774) |
| `src/vector_fitting.rs` | `skrf/vectorFitting.py` | `VectorFitting` (L22) |
| `src/vi/scpi_errors.rs` | `skrf/vi/scpi_errors.py` | `SCPIError` (L131) |
| `src/vi/tests/test_validators.rs` | `skrf/vi/tests/test_validators.py` | — |
| `src/vi/tests/test_vna.rs` | `skrf/vi/tests/test_vna.py` | — |
| `src/vi/validators.rs` | `skrf/vi/validators.py` | `ValidationError` (L22)<br>`Validator` (L26)<br>`IntValidator` (L41)<br>`FloatValidator` (L71)<br>`FreqValidator` (L107)<br>`EnumValidator` (L132)<br>`SetValidator` (L152)<br>`DictValidator` (L169)<br>`DelimitedStrValidator` (L193)<br>`BooleanValidator` (L208) |
| `src/vi/vna/hp/hp8510c_sweep_plan.rs` | `skrf/vi/vna/hp/hp8510c_sweep_plan.py` | `SweepSection` (L13)<br>`LinearBuiltinSweepSection` (L32)<br>`LinearMaskedSweepSection` (L45)<br>`LinearCustomSweepSection` (L65)<br>`RandomSweepSection` (L92)<br>`SweepPlan` (L184) |
| `src/vi/vna/hp/hp8510c.rs` | `skrf/vi/vna/hp/hp8510c.py` | `HP8510C` (L29) |
| `src/vi/vna/hp/hp8720b.rs` | `skrf/vi/vna/hp/hp8720b.py` | `HP8720B` (L24) |
| `src/vi/vna/hp/tests/test_8510c_sweep_plan.rs` | `skrf/vi/vna/hp/tests/test_8510c_sweep_plan.py` | — |
| `src/vi/vna/hp/tests/test_8510c.rs` | `skrf/vi/vna/hp/tests/test_8510c.py` | — |
| `src/vi/vna/keysight/fieldfox.rs` | `skrf/vi/vna/keysight/fieldfox.py` | `WindowFormat` (L48)<br>`FieldFox` (L66) |
| `src/vi/vna/keysight/pna.rs` | `skrf/vi/vna/keysight/pna.py` | `SweepType` (L28)<br>`SweepMode` (L37)<br>`TriggerSource` (L44)<br>`AveragingMode` (L50)<br>`PNA` (L55) |
| `src/vi/vna/keysight/tests/test_fieldfox.rs` | `skrf/vi/vna/keysight/tests/test_fieldfox.py` | — |
| `src/vi/vna/keysight/tests/test_pna.rs` | `skrf/vi/vna/keysight/tests/test_pna.py` | — |
| `src/vi/vna/nanovna/nanovna.rs` | `skrf/vi/vna/nanovna/nanovna.py` | `OP` (L18)<br>`REG_ADDR` (L32)<br>`NanoVNAv2` (L46) |
| `src/vi/vna/rohde_schwarz/rs_vna.rs` | `skrf/vi/vna/rohde_schwarz/rs_vna.py` | `SweepType` (L27)<br>`SweepMode` (L39)<br>`AveragingMode` (L44)<br>`RSVNA` (L51) |
| `src/vi/vna/rohde_schwarz/tests/test_zna.rs` | `skrf/vi/vna/rohde_schwarz/tests/test_zna.py` | — |
| `src/vi/vna/rohde_schwarz/tests/test_zva.rs` | `skrf/vi/vna/rohde_schwarz/tests/test_zva.py` | — |
| `src/vi/vna/rohde_schwarz/zna.rs` | `skrf/vi/vna/rohde_schwarz/zna.py` | `ZNA` (L4) |
| `src/vi/vna/rohde_schwarz/zva.rs` | `skrf/vi/vna/rohde_schwarz/zva.py` | `ZVA` (L4) |
| `src/vi/vna/vna.rs` | `skrf/vi/vna/vna.py` | `ValuesFormat` (L45)<br>`Channel` (L56)<br>`VNA` (L80) |

Bead status remains the fine-grained implementation ledger; this graph-derived
index is the authoritative path and upstream type-origin cross-reference.
