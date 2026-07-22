# Generated port coverage audit

## Current completion evidence

The authoritative audit at upstream commit
`ca243628ee5fd91ed030fec52bcec08d778a8516` covers the complete tracked
`skrf/` tree while intentionally excluding the separately ported `doc/` tree:

- 394 tracked files under `skrf/` inspected.
- 101 Python files found: 88 eligible implementation/test files and 13 package
  initializers or Python-only `conftest.py` files.
- 88 of 88 eligible Python files have nonempty Rust counterparts at their
  auditable paths below `src/`.
- 31 of 31 upstream directories have matching directories below `src/`.
- 293 of 293 non-Python fixtures/assets exist at matching paths and have
  identical SHA-256 hashes.
- The Understand Anything graph contains 1,301 nodes and 1,658 edges. All 88
  eligible Python files have graph file nodes, and the 208 class/type nodes in
  those files are indexed in `PORTING_MAP.md`.
- `cargo fmt --all --check` and `cargo test --locked --all-features` pass in the
  current completion audit.
- Both non-mutating equivalents of the repository's production/test Clippy
  commands exit successfully; production has zero `unwrap()`/`expect()`
  diagnostics. Pedantic/nursery rules remain warning-level by repository
  policy.
- The native VISA dependency graph is present for Windows x86/x64, absent on
  macOS and Linux, and the Windows native adapter type-checks when
  `RUST_RF_NATIVE_VISA=1`.

## Initial symbol-name snapshot

The table below is retained as the reproducible initial inventory. A literal
identifier name match was only structural evidence; Python/Rust naming,
operator, property, and generated-method differences make unmatched names
expected. Behavioral completion is established by the implementation ledger,
ported tests, fixture parity, and full Rust gates rather than this initial
heuristic.

- Understand Anything graph nodes: 1301
- Understand Anything file nodes: 555
- Tracked Python files: 102
- Python classes/functions/methods: 2559
- Rust identifier name matches: 563
- Unmatched names: 1996

| Python file | Graph degree | Symbols | Name matches | Unmatched examples |
| --- | ---: | ---: | ---: | --- |
| `skrf/calibration/tests/test_calibration.py` | 162 | 222 | 6 | `_compare_dicts_allclose`, `DetermineTest`, `DetermineTest.setUp`, `DetermineTest.test_determine_line`, `DetermineTest.test_determine_reflect`, `DetermineTest.test_determine_reflect_matched_thru_and_line_ideal_reflect`, `DetermineTest.test_determine_reflect_matched_thru_and_line`, `DetermineTest.test_determine_reflect_regression`, +208 more |
| `skrf/network.py` | 319 | 219 | 51 | `Network._generated_functions`, `Network.__init__`, `Network.from_z`, `Network.__pow__`, `Network.__rshift__`, `Network.__floordiv__`, `Network.__mul__`, `Network.__rmul__`, +160 more |
| `skrf/tests/test_network.py` | 16 | 140 | 0 | `NetworkTestCase`, `NetworkTestCase.setUp`, `NetworkTestCase.test_network_copy`, `NetworkTestCase.test_two_port_reflect`, `NetworkTestCase.test_network_empty_frequency_range`, `NetworkTestCase.test_network_sequence_frequency_with_f_unit`, `NetworkTestCase.test_timedomain`, `NetworkTestCase.test_time_gate`, +132 more |
| `skrf/calibration/calibration.py` | 156 | 170 | 47 | `Calibration.__init__`, `Calibration.__repr__`, `Calibration.apply_cal`, `Calibration.apply_cal_to_list`, `Calibration.apply_cal_to_all_in_dir`, `Calibration.apply_cal_to_network_set`, `Calibration.pop`, `Calibration.remove_and_cal`, +115 more |
| `skrf/tests/test_circuit.py` | 47 | 83 | 1 | `CircuitTestConstructor`, `CircuitTestConstructor.setUp`, `CircuitTestConstructor.test_all_networks_have_name`, `CircuitTestConstructor.test_all_networks_have_same_frequency`, `CircuitTestConstructor.test_no_duplicate_node`, `CircuitTestConstructor.test_s_active`, `CircuitTestConstructor.test_auto_reduce`, `CircuitTestConstructor.test_auto_reduce_with_passed_arguments`, +74 more |
| `skrf/instances.py` | 12 | 78 | 3 | `StaticInstances.f_wr51`, `StaticInstances.f_wr42`, `StaticInstances.f_wr34`, `StaticInstances.f_wr28`, `StaticInstances.f_wr22p4`, `StaticInstances.f_wr18p8`, `StaticInstances.f_wr14p8`, `StaticInstances.f_wr12p2`, +67 more |
| `skrf/calibration/deembedding.py` | 74 | 114 | 41 | `Deembedding.__init__`, `Deembedding.__repr_`, `OpenShort.__init__`, `Open.__init__`, `ShortOpen.__init__`, `Short.__init__`, `SplitPi.__init__`, `SplitTee.__init__`, +65 more |
| `skrf/media/tests/test_media.py` | 22 | 52 | 0 | `DefinedGammaZ0TestCase`, `DefinedGammaZ0TestCase.setUp`, `DefinedGammaZ0TestCase.test_impedance_mismatch`, `DefinedGammaZ0TestCase.test_tee`, `DefinedGammaZ0TestCase.test_splitter`, `DefinedGammaZ0TestCase.test_mismatch_splitter`, `DefinedGammaZ0TestCase.test_complex_impedance_mismatch_tee`, `DefinedGammaZ0TestCase.test_splitter_is_reciprocal_and_unitary`, +44 more |
| `skrf/circuit.py` | 19 | 60 | 12 | `Circuit._get_nx`, `Circuit._REDUCE_OPTIONS`, `Circuit.__init__`, `Circuit.connections`, `Circuit.connections`, `Circuit.update_networks`, `Circuit.check_duplicate_names`, `Circuit._is_named`, +40 more |
| `skrf/util.py` | 103 | 61 | 17 | `partial_with_docs`, `copy_doc`, `now_string_2_dt`, `get_fid`, `get_extn`, `basename_noext`, `dict_2_recarray`, `findReplace`, +36 more |
| `skrf/plotting.py` | 185 | 44 | 2 | `plotting_available`, `axes_kwarg`, `figure`, `subplots`, `_get_label_str`, `scale_frequency_ticks`, `smith`, `plot_rectangular`, +34 more |
| `skrf/networkSet.py` | 39 | 61 | 20 | `NetworkSet.__init__`, `NetworkSet.from_dir`, `NetworkSet.from_s_dict`, `NetworkSet.__add_a_operator`, `NetworkSet.__repr__`, `NetworkSet.__getitem__`, `NetworkSet.__add_a_element_wise_method`, `NetworkSet.__add_a_func_on_property`, +33 more |
| `skrf/media/tests/test_all_construction.py` | 27 | 40 | 0 | `MediaTestCase`, `MediaTestCase.test_gamma`, `MediaTestCase.test_z0_value`, `MediaTestCase.test_mode_original_not_modified`, `MediaTestCase.test_match`, `MediaTestCase.test_load`, `MediaTestCase.test_short`, `MediaTestCase.test_open`, +32 more |
| `skrf/mathFunctions.py` | 219 | 51 | 15 | `complex_2_magnitude`, `complex_2_db`, `complex_2_db10`, `complex_2_radian`, `complex_2_degree`, `complex_2_quadrature`, `complex_2_reim`, `magnitude_2_db`, +28 more |
| `skrf/media/media.py` | 37 | 72 | 36 | `Media.__init__`, `Media.mode`, `Media.copy`, `Media.npoints`, `Media.npoints`, `Media.z0_port`, `Media.z0_port`, `Media.z0_override`, +28 more |
| `skrf/frequency.py` | 35 | 51 | 19 | `InvalidFrequencyWarning`, `Frequency.__init__`, `Frequency.__repr__`, `Frequency.__getitem__`, `Frequency.from_f`, `Frequency.__ne__`, `Frequency.__add__`, `Frequency.__sub__`, +24 more |
| `skrf/tests/test_networkSet.py` | 7 | 28 | 0 | `NetworkSetTestCase`, `NetworkSetTestCase.setUp`, `NetworkSetTestCase.test_constructor`, `NetworkSetTestCase.test_from_zip`, `NetworkSetTestCase.test_from_dir`, `NetworkSetTestCase.test_from_s_dict`, `NetworkSetTestCase.test_to_dict`, `NetworkSetTestCase.test_to_s_dict`, +20 more |
| `skrf/tests/test_mathFunctions.py` | 11 | 27 | 0 | `TestUnitConversions`, `TestUnitConversions.setUp`, `TestUnitConversions.test_complex_2_magnitude`, `TestUnitConversions.test_complex_2_db10`, `TestUnitConversions.test_complex_2_degree`, `TestUnitConversions.test_complex_2_quadrature`, `TestUnitConversions.test_complex_components`, `TestUnitConversions.test_complex_2_reim`, +19 more |
| `skrf/tests/test_taper.py` | 23 | 24 | 0 | `Taper1DTestCase`, `Taper1DTestCase.f`, `Taper1DTestCase.setUp`, `Taper1DTestCase.test_value_vector`, `Taper1DTestCase.test_section_length`, `Taper1DTestCase.test_media_at`, `Taper1DTestCase.test_section_at`, `LinearTestCase`, +16 more |
| `skrf/vectorFitting.py` | 7 | 31 | 7 | `VectorFitting.__init__`, `VectorFitting.get_spurious`, `VectorFitting.get_model_order`, `VectorFitting._init_poles`, `VectorFitting._pole_relocation`, `VectorFitting._fit_residues`, `VectorFitting._get_delta`, `VectorFitting._find_error_bands`, +16 more |
| `skrf/tests/test_plotting.py` | 95 | 23 | 0 | `primary_properties`, `primary_methods`, `generated_functions`, `test_primary_plotting`, `test_generated_function_plots`, `test_plot_passivity`, `test_plot_reciprocity`, `test_plot_reciprocity2`, +15 more |
| `skrf/io/touchstone.py` | 35 | 32 | 9 | `remove_prefix`, `ParserState`, `ParserState.n_ansys_impedance_values`, `ParserState.numbers_per_line`, `ParserState.parse_noise`, `ParserState.parse_noise`, `ParserState.frequency_mult`, `ParserState.parse_port`, +15 more |
| `skrf/vi/vna/hp/hp8510c_sweep_plan.py` | 30 | 31 | 8 | `SweepSection.get_hz`, `SweepSection.get_raw_hz`, `SweepSection.apply_8510`, `SweepSection.mask_8510`, `LinearBuiltinSweepSection.get_hz`, `LinearBuiltinSweepSection.apply_8510`, `LinearMaskedSweepSection.get_hz`, `LinearMaskedSweepSection.get_raw_hz`, +15 more |
| `skrf/io/tests/test_io.py` | 9 | 23 | 0 | `IOTestCase`, `IOTestCase.setUp`, `IOTestCase.read_write`, `IOTestCase.test_read_all`, `IOTestCase.test_read_all_files`, `IOTestCase.test_save_sesh`, `IOTestCase.test_write_all_dict`, `IOTestCase.test_readwrite_network`, +15 more |
| `skrf/io/tests/test_ts_spec.py` | 79 | 19 | 0 | `test_ex_1`, `test_ex_2`, `test_ex_2_write`, `test_ex_3`, `test_ex_4`, `test_ts_example_5_6`, `test_ts_example_7`, `test_example_8`, +11 more |
| `skrf/calibration/tests/test_deembedding.py` | 8 | 19 | 0 | `DeembeddingTestCase`, `DeembeddingTestCase.setUp`, `DeembeddingTestCase.test_freqmismatch`, `DeembeddingTestCase.test_openshort`, `DeembeddingTestCase.test_open`, `DeembeddingTestCase.test_shortopen`, `DeembeddingTestCase.test_short`, `DeembeddingTestCase.test_splitpi`, +11 more |
| `skrf/media/rectangularWaveguide.py` | 10 | 20 | 2 | `RectangularWaveguide.__init__`, `RectangularWaveguide.__repr__`, `RectangularWaveguide.from_z0`, `RectangularWaveguide.ep`, `RectangularWaveguide.mu`, `RectangularWaveguide.k0`, `RectangularWaveguide.ky`, `RectangularWaveguide.kx`, +10 more |
| `skrf/media/circularWaveguide.py` | 8 | 18 | 2 | `CircularWaveguide.__init__`, `CircularWaveguide.__repr__`, `CircularWaveguide.from_z0`, `CircularWaveguide.ep`, `CircularWaveguide.mu`, `CircularWaveguide.k0`, `CircularWaveguide.kc`, `CircularWaveguide.f_cutoff`, +8 more |
| `skrf/io/csv.py` | 65 | 24 | 9 | `pna_csv_2_df`, `pna_csv_2_ntwks2`, `pna_csv_2_ntwks3`, `AgilentCSV.__init__`, `AgilentCSV.n_traces`, `AgilentCSV.dict`, `AgilentCSV.dataframe`, `pna_csv_header_split`, +7 more |
| `skrf/vi/vna/keysight/tests/test_pna.py` | 62 | 15 | 0 | `mocked_ff`, `test_params`, `test_frequency_query`, `test_frequency_write`, `test_active_channel_query`, `test_active_channel_setter`, `test_query_fmt_query`, `test_query_fmt_write`, +7 more |
| `skrf/media/device.py` | 26 | 21 | 6 | `Device.__init__`, `Device.ntwk`, `MatchedSymmetricCoupler.__init__`, `MatchedSymmetricCoupler.from_dbdeg`, `MatchedSymmetricCoupler.c`, `MatchedSymmetricCoupler.c`, `MatchedSymmetricCoupler.t`, `MatchedSymmetricCoupler.t`, +7 more |
| `skrf/tlineFunctions.py` | 77 | 17 | 3 | `distributed_circuit_2_propagation_impedance`, `propagation_impedance_2_distributed_circuit`, `electrical_length_2_distance`, `load_impedance_2_reflection_coefficient`, `reflection_coefficient_2_input_impedance`, `reflection_coefficient_at_theta`, `input_impedance_at_theta`, `load_impedance_2_reflection_coefficient_at_theta`, +6 more |
| `skrf/vi/vna/rohde_schwarz/tests/test_zna.py` | 58 | 14 | 0 | `mocked_ff`, `test_params`, `test_frequency_query`, `test_frequency_write`, `test_active_channel_query`, `test_active_channel_setter`, `test_query_fmt_query`, `test_query_fmt_write`, +6 more |
| `skrf/vi/vna/rohde_schwarz/tests/test_zva.py` | 58 | 14 | 0 | `mocked_ff`, `test_params`, `test_frequency_query`, `test_frequency_write`, `test_active_channel_query`, `test_active_channel_setter`, `test_query_fmt_query`, `test_query_fmt_write`, +6 more |
| `skrf/media/mline.py` | 27 | 16 | 2 | `MLine.__init__`, `MLine.__repr__`, `MLine.gamma`, `MLine.z0_characteristic`, `MLine.Z0_f`, `MLine.analyse_dielectric`, `MLine.analyse_quasi_static`, `MLine.analyse_dispersion`, +6 more |
| `skrf/tests/test_tlineFunctions.py` | 14 | 14 | 0 | `TestBasicTransmissionLine`, `TestBasicTransmissionLine.setUp`, `TestBasicTransmissionLine.test_input_reflection_coefficient`, `TestBasicTransmissionLine.test_propagation_constant_from_reflection_coefficient`, `ElectricalLengthTests`, `ElectricalLengthTests.setUp`, `ElectricalLengthTests.gamma_from_f`, `ElectricalLengthTests.test_electrical_length_from_length`, +6 more |
| `skrf/tests/test_convenience.py` | 7 | 14 | 0 | `ConvenienceTestCase`, `ConvenienceTestCase.setUp`, `ConvenienceTestCase.test_hfss_high_port_number`, `ConvenienceTestCase.test_hfss_touchstone_2_media`, `ConvenienceTestCase.test_hfss_touchstone_renormalization`, `ConvenienceTestCase.test_is_hfss_touchstone`, `ConvenienceTestCase.test_hfss_touchstone_2_network`, `ConvenienceTestCase.test_cst_touchstone_2_network`, +6 more |
| `skrf/qfactor.py` | 15 | 21 | 8 | `OptimizedResult.__getattr__`, `OptimizedResult.__repr__`, `OptimizedResult.__dir__`, `Qfactor.__init__`, `Qfactor.__repr__`, `Qfactor._initial_fit`, `Qfactor._optimise_fit6`, `Qfactor._optimise_fit7`, +5 more |
| `skrf/tests/test_qfactor.py` | 8 | 13 | 0 | `QfactorTests`, `QfactorTests.setUp`, `QfactorTests.csv_file_example_to_network`, `QfactorTests.test_constructor`, `QfactorTests.test_exceptions`, `QfactorTests.test_NLQFIT6`, `QfactorTests.test_NLQFIT6_2`, `QfactorTests.test_NLQFIT7`, +5 more |
| `skrf/io/general.py` | 77 | 18 | 6 | `_get_extension`, `read_all`, `write_all`, `save_sesh`, `load_all_touchstones`, `write_dict_of_networks`, `read_csv`, `statistical_2_touchstone`, +4 more |
| `skrf/media/freespace.py` | 10 | 15 | 3 | `Freespace.__init__`, `Freespace.__repr__`, `Freespace.ep`, `Freespace.mu`, `Freespace.rho`, `Freespace.rho`, `Freespace.ep_with_rho`, `Freespace.gamma`, +4 more |
| `skrf/io/tests/test_touchstone.py` | 7 | 12 | 0 | `TouchstoneTestCase`, `TouchstoneTestCase.setUp`, `TouchstoneTestCase.test_read_data`, `TouchstoneTestCase.test_double_option_line`, `TouchstoneTestCase.test_read_with_special_encoding`, `TouchstoneTestCase.test_read_from_fid`, `TouchstoneTestCase.test_get_sparameter_data`, `TouchstoneTestCase.test_HFSS_touchstone_files`, +4 more |
| `skrf/tests/test_vectorfitting.py` | 6 | 12 | 0 | `VectorFittingTestCase`, `VectorFittingTestCase.test_ringslot_with_proportional`, `VectorFittingTestCase.test_ringslot_default_log`, `VectorFittingTestCase.test_ringslot_without_prop_const`, `VectorFittingTestCase.test_ringslot_custompoles`, `VectorFittingTestCase.test_190ghz_measured`, `VectorFittingTestCase.test_no_convergence`, `VectorFittingTestCase.test_dc_enforcement`, +4 more |
| `skrf/taper.py` | 23 | 19 | 8 | `Taper1D.__init__`, `Taper1D.media_at`, `Taper1D.medias`, `Linear`, `Linear.__init__`, `Exponential`, `Exponential.__init__`, `SmoothStep`, +3 more |
| `skrf/vi/vna/nanovna/nanovna.py` | 16 | 24 | 13 | `REG_ADDR`, `NanoVNAv2.__init__`, `NanoVNAv2.freq_start`, `NanoVNAv2.freq_start`, `NanoVNAv2.freq_stop`, `NanoVNAv2.freq_stop`, `NanoVNAv2.freq_step`, `NanoVNAv2.freq_step`, +3 more |
| `skrf/vi/vna/hp/hp8510c.py` | 10 | 37 | 26 | `HP8510C.__init__`, `HP8510C.get_switch_terms`, `HP8510C.freq_start`, `HP8510C.freq_start`, `HP8510C.freq_stop`, `HP8510C.freq_stop`, `HP8510C._npoints`, `HP8510C.npoints`, +3 more |
| `skrf/io/tests/test_citi.py` | 9 | 11 | 0 | `CitiTestCase`, `CitiTestCase.setUp`, `CitiTestCase.test_to_networks`, `CitiTestCase.test_to_networkset`, `CitiTestCase.test_params`, `CitiTestCase.test_only_freq_in_var`, `CitiTestCase.test_values_1p_1`, `CitiTestCase.test_values_1p_2`, +3 more |
| `skrf/vi/vna/hp/hp8720b.py` | 7 | 36 | 25 | `HP8720B.__init__`, `HP8720B.get_switch_terms`, `HP8720B.freq_start`, `HP8720B.freq_start`, `HP8720B.freq_stop`, `HP8720B.freq_stop`, `HP8720B._npoints`, `HP8720B.npoints`, +3 more |
| `skrf/vi/vna/keysight/tests/test_fieldfox.py` | 42 | 10 | 0 | `mocked_ff`, `test_params`, `test_freq_query`, `test_freq_write`, `test_query_fmt_query`, `test_query_fmt_write`, `test_define_msmnt`, `test_get_measurement_parameter`, +2 more |
| `skrf/media/coaxial.py` | 10 | 12 | 2 | `Coaxial.__init__`, `Coaxial.from_attenuation_VF`, `Coaxial.from_Z0_Dout`, `Coaxial.Rs`, `Coaxial.a`, `Coaxial.b`, `Coaxial.R`, `Coaxial.L`, +2 more |
| `skrf/tests/test_util.py` | 10 | 10 | 0 | `HomoDictTest`, `HomoDictTest.setUp`, `HomoDictTest.test_get_item`, `HomoDictTest.test_call`, `HomoDictTest.test_boolean_mask`, `HomoListTest`, `HomoListTest.setUp`, `HomoListTest.test_get_item`, +2 more |
| `skrf/media/definedAEpTandZ0.py` | 8 | 12 | 2 | `DefinedAEpTandZ0.__init__`, `DefinedAEpTandZ0.__repr__`, `DefinedAEpTandZ0.ep_r_f`, `DefinedAEpTandZ0.tand_f`, `DefinedAEpTandZ0.alpha_conductor`, `DefinedAEpTandZ0.alpha_dielectric`, `DefinedAEpTandZ0.beta_phase`, `DefinedAEpTandZ0.gamma`, +2 more |
| `skrf/tests/test_frequency.py` | 8 | 10 | 0 | `FrequencyTestCase`, `FrequencyTestCase.setUp`, `FrequencyTestCase.test_create_linear_sweep`, `FrequencyTestCase.test_create_log_sweep`, `FrequencyTestCase.test_create_rando_sweep`, `FrequencyTestCase.test_rando_sweep_from_touchstone`, `FrequencyTestCase.test_slicer`, `FrequencyTestCase.test_frequency_check`, +2 more |
| `skrf/calibration/tests/test_calibrationSet.py` | 21 | 9 | 0 | `CalsetTest`, `CalsetTest.test_run`, `CalsetTest.test_correct_ntwk`, `DotOneport`, `DotOneport.setUp`, `DotOneport.measure`, `DotEightTerm`, `DotEightTerm.setUp`, +1 more |
| `skrf/vi/vna/vna.py` | 19 | 18 | 9 | `_format_cmd`, `Channel.__init__`, `VNA.__init__`, `VNA.__init_subclass__`, `VNA._add_channel_support`, `VNA._setup_scpi`, `VNA.timeout`, `VNA.timeout`, +1 more |
| `skrf/media/cpw.py` | 12 | 11 | 2 | `CPW.__init__`, `CPW.__repr__`, `CPW.z0_characteristic`, `CPW.gamma`, `CPW.analyse_dielectric`, `CPW.analyse_quasi_static`, `CPW.analyse_dispersion`, `CPW.analyse_loss`, +1 more |
| `skrf/io/tests/test_mdif.py` | 9 | 9 | 0 | `MdifTestCase`, `MdifTestCase.setUp`, `MdifTestCase.test_equal`, `MdifTestCase.test_to_networkset`, `MdifTestCase.test_params`, `MdifTestCase.test_to_to_networkset_params`, `MdifTestCase.test_to_networkset_values`, `MdifTestCase.test_comment_after_BEGIN`, +1 more |
| `skrf/media/tests/test_cpw.py` | 9 | 9 | 0 | `CPWTestCase`, `CPWTestCase.setUp`, `CPWTestCase.test_qucs_network`, `CPWTestCase.test_ads_network`, `CPWTestCase.test_z0`, `CPWTestCase.test_ep_reff`, `CPWTestCase.test_z0_vs_f`, `CPWTestCase.test_alpha_warning`, +1 more |
| `skrf/io/tests/test_csv.py` | 7 | 9 | 0 | `AgilentCSVTestCase`, `AgilentCSVTestCase.setUp`, `AgilentCSVTestCase.test_columns`, `AgilentCSVTestCase.test_comments`, `AgilentCSVTestCase.test_data`, `AgilentCSVTestCase.test_frequency`, `AgilentCSVTestCase.test_networks`, `AgilentCSVTestCase.test_scalar_networks`, +1 more |
| `skrf/vi/validators.py` | 47 | 34 | 26 | `Validator`, `IntValidator.__init__`, `FloatValidator.__init__`, `EnumValidator.__init__`, `SetValidator.__init__`, `DictValidator.__init__`, `DelimitedStrValidator.__init__`, `BooleanValidator.__init__` |
| `skrf/vi/tests/test_validators.py` | 36 | 8 | 0 | `test_int_validator`, `test_int_validator_out_of_bounds`, `test_float_validator`, `test_float_validator_out_of_bounds`, `test_freq_validator`, `test_enum_validator`, `test_set_validator`, `test_dict_validator` |
| `skrf/calibration/calibrationSet.py` | 22 | 12 | 5 | `CalibrationSet`, `CalibrationSet.__init__`, `CalibrationSet.__getitem__`, `CalibrationSet.apply_cal`, `CalibrationSet.plot_uncertainty_per_standard`, `CalibrationSet.dankness`, `Dot` |
| `skrf/io/mdif.py` | 12 | 12 | 5 | `Mdif.__init__`, `Mdif.params`, `Mdif.comments`, `Mdif._parse_comments`, `Mdif._parse_data`, `Mdif._parse_mdif`, `Mdif.__create_optionstring` |
| `skrf/media/distributedCircuit.py` | 8 | 11 | 4 | `DistributedCircuit.__init__`, `DistributedCircuit._format_param`, `DistributedCircuit.__repr__`, `DistributedCircuit.Z`, `DistributedCircuit.Y`, `DistributedCircuit.z0_characteristic`, `DistributedCircuit.gamma` |
| `skrf/media/tests/test_coaxial.py` | 8 | 7 | 0 | `MediaTestCase`, `MediaTestCase.setUp`, `MediaTestCase.test_line`, `MediaTestCase.test_init_from_attenuation_VF_units`, `MediaTestCase.test_init_from_attenuation_VF_array_att`, `MediaTestCase.test_R`, `MediaTestCase.test_LC` |
| `skrf/vi/vna/hp/tests/test_8510c.py` | 26 | 6 | 0 | `mocked_ff`, `test_params`, `test_freq_query`, `test_freq_write`, `test_reset`, `test_wait_until_finished` |
| `skrf/vi/vna/keysight/pna.py` | 25 | 35 | 29 | `PNA.Channel.__init__`, `PNA.Channel._on_delete`, `PNA.Channel.freq_step`, `PNA.Channel.freq_step`, `PNA.__init__`, `PNA._model_param` |
| `skrf/vi/vna/hp/tests/test_8510c_sweep_plan.py` | 24 | 6 | 0 | `test_800pt_swp`, `test_801pt_swp`, `test_802pt_swp`, `test_1001pt_swp`, `test_multi_swp`, `test_multi_swp_with_single` |
| `skrf/vi/vna/rohde_schwarz/rs_vna.py` | 23 | 33 | 27 | `RSVNA`, `RSVNA.Channel.__init__`, `RSVNA.Channel._on_delete`, `RSVNA.Channel.create_sparam_group`, `RSVNA.__init__`, `RSVNA._model_param` |
| `skrf/media/tests/test_mline.py` | 9 | 6 | 0 | `MLineTestCase`, `MLineTestCase.setUp`, `MLineTestCase.test_z0_ep_reff`, `MLineTestCase.test_line_qucs`, `MLineTestCase.test_line_ads`, `MLineTestCase.test_alpha_warning` |
| `skrf/media/tests/test_rectangularWaveguide.py` | 9 | 5 | 0 | `MediaTestCase`, `MediaTestCase.setUp`, `MediaTestCase.test_line`, `MediaTestCase.test_conductor_loss`, `MediaTestCase.test_roughness` |
| `skrf/media/tests/test_definedaeptandz0.py` | 7 | 5 | 0 | `DefinedAEpTandZ0TestCase`, `DefinedAEpTandZ0TestCase.setUp`, `DefinedAEpTandZ0TestCase.test_line_awr`, `DefinedAEpTandZ0TestCase.test_nominal_impedance_dispersion`, `DefinedAEpTandZ0TestCase.test_raw_z0_array` |
| `skrf/media/tests/test_distributedCircuit.py` | 7 | 5 | 0 | `MediaTestCase`, `MediaTestCase.setUp`, `MediaTestCase.test_constructor`, `MediaTestCase.test_line`, `MediaTestCase.test_write_csv` |
| `skrf/tests/test_static_data.py` | 18 | 4 | 0 | `test_static_data`, `test_static_airs`, `test_static_frequencies`, `test_static_waveguides` |
| `skrf/vi/vna/keysight/fieldfox.py` | 13 | 16 | 12 | `FieldFox.__init__`, `FieldFox.freq_step`, `FieldFox.freq_step`, `FieldFox.get_measurement_parameter` |
| `skrf/io/citi.py` | 12 | 8 | 4 | `Citi.__init__`, `Citi.comments`, `Citi.params`, `Citi._parse_citi` |
| `skrf/notebook/bokeh_.py` | 14 | 3 | 0 | `plot_rectangular`, `plot_polar`, `use_bokeh` |
| `skrf/vi/tests/test_vna.py` | 12 | 3 | 0 | `test_format_cmd`, `test_vna_add_channel_support`, `test_vna_create_delete_channels` |
| `skrf/constants.py` | 38 | 3 | 1 | `get_distance_dict`, `__getattr__` |
| `skrf/time.py` | 24 | 5 | 3 | `indexes`, `get_window` |
| `skrf/data/__init__.py` | 14 | 20 | 18 | `StaticData.one_port_cal`, `__getattr__` |
| `skrf/tests/conftest.py` | 14 | 3 | 1 | `ntwk1_dc`, `ntwk_set_zip` |
| `skrf/tests/test_init.py` | 8 | 2 | 0 | `_modules_loaded_by`, `test_no_heavy_modules_on_import` |
| `skrf/io/metas.py` | 8 | 1 | 0 | `ns_2_sdatcv` |
| `skrf/vi/scpi_errors.py` | 5 | 2 | 1 | `SCPIError.__init__` |
| `skrf/__init__.py` | 85 | 1 | 1 |  |
| `skrf/media/__init__.py` | 29 | 0 | 0 |  |
| `skrf/io/__init__.py` | 17 | 0 | 0 |  |
| `skrf/vi/vna/__init__.py` | 11 | 0 | 0 |  |
| `skrf/calibration/__init__.py` | 8 | 0 | 0 |  |
| `skrf/vi/__init__.py` | 8 | 0 | 0 |  |
| `skrf/vi/vna/rohde_schwarz/zna.py` | 6 | 1 | 1 |  |
| `skrf/vi/vna/rohde_schwarz/zva.py` | 6 | 1 | 1 |  |
| `skrf/notebook/utils.py` | 5 | 1 | 1 |  |
| `skrf/programs/plot_touchstone.py` | 5 | 1 | 1 |  |
| `skrf/vi/vna/rohde_schwarz/__init__.py` | 4 | 0 | 0 |  |
| `skrf/vi/vna/hp/__init__.py` | 3 | 0 | 0 |  |
| `skrf/vi/vna/keysight/__init__.py` | 3 | 0 | 0 |  |
| `skrf/vi/vna/nanovna/__init__.py` | 2 | 0 | 0 |  |
| `apps/__init__.py` | 0 | 0 | 0 |  |
| `skrf/notebook/__init__.py` | 0 | 0 | 0 |  |
| `skrf/notebook/matplotlib_.py` | 0 | 0 | 0 |  |
