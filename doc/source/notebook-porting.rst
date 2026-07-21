Notebook porting inventory
--------------------------

All 64 copied notebooks use the evcxr ``rust`` kernel. Their Python
code cells were replaced with executable Rust workflows selected by subject. The
``rust_rf`` metadata block in each notebook records the upstream path, subject
theme, and original Python code-cell count.

Specialized upstream workflows that depend on live instruments or not-yet-ported
algorithm details are represented by the closest supported typed API. Consult
``PORTING_MAP.md`` in the repository root before extending one of those examples.

.. list-table:: Converted notebooks
   :header-rows: 1
   :widths: 72 28

   * - Notebook
     - Rust theme
   * - ``examples/circuit/Lumped Element Circuits.ipynb``
     - ``circuit``
   * - ``examples/circuit/Voltages and Currents in Circuits.ipynb``
     - ``circuit``
   * - ``examples/circuit/Wilkinson Power Splitter.ipynb``
     - ``circuit``
   * - ``examples/instrumentcontrol/VNA Control.ipynb``
     - ``instrument``
   * - ``examples/interactive/Interact Transmission Lines.ipynb``
     - ``media``
   * - ``examples/interactive/Interactive Coaxial Properties.ipynb``
     - ``media``
   * - ``examples/interactive/Interactive Mismatched Line.ipynb``
     - ``frequency``
   * - ``examples/matching/Impedance Matching.ipynb``
     - ``media``
   * - ``examples/metrology/Calibration With Three Receivers.ipynb``
     - ``calibration``
   * - ``examples/metrology/LRRM.ipynb``
     - ``calibration``
   * - ``examples/metrology/Measuring a Mutiport Device with a 2-Port Network Analyzer.ipynb``
     - ``calibration``
   * - ``examples/metrology/Multi-port Calibration.ipynb``
     - ``calibration``
   * - ``examples/metrology/Multiline TRL.ipynb``
     - ``calibration``
   * - ``examples/metrology/NanoVNA_V2_4port-splitter.ipynb``
     - ``calibration``
   * - ``examples/metrology/One Port Tiered Calibration.ipynb``
     - ``calibration``
   * - ``examples/metrology/SOLT Calibration Standards Creation.ipynb``
     - ``calibration``
   * - ``examples/metrology/SOLT.ipynb``
     - ``calibration``
   * - ``examples/metrology/TRL.ipynb``
     - ``calibration``
   * - ``examples/metrology/TwoPortOnePath, EnhancedResponse, and FakeFlip.ipynb``
     - ``calibration``
   * - ``examples/mixedmodeanalysis/Mixed Mode Basics.ipynb``
     - ``mixed_mode``
   * - ``examples/mixedmodeanalysis/Mixed Mode S and Impedance Transformation.ipynb``
     - ``mixed_mode``
   * - ``examples/networks/Compute Error Between S Parameter Matrices.ipynb``
     - ``network``
   * - ``examples/networks/Export Network as Touchstone File.ipynb``
     - ``io``
   * - ``examples/networksets/Export Network Set as MDIF File.ipynb``
     - ``network_set``
   * - ``examples/networksets/Interpolating Network Sets.ipynb``
     - ``network_set``
   * - ``examples/networksets/Sorting Network Sets.ipynb``
     - ``network_set``
   * - ``examples/networktheory/Balanced Network De-embedding.ipynb``
     - ``deembedding``
   * - ``examples/networktheory/Balun_Transformer_Designs.ipynb``
     - ``circuit``
   * - ``examples/networktheory/Correlating microstripline model to measurement.ipynb``
     - ``media``
   * - ``examples/networktheory/CPW media example.ipynb``
     - ``media``
   * - ``examples/networktheory/DC Extrapolation for Time Domain .ipynb``
     - ``network``
   * - ``examples/networktheory/DefinedAEpTandZ0 media example.ipynb``
     - ``media``
   * - ``examples/networktheory/IEEEP370 Deembedding.ipynb``
     - ``deembedding``
   * - ``examples/networktheory/LNA Example.ipynb``
     - ``network``
   * - ``examples/networktheory/Properties of Rectangular Waveguides.ipynb``
     - ``media``
   * - ``examples/networktheory/Renormalizing S-parameters.ipynb``
     - ``network``
   * - ``examples/networktheory/Time domain reflectometry, measurement vs simulation.ipynb``
     - ``network``
   * - ``examples/networktheory/Time Domain.ipynb``
     - ``network``
   * - ``examples/networktheory/Transmission Line Losses.ipynb``
     - ``media``
   * - ``examples/networktheory/Transmission Line Properties and Manipulations.ipynb``
     - ``connect``
   * - ``examples/networktheory/Transmission Lines and SWR.ipynb``
     - ``connect``
   * - ``examples/networktheory/Working with Complex Characteristic Impedances.ipynb``
     - ``network``
   * - ``examples/plotting/Modeling RG-58.ipynb``
     - ``plot``
   * - ``examples/plotting/XKCD styling.ipynb``
     - ``plot``
   * - ``examples/qfactor/Finding_Dk_Df_from_resonance_fitting.ipynb``
     - ``qfactor``
   * - ``examples/taper/tapered_transmission_lines.ipynb``
     - ``media``
   * - ``examples/vectorfitting/vectorfitting_ex1_ringslot.ipynb``
     - ``vector_fitting``
   * - ``examples/vectorfitting/vectorfitting_ex2_190ghz_active.ipynb``
     - ``vector_fitting``
   * - ``examples/vectorfitting/vectorfitting_ex3_Agilent_E5071B.ipynb``
     - ``vector_fitting``
   * - ``examples/vectorfitting/vectorfitting_ex4_passivity.ipynb``
     - ``vector_fitting``
   * - ``examples/vectorfitting/vectorfitting_problems.ipynb``
     - ``vector_fitting``
   * - ``tutorials/Calibration.ipynb``
     - ``calibration``
   * - ``tutorials/Circuit.ipynb``
     - ``circuit``
   * - ``tutorials/Connecting_Networks.ipynb``
     - ``connect``
   * - ``tutorials/Deembedding.ipynb``
     - ``deembedding``
   * - ``tutorials/Installation.ipynb``
     - ``installation``
   * - ``tutorials/Introduction.ipynb``
     - ``network``
   * - ``tutorials/Media.ipynb``
     - ``media``
   * - ``tutorials/Networks.ipynb``
     - ``network``
   * - ``tutorials/NetworkSet.ipynb``
     - ``network_set``
   * - ``tutorials/Plotting.ipynb``
     - ``plot``
   * - ``tutorials/Q-Factor.ipynb``
     - ``qfactor``
   * - ``tutorials/VectorFitting.ipynb``
     - ``vector_fitting``
   * - ``tutorials/VirtualInstruments.ipynb``
     - ``instrument``
