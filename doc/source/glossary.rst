.. _glossary:

Glossary
--------

.. glossary::

   frequency axis
      A typed :class:`Frequency` sweep stored internally in hertz, with an
      explicit display unit and sweep classification.

   network
      A frequency-dependent microwave N-port represented by S-parameters and a
      reference impedance for every port and frequency point.

   port impedance
      The reference impedance associated with a network port. Terminating a
      port with a different impedance generally creates a reflection.

   characteristic impedance
      The voltage-to-current ratio for a single traveling wave along a
      transmission-line mode.

   switch terms
      Corrections for the change in a VNA's error networks when its internal
      source switch changes the excited port.

   Cargo feature
      A compile-time switch for optional rust-rf integrations. The crate uses
      ``plot``, ``visa``, ``xlsx``, ``dataframe``, and the aggregate ``full``
      feature.

   evcxr
      A Rust evaluation context used by the ``evcxr_jupyter`` kernel to compile
      and run Rust cells in Jupyter notebooks.
