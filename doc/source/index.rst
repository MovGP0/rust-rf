.. _home:

rust-rf
-------

**rust-rf** is a BSD-licensed Rust port of scikit-rf for RF and microwave
engineering. It provides typed frequency axes, networks, calibration,
de-embedding, media, circuit solving, vector fitting, instrument support, and
optional plotting/data integrations.

The port preserves scikit-rf's domain model while adapting Python conventions to
Rust: methods use ``snake_case``, array data uses ``ndarray``, operations return
``Result`` instead of raising exceptions, and optional integrations are Cargo
features.

.. toctree::
   :maxdepth: 2

   tutorials/index
   examples/index
   api/index
   notebook-porting
   contributing/index
   citations_ack/index
   glossary
   license

The authoritative item-level reference is built directly from the crate:

.. code-block:: console

   cargo doc --all-features --open
