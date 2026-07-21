.. _contributing:

Contributing to rust-rf
-----------------------

rust-rf follows the standard fork-and-pull-request workflow. Keep ports aligned
with the upstream scikit-rf behavior, add focused Rust tests, and document the
Python source symbol when implementing an equivalent API.

Before opening a pull request, run:

.. code-block:: console

   cargo fmt --check
   cargo test --all-features
   cargo clippy --all-targets --all-features -- -D warnings
   cargo doc --all-features

Documentation notebooks use the evcxr ``rust`` kernel. Do not commit execution
outputs; the Sphinx build executes notebooks from a clean state.

.. code-block:: console

   python -m pip install -r doc/requirements.txt
   evcxr_jupyter --install
   python -m sphinx -W --keep-going -b html doc/source doc/build/html
