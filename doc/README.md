# rust-rf documentation

This directory is the Rust port of the scikit-rf documentation corpus. It keeps
the upstream figures and RF data sets while using rust-rf APIs and evcxr Rust
notebooks.

## Prerequisites

```text
python -m pip install -r requirements.txt
evcxr_jupyter --install
```

The second command registers the already-installed evcxr kernel with Jupyter.

## Build

```text
python -m sphinx -W --keep-going -b html source build/html
cargo doc --all-features
```

Notebook outputs are intentionally not stored in Git. Sphinx executes them with
the `rust` kernel while building.
