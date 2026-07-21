# rust-rf documentation build configuration

project = "rust-rf"
copyright = "2026, rust-rf contributors and scikit-rf contributors"
author = "rust-rf contributors"

extensions = [
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.mathjax",
    "nbsphinx",
]

autosectionlabel_prefix_document = True
nbsphinx_execute = "always"
nbsphinx_allow_errors = False
nbsphinx_kernel_name = "rust"
nbsphinx_timeout = 300

templates_path = []
exclude_patterns = ["_build", "_templates", "Thumbs.db", ".DS_Store", "**/.ipynb_checkpoints"]
pygments_style = "sphinx"

html_theme = "sphinx_rtd_theme"
html_title = "rust-rf Documentation"
html_short_title = "rust-rf"
html_static_path = ["_static"]
htmlhelp_basename = "rustrfdoc"

latex_documents = [
    ("index", "rust-rf.tex", "rust-rf Documentation", "rust-rf contributors", "manual"),
]
man_pages = [("index", "rust-rf", "rust-rf Documentation", ["rust-rf contributors"], 1)]
