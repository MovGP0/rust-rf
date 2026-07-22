//! Documentation tooling originating from `doc/source/conf.py` and
//! `doc/sphinxext/docscrape.py` as exercised by `doc/sphinxext/tests/test_docscrape.py`.

use std::collections::BTreeMap;

/// Stable documentation-build settings translated from `doc/source/conf.py`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DocumentationConfig {
    /// Project name displayed by generated documentation.
    pub project: String,
    /// Project author displayed by generated documentation.
    pub author: String,
    /// HTML page title.
    pub html_title: String,
    /// Documentation source-file suffix.
    pub source_suffix: String,
    /// Root document used to build the documentation tree.
    pub master_document: String,
    /// Enabled documentation extensions.
    pub extensions: Vec<String>,
    /// Source patterns excluded from documentation builds.
    pub excluded_patterns: Vec<String>,
    /// External documentation inventories keyed by project name.
    pub intersphinx: BTreeMap<String, String>,
}

impl DocumentationConfig {
    /// Typed equivalent of the stable, non-Python portions of `doc/source/conf.py`.
    pub fn scikit_rf() -> Self {
        Self {
            project: "scikit-rf".to_owned(),
            author: "scikit-rf team".to_owned(),
            html_title: "scikit-rf Documentation".to_owned(),
            source_suffix: ".rst".to_owned(),
            master_document: "index".to_owned(),
            extensions: [
                "sphinx.ext.autodoc",
                "sphinx.ext.autosectionlabel",
                "sphinx.ext.autosummary",
                "sphinx.ext.napoleon",
                "sphinx.ext.mathjax",
                "sphinx.ext.viewcode",
                "sphinx.ext.intersphinx",
                "sphinx_rtd_theme",
                "nbsphinx",
                "IPython.sphinxext.ipython_directive",
                "IPython.sphinxext.ipython_console_highlighting",
            ]
            .into_iter()
            .map(str::to_owned)
            .collect(),
            excluded_patterns: [
                "_build",
                "Thumbs.db",
                ".DS_Store",
                "**/*.rst.rst",
                "**.ipynb_checkpoints",
            ]
            .into_iter()
            .map(str::to_owned)
            .collect(),
            intersphinx: BTreeMap::from([
                (
                    "numpy".to_owned(),
                    "https://numpy.org/doc/stable".to_owned(),
                ),
                (
                    "pandas".to_owned(),
                    "https://pandas.pydata.org/docs".to_owned(),
                ),
                ("python".to_owned(), "https://docs.python.org/3".to_owned()),
                (
                    "scipy".to_owned(),
                    "https://docs.scipy.org/doc/scipy/".to_owned(),
                ),
            ]),
        }
    }
}

/// Removes the generated `self` and plotting-attribute arguments from generated plot signatures.
///
/// Signatures for ordinary functions, or plot functions not listed in `generated_plot_methods`,
/// are returned unchanged.
pub fn process_signature(
    qualified_name: &str,
    signature: &str,
    generated_plot_methods: &[&str],
) -> String {
    let function = qualified_name.rsplit('.').next().unwrap_or(qualified_name);
    let Some(property) = function.strip_prefix("plot_") else {
        return signature.to_owned();
    };
    if !generated_plot_methods.contains(&property) {
        return signature.to_owned();
    }
    let Some(arguments) = signature
        .strip_prefix('(')
        .and_then(|value| value.strip_suffix(')'))
    else {
        return signature.to_owned();
    };
    let retained = arguments
        .split(',')
        .map(str::trim)
        .skip(2)
        .collect::<Vec<_>>();
    format!("({})", retained.join(", "))
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// Named entry in a structured NumPy-style documentation section.
pub struct DocField {
    /// Field, parameter, return value, or exception name.
    pub name: String,
    /// Declared type or value description.
    pub field_type: String,
    /// Dedented Markdown description lines.
    pub description: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
/// Parsed representation of a NumPy-style docstring.
pub struct NumpyDocString {
    /// Optional callable signature from the first line.
    pub signature: Option<String>,
    /// Initial one-paragraph summary.
    pub summary: Vec<String>,
    /// Additional prose before the first named section.
    pub extended_summary: Vec<String>,
    /// Free-form named sections such as Notes or Examples.
    pub sections: BTreeMap<String, Vec<String>>,
    /// Structured sections such as Parameters, Returns, and Raises.
    pub fields: BTreeMap<String, Vec<DocField>>,
    /// Parsed entries from the reStructuredText `index` directive.
    pub index: BTreeMap<String, Vec<String>>,
}

impl NumpyDocString {
    /// Parses the section and field model used by the bundled `NumPy` docscrape extension.
    #[must_use]
    pub fn parse(input: &str) -> Self {
        let lines = dedent(input);
        let mut output = Self::default();
        let mut cursor = 0;
        while cursor < lines.len() && lines[cursor].trim().is_empty() {
            cursor += 1;
        }
        if cursor < lines.len() && looks_like_signature(lines[cursor].trim()) {
            output.signature = Some(lines[cursor].trim().to_owned());
            cursor += 1;
            skip_blank(&lines, &mut cursor);
        }
        output.summary = take_paragraph(&lines, &mut cursor);
        skip_blank(&lines, &mut cursor);
        while cursor < lines.len() && !is_section_header(&lines, cursor) {
            output.extended_summary.push(lines[cursor].clone());
            cursor += 1;
        }
        trim_blank_lines(&mut output.extended_summary);
        while cursor + 1 < lines.len() {
            if !is_section_header(&lines, cursor) {
                cursor += 1;
                continue;
            }
            let name = lines[cursor].trim().to_owned();
            cursor += 2;
            let start = cursor;
            while cursor < lines.len() && !is_section_header(&lines, cursor) {
                cursor += 1;
            }
            let mut body = lines[start..cursor].to_vec();
            trim_blank_lines(&mut body);
            if name == "Parameters"
                || name == "Other Parameters"
                || name == "Returns"
                || name == "Yields"
                || name == "Raises"
                || name == "Warns"
                || name == "Attributes"
                || name == "Methods"
            {
                output.fields.insert(name, parse_fields(&body));
            } else {
                if name == "index" {
                    parse_index(&body, &mut output.index);
                }
                output.sections.insert(name, body);
            }
        }
        if let Some(index_start) = lines
            .iter()
            .position(|line| line.trim_start().starts_with(".. index::"))
        {
            parse_index(&lines[index_start..], &mut output.index);
        }
        output
    }

    /// Returns the lines in a free-form named section, or an empty slice.
    pub fn section(&self, name: &str) -> &[String] {
        self.sections.get(name).map_or(&[], Vec::as_slice)
    }

    /// Returns the entries in a structured named section, or an empty slice.
    pub fn field_section(&self, name: &str) -> &[DocField] {
        self.fields.get(name).map_or(&[], Vec::as_slice)
    }
}

fn dedent(input: &str) -> Vec<String> {
    let raw = input.lines().collect::<Vec<_>>();
    let indentation = raw
        .iter()
        .filter(|line| !line.trim().is_empty())
        .map(|line| line.len() - line.trim_start().len())
        .min()
        .unwrap_or(0);
    raw.into_iter()
        .map(|line| line.get(indentation..).unwrap_or("").to_owned())
        .collect()
}

fn looks_like_signature(line: &str) -> bool {
    line.contains('(') && line.ends_with(')')
}

fn is_section_header(lines: &[String], index: usize) -> bool {
    index + 1 < lines.len()
        && !lines[index].trim().is_empty()
        && !lines[index + 1].trim().is_empty()
        && lines[index + 1]
            .trim()
            .chars()
            .all(|character| character == '-')
        && lines[index + 1].trim().len() >= 3
}

fn skip_blank(lines: &[String], cursor: &mut usize) {
    while *cursor < lines.len() && lines[*cursor].trim().is_empty() {
        *cursor += 1;
    }
}

fn take_paragraph(lines: &[String], cursor: &mut usize) -> Vec<String> {
    let mut output = Vec::new();
    while *cursor < lines.len() && !lines[*cursor].trim().is_empty() {
        output.push(lines[*cursor].trim().to_owned());
        *cursor += 1;
    }
    output
}

fn trim_blank_lines(lines: &mut Vec<String>) {
    while lines.first().is_some_and(|line| line.trim().is_empty()) {
        lines.remove(0);
    }
    while lines.last().is_some_and(|line| line.trim().is_empty()) {
        lines.pop();
    }
}

fn parse_fields(lines: &[String]) -> Vec<DocField> {
    let mut fields = Vec::new();
    let mut cursor = 0;
    while cursor < lines.len() {
        if lines[cursor].trim().is_empty() {
            cursor += 1;
            continue;
        }
        let header = lines[cursor].trim();
        let (name, field_type) = header
            .split_once(':')
            .map_or((header, ""), |(name, field_type)| {
                (name.trim(), field_type.trim())
            });
        cursor += 1;
        let mut description = Vec::new();
        while cursor < lines.len() {
            let line = &lines[cursor];
            if !line.trim().is_empty() && !line.starts_with(' ') && !line.starts_with('\t') {
                break;
            }
            description.push(line.strip_prefix("    ").unwrap_or(line).to_owned());
            cursor += 1;
        }
        trim_blank_lines(&mut description);
        fields.push(DocField {
            name: name.to_owned(),
            field_type: field_type.to_owned(),
            description,
        });
    }
    fields
}

fn parse_index(lines: &[String], index: &mut BTreeMap<String, Vec<String>>) {
    for line in lines {
        let trimmed = line.trim();
        if let Some(value) = trimmed.strip_prefix(".. index::") {
            index.insert("default".to_owned(), vec![value.trim().to_owned()]);
        } else if let Some(value) = trimmed.strip_prefix(':') {
            if let Some((key, values)) = value.split_once(':') {
                index.insert(
                    key.trim().to_owned(),
                    values
                        .split(',')
                        .map(|entry| entry.trim().to_owned())
                        .collect(),
                );
            }
        }
    }
}
