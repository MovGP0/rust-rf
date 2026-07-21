use rust_rf::docs::{DocumentationConfig, NumpyDocString, process_signature};

const DOC: &str = r#"
  numpy.multivariate_normal(mean, cov, shape=None, spam=None)

  Draw values from a multivariate normal distribution with specified
  mean and covariance.

  The multivariate normal or Gaussian distribution is a generalisation
  of the one-dimensional normal distribution to higher dimensions.

  Parameters
  ----------
  mean : (N,) ndarray
      Mean of the N-dimensional distribution.

      .. math::

         (1+2+3)/3

  cov : (N,N) ndarray
      Covariance matrix of the distribution.
  shape : tuple of ints
      Given a shape of samples.

  Returns
  -------
  out : ndarray
      The drawn samples.

  Other Parameters
  ----------------
  spam : parrot
      A parrot off its mortal coil.

  Raises
  ------
  RuntimeError
      Some error

  Warns
  -----
  RuntimeWarning
      Some warning

  Notes
  -----
  Instead of specifying the full covariance matrix, approximations exist.

  References
  ----------
  .. [1] A reference.

  Examples
  --------
  >>> mean = (1,2)

  .. index:: random
     :refguide: random;distributions, random;gauss
"#;

#[test]
fn exposes_the_documentation_build_configuration() {
    let configuration = DocumentationConfig::scikit_rf();
    assert_eq!(configuration.project, "scikit-rf");
    assert_eq!(configuration.master_document, "index");
    assert!(
        configuration
            .extensions
            .contains(&"sphinx.ext.autodoc".to_owned())
    );
    assert_eq!(
        configuration.intersphinx["numpy"],
        "https://numpy.org/doc/stable"
    );
}

#[test]
fn strips_generated_plot_receiver_arguments_from_signatures() {
    assert_eq!(
        process_signature(
            "skrf.Network.plot_s_db",
            "(self, attribute, m=None, n=None)",
            &["s_db"]
        ),
        "(m=None, n=None)"
    );
    assert_eq!(
        process_signature("skrf.Network.crop", "(self, start, stop)", &["s_db"]),
        "(self, start, stop)"
    );
}

#[test]
fn parses_numpy_docstring_summaries_sections_fields_and_index() {
    let documentation = NumpyDocString::parse(DOC);
    assert_eq!(
        documentation.signature.as_deref(),
        Some("numpy.multivariate_normal(mean, cov, shape=None, spam=None)")
    );
    assert!(documentation.summary[0].starts_with("Draw values"));
    assert!(documentation.summary[1].ends_with("covariance."));
    assert!(documentation.extended_summary[0].starts_with("The multivariate normal"));
    let parameters = documentation.field_section("Parameters");
    assert_eq!(parameters.len(), 3);
    assert_eq!(parameters[0].name, "mean");
    assert_eq!(parameters[1].field_type, "(N,N) ndarray");
    assert!(parameters[1].description[0].starts_with("Covariance matrix"));
    assert_eq!(
        documentation.field_section("Other Parameters")[0].name,
        "spam"
    );
    assert_eq!(documentation.field_section("Returns")[0].name, "out");
    assert_eq!(
        documentation.field_section("Raises")[0].name,
        "RuntimeError"
    );
    assert_eq!(
        documentation.field_section("Warns")[0].name,
        "RuntimeWarning"
    );
    assert!(documentation.section("Notes")[0].starts_with("Instead"));
    assert!(documentation.section("References")[0].starts_with(".. [1]"));
    assert!(documentation.section("Examples")[0].starts_with(">>>"));
    assert_eq!(documentation.index["default"], vec!["random"]);
    assert_eq!(documentation.index["refguide"].len(), 2);
}

#[test]
fn parses_parameter_sections_without_extended_summary() {
    let documentation = NumpyDocString::parse(
        "Returns indices.\n\nParameters\n----------\na : {array_like}\n    Array to inspect.\n",
    );
    assert!(documentation.extended_summary.is_empty());
    assert_eq!(documentation.field_section("Parameters").len(), 1);
}
