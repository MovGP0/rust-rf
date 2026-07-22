//! Backend-neutral plotting and optional SVG-rendering tests.

use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use rust_rf::NetworkSet;
use rust_rf::data::DATA;
use rust_rf::notebook::{
    BOKEH_NETWORK_METHODS, BokehPlotOptions, NOTEBOOK_COLORS, color, plot_polar, plot_rectangular,
    rectangular_plot, trace_color_cycle, use_bokeh,
};
use rust_rf::plotting::{
    Component, Parameter, animation_frames, complex_plot, contour_data, log_sigma_plot,
    minmax_bounds_plot, network_plot, passivity_plot, reciprocity_plot, reciprocity2_plot,
    shaded_bands, signature, smith_plot, time_domain_plot, uncertainty_bounds_plot,
    uncertainty_decomposition_plot, uncertainty_plot, vector_plot, violin_data,
};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("convenience")
        .join(name)
}

/// Checks rectangular, complex-plane, polar, and Smith plot series.
#[test]
fn builds_rectangular_complex_polar_and_smith_series() {
    let network = DATA.ntwk1().expect("network should load");
    for component in [
        Component::Decibels,
        Component::Decibels10,
        Component::Magnitude,
        Component::PhaseDegrees,
        Component::Real,
        Component::Imaginary,
        Component::Vswr,
    ] {
        let plot = network_plot(&network, Parameter::Scattering, component, None)
            .expect("plot data should build");
        assert_eq!(plot.series.len(), 4);
        assert_eq!(plot.x_label, "Frequency (Hz)");
    }
    assert_eq!(
        complex_plot(&network, Parameter::Impedance, Some((0, 0)), false)
            .expect("complex plot")
            .series
            .len(),
        1
    );
    assert_eq!(
        complex_plot(&network, Parameter::Admittance, None, true)
            .expect("polar plot")
            .series
            .len(),
        4
    );
    assert!(
        smith_plot(&network, None)
            .expect("Smith plot")
            .title
            .contains("Smith")
    );
}

/// Checks passivity, reciprocity, and network-set uncertainty traces.
#[test]
fn builds_network_metrics_and_uncertainty_series() {
    let network = DATA.ntwk1().expect("network should load");
    assert_eq!(
        passivity_plot(&network, None)
            .expect("passivity")
            .series
            .len(),
        2
    );
    assert_eq!(
        reciprocity_plot(&network, true)
            .expect("reciprocity")
            .series
            .len(),
        1
    );

    let set = NetworkSet::from_zip(fixture("ntwks.zip")).expect("network set should load");
    let uncertainty =
        uncertainty_plot(&set, Component::Decibels, 0, 0).expect("uncertainty plot should build");
    assert_eq!(uncertainty.series.len(), 3);
}

/// Checks notebook colors, generated method names, and Bokeh-compatible plot data.
#[test]
fn exposes_notebook_color_cycle_and_backend_neutral_plot() {
    let colors = trace_color_cycle(1).take(5).collect::<Vec<_>>();
    assert_eq!(
        colors,
        vec!["#FF0000", "#FF00FF", "#00AA00", "#0000FF", "#FF0000"]
    );
    assert_eq!(NOTEBOOK_COLORS[3], ("blue", "#0000FF"));
    assert_eq!(color("lime_green"), Some("#00FF00"));
    assert_eq!(color("missing"), None);
    assert_eq!(trace_color_cycle(999).collect::<Vec<_>>(), vec!["#00AA00"]);
    assert_eq!(trace_color_cycle(1_000).count(), 0);
    let network = DATA.ntwk1().expect("network");
    assert_eq!(
        rectangular_plot(&network, Component::Magnitude)
            .expect("notebook plot")
            .series
            .len(),
        4
    );
    assert_eq!(
        plot_rectangular(
            &network,
            BokehPlotOptions {
                parameter: Parameter::Impedance,
                component: Component::Real,
                ports: Some((0, 0)),
                show: false,
            },
        )
        .expect("configured notebook plot")
        .series
        .len(),
        1
    );
    assert_eq!(
        plot_polar(&network, Parameter::Scattering, Some((1, 0)))
            .expect("notebook polar plot")
            .series
            .len(),
        1
    );
    assert_eq!(use_bokeh(), BOKEH_NETWORK_METHODS);
}

/// Checks time-domain and advanced uncertainty, heatmap, violin, animation,
/// contour, band, and vector plot data.
#[test]
fn builds_advanced_backend_neutral_plot_data() {
    let network = DATA.ntwk1().expect("network should load");
    assert_eq!(
        reciprocity2_plot(&network, false)
            .expect("reciprocity #2")
            .series
            .len(),
        1
    );
    assert_eq!(
        time_domain_plot(&network, Component::Decibels, Some((0, 0)))
            .expect("time-domain plot")
            .series
            .len(),
        1
    );
    let set = NetworkSet::from_zip(fixture("ntwks.zip")).expect("network set should load");
    assert_eq!(
        uncertainty_bounds_plot(&set, Component::Magnitude, 0, 0, 3.0)
            .expect("uncertainty bounds")
            .series
            .len(),
        3
    );
    assert_eq!(
        minmax_bounds_plot(&set, Component::Decibels, 0, 0)
            .expect("min/max bounds")
            .series
            .len(),
        2
    );
    assert_eq!(
        uncertainty_decomposition_plot(&set, 0, 0)
            .expect("uncertainty decomposition")
            .series
            .len(),
        2
    );
    assert_eq!(
        log_sigma_plot(&set, 0, 0).expect("log sigma").series.len(),
        1
    );
    let heatmap = signature(&set, Component::Magnitude, 0, 0).expect("signature");
    assert_eq!(heatmap.values.len(), set.len());
    assert_eq!(
        violin_data(&set, Component::Real, 0, 0)
            .expect("violin distributions")
            .len(),
        set.networks[0].frequency_points()
    );
    assert_eq!(
        animation_frames(&set, Component::PhaseDegrees, Some((0, 0)))
            .expect("animation frames")
            .len(),
        set.len()
    );
    assert_eq!(
        shaded_bands(&[1.0, 2.0, 4.0], (-1.0, 1.0))
            .expect("shaded bands")
            .len(),
        2
    );
    assert_eq!(
        vector_plot(
            num_complex::Complex64::new(2.0, 3.0),
            num_complex::Complex64::new(1.0, 1.0)
        )
        .series[0]
            .x,
        vec![1.0, 3.0]
    );
    assert_eq!(
        contour_data(&[1.0, 2.0], &[3.0], &[vec![4.0, 5.0]], "contour")
            .expect("contour")
            .values,
        vec![vec![4.0, 5.0]]
    );
}

#[cfg(feature = "plot")]
/// Checks SVG rendering and the `plot-touchstone` command-line utility.
#[test]
fn renders_svg_and_runs_the_touchstone_plotter() {
    use std::process::Command;

    use rust_rf::plotting::render_svg;

    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock should follow the Unix epoch")
        .as_nanos();
    let temporary_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(".temp");
    let temporary = temporary_root.join(format!("plotting-{}-{unique}", std::process::id()));
    fs::create_dir_all(&temporary).expect("temporary directory should be created");
    let library_output = temporary.join("library-plot.svg");
    let cli_output = temporary.join("cli-plot.svg");
    let cli_phase_output = temporary.join("cli-plot-phase.svg");
    let cli_smith_output = temporary.join("cli-plot-smith.svg");
    let selected_output = temporary.join("selected.svg");
    let plot = network_plot(
        &DATA.ntwk1().expect("network"),
        Parameter::Scattering,
        Component::Decibels,
        None,
    )
    .expect("plot");
    render_svg(&plot, &library_output, (800, 600)).expect("SVG should render");
    assert!(fs::metadata(&library_output).expect("SVG metadata").len() > 1000);

    let status = Command::new(env!("CARGO_BIN_EXE_plot-touchstone"))
        .arg(fixture("ntwk1.s2p"))
        .arg(&cli_output)
        .status()
        .expect("plot-touchstone should run");
    assert!(status.success());
    assert!(fs::metadata(&cli_output).expect("CLI SVG metadata").len() > 1000);
    assert!(
        fs::metadata(&cli_phase_output)
            .expect("phase SVG metadata")
            .len()
            > 1000
    );
    assert!(
        fs::metadata(&cli_smith_output)
            .expect("Smith SVG metadata")
            .len()
            > 1000
    );

    let status = Command::new(env!("CARGO_BIN_EXE_plot-touchstone"))
        .args(["-m", "1", "-n", "2", "-o"])
        .arg(&selected_output)
        .arg(fixture("ntwk1.s2p"))
        .arg(fixture("ntwk1.s2p"))
        .status()
        .expect("plot-touchstone should accept port selection and multiple files");
    assert!(status.success());
    assert!(
        fs::metadata(&selected_output)
            .expect("selected SVG metadata")
            .len()
            > 1000
    );
    fs::remove_dir_all(temporary).expect("temporary plotting files should be removed");
    let _ = fs::remove_dir(temporary_root);
}
