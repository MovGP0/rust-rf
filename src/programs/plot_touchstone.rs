//! Rust port of `skrf/programs/plot_touchstone.py`.

use std::ffi::OsString;
use std::path::Path;
use std::path::PathBuf;

use rust_rf::Network;
use rust_rf::plotting::{Component, Parameter, Plot, network_plot, render_svg, smith_plot};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let options = Options::parse(std::env::args_os().skip(1))?;
    if options.help {
        println!("{}", Options::USAGE);
        return Ok(());
    }
    let networks = options
        .inputs
        .iter()
        .map(Network::read_touchstone)
        .collect::<rust_rf::Result<Vec<_>>>()?;
    let db = rectangular_plots(
        &networks,
        Component::Decibels,
        options.output_port,
        options.input_port,
    )?;
    let phase = rectangular_plots(
        &networks,
        Component::PhaseDegrees,
        options.output_port,
        options.input_port,
    )?;
    let smith = smith_plots(&networks, options.output_port, options.input_port)?;
    render_svg(&db, &options.output, (1_200, 800))?;
    render_svg(
        &phase,
        sibling_output(&options.output, "phase"),
        (1_200, 800),
    )?;
    render_svg(
        &smith,
        sibling_output(&options.output, "smith"),
        (1_200, 800),
    )?;
    Ok(())
}

#[derive(Debug)]
struct Options {
    inputs: Vec<PathBuf>,
    output: PathBuf,
    output_port: Option<usize>,
    input_port: Option<usize>,
    help: bool,
}

impl Options {
    const USAGE: &'static str =
        "usage: plot-touchstone [-m PORT] [-n PORT] [-o OUTPUT.svg] file.sNp [file2.sNp ...]";

    fn parse(
        arguments: impl Iterator<Item = OsString>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let arguments = arguments.collect::<Vec<_>>();
        let mut inputs = Vec::new();
        let mut output = None;
        let mut output_port = None;
        let mut input_port = None;
        let mut help = false;
        let mut index = 0;
        while index < arguments.len() {
            let argument = arguments[index].to_string_lossy();
            match argument.as_ref() {
                "-h" | "--help" => help = true,
                "-m" => {
                    index += 1;
                    output_port = Some(parse_port(arguments.get(index), "-m")?);
                }
                "-n" => {
                    index += 1;
                    input_port = Some(parse_port(arguments.get(index), "-n")?);
                }
                "-o" | "--output" => {
                    index += 1;
                    output = Some(PathBuf::from(
                        arguments
                            .get(index)
                            .ok_or("missing path after output option")?,
                    ));
                }
                value if value.starts_with('-') => {
                    return Err(format!("unknown option `{value}`\n{}", Self::USAGE).into());
                }
                _ => inputs.push(PathBuf::from(&arguments[index])),
            }
            index += 1;
        }
        if help {
            return Ok(Self {
                inputs,
                output: output.unwrap_or_else(|| PathBuf::from("touchstone.svg")),
                output_port,
                input_port,
                help,
            });
        }
        if output.is_none()
            && inputs.len() == 2
            && inputs[1]
                .extension()
                .is_some_and(|extension| extension.eq_ignore_ascii_case("svg"))
        {
            output = inputs.pop();
        }
        if inputs.is_empty() {
            return Err(Self::USAGE.into());
        }
        Ok(Self {
            inputs,
            output: output.unwrap_or_else(|| PathBuf::from("touchstone.svg")),
            output_port,
            input_port,
            help,
        })
    }
}

fn parse_port(value: Option<&OsString>, option: &str) -> Result<usize, Box<dyn std::error::Error>> {
    let value = value.ok_or_else(|| format!("missing one-based port number after {option}"))?;
    let port = value
        .to_string_lossy()
        .parse::<usize>()
        .map_err(|error| format!("invalid port number for {option}: {error}"))?;
    port.checked_sub(1)
        .ok_or_else(|| format!("port number for {option} must be at least 1").into())
}

fn rectangular_plots(
    networks: &[Network],
    component: Component,
    output_port: Option<usize>,
    input_port: Option<usize>,
) -> rust_rf::Result<Plot> {
    merge_plots(
        networks,
        |network, ports| network_plot(network, Parameter::Scattering, component, ports),
        output_port,
        input_port,
    )
}

fn smith_plots(
    networks: &[Network],
    output_port: Option<usize>,
    input_port: Option<usize>,
) -> rust_rf::Result<Plot> {
    merge_plots(networks, smith_plot, output_port, input_port)
}

fn merge_plots(
    networks: &[Network],
    mut plot: impl FnMut(&Network, Option<(usize, usize)>) -> rust_rf::Result<Plot>,
    output_port: Option<usize>,
    input_port: Option<usize>,
) -> rust_rf::Result<Plot> {
    let first = &networks[0];
    let pairs = selected_pairs(first.ports(), output_port, input_port);
    if pairs.is_empty() {
        return Err(rust_rf::Error::InvalidPort {
            port: output_port.or(input_port).unwrap_or_default(),
            ports: first.ports(),
        });
    }
    let mut merged = plot(first, pairs.first().copied())?;
    merged.series.clear();
    for network in networks {
        for ports in selected_pairs(network.ports(), output_port, input_port) {
            let mut current = plot(network, Some(ports))?;
            if networks.len() > 1 {
                let name = network.name.as_deref().unwrap_or("network");
                for series in &mut current.series {
                    series.label = format!("{name}: {}", series.label);
                }
            }
            merged.series.extend(current.series);
        }
    }
    Ok(merged)
}

fn selected_pairs(
    ports: usize,
    output_port: Option<usize>,
    input_port: Option<usize>,
) -> Vec<(usize, usize)> {
    (0..ports)
        .flat_map(|output| (0..ports).map(move |input| (output, input)))
        .filter(|(output, input)| {
            output_port.is_none_or(|selected| selected == *output)
                && input_port.is_none_or(|selected| selected == *input)
        })
        .collect()
}

fn sibling_output(output: &Path, suffix: &str) -> PathBuf {
    let mut sibling = output.to_path_buf();
    let stem = output
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("touchstone");
    let extension = output
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or("svg");
    sibling.set_file_name(format!("{stem}-{suffix}.{extension}"));
    sibling
}
