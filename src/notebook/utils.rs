//! Port of `skrf/notebook/utils.py`.

/// Ordered notebook color definitions used by the interactive plotting helpers.
pub const NOTEBOOK_COLORS: [(&str, &str); 9] = [
    ("lime_green", "#00FF00"),
    ("green", "#00AA00"),
    ("cyan", "#00FFFF"),
    ("blue", "#0000FF"),
    ("red", "#FF0000"),
    ("magenta", "#FF00FF"),
    ("yellow", "#FFFF00"),
    ("purple", "#990099"),
    ("orange", "#FFA500"),
];

const TRACE_COLORS: [&str; 4] = ["#0000FF", "#FF0000", "#FF00FF", "#00AA00"];

/// Cycles through blue, red, magenta, and green trace colors.
///
/// `start` selects the initial color index. Like the upstream generator, the
/// iterator stops before index 1,000.
pub fn trace_color_cycle(start: usize) -> impl Iterator<Item = &'static str> {
    (start..1_000).map(|index| TRACE_COLORS[index % TRACE_COLORS.len()])
}

/// Finds a notebook color by its upstream name.
#[must_use]
pub fn color(name: &str) -> Option<&'static str> {
    NOTEBOOK_COLORS
        .iter()
        .find_map(|(candidate, value)| (*candidate == name).then_some(*value))
}
