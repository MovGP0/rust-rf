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

/// Port of `trace_color_cycle`; yields the same finite sequence through index 999.
pub fn trace_color_cycle(start: usize) -> impl Iterator<Item = &'static str> {
    (start..1_000).map(|index| TRACE_COLORS[index % TRACE_COLORS.len()])
}

/// Finds a notebook color by its upstream name.
pub fn color(name: &str) -> Option<&'static str> {
    NOTEBOOK_COLORS
        .iter()
        .find_map(|(candidate, value)| (*candidate == name).then_some(*value))
}
