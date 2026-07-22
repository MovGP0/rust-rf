//! Integration tests for general utility helpers and homogeneous collections.

use std::collections::BTreeMap;
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

use approx::assert_relative_eq;
use chrono::{Datelike, Timelike};
use ndarray::array;
use rust_rf::util::{
    HomoDict, HomoList, ProgressBar, SmoothingWindow, basename_without_extension,
    dictionary_to_records, duplicate_index, extension, find_nearest, find_nearest_index,
    git_version, now_string, parse_now_string, replace_in_files, slice_domain, smooth, unique_name,
};

#[test]
/// Finds the nearest sample and the inclusive index range for a domain.
fn finds_nearest_values_and_domain_ranges() {
    let values = array![0.0, 2.0, 5.0, 9.0];
    assert_eq!(
        find_nearest_index(&values, 4.0).expect("nearest should exist"),
        2
    );
    assert_relative_eq!(
        find_nearest(&values, 4.0).expect("nearest should exist"),
        5.0,
        epsilon = f64::EPSILON
    );
    assert_eq!(
        slice_domain(&values, (1.0, 8.0)).expect("range should exist"),
        0..=3
    );
    assert!(find_nearest_index(&array![], 1.0).is_err());
}

#[test]
/// Handles path extensions, duplicate detection, and unique-name suffixes.
fn handles_path_names_duplicates_and_unique_suffixes() {
    assert_eq!(extension("fixtures/example.s2p").as_deref(), Some("s2p"));
    assert_eq!(extension("fixtures/example"), None);
    assert_eq!(
        basename_without_extension("fixtures/example.s2p").as_deref(),
        Some("example")
    );

    let names = vec!["dut".to_owned(), "dut_01".to_owned(), "other".to_owned()];
    assert_eq!(duplicate_index(&"dut".to_owned(), &names, None), Some(0));
    assert_eq!(duplicate_index(&"dut".to_owned(), &names, Some(0)), None);
    assert_eq!(unique_name("dut", &names, None), "dut_02");
    assert_eq!(unique_name("dut_01", &names, None), "dut_02");
    assert_eq!(unique_name("dut", &names, Some(0)), "dut");
}

#[test]
/// Smooths reflected one-dimensional signals and rejects invalid window lengths.
fn smooths_reflected_one_dimensional_signals() {
    let values = array![1.0, 2.0, 3.0, 4.0, 5.0];
    let smoothed =
        smooth(&values, 3, SmoothingWindow::Flat).expect("flat smoothing should succeed");
    let expected = [5.0 / 3.0, 2.0, 3.0, 4.0, 13.0 / 3.0];
    for (actual, expected) in smoothed.iter().zip(expected) {
        assert_relative_eq!(*actual, expected, epsilon = 1.0e-12);
    }
    assert_eq!(
        smooth(&values, 2, SmoothingWindow::Hanning).expect("short windows are identity"),
        values
    );
    assert!(smooth(&values, 6, SmoothingWindow::Flat).is_err());
}

#[test]
/// Exercises indexing, mapping, boolean-style matching, and selection for
/// homogeneous lists and dictionaries.
fn maps_and_selects_homogeneous_collections() {
    let list = HomoList::new(["asdf".to_owned(), "ZZZZ".to_owned()]);
    assert_eq!(&list[0], "asdf");
    assert_eq!(list.map(|value| value.to_uppercase())[0], "ASDF");
    let indexes = list.matching_indexes(|value| value.starts_with('a'));
    assert_eq!(
        list.select(&indexes).expect("selection should work").values,
        vec!["asdf"]
    );

    let dictionary = HomoDict::new(BTreeMap::from([
        ("a".to_owned(), "asdf".to_owned()),
        ("b".to_owned(), "ZZZZ".to_owned()),
    ]));
    assert_eq!(&dictionary[&"a".to_owned()], "asdf");
    assert_eq!(
        dictionary.map_values(|value| value.to_uppercase())[&"a".to_owned()],
        "ASDF"
    );
    let keys = dictionary
        .matching_keys(|value| value.starts_with('a'))
        .into_iter()
        .cloned()
        .collect::<Vec<_>>();
    assert_eq!(
        dictionary
            .select(&keys)
            .expect("selection should work")
            .values,
        BTreeMap::from([("a".to_owned(), "asdf".to_owned())])
    );
}

#[test]
/// Round-trips sortable timestamp strings and rejects invalid timestamps.
fn round_trips_sortable_timestamps() {
    let timestamp = now_string();
    let parsed = parse_now_string(&timestamp).expect("generated timestamp should parse");
    assert_eq!(
        parsed.format("%Y.%m.%d.%H.%M.%S.%6f").to_string(),
        timestamp
    );

    let known =
        parse_now_string("2026.07.21.13.14.15.123456").expect("known timestamp should parse");
    assert_eq!(known.year(), 2026);
    assert_eq!(known.month(), 7);
    assert_eq!(known.day(), 21);
    assert_eq!(known.hour(), 13);
    assert_eq!(known.minute(), 14);
    assert_eq!(known.second(), 15);
    assert_eq!(known.nanosecond(), 123_456_000);

    assert!(parse_now_string("not-a-timestamp").is_err());
    assert!(parse_now_string("2026.02.30.00.00.00").is_err());
}

#[test]
/// Converts delimited dictionary keys into structured records.
fn converts_structured_dictionary_keys_to_records() {
    let values = BTreeMap::from([("a,1.5,-2".to_owned(), 10), ("b,3.0,4".to_owned(), 20)]);
    let records = dictionary_to_records(&values, ",").expect("records should convert");
    assert_eq!(records.len(), 2);
    assert_eq!(records[0].fields, ["a", "1.5", "-2"]);
    assert_eq!(records[0].value, 10);
    assert_eq!(records[1].fields, ["b", "3.0", "4"]);
    assert_eq!(records[1].value, 20);
    assert!(dictionary_to_records(&values, "").is_err());
}

#[test]
/// Recursively replaces matching text only in files selected by the glob.
fn recursively_replaces_only_matching_files() {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock should follow the Unix epoch")
        .as_nanos();
    let root = std::env::current_dir()
        .expect("current directory should exist")
        .join(".temp")
        .join(format!("util-replace-{}-{unique}", std::process::id()));
    let nested = root.join("nested");
    fs::create_dir_all(&nested).expect("temporary directories should be created");
    fs::write(root.join("first.txt"), "find this twice: find this")
        .expect("first fixture should be written");
    fs::write(nested.join("second.txt"), "find this here")
        .expect("second fixture should be written");
    fs::write(root.join("ignored.md"), "find this unchanged")
        .expect("ignored fixture should be written");

    let mut changed = replace_in_files(&root, "find this", "replacement", "*.txt")
        .expect("recursive replacement should succeed");
    changed.sort();
    assert_eq!(changed, [root.join("first.txt"), nested.join("second.txt")]);
    assert_eq!(
        fs::read_to_string(root.join("first.txt")).expect("first fixture should be readable"),
        "replacement twice: replacement"
    );
    assert_eq!(
        fs::read_to_string(root.join("ignored.md")).expect("ignored fixture should be readable"),
        "find this unchanged"
    );

    fs::remove_dir_all(&root).expect("temporary fixtures should be removed");
    let _ = fs::remove_dir(root.parent().expect("temporary root should have a parent"));
}

#[test]
/// Renders progress updates and reads the repository's Git description.
fn renders_progress_and_reads_git_description() {
    let mut progress = ProgressBar::new(10, "measurements").expect("progress bar should construct");
    assert!(progress.render().contains("0%"));
    progress.update(5);
    assert!(progress.to_string().contains("50%"));
    assert!(
        progress
            .to_string()
            .contains("5 of 10 measurements complete")
    );
    progress.advance();
    assert!(progress.render().contains("6 of 10 measurements complete"));
    progress.update(20);
    assert!(progress.render().contains("100%"));
    assert!(ProgressBar::new(0, "invalid").is_err());

    assert!(git_version(".").is_ok());
}
