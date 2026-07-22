//! Rust port of `doc/gh-pages.py`.

use std::env;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

/// Local checkout of the GitHub Pages repository.
const PAGES_DIRECTORY: &str = "gh-pages";
/// Generated HTML documentation directory.
const HTML_DIRECTORY: &str = "build/html";
/// Generated PDF documentation path.
const PDF_PATH: &str = "build/latex/scikit-rf.pdf";
/// SSH URL of the documentation repository.
const PAGES_REPOSITORY: &str = "git@github.com:scikit-rf/doc.git";

/// Publishes a generated documentation build into the GitHub Pages checkout.
///
/// The optional positional argument selects the destination tag. Without one, the command uses
/// `git describe --exact-match` and falls back to `dev`.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let start = env::current_dir()?;
    let tag = env::args().nth(1).unwrap_or_else(|| {
        command_output(&start, "git", &["describe", "--exact-match"])
            .ok()
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| "dev".to_owned())
    });
    validate_component(&tag)?;
    let pages = start.join(PAGES_DIRECTORY);
    if pages.exists() {
        run(&pages, "git", &["checkout", "gh-pages"])?;
        run(&pages, "git", &["pull", "--ff-only"])?;
    } else {
        run(&start, "git", &["clone", PAGES_REPOSITORY, PAGES_DIRECTORY])?;
        run(&pages, "git", &["checkout", "gh-pages"])?;
    }
    let branch = command_output(&pages, "git", &["branch", "--show-current"])?;
    if branch != "gh-pages" {
        return Err(format!("documentation repository is on {branch}, expected gh-pages").into());
    }
    let destination = pages.join(&tag);
    ensure_child(&pages, &destination)?;
    if destination.exists() {
        fs::remove_dir_all(&destination)?;
    }
    copy_directory(&start.join(HTML_DIRECTORY), &destination)?;
    let pdf = start.join(PDF_PATH);
    if pdf.exists() {
        fs::copy(&pdf, destination.join("scikit-rf.pdf"))?;
    }
    run(&pages, "git", &["add", "-A", &tag])?;
    run(
        &pages,
        "git",
        &["commit", "-m", &format!("Updated doc release: {tag}")],
    )?;
    run(&pages, "git", &["--no-pager", "log", "--oneline", "-3"])?;
    println!("Documentation staged in {}", destination.display());
    println!("Verify the build and push the gh-pages repository explicitly.");
    Ok(())
}

/// Ensures a documentation tag is one safe path component.
fn validate_component(value: &str) -> Result<(), Box<dyn std::error::Error>> {
    if value.is_empty()
        || value == "."
        || value == ".."
        || value.contains('/')
        || value.contains('\\')
    {
        return Err("documentation tag must be one safe path component".into());
    }
    Ok(())
}

/// Ensures `child` resolves beneath `parent` before any replacement occurs.
fn ensure_child(parent: &Path, child: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let parent = parent.canonicalize()?;
    let candidate = child
        .parent()
        .ok_or("documentation destination has no parent")?
        .canonicalize()?
        .join(
            child
                .file_name()
                .ok_or("documentation destination has no name")?,
        );
    if !candidate.starts_with(&parent) || candidate == parent {
        return Err("documentation destination escapes the gh-pages repository".into());
    }
    Ok(())
}

/// Recursively copies the generated documentation tree.
fn copy_directory(source: &Path, destination: &Path) -> io::Result<()> {
    if !source.is_dir() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!(
                "documentation HTML directory {} does not exist",
                source.display()
            ),
        ));
    }
    fs::create_dir_all(destination)?;
    for entry in fs::read_dir(source)? {
        let entry = entry?;
        let target = destination.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            copy_directory(&entry.path(), &target)?;
        } else {
            fs::copy(entry.path(), target)?;
        }
    }
    Ok(())
}

/// Executes a command and forwards its standard output and standard error.
fn run(
    directory: &Path,
    program: &str,
    arguments: &[&str],
) -> Result<(), Box<dyn std::error::Error>> {
    let output = command(directory, program, arguments)?;
    if !output.status.success() {
        return Err(command_error(program, arguments, &output).into());
    }
    print!("{}", String::from_utf8_lossy(&output.stdout));
    eprint!("{}", String::from_utf8_lossy(&output.stderr));
    Ok(())
}

/// Executes a command and returns its trimmed standard output.
fn command_output(
    directory: &Path,
    program: &str,
    arguments: &[&str],
) -> Result<String, Box<dyn std::error::Error>> {
    let output = command(directory, program, arguments)?;
    if !output.status.success() {
        return Err(command_error(program, arguments, &output).into());
    }
    Ok(String::from_utf8(output.stdout)?.trim().to_owned())
}

/// Executes `program` with `arguments` in `directory`.
fn command(directory: &Path, program: &str, arguments: &[&str]) -> io::Result<Output> {
    Command::new(program)
        .args(arguments)
        .current_dir(directory)
        .output()
}

/// Formats a failed command and its captured error output.
fn command_error(program: &str, arguments: &[&str], output: &Output) -> String {
    format!(
        "{} {} failed with {}: {}",
        program,
        arguments.join(" "),
        output.status,
        String::from_utf8_lossy(&output.stderr).trim()
    )
}

#[allow(dead_code)]
/// Returns the publication destination for `tag`.
fn destination_for(root: &Path, tag: &str) -> PathBuf {
    root.join(PAGES_DIRECTORY).join(tag)
}
