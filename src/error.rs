use thiserror::Error;

#[derive(Debug, Error)]
/// Error returned by rust-rf operations.
pub enum Error {
    /// A frequency axis or frequency value is invalid.
    #[error("invalid frequency axis: {0}")]
    InvalidFrequency(String),

    /// Array or network dimensions are incompatible.
    #[error("incompatible network shapes: {0}")]
    IncompatibleShape(String),

    /// A port index lies outside the available network ports.
    #[error("port index {port} is outside a {ports}-port network")]
    InvalidPort {
        /// Invalid one-based port index.
        port: usize,
        /// Number of ports available on the network.
        ports: usize,
    },

    /// Input data or an instrument response could not be parsed.
    #[error("parse error: {0}")]
    Parse(String),

    /// The requested operation is not supported.
    #[error("unsupported operation: {0}")]
    Unsupported(String),

    /// An input/output operation failed.
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

/// Result type used throughout rust-rf.
pub type Result<T> = std::result::Result<T, Error>;
