use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("invalid frequency axis: {0}")]
    InvalidFrequency(String),

    #[error("incompatible network shapes: {0}")]
    IncompatibleShape(String),

    #[error("port index {port} is outside a {ports}-port network")]
    InvalidPort { port: usize, ports: usize },

    #[error("parse error: {0}")]
    Parse(String),

    #[error("unsupported operation: {0}")]
    Unsupported(String),

    #[error(transparent)]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
