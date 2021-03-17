use std::error::Error;
use std::fmt::{Display, Formatter};
use std::io;
use std::num;

pub type Result<T> = std::result::Result<T, GbdtError>;

#[derive(Debug)]
pub enum GbdtError {
    NotSupportExtraMissingNode,
    ChildrenNotFound,
    IO(io::Error),
    ParseInt(num::ParseIntError),
    ParseFloat(num::ParseFloatError),
    SerdeJson(serde_json::Error),
}

impl From<&str> for GbdtError {
    fn from(err: &str) -> GbdtError {
        GbdtError::IO(io::Error::new(io::ErrorKind::Other, err))
    }
}

impl From<serde_json::Error> for GbdtError {
    fn from(err: serde_json::Error) -> GbdtError {
        GbdtError::SerdeJson(err)
    }
}

impl From<num::ParseFloatError> for GbdtError {
    fn from(err: num::ParseFloatError) -> GbdtError {
        GbdtError::ParseFloat(err)
    }
}

impl From<num::ParseIntError> for GbdtError {
    fn from(err: num::ParseIntError) -> GbdtError {
        GbdtError::ParseInt(err)
    }
}

impl From<io::Error> for GbdtError {
    fn from(err: io::Error) -> GbdtError {
        GbdtError::IO(err)
    }
}

impl Display for GbdtError {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        match *self {
            GbdtError::NotSupportExtraMissingNode => write!(f, "Not support extra missing node"),
            GbdtError::ChildrenNotFound => write!(f, "Children not found"),
            GbdtError::IO(ref e) => write!(f, "IO error: {}", e),
            GbdtError::ParseInt(ref e) => write!(f, "ParseInt error: {}", e),
            GbdtError::ParseFloat(ref e) => write!(f, "ParseFloat error: {}", e),
            GbdtError::SerdeJson(ref e) => write!(f, "SerdeJson error: {}", e),
        }
    }
}

impl Error for GbdtError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match *self {
            GbdtError::NotSupportExtraMissingNode => None,
            GbdtError::ChildrenNotFound => None,
            GbdtError::IO(ref e) => Some(e),
            GbdtError::ParseInt(ref e) => Some(e),
            GbdtError::ParseFloat(ref e) => Some(e),
            GbdtError::SerdeJson(ref e) => Some(e),
        }
    }
}
