use crate::sys;
use std::{
    ffi::{CStr, CString},
    fmt,
    path::Path,
};

/// Status reported by the C API, or an equivalent Rust-side validation failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Status {
    InvalidArgument,
    UnknownKey,
    TypeMismatch,
    OutOfRange,
    Unsupported,
    OutOfMemory,
    LoadFailed,
    GenerationFailed,
    Busy,
    Cancelled,
    Unknown(u32),
}

/// An owned diagnostic. It survives subsequent native API calls.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error {
    pub status: Status,
    pub message: String,
}

pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    pub(crate) fn new(status: Status, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
        }
    }

    pub(crate) fn last(status: Status) -> Self {
        // SAFETY: the thread-local diagnostic is copied before the next API call.
        let message = unsafe { copy_string(sys::sd_get_last_error()) };
        Self::new(
            status,
            if message.is_empty() {
                "native operation failed".into()
            } else {
                message
            },
        )
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}: {}", self.status, self.message)
    }
}

impl std::error::Error for Error {}

pub(crate) fn check(status: sys::sd_status_t) -> Result<()> {
    let status = match status {
        sys::SD_STATUS_OK => return Ok(()),
        sys::SD_STATUS_INVALID_ARGUMENT => Status::InvalidArgument,
        sys::SD_STATUS_UNKNOWN_KEY => Status::UnknownKey,
        sys::SD_STATUS_TYPE_MISMATCH => Status::TypeMismatch,
        sys::SD_STATUS_OUT_OF_RANGE => Status::OutOfRange,
        sys::SD_STATUS_UNSUPPORTED => Status::Unsupported,
        sys::SD_STATUS_OUT_OF_MEMORY => Status::OutOfMemory,
        sys::SD_STATUS_LOAD_FAILED => Status::LoadFailed,
        sys::SD_STATUS_GENERATION_FAILED => Status::GenerationFailed,
        sys::SD_STATUS_BUSY => Status::Busy,
        sys::SD_STATUS_CANCELLED => Status::Cancelled,
        other => Status::Unknown(other),
    };
    Err(Error::last(status))
}

pub(crate) fn cstring(value: &str) -> Result<CString> {
    CString::new(value).map_err(|_| {
        Error::new(
            Status::InvalidArgument,
            "string contains an interior NUL byte",
        )
    })
}

pub(crate) fn path_string(path: &Path) -> Result<CString> {
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        CString::new(path.as_os_str().as_bytes()).map_err(|_| {
            Error::new(
                Status::InvalidArgument,
                "path contains an interior NUL byte",
            )
        })
    }
    #[cfg(not(unix))]
    {
        let path = path
            .to_str()
            .ok_or_else(|| Error::new(Status::InvalidArgument, "path is not valid Unicode"))?;
        cstring(path)
    }
}

// SAFETY: caller guarantees a null pointer or a live NUL-terminated C string.
pub(crate) unsafe fn copy_string(ptr: *const std::ffi::c_char) -> String {
    if ptr.is_null() {
        String::new()
    } else {
        // SAFETY: guaranteed by the caller; the result owns a copy.
        unsafe { CStr::from_ptr(ptr) }
            .to_string_lossy()
            .into_owned()
    }
}
