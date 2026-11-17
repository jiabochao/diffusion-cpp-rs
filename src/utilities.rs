use crate::{
    Error, Result, Status, Tensor,
    error::{check, copy_string},
    sys,
};
use std::ptr;

/// Parameters for native Canny image preprocessing.
#[derive(Debug, Clone, Copy)]
pub struct CannyOptions {
    pub high_threshold: f32,
    pub low_threshold: f32,
    pub weak: f32,
    pub strong: f32,
    pub inverse: bool,
}

impl Default for CannyOptions {
    fn default() -> Self {
        Self {
            high_threshold: 0.08,
            low_threshold: 0.08,
            weak: 0.8,
            strong: 1.0,
            inverse: false,
        }
    }
}

impl CannyOptions {
    pub fn process(&self, image: &Tensor) -> Result<Tensor> {
        for value in [
            self.high_threshold,
            self.low_threshold,
            self.weak,
            self.strong,
        ] {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(Error::new(
                    Status::OutOfRange,
                    "Canny thresholds and strengths must be finite values in [0, 1]",
                ));
            }
        }
        if self.weak > self.strong {
            return Err(Error::new(
                Status::OutOfRange,
                "Canny weak strength cannot exceed strong strength",
            ));
        }
        let mut out = ptr::null_mut();
        // SAFETY: image is immutable and live, output storage is writable.
        check(unsafe {
            sys::sd_preprocess_canny(
                image.as_ptr(),
                self.high_threshold,
                self.low_threshold,
                self.weak,
                self.strong,
                self.inverse,
                &mut out,
            )
        })?;
        // SAFETY: successful preprocessing transfers ownership.
        unsafe { Tensor::from_owned(out) }
    }
}

pub fn build_version() -> String {
    // SAFETY: native code returns a static C string.
    unsafe { copy_string(sys::sd_version()) }
}

pub fn build_commit() -> String {
    // SAFETY: native code returns a static C string.
    unsafe { copy_string(sys::sd_commit()) }
}

pub fn physical_cores() -> i32 {
    // SAFETY: no arguments or ownership obligations.
    unsafe { sys::sd_get_num_physical_cores() }
}

pub fn system_info() -> String {
    // SAFETY: native code returns a NUL-terminated string; copy it immediately.
    unsafe { copy_string(sys::sd_get_system_info()) }
}

/// Available backend devices, one `name\tdescription` per line.
pub fn devices() -> String {
    // SAFETY: a null buffer with size zero queries the required byte count.
    let mut needed = unsafe { sys::sd_list_devices(ptr::null_mut(), 0) };
    loop {
        let mut buffer = vec![0u8; needed.saturating_add(1)];
        // SAFETY: buffer is writable for exactly the supplied size.
        let actual = unsafe { sys::sd_list_devices(buffer.as_mut_ptr().cast(), buffer.len()) };
        if actual <= needed {
            return String::from_utf8_lossy(&buffer[..actual]).into_owned();
        }
        needed = actual;
    }
}
