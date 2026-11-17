//! Raw bindings generated from the pinned stable-diffusion.cpp header.
//!
//! All native ownership, pointer validity, callback lifetime and synchronization
//! requirements apply. Prefer `diffusion-cpp-rs` for safe resource management.
#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals)]
#![allow(clippy::all)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

/// The upstream revision bundled with this crate.
pub const UPSTREAM_REVISION: &str = "532d2c15c430a785a712b2d94281f04e733008a4";
