//! Safe, owning Rust bindings to stable-diffusion.cpp.
//!
//! ```no_run
//! use diffusion_cpp_rs::{Context, ContextOptions, ImageOptions};
//!
//! # fn main() -> diffusion_cpp_rs::Result<()> {
//! let mut model = ContextOptions::new()?;
//! model.set_path("model_path", "model.safetensors")?;
//! let mut ctx = Context::new(&model)?;
//! let mut options = ImageOptions::new()?;
//! options.set_string("prompt", "a small house in the mountains")?
//!     .set_int("width", 512)?
//!     .set_int("height", 512)?
//!     .set_int("seed", 42)?
//!     .set_int("sample.steps", 20)?;
//! let output = ctx.generate_image(&options)?;
//! let image = output.image(0)?;
//! let pixels = image.as_slice::<u8>()?;
//! # Ok(())
//! # }
//! ```
//!
//! Option keys follow the pinned upstream C API. Setters validate their type and
//! purpose in native code. Handles release their resources on drop and are
//! deliberately neither `Send` nor `Sync`. Create and use a context on the same
//! worker thread; the bindings do not promise concurrent native inference.

#![deny(unsafe_op_in_unsafe_fn)]

mod context;
mod error;
mod options;
mod output;
mod tensor;
mod utilities;

pub use context::{ADetailer, Context, GenerationProgress, Upscaler, VideoUpscaler};
pub use error::{Error, Result, Status};
pub use options::{
    ADetailOptions, ADetailerOptions, ContextOptions, Embedding, ImageOptions, Lora, Options,
    Purpose, UpscaleOptions, UpscalerOptions, VideoOptions, VideoUpscaleOptions,
    VideoUpscalerOptions,
};
pub use output::{Audio, GenerationOutput, ResultKind, Video};
pub use tensor::{Tensor, TensorElement, TensorType};
pub use utilities::{
    CannyOptions, build_commit, build_version, devices, physical_cores, system_info,
};

/// Complete unsafe C API, including global callbacks and advanced conversion APIs.
pub use diffusion_cpp_sys as sys;

// Native handles are thread-affine until upstream specifies a stronger contract.
type ThreadBound = std::marker::PhantomData<std::rc::Rc<()>>;
