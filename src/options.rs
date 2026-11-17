use crate::{
    Audio, Error, Result, Status, Tensor, ThreadBound, Video,
    error::{check, cstring, path_string},
    sys,
};
use std::{marker::PhantomData, path::Path, ptr::NonNull};

mod sealed {
    pub trait Sealed {}
}

/// The purpose of a native options object. Only built-in purposes are allowed.
pub trait Purpose: sealed::Sealed {
    #[doc(hidden)]
    fn create() -> *mut sys::sd_opt_t;
}

/// Mutable options whose purpose is checked at compile time.
///
/// Setters copy strings and arrays and retain tensors, so inputs may be dropped
/// after a successful call. An unsuccessful setter preserves the previous value.
/// Key names and defaults follow `docs/guides/c_api.md` in the pinned upstream.
///
/// ```compile_fail
/// use diffusion_cpp_rs::{Context, ImageOptions};
/// let options = ImageOptions::new().unwrap();
/// Context::new(&options); // Image options cannot load a context.
/// ```
#[derive(Debug)]
pub struct Options<P: Purpose> {
    ptr: NonNull<sys::sd_opt_t>,
    _purpose: PhantomData<P>,
    _thread_bound: ThreadBound,
}

macro_rules! purpose {
    ($kind:ident, $alias:ident, $ctor:ident, $doc:literal) => {
        #[doc = $doc]
        #[derive(Debug)]
        pub enum $kind {}
        impl sealed::Sealed for $kind {}
        impl Purpose for $kind {
            fn create() -> *mut sys::sd_opt_t {
                // SAFETY: the constructor has no preconditions.
                unsafe { sys::$ctor() }
            }
        }
        #[doc = $doc]
        pub type $alias = Options<$kind>;
    };
}
purpose!(
    ContextPurpose,
    ContextOptions,
    sd_ctx_opt_create,
    "Model loading options."
);
purpose!(
    ImagePurpose,
    ImageOptions,
    sd_img_gen_opt_create,
    "Image generation options, including img2img and inpainting."
);
purpose!(
    VideoPurpose,
    VideoOptions,
    sd_vid_gen_opt_create,
    "Video generation options."
);
purpose!(
    UpscalerPurpose,
    UpscalerOptions,
    sd_upscaler_opt_create,
    "Image upscaler loading options."
);
purpose!(
    UpscalePurpose,
    UpscaleOptions,
    sd_upscale_opt_create,
    "Image upscaling options."
);
purpose!(
    VideoUpscalerPurpose,
    VideoUpscalerOptions,
    sd_vid_upscaler_opt_create,
    "Video upscaler loading options."
);
purpose!(
    VideoUpscalePurpose,
    VideoUpscaleOptions,
    sd_vid_upscale_opt_create,
    "Video upscaling options."
);
purpose!(
    ADetailerPurpose,
    ADetailerOptions,
    sd_adetailer_opt_create,
    "ADetailer loading options."
);
purpose!(
    ADetailPurpose,
    ADetailOptions,
    sd_adetail_opt_create,
    "ADetailer processing options."
);

#[derive(Debug, Clone, Copy)]
pub struct Lora<'a> {
    pub path: &'a Path,
    pub multiplier: f32,
    pub is_high_noise: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct Embedding<'a> {
    pub name: &'a str,
    pub path: &'a Path,
}

macro_rules! scalar_setter {
    ($name:ident, $ty:ty, $native:ident, $doc:literal) => {
        #[doc = $doc]
        pub fn $name(&mut self, key: &str, value: $ty) -> Result<&mut Self> {
            let key = cstring(key)?;
            // SAFETY: self is exclusively borrowed, key is live and NUL-terminated.
            check(unsafe { sys::$native(self.ptr.as_ptr(), key.as_ptr(), value) })?;
            Ok(self)
        }
    };
}

impl<P: Purpose> Options<P> {
    pub fn new() -> Result<Self> {
        let ptr = NonNull::new(P::create()).ok_or_else(|| Error::last(Status::OutOfMemory))?;
        Ok(Self {
            ptr,
            _purpose: PhantomData,
            _thread_bound: PhantomData,
        })
    }

    scalar_setter!(set_bool, bool, sd_opt_set_bool, "Sets a boolean option.");
    scalar_setter!(set_int, i64, sd_opt_set_int, "Sets an integer option.");
    scalar_setter!(
        set_float,
        f64,
        sd_opt_set_float,
        "Sets a floating-point option."
    );

    pub fn set_string(&mut self, key: &str, value: &str) -> Result<&mut Self> {
        let key = cstring(key)?;
        let value = cstring(value)?;
        // SAFETY: strings are live through the call; the native setter copies them.
        check(unsafe { sys::sd_opt_set_string(self.ptr.as_ptr(), key.as_ptr(), value.as_ptr()) })?;
        Ok(self)
    }

    pub fn set_path(&mut self, key: &str, path: impl AsRef<Path>) -> Result<&mut Self> {
        let key = cstring(key)?;
        let path = path_string(path.as_ref())?;
        // SAFETY: the path is copied before returning.
        check(unsafe { sys::sd_opt_set_string(self.ptr.as_ptr(), key.as_ptr(), path.as_ptr()) })?;
        Ok(self)
    }

    pub fn set_tensor(&mut self, key: &str, tensor: &Tensor) -> Result<&mut Self> {
        let key = cstring(key)?;
        // SAFETY: native code retains the valid tensor on success.
        check(unsafe { sys::sd_opt_set_tensor(self.ptr.as_ptr(), key.as_ptr(), tensor.as_ptr()) })?;
        Ok(self)
    }

    pub fn set_tensors(&mut self, key: &str, tensors: &[Tensor]) -> Result<&mut Self> {
        let key = cstring(key)?;
        let pointers: Vec<_> = tensors.iter().map(Tensor::as_ptr).collect();
        // SAFETY: all pointers are live; the native setter retains their storage.
        check(unsafe {
            sys::sd_opt_set_tensors(
                self.ptr.as_ptr(),
                key.as_ptr(),
                pointers.as_ptr(),
                pointers.len(),
            )
        })?;
        Ok(self)
    }

    pub fn set_floats(&mut self, key: &str, values: &[f32]) -> Result<&mut Self> {
        let key = cstring(key)?;
        // SAFETY: the slice is valid and copied synchronously.
        check(unsafe {
            sys::sd_opt_set_floats(
                self.ptr.as_ptr(),
                key.as_ptr(),
                values.as_ptr(),
                values.len(),
            )
        })?;
        Ok(self)
    }

    pub fn set_ints(&mut self, key: &str, values: &[i32]) -> Result<&mut Self> {
        let key = cstring(key)?;
        // SAFETY: the slice is valid and copied synchronously.
        check(unsafe {
            sys::sd_opt_set_ints(
                self.ptr.as_ptr(),
                key.as_ptr(),
                values.as_ptr(),
                values.len(),
            )
        })?;
        Ok(self)
    }

    pub fn set_loras(&mut self, key: &str, values: &[Lora<'_>]) -> Result<&mut Self> {
        let key = cstring(key)?;
        let paths = values
            .iter()
            .map(|lora| path_string(lora.path))
            .collect::<Result<Vec<_>>>()?;
        let raw: Vec<_> = values
            .iter()
            .zip(&paths)
            .map(|(lora, path)| sys::sd_lora_t {
                is_high_noise: lora.is_high_noise,
                multiplier: lora.multiplier,
                path: path.as_ptr(),
            })
            .collect();
        // SAFETY: descriptors and their strings remain live; all are copied.
        check(unsafe {
            sys::sd_opt_set_loras(self.ptr.as_ptr(), key.as_ptr(), raw.as_ptr(), raw.len())
        })?;
        Ok(self)
    }

    pub fn set_embeddings(&mut self, key: &str, values: &[Embedding<'_>]) -> Result<&mut Self> {
        let key = cstring(key)?;
        let names = values
            .iter()
            .map(|item| cstring(item.name))
            .collect::<Result<Vec<_>>>()?;
        let paths = values
            .iter()
            .map(|item| path_string(item.path))
            .collect::<Result<Vec<_>>>()?;
        let raw: Vec<_> = names
            .iter()
            .zip(&paths)
            .map(|(name, path)| sys::sd_embedding_t {
                name: name.as_ptr(),
                path: path.as_ptr(),
            })
            .collect();
        // SAFETY: descriptors and their strings are live and copied synchronously.
        check(unsafe {
            sys::sd_opt_set_embeddings(self.ptr.as_ptr(), key.as_ptr(), raw.as_ptr(), raw.len())
        })?;
        Ok(self)
    }

    pub fn set_audios(&mut self, key: &str, values: &[Audio]) -> Result<&mut Self> {
        let key = cstring(key)?;
        let raw: Vec<_> = values.iter().map(Audio::raw).collect();
        // SAFETY: descriptors and tensor handles are live; native code retains them.
        check(unsafe {
            sys::sd_opt_set_audios(self.ptr.as_ptr(), key.as_ptr(), raw.as_ptr(), raw.len())
        })?;
        Ok(self)
    }

    pub fn set_videos(&mut self, key: &str, values: &[Video]) -> Result<&mut Self> {
        let key = cstring(key)?;
        let raw: Vec<_> = values.iter().map(Video::raw).collect();
        // SAFETY: descriptors and tensor handles are live; native code retains them.
        check(unsafe {
            sys::sd_opt_set_videos(self.ptr.as_ptr(), key.as_ptr(), raw.as_ptr(), raw.len())
        })?;
        Ok(self)
    }

    pub fn unset(&mut self, key: &str) -> Result<&mut Self> {
        let key = cstring(key)?;
        // SAFETY: self is exclusively borrowed and key is a valid C string.
        check(unsafe { sys::sd_opt_unset(self.ptr.as_ptr(), key.as_ptr()) })?;
        Ok(self)
    }

    /// A borrowed handle. Mutating it through FFI requires exclusive access.
    pub fn as_ptr(&self) -> *const sys::sd_opt_t {
        self.ptr.as_ptr()
    }
}

impl<P: Purpose> Drop for Options<P> {
    fn drop(&mut self) {
        // SAFETY: self owns this options object and it is no longer in use.
        unsafe { sys::sd_opt_free(self.ptr.as_ptr()) }
    }
}
