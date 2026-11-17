use crate::{
    ADetailOptions, ADetailerOptions, ContextOptions, Error, GenerationOutput, ImageOptions,
    Result, Status, ThreadBound, UpscaleOptions, UpscalerOptions, VideoOptions,
    VideoUpscaleOptions, VideoUpscalerOptions,
    error::{check, copy_string, path_string},
    sys,
};
use std::{
    ffi::c_void,
    marker::PhantomData,
    panic::{AssertUnwindSafe, catch_unwind},
    path::Path,
    ptr::{self, NonNull},
};

/// A copied context-local progress event. A total of zero means unknown.
#[derive(Debug, Clone)]
pub struct GenerationProgress {
    pub stage: String,
    pub detail: String,
    pub unit: String,
    pub sample_pass: i32,
    pub completed: i64,
    pub total: i64,
}

struct ProgressCallback(Box<dyn Fn(GenerationProgress) + Send + Sync>);

unsafe extern "C" fn progress_trampoline(
    progress: *const sys::sd_generation_progress_t,
    data: *mut c_void,
) {
    if progress.is_null() || data.is_null() {
        return;
    }
    // Catch the whole conversion and invocation: unwinding must never cross FFI.
    let result = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: native code supplies a live event and our registered boxed state.
        let (raw, callback) = unsafe { (&*progress, &*data.cast::<ProgressCallback>()) };
        let event = GenerationProgress {
            // SAFETY: event strings remain live during the synchronous callback.
            stage: unsafe { copy_string(raw.stage) },
            detail: unsafe { copy_string(raw.detail) },
            unit: unsafe { copy_string(raw.unit) },
            sample_pass: raw.sample_pass,
            completed: raw.completed,
            total: raw.total,
        };
        (callback.0)(event);
    }));
    if let Err(payload) = result {
        // Even a panic payload with a panicking destructor must not unwind into C.
        std::mem::forget(payload);
    }
}

/// A loaded diffusion model, released automatically on drop.
///
/// ```compile_fail
/// use diffusion_cpp_rs::Context;
/// fn require_send<T: Send>() {}
/// require_send::<Context>();
/// ```
pub struct Context {
    ptr: NonNull<sys::sd_ctx_t>,
    progress: Option<Box<ProgressCallback>>,
    _thread_bound: ThreadBound,
}

impl Context {
    /// Loads a model. The context retains its own configuration.
    pub fn new(options: &ContextOptions) -> Result<Self> {
        let mut ptr = ptr::null_mut();
        // SAFETY: options are live and output storage is writable.
        check(unsafe { sys::sd_ctx_create(options.as_ptr(), &mut ptr) })?;
        let ptr = NonNull::new(ptr).ok_or_else(|| Error::last(Status::LoadFailed))?;
        Ok(Self {
            ptr,
            progress: None,
            _thread_bound: PhantomData,
        })
    }

    pub fn generate_image(&mut self, options: &ImageOptions) -> Result<GenerationOutput> {
        let mut out = ptr::null_mut();
        // SAFETY: exclusive context access prevents overlapping operations.
        check(unsafe { sys::sd_generate_image(self.ptr.as_ptr(), options.as_ptr(), &mut out) })?;
        // SAFETY: successful generation transfers ownership of the result.
        unsafe { GenerationOutput::from_owned(out) }
    }

    pub fn generate_video(&mut self, options: &VideoOptions) -> Result<GenerationOutput> {
        let mut out = ptr::null_mut();
        // SAFETY: exclusive context access prevents overlapping operations.
        check(unsafe { sys::sd_generate_video(self.ptr.as_ptr(), options.as_ptr(), &mut out) })?;
        // SAFETY: successful generation transfers ownership of the result.
        unsafe { GenerationOutput::from_owned(out) }
    }

    pub fn supports_image_generation(&self) -> bool {
        // SAFETY: the context is live and idle.
        unsafe { sys::sd_ctx_supports_image_generation(self.ptr.as_ptr()) }
    }

    pub fn supports_video_generation(&self) -> bool {
        // SAFETY: the context is live and idle.
        unsafe { sys::sd_ctx_supports_video_generation(self.ptr.as_ptr()) }
    }

    pub fn model_version(&self) -> String {
        // SAFETY: upstream returns a static, NUL-terminated model name.
        unsafe { copy_string(sys::sd_get_model_version_name(self.ptr.as_ptr())) }
    }

    pub fn load_control_net(&mut self, path: impl AsRef<Path>) -> Result<()> {
        let path = path_string(path.as_ref())?;
        // SAFETY: the context is exclusively borrowed and the string is live.
        if unsafe { sys::sd_ctx_load_control_net(self.ptr.as_ptr(), path.as_ptr()) } {
            Ok(())
        } else {
            Err(Error::new(
                Status::LoadFailed,
                "ControlNet loading failed; see the native log for details",
            ))
        }
    }

    pub fn unload_control_net(&mut self) -> Result<()> {
        // SAFETY: the context is exclusively borrowed and idle.
        if unsafe { sys::sd_ctx_unload_control_net(self.ptr.as_ptr()) } {
            Ok(())
        } else {
            Err(Error::new(
                Status::InvalidArgument,
                "ControlNet unloading failed; see the native log for details",
            ))
        }
    }

    pub fn has_control_net(&self) -> bool {
        // SAFETY: the context is live and idle.
        unsafe { sys::sd_ctx_has_control_net(self.ptr.as_ptr()) }
    }

    /// Registers an owned callback. Panics are contained at the FFI boundary;
    /// callbacks should handle their own errors. Replaces any previous callback.
    pub fn set_progress_callback<F>(&mut self, callback: F)
    where
        F: Fn(GenerationProgress) + Send + Sync + 'static,
    {
        let mut callback = Box::new(ProgressCallback(Box::new(callback)));
        let data = (&mut *callback as *mut ProgressCallback).cast();
        // SAFETY: context is idle. The box stays at a fixed address until cleared.
        unsafe {
            sys::sd_set_generation_progress_callback(
                self.ptr.as_ptr(),
                Some(progress_trampoline),
                data,
            )
        };
        self.progress = Some(callback);
    }

    pub fn clear_progress_callback(&mut self) {
        // SAFETY: unregister before freeing callback state; no generation is active.
        unsafe {
            sys::sd_set_generation_progress_callback(self.ptr.as_ptr(), None, ptr::null_mut())
        };
        self.progress = None;
    }

    /// A borrowed native context. Callers of raw APIs must uphold exclusivity.
    pub fn as_ptr(&self) -> *mut sys::sd_ctx_t {
        self.ptr.as_ptr()
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        self.clear_progress_callback();
        // SAFETY: context is idle, no registered callback references freed state.
        unsafe { sys::sd_ctx_free(self.ptr.as_ptr()) }
    }
}

macro_rules! model_handle {
    ($name:ident, $native:ty, $options:ty, $create:ident, $free:ident, $doc:literal) => {
        #[doc = $doc]
        #[derive(Debug)]
        pub struct $name {
            ptr: NonNull<$native>,
            _thread_bound: ThreadBound,
        }
        impl $name {
            pub fn new(options: &$options) -> Result<Self> {
                let mut ptr = ptr::null_mut();
                // SAFETY: options and output storage are valid throughout the call.
                check(unsafe { sys::$create(options.as_ptr(), &mut ptr) })?;
                let ptr = NonNull::new(ptr).ok_or_else(|| Error::last(Status::LoadFailed))?;
                Ok(Self {
                    ptr,
                    _thread_bound: PhantomData,
                })
            }
        }
        impl Drop for $name {
            fn drop(&mut self) {
                // SAFETY: self exclusively owns an idle handle.
                unsafe { sys::$free(self.ptr.as_ptr()) }
            }
        }
    };
}
model_handle!(
    Upscaler,
    sys::sd_upscaler_t,
    UpscalerOptions,
    sd_upscaler_create,
    sd_upscaler_free,
    "A loaded image upscaler."
);
model_handle!(
    VideoUpscaler,
    sys::sd_vid_upscaler_t,
    VideoUpscalerOptions,
    sd_vid_upscaler_create,
    sd_vid_upscaler_free,
    "A loaded video upscaler."
);
model_handle!(
    ADetailer,
    sys::sd_adetailer_t,
    ADetailerOptions,
    sd_adetailer_create,
    sd_adetailer_free,
    "A loaded ADetailer detector."
);

impl Upscaler {
    pub fn upscale(&mut self, options: &UpscaleOptions) -> Result<GenerationOutput> {
        let mut out = ptr::null_mut();
        // SAFETY: the upscaler is exclusive, options live, and out writable.
        check(unsafe { sys::sd_upscale(self.ptr.as_ptr(), options.as_ptr(), &mut out) })?;
        // SAFETY: successful operation transfers ownership.
        unsafe { GenerationOutput::from_owned(out) }
    }

    pub fn scale(&self) -> i32 {
        // SAFETY: the upscaler is live and idle.
        unsafe { sys::sd_upscaler_get_scale(self.ptr.as_ptr()) }
    }

    /// Reads model metadata only; `None` means an unrecognized RGB upscaler.
    pub fn probe_scale(path: impl AsRef<Path>) -> Result<Option<i32>> {
        let path = path_string(path.as_ref())?;
        // SAFETY: path is a valid C string throughout the synchronous probe.
        let scale = unsafe { sys::sd_upscaler_probe_scale(path.as_ptr()) };
        Ok((scale > 0).then_some(scale))
    }
}

impl VideoUpscaler {
    pub fn upscale(&mut self, options: &VideoUpscaleOptions) -> Result<GenerationOutput> {
        let mut out = ptr::null_mut();
        // SAFETY: the upscaler is exclusive, options live, and out writable.
        check(unsafe { sys::sd_upscale_video(self.ptr.as_ptr(), options.as_ptr(), &mut out) })?;
        // SAFETY: successful operation transfers ownership.
        unsafe { GenerationOutput::from_owned(out) }
    }
}

impl ADetailer {
    pub fn process(
        &mut self,
        context: &mut Context,
        options: &ADetailOptions,
    ) -> Result<GenerationOutput> {
        let mut out = ptr::null_mut();
        // SAFETY: both model handles are exclusively borrowed, options stay live.
        check(unsafe {
            sys::sd_adetail(
                self.ptr.as_ptr(),
                context.ptr.as_ptr(),
                options.as_ptr(),
                &mut out,
            )
        })?;
        // SAFETY: successful operation transfers ownership.
        unsafe { GenerationOutput::from_owned(out) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        Arc,
        atomic::{AtomicI64, Ordering},
    };

    #[test]
    fn callback_copies_event_and_contains_panics() {
        let completed = Arc::new(AtomicI64::new(0));
        let capture = completed.clone();
        let mut state = ProgressCallback(Box::new(move |event| {
            assert_eq!(event.stage, "sampling");
            capture.store(event.completed, Ordering::Relaxed);
            panic!("a user callback may panic");
        }));
        let event = sys::sd_generation_progress_t {
            stage: c"sampling".as_ptr(),
            detail: ptr::null(),
            unit: c"steps".as_ptr(),
            sample_pass: 1,
            completed: 3,
            total: 20,
        };
        // SAFETY: both event and callback state remain live throughout invocation.
        unsafe { progress_trampoline(&event, (&mut state as *mut ProgressCallback).cast()) };
        assert_eq!(completed.load(Ordering::Relaxed), 3);
    }
}
