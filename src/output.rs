use crate::{Error, Result, Status, Tensor, ThreadBound, error::copy_string, sys};
use std::{
    marker::PhantomData,
    ptr::{self, NonNull},
};

/// Audio samples in an F32 tensor with shape `[channels, samples]`.
#[derive(Debug, Clone)]
pub struct Audio {
    pub sample_rate: u32,
    pub data: Tensor,
}

impl Audio {
    pub(crate) fn raw(&self) -> sys::sd_audio_t {
        sys::sd_audio_t {
            sample_rate: self.sample_rate,
            data: self.data.as_ptr(),
        }
    }

    // SAFETY: the descriptor and its optional tensor are live during this call.
    pub(crate) unsafe fn from_raw(raw: &sys::sd_audio_t) -> Result<Option<Self>> {
        if raw.data.is_null() {
            return Ok(None);
        }
        // SAFETY: guaranteed by the caller; retaining makes the audio independent.
        let data = unsafe { Tensor::retain(raw.data) }?;
        Ok(Some(Self {
            sample_rate: raw.sample_rate,
            data,
        }))
    }
}

/// Interleaved U8 video storage with shape `[channels, width, height, frames]`.
#[derive(Debug, Clone)]
pub struct Video {
    pub data: Tensor,
    pub fps: i32,
    pub audio: Option<Audio>,
}

impl Video {
    pub(crate) fn raw(&self) -> sys::sd_video_t {
        sys::sd_video_t {
            data: self.data.as_ptr(),
            fps: self.fps,
            audio: self
                .audio
                .as_ref()
                .map(Audio::raw)
                .unwrap_or(sys::sd_audio_t {
                    sample_rate: 0,
                    data: ptr::null(),
                }),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResultKind {
    Image,
    Video,
}

/// Owns generated images, video/audio and metadata independently of the context.
/// Tensors returned by accessors retain their own references and may outlive this
/// result. This also avoids borrowing pointers invalidated by lazy video assembly.
#[derive(Debug)]
pub struct GenerationOutput {
    ptr: NonNull<sys::sd_result_t>,
    _thread_bound: ThreadBound,
}

impl GenerationOutput {
    // SAFETY: ptr must be null or transfer one owned result.
    pub(crate) unsafe fn from_owned(ptr: *mut sys::sd_result_t) -> Result<Self> {
        let ptr = NonNull::new(ptr).ok_or_else(|| Error::last(Status::GenerationFailed))?;
        Ok(Self {
            ptr,
            _thread_bound: PhantomData,
        })
    }

    pub fn kind(&self) -> ResultKind {
        // SAFETY: self owns the result.
        match unsafe { sys::sd_result_get_kind(self.as_ptr()) } {
            sys::SD_RESULT_IMAGE => ResultKind::Image,
            sys::SD_RESULT_VIDEO => ResultKind::Video,
            other => panic!("unexpected native result kind {other}"),
        }
    }

    /// Image batch size, or the number of video frames.
    pub fn image_count(&self) -> usize {
        // SAFETY: self owns the result.
        unsafe { sys::sd_result_get_image_count(self.as_ptr()) }
    }

    pub fn image(&self, index: usize) -> Result<Tensor> {
        if index >= self.image_count() {
            return Err(Error::new(
                Status::OutOfRange,
                "image index is out of range",
            ));
        }
        // SAFETY: the index is valid; retaining the borrowed tensor grants ownership.
        unsafe {
            let tensor = sys::sd_result_get_image(self.as_ptr(), index);
            if tensor.is_null() {
                return Err(Error::last(Status::GenerationFailed));
            }
            Tensor::retain(tensor)
        }
    }

    pub fn images(&self) -> Result<Vec<Tensor>> {
        (0..self.image_count())
            .map(|index| self.image(index))
            .collect()
    }

    /// The native JSON generation metadata for an image or video frame.
    pub fn generation_info(&self, index: usize) -> Result<String> {
        if index >= self.image_count() {
            return Err(Error::new(
                Status::OutOfRange,
                "metadata index is out of range",
            ));
        }
        // SAFETY: result is live; copy the borrowed string before any other API call.
        unsafe {
            let text = sys::sd_result_get_generation_info(self.as_ptr(), index);
            if text.is_null() {
                return Err(Error::last(Status::GenerationFailed));
            }
            Ok(copy_string(text))
        }
    }

    pub fn audio(&self) -> Result<Option<Audio>> {
        // SAFETY: self owns the descriptor and tensors throughout the retain call.
        unsafe {
            let audio = sys::sd_result_get_audio(self.as_ptr());
            if audio.is_null() {
                return Ok(None);
            }
            Audio::from_raw(&*audio)
        }
    }

    /// Materializes contiguous video storage on first access.
    pub fn video(&self) -> Result<Option<Video>> {
        if self.kind() != ResultKind::Video {
            return Ok(None);
        }
        // SAFETY: self owns all borrowed descriptors until they have been retained.
        unsafe {
            let video = sys::sd_result_get_video(self.as_ptr());
            if video.is_null() {
                return Err(Error::last(Status::GenerationFailed));
            }
            let video = &*video;
            let data = Tensor::retain(video.data)?;
            let audio = Audio::from_raw(&video.audio)?;
            Ok(Some(Video {
                data,
                fps: video.fps,
                audio,
            }))
        }
    }

    pub fn fps(&self) -> i32 {
        // SAFETY: self owns the result.
        unsafe { sys::sd_result_get_fps(self.as_ptr()) }
    }

    pub fn as_ptr(&self) -> *const sys::sd_result_t {
        self.ptr.as_ptr()
    }
}

impl Drop for GenerationOutput {
    fn drop(&mut self) {
        // SAFETY: release exactly the result owned by self.
        unsafe { sys::sd_result_free(self.ptr.as_ptr()) }
    }
}
