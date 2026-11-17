use crate::{Error, Result, Status, ThreadBound, sys};
use std::{marker::PhantomData, mem, ptr::NonNull, slice};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum TensorType {
    U8 = sys::SD_TENSOR_U8 as i32,
    I32 = sys::SD_TENSOR_I32 as i32,
    I64 = sys::SD_TENSOR_I64 as i32,
    F32 = sys::SD_TENSOR_F32 as i32,
    F64 = sys::SD_TENSOR_F64 as i32,
}

mod sealed {
    pub trait Sealed {}
    impl Sealed for u8 {}
    impl Sealed for i32 {}
    impl Sealed for i64 {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

/// A supported tensor element. Sealed to types with no invalid bit patterns.
pub trait TensorElement: sealed::Sealed + Copy + 'static {
    const TYPE: TensorType;
}

macro_rules! element {
    ($ty:ty, $kind:ident) => {
        impl TensorElement for $ty {
            const TYPE: TensorType = TensorType::$kind;
        }
    };
}
element!(u8, U8);
element!(i32, I32);
element!(i64, I64);
element!(f32, F32);
element!(f64, F64);

/// An immutable, reference-counted native tensor.
///
/// Dimensions are contiguous-first: RGB images use `[3, width, height]`, audio
/// uses `[channels, samples]`, and video uses `[channels, width, height, frames]`.
/// Cloning retains native storage; it does not copy pixels.
#[derive(Debug)]
pub struct Tensor {
    pub(crate) ptr: NonNull<sys::sd_tensor_t>,
    _thread_bound: ThreadBound,
}

impl Tensor {
    /// Copies data into native storage after checking the shape and byte count.
    pub fn new<T: TensorElement>(shape: &[i64], data: &[T]) -> Result<Self> {
        if shape.len() > 8 || shape.iter().any(|&dim| dim < 0) {
            return Err(Error::new(
                Status::InvalidArgument,
                "tensor rank must be <= 8 and dimensions nonnegative",
            ));
        }
        let elements = shape
            .iter()
            .try_fold(1usize, |count, &dim| {
                if count == 0 {
                    Some(0)
                } else {
                    usize::try_from(dim)
                        .ok()
                        .and_then(|dim| count.checked_mul(dim))
                }
            })
            .filter(|&count| count as u128 <= i64::MAX as u128)
            .ok_or_else(|| {
                Error::new(
                    Status::OutOfRange,
                    "tensor shape overflows addressable memory",
                )
            })?;
        if elements != data.len() {
            return Err(Error::new(
                Status::InvalidArgument,
                "tensor shape does not match the number of elements",
            ));
        }
        // SAFETY: data and shape are valid slices. Native creation copies them.
        let ptr = unsafe {
            sys::sd_tensor_create(
                T::TYPE as i32,
                shape.len(),
                shape.as_ptr(),
                data.as_ptr().cast(),
                mem::size_of_val(data),
            )
        };
        // SAFETY: the returned pointer is either null or a newly owned tensor.
        unsafe { Self::from_owned(ptr) }
    }

    /// Creates an interleaved, 8-bit RGB image.
    pub fn rgb(width: u32, height: u32, pixels: &[u8]) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(Error::new(
                Status::InvalidArgument,
                "image dimensions must be positive",
            ));
        }
        Self::new(&[3, i64::from(width), i64::from(height)], pixels)
    }

    pub fn element_type(&self) -> TensorType {
        // SAFETY: self owns a valid immutable handle.
        match unsafe { sys::sd_tensor_type(self.as_ptr()) } {
            1 => TensorType::U8,
            2 => TensorType::I32,
            3 => TensorType::I64,
            4 => TensorType::F32,
            5 => TensorType::F64,
            other => panic!("unsupported native tensor type {other}"),
        }
    }

    pub fn shape(&self) -> Vec<i64> {
        // SAFETY: the handle is live, and each axis is below its rank.
        unsafe {
            (0..sys::sd_tensor_rank(self.as_ptr()))
                .map(|axis| sys::sd_tensor_dim(self.as_ptr(), axis))
                .collect()
        }
    }

    pub fn byte_len(&self) -> usize {
        // SAFETY: self owns the tensor.
        unsafe { sys::sd_tensor_byte_size(self.as_ptr()) }
    }

    /// Borrows typed data for the lifetime of this tensor.
    pub fn as_slice<T: TensorElement>(&self) -> Result<&[T]> {
        if self.element_type() != T::TYPE {
            return Err(Error::new(
                Status::TypeMismatch,
                "requested element type differs from the tensor type",
            ));
        }
        let bytes = self.byte_len();
        if bytes == 0 {
            return Ok(&[]);
        }
        // SAFETY: data is immutable and lives until the last owning handle drops.
        let ptr = unsafe { sys::sd_tensor_data(self.as_ptr()) }.cast::<T>();
        if ptr.is_null()
            || !ptr.is_aligned()
            || bytes > isize::MAX as usize
            || bytes % mem::size_of::<T>() != 0
        {
            return Err(Error::new(
                Status::InvalidArgument,
                "invalid native tensor storage",
            ));
        }
        // SAFETY: type, alignment and size were validated. All supported element
        // types accept every bit pattern. The borrow cannot outlive self.
        Ok(unsafe { slice::from_raw_parts(ptr, bytes / mem::size_of::<T>()) })
    }

    /// A borrowed raw handle, valid until this tensor is dropped.
    pub fn as_ptr(&self) -> *const sys::sd_tensor_t {
        self.ptr.as_ptr()
    }

    // SAFETY: ptr must be null or transfer one native ownership reference.
    pub(crate) unsafe fn from_owned(ptr: *mut sys::sd_tensor_t) -> Result<Self> {
        let ptr = NonNull::new(ptr).ok_or_else(|| Error::last(Status::OutOfMemory))?;
        Ok(Self {
            ptr,
            _thread_bound: PhantomData,
        })
    }

    // SAFETY: ptr must be a live tensor borrowed for the duration of this call.
    pub(crate) unsafe fn retain(ptr: *const sys::sd_tensor_t) -> Result<Self> {
        // SAFETY: guaranteed by the caller; retain grants a new ownership reference.
        unsafe { Self::from_owned(sys::sd_tensor_retain(ptr)) }
    }
}

impl Clone for Tensor {
    fn clone(&self) -> Self {
        // SAFETY: retaining a valid tensor cannot fail and allocates no storage.
        unsafe { Self::retain(self.as_ptr()) }.expect("retaining a valid tensor failed")
    }
}

impl Drop for Tensor {
    fn drop(&mut self) {
        // SAFETY: release exactly the reference owned by self.
        unsafe { sys::sd_tensor_free(self.ptr.as_ptr()) }
    }
}
