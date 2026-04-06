use anyhow::{Result, bail};

/// PFFFT wrapper for real-to-complex and complex-to-real transforms. This module is only available when the `pffft`
/// feature is enabled.
use std::ffi::c_int;
use std::os::raw::c_void;

use realfft::num_complex::Complex;

// FFI bindings to the PFFFT C library
#[repr(C)]
struct PffftSetup {
    _private: [u8; 0],
}

#[repr(C)]
#[derive(Copy, Clone)]
enum PffftDirection {
    Forward = 0,
    Backward = 1,
}

#[repr(C)]
#[derive(Copy, Clone)]
enum PffftTransform {
    Real = 0,
    #[allow(dead_code)]
    Complex = 1,
}

#[link(name = "pffft", kind = "static")]
unsafe extern "C" {
    fn pffft_new_setup(n: c_int, transform: PffftTransform) -> *mut PffftSetup;
    fn pffft_destroy_setup(setup: *mut PffftSetup);
    fn pffft_transform_ordered(
        setup: *mut PffftSetup,
        input: *const f32,
        output: *mut f32,
        work: *mut f32,
        direction: PffftDirection,
    );
    fn pffft_aligned_malloc(nb_bytes: usize) -> *mut c_void;
    fn pffft_aligned_free(ptr: *mut c_void);
}

/// Helper struct for aligned memory allocation required by PFFFT.
struct AlignedBuffer {
    ptr: *mut f32,
    len: usize,
}

impl AlignedBuffer {
    fn new(len: usize) -> Option<Self> {
        let bytes = len * std::mem::size_of::<f32>();
        let ptr = unsafe { pffft_aligned_malloc(bytes) as *mut f32 };
        if ptr.is_null() { None } else { Some(AlignedBuffer { ptr, len }) }
    }

    unsafe fn as_slice_mut(&mut self) -> &mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    unsafe fn as_slice(&self) -> &[f32] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        unsafe { pffft_aligned_free(self.ptr as *mut c_void) };
    }
}

/// PFFFT wrapper for real-to-complex transforms.
pub struct PffftRealToComplex {
    setup: *mut PffftSetup,
    size: usize,
    aligned_input: AlignedBuffer,
    aligned_output: AlignedBuffer,
}

/// PFFFT wrapper for complex-to-real transforms.
pub struct PffftComplexToReal {
    setup: *mut PffftSetup,
    size: usize,
    aligned_input: AlignedBuffer,
    aligned_output: AlignedBuffer,
}

impl PffftRealToComplex {
    /// Create a new real-to-complex PFFFT transformer for the given size. The size must be power of two.
    pub fn new(size: usize) -> Result<Self> {
        if !size.is_power_of_two() {
            bail!("{} is not a power of two", size);
        }

        let setup = unsafe {
            // SAFETY: size is a positive integer, PffftTransform::Real is a valid enum value.
            pffft_new_setup(size as c_int, PffftTransform::Real)
        };
        if setup.is_null() {
            bail!("pffft_new_setup failed for size {}", size);
        }

        let aligned_input = AlignedBuffer::new(size).expect("Failed to allocate aligned input buffer");
        let aligned_output = AlignedBuffer::new(size).expect("Failed to allocate aligned output buffer");

        Ok(Self { setup, size, aligned_input, aligned_output })
    }

    /// Process a real input slice and write complex output.
    pub fn process(&mut self, input: &[f32], output: &mut [Complex<f32>]) {
        assert_eq!(input.len(), self.size);
        assert_eq!(output.len(), self.size / 2 + 1);

        unsafe {
            let input_slice = self.aligned_input.as_slice_mut();
            input_slice.copy_from_slice(input);

            pffft_transform_ordered(
                self.setup,
                self.aligned_input.ptr,
                self.aligned_output.ptr,
                std::ptr::null_mut(),
                PffftDirection::Forward,
            );

            let out_slice = self.aligned_output.as_slice();
            // DC component (real) is at index 0, Nyquist (real) is at index 1
            *output.get_unchecked_mut(0) = Complex::new(*out_slice.get_unchecked(0), 0.0);
            for i in 1..self.size / 2 {
                *output.get_unchecked_mut(i) =
                    Complex::new(*out_slice.get_unchecked(2 * i), *out_slice.get_unchecked(2 * i + 1));
            }
            *output.get_unchecked_mut(self.size / 2) = Complex::new(*out_slice.get_unchecked(1), 0.0);
        }
    }
}

impl Drop for PffftRealToComplex {
    fn drop(&mut self) {
        unsafe {
            pffft_destroy_setup(self.setup);
        }
    }
}

impl PffftComplexToReal {
    /// Create a new complex-to-real PFFFT transformer for the given size.
    pub fn new(size: usize) -> Result<Self> {
        if !size.is_power_of_two() {
            bail!("{} is not a power of two", size);
        }

        let setup = unsafe {
            // SAFETY: size is a positive integer, PffftTransform::Real is a valid enum value.
            pffft_new_setup(size as c_int, PffftTransform::Real)
        };
        if setup.is_null() {
            bail!("pffft_new_setup failed for size {}", size);
        }

        let aligned_input = AlignedBuffer::new(size).expect("Failed to allocate aligned input buffer");
        let aligned_output = AlignedBuffer::new(size).expect("Failed to allocate aligned output buffer");

        Ok(Self { setup, size, aligned_input, aligned_output })
    }

    /// Process a complex input slice and write real output.
    pub fn process(&mut self, input: &[Complex<f32>], output: &mut [f32]) {
        assert_eq!(input.len(), self.size / 2 + 1);
        assert_eq!(output.len(), self.size);

        // Convert complex input to packed format in aligned buffer
        unsafe {
            let in_slice = self.aligned_input.as_slice_mut();
            // DC component (real) goes to index 0, Nyquist (real) goes to index 1
            *in_slice.get_unchecked_mut(0) = input.get_unchecked(0).re;
            *in_slice.get_unchecked_mut(1) = input.get_unchecked(self.size / 2).re;
            for i in 1..self.size / 2 {
                *in_slice.get_unchecked_mut(2 * i) = input.get_unchecked(i).re;
                *in_slice.get_unchecked_mut(2 * i + 1) = input.get_unchecked(i).im;
            }
        }

        // Perform the backward transform
        unsafe {
            pffft_transform_ordered(
                self.setup,
                self.aligned_input.ptr,
                self.aligned_output.ptr,
                std::ptr::null_mut(),
                PffftDirection::Backward,
            );
        }

        // Copy the result to output
        unsafe {
            let out_slice = self.aligned_output.as_slice();
            output.copy_from_slice(out_slice);
        }
    }
}

impl Drop for PffftComplexToReal {
    fn drop(&mut self) {
        unsafe {
            pffft_destroy_setup(self.setup);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    #[test]
    fn test_pffft_roundtrip() {
        // We only care about 2^N
        for size in [256, 1024, 4096] {
            // Generate random input in [-5, 5)
            let rng = SmallRng::seed_from_u64(size as u64);
            let input: Vec<f32> = rng.random_iter::<f32>().take(size).map(|x| x * 10.0 - 5.0).collect();

            let mut forward = PffftRealToComplex::new(size).unwrap();
            let mut spectrum = vec![Complex::new(0.0, 0.0); size / 2 + 1];
            forward.process(&input, &mut spectrum);

            let mut backward = PffftComplexToReal::new(size).unwrap();
            let mut output = vec![0.0; size];
            backward.process(&spectrum, &mut output);

            let scale = 1.0 / size as f32;
            for (i, (orig, out)) in input.iter().zip(output.iter()).enumerate() {
                let scaled = out * scale;
                let diff = (*orig - scaled).abs();
                // Allow for some floating point error
                assert!(diff < 1e-4, "Mismatch at index {}: original {}, scaled {}, diff {}", i, orig, scaled, diff);
            }
        }
    }
}
