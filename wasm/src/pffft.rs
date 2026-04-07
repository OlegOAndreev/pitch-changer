use anyhow::{Result, bail};

/// PFFFT wrapper for real-to-complex, complex-to-real, and complex-to-complex transforms. This module is only available when the `pffft`
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

    fn as_slice_mut(&mut self) -> &mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    fn as_slice(&self) -> &[f32] {
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

/// PFFFT wrapper for complex-to-complex transforms.
pub struct PffftComplex {
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

        let input_slice = self.aligned_input.as_slice_mut();
        input_slice.copy_from_slice(input);

        unsafe {
            pffft_transform_ordered(
                self.setup,
                self.aligned_input.ptr,
                self.aligned_output.ptr,
                std::ptr::null_mut(),
                PffftDirection::Forward,
            );
        }

        let output_f32 = complex_as_f32_slice_mut(output);
        let out_slice = self.aligned_output.as_slice();
        // DC component (real) is at index 0, Nyquist (real) is at index 1
        output_f32[0] = out_slice[0];
        output_f32[1] = 0.0;
        output_f32[2..self.size].copy_from_slice(&out_slice[2..]);
        output_f32[self.size] = out_slice[1];
        output_f32[self.size + 1] = 0.0;
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
        let input_f32 = complex_as_f32_slice(input);
        let in_slice = self.aligned_input.as_slice_mut();
        // DC component (real) goes to index 0, Nyquist (real) goes to index 1
        in_slice[0] = input_f32[0];
        in_slice[1] = input_f32[self.size];
        in_slice[2..].copy_from_slice(&input_f32[2..self.size]);

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
        output.copy_from_slice(self.aligned_output.as_slice());
    }
}

impl PffftComplex {
    /// Create a new complex-to-complex PFFFT transformer for the given size. The size must be power of two.
    pub fn new(size: usize) -> Result<Self> {
        if !size.is_power_of_two() {
            bail!("{} is not a power of two", size);
        }

        let setup = unsafe {
            // SAFETY: size is a positive integer, PffftTransform::Complex is a valid enum value.
            pffft_new_setup(size as c_int, PffftTransform::Complex)
        };
        if setup.is_null() {
            bail!("pffft_new_setup failed for size {}", size);
        }

        let aligned_input = AlignedBuffer::new(2 * size).expect("Failed to allocate aligned input buffer");
        let aligned_output = AlignedBuffer::new(2 * size).expect("Failed to allocate aligned output buffer");

        Ok(Self { setup, size, aligned_input, aligned_output })
    }

    /// Perform a forward complex-to-complex transform.
    pub fn forward(&mut self, input: &[Complex<f32>], output: &mut [Complex<f32>]) {
        assert_eq!(input.len(), self.size);
        assert_eq!(output.len(), self.size);

        let in_slice = self.aligned_input.as_slice_mut();
        in_slice.copy_from_slice(complex_as_f32_slice(input));

        unsafe {
            pffft_transform_ordered(
                self.setup,
                self.aligned_input.ptr,
                self.aligned_output.ptr,
                std::ptr::null_mut(),
                PffftDirection::Forward,
            );
        }

        // Convert interleaved output back to complex
        let out_slice = self.aligned_output.as_slice();
        let output_f32 = complex_as_f32_slice_mut(output);
        output_f32.copy_from_slice(out_slice);
    }

    /// Perform a backward complex-to-complex transform.
    pub fn backward(&mut self, input: &[Complex<f32>], output: &mut [Complex<f32>]) {
        assert_eq!(input.len(), self.size);
        assert_eq!(output.len(), self.size);

        let input_f32 = complex_as_f32_slice(input);
        let in_slice = self.aligned_input.as_slice_mut();
        in_slice.copy_from_slice(input_f32);

        unsafe {
            pffft_transform_ordered(
                self.setup,
                self.aligned_input.ptr,
                self.aligned_output.ptr,
                std::ptr::null_mut(),
                PffftDirection::Backward,
            );
        }

        let output_f32 = complex_as_f32_slice_mut(output);
        output_f32.copy_from_slice(self.aligned_output.as_slice());
    }
}

impl Drop for PffftComplex {
    fn drop(&mut self) {
        unsafe {
            pffft_destroy_setup(self.setup);
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
    use realfft::RealFftPlanner;
    use rustfft::FftPlanner;

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

    #[test]
    fn test_pffft_complex_roundtrip() {
        // We only care about 2^N
        for size in [256, 1024, 4096] {
            // Generate random complex input
            let rng = SmallRng::seed_from_u64(size as u64);
            let input: Vec<Complex<f32>> = rng
                .random_iter::<f32>()
                .take(2 * size)
                .map(|x| x * 10.0 - 5.0)
                .collect::<Vec<f32>>()
                .chunks_exact(2)
                .map(|chunk| Complex::new(chunk[0], chunk[1]))
                .collect();

            let mut transform = PffftComplex::new(size).unwrap();
            let mut spectrum = vec![Complex::new(0.0, 0.0); size];
            transform.forward(&input, &mut spectrum);

            let mut output = vec![Complex::new(0.0, 0.0); size];
            transform.backward(&spectrum, &mut output);

            let scale = 1.0 / size as f32;
            for (i, (orig, out)) in input.iter().zip(output.iter()).enumerate() {
                let scaled = *out * scale;
                let diff = (*orig - scaled).norm();
                // Allow for some floating point error
                assert!(
                    diff < 1e-4,
                    "Mismatch at index {}: original {:?}, scaled {:?}, diff {}",
                    i,
                    orig,
                    scaled,
                    diff
                );
            }
        }
    }

    #[test]
    fn test_pffft_vs_realfft_forward() {
        for size in [256, 1024, 4096] {
            let rng = SmallRng::seed_from_u64(size as u64);
            let input: Vec<f32> = rng.random_iter::<f32>().take(size).map(|x| x * 10.0 - 5.0).collect();

            let mut pffft_forward = PffftRealToComplex::new(size).unwrap();
            let mut spectrum_pffft = vec![Complex::new(0.0, 0.0); size / 2 + 1];
            pffft_forward.process(&input, &mut spectrum_pffft);

            let mut planner = RealFftPlanner::new();
            let r2c = planner.plan_fft_forward(size);
            let mut in_buf = input.clone();
            let mut spectrum_realfft = r2c.make_output_vec();
            let mut scratch = r2c.make_scratch_vec();
            r2c.process_with_scratch(&mut in_buf, &mut spectrum_realfft, &mut scratch).unwrap();

            // Compare
            for (p, r) in spectrum_pffft.iter().zip(spectrum_realfft.iter()) {
                let diff_re = (p.re - r.re).abs();
                let diff_im = (p.im - r.im).abs();
                assert!(diff_re < 1e-4, "Real part mismatch: PFFFT {}, realfft {}, diff {}", p.re, r.re, diff_re);
                assert!(diff_im < 1e-4, "Imag part mismatch: PFFFT {}, realfft {}, diff {}", p.im, r.im, diff_im);
            }
        }
    }

    #[test]
    fn test_pffft_vs_rustfft_forward() {
        for size in [256, 1024, 4096] {
            let rng = SmallRng::seed_from_u64(size as u64);
            let input: Vec<Complex<f32>> = rng
                .random_iter::<f32>()
                .take(2 * size)
                .map(|x| x * 10.0 - 5.0)
                .collect::<Vec<f32>>()
                .chunks_exact(2)
                .map(|chunk| Complex::new(chunk[0], chunk[1]))
                .collect();

            let mut pffft = PffftComplex::new(size).unwrap();
            let mut spectrum_pffft = vec![Complex::new(0.0, 0.0); size];
            pffft.forward(&input, &mut spectrum_pffft);

            let mut planner = FftPlanner::new();
            let fft = planner.plan_fft_forward(size);
            let mut in_buf = input.clone();
            let mut spectrum_rustfft = vec![Complex::new(0.0, 0.0); size];
            let mut scratch = vec![Complex::new(0.0, 0.0); size];
            fft.process_immutable_with_scratch(&mut in_buf, &mut spectrum_rustfft, &mut scratch);

            for (p, r) in spectrum_pffft.iter().zip(spectrum_rustfft.iter()) {
                let diff_re = (p.re - r.re).abs();
                let diff_im = (p.im - r.im).abs();
                assert!(diff_re < 1e-3, "Real part mismatch: PFFFT {}, rustfft {}, diff {}", p.re, r.re, diff_re);
                assert!(diff_im < 1e-3, "Imag part mismatch: PFFFT {}, rustfft {}, diff {}", p.im, r.im, diff_im);
            }
        }
    }
}

fn complex_as_f32_slice(slice: &[Complex<f32>]) -> &[f32] {
    unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const f32, slice.len() * 2) }
}

fn complex_as_f32_slice_mut(slice: &mut [Complex<f32>]) -> &mut [f32] {
    unsafe { std::slice::from_raw_parts_mut(slice.as_mut_ptr() as *mut f32, slice.len() * 2) }
}
