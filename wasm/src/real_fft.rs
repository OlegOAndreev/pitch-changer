/// real_fft is a replacement for realfft crate, which supports only 2^N sizes and uses SIMD instructions for better
/// performance. Ported from https://github.com/HEnquist/realfft/blob/master/src/lib.rs
use std::f32;
use std::sync::Arc;

use anyhow::{Result, bail};

use rustfft::Fft;
use rustfft::num_complex::Complex;

use crate::float4::{Float4, SimdFloat4};

pub struct FftRealToComplex {
    fft: Arc<dyn Fft<f32>>,
    size: usize,
    twiddles: Vec<Complex<f32>>,
}

impl FftRealToComplex {
    /// Create new FftRealToComplex
    pub fn new(size: usize) -> Result<FftRealToComplex> {
        if !size.is_power_of_two() || !size.is_multiple_of(16) {
            bail!("FftRealToComplex size must be power of two, is {}", size);
        }
        let mut planner = rustfft::FftPlanner::new();
        let fft = planner.plan_fft_forward(size / 2);
        let twiddle_count = size / 4;
        let twiddles: Vec<Complex<f32>> = (1..twiddle_count).map(|i| compute_twiddle(i, size) * 0.5).collect();
        Ok(Self { fft, size, twiddles })
    }

    /// Return the size of scratch buffer that must be passed to process().
    pub fn get_scratch_len(&self) -> usize {
        return self.fft.get_outofplace_scratch_len();
    }

    /// Process input and store the result in output. Output is sized the same way as realfft does: the first element
    /// is DC and the last element is Nyqist, both have zero immediate component. The input is used as a scratch buffer.
    #[inline(never)]
    pub fn process(&self, input: &mut [f32], output: &mut [Complex<f32>], scratch: &mut [Complex<f32>]) -> Result<()> {
        if input.len() != self.size {
            bail!("Expected input size {}, got {}", self.size, input.len());
        }
        if output.len() != self.size / 2 + 1 {
            bail!("Expected output size {}, got {}", self.size / 2 + 1, input.len());
        }
        if scratch.len() < self.get_scratch_len() {
            bail!("Expected scratch size {}, got {}", self.get_scratch_len(), input.len());
        }

        let fft_input = f32_as_complex_slice_mut(input);
        self.fft.process_outofplace_with_scratch(fft_input, &mut output[..self.size / 2], scratch);

        let re0 = output[0].re;
        let im0 = output[0].im;
        output[0].re = re0 + im0;
        output[0].im = 0.0;
        output[self.size / 2].re = re0 - im0;
        output[self.size / 2].im = 0.0;

        let twiddle_f32_ptr = self.twiddles.as_ptr() as *const f32;
        let output_f32_ptr = output.as_mut_ptr() as *mut f32;
        let simd_half = SimdFloat4::splat(0.5);
        // See remainder plain loop to see how the original loop looks like.
        for i in (1..(self.size / 4 - 3)).step_by(4) {
            let (out_re, out_im) = unsafe { SimdFloat4::load_deinterleave2(output_f32_ptr.add(2 * i)) };
            let (out_rev_re, out_rev_im) =
                unsafe { SimdFloat4::load_deinterleave2_rev(output_f32_ptr.add(self.size - 2 * i - 6)) };
            let (twiddle_re, twiddle_im) = unsafe { SimdFloat4::load_deinterleave2(twiddle_f32_ptr.add(2 * i - 2)) };

            let sum_re = SimdFloat4::add(out_re, out_rev_re);
            let sum_im = SimdFloat4::add(out_im, out_rev_im);
            let diff_re = SimdFloat4::sub(out_re, out_rev_re);
            let diff_im = SimdFloat4::sub(out_im, out_rev_im);

            let half_sum_re = SimdFloat4::mul(simd_half, sum_re);
            let half_diff_im = SimdFloat4::mul(simd_half, diff_im);

            let output_twiddled_re =
                SimdFloat4::add(SimdFloat4::mul(sum_im, twiddle_re), SimdFloat4::mul(diff_re, twiddle_im));
            let output_twiddled_im =
                SimdFloat4::sub(SimdFloat4::mul(sum_im, twiddle_im), SimdFloat4::mul(diff_re, twiddle_re));

            let out_re = SimdFloat4::add(half_sum_re, output_twiddled_re);
            let out_im = SimdFloat4::add(half_diff_im, output_twiddled_im);
            unsafe {
                SimdFloat4::store_interleave2(output_f32_ptr.add(2 * i), out_re, out_im);
            }
            let out_rev_re = SimdFloat4::sub(half_sum_re, output_twiddled_re);
            let out_rev_im = SimdFloat4::sub(output_twiddled_im, half_diff_im);
            unsafe {
                SimdFloat4::store_interleave2_rev(output_f32_ptr.add(self.size - 2 * i - 6), out_rev_re, out_rev_im);
            }
        }

        // Remainder plain loop
        for i in (self.size / 4 - 3)..(self.size / 4) {
            let out = output[i];
            let out_rev = output[self.size / 2 - i];
            let twiddle = self.twiddles[i - 1];

            let sum = out + out_rev;
            let diff = out - out_rev;

            let half_sum_re = 0.5 * sum.re;
            let half_diff_im = 0.5 * diff.im;

            let output_twiddled_re = sum.im * twiddle.re + diff.re * twiddle.im;
            let output_twiddled_im = sum.im * twiddle.im - diff.re * twiddle.re;

            output[i].re = half_sum_re + output_twiddled_re;
            output[i].im = half_diff_im + output_twiddled_im;
            output[self.size / 2 - i].re = half_sum_re - output_twiddled_re;
            output[self.size / 2 - i].im = -half_diff_im + output_twiddled_im;
        }

        output[self.size / 4].im *= -1.0;

        Ok(())
    }

    /// Create a vector suitable for passing as scratch parameter to process.
    #[inline(never)]
    pub fn make_scratch_vec(&self) -> Vec<Complex<f32>> {
        vec![Complex::ZERO; self.size / 2]
    }
}

// fn simd_assert_eq(v: SimdFloat4, s0: f32, s1: f32, s2: f32, s3: f32) {
//     let mut arr = [0.0f32; 4];
//     SimdFloat4::store_to_slice(v, &mut arr);
//     assert_eq!(arr[0], s0, "0");
//     assert_eq!(arr[1], s1, "1");
//     assert_eq!(arr[2], s2, "2");
//     assert_eq!(arr[3], s3, "3");
// }

pub struct FftComplexToReal {
    fft: Arc<dyn Fft<f32>>,
    size: usize,
    twiddles: Vec<Complex<f32>>,
}

impl FftComplexToReal {
    /// Create new FftComplexToReal
    pub fn new(size: usize) -> Result<FftComplexToReal> {
        if !size.is_power_of_two() || !size.is_multiple_of(4) {
            bail!("FftComplexToReal size must be power of two and multiple of 4, is {}", size);
        }
        let mut planner = rustfft::FftPlanner::new();
        let fft = planner.plan_fft_inverse(size / 2);
        let twiddle_count = size / 4;
        let twiddles: Vec<Complex<f32>> = (1..twiddle_count).map(|i| compute_twiddle(i, size).conj()).collect();
        Ok(Self { fft, size, twiddles })
    }

    /// Return the size of scratch buffer that must be passed to process().
    pub fn get_scratch_len(&self) -> usize {
        return self.fft.get_outofplace_scratch_len();
    }

    /// Process input and store the result in output. Input is sized the same way as realfft does: the first element is
    /// DC and the last element is Nyqist, both have zero immediate component (unlike realfft, we simply ignore the
    /// immediate components). The input is used as a scratch buffer.
    #[inline(never)]
    pub fn process(&self, input: &mut [Complex<f32>], output: &mut [f32], scratch: &mut [Complex<f32>]) -> Result<()> {
        if input.len() != self.size / 2 + 1 {
            bail!("Expected input size {}, got {}", self.size / 2 + 1, input.len());
        }
        if output.len() != self.size {
            bail!("Expected output size {}, got {}", self.size, input.len());
        }
        if scratch.len() < self.get_scratch_len() {
            bail!("Expected scratch size {}, got {}", self.get_scratch_len(), input.len());
        }

        let input_f32 = complex_as_f32_slice_mut(input);
        unsafe {
            let first_re = input_f32[0] + input_f32[self.size];
            let first_im = input_f32[0] - input_f32[self.size];
            input_f32[0] = first_re;
            input_f32[1] = first_im;

            for i in 1..self.size / 4 {
                let out_re = *input_f32.get_unchecked(2 * i);
                let out_im = *input_f32.get_unchecked(2 * i + 1);
                let out_rev_re = *input_f32.get_unchecked(self.size - 2 * i);
                let out_rev_im = *input_f32.get_unchecked(self.size - 2 * i + 1);

                let twiddle = *self.twiddles.get_unchecked(i - 1);

                let sum_re = out_re + out_rev_re;
                let sum_im = out_im + out_rev_im;
                let diff_re = out_re - out_rev_re;
                let diff_im = out_im - out_rev_im;

                let output_twiddled_real = sum_im * twiddle.re + diff_re * twiddle.im;
                let output_twiddled_im = sum_im * twiddle.im - diff_re * twiddle.re;

                *input_f32.get_unchecked_mut(2 * i) = sum_re - output_twiddled_real;
                *input_f32.get_unchecked_mut(2 * i + 1) = diff_im - output_twiddled_im;

                *input_f32.get_unchecked_mut(self.size - 2 * i) = sum_re + output_twiddled_real;
                *input_f32.get_unchecked_mut(self.size - 2 * i + 1) = -output_twiddled_im - diff_im;
            }

            input_f32[self.size / 2] *= 2.0;
            input_f32[self.size / 2 + 1] *= -2.0;
        }

        let fft_output = f32_as_complex_slice_mut(output);
        self.fft.process_outofplace_with_scratch(&mut input[..self.size / 2], fft_output, scratch);

        Ok(())
    }

    /// Create a vector suitable for passing as scratch parameter to process.
    #[inline(never)]
    pub fn make_scratch_vec(&self) -> Vec<Complex<f32>> {
        vec![Complex::ZERO; self.size / 2]
    }
}

#[inline(always)]
fn compute_twiddle(i: usize, size: usize) -> Complex<f32> {
    let angle = -2.0 * f32::consts::PI * i as f32 / size as f32;
    Complex { re: angle.cos(), im: angle.sin() }
}

fn f32_as_complex_slice_mut(slice: &mut [f32]) -> &mut [Complex<f32>] {
    assert!(slice.len().is_multiple_of(2));
    unsafe { std::slice::from_raw_parts_mut(slice.as_mut_ptr() as *mut Complex<f32>, slice.len() / 2) }
}

fn complex_as_f32_slice_mut(slice: &mut [Complex<f32>]) -> &mut [f32] {
    unsafe { std::slice::from_raw_parts_mut(slice.as_mut_ptr() as *mut f32, slice.len() * 2) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};
    use realfft::RealFftPlanner;

    #[test]
    fn test_real_fft_roundtrip() {
        for size in [16, 256, 512, 1024, 2048, 4096] {
            // Generate random input in [-5, 5)
            let rng = SmallRng::seed_from_u64(size as u64);
            let mut input: Vec<f32> = rng.random_iter::<f32>().take(size).map(|x| x * 10.0 - 5.0).collect();
            let orig_input = input.clone();

            let forward = FftRealToComplex::new(size).unwrap();
            let mut spectrum = vec![Complex::ZERO; size / 2 + 1];
            let mut scratch = vec![Complex::ZERO; forward.get_scratch_len()];
            forward.process(&mut input, &mut spectrum, &mut scratch).unwrap();

            let backward = FftComplexToReal::new(size).unwrap();
            let mut output = vec![0.0; size];
            let mut backward_scratch = vec![Complex::ZERO; backward.get_scratch_len()];
            backward.process(&mut spectrum, &mut output, &mut backward_scratch).unwrap();

            let scale = 1.0 / size as f32;
            for (i, (orig, out)) in orig_input.iter().zip(output.iter()).enumerate() {
                let scaled = out * scale;
                let diff = (*orig - scaled).abs();
                assert!(
                    diff < 1e-4,
                    "Mismatch at size {}, index {}: original {}, scaled {}, diff {}",
                    size,
                    i,
                    orig,
                    scaled,
                    diff
                );
            }
        }
    }

    #[test]
    fn test_real_fft_vs_realfft_forward() {
        for size in [16, 256, 512, 1024, 2048, 4096] {
            // Generate random input in [-5, 5)
            for i in 0..10 {
                let rng = SmallRng::seed_from_u64((size + i) as u64);
                let mut input: Vec<f32> = rng.random_iter::<f32>().take(size).map(|x| x * 10.0 - 5.0).collect();
                let mut orig_input = input.clone();

                let forward = FftRealToComplex::new(size).unwrap();
                let mut spectrum = vec![Complex::ZERO; size / 2 + 1];
                let mut scratch = vec![Complex::ZERO; forward.get_scratch_len()];
                forward.process(&mut input, &mut spectrum, &mut scratch).unwrap();

                let mut planner = RealFftPlanner::new();
                let r2c = planner.plan_fft_forward(size);
                let mut spectrum_realfft = r2c.make_output_vec();
                let mut scratch = r2c.make_scratch_vec();
                r2c.process_with_scratch(&mut orig_input, &mut spectrum_realfft, &mut scratch).unwrap();

                for (i, (p, r)) in spectrum.iter().zip(spectrum_realfft.iter()).enumerate() {
                    let diff = (p - r).norm();
                    assert!(
                        diff < 1e-4,
                        "Real part mismatch at size {}, index {}: real_fft {}, realfft {}, diff {}",
                        size,
                        i,
                        p,
                        r,
                        diff
                    );
                }
            }
        }
    }
}
