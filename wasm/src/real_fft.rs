/// Replacement for realfft crate, which supports only 2^N powers. Ported from
/// https://github.com/HEnquist/realfft/blob/master/src/lib.rs
use std::f32;
use std::sync::Arc;

use anyhow::{Result, bail};

use rustfft::Fft;
use rustfft::num_complex::Complex;

pub struct FftRealToComplex {
    fft: Arc<dyn Fft<f32>>,
    size: usize,
    twiddles: Vec<Complex<f32>>,
}

impl FftRealToComplex {
    /// Create new FftRealToComplex
    pub fn new(size: usize) -> Result<FftRealToComplex> {
        if !size.is_power_of_two() || !size.is_multiple_of(4) {
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

        let output_f32 = complex_as_f32_slice_mut(output);
        unsafe {
            let re0 = output_f32[0];
            let im0 = output_f32[1];
            output_f32[0] = re0 + im0;
            output_f32[1] = 0.0;
            output_f32[self.size] = re0 - im0;
            output_f32[self.size + 1] = 0.0;

            for i in 1..self.size / 4 {
                let re_idx = 2 * i;
                let im_idx = 2 * i + 1;
                let re_rev = self.size - 2 * i;
                let im_rev = self.size - 2 * i + 1;

                let out_re = *output_f32.get_unchecked(re_idx);
                let out_im = *output_f32.get_unchecked(im_idx);
                let out_rev_re = *output_f32.get_unchecked(re_rev);
                let out_rev_im = *output_f32.get_unchecked(im_rev);

                let twiddle = *self.twiddles.get_unchecked(i - 1);

                let sum_re = out_re + out_rev_re;
                let sum_im = out_im + out_rev_im;
                let diff_re = out_re - out_rev_re;
                let diff_im = out_im - out_rev_im;

                let half_sum_re = 0.5 * sum_re;
                let half_diff_im = 0.5 * diff_im;

                let output_twiddled_real = sum_im * twiddle.re + diff_re * twiddle.im;
                let output_twiddled_im = sum_im * twiddle.im - diff_re * twiddle.re;

                *output_f32.get_unchecked_mut(re_idx) = half_sum_re + output_twiddled_real;
                *output_f32.get_unchecked_mut(im_idx) = half_diff_im + output_twiddled_im;
                *output_f32.get_unchecked_mut(re_rev) = half_sum_re - output_twiddled_real;
                *output_f32.get_unchecked_mut(im_rev) = -half_diff_im + output_twiddled_im;
            }

            output_f32[self.size / 2 + 1] *= -1.0;
        }

        Ok(())
    }

    /// Create a vector suitable for passing as scratch parameter to process.
    pub fn make_scratch_vec(&self) -> Vec<Complex<f32>> {
        vec![Complex::ZERO; self.size / 2]
    }
}

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
                let re_idx = 2 * i;
                let im_idx = 2 * i + 1;
                let re_rev = self.size - 2 * i;
                let im_rev = self.size - 2 * i + 1;

                let out_re = *input_f32.get_unchecked(re_idx);
                let out_im = *input_f32.get_unchecked(im_idx);
                let out_rev_re = *input_f32.get_unchecked(re_rev);
                let out_rev_im = *input_f32.get_unchecked(im_rev);

                let twiddle = *self.twiddles.get_unchecked(i - 1);

                let sum_re = out_re + out_rev_re;
                let sum_im = out_im + out_rev_im;
                let diff_re = out_re - out_rev_re;
                let diff_im = out_im - out_rev_im;

                let output_twiddled_real = sum_im * twiddle.re + diff_re * twiddle.im;
                let output_twiddled_im = sum_im * twiddle.im - diff_re * twiddle.re;

                *input_f32.get_unchecked_mut(re_idx) = sum_re - output_twiddled_real;
                *input_f32.get_unchecked_mut(im_idx) = diff_im - output_twiddled_im;

                *input_f32.get_unchecked_mut(re_rev) = sum_re + output_twiddled_real;
                *input_f32.get_unchecked_mut(im_rev) = -output_twiddled_im - diff_im;
            }

            input_f32[self.size / 2] *= 2.0;
            input_f32[self.size / 2 + 1] *= -2.0;
        }

        let fft_output = f32_as_complex_slice_mut(output);
        self.fft.process_outofplace_with_scratch(&mut input[..self.size / 2], fft_output, scratch);

        Ok(())
    }

    /// Create a vector suitable for passing as scratch parameter to process.
    pub fn make_scratch_vec(&self) -> Vec<Complex<f32>> {
        vec![Complex::ZERO; self.size / 2]
    }
}

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
            // for size in [8, 16, 256, 512, 1024, 2048, 4096] {
            // Generate random input in [-5, 5)
            let rng = SmallRng::seed_from_u64(size as u64);
            let mut input: Vec<f32> = rng.random_iter::<f32>().take(size).map(|x| x * 10.0 - 5.0).collect();
            let orig_input = input.clone();

            let forward = FftRealToComplex::new(size).unwrap();
            let mut spectrum = vec![Complex::ZERO; size / 2 + 1];
            let mut scratch = vec![Complex::ZERO; forward.get_scratch_len()];
            forward.process(&mut input, &mut spectrum, &mut scratch).unwrap();

            let mut in_buf = spectrum.clone();

            let backward = FftComplexToReal::new(size).unwrap();
            let mut output = vec![0.0; size];
            let mut backward_scratch = vec![Complex::ZERO; backward.get_scratch_len()];
            backward.process(&mut spectrum, &mut output, &mut backward_scratch).unwrap();

            let mut planner = RealFftPlanner::new();
            let c2r = planner.plan_fft_inverse(size);
            let mut output_rfft = c2r.make_output_vec();
            let mut another_scratch = c2r.make_scratch_vec();
            let _ = c2r.process_with_scratch(&mut in_buf, &mut output_rfft, &mut another_scratch);

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
        for size in [8, 16, 256, 512, 1024, 2048, 4096] {
            // Generate random input in [-5, 5)
            for i in 0..10 {
                let rng = SmallRng::seed_from_u64((size + i) as u64);
                let mut input: Vec<f32> = rng.random_iter::<f32>().take(size).map(|x| x * 10.0 - 5.0).collect();

                let forward = FftRealToComplex::new(size).unwrap();
                let mut spectrum = vec![Complex::ZERO; size / 2 + 1];
                let mut scratch = vec![Complex::ZERO; forward.get_scratch_len()];
                forward.process(&mut input, &mut spectrum, &mut scratch).unwrap();

                let mut planner = RealFftPlanner::new();
                let r2c = planner.plan_fft_forward(size);
                let mut in_buf = input.clone();
                let mut spectrum_realfft = r2c.make_output_vec();
                let mut scratch = r2c.make_scratch_vec();
                r2c.process_with_scratch(&mut in_buf, &mut spectrum_realfft, &mut scratch).unwrap();

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
