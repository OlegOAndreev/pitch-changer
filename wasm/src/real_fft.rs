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
        if !size.is_power_of_two() || size < 4 {
            bail!("FftRealToComplex size must be power of two, is {}", size);
        }
        let mut planner = rustfft::FftPlanner::new();
        let fft = planner.plan_fft_forward(size / 2);
        let twiddle_count = size / 4;
        let twiddles: Vec<Complex<f32>> = (1..twiddle_count).map(|i| compute_twiddle(i, size) * 0.5).collect();
        Ok(Self { fft, size, twiddles })
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
        if scratch.len() < self.size / 2 {
            bail!("Expected scratch size {}, got {}", self.size / 2, input.len());
        }

        let fft_input =
            unsafe { std::slice::from_raw_parts_mut(input.as_mut_ptr() as *mut Complex<f32>, input.len() / 2) };
        self.fft.process_outofplace_with_scratch(fft_input, &mut output[..self.size / 2], scratch);

        unsafe {
            let first = *output.get_unchecked(0);
            *output.get_unchecked_mut(0) = Complex { re: first.re + first.im, im: 0.0 };
            *output.get_unchecked_mut(self.size / 2) = Complex { re: first.re - first.im, im: 0.0 };

            for i in 1..self.size / 4 {
                let out = *output.get_unchecked(i);
                let out_rev = *output.get_unchecked(self.size / 2 - i);
                let twiddle = *self.twiddles.get_unchecked(i - 1);
                let sum = out + out_rev;
                let diff = out - out_rev;

                let twiddled_re_sum = sum.im * twiddle.re;
                let twiddled_im_sum = sum.im * twiddle.im;
                let twiddled_re_diff = diff.re * twiddle.re;
                let twiddled_im_diff = diff.re * twiddle.im;
                let half_sum_re = 0.5 * sum.re;
                let half_diff_im = 0.5 * diff.im;

                let output_twiddled_real = twiddled_re_sum + twiddled_im_diff;
                let output_twiddled_im = twiddled_im_sum - twiddled_re_diff;

                *output.get_unchecked_mut(i) =
                    Complex { re: half_sum_re + output_twiddled_real, im: half_diff_im + output_twiddled_im };
                *output.get_unchecked_mut(self.size / 2 - i) =
                    Complex { re: half_sum_re - output_twiddled_real, im: output_twiddled_im - half_diff_im };
            }
            output.get_unchecked_mut(self.size / 4).im = -output.get_unchecked(self.size / 4).im;
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
        if !size.is_power_of_two() || size < 4 {
            bail!("FftComplexToReal size must be power of two, is {}", size);
        }
        let mut planner = rustfft::FftPlanner::new();
        let fft = planner.plan_fft_inverse(size / 2);
        let twiddle_count = size / 4;
        let twiddles: Vec<Complex<f32>> = (1..twiddle_count).map(|i| compute_twiddle(i, size).conj()).collect();
        Ok(Self { fft, size, twiddles })
    }

    /// Process input and store the result in output. Input is sized the same way as realfft does: the first element is
    /// DC and the last element is Nyqist, both have zero immediate component. The input is used as a scratch buffer.
    pub fn process(&self, input: &mut [Complex<f32>], output: &mut [f32], scratch: &mut [Complex<f32>]) -> Result<()> {
        if input.len() != self.size / 2 + 1 {
            bail!("Expected input size {}, got {}", self.size / 2 + 1, input.len());
        }
        if output.len() != self.size {
            bail!("Expected output size {}, got {}", self.size, input.len());
        }
        if scratch.len() < self.size / 2 {
            bail!("Expected scratch size {}, got {}", self.size / 2, input.len());
        }

        unsafe {
            let first_re = input.get_unchecked(0).re + input.get_unchecked(self.size / 2).re;
            let first_im = input.get_unchecked(0).re - input.get_unchecked(self.size / 2).re;
            *input.get_unchecked_mut(0) = Complex { re: first_re, im: first_im };

            for i in 1..self.size / 4 {
                let out = *input.get_unchecked(i);
                let out_rev = *input.get_unchecked(self.size / 2 - i);
                let twiddle = *self.twiddles.get_unchecked(i - 1);
                let sum = out + out_rev;
                let diff = out - out_rev;

                let twiddled_re_sum = sum.im * twiddle.re;
                let twiddled_im_sum = sum.im * twiddle.im;
                let twiddled_re_diff = diff.re * twiddle.re;
                let twiddled_im_diff = diff.re * twiddle.im;

                let output_twiddled_real = twiddled_re_sum + twiddled_im_diff;
                let output_twiddled_im = twiddled_im_sum - twiddled_re_diff;

                // We finally have all the data we need to write our preprocessed data back where we got it from.
                *input.get_unchecked_mut(i) =
                    Complex { re: sum.re - output_twiddled_real, im: diff.im - output_twiddled_im };
                *input.get_unchecked_mut(self.size / 2 - i) =
                    Complex { re: sum.re + output_twiddled_real, im: -output_twiddled_im - diff.im };
            }
            *input.get_unchecked_mut(self.size / 4) = (input.get_unchecked(self.size / 4) * 2.0).conj();
        }

        let fft_output =
            unsafe { std::slice::from_raw_parts_mut(output.as_mut_ptr() as *mut Complex<f32>, output.len() / 2) };
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

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};
    use realfft::RealFftPlanner;

    #[test]
    fn test_realfft_roundtrip() {
        for size in [8, 16, 256, 512, 1024, 2048, 4096] {
            // Generate random input in [-5, 5)
            let rng = SmallRng::seed_from_u64(size as u64);
            let mut input: Vec<f32> = rng.random_iter::<f32>().take(size).map(|x| x * 10.0 - 5.0).collect();

            let forward = FftRealToComplex::new(size).unwrap();
            let mut spectrum = vec![Complex::ZERO; size / 2 + 1];
            let mut scratch = vec![Complex::ZERO; size / 2];
            forward.process(&mut input, &mut spectrum, &mut scratch).unwrap();

            let backward = FftComplexToReal::new(size).unwrap();
            let mut output = vec![0.0; size];
            backward.process(&mut spectrum, &mut output, &mut scratch).unwrap();

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
    fn test_realfft_vs_real_fft_forward() {
        for size in [8, 16, 256, 512, 1024, 2048, 4096] {
            // Generate random input in [-5, 5)
            let rng = SmallRng::seed_from_u64(size as u64);
            let mut input: Vec<f32> = rng.random_iter::<f32>().take(size).map(|x| x * 10.0 - 5.0).collect();

            let forward = FftRealToComplex::new(size).unwrap();
            let mut spectrum = vec![Complex::ZERO; size / 2 + 1];
            let mut scratch = vec![Complex::ZERO; size / 2];
            forward.process(&mut input, &mut spectrum, &mut scratch).unwrap();

            let mut planner = RealFftPlanner::new();
            let r2c = planner.plan_fft_forward(size);
            let mut in_buf = input.clone();
            let mut spectrum_realfft = r2c.make_output_vec();
            let mut scratch = r2c.make_scratch_vec();
            r2c.process_with_scratch(&mut in_buf, &mut spectrum_realfft, &mut scratch).unwrap();

            // Compare
            for (p, r) in spectrum.iter().zip(spectrum_realfft.iter()) {
                let diff_re = (p.re - r.re).abs();
                let diff_im = (p.im - r.im).abs();
                assert!(diff_re < 1e-4, "Real part mismatch: real_fft {}, realfft {}, diff {}", p.re, r.re, diff_re);
                assert!(diff_im < 1e-4, "Imag part mismatch: real_fft {}, realfft {}, diff {}", p.im, r.im, diff_im);
            }
        }
    }
}
