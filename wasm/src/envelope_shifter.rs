use std::sync::Arc;

use realfft::num_complex::Complex;
use realfft::{ComplexToReal, RealToComplex};

use crate::util::{approx_exp2, approx_log2, linear_sample};

/// An implementation of spectral envelope shifting. It takes spectrum, computes the spectral envelope using cepstrum
/// and after that updates the spectrum magnitudes so that the spectrum envelopes "shifts down":
///   output_magnitude[i] = magnitude[i] * spectral_envelope[i * shift_ratio] / spectral_envelope[i]
///
/// The envelope is computed on a 4x downsampled spectrum (using max-pooling) for performance. Full-resolution envelope
/// values are obtained via linear interpolation of the downsampled envelope.
pub struct EnvelopeShifter {
    num_bins: usize,
    downsample_size: usize,
    cepstrum_cutoff_bins: usize,
    shift_ratio: f32,
    forward_plan: Arc<dyn RealToComplex<f32>>,
    inverse_plan: Arc<dyn ComplexToReal<f32>>,
    // Scratch buffers, all sized for the downsampled spectrum
    magnitudes_buf: Vec<f32>,
    new_magnitudes_buf: Vec<f32>,
    cepstrum_buf: Vec<Complex<f32>>,
    scratch_forward: Vec<Complex<f32>>,
    scratch_inverse: Vec<Complex<f32>>,
}

impl EnvelopeShifter {
    const DOWNSAMPLE_BY: usize = 4;

    pub fn new(num_bins: usize, cepstrum_cutoff_bins: usize, shift_ratio: f32) -> Self {
        use realfft::RealFftPlanner;

        let full_size = num_bins - 1;
        assert!(full_size.is_power_of_two());
        let downsample_size = full_size / Self::DOWNSAMPLE_BY;

        let mut planner = RealFftPlanner::<f32>::new();
        let forward_plan = planner.plan_fft_forward(downsample_size);
        let inverse_plan = planner.plan_fft_inverse(downsample_size);
        let cepstrum_size = downsample_size / 2 + 1;
        let magnitudes_buf = vec![0.0; downsample_size];
        let new_magnitudes_buf = vec![0.0; downsample_size];
        let cepstrum_buf = vec![Complex::ZERO; cepstrum_size];
        let scratch_forward = forward_plan.make_scratch_vec();
        let scratch_inverse = inverse_plan.make_scratch_vec();

        Self {
            num_bins,
            downsample_size,
            cepstrum_cutoff_bins,
            shift_ratio,
            forward_plan,
            inverse_plan,
            magnitudes_buf,
            new_magnitudes_buf,
            cepstrum_buf,
            scratch_forward,
            scratch_inverse,
        }
    }

    /// Compute spectral envelope, shift the frequencies magnitudes by pitch
    pub fn shift_envelope(&mut self, freq: &mut [Complex<f32>]) {
        assert_eq!(freq.len(), self.num_bins);

        self.fill_magnitudes_buf(freq);
        self.compute_envelope_impl();

        // magnitudes_buf now contains the spectral envelope at small resolution. Use linear interpolation to sample
        // both current and shifted envelope at full resolution.
        for k in 1..self.num_bins {
            let pos = (k - 1) as f32 / Self::DOWNSAMPLE_BY as f32;
            let cur_envelope = linear_sample(&self.magnitudes_buf, pos);
            if cur_envelope > 1e-7 {
                let shifted_envelope = linear_sample(&self.magnitudes_buf, pos * self.shift_ratio);
                freq[k] *= shifted_envelope / cur_envelope;
            }
        }
    }

    pub fn compute_envelope(&mut self, magnitudes: &[f32], output: &mut Vec<f32>) {
        assert_eq!(magnitudes.len(), self.num_bins);

        self.fill_magnitudes_buf_from_slice(magnitudes);
        self.compute_envelope_impl();

        output.clear();
        output.push(magnitudes[0]);
        for k in 0..self.num_bins - 1 {
            output.push(linear_sample(&self.magnitudes_buf, k as f32 / Self::DOWNSAMPLE_BY as f32));
        }
    }

    /// Update the pitch shift ratio.
    pub fn set_params(&mut self, cepstrum_cutoff_bins: usize, shift_ratio: f32) {
        self.cepstrum_cutoff_bins = cepstrum_cutoff_bins;
        self.shift_ratio = shift_ratio;
    }

    /// Downsample magnitudes of a freq slice into small_magnitudes_buf using max-pooling. Initially we store squared
    /// magnitudes in magnitudes_buf, see compute_envelope_impl.
    fn fill_magnitudes_buf(&mut self, freq: &[Complex<f32>]) {
        match Self::DOWNSAMPLE_BY {
            1 => {
                for i in 1..self.num_bins {
                    self.magnitudes_buf[i - 1] = freq[i].norm_sqr();
                }
            }
            2 => {
                for i in (1..self.num_bins).step_by(2) {
                    let a = freq[i].norm_sqr();
                    let b = freq[i + 1].norm_sqr();
                    self.magnitudes_buf[(i - 1) / 2] = a.max(b);
                }
            }
            4 => {
                for i in (1..self.num_bins).step_by(4) {
                    let a = freq[i].norm_sqr();
                    let b = freq[i + 1].norm_sqr();
                    let c = freq[i + 2].norm_sqr();
                    let d = freq[i + 3].norm_sqr();
                    self.magnitudes_buf[(i - 1) / 4] = a.max(b).max(c.max(d));
                }
            }
            _ => {
                unreachable!("Downsampling by {} is not supported", Self::DOWNSAMPLE_BY);
            }
        }
    }

    /// Downsample a magnitude slice into small_magnitudes_buf using max-pooling. Initially we store squared
    /// magnitudes in magnitudes_buf, see compute_envelope_impl.
    fn fill_magnitudes_buf_from_slice(&mut self, magnitudes: &[f32]) {
        match Self::DOWNSAMPLE_BY {
            1 => {
                for i in 1..self.num_bins {
                    self.magnitudes_buf[i - 1] = magnitudes[i] * magnitudes[i];
                }
            }
            2 => {
                for i in (1..self.num_bins).step_by(2) {
                    let a = magnitudes[i] * magnitudes[i];
                    let b = magnitudes[i + 1] * magnitudes[i + 1];
                    self.magnitudes_buf[(i - 1) / 2] = a.max(b);
                }
            }
            4 => {
                for i in (1..self.num_bins).step_by(4) {
                    let a = magnitudes[i] * magnitudes[i];
                    let b = magnitudes[i + 1] * magnitudes[i + 1];
                    let c = magnitudes[i + 2] * magnitudes[i + 2];
                    let d = magnitudes[i + 3] * magnitudes[i + 3];
                    self.magnitudes_buf[(i - 1) / 4] = a.max(b).max(c.max(d));
                }
            }
            _ => {
                unreachable!("Downsampling by {} is not supported", Self::DOWNSAMPLE_BY);
            }
        }
    }

    /// Compute the spectral envelope in-place on small_magnitudes_buf using iterative cepstral smoothing.
    fn compute_envelope_impl(&mut self) {
        const EPSILON: f32 = 1e-6;
        // This code implements true envelope estimation by doing multiple iterations of spectrum -> cepstrum ->
        // cepstrum cutoff -> spectrum loop. The more iterations you set, the larger the performance hit, and the more
        // accurate the estimation is.
        const ITERATIONS: usize = 5;

        for k in 0..self.downsample_size {
            // Initially magnitudes_buf contains the squared magnitudes, see fill_magnitudes_buf. We user the equality
            // log2(x^2) = log2(x) * 2 here.
            self.magnitudes_buf[k] = approx_log2(self.magnitudes_buf[k] + EPSILON) * 0.5;
        }

        let cutoff = self.cepstrum_cutoff_bins.min(self.cepstrum_buf.len());
        let norm = 1.0 / self.downsample_size as f32;
        for iteration in 0..ITERATIONS {
            self.forward_plan
                .process_with_scratch(&mut self.magnitudes_buf, &mut self.cepstrum_buf, &mut self.scratch_forward)
                .expect("failed forward STFT pass");

            self.cepstrum_buf[cutoff..].fill(Complex::ZERO);

            self.inverse_plan
                .process_with_scratch(&mut self.cepstrum_buf, &mut self.new_magnitudes_buf, &mut self.scratch_inverse)
                .expect("failed inverse STFT pass");

            if iteration < ITERATIONS - 1 {
                for k in 0..self.downsample_size {
                    self.magnitudes_buf[k] = self.magnitudes_buf[k].max(self.new_magnitudes_buf[k] * norm);
                }
            }
        }

        let upper_bin = self.downsample_size * 3 / 4;
        for k in 0..upper_bin {
            self.magnitudes_buf[k] = approx_exp2(self.new_magnitudes_buf[k] * norm);
        }

        // Make the upper part of envelope constant (see www.diva-portal.org/smash/get/diva2%3A1381398/FULLTEXT01.pdf)
        for k in upper_bin..self.downsample_size {
            self.magnitudes_buf[k] = self.magnitudes_buf[upper_bin - 1];
        }
    }
}
