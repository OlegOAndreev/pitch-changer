use std::sync::Arc;

use realfft::num_complex::Complex;
use realfft::{ComplexToReal, RealToComplex};

use crate::util::linear_sample;

/// An implementation of spectral envelope shifting. It takes spectrum, computes the spectral envelope using cepstrum
/// and after that updates the spectrum magnitudes so that the spectrum envelopes "shifts down":
///   output_magnitude[i] = magnitude[i] * spectral_envelope[i * shift_ratio] / spectral_envelope[i]
pub struct EnvelopeShifter {
    num_bins: usize,
    cepstrum_cutoff_samples: usize,
    shift_ratio: f32,
    forward_plan: Arc<dyn RealToComplex<f32>>,
    inverse_plan: Arc<dyn ComplexToReal<f32>>,
    // Scratch buffers
    magnitudes_buf: Vec<f32>,
    new_magnitudes_buf: Vec<f32>,
    cepstrum_buf: Vec<Complex<f32>>,
    scratch_forward: Vec<Complex<f32>>,
    scratch_inverse: Vec<Complex<f32>>,
}

impl EnvelopeShifter {
    pub fn new(num_bins: usize, cepstrum_cutoff_samples: usize, shift_ratio: f32) -> Self {
        use realfft::RealFftPlanner;

        // We use num_bins - 1 because we do not touch the DC anyway and num_bins-1 is very likely to be a power-of-two
        // (it is equal to original fft_size / 2 + 1).
        let mut planner = RealFftPlanner::<f32>::new();
        let forward_plan = planner.plan_fft_forward(num_bins - 1);
        let inverse_plan = planner.plan_fft_inverse(num_bins - 1);
        let cepstrum_size = (num_bins - 1) / 2 + 1;
        let magnitudes_buf = vec![0.0; num_bins - 1];
        let new_magnitudes_buf = vec![0.0; num_bins - 1];
        let cepstrum_buf = vec![Complex::ZERO; cepstrum_size];
        let scratch_forward = forward_plan.make_scratch_vec();
        let scratch_inverse = inverse_plan.make_scratch_vec();

        Self {
            num_bins,
            cepstrum_cutoff_samples,
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
        for k in 1..self.num_bins {
            self.magnitudes_buf[k - 1] = freq[k].norm();
        }
        self.compute_envelope_impl();

        // magnitudes_buf now contains the spectral envelope. We need to shift this envelope down by shift_ratio:
        // shifted_envelope[k] ~= envelope[k * pitch_shift],
        for k in 1..self.num_bins {
            let cur_envelope = self.magnitudes_buf[k - 1];
            let shifted_envelope = linear_sample(&self.magnitudes_buf, (k - 1) as f32 * self.shift_ratio);
            freq[k] *= shifted_envelope / cur_envelope;
        }
    }

    pub fn compute_envelope(&mut self, magnitudes: &[f32], output: &mut Vec<f32>) {
        assert_eq!(magnitudes.len(), self.num_bins);
        self.magnitudes_buf.copy_from_slice(&magnitudes[1..]);

        self.compute_envelope_impl();
        output.clear();
        output.push(magnitudes[0]);
        output.extend_from_slice(&self.magnitudes_buf);
    }

    // Assume that magnitudes_buf contains current magnitudes and update it to contain the envelope.
    fn compute_envelope_impl(&mut self) {
        const EPSILON: f32 = 1e-6;
        // This code implements true envelope estimation by doing multiple iterations of spectrum -> cepstrum ->
        // cepstrum cutoff -> spectrum loop. The more iterations you set, the larger the performance hit, and the more
        // accurate the estimation is.
        const ITERATIONS: usize = 5;

        for k in 0..self.num_bins - 1 {
            self.magnitudes_buf[k] = (self.magnitudes_buf[k] + EPSILON).log2();
        }

        let norm = 1.0 / (self.num_bins as f32 - 1.0);
        for _ in 0..ITERATIONS {
            self.forward_plan
                .process_with_scratch(&mut self.magnitudes_buf, &mut self.cepstrum_buf, &mut self.scratch_forward)
                .expect("failed forward STFT pass");

            // Do filtering
            for k in self.cepstrum_cutoff_samples..self.cepstrum_buf.len() {
                self.cepstrum_buf[k] = Complex::ZERO;
            }

            self.inverse_plan
                .process_with_scratch(&mut self.cepstrum_buf, &mut self.new_magnitudes_buf, &mut self.scratch_inverse)
                .expect("failed inverse STFT pass");

            for k in 0..self.num_bins - 1 {
                self.magnitudes_buf[k] = self.magnitudes_buf[k].max(self.new_magnitudes_buf[k] * norm);
            }
        }

        // Make the upper part of envelope constant (see www.diva-portal.org/smash/get/diva2%3A1381398/FULLTEXT01.pdf)
        for k in 0..self.num_bins - 1 {
            self.magnitudes_buf[k] = (self.new_magnitudes_buf[k] * norm).exp2();
        }

        let upper_bin = self.num_bins * 3 / 4;
        for k in upper_bin..self.num_bins - 1 {
            self.magnitudes_buf[k] = self.magnitudes_buf[upper_bin];
        }
    }
}
