use std::sync::Arc;

use realfft::num_complex::Complex;
use realfft::{ComplexToReal, RealToComplex};

use crate::util::{LinearSample, approx_exp2, approx_log2};

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
    orig_magnitudes_buf: Vec<f32>,
    new_magnitudes_buf: Vec<f32>,
    cepstrum_buf: Vec<Complex<f32>>,
    scratch_forward: Vec<Complex<f32>>,
    scratch_inverse: Vec<Complex<f32>>,
}

impl EnvelopeShifter {
    const DOWNSAMPLE_BY: usize = 4;

    const MAX_GAIN: f32 = 10.0;

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
        let orig_magnitudes_buf = vec![0.0; downsample_size];
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
            orig_magnitudes_buf,
            new_magnitudes_buf,
            cepstrum_buf,
            scratch_forward,
            scratch_inverse,
        }
    }

    /// Compute spectral envelope, shift the frequencies magnitudes by pitch
    pub fn shift_envelope(&mut self, freq: &mut [Complex<f32>]) {
        let num_bins = self.num_bins;
        assert_eq!(freq.len(), num_bins);

        self.fill_orig_magnitudes_buf(freq);
        self.compute_envelope_impl();

        // Find peak in spectral envelope and reduce shifting envelope for bins before it. See EFFICIENT SPECTRAL
        // ENVELOPE ESTIMATION AND ITS APPLICATION TO PITCH SHIFTING AND ENVELOPE PRESERVATION by A. R¨obel and X. Rodet
        // for explanation:
        //
        // > Further inspection of the problem reveals the following issues. The spectral envelope below the fundamental
        // > partial will generally have a rather steep slope towards 0. Pitch shifting down will therefore attenuate
        // > the fundamental and create a less complete sound perception. Moreover, due to the fact that the cepstral
        // > order needs to be adjusted to fit the highest fundamental frequency that is present in the whole signal,
        // > the formants that may be observed for lower pitched signals will be smoothed such that the sound is
        // > perceived as dull.
        let peak_bin = (Self::find_peak(&self.new_magnitudes_buf) + 1) * Self::DOWNSAMPLE_BY;
        let start_bin = if self.shift_ratio < 1.0 { (peak_bin as f32 * self.shift_ratio) as usize } else { 1 };
        let upper_bin = num_bins * 3 / 4;

        // new_magnitudes_buf now contains the spectral envelope at small resolution. Use linear interpolation to sample
        // both current and shifted envelope at full resolution.
        let mut cur_sample = LinearSample::new(1.0 / Self::DOWNSAMPLE_BY as f32);
        let mut shifted_sample = LinearSample::new(self.shift_ratio / Self::DOWNSAMPLE_BY as f32);
        // Do not shift envelope at all until start_bin.
        for _ in 1..start_bin {
            cur_sample.step();
            shifted_sample.step();
        }
        // Introduce envelope shift after start_bin
        for k in start_bin..peak_bin {
            let cur_envelope = cur_sample.sample(&self.new_magnitudes_buf);
            if cur_envelope > 1e-5 {
                let shifted_envelope = shifted_sample.sample(&self.new_magnitudes_buf);
                let alpha = (k - start_bin) as f32 / (peak_bin - start_bin) as f32;
                let ratio = (shifted_envelope / cur_envelope) * alpha + 1.0 - alpha;
                freq[k] *= ratio;
            }
            cur_sample.step();
            shifted_sample.step();
        }
        for k in peak_bin..upper_bin {
            let cur_envelope = cur_sample.sample(&self.new_magnitudes_buf);
            if cur_envelope > 1e-5 {
                let shifted_envelope = shifted_sample.sample(&self.new_magnitudes_buf);
                let mut ratio = shifted_envelope / cur_envelope;
                // Limit the gain for upper parts of spectrum
                if k > peak_bin * 10 && ratio > Self::MAX_GAIN {
                    ratio = Self::MAX_GAIN;
                }
                freq[k] *= ratio;
            }
            cur_sample.step();
            shifted_sample.step();
        }
    }

    pub fn compute_envelope(&mut self, magnitudes: &[f32], output: &mut Vec<f32>) {
        assert_eq!(magnitudes.len(), self.num_bins);

        self.fill_orig_magnitudes_buf_from_slice(magnitudes);
        self.compute_envelope_impl();

        output.clear();
        output.push(magnitudes[0]);

        let mut sample = LinearSample::new(1.0 / Self::DOWNSAMPLE_BY as f32);
        for _ in 0..self.num_bins - 1 {
            output.push(sample.sample(&self.new_magnitudes_buf));
            sample.step();
        }
    }

    /// Update the pitch shift ratio.
    pub fn set_params(&mut self, cepstrum_cutoff_bins: usize, shift_ratio: f32) {
        self.cepstrum_cutoff_bins = cepstrum_cutoff_bins;
        self.shift_ratio = shift_ratio;
    }

    /// Downsample magnitudes of freq using max-pooling. Initially we store squared magnitudes in orig_magnitudes_buf,
    /// see compute_envelope_impl.
    fn fill_orig_magnitudes_buf(&mut self, freq: &[Complex<f32>]) {
        match Self::DOWNSAMPLE_BY {
            1 => {
                for (magn, freq) in self.orig_magnitudes_buf.iter_mut().zip(&freq[1..]) {
                    *magn = freq.norm_sqr();
                }
            }
            2 => {
                let (freq_chunks, freq_remainder) = freq[1..].as_chunks::<2>();
                assert!(freq_remainder.is_empty());
                for (magn, freq_chunk) in self.orig_magnitudes_buf.iter_mut().zip(freq_chunks) {
                    let a = freq_chunk[0].norm_sqr();
                    let b = freq_chunk[1].norm_sqr();
                    *magn = a.max(b);
                }
            }
            4 => {
                let (freq_chunks, freq_remainder) = freq[1..].as_chunks::<4>();
                assert!(freq_remainder.is_empty());
                for (magn, freq_chunk) in self.orig_magnitudes_buf.iter_mut().zip(freq_chunks) {
                    let a = freq_chunk[0].norm_sqr();
                    let b = freq_chunk[1].norm_sqr();
                    let c = freq_chunk[2].norm_sqr();
                    let d = freq_chunk[3].norm_sqr();
                    *magn = a.max(b).max(c.max(d));
                }
            }
            _ => {
                unreachable!("Downsampling by {} is not supported", Self::DOWNSAMPLE_BY);
            }
        }
    }

    /// Downsample a magnitude slice using max-pooling. Initially we store squared magnitudes in orig_magnitudes_buf,
    /// see compute_envelope_impl.
    fn fill_orig_magnitudes_buf_from_slice(&mut self, magnitudes: &[f32]) {
        match Self::DOWNSAMPLE_BY {
            1 => {
                for (magn, src) in self.orig_magnitudes_buf.iter_mut().zip(&magnitudes[1..]) {
                    *magn = src * src;
                }
            }
            2 => {
                let (magn_chunks, magn_remainder) = magnitudes[1..].as_chunks::<2>();
                assert!(magn_remainder.is_empty());
                for (magn, magn_chunk) in self.orig_magnitudes_buf.iter_mut().zip(magn_chunks) {
                    let a = magn_chunk[0] * magn_chunk[0];
                    let b = magn_chunk[1] * magn_chunk[1];
                    *magn = a.max(b);
                }
            }
            4 => {
                let (magn_chunks, magn_remainder) = magnitudes[1..].as_chunks::<4>();
                assert!(magn_remainder.is_empty());
                for (magn, magn_chunk) in self.orig_magnitudes_buf.iter_mut().zip(magn_chunks) {
                    let a = magn_chunk[0] * magn_chunk[0];
                    let b = magn_chunk[1] * magn_chunk[1];
                    let c = magn_chunk[2] * magn_chunk[2];
                    let d = magn_chunk[3] * magn_chunk[3];
                    *magn = a.max(b).max(c.max(d));
                }
            }
            _ => {
                unreachable!("Downsampling by {} is not supported", Self::DOWNSAMPLE_BY);
            }
        }
    }

    /// Compute the spectral envelope in-place on orig_magnitudes_buf using iterative cepstral smoothing.
    fn compute_envelope_impl(&mut self) {
        const EPSILON: f32 = 1e-6;
        // This code implements true envelope estimation by doing multiple iterations of spectrum -> cepstrum ->
        // cepstrum cutoff -> spectrum loop. The more iterations you set, the larger is the performance hit. Also, the
        // we envelope may become much larger for lower frequencies.
        const ITERATIONS: usize = 3;

        for magn in &mut self.orig_magnitudes_buf {
            // Initially orig_magnitudes_buf contains the squared magnitudes, see fill_magnitudes_buf. We use the
            // equality log2(x^2) = log2(x) * 2 here.
            *magn = approx_log2(*magn + EPSILON) * 0.5;
        }

        let cutoff = self.cepstrum_cutoff_bins.min(self.cepstrum_buf.len());
        let norm = 1.0 / self.downsample_size as f32;
        for iteration in 0..ITERATIONS {
            if iteration == 0 {
                // realfft destroys the input buffer, which is why we need to store orig_magnitudes_buf separately
                self.magnitudes_buf.copy_from_slice(&self.orig_magnitudes_buf);
            } else {
                for (magn, (orig_magn, new_magn)) in self
                    .magnitudes_buf
                    .iter_mut()
                    .zip(self.orig_magnitudes_buf.iter().zip(&self.new_magnitudes_buf))
                {
                    *magn = orig_magn.max(new_magn * norm);
                }
            }
            self.forward_plan
                .process_with_scratch(&mut self.magnitudes_buf, &mut self.cepstrum_buf, &mut self.scratch_forward)
                .expect("failed forward STFT pass");

            self.cepstrum_buf[cutoff..].fill(Complex::ZERO);

            self.inverse_plan
                .process_with_scratch(&mut self.cepstrum_buf, &mut self.new_magnitudes_buf, &mut self.scratch_inverse)
                .expect("failed inverse STFT pass");
        }

        let upper_bin = self.downsample_size * 3 / 4;
        for magn in &mut self.new_magnitudes_buf[0..upper_bin] {
            *magn = approx_exp2(*magn * norm);
        }

        // Make the upper part of envelope constant (see www.diva-portal.org/smash/get/diva2%3A1381398/FULLTEXT01.pdf)
        let fixed_magn = self.new_magnitudes_buf[upper_bin - 1];
        for magn in &mut self.new_magnitudes_buf[upper_bin..] {
            *magn = fixed_magn;
        }
    }

    // The returns the id of peak, which is defined as a bin which is greater than 4 bins before and after it or 0 if no
    // such bin exist.
    pub fn find_peak(spectrum: &[f32]) -> usize {
        assert!(spectrum.len() >= 9);
        // Process the first part.
        for i in 0..4 {
            if spectrum[i] < spectrum[i + 1]
                || spectrum[i] < spectrum[i + 2]
                || spectrum[i] < spectrum[i + 3]
                || spectrum[i] < spectrum[i + 4]
            {
                continue;
            }
            if i > 0 && spectrum[i] < spectrum[i - 1] {
                continue;
            }
            if i > 1 && spectrum[i] < spectrum[i - 2] {
                continue;
            }
            if i > 2 && spectrum[i] < spectrum[i - 3] {
                continue;
            }
            if i > 3 && spectrum[i] < spectrum[i - 4] {
                continue;
            }
            return i;
        }
        // Process the last part (we ignore the tail)
        for i in 4..spectrum.len() - 4 {
            if spectrum[i] < spectrum[i + 1]
                || spectrum[i] < spectrum[i + 2]
                || spectrum[i] < spectrum[i + 3]
                || spectrum[i] < spectrum[i + 4]
                || spectrum[i] < spectrum[i - 1]
                || spectrum[i] < spectrum[i - 2]
                || spectrum[i] < spectrum[i - 3]
                || spectrum[i] < spectrum[i - 4]
            {
                continue;
            }
            return i;
        }
        0
    }
}
