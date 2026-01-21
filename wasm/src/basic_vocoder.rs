use std::f32::consts::PI;

use realfft::num_complex::Complex;

use crate::util::normalize_phase;

/// Basic vocoder
pub struct BasicVocoder {
    num_bins: usize,

    prev_ana_phases: Vec<f32>,
    prev_syn_phases: Vec<f32>,

    // Scratch buffers
    syn_magnitudes: Vec<f32>,
    syn_phase_diffs: Vec<f32>,
    max_magnitudes: Vec<f32>,
}

impl BasicVocoder {
    pub fn new(fft_size: usize) -> Self {
        let num_bins = fft_size / 2 + 1;
        let prev_ana_phases = vec![0.0; num_bins];
        let prev_syn_phases = vec![0.0; num_bins];
        let syn_magnitudes = vec![0.0; num_bins];
        let syn_phase_diffs = vec![0.0; num_bins];
        let max_magnitudes = vec![0.0; num_bins];

        Self { num_bins, prev_ana_phases, prev_syn_phases, syn_magnitudes, syn_phase_diffs, max_magnitudes }
    }

    /// Standard phase vocoder implementation: takes frequency data and computes output for given channel with given
    /// pitch shift.
    pub fn stretch_freq(
        &mut self,
        freq: &[Complex<f32>],
        pitch_shift: f32,
        overlap: usize,
        dst_freq: &mut [Complex<f32>],
    ) {
        let num_bins = freq.len();
        assert_eq!(self.num_bins, num_bins);

        // Overlap must be power-of-two, so use this fact for optimization: replace the k % overlap with
        // k & overlap_mask.
        let overlap_mask = overlap - 1;
        if (overlap_mask & overlap) != 0 {
            panic!("Overlap must be power of two");
        }
        // orig_phase_mult is used to calculate the expected phase difference for a basis for frequency bin k:
        // e^(2 * pi * k * t / num_bins) = e^(2 * pi * k / overlap) when t = hop_size, therefore the expected phase
        // at t = hop_size equals orig_phase_mult * k.
        //
        // In order to constrain the phase difference to [0, 2 * pi), simply replace the k with (k % overlap):
        // orig_phase_mult * (k % overlap). As noted above, this is equivalent to orig_phase_mult * (k & overlap_mask).
        let orig_phase_mult = 2.0 * PI / overlap as f32;

        // Initialize scratch buffers
        self.syn_magnitudes.fill(0.0);
        self.syn_phase_diffs.fill(0.0);
        self.max_magnitudes.fill(0.0);

        #[allow(clippy::needless_range_loop)]
        for k in 0..num_bins {
            let new_k = (k as f32 * pitch_shift) as usize;
            if new_k >= num_bins {
                break;
            }

            let magn = freq[k].norm();
            let max_magn = magn > self.max_magnitudes[new_k];
            if max_magn {
                self.max_magnitudes[new_k] = magn;
            }
            // If pitch_shift < 1.0, several analysis frequency bins may correspond to one synthesis bin, sum the
            // magnitudes.
            self.syn_magnitudes[new_k] += magn;

            let phase = freq[k].arg();

            // If multiple old phase bins get into one new phase bin (when pitch_shift < 1.0), we add their magnitudes
            // but want to choose only one phase (summing up phases from multiple bins make no sense). Therefore,
            // we separately compute new_phase_diffs and then add them to prev_new_phases in the end when finally
            // computing dst_freq. Ideally we would choose the phase from the bin with the highest magnitude here.
            if max_magn {
                // Original phase diff is the difference between the potential phase (phase of the frequency bin k
                // at the end of the previous block) and the actual phase for the frequency bin k at the start of
                // the new block. This phase diff is then applied to the stretched frequencies.
                let ana_phase_diff =
                    normalize_phase(phase - self.prev_ana_phases[k] - orig_phase_mult * ((k & overlap_mask) as f32));

                // The following two lines are equivalent to
                //   syn_phase_diffs[new_k] = pitch_shift * (ana_phase_diff + orig_phase_mult * k).
                // The main difference is that the syn_phase_diffs is much closer to zero, so that normalize_phase
                // would take less time later.
                let syn_phase_diff =
                    ana_phase_diff * pitch_shift + (k as f32 * pitch_shift - new_k as f32) * orig_phase_mult;
                self.syn_phase_diffs[new_k] = orig_phase_mult * ((new_k & overlap_mask) as f32) + syn_phase_diff;
            }

            self.prev_ana_phases[k] = phase;
        }

        // Compute output frequency bins
        #[allow(clippy::needless_range_loop)]
        for k in 0..num_bins {
            // Do the normalize_phase here so that the new_phases do not become too large, reducing the floating point
            // error.
            self.prev_syn_phases[k] = normalize_phase(self.prev_syn_phases[k] + self.syn_phase_diffs[k]);
            dst_freq[k] = Complex::from_polar(self.syn_magnitudes[k], self.prev_syn_phases[k]);
        }
    }

    pub fn reset(&mut self) {
        self.prev_ana_phases.fill(0.0);
        self.prev_syn_phases.fill(0.0);
    }
}
