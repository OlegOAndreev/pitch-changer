use std::f32::consts::PI;
use std::mem;

use realfft::num_complex::Complex;

use crate::util::{approx_atan2, approx_sincos, normalize_phase};

// BinWithMagnitude stores bin magnitude + bin index compacted into once i64 value. This allows much faster sorting of
// bins by magnitude.
#[derive(PartialEq, Eq, PartialOrd, Ord, Debug)]
struct BinWithMagnitude {
    repr: i64,
}

impl BinWithMagnitude {
    fn new(index: usize, sqr_magnitude: f32) -> Self {
        let repr = (sqr_magnitude.to_bits() as i64) << 32 | index as i64;
        Self { repr }
    }

    fn index(&self) -> usize {
        (self.repr & 0xFFFFFFFF) as usize
    }

    #[allow(unused)]
    fn sqr_magnitude(&self) -> f32 {
        f32::from_bits((self.repr >> 32) as u32)
    }
}

/// Phase vocoder with the phase gradient implementation: the implementation of paper Phase Vocoder Done Right by
/// Zdeneˇk Pru ̊ša and Nicki Holighaus.
///
/// TLDR: Phase of a bin in the current frame can be either computed from phase of the same bin from the previous frame
/// or from the phase of a neighbor bin of the current frame.
pub struct PhaseGradientTimeStretch {
    fft_size: usize,
    num_bins: usize,

    prev_sqr_magnitudes: Vec<f32>,
    prev_ana_phases: Vec<f32>,
    prev_syn_phases: Vec<f32>,

    sqr_magnitudes: Vec<f32>,
    ana_phases: Vec<f32>,
    syn_phases: Vec<f32>,
    // Unlike the original paper, we do not use max-heap for all operations. Heap operations take more than ~50% of all
    // time in PitchShifter and we try to optimize them as much as we can.
    //
    // Instead we slightly modify the original algorithm: we sort the bins from the previous frame, propagate the phase
    // from the largest bin from the previous frame and then try to propagate the phase to the neighbors in the current
    // frame.
    //
    // This results in almost the same phase assignment as the previous algorithm, but is considerably faster.
    prev_sorted_bins: Vec<BinWithMagnitude>,
    // If true, syn_phases is assigned.
    phase_assigned: Vec<bool>,
}

// The implementation uses inline(never) in a few places to make profiling easier.
impl PhaseGradientTimeStretch {
    // Ignore the frequencies with magnitude < (max magnitude * MIN_MAGNITUDE_TOLERANCE). Note that we deal with
    // squared magnitudes.
    const MIN_MAGNITUDE_TOLERANCE: f32 = 1e-6;

    /// Create a new phase gradient time stretcher with given FFT size.
    pub fn new(fft_size: usize) -> Self {
        let num_bins = fft_size / 2 + 1;

        let prev_sqr_magnitudes = vec![0.0; num_bins];
        let prev_ana_phases = vec![0.0; num_bins];
        let prev_syn_phases = vec![0.0; num_bins];

        // Scratch buffers
        let sqr_magnitudes = vec![0.0; num_bins];
        let ana_phases = vec![0.0; num_bins];
        let syn_phases = vec![0.0; num_bins];
        let prev_sorted_bins = Vec::with_capacity(num_bins);
        let phase_assigned = vec![false; num_bins];

        Self {
            fft_size,
            num_bins,
            prev_sqr_magnitudes,
            prev_ana_phases,
            prev_syn_phases,
            sqr_magnitudes,
            ana_phases,
            syn_phases,
            prev_sorted_bins,
            phase_assigned,
        }
    }

    /// Process a single STFT frame.
    pub fn process(
        &mut self,
        ana_freq: &[Complex<f32>],
        ana_hop_size: usize,
        syn_freq: &mut [Complex<f32>],
        syn_hop_size: usize,
    ) {
        let num_bins = self.num_bins;
        assert_eq!(ana_freq.len(), num_bins);
        assert_eq!(syn_freq.len(), num_bins);

        let min_magn = self.prepare_magnitude_and_phases(ana_freq);
        self.assign_phases(ana_hop_size as f32, syn_hop_size as f32);
        self.convert_to_freq(ana_freq, syn_freq, min_magn);

        // Save previous analysis data for next frame
        mem::swap(&mut self.prev_sqr_magnitudes, &mut self.sqr_magnitudes);
        mem::swap(&mut self.prev_ana_phases, &mut self.ana_phases);
    }

    // Helper that assigns the phase to bin k based on the phase of bin k in the previous frame.
    //
    // From the paper
    //   φ_s(m_h, n) ← φ_s(m_h, n − 1) + a_s / 2 * ((∆_t φ_a) (m_h, n − 1) + (∆_t φ_a) (m_h, n))
    //
    // We simplify it by removing one of the derivatives to
    //   φ_s(m_h, n) ← φ_s(m_h, n − 1) + a_s * (∆_t φ_a) (m_h, n),
    // where the derivative is
    //   (∆_t φ_a) (m_h, n) = 1 / a_a * [φ_a(m, n) − φ_a(m, n − 1) − 2πma_a / M]_2π + 2πm/M
    //
    // The idea is pretty simple: remove the "inherent" phase shift for the bin which is 2πk∆ / M where k is
    // the bin size and ∆ is the difference in time (account for difference between analysis and synthesis
    // time delta which is syn_hop_size - ana_hop_size). Then normalize phases to [-π, π) and compute
    // the "residual" delta between two analysis phases (this happens e.g. when the true frequency differs
    // from the bin frequency). After that interpolate (or extrapolate if syn_hop_size > ana_hop_size)
    // the "residual" phase difference.
    #[inline(always)]
    fn assign_phase_from_prev(
        &mut self,
        k: usize,
        ana_hop_size: f32,
        ana_phase_diff: f32,
        syn_hop_size: f32,
        syn_phase_diff: f32,
    ) {
        let ana_phase_derivative =
            normalize_phase(self.ana_phases[k] - self.prev_ana_phases[k] - ana_phase_diff * k as f32) / ana_hop_size;
        self.syn_phases[k] = self.prev_syn_phases[k] + ana_phase_derivative * syn_hop_size + syn_phase_diff * k as f32;
        self.phase_assigned[k] = true;
    }

    // Helper that assigns the phase to bin k from another bin k_src in the current frame.
    //
    // From the paper
    //   φ_s(m_h + 1, n) ← φ_s(m_h, n) + b_s / 2 ((∆_f φ_a) (m_h, n) + (∆_f φ_a) (m_h + 1, n))
    //
    // We simplify it by removing one of the derivatives to
    //   φ_s(m_h + 1, n) ← φ_s(m_h, n) + b_s * (∆_f φ_a) (m_h, n),
    // where the derivative is
    //   (∆_f φ_a) (m_h, n) =  1 / b_a * [φ_a(m, n) − φ_a(m − 1, n)]_2π
    //
    // In simpler terms, we lock the phase in bin k_src to phase in bin k.
    #[inline(always)]
    fn assign_phase_from_neighbor(&mut self, k: usize, k_src: usize) {
        self.syn_phases[k] = self.syn_phases[k_src] + (self.ana_phases[k] - self.ana_phases[k_src]);
        self.phase_assigned[k] = true;
    }

    /// Reset internal state (clear previous frame data).
    pub fn reset(&mut self) {
        self.prev_sqr_magnitudes.fill(0.0);
        self.prev_ana_phases.fill(0.0);
        self.prev_syn_phases.fill(0.0);
    }

    // Fill in the sqr_magnitudes and return magnitude threshold.
    #[inline(never)]
    fn prepare_magnitude_and_phases(&mut self, ana_freq: &[Complex<f32>]) -> f32 {
        let num_bins = self.num_bins;

        // Find maximum magnitude for thresholding
        let mut max_magn = 0.0f32;
        for k in 0..num_bins {
            // We compare squared magnitudes and call sqrt() only before computing the final value.
            let magn = ana_freq[k].norm_sqr();
            self.sqr_magnitudes[k] = magn;
            max_magn = max_magn.max(magn).max(self.prev_sqr_magnitudes[k]);
        }
        let min_magn = max_magn * Self::MIN_MAGNITUDE_TOLERANCE;

        self.prev_sorted_bins.clear();

        for k in 0..num_bins {
            // Optimization: do not compute phase for frequencies below the min_magn threshold.
            if self.sqr_magnitudes[k] > min_magn {
                // Use approximation instead of ana_freq[k].arg();
                self.ana_phases[k] = approx_atan2(ana_freq[k].im, ana_freq[k].re);
                self.phase_assigned[k] = false;
                self.prev_sorted_bins.push(BinWithMagnitude::new(k, self.prev_sqr_magnitudes[k]));
            } else {
                // The original paper assigns random values to frequencies below the min magnitude, but we simply zero
                // them out.
                self.ana_phases[k] = 0.0;
                self.syn_phases[k] = 0.0;
                self.phase_assigned[k] = true;
            }
        }
        self.prev_sorted_bins.sort_unstable();

        min_magn
    }

    // Assign phases to syn_phases
    #[inline(never)]
    fn assign_phases(&mut self, ana_hop_size: f32, syn_hop_size: f32) {
        let num_bins = self.num_bins;

        // Original phase diff between frames for bin k = k * 2.0 * PI * ana_hop_size / fft_size.
        let ana_phase_diff = 2.0 * PI * ana_hop_size / self.fft_size as f32;
        let syn_phase_diff = 2.0 * PI * syn_hop_size / self.fft_size as f32;

        // Phase assignment algorithm: start with the bin with max magnitude.
        for bin in (0..self.prev_sorted_bins.len()).rev() {
            let k = self.prev_sorted_bins[bin].index();
            if self.phase_assigned[k] {
                continue;
            }

            // 1. Assign phase for bin k from previous frame.
            self.assign_phase_from_prev(k, ana_hop_size, ana_phase_diff, syn_hop_size, syn_phase_diff);

            // 2. Try propagating the phase to the bins on the left.
            if k > 0 {
                if self.phase_assigned[k - 1] {
                    continue;
                }
                // Check if we should propagate phase from previous or current frame.
                if self.sqr_magnitudes[k] < self.prev_sqr_magnitudes[k - 1] {
                    continue;
                }
                self.assign_phase_from_neighbor(k - 1, k);
            }

            // 3. Try propagating the phase to the bin on the right.
            if k < num_bins - 1 {
                if self.phase_assigned[k + 1] {
                    continue;
                }
                // Check if we should propagate from previous or current frame.
                if self.sqr_magnitudes[k] < self.prev_sqr_magnitudes[k + 1] {
                    continue;
                }
                self.assign_phase_from_neighbor(k + 1, k);
            }
        }
    }

    // Convert syn_phases back to syn_freq.
    #[inline(never)]
    fn convert_to_freq(&mut self, ana_freq: &[Complex<f32>], syn_freq: &mut [Complex<f32>], min_magn: f32) {
        let num_bins = self.num_bins;

        // Convert phase/magnitude back to complex frequency domain
        for k in 0..num_bins {
            if self.sqr_magnitudes[k] > min_magn {
                assert!(self.phase_assigned[k], "BUG: Phase {} not assigned", k);
                // Do the normalize_phase here so that the prev_syn_phases does not become too large, reducing the
                // floating point error. It also allows us to use approx_sincos implementation.
                self.prev_syn_phases[k] = normalize_phase(self.syn_phases[k]);
                // Use approximation instead of Complex::from_polar(self.sqr_magnitudes[k].sqrt(),
                // self.prev_syn_phases[k])
                let magn = self.sqr_magnitudes[k].sqrt();
                let (sin, cos) = approx_sincos(self.prev_syn_phases[k]);
                syn_freq[k] = Complex::new(magn * cos, magn * sin);
            } else {
                // We do not care what to fill this bin with.
                self.prev_syn_phases[k] = self.ana_phases[k];
                syn_freq[k] = ana_freq[k];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phase_gradient_bin_sorting() {
        // Test sorting of BinWithMagnitude vectors
        let mut bins = vec![
            BinWithMagnitude::new(0, 1.0),
            BinWithMagnitude::new(1, 3.0),
            BinWithMagnitude::new(2, 0.5),
            BinWithMagnitude::new(3, 2.0),
            BinWithMagnitude::new(4, 1.5),
            BinWithMagnitude::new(5, 3.0),
        ];
        bins.sort_unstable();

        let magnitudes: Vec<f32> = bins.iter().map(|b| b.sqr_magnitude()).collect();
        let indices: Vec<usize> = bins.iter().map(|b| b.index()).collect();
        assert_eq!(magnitudes, vec![0.5, 1.0, 1.5, 2.0, 3.0, 3.0]);
        assert_eq!(indices, vec![2, 0, 4, 3, 1, 5]);
    }

    #[test]
    fn test_phase_gradient_bin_edge_cases() {
        // Test with very large magnitude
        let bin = BinWithMagnitude::new(10, f32::MAX);
        assert_eq!(bin.sqr_magnitude(), f32::MAX);
        assert_eq!(bin.index(), 10);

        // Test with very small magnitude
        let bin = BinWithMagnitude::new(20, f32::MIN_POSITIVE);
        assert_eq!(bin.sqr_magnitude(), f32::MIN_POSITIVE);
        assert_eq!(bin.index(), 20);
    }
}
