use std::f32::consts::PI;
use std::mem;

use rustfft::num_complex::Complex;

use crate::float4::{Float4, SimdFloat4};
use crate::util::{approx_atan2, approx_sincos, normalize_phase, radix_sort_u32};

// CompactBin stores bin magnitude + bin index compacted into one u32 value. This allows much faster sorting of bins by
// magnitude, because sorting takes a large chunk of time of process().
#[derive(PartialEq, Eq, PartialOrd, Ord, Debug, Clone, Copy)]
#[repr(transparent)]
struct CompactBin {
    // Upper bits: bits 15-30 from f32 representation (the f32 should be non-negative, so the sign bit should always be
    // zero), lower bits: index. This truncates the lower 15 bits of fraction, but it does not matter much: approximate
    // order by magnitude is ok.
    repr: u32,
}

impl CompactBin {
    fn new(index: usize, sqr_magnitude: f32) -> Self {
        assert!(sqr_magnitude >= 0.0, "Magnitude {} is not positive", sqr_magnitude);
        // Skip the sign bit as it should always be zero, store the remainder.
        let magnitude_bits = (sqr_magnitude.to_bits() << 1) & 0xFFFF0000;
        let repr = magnitude_bits | (index as u32);
        Self { repr }
    }

    fn index(&self) -> usize {
        (self.repr & 0xFFFF) as usize
    }

    #[allow(unused)]
    fn sqr_magnitude(&self) -> f32 {
        f32::from_bits((self.repr & 0xFFFF0000) >> 1)
    }

    // We guarantee that the [CompactBin] can be interpreted as [u32]
    fn as_u32_slice_mut(slice: &mut [Self]) -> &mut [u32] {
        unsafe { mem::transmute(slice) }
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

    // Per-bin data.
    prev_sqr_magnitudes: Vec<f32>,
    prev_sqr_magnitudes_max: f32,
    prev_ana_phases: Vec<f32>,
    prev_syn_phases: Vec<f32>,

    // Current-frame buffers.
    sqr_magnitudes: Vec<f32>,
    ana_phases: Vec<f32>,
    syn_phases: Vec<f32>,
    phase_assigned: Vec<bool>,

    // Unlike the original paper, we do not use max-heap for all operations. Heap operations take more than ~50% of all
    // time in PitchShifter and we try to optimize them as much as we can.
    //
    // Instead we slightly modify the original algorithm: we sort the bins from the previous frame, propagate the phase
    // from the largest bin from the previous frame and then try to propagate the phase to the neighbors in the current
    // frame.
    //
    // This results in almost the same phase assignment as the previous algorithm, but is considerably faster.
    prev_sorted_bins: Vec<CompactBin>,
    prev_sorted_bins_scratch: Vec<u32>,
}

// The implementation uses inline(never) in a few places to make profiling easier.
impl PhaseGradientTimeStretch {
    // Ignore the frequencies with magnitude < (max magnitude * MIN_MAGNITUDE_TOLERANCE). Note that we deal with
    // squared magnitudes.
    const MIN_MAGNITUDE_TOLERANCE: f32 = 1e-6;

    /// Create a new phase gradient time stretcher with given FFT size.
    pub fn new(fft_size: usize) -> Self {
        assert!(fft_size <= u16::MAX as usize + 1, "FFT size must be at most {}", u16::MAX as usize + 1);

        let num_bins = fft_size / 2 + 1;

        let prev_sorted_bins = Vec::with_capacity(num_bins);
        let prev_sorted_bins_scratch = Vec::with_capacity(num_bins);

        Self {
            fft_size,
            num_bins,
            prev_sqr_magnitudes: vec![0.0; num_bins],
            prev_sqr_magnitudes_max: 0.0,
            prev_ana_phases: vec![0.0; num_bins],
            prev_syn_phases: vec![0.0; num_bins],
            sqr_magnitudes: vec![0.0; num_bins],
            ana_phases: vec![0.0; num_bins],
            syn_phases: vec![0.0; num_bins],
            phase_assigned: vec![false; num_bins],
            prev_sorted_bins,
            prev_sorted_bins_scratch,
        }
    }

    /// Process a single STFT frame of time stretching.
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
        let k_f = k as f32;
        let ana_phase_derivative =
            normalize_phase(self.ana_phases[k] - self.prev_ana_phases[k] - ana_phase_diff * k_f) / ana_hop_size;
        self.syn_phases[k] = self.prev_syn_phases[k] + ana_phase_derivative * syn_hop_size + syn_phase_diff * k_f;
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

        // Compute squared magnitudes and max of them. Process 4 bins at a time using SIMD.
        let num_chunks = num_bins / 4;
        let ana_freq_ptr = ana_freq.as_ptr() as *const f32;
        let sqr_magn_ptr = self.sqr_magnitudes.as_mut_ptr();
        let mut magn_max_chunked = Float4::splat(0.0);
        for k in 0..num_chunks {
            // Load 4 complex numbers (8 interleaved f32 values), deinterleave into re and im.
            let (re, im) = unsafe { SimdFloat4::load_deinterleave2(ana_freq_ptr.add(k * 8)) };
            // sqr_magn = re*re + im*im
            let sqr_magn = SimdFloat4::add(SimdFloat4::mul(re, re), SimdFloat4::mul(im, im));
            unsafe {
                SimdFloat4::store(sqr_magn_ptr.add(k * 4), sqr_magn);
            }
            magn_max_chunked = Float4::max(magn_max_chunked, sqr_magn);
        }
        let mut cur_magn_max = Float4::horizontal_max(magn_max_chunked);
        for k in num_chunks * 4..num_bins {
            let magn = ana_freq[k].norm_sqr();
            self.sqr_magnitudes[k] = magn;
            cur_magn_max = cur_magn_max.max(magn).max(self.prev_sqr_magnitudes[k]);
        }
        let magn_max = cur_magn_max.max(self.prev_sqr_magnitudes_max);
        self.prev_sqr_magnitudes_max = cur_magn_max;

        let min_magn = magn_max * Self::MIN_MAGNITUDE_TOLERANCE;

        self.prev_sorted_bins.clear();

        for k in 0..num_bins {
            // Optimization: do not compute phase for frequencies below the min_magn threshold.
            if self.sqr_magnitudes[k] > min_magn {
                let freq = ana_freq[k];
                // Use approximation instead of ana_freq[k].arg();
                self.ana_phases[k] = approx_atan2(freq.im, freq.re);
                self.phase_assigned[k] = false;
                self.prev_sorted_bins.push(CompactBin::new(k, self.prev_sqr_magnitudes[k]));
            } else {
                // The original paper assigns random values to frequencies below the min magnitude, but we simply zero
                // them out.
                self.ana_phases[k] = 0.0;
                self.syn_phases[k] = 0.0;
                self.phase_assigned[k] = true;
            }
        }
        radix_sort_u32(CompactBin::as_u32_slice_mut(&mut self.prev_sorted_bins), &mut self.prev_sorted_bins_scratch);

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

            let sqr_magn_k = self.sqr_magnitudes[k];

            // 2. Try propagating the phase to the bins on the left.
            if k > 0 && !self.phase_assigned[k - 1] && sqr_magn_k >= self.prev_sqr_magnitudes[k - 1] {
                self.assign_phase_from_neighbor(k - 1, k);
            }

            // 3. Try propagating the phase to the bin on the right.
            if k < num_bins - 1 && !self.phase_assigned[k + 1] && sqr_magn_k >= self.prev_sqr_magnitudes[k + 1] {
                self.assign_phase_from_neighbor(k + 1, k);
            }
        }
    }

    // Convert syn_phases back to syn_freq.
    #[inline(never)]
    fn convert_to_freq(&mut self, ana_freq: &[Complex<f32>], syn_freq: &mut [Complex<f32>], min_magn: f32) {
        let num_bins = self.num_bins;

        for k in 0..num_bins {
            let sqr_magn = self.sqr_magnitudes[k];
            if sqr_magn > min_magn {
                assert!(self.phase_assigned[k], "BUG: Phase {} not assigned", k);
                // Do the normalize_phase here so that prev_syn_phases does not become too large, reducing the
                // floating point error. It also allows us to use approx_sincos implementation.
                let normalized = normalize_phase(self.syn_phases[k]);
                self.prev_syn_phases[k] = normalized;
                // Use approximation instead of Complex::from_polar(magn, normalized)
                let magn = sqr_magn.sqrt();
                let (sin, cos) = approx_sincos(normalized);
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
    fn test_compact_bin_sorting() {
        // Test sorting of CompactBin vectors
        let mut bins = vec![
            CompactBin::new(0, 1.0),
            CompactBin::new(1, 3.0),
            CompactBin::new(2, 0.5),
            CompactBin::new(3, 2.0),
            CompactBin::new(4, 1.5),
            CompactBin::new(5, 3.0),
        ];
        bins.sort_unstable();

        let magnitudes: Vec<f32> = bins.iter().map(|b| b.sqr_magnitude()).collect();
        let indices: Vec<usize> = bins.iter().map(|b| b.index()).collect();
        assert_eq!(magnitudes, vec![0.5, 1.0, 1.5, 2.0, 3.0, 3.0]);
        assert_eq!(indices, vec![2, 0, 4, 3, 1, 5]);
    }

    #[test]
    fn test_compact_bin_edge_cases() {
        // Test with very large magnitude. Note: we only store top 16 bits of f32 representation, so we lose some
        // precision
        let bin = CompactBin::new(10, f32::MAX);
        // When we reconstruct from 16 bits, we get an approximation
        let reconstructed = bin.sqr_magnitude();
        // Check that it's close (we keep top 16 bits of 32-bit float, losing lower 15 bits). The relative error should
        // be about 2^(-8) = 1/256 ≈ 0.004 for normalized numbers
        let expected = f32::MAX;
        let tolerance = expected * 0.005; // 1% tolerance
        assert!(
            (reconstructed - expected).abs() <= tolerance,
            "reconstructed={}, expected={}, diff={}",
            reconstructed,
            expected,
            (reconstructed - expected).abs()
        );
        assert_eq!(bin.index(), 10);

        // Test with very small magnitude
        let bin = CompactBin::new(20, f32::MIN_POSITIVE);
        // For very small values, the top 16 bits might be all zeros
        let reconstructed = bin.sqr_magnitude();
        // Check that it's either 0.0 or very close to MIN_POSITIVE
        if reconstructed != 0.0 {
            let expected = f32::MIN_POSITIVE;
            let tolerance = expected * 0.01; // 1% tolerance
            assert!(
                (reconstructed - expected).abs() <= tolerance,
                "reconstructed={}, expected={}, diff={}",
                reconstructed,
                expected,
                (reconstructed - expected).abs()
            );
        }
        assert_eq!(bin.index(), 20);
    }

    #[test]
    fn test_phase_gradient_bin_sorting_precision() {
        // Test that sorting works correctly even with 16-bit magnitude precision
        // Create bins with magnitudes that differ only in lower bits
        let mut bins = vec![
            CompactBin::new(0, 1.0),
            CompactBin::new(1, 1.0 + f32::EPSILON * 1000.0),
            CompactBin::new(2, 1.0 - f32::EPSILON * 1000.0),
            CompactBin::new(3, 2.0),
            CompactBin::new(4, 2.0 + f32::EPSILON * 1000.0),
        ];
        bins.sort_unstable();

        // With 16-bit precision, bins with very close magnitudes might get the same compressed representation and sort
        // by index instead of exact magnitude This is acceptable for the algorithm
        let indices: Vec<usize> = bins.iter().map(|b| b.index()).collect();

        // Basic checks: bins should be roughly sorted by magnitude. Index 3 (magnitude ~2.0) should come after indices
        // 0,1,2 (magnitude ~1.0)
        let pos_of_3 = indices.iter().position(|&i| i == 3).unwrap();
        let pos_of_0 = indices.iter().position(|&i| i == 0).unwrap();
        let pos_of_1 = indices.iter().position(|&i| i == 1).unwrap();
        let pos_of_2 = indices.iter().position(|&i| i == 2).unwrap();

        // Magnitude ~2.0 should come after magnitude ~1.0
        assert!(pos_of_3 > pos_of_0);
        assert!(pos_of_3 > pos_of_1);
        assert!(pos_of_3 > pos_of_2);

        // Index 4 (magnitude ~2.0) should be near index 3
        let pos_of_4 = indices.iter().position(|&i| i == 4).unwrap();
        // Should be close to index 3 (both ~2.0 magnitude)
        assert!((pos_of_4 as isize - pos_of_3 as isize).abs() <= 1);
    }
}
