use std::collections::BinaryHeap;
use std::f32::consts::PI;
use std::mem;

use realfft::num_complex::Complex;

use crate::util::{approx_atan2, approx_sincos};

// PhaseGradientBin stored bin magnitude + bin index compacted into once i64 value. This allows much faster sorting of bins by
// magnitude.
#[derive(PartialEq, Eq, PartialOrd, Ord, Debug)]
struct PhaseGradientBin {
    repr: i64,
}

impl PhaseGradientBin {
    fn new(magn: f32, index: usize) -> Self {
        let repr = (magn.to_bits() as i64) << 32 | index as i64;
        Self { repr }
    }

    fn index(&self) -> usize {
        (self.repr & 0xFFFFFFFF) as usize
    }

    fn magnitude(&self) -> f32 {
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
    // Unlike the original paper, we split the max-heap into two heaps, where the first heap is simply a sorted vector
    // of previous magnitudes. Heap operations take ~50% of all time in PitchShifter and we try to optimize them as much
    // as we can. By segregating the HeapElem from prev stft frame and current stft frame, we reduce the BinaryHeap
    // size and can use sort, which is faster than a bunch of push/pop operations.
    prev_max_heap: Vec<PhaseGradientBin>,
    max_heap: BinaryHeap<PhaseGradientBin>,
    // If true, syn_phases is assigned.
    phase_assigned: Vec<bool>,
}

impl PhaseGradientTimeStretch {
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
        let prev_max_heap = Vec::with_capacity(num_bins);
        let max_heap = BinaryHeap::with_capacity(num_bins);
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
            prev_max_heap,
            max_heap,
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
        use crate::util::normalize_phase;

        // Ignore the frequencies with magnitude < (max magnitude * MIN_MAGNITUDE_TOLERANCE). Note that we process
        // squared magnitudes.
        const MIN_MAGNITUDE_TOLERANCE: f32 = 1e-6;

        let num_bins = self.num_bins;
        assert_eq!(ana_freq.len(), num_bins);
        assert_eq!(syn_freq.len(), num_bins);
        // Original phase diff between frames for bin k = k * 2.0 * PI * ana_hop_size / fft_size.
        let ana_phase_diff = 2.0 * PI * ana_hop_size as f32 / self.fft_size as f32;
        let syn_phase_diff = 2.0 * PI * syn_hop_size as f32 / self.fft_size as f32;

        // Find maximum magnitude for thresholding
        let mut max_magn = 0.0f32;
        for k in 0..num_bins {
            // We compare squared magnitudes and call sqrt() only before computing the final value.
            let magn = ana_freq[k].norm_sqr();
            self.sqr_magnitudes[k] = magn;
            max_magn = max_magn.max(magn).max(self.prev_sqr_magnitudes[k]);
        }
        let min_magn = max_magn * MIN_MAGNITUDE_TOLERANCE;

        //

        self.prev_max_heap.clear();
        self.max_heap.clear();
        self.phase_assigned.fill(false);

        // Number of false values in self.phase_assigned
        let mut num_phase_unassigned = 0;
        for k in 0..num_bins {
            // Optimization: do not compute phase for frequencies below the min_magn threshold.
            if self.sqr_magnitudes[k] > min_magn {
                // Use approximation instead of ana_freq[k].arg();
                self.ana_phases[k] = approx_atan2(ana_freq[k].im, ana_freq[k].re);
                self.phase_assigned[k] = false;
                num_phase_unassigned += 1;
                self.prev_max_heap.push(PhaseGradientBin::new(self.prev_sqr_magnitudes[k], k));
            } else {
                // The original paper assigns random values to frequencies below the min magnitude, but we simply zero
                // them out.
                self.ana_phases[k] = 0.0;
                self.syn_phases[k] = 0.0;
                self.phase_assigned[k] = true;
            }
        }
        self.prev_max_heap.sort_unstable();

        // Phase assignment algorithm
        while num_phase_unassigned > 0 {
            let (k, from_prev) = self.find_next_heap_elem();
            if from_prev {
                // Element is from previous frame -- compute the phase based on previous frame only if the phase was not
                // already propagated from neighbor (see next branch of this if).
                if self.phase_assigned[k] {
                    continue;
                }
                self.phase_assigned[k] = true;
                num_phase_unassigned -= 1;
                self.max_heap.push(PhaseGradientBin::new(self.sqr_magnitudes[k], k));

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
                let ana_phase_derivative =
                    normalize_phase(self.ana_phases[k] - self.prev_ana_phases[k] - ana_phase_diff * k as f32)
                        / ana_hop_size as f32;
                self.syn_phases[k] =
                    self.prev_syn_phases[k] + ana_phase_derivative * syn_hop_size as f32 + syn_phase_diff * k as f32;
            } else {
                // Element is from current frame - propagate phase gradient to neighbors
                if k > 0 && !self.phase_assigned[k - 1] {
                    self.phase_assigned[k - 1] = true;
                    num_phase_unassigned -= 1;
                    self.max_heap.push(PhaseGradientBin::new(self.sqr_magnitudes[k - 1], k - 1));

                    // From the paper
                    //   φ_s(m_h + 1, n) ← φ_s(m_h, n) + b_s / 2 ((∆_f φ_a) (m_h, n) + (∆_f φ_a) (m_h + 1, n))
                    //
                    // We simplify it by removing one of the derivatives to
                    //   φ_s(m_h + 1, n) ← φ_s(m_h, n) + b_s * (∆_f φ_a) (m_h, n),
                    // where the derivative is
                    //   (∆_f φ_a) (m_h, n) =  1 / b_a * [φ_a(m, n) − φ_a(m − 1, n)]_2π
                    //
                    // In simpler terms, we lock the phase in bin k-1 to phase in bin k.
                    self.syn_phases[k - 1] = self.syn_phases[k] + self.ana_phases[k - 1] - self.ana_phases[k];
                }

                if k < self.num_bins - 1 && !self.phase_assigned[k + 1] {
                    self.phase_assigned[k + 1] = true;
                    num_phase_unassigned -= 1;
                    self.max_heap.push(PhaseGradientBin::new(self.sqr_magnitudes[k + 1], k + 1));

                    self.syn_phases[k + 1] = self.syn_phases[k] + self.ana_phases[k + 1] - self.ana_phases[k];
                }
            }
        }

        // Convert phase/magnitude back to complex frequency domain
        for k in 0..num_bins {
            // Do the normalize_phase here so that the prev_syn_phases does not become too large, reducing the floating
            // point error. It also allows us to use approx_sincos implementation.
            self.prev_syn_phases[k] = normalize_phase(self.syn_phases[k]);
            if self.sqr_magnitudes[k] > min_magn {
                // Use approximation instead of Complex::from_polar(self.sqr_magnitudes[k].sqrt(), self.prev_syn_phases[k])
                let magn = self.sqr_magnitudes[k].sqrt();
                let (sin, cos) = approx_sincos(self.prev_syn_phases[k]);
                syn_freq[k] = Complex::new(magn * cos, magn * sin);
            } else {
                // We do not care what to fill this bin with.
                syn_freq[k] = ana_freq[k];
            }
        }
        // Save previous analysis data for next frame
        mem::swap(&mut self.prev_sqr_magnitudes, &mut self.sqr_magnitudes);
        mem::swap(&mut self.prev_ana_phases, &mut self.ana_phases);
    }

    // Find the next HeapElem with max magnited from prev_max_heap and max_heap, return true if it was from
    // prev_max_heap and the bin index.
    fn find_next_heap_elem(&mut self) -> (usize, bool) {
        let prev_top_magnitude = self.prev_max_heap.last().map_or(-1.0, |e| e.magnitude());
        let top_magnitude = self.max_heap.peek().map_or(-1.0, |e| e.magnitude());
        if prev_top_magnitude > top_magnitude {
            let top = self.prev_max_heap.pop().expect("INTERNAL ERROR: no more elements remaining in the heaps");
            (top.index() as usize, true)
        } else {
            let top = self.max_heap.pop().expect("INTERNAL ERROR: no more elements remaining in the heaps");
            (top.index() as usize, false)
        }
    }

    /// Reset internal state (clear previous frame data).
    pub fn reset(&mut self) {
        self.prev_sqr_magnitudes.fill(0.0);
        self.prev_ana_phases.fill(0.0);
        self.prev_syn_phases.fill(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phase_gradient_bin_binary_heap_ordering() {
        // Test that BinaryHeap returns bins in descending magnitude order
        let mut heap = BinaryHeap::new();

        heap.push(PhaseGradientBin::new(1.5, 1));
        heap.push(PhaseGradientBin::new(3.0, 2));
        heap.push(PhaseGradientBin::new(0.5, 3));
        heap.push(PhaseGradientBin::new(2.0, 4));

        assert_eq!(heap.pop().unwrap().magnitude(), 3.0);
        assert_eq!(heap.pop().unwrap().magnitude(), 2.0);
        assert_eq!(heap.pop().unwrap().magnitude(), 1.5);
        assert_eq!(heap.pop().unwrap().magnitude(), 0.5);
        assert!(heap.is_empty());
    }

    #[test]
    fn test_phase_gradient_bin_sorting() {
        // Test sorting of PhaseGradientBin vectors
        let mut bins = vec![
            PhaseGradientBin::new(1.0, 0),
            PhaseGradientBin::new(3.0, 1),
            PhaseGradientBin::new(0.5, 2),
            PhaseGradientBin::new(2.0, 3),
            PhaseGradientBin::new(1.5, 4),
            PhaseGradientBin::new(3.0, 5),
        ];
        bins.sort_unstable();

        let magnitudes: Vec<f32> = bins.iter().map(|b| b.magnitude()).collect();
        let indices: Vec<usize> = bins.iter().map(|b| b.index()).collect();
        assert_eq!(magnitudes, vec![0.5, 1.0, 1.5, 2.0, 3.0, 3.0]);
        assert_eq!(indices, vec![2, 0, 4, 3, 1, 5]);
    }

    #[test]
    fn test_phase_gradient_bin_edge_cases() {
        // Test with very large magnitude
        let bin = PhaseGradientBin::new(f32::MAX, 10);
        assert_eq!(bin.magnitude(), f32::MAX);
        assert_eq!(bin.index(), 10);

        // Test with very small magnitude
        let bin = PhaseGradientBin::new(f32::MIN_POSITIVE, 20);
        assert_eq!(bin.magnitude(), f32::MIN_POSITIVE);
        assert_eq!(bin.index(), 20);
    }
}
