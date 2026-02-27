use std::collections::BinaryHeap;
use std::f32::consts::PI;
use std::mem;

use realfft::num_complex::Complex;

use crate::util::{approx_atan2, approx_sincos};

#[derive(Debug, Clone, Copy, PartialEq)]
struct PhaseGradientBin {
    // Micro-optimization: store usize as u32.
    pub index: u32,
    pub magnitude: f32,
}

impl PhaseGradientBin {
    fn new(index: usize, magnitude: f32) -> Self {
        Self { index: index as u32, magnitude: magnitude }
    }
}

impl Eq for PhaseGradientBin {}

impl PartialOrd for PhaseGradientBin {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PhaseGradientBin {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.magnitude.total_cmp(&other.magnitude)
    }
}

/// Phase vocoder with the phase gradient implementation: the implementation of paper Phase Vocoder Done Right by
/// Zdeneˇk Pru ̊ša and Nicki Holighaus.
///
/// TLDR: Phase of a bin in the current frame can be either computed from phase of the same bin from the previous frame
/// or from the phase of a neighbor bin of the current frame.
pub(crate) struct PhaseGradientTimeStretch {
    fft_size: usize,
    num_bins: usize,

    prev_magnitudes: Vec<f32>,
    prev_ana_phases: Vec<f32>,
    prev_syn_phases: Vec<f32>,

    magnitudes: Vec<f32>,
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

        let prev_magnitudes = vec![0.0; num_bins];
        let prev_ana_phases = vec![0.0; num_bins];
        let prev_syn_phases = vec![0.0; num_bins];

        // Scratch buffers
        let magnitudes = vec![0.0; num_bins];
        let ana_phases = vec![0.0; num_bins];
        let syn_phases = vec![0.0; num_bins];
        let prev_max_heap = Vec::with_capacity(num_bins);
        let max_heap = BinaryHeap::with_capacity(num_bins);
        let phase_assigned = vec![false; num_bins];

        Self {
            fft_size,
            num_bins,
            prev_magnitudes,
            prev_ana_phases,
            prev_syn_phases,
            magnitudes,
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

        // Ignore the frequencies with magnitude < (max magnitude * MIN_MAGNITUDE_TOLERANCE).
        const MIN_MAGNITUDE_TOLERANCE: f32 = 1e-3;

        let freq_size = ana_freq.len();
        assert_eq!(self.num_bins, freq_size);
        // Original phase diff between frames for bin k = k * 2.0 * PI * ana_hop_size / fft_size.
        let ana_phase_diff = 2.0 * PI * ana_hop_size as f32 / self.fft_size as f32;
        let syn_phase_diff = 2.0 * PI * syn_hop_size as f32 / self.fft_size as f32;

        // Find maximum magnitude for thresholding
        let mut max_magn = 0.0f32;
        for k in 0..freq_size {
            let magn = ana_freq[k].norm();
            self.magnitudes[k] = magn;
            max_magn = max_magn.max(magn).max(self.prev_magnitudes[k]);
        }
        let min_magn = max_magn * MIN_MAGNITUDE_TOLERANCE;

        self.prev_max_heap.clear();
        self.max_heap.clear();
        self.syn_phases.fill(0.0);
        self.phase_assigned.fill(false);

        // Number of false values in self.phase_assigned
        let mut num_phase_unassigned = 0;
        for k in 0..freq_size {
            // Optimization: do not compute phase for frequencies below the min_magn threshold.
            if self.magnitudes[k] > min_magn {
                // Use approximation instead of ana_freq[k].arg();
                self.ana_phases[k] = approx_atan2(ana_freq[k].im, ana_freq[k].re);
                self.phase_assigned[k] = false;
                num_phase_unassigned += 1;
                self.prev_max_heap
                    .push(PhaseGradientBin { index: k as u32, magnitude: self.prev_magnitudes[k] });
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
                self.max_heap.push(PhaseGradientBin::new(k, self.magnitudes[k]));

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
                    self.max_heap.push(PhaseGradientBin::new(k - 1, self.magnitudes[k - 1]));

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

                if k < freq_size - 1 && !self.phase_assigned[k + 1] {
                    self.phase_assigned[k + 1] = true;
                    num_phase_unassigned -= 1;
                    self.max_heap.push(PhaseGradientBin::new(k + 1, self.magnitudes[k + 1]));

                    self.syn_phases[k + 1] = self.syn_phases[k] + self.ana_phases[k + 1] - self.ana_phases[k];
                }
            }
        }

        // Convert phase/magnitude back to complex frequency domain
        for k in 0..freq_size {
            // Do the normalize_phase here so that the prev_syn_phases does not become too large, reducing the floating
            // point error. It also allows us to use approx_sincos implementation.
            self.prev_syn_phases[k] = normalize_phase(self.syn_phases[k]);
            if self.magnitudes[k] > min_magn {
                // Use approximation instead of Complex::from_polar(self.magnitudes[k], self.prev_syn_phases[k])
                let (sin, cos) = approx_sincos(self.prev_syn_phases[k]);
                syn_freq[k] = Complex::new(self.magnitudes[k] * cos, self.magnitudes[k] * sin);
            } else {
                // We do not care what to fill this bin with.
                syn_freq[k] = ana_freq[k];
            }
        }
        // Save previous analysis data for next frame
        mem::swap(&mut self.prev_magnitudes, &mut self.magnitudes);
        mem::swap(&mut self.prev_ana_phases, &mut self.ana_phases);
    }

    // Find the next HeapElem with max magnited from prev_max_heap and max_heap, return true if it was from
    // prev_max_heap and the bin index.
    fn find_next_heap_elem(&mut self) -> (usize, bool) {
        let prev_top_magnitude = self.prev_max_heap.last().map_or(-1.0, |e| e.magnitude);
        let top_magnitude = self.max_heap.peek().map_or(-1.0, |e| e.magnitude);
        if prev_top_magnitude > top_magnitude {
            let top = self.prev_max_heap.pop().expect("INTERNAL ERROR: no more elements remaining in the heaps");
            (top.index as usize, true)
        } else {
            let top = self.max_heap.pop().expect("INTERNAL ERROR: no more elements remaining in the heaps");
            (top.index as usize, false)
        }
    }

    /// Reset internal state (clear previous frame data).
    pub fn reset(&mut self) {
        self.prev_magnitudes.fill(0.0);
        self.prev_ana_phases.fill(0.0);
        self.prev_syn_phases.fill(0.0);
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BinaryHeap;

    use super::*;

    #[test]
    fn test_heap_elem_ordering() {
        let mut heap = BinaryHeap::new();

        heap.push(PhaseGradientBin { index: 0, magnitude: 1.5 });
        heap.push(PhaseGradientBin { index: 1, magnitude: 3.2 });
        heap.push(PhaseGradientBin { index: 2, magnitude: 0.8 });
        heap.push(PhaseGradientBin { index: 3, magnitude: 2.7 });
        heap.push(PhaseGradientBin { index: 2, magnitude: 2.7 });

        assert_eq!(heap.pop().unwrap().magnitude, 3.2);
        assert_eq!(heap.pop().unwrap().magnitude, 2.7);
        assert_eq!(heap.pop().unwrap().magnitude, 2.7);
        assert_eq!(heap.pop().unwrap().magnitude, 1.5);
        assert_eq!(heap.pop().unwrap().magnitude, 0.8);
        assert!(heap.is_empty());
    }
}
