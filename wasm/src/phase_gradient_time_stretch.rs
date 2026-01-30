use std::collections::BinaryHeap;
use std::f32::consts::PI;

use realfft::num_complex::Complex;

use crate::util::normalize_phase;

#[derive(Debug, Clone, Copy, PartialEq)]
struct HeapElem {
    pub bin: usize,
    pub magnitude: f32,
    // True if the element is from the previous STFT frame, false if from the current frame.
    pub from_prev: bool,
}

impl Eq for HeapElem {}

impl PartialOrd for HeapElem {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapElem {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.magnitude.total_cmp(&other.magnitude)
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

    prev_magnitudes: Vec<f32>,
    prev_ana_phases: Vec<f32>,
    prev_syn_phases: Vec<f32>,

    magnitudes: Vec<f32>,
    ana_phases: Vec<f32>,
    syn_phases: Vec<f32>,
    max_heap: BinaryHeap<HeapElem>,
    // If true, syn_phases is assigned.
    phase_assigned: Vec<bool>,
}

impl PhaseGradientTimeStretch {
    pub fn new(fft_size: usize) -> Self {
        let num_bins = fft_size / 2 + 1;

        let prev_magnitudes = vec![0.0; num_bins];
        let prev_ana_phases = vec![0.0; num_bins];
        let prev_syn_phases = vec![0.0; num_bins];

        // Scratch buffers
        let magnitudes = vec![0.0; num_bins];
        let ana_phases = vec![0.0; num_bins];
        let syn_phases = vec![0.0; num_bins];
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
            max_heap,
            phase_assigned,
        }
    }

    pub fn process(
        &mut self,
        ana_freq: &[Complex<f32>],
        ana_hop_size: usize,
        syn_freq: &mut [Complex<f32>],
        syn_hop_size: usize,
    ) {
        // Ignore the frequencies with magnitude < (max magnitude * MIN_MAGNITUDE_TOLERANCE).
        const MIN_MAGNITUDE_TOLERANCE: f32 = 1e-5;

        let freq_size = ana_freq.len();
        assert_eq!(self.num_bins, freq_size);
        // Original phase diff between frames for bin k = k * 2.0 * PI * ana_hop_size / fft_size.
        let ana_phase_diff = 2.0 * PI * ana_hop_size as f32 / self.fft_size as f32;
        let syn_phase_diff = 2.0 * PI * syn_hop_size as f32 / self.fft_size as f32;

        // Find maximum magnitude for thresholding
        let mut max_magn = 0.0f32;
        #[allow(clippy::needless_range_loop)]
        for k in 0..freq_size {
            let magn = ana_freq[k].norm();
            self.magnitudes[k] = magn;
            max_magn = max_magn.max(self.magnitudes[k]).max(self.prev_magnitudes[k]);
        }
        let min_magn = max_magn * MIN_MAGNITUDE_TOLERANCE;

        self.max_heap.clear();
        self.syn_phases.fill(0.0);
        self.phase_assigned.fill(false);

        // Number of false values in self.phase_assigned
        let mut num_unassigned = 0;
        #[allow(clippy::needless_range_loop)]
        for k in 0..freq_size {
            // Optimization: do not compute phase for frequencies below the min_magn threshold.
            if self.magnitudes[k] > min_magn {
                self.ana_phases[k] = ana_freq[k].arg();
                self.phase_assigned[k] = false;
                num_unassigned += 1;
                self.max_heap
                    .push(HeapElem { bin: k, magnitude: self.prev_magnitudes[k], from_prev: true });
            } else {
                // The original paper assigns random values to frequencies below the min magnitude, but we simply leave
                // the phase to be 0.0
                self.ana_phases[k] = 0.0;
                self.phase_assigned[k] = true;
            }
        }

        // Phase assignment algorithm
        while num_unassigned > 0 {
            let top_elem = self.max_heap.pop().unwrap_or_else(|| {
                panic!("INTERNAL ERROR: no more elements remaining in the heap, {} still unassigned", num_unassigned)
            });
            let k = top_elem.bin;
            if top_elem.from_prev {
                // Element is from previous frame -- compute the phase based on previous frame only if the phase was not
                // already propagated from neighbor (see next branch of this if).
                if self.phase_assigned[k] {
                    continue;
                }
                self.phase_assigned[k] = true;
                num_unassigned -= 1;
                self.max_heap.push(HeapElem { bin: k, magnitude: self.magnitudes[k], from_prev: false });

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
                    num_unassigned -= 1;
                    self.max_heap
                        .push(HeapElem { bin: k - 1, magnitude: self.magnitudes[k - 1], from_prev: false });

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
                    num_unassigned -= 1;
                    self.max_heap
                        .push(HeapElem { bin: k + 1, magnitude: self.magnitudes[k + 1], from_prev: false });

                    self.syn_phases[k + 1] = self.syn_phases[k] + self.ana_phases[k + 1] - self.ana_phases[k];
                }
            }
        }

        // Convert phase/magnitude back to complex frequency domain
        #[allow(clippy::needless_range_loop)]
        for k in 0..freq_size {
            // Do the normalize_phase here so that the prev_syn_phases does not become too large, reducing the floating
            // point error.
            self.prev_syn_phases[k] = normalize_phase(self.syn_phases[k]);
            syn_freq[k] = Complex::from_polar(self.magnitudes[k], self.prev_syn_phases[k]);
        }
        // Save previous analysis data for next frame
        self.prev_magnitudes.copy_from_slice(&self.magnitudes);
        self.prev_ana_phases.copy_from_slice(&self.ana_phases);
    }

    pub fn reset(&mut self) {
        self.prev_magnitudes.fill(0.0);
        self.prev_ana_phases.fill(0.0);
        self.prev_syn_phases.fill(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heap_elem_ordering() {
        let mut heap = BinaryHeap::new();

        heap.push(HeapElem { bin: 0, magnitude: 1.5, from_prev: false });
        heap.push(HeapElem { bin: 1, magnitude: 3.2, from_prev: false });
        heap.push(HeapElem { bin: 2, magnitude: 0.8, from_prev: false });
        heap.push(HeapElem { bin: 3, magnitude: 2.7, from_prev: false });
        heap.push(HeapElem { bin: 2, magnitude: 2.7, from_prev: true });

        assert_eq!(heap.pop().unwrap().magnitude, 3.2);
        assert_eq!(heap.pop().unwrap().magnitude, 2.7);
        assert_eq!(heap.pop().unwrap().magnitude, 2.7);
        assert_eq!(heap.pop().unwrap().magnitude, 1.5);
        assert_eq!(heap.pop().unwrap().magnitude, 0.8);
        assert!(heap.is_empty());
    }
}
