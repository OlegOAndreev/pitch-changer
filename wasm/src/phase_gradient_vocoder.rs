use std::collections::BinaryHeap;
use std::f32::consts::PI;

use realfft::num_complex::Complex;

use crate::util::normalize_phase;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HeapElem {
    pub freq_bin: usize,
    pub magnitude: f32,
    // True if the element is from the previous STFT frame, false if from the current frame.
    pub prev_frame: bool,
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

pub struct PhaseGradientVocoder {
    num_bins: usize,

    prev_ana_magnitudes: Vec<f32>,
    prev_ana_phases: Vec<f32>,
    prev_syn_phases: Vec<f32>,

    ana_magnitudes: Vec<f32>,
    ana_phases: Vec<f32>,
    syn_magnitudes: Vec<f32>,
    syn_phases: Vec<f32>,
    max_heap: BinaryHeap<HeapElem>,
    // If true, synPhases is assigned.
    phase_assigned: Vec<bool>,
}

impl PhaseGradientVocoder {
    pub fn new(fft_size: usize) -> Self {
        let num_bins = fft_size / 2 + 1;

        let prev_ana_magnitudes = vec![0.0; num_bins];
        let prev_ana_phases = vec![0.0; num_bins];
        let prev_syn_phases = vec![0.0; num_bins];

        // Scratch buffers
        let ana_magnitudes = vec![0.0; num_bins];
        let ana_phases = vec![0.0; num_bins];
        let syn_magnitudes = vec![0.0; num_bins];
        let syn_phases = vec![0.0; num_bins];
        let max_heap = BinaryHeap::with_capacity(num_bins);
        let phase_assigned = vec![false; num_bins];

        Self {
            num_bins,
            prev_ana_magnitudes,
            prev_ana_phases,
            prev_syn_phases,
            ana_magnitudes,
            ana_phases,
            syn_magnitudes,
            syn_phases,
            max_heap,
            phase_assigned,
        }
    }

    /// Phase vocoder with the phase gradient implementation: takes frequency data and computes output for given channel with given
    /// pitch shift.
    pub fn stretch_freq(
        &mut self,
        freq: &[Complex<f32>],
        pitch_shift: f32,
        overlap: usize,
        dst_freq: &mut [Complex<f32>],
    ) {
        // Ignore the frequencies with magnitude < (max magnitude * MIN_MAGNITUDE_TOLERANCE).
        const MIN_MAGNITUDE_TOLERANCE: f32 = 1e-3;

        let freq_size = freq.len();
        assert_eq!(self.num_bins, freq_size);

        // Overlap must be power-of-two, so use this fact for optimization: replace the k % overlap with
        // k & overlap_mask. You can always replace k * orig_phase_mult with (k % overlap) * orig_phase_mult.
        let overlap_mask = overlap - 1;
        if (overlap_mask & overlap) != 0 {
            panic!("Overlap must be pow-of-2");
        }
        let orig_phase_mult = 2.0 * PI / overlap as f32;

        // Find maximum magnitude for thresholding
        let mut max_magn = 0.0f32;
        #[allow(clippy::needless_range_loop)]
        for k in 0..freq_size {
            let magn = freq[k].norm();
            self.ana_magnitudes[k] = magn;
            max_magn = max_magn.max(self.ana_magnitudes[k]).max(self.prev_ana_magnitudes[k]);
        }
        let min_magn = max_magn * MIN_MAGNITUDE_TOLERANCE;

        self.max_heap.clear();
        self.syn_magnitudes.fill(0.0);
        self.syn_phases.fill(0.0);
        self.phase_assigned.fill(false);

        // Number of unassigned phases
        let mut num_unassigned = 0;
        #[allow(clippy::needless_range_loop)]
        for k in 0..freq_size {
            let new_k = (k as f32 * pitch_shift) as usize;
            if new_k >= freq_size {
                break;
            }

            let magn = self.ana_magnitudes[k];
            // If pitch_shift < 1.0, several analysis frequency bins may correspond to one synthesis bin, sum the
            // magnitudes.
            self.syn_magnitudes[new_k] += magn;

            // Optimization: do not compute phase for frequencies below the min_magn threshold.
            if magn > min_magn {
                self.ana_phases[k] = freq[k].arg();
                self.phase_assigned[k] = false;
                num_unassigned += 1;
                let prev_magn = self.prev_ana_magnitudes[k];
                self.max_heap.push(HeapElem { freq_bin: k, magnitude: prev_magn, prev_frame: true });
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
            let k = top_elem.freq_bin;
            if top_elem.prev_frame {
                // Element is from previous frame
                if self.phase_assigned[k] {
                    continue;
                }
                self.phase_assigned[k] = true;
                num_unassigned -= 1;
                self.max_heap
                    .push(HeapElem { freq_bin: k, magnitude: self.ana_magnitudes[k], prev_frame: false });

                let new_k = (k as f32 * pitch_shift) as usize;
                if new_k >= freq_size {
                    continue;
                }
                // Original phase diff is the difference between the potential phase (phase of the frequency bin k at
                // the end of the previous block) and the actual phase for the frequency bin k at the start of the new
                // block. This phase diff is then applied to the stretched frequencies. The final formula is simplified
                // a bit from the one from smbPitchShift.cpp Modulo overlapMask is important here, because otherwise the
                // origPhaseMult * k can be very large and normalizePhase expects values close to 0
                let ana_phase_diff = normalize_phase(
                    self.ana_phases[k] - self.prev_ana_phases[k] - orig_phase_mult * (k & overlap_mask) as f32,
                );

                // The following two lines are basically equivalent to
                //   syn_phase_diffs[newk] = pitch_shift * (ana_phase_diff + orig_phase_mult * k).
                // The main difference is that the syn_phase_diffs is much closer to zero, so that normalize_phase
                // would take less time later.
                let syn_phase_diff =
                    ana_phase_diff * pitch_shift + (k as f32 * pitch_shift - new_k as f32) * orig_phase_mult;
                self.syn_phases[new_k] =
                    self.prev_syn_phases[new_k] + syn_phase_diff + orig_phase_mult * (new_k & overlap_mask) as f32;
            } else {
                // Element is from current frame - propagate phase gradient to neighbors
                if k > 0 && !self.phase_assigned[k - 1] {
                    self.phase_assigned[k - 1] = true;
                    num_unassigned -= 1;
                    self.max_heap.push(HeapElem {
                        freq_bin: k - 1,
                        magnitude: self.ana_magnitudes[k - 1],
                        prev_frame: false,
                    });

                    let new_k1 = ((k - 1) as f32 * pitch_shift) as usize;
                    let new_k = (k as f32 * pitch_shift) as usize;
                    if new_k < freq_size && new_k1 != new_k {
                        self.syn_phases[new_k1] = self.syn_phases[new_k] - self.ana_phases[k] + self.ana_phases[k - 1];
                    }
                }

                if k < freq_size - 1 && !self.phase_assigned[k + 1] {
                    self.phase_assigned[k + 1] = true;
                    num_unassigned -= 1;
                    self.max_heap.push(HeapElem {
                        freq_bin: k + 1,
                        magnitude: self.ana_magnitudes[k + 1],
                        prev_frame: false,
                    });

                    let new_k1 = ((k + 1) as f32 * pitch_shift) as usize;
                    let new_k = (k as f32 * pitch_shift) as usize;
                    if new_k1 < freq_size && new_k1 != new_k {
                        self.syn_phases[new_k1] = self.syn_phases[new_k] - self.ana_phases[k] + self.ana_phases[k + 1];
                    }
                }
            }
        }

        // Convert phase/magnitude back to complex frequency domain
        #[allow(clippy::needless_range_loop)]
        for k in 0..freq_size {
            // Do the normalize_phase here so that the prev_syn_phases does not become too large, reducing the floating
            // point error.
            self.prev_syn_phases[k] = normalize_phase(self.syn_phases[k]);
            dst_freq[k] = Complex::from_polar(self.syn_magnitudes[k], self.prev_syn_phases[k]);
        }
        // Save previous analysis data for next frame
        self.prev_ana_magnitudes.copy_from_slice(&self.ana_magnitudes);
        self.prev_ana_phases.copy_from_slice(&self.ana_phases);
    }

    pub fn reset(&mut self) {
        self.prev_ana_magnitudes.fill(0.0);
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

        heap.push(HeapElem { freq_bin: 0, magnitude: 1.5, prev_frame: false });
        heap.push(HeapElem { freq_bin: 1, magnitude: 3.2, prev_frame: false });
        heap.push(HeapElem { freq_bin: 2, magnitude: 0.8, prev_frame: false });
        heap.push(HeapElem { freq_bin: 3, magnitude: 2.7, prev_frame: false });
        heap.push(HeapElem { freq_bin: 2, magnitude: 2.7, prev_frame: true });

        assert_eq!(heap.pop().unwrap().magnitude, 3.2);
        assert_eq!(heap.pop().unwrap().magnitude, 2.7);
        assert_eq!(heap.pop().unwrap().magnitude, 2.7);
        assert_eq!(heap.pop().unwrap().magnitude, 1.5);
        assert_eq!(heap.pop().unwrap().magnitude, 0.8);
        assert!(heap.is_empty());
    }
}
