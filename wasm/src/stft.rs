//! STFT (Short-Time Fourier Transform) state and operations.

use realfft::num_complex::Complex;
use realfft::{ComplexToReal, RealToComplex};
use std::sync::Arc;

use crate::window::{WindowType, get_window_squared_sum};

/// STFT state for performing forward and inverse FFT operations.
///
/// One window will be used both for analysis and synthesis (i.e. forward and inverse transforms). This idea is taken
/// from smbPitchShift.cpp and https://www.dsprelated.com/freebooks/sasp/Overlap_Add_OLA_STFT_Processing.html and
/// requires that for overlaps >= 4 the sum of squared windows is constant.
///
/// The explanation why having windows both for forward and inverse transform is more optimal (instead of just forward
/// transform) is given here: https://gauss256.github.io/blog/cola.html:
/// > As mentioned above, it is most common to choose a = 1. The reason is that in the case where we did modify the
/// > STFT, there may not be a time-domain signal whose STFT matches our modified version. Choosing a = 1 gives the
/// > Griffin-Lim [http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.306.7858&rep=rep1&type=pdf] optimal estimate
/// > (optimal in a least-squares sense) for a time-domain signal from a modified STFT. SciPy implements a = 1 for the
/// > signal reconstruction in the istft routine.
///
/// When you do synthesis with different hop size compared to analysis, you break constant overlap-add property, but
/// in general things smooth out nicely if your synthesis hop size is not too large. I do not know why =)
pub(crate) struct Stft {
    fft_size: usize,
    window_type: WindowType,
    forward_plan: Arc<dyn RealToComplex<f32>>,
    inverse_plan: Arc<dyn ComplexToReal<f32>>,
    window: Vec<f32>,
    // Scratch buffers
    input_buf: Vec<f32>,
    input_freq_buf: Vec<Complex<f32>>,
    scratch_forward: Vec<Complex<f32>>,
    output_buf: Vec<f32>,
    output_freq_buf: Vec<Complex<f32>>,
    scratch_inverse: Vec<Complex<f32>>,
}

impl Stft {
    /// Create a new STFT instance with given FFT size and window type.
    pub fn new(fft_size: usize, window_type: WindowType) -> Self {
        use crate::window::generate_window;
        use realfft::RealFftPlanner;

        let mut planner = RealFftPlanner::<f32>::new();
        let forward_plan = planner.plan_fft_forward(fft_size);
        let inverse_plan = planner.plan_fft_inverse(fft_size);

        let window = generate_window(window_type, fft_size);

        let input_buf = vec![0.0; fft_size];
        let input_freq_buf = vec![Complex::ZERO; fft_size / 2 + 1];
        let scratch_forward = forward_plan.make_scratch_vec();
        let output_buf = vec![0.0; fft_size];
        let output_freq_buf = vec![Complex::ZERO; fft_size / 2 + 1];
        let scratch_inverse = inverse_plan.make_scratch_vec();

        Self {
            fft_size,
            window_type,
            forward_plan,
            inverse_plan,
            window,
            input_buf,
            input_freq_buf,
            scratch_forward,
            output_buf,
            output_freq_buf,
            scratch_inverse,
        }
    }

    /// Apply window, run forward pass, call freq_func on frequencies, run inverse pass, apply window again and return
    /// the resulting series. The result is not normalized, you should normalize it using  The returned slice points to
    /// internal buffer.
    pub fn process<F>(&mut self, input: &[f32], mut processor: F) -> &[f32]
    where
        F: FnMut(&[Complex<f32>], &mut [Complex<f32>]),
    {
        assert_eq!(input.len(), self.fft_size);
        for i in 0..self.fft_size {
            self.input_buf[i] = input[i] * self.window[i];
        }
        self.forward_plan
            .process_with_scratch(&mut self.input_buf, &mut self.input_freq_buf, &mut self.scratch_forward)
            .expect("failed forward STFT pass");

        processor(&self.input_freq_buf, &mut self.output_freq_buf);

        self.inverse_plan
            .process_with_scratch(&mut self.output_freq_buf, &mut self.output_buf, &mut self.scratch_inverse)
            .expect("failed inverse STFT pass");
        for i in 0..self.fft_size {
            self.output_buf[i] *= self.window[i];
        }

        &self.output_buf
    }

    /// Return normalization factor for STFT output: inverse FFT scaling (1/fft_size) * squared windows sum
    pub fn get_norm_factor(&self, hop_size: usize) -> f32 {
        let window_norm = get_window_squared_sum(self.window_type, self.fft_size, hop_size);
        1.0 / (self.fft_size as f32 * window_norm)
    }
}

/// Buffer for accumulating results of STFT. Implements overlap-add accumulation: windows of length `fft_size` are added
/// at offset `pos`, and after each addition `hop_size` samples can be output from the beginning of the buffer while
/// shifting the remaining content left.
pub(crate) struct StftAccumBuf {
    buffer: Vec<f32>,
    pos: usize,
}

impl StftAccumBuf {
    /// Create a new accumulation buffer.
    pub(crate) fn new(fft_size: usize) -> Self {
        Self { buffer: vec![0.0; fft_size], pos: 0 }
    }

    /// Add input multiplied by factor to the accumulation buffer.
    pub(crate) fn add(&mut self, input: &[f32], factor: f32) {
        assert_eq!(input.len(), self.buffer.len());
        let remaining = self.buffer.len() - self.pos;
        for i in 0..remaining {
            self.buffer[self.pos + i] += input[i] * factor;
        }
        for i in remaining..input.len() {
            self.buffer[i - remaining] += input[i] * factor;
        }
    }

    /// Multiply the next `n` samples from the current position by window.
    pub(crate) fn multiply_next(&mut self, window: &[f32]) {
        assert!(window.len() <= self.buffer.len());
        let first_part = (self.buffer.len() - self.pos).min(window.len());
        for i in 0..first_part {
            self.buffer[self.pos + i] *= window[i];
        }
        for i in first_part..window.len() {
            self.buffer[i - first_part] *= window[i];
        }
    }

    /// Output `hop_size` samples from the current position in the buffer, shift the buffer by hop_size and zero new
    /// part.
    pub(crate) fn output_next(&mut self, hop_size: usize, output: &mut Vec<f32>) {
        // Output from pos to end of buffer
        let first_part = (self.buffer.len() - self.pos).min(hop_size);
        output.extend_from_slice(&self.buffer[self.pos..self.pos + first_part]);
        self.buffer[self.pos..self.pos + first_part].fill(0.0);
        let remaining = hop_size - first_part;
        if remaining > 0 {
            output.extend_from_slice(&self.buffer[0..remaining]);
            self.buffer[0..remaining].fill(0.0);
        }
        self.pos = (self.pos + hop_size) % self.buffer.len();
    }

    /// Clear the buffer and reset position to zero.
    pub(crate) fn reset(&mut self) {
        self.buffer.fill(0.0);
        self.pos = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::StftAccumBuf;

    #[test]
    fn test_stft_accum_buf_multiple_outputs() {
        let fft_size = 6;
        let mut accum = StftAccumBuf::new(fft_size);

        let window_a = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let window_b = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0];
        let window_c = [3.0, 3.0, 3.0, 3.0, 3.0, 3.0];
        let window_d = [4.0, 4.0, 4.0, 4.0, 4.0, 4.0];

        accum.add(&window_a, 2.0);
        let mut output = Vec::new();
        accum.output_next(2, &mut output);
        assert_eq!(output, &[2.0, 2.0]);

        accum.add(&window_b, 3.0);
        accum.output_next(2, &mut output);
        assert_eq!(output, &[2.0, 2.0, 8.0, 8.0]);

        accum.add(&window_c, 5.0);
        accum.output_next(2, &mut output);
        assert_eq!(output, &[2.0, 2.0, 8.0, 8.0, 23.0, 23.0]);

        accum.add(&window_d, 1.0);
        accum.output_next(3, &mut output);
        assert_eq!(output, &[2.0, 2.0, 8.0, 8.0, 23.0, 23.0, 25.0, 25.0, 19.0]);

        accum.output_next(fft_size, &mut output);
        assert_eq!(output, &[2.0, 2.0, 8.0, 8.0, 23.0, 23.0, 25.0, 25.0, 19.0, 19.0, 4.0, 4.0, 0.0, 0.0, 0.0]);

        accum.output_next(fft_size, &mut output);
        assert_eq!(
            output,
            &[
                2.0, 2.0, 8.0, 8.0, 23.0, 23.0, 25.0, 25.0, 19.0, 19.0, 4.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0
            ]
        );
    }

    #[test]
    fn test_stft_accum_buf_multiply_next() {
        let fft_size = 8;
        let mut accum = StftAccumBuf::new(fft_size);
        for i in 0..fft_size {
            accum.buffer[i] = (i + 1) as f32;
        }
        accum.pos = 0;

        let window = [2.0, 3.0, 4.0, 5.0];
        accum.multiply_next(&window);

        assert_eq!(accum.buffer, &[2.0, 6.0, 12.0, 20.0, 5.0, 6.0, 7.0, 8.0]);
        assert_eq!(accum.pos, 0);

        let mut accum2 = StftAccumBuf::new(fft_size);
        for i in 0..fft_size {
            accum2.buffer[i] = (i + 1) as f32;
        }
        accum2.pos = 6;

        let window2 = [0.5, 2.0, 3.0, 4.0];
        accum2.multiply_next(&window2);
        assert_eq!(accum2.buffer, &[3.0, 8.0, 3.0, 4.0, 5.0, 6.0, 3.5, 16.0]);
        assert_eq!(accum2.pos, 6);
    }
}
