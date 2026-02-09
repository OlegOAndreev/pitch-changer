//! STFT (Short-Time Fourier Transform) state and operations.

use realfft::num_complex::Complex;
use realfft::{ComplexToReal, RealToComplex};
use std::sync::Arc;

use crate::window::WindowType;

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
pub struct Stft {
    fft_size: usize,
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
        let input_freq_buf = vec![Complex::new(0.0, 0.0); fft_size / 2 + 1];
        let scratch_forward = forward_plan.make_scratch_vec();
        let output_buf = vec![0.0; fft_size];
        let output_freq_buf = vec![Complex::new(0.0, 0.0); fft_size / 2 + 1];
        let scratch_inverse = inverse_plan.make_scratch_vec();

        Self {
            fft_size,
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

    // Apply window, run forward pass, call freq_func on frequencies, run inverse pass, apply window again and return
    // the resulting series. The returned slice points to internal buffer.
    pub fn process<F>(&mut self, input: &[f32], mut processor: F) -> &[f32]
    where
        F: FnMut(&[Complex<f32>], &mut [Complex<f32>]),
    {
        assert_eq!(input.len(), self.fft_size);
        self.input_buf.copy_from_slice(input);
        for (s, w) in self.input_buf.iter_mut().zip(&self.window) {
            *s *= w;
        }
        self.forward_plan
            .process_with_scratch(&mut self.input_buf, &mut self.input_freq_buf, &mut self.scratch_forward)
            .expect("failed forward STFT pass");

        processor(&self.input_freq_buf, &mut self.output_freq_buf);

        self.inverse_plan
            .process_with_scratch(&mut self.output_freq_buf, &mut self.output_buf, &mut self.scratch_inverse)
            .expect("failed inverse STFT pass");
        for (s, w) in self.output_buf.iter_mut().zip(&self.window) {
            *s *= w;
        }

        &self.output_buf
    }
}
