//! STFT (Short-Time Fourier Transform) state and operations.

use realfft::{RealFftPlanner, num_complex::Complex};
use std::sync::Arc;

use crate::util::fill_hann_window;

/// STFT state for performing forward and inverse FFT operations.
///
/// One window will be used both for analysis and synthesis (i.e. forward and inverse transforms). This idea is taken
/// from smbPitchShift.cpp and is based on the fact that for overlaps >= 4 the sum of squared hann window is constant.
/// The explanation why having windows both for forward and inverse transform is more optimal (instead of just forward
/// transform) is given here: https://gauss256.github.io/blog/cola.html:
///
/// > As mentioned above, it is most common to choose a = 1. The reason is that in the case where we did modify
/// > the STFT, there may not be a time-domain signal whose STFT matches our modified version. Choosing a = 1 gives
/// > the Griffin-Lim [http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.306.7858&rep=rep1&type=pdf]
/// > optimal estimate (optimal in a least-squares sense) for a time-domain signal from a modified STFT. SciPy
/// > implements a = 1 for the signal reconstruction in the istft routine.
pub struct Stft {
    fft_size: usize,
    forward_plan: Arc<dyn realfft::RealToComplex<f32>>,
    inverse_plan: Arc<dyn realfft::ComplexToReal<f32>>,
    window: Vec<f32>,
    // Scratch buffers
    src_buf: Vec<f32>,
    src_freq_buf: Vec<Complex<f32>>,
    dst_buf: Vec<f32>,
    dst_freq_buf: Vec<Complex<f32>>,
}

impl Stft {
    pub fn new(fft_size: usize) -> Self {
        let mut planner = RealFftPlanner::<f32>::new();
        let forward_plan = planner.plan_fft_forward(fft_size);
        let inverse_plan = planner.plan_fft_inverse(fft_size);

        let mut window = vec![0.0; fft_size];
        fill_hann_window(&mut window);

        let src_buf = vec![0.0; fft_size];
        let src_freq_buf = vec![Complex::ZERO; fft_size / 2 + 1];
        let dst_buf = vec![0.0; fft_size];
        let dst_freq_buf = vec![Complex::ZERO; fft_size / 2 + 1];

        Self { fft_size, forward_plan, inverse_plan, window, src_buf, src_freq_buf, dst_buf, dst_freq_buf }
    }

    // Run forward pass, call freq_func on frequencies, run inverse pass and return the resulting series. The returned
    // slice points to internal buffer.
    pub fn process<F>(&mut self, input: &[f32], mut freq_func: F) -> &[f32]
    where
        F: FnMut(&[Complex<f32>], &mut [Complex<f32>]),
    {
        assert_eq!(input.len(), self.fft_size);
        self.src_buf.copy_from_slice(input);
        for (s, w) in self.src_buf.iter_mut().zip(&self.window) {
            *s *= w;
        }
        self.forward_plan
            .process(&mut self.src_buf, &mut self.src_freq_buf)
            .expect("failed forward STFT pass");

        freq_func(&self.src_freq_buf, &mut self.dst_freq_buf);

        self.inverse_plan
            .process(&mut self.dst_freq_buf, &mut self.dst_buf)
            .expect("failed inverse STFT pass");
        for (s, w) in self.dst_buf.iter_mut().zip(&self.window) {
            *s *= w;
        }

        &self.dst_buf
    }
}

#[cfg(test)]
mod tests {}
