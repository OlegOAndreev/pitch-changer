
use std::sync::Arc;

use realfft::num_complex::Complex;
use wasm_bindgen::prelude::*;

use crate::web::Float32Vec;

/// SpectralHistogram computes the spectral histogram: the magnitudes of FFT bins.
#[allow(dead_code)]
#[wasm_bindgen]
struct SpectralHistogram {
    fft_size: usize,
    forward_plan: Arc<dyn realfft::RealToComplex<f32>>,
    // Scratch buffer
    input_buf: Vec<f32>,
    scratch_buf: Vec<Complex<f32>>,
    output_buf: Vec<Complex<f32>>,
}

#[allow(dead_code)]
#[wasm_bindgen]
impl SpectralHistogram {
    /// Create a new SpectralHistogram for given size.
    #[wasm_bindgen(constructor)]
    pub fn new(fft_size: usize) -> Self {
        use realfft::RealFftPlanner;

        let mut planner = RealFftPlanner::<f32>::new();
        let forward_plan = planner.plan_fft_forward(fft_size);
        let input_buf = vec![0.0; fft_size];
        let scratch_buf = forward_plan.make_scratch_vec();
        let output_buf = vec![Complex::new(0.0, 0.0); fft_size / 2 + 1];
        Self { fft_size, forward_plan, input_buf, scratch_buf, output_buf }
    }

    /// Compute spectral histogram of multi-channel interleaved input. Averages the channel values before computing the
    /// spectrogram.
    #[wasm_bindgen]
    pub fn compute(&mut self, input: &Float32Vec, num_channels: usize, output: &mut Float32Vec) {
        output.clear();
        self.input_buf.fill(0.0);
        let len = self.fft_size.min(input.len() / num_channels);
        for i in 0..len {
            for ch in 0..num_channels {
                self.input_buf[i] += input.0[i * num_channels + ch];
            }
            self.input_buf[i] /= num_channels as f32;
        }
        self.forward_plan
            .process_with_scratch(&mut self.input_buf, &mut self.output_buf, &mut self.scratch_buf)
            .expect("failed forward STFT pass");
        output.resize(self.output_buf.len());
        for (o, freq) in output.0.iter_mut().zip(&self.output_buf) {
            *o = freq.norm();
        }
    }
}
