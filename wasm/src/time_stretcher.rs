use wasm_bindgen::prelude::*;

use anyhow::{Result, bail};

use crate::WindowType;
use crate::phase_gradient_time_stretch::PhaseGradientTimeStretch;
use crate::stft::Stft;
use crate::web::WrapAnyhowError;

/// Parameters for audio time stretching/.
#[derive(Debug, Clone, Copy)]
#[wasm_bindgen]
pub struct TimeStretchParams {
    /// Time stretch factor (e.g., 2.0 = twice as long, 0.5 = half length)
    pub time_stretch: f32,
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// FFT window size in samples
    pub fft_size: usize,
    /// Overlap factor (number of windows overlapping one point, must be power of two)
    pub overlap: u32,
    /// Window type to use for STFT
    pub window_type: WindowType,
}

#[wasm_bindgen]
impl TimeStretchParams {
    #[wasm_bindgen(constructor)]
    pub fn new(sample_rate: u32, time_stretch: f32) -> Self {
        Self { time_stretch, sample_rate, fft_size: 4096, overlap: 8, window_type: WindowType::Hann }
    }

    #[wasm_bindgen]
    pub fn to_debug_string(&self) -> String {
        format!("{:?}", self)
    }
}

#[wasm_bindgen]
pub struct TimeStretcher {
    params: TimeStretchParams,
    ana_hop_size: usize,
    syn_hop_size: usize,
    stft: Stft,
    phase_gradient_vocoder: PhaseGradientTimeStretch,
    // See finish()
    tail_window: Vec<f32>,

    // Buffer management: source data is accumulated in input_buf until there is enough data to run STFT. After each
    // STFT results are accumulated into output_accum_buf. After that the input_buf is shifted by ana_hop_size and
    // output_accum_buf is shifted by syn_hop_size.
    //
    // Note: input_buf is not padded with zeros, so the initial part (until fft_size samples are processed) will be
    // smoothed by windows of STFT. There is no particular good way to work around this given that syn_hop_size !=
    // ana_hop_size. See also finish() for reverse problem: smoothing of the final part.
    input_buf: Vec<f32>,
    output_accum_buf: Vec<f32>,
}

#[wasm_bindgen]
impl TimeStretcher {
    #[wasm_bindgen(constructor)]
    pub fn new(params: &TimeStretchParams) -> std::result::Result<Self, WrapAnyhowError> {
        use crate::window::generate_tail_window;

        Self::validate_params(params).map_err(WrapAnyhowError)?;

        let ana_hop_size = params.fft_size / params.overlap as usize;
        let syn_hop_size = (ana_hop_size as f32 * params.time_stretch) as usize;
        let stft = Stft::new(params.fft_size, params.window_type);
        let phase_gradient_vocoder = PhaseGradientTimeStretch::new(params.fft_size);
        let tail_len = params.fft_size / ana_hop_size * syn_hop_size;
        let tail_window = generate_tail_window(params.window_type, tail_len);
        let input_buf = Vec::with_capacity(params.fft_size);
        let output_accum_buf = vec![0.0f32; params.fft_size];

        Ok(Self {
            params: *params,
            ana_hop_size,
            syn_hop_size,
            stft,
            phase_gradient_vocoder,
            tail_window,
            input_buf,
            output_accum_buf,
        })
    }

    fn validate_params(params: &TimeStretchParams) -> Result<()> {
        if params.time_stretch < 0.25 || params.time_stretch > 4.0 {
            bail!("Time stretching factor cannot be lower than 0.25 or higher than 4");
        }
        if params.time_stretch > params.overlap as f32 / 3.0 {
            bail!("Time stretching factor cannot be bigger than overlap/3");
        }
        if params.fft_size.next_power_of_two() != params.fft_size {
            bail!("FFT size must be power of two");
        }
        if params.fft_size < 512 || params.fft_size > 32768 {
            bail!("FFT size must be in [512, 32768] range");
        }
        if params.overlap < 4 {
            bail!("Overlap must be at least 4");
        }
        if params.overlap.next_power_of_two() != params.overlap {
            bail!("Overlap must be power of two");
        }
        if !params.fft_size.is_multiple_of(params.overlap as usize) {
            bail!("FFT size must be divisible by overlap");
        }
        Ok(())
    }

    /// Process a chunk of audio samples through the time stretcher.
    ///
    /// Note: you need to call `finish()` to receive the last output chunks after you call `process()` for all input
    /// samples.
    #[wasm_bindgen]
    pub fn process(&mut self, input: &[f32]) -> Vec<f32> {
        let output_capacity = input.len() / self.ana_hop_size * self.syn_hop_size;
        let mut output = Vec::with_capacity(output_capacity);
        let mut input_pos = 0;
        while input_pos < input.len() {
            let needed = self.params.fft_size - self.input_buf.len();
            let available = input.len() - input_pos;
            let n = available.min(needed);
            self.input_buf.extend_from_slice(&input[input_pos..input_pos + n]);
            input_pos += n;

            if self.input_buf.len() == self.params.fft_size {
                self.do_stft();
                self.append_output_and_shift(&mut output);
            }
        }
        output
    }

    /// Finish processing any remaining audio data in the internal buffers. This method must be called after all input
    /// has been processed via `process()`.
    ///
    /// Note: after calling `finish()`, the pitch shifter is reset and ready to process new audio data.
    #[wasm_bindgen]
    pub fn finish(&mut self) -> Vec<f32> {
        // We want to process all data remaining in input_buf, which is done by running stft fft_size/ana_hop_size times
        // and padding with zeros after each iteration. We want to fade out this tail to zero by applying half-window
        // stored in tail_window.
        //
        // This is especially important for large stretches where syn_hop_size > ana_hop_size leads to rapid amplitude
        // changes and noise.
        //
        // Another way to solve this problem is simply appending the whole output_accum_buf into result on final
        // iteration, but this makes testing more annoying =)
        let result_capacity = (self.params.fft_size / self.ana_hop_size) * self.syn_hop_size;
        let mut result = Vec::with_capacity(result_capacity);

        let iters = self.params.fft_size / self.ana_hop_size;
        for i in 0..iters {
            self.input_buf.resize(self.params.fft_size, 0.0);
            self.do_stft();

            let tail_window_offset = i * self.syn_hop_size;
            // After the last iteration do windowing first.
            for (a, w) in self.output_accum_buf[..self.syn_hop_size]
                .iter_mut()
                .zip(&self.tail_window[tail_window_offset..])
            {
                *a *= *w;
            }
            self.append_output_and_shift(&mut result);
        }
        debug_assert_eq!(result.len(), result_capacity);

        self.reset();
        result
    }

    /// Do one iteration of stft
    fn do_stft(&mut self) {
        use crate::window::get_window_squared_sum;

        let output = self.stft.process(&self.input_buf, |ana_freq, syn_freq| {
            // syn_freq.copy_from_slice(ana_freq);
            self.phase_gradient_vocoder
                .process(ana_freq, self.ana_hop_size, syn_freq, self.syn_hop_size);
            // Ensure conjugate symmetry for real-valued inverse FFT. The first bin and last bin should have zero
            // imaginary part. After processing, they may become non-zero (even if very small).
            syn_freq[0].im = 0.0;
            syn_freq[syn_freq.len() - 1].im = 0.0;
        });
        // Normalization factor: inverse FFT scaling (1/fft_size) * squared windows sum * time stretch factor
        // The last factor is based on the fact that synthesis windows have different overlap contained
        let window_norm = get_window_squared_sum(self.params.window_type, self.params.fft_size, self.ana_hop_size);
        let norm = self.params.time_stretch / (self.params.fft_size as f32 * window_norm);
        // let norm = 0.5 / (self.params.fft_size as f32 * window_norm);
        for (a, o) in self.output_accum_buf.iter_mut().zip(output) {
            *a += *o * norm;
        }
    }

    /// Append next part of output accum buf to result and shift the buffers.
    fn append_output_and_shift(&mut self, output: &mut Vec<f32>) {
        output.extend_from_slice(&self.output_accum_buf[..self.syn_hop_size]);
        self.output_accum_buf.drain(0..self.syn_hop_size);
        self.output_accum_buf.resize(self.params.fft_size, 0.0);
        self.input_buf.drain(0..self.ana_hop_size);
    }

    /// Reset the pitch shifter to its initial state.
    #[wasm_bindgen]
    pub fn reset(&mut self) {
        self.phase_gradient_vocoder.reset();
        self.input_buf.clear();
        self.output_accum_buf.fill(0.0);
    }
}

#[cfg(test)]
mod tests {
    use crate::generate_sine_wave;
    use crate::util::{compute_dominant_frequency, compute_magnitude};

    use super::*;
    use rand::Rng;

    fn process_all(stretcher: &mut TimeStretcher, input: &[f32]) -> Vec<f32> {
        let mut result = stretcher.process(input);
        result.append(&mut stretcher.finish());
        result
    }

    fn process_all_small_chunks(stretcher: &mut TimeStretcher, input: &[f32]) -> Vec<f32> {
        const CHUNK_SIZE: usize = 100;

        let mut result = vec![];
        for chunk in input.chunks(CHUNK_SIZE) {
            result.append(&mut stretcher.process(chunk));
        }
        result.append(&mut stretcher.finish());
        result
    }

    #[test]
    fn test_randomized_pitch_stretcher_no_crash() {
        use rand;

        let mut rng = rand::rng();
        const ITERATIONS: usize = 100;

        for _ in 0..ITERATIONS {
            let sample_rate = rng.random_range(10000..=100000);
            let time_stretch = rng.random_range(0.25..1.2);
            let fft_size = 1 << rng.random_range(9..=12); // 2^9=512, 2^12=4096
            let mut params = TimeStretchParams::new(sample_rate, time_stretch);
            params.fft_size = fft_size;
            params.overlap = 1 << rng.random_range(2..=5); // 2^2=4, 2^5=32

            let len = rng.random_range(0..=4 * fft_size);
            let audio_data: Vec<f32> = (0..len).map(|_| rng.random_range(-1.0..1.0)).collect();

            let mut stretcher = TimeStretcher::new(&params).unwrap();
            let _output = process_all(&mut stretcher, &audio_data);
        }
    }

    #[test]
    fn test_time_stretch_single_sine_wave() -> Result<()> {
        const DURATION: f32 = 0.5;
        const MAGNITUDE: f32 = 3.2;

        for sample_rate in [44100.0, 96000.0] {
            for input_freq in [200.0, 500.0, 800.0, 5000.0] {
                for fft_size in [1024, 4096] {
                    for overlap in [8, 16] {
                        for window_type in [WindowType::Hann, WindowType::SqrtBlackman, WindowType::SqrtHann] {
                            for time_stretch in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0] {
                                let input = generate_sine_wave(input_freq, sample_rate, MAGNITUDE, DURATION);

                                let mut params = TimeStretchParams::new(sample_rate as u32, time_stretch);
                                params.fft_size = fft_size;
                                params.overlap = overlap;
                                let mut stretcher = TimeStretcher::new(&params).unwrap();
                                let output = process_all(&mut stretcher, &input);

                                let expected_freq = input_freq;
                                let output_freq = compute_dominant_frequency(&output, sample_rate);
                                let output_magn = compute_magnitude(&output);

                                let bin_width = sample_rate as f32 / params.fft_size as f32;
                                let tolerance = bin_width * 2.0;
                                println!(
                                    "Sample rate {}, fft size {}, overlap {}, window {:?}, stretch {}, input {} Hz, output {} Hz, expected {} Hz, magnitude {}",
                                    sample_rate,
                                    fft_size,
                                    overlap,
                                    window_type,
                                    time_stretch,
                                    input_freq,
                                    output_freq,
                                    expected_freq,
                                    output_magn
                                );
                                assert!(
                                    (output_freq - expected_freq).abs() < tolerance,
                                    "expected {} Hz, got {} Hz for sample rate {}, fft size {}, overlap {}, window {:?}, stretch {}, input {} Hz",
                                    expected_freq,
                                    output_freq,
                                    sample_rate,
                                    fft_size,
                                    overlap,
                                    window_type,
                                    time_stretch,
                                    input_freq,
                                );
                                assert!(
                                    (output_magn - MAGNITUDE).abs() < MAGNITUDE * 0.1,
                                    "expected magnitude {}, got {} for sample rate {}, fft size {}, overlap {}, window {:?}, stretch {}, input {} Hz",
                                    MAGNITUDE,
                                    output_magn,
                                    sample_rate,
                                    fft_size,
                                    overlap,
                                    window_type,
                                    time_stretch,
                                    input_freq,
                                );
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    #[test]
    fn test_time_stretch_identity() -> Result<()> {
        const FREQ: f32 = 440.0;
        const MAGNITUDE: f32 = 2.2;
        const DURATION: f32 = 1.0;

        for sample_rate in [44100.0, 96000.0] {
            for fft_size in [1024, 4096] {
                for overlap in [8, 16] {
                    for window_type in [WindowType::Hann, WindowType::SqrtBlackman, WindowType::SqrtHann] {
                        let input = generate_sine_wave(FREQ, sample_rate, MAGNITUDE, DURATION);

                        let mut params = TimeStretchParams::new(sample_rate as u32, 1.0);
                        params.fft_size = fft_size;
                        params.overlap = overlap;
                        params.window_type = window_type;
                        let mut stretcher = TimeStretcher::new(&params).unwrap();
                        let output = process_all_small_chunks(&mut stretcher, &input);

                        // Skip transient at start and end, compare the middle
                        let offset = fft_size * 2;
                        let middle_len = (input.len() - offset * 2).min(output.len() - offset * 2);
                        let input_slice = &input[offset..offset + middle_len];
                        let output_slice = &output[offset..offset + middle_len];

                        let mut max_diff = 0.0f32;
                        for (i, o) in input_slice.iter().zip(output_slice) {
                            let diff = (i - o).abs();
                            if diff > max_diff {
                                max_diff = diff;
                            }
                        }

                        assert!(
                            max_diff < 1e-3,
                            "Max difference {} for sample_rate {}, fft_size {}, overlap {}, window {:?}",
                            max_diff,
                            sample_rate,
                            fft_size,
                            overlap,
                            window_type
                        );
                    }
                }
            }
        }

        Ok(())
    }
}
