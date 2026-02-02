use wasm_bindgen::prelude::*;

use anyhow::{Result, bail};
use audioadapter_buffers::direct::InterleavedSlice;
use rubato::{Async, FixedAsync, Resampler, SincInterpolationParameters, SincInterpolationType, WindowFunction};

use crate::WindowType;
use crate::time_stretcher::{TimeStretchParams, TimeStretcher};
use crate::web::WrapAnyhowError;

/// Parameters for audio pitch shifting
#[derive(Debug, Clone, Copy)]
#[wasm_bindgen]
pub struct PitchShiftParams {
    /// Pitch shift factor (e.g., 2.0 = one octave up, 0.5 = one octave down)
    pub pitch_shift: f32,
    /// Sample rate in Hz
    pub rate: u32,
    /// FFT window size in samples
    pub fft_size: usize,
    /// Overlap factor (number of windows overlapping one point, must be power of two)
    pub overlap: u32,
    /// Window type to use for STFT
    pub window_type: WindowType,
}

#[wasm_bindgen]
impl PitchShiftParams {
    #[wasm_bindgen(constructor)]
    pub fn new(rate: u32, pitch_shift: f32) -> Self {
        let stretch_params = TimeStretchParams::new(rate, 1.0);
        Self {
            pitch_shift,
            rate,
            fft_size: stretch_params.fft_size,
            overlap: stretch_params.overlap,
            window_type: stretch_params.window_type,
        }
    }
}

#[wasm_bindgen]
pub struct PitchShifter {
    time_stretcher: TimeStretcher,
    resampler: Async<f32>,
}

// PitchShifter stretches the time by pitch_shift factor and then resamples the rate back so that the output has the
// ~same length as the input.
#[wasm_bindgen]
impl PitchShifter {
    #[wasm_bindgen(constructor)]
    pub fn new(params: &PitchShiftParams) -> std::result::Result<Self, WrapAnyhowError> {
        Self::validate_params(params).map_err(WrapAnyhowError)?;

        let mut time_stretch_params = TimeStretchParams::new(params.rate, params.pitch_shift);
        time_stretch_params.fft_size = params.fft_size;
        time_stretch_params.overlap = params.overlap;
        time_stretch_params.window_type = params.window_type;

        let time_stretcher = TimeStretcher::new(&time_stretch_params)?;

        // All the parameters are copied from documentation.
        let sinc_params = SincInterpolationParameters {
            sinc_len: 128,
            f_cutoff: 0.95,
            oversampling_factor: 128,
            interpolation: SincInterpolationType::Quadratic,
            window: Self::window_type_to_rubato(params.window_type),
        };
        let resampler = Async::new_sinc(
            1.0 / (params.pitch_shift as f64),
            1.0,
            &sinc_params,
            params.fft_size, // use FFT size as chunk size
            1,
            FixedAsync::Input,
        )
        .map_err(|e| WrapAnyhowError(anyhow::anyhow!("Failed to create resampler: {:?}", e)))?;

        Ok(Self { time_stretcher, resampler })
    }

    fn validate_params(params: &PitchShiftParams) -> Result<()> {
        if params.pitch_shift < 0.25 || params.pitch_shift > 4.0 {
            bail!("Pitch shifting factor cannot be lower than 0.25 or higher than 4");
        }
        // Most of the parameter validation will be done by TimeStretcher.
        Ok(())
    }

    fn window_type_to_rubato(window_type: WindowType) -> WindowFunction {
        match window_type {
            WindowType::Hann => WindowFunction::Hann2,
            WindowType::SqrtHann => WindowFunction::Hann,
            WindowType::SqrtBlackman => WindowFunction::Blackman,
        }
    }

    /// Process a chunk of audio samples through the pitch shifter.
    ///
    /// Note: you need to call `finish()` to receive the last output chunks after you call `process()` for all input
    /// samples.
    #[wasm_bindgen]
    pub fn process(&mut self, src: &[f32]) -> Vec<f32> {
        let time_stretched = self.time_stretcher.process(src);
        self.resample(&time_stretched)
    }

    /// Finish processing any remaining audio data in the internal buffers. This method must be called after all input
    /// has been processed via `process()`.
    ///
    /// Note: after calling `finish()`, the pitch shifter is reset and ready to process new audio data.
    #[wasm_bindgen]
    pub fn finish(&mut self) -> Vec<f32> {
        let time_stretched = self.time_stretcher.finish();
        let result = self.resample(&time_stretched);
        self.reset();
        result
    }

    /// Reset the pitch shifter to its initial state.
    #[wasm_bindgen]
    pub fn reset(&mut self) {
        self.time_stretcher.reset();
        self.resampler.reset();
    }

    fn resample(&mut self, input: &[f32]) -> Vec<f32> {
        if input.is_empty() {
            return Vec::new();
        }

        let input_slice = InterleavedSlice::new(input, 1, input.len()).expect("InterleavedSlice::new error");
        let output_len = self.resampler.process_all_needed_output_len(input.len());
        let mut output_vec = vec![0.0f32; output_len];
        let mut output_slice =
            InterleavedSlice::new_mut(&mut output_vec, 1, output_len).expect("InterleavedSlice::new_mut error");

        // Resample the entire input
        let (input_consumed, output_writtern) = self
            .resampler
            .process_all_into_buffer(&input_slice, &mut output_slice, input.len(), None)
            .expect("process_all_into_buffer error");
        assert_eq!(input_consumed, input.len());

        output_vec.truncate(output_writtern);
        output_vec
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generate_sine_wave;
    use crate::util::{compute_dominant_frequency, compute_magnitude};
    use anyhow::Result;

    fn process_all(shifter: &mut PitchShifter, input: &[f32]) -> Vec<f32> {
        let mut result = shifter.process(input);
        result.append(&mut shifter.finish());
        result
    }

    #[test]
    fn test_randomized_pitch_shifter_no_crash() {
        use rand::Rng;

        let mut rng = rand::rng();
        const ITERATIONS: usize = 100;

        for _ in 0..ITERATIONS {
            let sample_rate = rng.random_range(10000..=100000);
            // overlap must be power of two, 4..=32
            let overlap = 1 << rng.random_range(2..=5); // 4, 8, 16, 32
            // pitch shift must be within 0.25..=4.0 and also <= overlap/3
            let max_pitch = (overlap as f32 / 3.0).min(4.0);
            let pitch_shift = rng.random_range(0.25..=max_pitch);
            // fft_size must be power of two, >= overlap, and divisible by overlap
            // generate exponent 9..=12 (512..=4096) which is >= overlap (max 32)
            let fft_size = 1 << rng.random_range(9..=12); // 512, 1024, 2048, 4096

            let mut params = PitchShiftParams::new(sample_rate, pitch_shift);
            params.fft_size = fft_size;
            params.overlap = overlap;

            let len = rng.random_range(0..=4 * fft_size);
            let audio_data: Vec<f32> = (0..len).map(|_| rng.random_range(-1.0..1.0)).collect();

            let mut shifter = PitchShifter::new(&params).unwrap();
            let _output = process_all(&mut shifter, &audio_data);
        }
    }

    #[test]
    fn test_pitch_shift_single_sine_wave() -> Result<()> {
        const DURATION: f32 = 0.5;
        const MAGNITUDE: f32 = 3.2;
        const INPUT_FREQ: f32 = 400.0;

        for sample_rate in [44100.0, 96000.0] {
            for fft_size in [1024, 4096] {
                for overlap in [8, 16] {
                    for window_type in [WindowType::Hann, WindowType::SqrtBlackman, WindowType::SqrtHann] {
                        for pitch_shift in [0.5, 0.75, 1.25, 1.5, 2.0] {
                            let input = generate_sine_wave(INPUT_FREQ, sample_rate, MAGNITUDE, DURATION);

                            let mut params = PitchShiftParams::new(sample_rate as u32, pitch_shift);
                            params.fft_size = fft_size;
                            params.overlap = overlap;
                            params.window_type = window_type;
                            let mut shifter = PitchShifter::new(&params).unwrap();
                            let output = process_all(&mut shifter, &input);

                            let expected_freq = INPUT_FREQ * pitch_shift;
                            let output_freq = compute_dominant_frequency(&output, sample_rate);
                            let output_magn = compute_magnitude(&output);

                            let bin_width = sample_rate as f32 / params.fft_size as f32;
                            let tolerance = bin_width * 2.0;
                            println!(
                                "Sample rate {}, fft size {}, overlap {}, window {:?}, pitch shift {}, input {} Hz, output {} Hz, expected {} Hz, magnitude {}",
                                sample_rate,
                                fft_size,
                                overlap,
                                window_type,
                                pitch_shift,
                                INPUT_FREQ,
                                output_freq,
                                expected_freq,
                                output_magn
                            );
                            assert!(
                                (output_freq - expected_freq).abs() < tolerance,
                                "expected {} Hz, got {} Hz for sample rate {}, fft size {}, overlap {}, window {:?}, pitch shift {}, input {} Hz",
                                expected_freq,
                                output_freq,
                                sample_rate,
                                fft_size,
                                overlap,
                                window_type,
                                pitch_shift,
                                INPUT_FREQ,
                            );
                            assert!(
                                (output_magn - MAGNITUDE).abs() < MAGNITUDE * 0.1,
                                "expected magnitude {}, got {} for sample rate {}, fft size {}, overlap {}, window {:?}, pitch shift {}, input {} Hz",
                                MAGNITUDE,
                                output_magn,
                                sample_rate,
                                fft_size,
                                overlap,
                                window_type,
                                pitch_shift,
                                INPUT_FREQ,
                            );
                        }
                    }
                }
            }
        }

        Ok(())
    }
}
