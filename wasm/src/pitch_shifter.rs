use wasm_bindgen::prelude::*;

use anyhow::{Result, bail};

use crate::WindowType;
use crate::resampler::StreamingResampler;
use crate::time_stretcher::{TimeStretchParams, TimeStretcher};
use crate::util::{deinterleave_samples, interleave_samples};
use crate::web::{Float32Vec, WrapAnyhowError};

/// Parameters for audio pitch shifting.
#[derive(Debug, Clone, Copy)]
#[wasm_bindgen]
pub struct PitchShiftParams {
    /// Pitch shift factor (e.g., 2.0 = one octave up, 0.5 = one octave down)
    pub pitch_shift: f32,
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
impl PitchShiftParams {
    #[wasm_bindgen(constructor)]
    /// Create new pitch shift parameters with default FFT size, overlap, and window.
    pub fn new(sample_rate: u32, pitch_shift: f32) -> Self {
        let stretch_params = TimeStretchParams::new(sample_rate, 1.0);
        Self {
            pitch_shift,
            sample_rate,
            fft_size: stretch_params.fft_size,
            overlap: stretch_params.overlap,
            window_type: stretch_params.window_type,
        }
    }

    #[wasm_bindgen]
    /// Return a debug representation of the parameters for JS.
    pub fn to_debug_string(&self) -> String {
        format!("{:?}", self)
    }
}

/// PitchShifter stretches the mono audio time by pitch_shift factor and then resamples the rate back so that the output
/// has the ~same length as the input.
#[wasm_bindgen]
pub struct PitchShifter {
    time_stretcher: TimeStretcher,
    resampler: StreamingResampler,
    // Scratch buffer
    stretched: Vec<f32>,
}

#[wasm_bindgen]
impl PitchShifter {
    #[wasm_bindgen(constructor)]
    /// Create a new pitch shifter instance with given parameters.
    pub fn new(params: &PitchShiftParams) -> std::result::Result<Self, WrapAnyhowError> {
        Self::validate_params(params).map_err(WrapAnyhowError)?;

        let mut time_stretch_params = TimeStretchParams::new(params.sample_rate, params.pitch_shift);
        time_stretch_params.fft_size = params.fft_size;
        time_stretch_params.overlap = params.overlap;
        time_stretch_params.window_type = params.window_type;

        let time_stretcher = TimeStretcher::new(&time_stretch_params)?;

        let resampler = StreamingResampler::new(1.0 / params.pitch_shift)?;

        Ok(Self { time_stretcher, resampler, stretched: vec![] })
    }

    /// Process a chunk of audio samples through the pitch shifter. Output is NOT cleared.
    ///
    /// Note: you need to call `finish()` to receive the last output chunks after you call `process()` for all input
    /// samples.
    #[wasm_bindgen]
    pub fn process(&mut self, input: &Float32Vec, output: &mut Float32Vec) {
        self.process_vec(&input.0, &mut output.0);
    }

    /// Finish processing any remaining audio data in the internal buffers. This method must be called after all input
    /// has been processed via `process()`. Output is NOT cleared.
    ///
    /// Note: after calling `finish()`, the pitch shifter is reset and ready to process new audio data.
    #[wasm_bindgen]
    pub fn finish(&mut self, output: &mut Float32Vec) {
        self.finish_vec(&mut output.0);
    }

    /// Reset the pitch shifter to its initial state.
    #[wasm_bindgen]
    pub fn reset(&mut self) {
        self.time_stretcher.reset();
        self.resampler.reset();
        self.stretched.clear();
    }

    /// Copy the current magnitude spectrum from the last processed STFT frame.
    #[wasm_bindgen]
    pub fn get_magnitudes(&self, output: &mut Float32Vec) {
        self.time_stretcher.get_magnitudes(output);
    }

    fn validate_params(params: &PitchShiftParams) -> Result<()> {
        if params.pitch_shift < 0.25 || params.pitch_shift > 4.0 {
            bail!("Pitch shifting factor cannot be lower than 0.25 or higher than 4");
        }
        // Most of the parameter validation will be done by TimeStretcher.
        Ok(())
    }
}

// Why do we need a separate mod:
// https://stackoverflow.com/questions/51388721/is-it-possible-to-have-wasm-bindgen-ignore-certain-public-functions-in-an-impl
impl PitchShifter {
    /// See documentation for process()
    pub fn process_vec(&mut self, input: &[f32], output: &mut Vec<f32>) {
        self.stretched.clear();
        self.time_stretcher.process_vec(input, &mut self.stretched);
        self.resampler.resample(&self.stretched, output);
    }

    /// See documentation for finish()
    pub fn finish_vec(&mut self, output: &mut Vec<f32>) {
        self.stretched.clear();
        self.time_stretcher.finish_vec(&mut self.stretched);
        self.resampler.resample(&self.stretched, output);
        self.resampler.finish(output);
        self.reset();
    }

    /// Copy the current magnitude spectrum from the last processed STFT frame.
    pub fn get_magnitudes_vec(&self, output: &mut Vec<f32>) {
        self.time_stretcher.get_magnitudes_vec(output);
    }
}

/// Multi-channel pitch shifter that processes interleaved audio data.
#[wasm_bindgen]
pub struct MultiPitchShifter {
    shifters: Vec<PitchShifter>,
    num_channels: usize,
    // Scratch buffers
    deinterleaved_buf: Vec<f32>,
    output_buf: Vec<f32>,
}

#[wasm_bindgen]
impl MultiPitchShifter {
    #[wasm_bindgen(constructor)]
    /// Create a new multi-channel pitch shifter for given number of channels.
    pub fn new(params: &PitchShiftParams, num_channels: usize) -> std::result::Result<Self, WrapAnyhowError> {
        if num_channels == 0 {
            return Err(WrapAnyhowError(anyhow::anyhow!("Number of channels must be at least 1")));
        }

        let mut shifters = vec![];
        for _ in 0..num_channels {
            shifters.push(PitchShifter::new(params)?);
        }

        Ok(Self { shifters, num_channels, deinterleaved_buf: vec![], output_buf: vec![] })
    }

    /// Process a chunk of interleaved audio samples through the pitch shifter. Output is NOT cleared.
    ///
    /// Note: you need to call `finish()` to receive the last output chunks after you call `process()` for all input
    /// samples.
    #[wasm_bindgen]
    pub fn process(&mut self, input: &Float32Vec, output: &mut Float32Vec) {
        self.process_vec(&input.0, &mut output.0);
    }

    /// Finish processing any remaining audio data in the internal buffers. This method must be called after all input
    /// has been processed via `process()`. Output is NOT cleared.
    ///
    /// Note: after calling `finish()`, the pitch shifter is reset and ready to process new audio data.
    #[wasm_bindgen]
    pub fn finish(&mut self, output: &mut Float32Vec) {
        self.finish_vec(&mut output.0);
    }

    /// Reset all internal pitch shifters.
    #[wasm_bindgen]
    pub fn reset(&mut self) {
        for shifter in &mut self.shifters {
            shifter.reset();
        }
    }
}

impl MultiPitchShifter {
    /// Process interleaved multi-channel audio data.
    pub fn process_vec(&mut self, input: &[f32], output: &mut Vec<f32>) {
        if self.num_channels == 1 {
            self.shifters[0].process_vec(input, output);
            return;
        }

        assert!(input.len().is_multiple_of(self.num_channels));

        self.deinterleaved_buf.clear();
        self.output_buf.clear();

        let samples_per_channel = input.len() / self.num_channels;
        deinterleave_samples(input, self.num_channels, &mut self.deinterleaved_buf);

        for (ch, shifter) in self.shifters.iter_mut().enumerate() {
            let channel_start = ch * samples_per_channel;
            let channel_end = (ch + 1) * samples_per_channel;
            shifter.process_vec(&self.deinterleaved_buf[channel_start..channel_end], &mut self.output_buf);
        }

        interleave_samples(&self.output_buf, self.num_channels, output);
    }

    /// Finish processing for all channels.
    pub fn finish_vec(&mut self, output: &mut Vec<f32>) {
        if self.num_channels == 1 {
            self.shifters[0].finish_vec(output);
            return;
        }

        self.output_buf.clear();
        for shifter in &mut self.shifters {
            shifter.finish_vec(&mut self.output_buf);
        }

        interleave_samples(&self.output_buf, self.num_channels, output);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generate_sine_wave;
    use crate::util::{compute_dominant_frequency, compute_magnitude};
    use anyhow::Result;

    fn process_all(shifter: &mut PitchShifter, input: &[f32]) -> Vec<f32> {
        const PREFIX_SIZE: usize = 1000;
        const PREFIX: f32 = -1234.0;

        let input = Float32Vec(input.to_vec());
        let mut output = Float32Vec(vec![PREFIX; PREFIX_SIZE]);
        shifter.process(&input, &mut output);
        shifter.finish(&mut output);

        let mut result = output.0;
        for i in 0..PREFIX_SIZE {
            assert_eq!(result[i], PREFIX);
        }
        result.drain(..PREFIX_SIZE);
        result
    }

    fn process_all_small_chunks(shifter: &mut PitchShifter, input: &[f32]) -> Vec<f32> {
        const PREFIX_SIZE: usize = 1000;
        const PREFIX: f32 = -1234.0;
        const CHUNK_SIZE: usize = 100;

        let mut output = Float32Vec(vec![PREFIX; PREFIX_SIZE]);
        for chunk in input.chunks(CHUNK_SIZE) {
            shifter.process(&Float32Vec(chunk.to_vec()), &mut output);
        }
        shifter.finish(&mut output);

        let mut result = output.0;
        for i in 0..PREFIX_SIZE {
            assert_eq!(result[i], PREFIX);
        }
        result.drain(..PREFIX_SIZE);
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
        const MAGNITUDE: f32 = 0.37;
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
                                "Sample rate {}, fft size {}, overlap {}, window {:?}, pitch shift {}, input {} Hz, output {} Hz, expected {} Hz, magnitude {}, input length {}, output length {}",
                                sample_rate,
                                fft_size,
                                overlap,
                                window_type,
                                pitch_shift,
                                INPUT_FREQ,
                                output_freq,
                                expected_freq,
                                output_magn,
                                input.len(),
                                output.len()
                            );
                            // Check that frequency, magnitude and output length are roughtly what is expected.
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
                            assert!(
                                output.len().abs_diff(input.len()) < input.len() / 20,
                                "expected output length {}, got {} for sample rate {}, fft size {}, overlap {}, window {:?}, pitch shift {}, input {} Hz",
                                input.len(),
                                output.len(),
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

    #[test]
    fn test_pitch_shift_identity() -> Result<()> {
        const FREQ: f32 = 440.0;
        const MAGNITUDE: f32 = 1.0;
        const DURATION: f32 = 0.5;

        for sample_rate in [44100.0, 96000.0] {
            for fft_size in [1024, 4096] {
                for overlap in [8, 16] {
                    for window_type in [WindowType::Hann, WindowType::SqrtBlackman, WindowType::SqrtHann] {
                        let input = generate_sine_wave(FREQ, sample_rate, MAGNITUDE, DURATION);

                        let mut params = PitchShiftParams::new(sample_rate as u32, 1.0);
                        params.fft_size = fft_size;
                        params.overlap = overlap;
                        params.window_type = window_type;
                        let mut shifter = PitchShifter::new(&params).unwrap();
                        let output = process_all_small_chunks(&mut shifter, &input);

                        // Skip transient at start and end, compare the middle
                        let offset = fft_size * 2;
                        let middle_len = (input.len() - offset * 2).min(output.len() - offset * 2);
                        let input_slice = &input[offset..offset + middle_len];
                        // Account of resampling latency.
                        let output_off = StreamingResampler::LATENCY;
                        let output_slice = &output[offset + output_off..offset + output_off + middle_len];

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

    /// Helper function to process all input through MultiPitchShifter
    fn process_all_multi(shifter: &mut MultiPitchShifter, input: &[f32]) -> Vec<f32> {
        const PREFIX_SIZE: usize = 1000;
        const PREFIX: f32 = -1234.0;

        let input = Float32Vec(input.to_vec());
        let mut output = Float32Vec(vec![PREFIX; PREFIX_SIZE]);
        shifter.process(&input, &mut output);
        shifter.finish(&mut output);

        let mut result = output.0;
        for i in 0..PREFIX_SIZE {
            assert_eq!(result[i], PREFIX);
        }
        result.drain(..PREFIX_SIZE);
        result
    }

    #[test]
    fn test_multi_pitch_shift_single_sine_wave() -> Result<()> {
        const DURATION: f32 = 0.5;
        const MAGNITUDE: f32 = 0.32;
        const SAMPLE_RATE: f32 = 48000.0;
        const OVERLAP: u32 = 8;
        const WINDOW_TYPE: WindowType = WindowType::SqrtHann;
        const PITCH_SHIFT: f32 = 1.5;

        let input_freqs = [500.0, 750.0, 850.0];
        let num_channels = input_freqs.len();

        for fft_size in [1024, 4096] {
            // Generate individual channel data and interleave it
            let mut separate_input = generate_sine_wave(input_freqs[0], SAMPLE_RATE, MAGNITUDE, DURATION);
            for i in 1..num_channels {
                separate_input.append(&mut generate_sine_wave(input_freqs[i], SAMPLE_RATE, MAGNITUDE, DURATION));
            }
            let mut input = vec![];
            interleave_samples(&separate_input, num_channels, &mut input);

            let mut params = PitchShiftParams::new(SAMPLE_RATE as u32, PITCH_SHIFT);
            params.fft_size = fft_size;
            params.overlap = OVERLAP;
            params.window_type = WINDOW_TYPE;

            let mut stretcher = MultiPitchShifter::new(&params, num_channels).unwrap();
            let output = process_all_multi(&mut stretcher, &input);

            assert!(output.len().is_multiple_of(num_channels));
            let output_len = output.len() / num_channels;
            let mut deinterleaved = vec![];
            deinterleave_samples(&output, num_channels, &mut deinterleaved);

            for ch in 0..num_channels {
                let expected_freq = input_freqs[ch] * PITCH_SHIFT;
                let channel_start = ch * output_len;
                let channel_end = (ch + 1) * output_len;
                let output_freq = compute_dominant_frequency(&deinterleaved[channel_start..channel_end], SAMPLE_RATE);
                let output_magn = compute_magnitude(&deinterleaved[channel_start..channel_end]);

                let bin_width = SAMPLE_RATE as f32 / fft_size as f32;
                let tolerance = bin_width * 2.0;
                let channel_input_len = input.len() / num_channels;
                println!(
                    "Channel {}: input {} Hz, output {} Hz, magnitude {}, input length {}, output length {}",
                    ch, expected_freq, output_freq, output_magn, channel_input_len, output_len
                );

                assert!(
                    (output_freq - expected_freq).abs() < tolerance,
                    "channel {}: expected {} Hz, got {} Hz for fft_size {}",
                    ch,
                    expected_freq,
                    output_freq,
                    fft_size
                );
                assert!(
                    (output_magn - MAGNITUDE).abs() < MAGNITUDE * 0.1,
                    "channel {}: expected magnitude {}, got {} for fft_size {}",
                    ch,
                    MAGNITUDE,
                    output_magn,
                    fft_size
                );

                assert!(
                    output_len.abs_diff(channel_input_len) < channel_input_len / 20,
                    "channel {}: expected output length {}, got {} for fft_size {}",
                    ch,
                    channel_input_len,
                    output_len,
                    fft_size
                );
            }
        }

        Ok(())
    }
}
