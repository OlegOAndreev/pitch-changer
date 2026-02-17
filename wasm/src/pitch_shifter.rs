use wasm_bindgen::prelude::*;

use anyhow::{Result, bail};

use crate::envelope_shifter::EnvelopeShifter;
use crate::multi_processor::{MonoProcessor, MultiProcessor};
use crate::peak_corrector::PeakCorrector;
use crate::resampler::StreamingResampler;
use crate::stft::{Stft, StftAccumBuf};
use crate::time_stretcher::{TimeStretchParams, TimeStretcher};
use crate::web::{Float32Vec, WrapAnyhowError};
use crate::window::WindowType;

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
    /// Cepstral lifter cutoff for formant preservation. Controls the separation between spectral envelope (formants)
    /// and fine structure (pitch harmonics). Lower values preserve more formant detail, higher values smooth more.
    /// Typical range: 0.5-1.5, try 1.0 for speech.
    ///
    /// If the default value of 0.0 is used, no formant preservation is applied.
    pub quefrency_cutoff: f32,
    /// Peak correction block size
    pub peak_correction_block_size: usize,
    /// Peak correction recovery rate per block
    pub peak_correction_recovery_rate: f32,
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
            quefrency_cutoff: 0.0,
            peak_correction_block_size: 256,
            peak_correction_recovery_rate: 0.01,
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
    params: PitchShiftParams,
    time_stretcher: TimeStretcher,
    resampler: StreamingResampler,

    // Formant processing, used if quefrency_cutoff != 0.0
    envelope_shift_enabled: bool,
    envelope_fft_size: usize,
    envelope_hop_size: usize,
    envelope_stft: Stft,
    envelope_shifter: EnvelopeShifter,

    // Scratch buffers. See TimeStretcher for description on how the output_accum_buf is filled. Unlike TimeStretcher,
    // we have a single hop size.
    stretched_buf: Vec<f32>,
    shifted_buf: Vec<f32>,
    output_accum_buf: StftAccumBuf,
}

impl PitchShifter {
    fn validate_params(params: &PitchShiftParams) -> Result<()> {
        if params.pitch_shift < 0.25 || params.pitch_shift > 4.0 {
            bail!("Pitch shifting factor cannot be lower than 0.25 or higher than 4");
        }
        // Most of the parameter validation will be done by TimeStretcher.
        Ok(())
    }

    fn do_envelope_processing(&mut self, output: &mut Vec<f32>) {
        assert!(self.envelope_shift_enabled);
        let output_capacity = self.shifted_buf.len() / self.envelope_hop_size * self.envelope_hop_size;
        output.reserve(output_capacity);

        let mut shifted_pos = 0;
        while shifted_pos + self.envelope_fft_size < self.shifted_buf.len() {
            // Do one STFT iteration.
            self.do_stft(shifted_pos);
            self.output_accum_buf.output_next(self.envelope_hop_size, output);

            shifted_pos += self.envelope_hop_size;
        }
        self.shifted_buf.drain(..shifted_pos);
    }

    fn finish_envelope_processing(&mut self, output: &mut Vec<f32>) {
        self.do_envelope_processing(output);
        // Process the remainder in self.shifted_buf by doing the simplest thing: the end has already been windowed by
        // TimeStretcher, so we do not care about pops/cracks.
        let remainder = self.shifted_buf.len();
        self.shifted_buf.resize(self.envelope_fft_size, 0.0);
        self.do_stft(0);
        self.output_accum_buf.output_next(remainder, output);
    }

    fn do_stft(&mut self, shifted_buf_pos: usize) {
        let norm_factor = self.envelope_stft.get_norm_factor(self.envelope_hop_size);
        let input = &self.shifted_buf[shifted_buf_pos..shifted_buf_pos + self.envelope_fft_size];
        let stft_output = self.envelope_stft.process(input, |ana_freq, syn_freq| {
            syn_freq.copy_from_slice(ana_freq);
            self.envelope_shifter.shift_envelope(syn_freq);
        });
        self.output_accum_buf.add(stft_output, norm_factor);
    }
}

impl MonoProcessor for PitchShifter {
    type Params = PitchShiftParams;

    fn new(params: &PitchShiftParams) -> Result<Self> {
        Self::validate_params(params)?;

        let mut time_stretch_params = TimeStretchParams::new(params.sample_rate, params.pitch_shift);
        time_stretch_params.fft_size = params.fft_size;
        time_stretch_params.overlap = params.overlap;
        time_stretch_params.window_type = params.window_type;

        let time_stretcher = TimeStretcher::new(&time_stretch_params)?;

        let resampler = StreamingResampler::new(1.0 / params.pitch_shift)?;

        let envelope_shift_enabled = params.quefrency_cutoff != 0.0;
        let envelope_fft_size = params.fft_size;
        let envelope_hop_size = envelope_fft_size / params.overlap as usize;
        let envelope_stft = Stft::new(envelope_fft_size, params.window_type);
        let num_bins = envelope_fft_size / 2 + 1;
        // We normalize the quefrency cutoff by pitch shift because we analyze the pitch shifted spectrum.
        let cepstrum_cutoff_samples =
            (params.quefrency_cutoff * params.sample_rate as f32 / (1000.0 * params.pitch_shift)) as usize;
        let envelope_shifter = EnvelopeShifter::new(num_bins, cepstrum_cutoff_samples, params.pitch_shift);

        Ok(Self {
            params: *params,
            time_stretcher,
            resampler,
            envelope_shift_enabled,
            envelope_fft_size,
            envelope_hop_size,
            envelope_stft,
            envelope_shifter,
            stretched_buf: vec![],
            shifted_buf: vec![],
            output_accum_buf: StftAccumBuf::new(envelope_fft_size),
        })
    }

    fn process(&mut self, input: &[f32], output: &mut Vec<f32>) {
        self.stretched_buf.clear();
        self.time_stretcher.process(input, &mut self.stretched_buf);
        if self.envelope_shift_enabled {
            self.resampler.resample(&self.stretched_buf, &mut self.shifted_buf);
            self.do_envelope_processing(output);
        } else {
            self.resampler.resample(&self.stretched_buf, output);
        }
    }

    fn finish(&mut self, output: &mut Vec<f32>) {
        self.stretched_buf.clear();
        self.time_stretcher.finish(&mut self.stretched_buf);
        if self.envelope_shift_enabled {
            self.resampler.resample(&self.stretched_buf, &mut self.shifted_buf);
            self.resampler.finish(&mut self.shifted_buf);
            self.finish_envelope_processing(output);
        } else {
            self.resampler.resample(&self.stretched_buf, output);
            self.resampler.finish(output);
        }
        self.reset();
    }

    fn reset(&mut self) {
        self.time_stretcher.reset();
        self.resampler.reset();
        self.stretched_buf.clear();
        self.shifted_buf.clear();
        self.output_accum_buf.reset();
    }

    fn set_param_value(&mut self, value: f32) {
        self.params.pitch_shift = value;
        self.time_stretcher.set_param_value(value);
        self.resampler.set_ratio(1.0 / self.params.pitch_shift);
        let cepstrum_cutoff_samples = (self.params.quefrency_cutoff * self.params.sample_rate as f32
            / (1000.0 * self.params.pitch_shift)) as usize;
        self.envelope_shifter.set_params(cepstrum_cutoff_samples, self.params.pitch_shift);
    }
}

/// Multi-channel pitch shifter that processes interleaved audio data with optional automatic peak correction.
#[wasm_bindgen]
pub struct MultiPitchShifter {
    inner: MultiProcessor<PitchShifter>,
    correction: PeakCorrector,
}

#[wasm_bindgen]
impl MultiPitchShifter {
    #[wasm_bindgen(constructor)]
    /// Create a new multi-channel pitch shifter for given number of channels.
    pub fn new(params: &PitchShiftParams, num_channels: usize) -> std::result::Result<Self, WrapAnyhowError> {
        let inner = MultiProcessor::<PitchShifter>::new(params, num_channels).map_err(WrapAnyhowError)?;

        let correction =
            PeakCorrector::new(params.peak_correction_block_size, params.peak_correction_recovery_rate, num_channels)
                .map_err(WrapAnyhowError)?;

        Ok(Self { inner, correction })
    }

    /// Process a chunk of interleaved audio samples through the pitch shifter with optional peak correction.
    #[wasm_bindgen]
    pub fn process(&mut self, input: &Float32Vec, output: &mut Float32Vec) {
        let mut inner_output = Float32Vec::new(0);
        self.inner.process(&input.0, &mut inner_output.0);
        self.correction.process(&inner_output.0, &mut output.0);
    }

    /// Finish processing any remaining audio data.
    #[wasm_bindgen]
    pub fn finish(&mut self, output: &mut Float32Vec) {
        let mut inner_output = Float32Vec::new(0);
        self.inner.finish(&mut inner_output.0);
        self.correction.process(&inner_output.0, &mut output.0);
        self.correction.finish(&mut output.0);
        self.reset();
    }

    /// Reset all internal state.
    #[wasm_bindgen]
    pub fn reset(&mut self) {
        self.inner.reset();
        self.correction.reset();
    }

    /// Set the pitch shift factor for all internal processors.
    #[wasm_bindgen]
    pub fn set_param_value(&mut self, value: f32) {
        self.inner.set_param_value(value);
    }
}

impl MultiPitchShifter {
    pub fn process_vec(&mut self, input: &[f32], output: &mut Vec<f32>) {
        let mut inner_output = Float32Vec::new(0);
        self.inner.process(&input, &mut inner_output.0);
        self.correction.process(&inner_output.0, output);
    }

    pub fn finish_vec(&mut self, output: &mut Vec<f32>) {
        let mut inner_output = Float32Vec::new(0);
        self.inner.finish(&mut inner_output.0);
        self.correction.process(&inner_output.0, output);
        self.correction.finish(output);
        self.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::{compute_dominant_frequency, compute_magnitude, generate_sine_wave};
    use anyhow::Result;

    fn process_all(shifter: &mut PitchShifter, input: &[f32]) -> Vec<f32> {
        const PREFIX_SIZE: usize = 1000;
        const PREFIX: f32 = -1234.0;

        let mut output = vec![PREFIX; PREFIX_SIZE];
        shifter.process(&input.to_vec(), &mut output);
        shifter.finish(&mut output);

        for i in 0..PREFIX_SIZE {
            assert_eq!(output[i], PREFIX);
        }
        output.drain(..PREFIX_SIZE);
        output
    }

    fn process_all_small_chunks(shifter: &mut PitchShifter, input: &[f32]) -> Vec<f32> {
        const PREFIX_SIZE: usize = 1000;
        const PREFIX: f32 = -1234.0;
        const CHUNK_SIZE: usize = 100;

        let mut output = vec![PREFIX; PREFIX_SIZE];
        for chunk in input.chunks(CHUNK_SIZE) {
            shifter.process(&chunk.to_vec(), &mut output);
        }
        shifter.finish(&mut output);

        for i in 0..PREFIX_SIZE {
            assert_eq!(output[i], PREFIX);
        }
        output.drain(..PREFIX_SIZE);
        output
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
            // pitch shift must be within 0.25..=2.0 and also <= overlap/3
            let max_pitch = (overlap as f32 / 3.0).min(2.0);
            let pitch_shift = rng.random_range(0.5..=max_pitch);
            // fft_size must be power of two, >= overlap, and divisible by overlap
            // generate exponent 9..=12 (512..=4096) which is >= overlap (max 32)
            let fft_size = 1 << rng.random_range(9..=12); // 512, 1024, 2048, 4096
            let quefrency_cutoff = if rng.random_bool(0.5) { 0.0 } else { rng.random_range(0.0..5.0) };

            let mut params = PitchShiftParams::new(sample_rate, pitch_shift);
            params.fft_size = fft_size;
            params.overlap = overlap;
            params.quefrency_cutoff = quefrency_cutoff;

            let len = rng.random_range(0..=4 * fft_size);
            let audio_data: Vec<f32> = (0..len).map(|_| rng.random_range(-1.0..1.0)).collect();

            let mut shifter = PitchShifter::new(&params).unwrap();
            let _ = process_all(&mut shifter, &audio_data);
            shifter.reset();
            let _ = process_all(&mut shifter, &audio_data);
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
        const MAGNITUDE: f32 = 0.94;
        const DURATION: f32 = 0.5;

        for sample_rate in [44100.0, 96000.0] {
            for fft_size in [1024, 4096] {
                for overlap in [8, 16] {
                    for window_type in [WindowType::Hann, WindowType::SqrtBlackman, WindowType::SqrtHann] {
                        for quefrency_cutoff in [0.0, 0.5, 100.0] {
                            let input = generate_sine_wave(FREQ, sample_rate, MAGNITUDE, DURATION);

                            let mut params = PitchShiftParams::new(sample_rate as u32, 1.0);
                            params.fft_size = fft_size;
                            params.overlap = overlap;
                            params.window_type = window_type;
                            params.quefrency_cutoff = quefrency_cutoff;
                            let mut shifter = PitchShifter::new(&params).unwrap();
                            let output = process_all_small_chunks(&mut shifter, &input);

                            // Skip transient at start and end, compare the middle
                            let offset = fft_size * 2;
                            let middle_len = (input.len() - offset * 2).min(output.len() - offset * 2);
                            // Account of resampling latency.
                            let output_offset = offset + StreamingResampler::LATENCY;
                            let mut max_diff = 0.0f32;
                            for i in 0..middle_len {
                                let diff = (input[offset + i] - output[output_offset + i]).abs();
                                if diff > max_diff {
                                    max_diff = diff;
                                }
                            }

                            println!(
                                "Sample rate {}, fft size {}, overlap {}, window {:?}, identity pitch shift, input length {}, output length {}",
                                sample_rate,
                                fft_size,
                                overlap,
                                window_type,
                                input.len(),
                                output.len()
                            );

                            assert!(
                                max_diff < 1e-3,
                                "Max difference {} for sample_rate {}, fft_size {}, overlap {}, window {:?}, quefrency cutoff {}",
                                max_diff,
                                sample_rate,
                                fft_size,
                                overlap,
                                window_type,
                                quefrency_cutoff
                            );
                        }
                    }
                }
            }
        }

        Ok(())
    }
}
