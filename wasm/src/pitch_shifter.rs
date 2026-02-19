use wasm_bindgen::prelude::*;

use anyhow::{Result, bail};

use crate::envelope_shifter::EnvelopeShifter;
use crate::peak_corrector::PeakCorrector;
use crate::resampler::StreamingResampler;
use crate::stft::{Stft, StftAccumBuf};
use crate::time_stretcher::{TimeStretchParams, TimeStretcher};
use crate::util::{deinterleave_samples, interleave_samples};
use crate::web::{Float32Vec, WrapAnyhowError};
use crate::window::WindowType;

/// Parameters for audio pitch shifting.
#[derive(Debug, Clone, Copy)]
#[wasm_bindgen]
pub struct PitchShiftParams {
    /// Pitch shift factor (e.g., 2.0 = one octave up, 0.5 = one octave down)
    pub pitch_shift: f32,
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
    pub fn new(sample_rate: u32, pitch_shift: f32, time_stretch: f32) -> Self {
        let stretch_params = TimeStretchParams::new(1.0);
        Self {
            pitch_shift,
            time_stretch,
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

impl PitchShiftParams {
    fn to_time_stretch(&self) -> TimeStretchParams {
        let effective_time_stretch = self.pitch_shift * self.time_stretch;
        TimeStretchParams {
            time_stretch: effective_time_stretch,
            fft_size: self.fft_size,
            overlap: self.overlap,
            window_type: self.window_type,
        }
    }
}

/// PitchShifter stretches the mono audio time by pitch_shift factor and then resamples the rate back so that the output
/// has the ~same length as the input.
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
    fn new(params: &PitchShiftParams) -> Result<Self> {
        Self::validate_params(params)?;

        let time_stretch_params = params.to_time_stretch();
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
        if self.params.pitch_shift == 1.0 {
            self.time_stretcher.process(input, output);
            return;
        }

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
        if self.params.pitch_shift == 1.0 {
            self.time_stretcher.finish(output);
            return;
        }

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

    fn update_params(&mut self, params: &PitchShiftParams) -> Result<()> {
        Self::validate_params(params)?;

        self.params = *params;
        let time_stretch_params = self.params.to_time_stretch();
        self.time_stretcher.update_params(&time_stretch_params);
        self.resampler.set_ratio(1.0 / params.pitch_shift);
        self.envelope_shift_enabled = params.quefrency_cutoff != 0.0;
        let cepstrum_cutoff_samples =
            (params.quefrency_cutoff * params.sample_rate as f32 / (1000.0 * params.pitch_shift)) as usize;
        self.envelope_shifter.set_params(cepstrum_cutoff_samples, params.pitch_shift);

        Ok(())
    }

    fn validate_params(params: &PitchShiftParams) -> Result<()> {
        if params.pitch_shift < 0.25 || params.pitch_shift > 4.0 {
            bail!("Pitch shifting factor cannot be lower than 0.25 or higher than 4");
        }
        // Validate that effective time stretch is within TimeStretcher's valid range
        let effective_time_stretch = params.pitch_shift * params.time_stretch;
        if effective_time_stretch < 0.5 || effective_time_stretch > 2.0 {
            bail!(
                "Effective time stretch (pitch_shift * time_stretch = {}) must be between 0.5 and 2.0",
                effective_time_stretch
            );
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

/// Multi-channel pitch shifter that processes interleaved audio data with optional automatic peak correction.
#[wasm_bindgen]
pub struct MultiPitchShifter {
    processors: Vec<PitchShifter>,
    num_channels: usize,
    corrector: PeakCorrector,
    // Scratch buffers for deinterleaving
    deinterleaved_buf: Vec<f32>,
    interleaved_buf: Vec<f32>,
    output_buf: Vec<f32>,
}

#[wasm_bindgen]
impl MultiPitchShifter {
    #[wasm_bindgen(constructor)]
    /// Create a new multi-channel pitch shifter for given number of channels.
    pub fn new(params: &PitchShiftParams, num_channels: usize) -> std::result::Result<Self, WrapAnyhowError> {
        let mut processors = vec![];
        for _ in 0..num_channels {
            processors.push(PitchShifter::new(params).map_err(WrapAnyhowError)?);
        }

        let corrector =
            PeakCorrector::new(params.peak_correction_block_size, params.peak_correction_recovery_rate, num_channels)
                .map_err(WrapAnyhowError)?;

        Ok(Self {
            processors,
            num_channels,
            corrector,
            deinterleaved_buf: vec![],
            interleaved_buf: vec![],
            output_buf: vec![],
        })
    }

    /// Process a chunk of interleaved audio samples through the pitch shifter with optional peak correction. output is
    /// NOT cleared.
    #[wasm_bindgen]
    pub fn process(&mut self, input: &Float32Vec, output: &mut Float32Vec) {
        self.process_vec(&input.0, &mut output.0);
    }

    /// Finish processing any remaining audio data. output is NOT cleared.
    #[wasm_bindgen]
    pub fn finish(&mut self, output: &mut Float32Vec) {
        self.finish_vec(&mut output.0);
    }

    /// Reset all internal state.
    #[wasm_bindgen]
    pub fn reset(&mut self) {
        for processor in &mut self.processors {
            processor.reset();
        }
        self.corrector.reset();
    }

    /// Update parameters.
    #[wasm_bindgen]
    pub fn update_params(&mut self, params: &PitchShiftParams) -> std::result::Result<(), WrapAnyhowError> {
        for processor in &mut self.processors {
            processor.update_params(params).map_err(WrapAnyhowError)?;
        }
        Ok(())
    }
}

impl MultiPitchShifter {
    pub fn process_vec(&mut self, input: &[f32], output: &mut Vec<f32>) {
        self.output_buf.clear();

        if self.num_channels == 1 {
            self.processors[0].process(input, &mut self.output_buf);
        } else {
            assert!(input.len().is_multiple_of(self.num_channels));
            self.deinterleaved_buf.clear();
            self.interleaved_buf.clear();

            let samples_per_channel = input.len() / self.num_channels;
            deinterleave_samples(input, self.num_channels, &mut self.deinterleaved_buf);

            for (ch, processor) in self.processors.iter_mut().enumerate() {
                let channel_start = ch * samples_per_channel;
                let channel_end = (ch + 1) * samples_per_channel;
                processor.process(&self.deinterleaved_buf[channel_start..channel_end], &mut self.interleaved_buf);
            }

            interleave_samples(&self.interleaved_buf, self.num_channels, &mut self.output_buf);
        }

        self.corrector.process(&self.output_buf, output);
    }

    pub fn finish_vec(&mut self, output: &mut Vec<f32>) {
        self.output_buf.clear();

        if self.num_channels == 1 {
            self.processors[0].finish(&mut self.output_buf);
        } else {
            self.interleaved_buf.clear();
            for processor in &mut self.processors {
                processor.finish(&mut self.interleaved_buf);
            }
            interleave_samples(&self.interleaved_buf, self.num_channels, &mut self.output_buf);
        }

        self.corrector.process(&self.output_buf, output);
        self.corrector.finish(output);
        self.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::{compute_dominant_frequency, compute_magnitude, generate_sine_wave, interleave_samples};
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
            let pitch_shift = rng.random_range(0.5..=2.0);
            let max_time_stretch = 1.95 / pitch_shift;
            let min_time_stretch = 0.55 / pitch_shift;
            let time_stretch = rng.random_range(min_time_stretch..=max_time_stretch);
            // fft_size must be power of two, >= overlap, and divisible by overlap
            // generate exponent 9..=12 (512..=4096) which is >= overlap (max 32)
            let fft_size = 1 << rng.random_range(9..=12); // 512, 1024, 2048, 4096
            let quefrency_cutoff = if rng.random_bool(0.5) { 0.0 } else { rng.random_range(0.0..5.0) };

            let mut params = PitchShiftParams::new(sample_rate as u32, pitch_shift, time_stretch);
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

                            let mut params = PitchShiftParams::new(sample_rate as u32, pitch_shift, 1.0);
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

                            // The + 1e-7 is a hack: it makes the pitch_shift != 1.0, while essentially not changing
                            // the rate.
                            let mut params = PitchShiftParams::new(sample_rate as u32, 1.0 + 1e-7, 1.0);
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

    #[test]
    fn test_multi_pitch_shifter_three_sine_waves() -> Result<()> {
        const DURATION: f32 = 0.5;
        const MAGNITUDE: f32 = 0.37;
        const SAMPLE_RATE: f32 = 44100.0;
        const FFT_SIZE: usize = 1024;
        const OVERLAP: u32 = 8;
        const PITCH_SHIFT: f32 = 1.2;
        const TIME_STRETCH: f32 = 1.4;

        let test_frequencies = [250.0, 440.0, 880.0];
        let num_channels = test_frequencies.len();

        let mut planar_data = vec![];

        for &freq in &test_frequencies {
            let sine_wave = generate_sine_wave(freq, SAMPLE_RATE, MAGNITUDE, DURATION);
            planar_data.extend(sine_wave);
        }

        let mut input_data = Vec::with_capacity(planar_data.len());
        interleave_samples(&planar_data, num_channels, &mut input_data);

        let mut params = PitchShiftParams::new(SAMPLE_RATE as u32, PITCH_SHIFT, TIME_STRETCH);
        params.fft_size = FFT_SIZE;
        params.overlap = OVERLAP;
        params.window_type = WindowType::Hann;

        let mut shifter = MultiPitchShifter::new(&params, num_channels)?;

        let mut output_data = vec![];
        shifter.process_vec(&input_data, &mut output_data);
        shifter.finish_vec(&mut output_data);

        let mut deinterleaved = vec![];
        deinterleave_samples(&output_data, num_channels, &mut deinterleaved);

        let output_samples_per_channel = output_data.len() / num_channels;

        for (ch, &input_freq) in test_frequencies.iter().enumerate() {
            let channel_output = &deinterleaved[output_samples_per_channel * ch..output_samples_per_channel * (ch + 1)];

            let output_freq = compute_dominant_frequency(&channel_output, SAMPLE_RATE);
            let expected_freq = input_freq * PITCH_SHIFT;
            let bin_width = SAMPLE_RATE / FFT_SIZE as f32;
            let tolerance = bin_width * 2.0;

            let expected_output_len = (planar_data.len() as f32 * TIME_STRETCH) as usize / num_channels;

            println!(
                "Channel {}: Input {} Hz, Output {:.2} Hz, Expected {:.2} Hz, Tolerance {:.2} Hz",
                ch, input_freq, output_freq, expected_freq, tolerance
            );

            assert!(
                (output_freq - expected_freq).abs() < tolerance,
                "Channel {}: Expected {} Hz, got {} Hz for input {} Hz with pitch shift {}",
                ch,
                expected_freq,
                output_freq,
                input_freq,
                PITCH_SHIFT
            );
            assert!(
                output_samples_per_channel.abs_diff(expected_output_len) < expected_output_len / 20,
                "expected output length {}, got {}",
                expected_output_len,
                output_samples_per_channel,
            );
        }

        Ok(())
    }
}
