use anyhow::{Result, bail};
use wasm_bindgen::prelude::*;

use crate::stft::Stft;
use crate::web::{Float32Vec, WrapAnyhowError};
use crate::{MonoProcessor, MultiProcessor, WindowType};

/// Parameters for formant-preserving pitch shifting.
#[derive(Debug, Clone, Copy)]
#[wasm_bindgen]
pub struct FormantPreservingPitchShifterParams {
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
    /// Cepstral lifter cutoff in milliseconds. Controls the separation between spectral envelope (formants) and fine
    /// structure (pitch harmonics). Lower values preserve more formant detail, higher values smooth more. Typical range:
    /// 1.0-5.0 ms.
    pub cepstrum_cutoff_ms: f32,
}

#[wasm_bindgen]
impl FormantPreservingPitchShifterParams {
    #[wasm_bindgen(constructor)]
    /// Create new formant-preserving pitch shift parameters with default FFT size, overlap, window and cepstrum cutoff.
    pub fn new(sample_rate: u32, pitch_shift: f32) -> Self {
        Self {
            pitch_shift,
            sample_rate,
            fft_size: 4096,
            overlap: 8,
            window_type: WindowType::Hann,
            cepstrum_cutoff_ms: 2.0,
        }
    }

    #[wasm_bindgen]
    /// Return a debug representation of the parameters for JS.
    pub fn to_debug_string(&self) -> String {
        format!("{:?}", self)
    }
}

/// Formant-preserving pitch shifter that delegates pitch shifting to `PitchShifter` and then applies spectrum envelope
/// correction as a post-processing step via STFT.
///
/// Note: maybe we should unify this with TimeStretcher?
#[wasm_bindgen]
pub struct FormantPreservingPitchShifter {
    params: FormantPreservingPitchShifterParams,
    hop_size: usize,
    stft: Stft,
    // phase_gradient_vocoder: PhaseGradientPitchShift,
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

impl FormantPreservingPitchShifter {
    fn validate_params(params: &FormantPreservingPitchShifterParams) -> Result<()> {
        if params.cepstrum_cutoff_ms <= 0.0 {
            bail!("Cepstrum cutoff must be positive");
        }
        if params.pitch_shift < 0.5 || params.pitch_shift > 2.0 {
            bail!("Pitch shifting factor cannot be lower than 0.5 or higher than 2");
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

    /// Do one iteration of stft
    fn do_stft(&mut self) {
        use crate::window::get_window_squared_sum;

        let output = self.stft.process(&self.input_buf, |ana_freq, syn_freq| {
            // syn_freq.copy_from_slice(ana_freq);
            self.phase_gradient_vocoder
                .process(ana_freq, self.hop_size, syn_freq, self.params.pitch_shift);
            // Ensure conjugate symmetry for real-valued inverse FFT. The first bin and last bin should have zero
            // imaginary part. After processing, they may become non-zero (even if very small).
            syn_freq[0].im = 0.0;
            syn_freq[syn_freq.len() - 1].im = 0.0;
        });
        // Normalization factor: inverse FFT scaling (1/fft_size) * squared windows sum
        let window_norm = get_window_squared_sum(self.params.window_type, self.params.fft_size, self.hop_size);
        let norm = 1.0 / (self.params.fft_size as f32 * window_norm);
        for (a, o) in self.output_accum_buf.iter_mut().zip(output) {
            *a += *o * norm;
        }
    }

    /// Append next part of output accum buf to result and shift the buffers.
    fn append_output_and_shift(&mut self, output: &mut Vec<f32>) {
        output.extend_from_slice(&self.output_accum_buf[..self.hop_size]);
        self.output_accum_buf.drain(0..self.hop_size);
        self.output_accum_buf.resize(self.params.fft_size, 0.0);
        self.input_buf.drain(0..self.hop_size);
    }
}

impl MonoProcessor for FormantPreservingPitchShifter {
    type Params = FormantPreservingPitchShifterParams;

    fn new(params: &FormantPreservingPitchShifterParams) -> Result<Self> {
        use crate::window::generate_tail_window;

        Self::validate_params(params).map_err(WrapAnyhowError)?;

        let hop_size = params.fft_size / params.overlap as usize;
        let stft = Stft::new(params.fft_size, params.window_type);
        let phase_gradient_vocoder = PhaseGradientPitchShift::new(params.fft_size);
        let tail_window = generate_tail_window(params.window_type, params.fft_size);
        let input_buf = Vec::with_capacity(params.fft_size);
        let output_accum_buf = vec![0.0f32; params.fft_size];

        Ok(Self { params: *params, hop_size, stft, phase_gradient_vocoder, tail_window, input_buf, output_accum_buf })
    }

    fn process(&mut self, input: &[f32], output: &mut Vec<f32>) {
        let output_capacity = input.len() / self.hop_size * self.hop_size;
        output.reserve(output_capacity);
        let mut input_pos = 0;
        while input_pos < input.len() {
            let needed = self.params.fft_size - self.input_buf.len();
            let available = input.len() - input_pos;
            let n = available.min(needed);
            self.input_buf.extend_from_slice(&input[input_pos..input_pos + n]);
            input_pos += n;

            if self.input_buf.len() == self.params.fft_size {
                self.do_stft();
                self.append_output_and_shift(output);
            }
        }
    }

    fn finish(&mut self, output: &mut Vec<f32>) {
        // See comment in TimeStretcher::finish()
        let output_capacity = self.params.fft_size;
        output.reserve(output_capacity);

        let iters = self.params.fft_size / self.hop_size;
        for i in 0..iters {
            self.input_buf.resize(self.params.fft_size, 0.0);
            self.do_stft();

            let tail_window_offset = i * self.hop_size;
            // After the last iteration do windowing first.
            for (a, w) in self.output_accum_buf[..self.hop_size]
                .iter_mut()
                .zip(&self.tail_window[tail_window_offset..])
            {
                *a *= *w;
            }
            self.append_output_and_shift(output);
        }

        self.reset();
    }

    fn reset(&mut self) {
        self.phase_gradient_vocoder.reset();
        self.input_buf.clear();
        self.output_accum_buf.fill(0.0);
    }
}

/// Multi-channel time stretcher that processes interleaved audio data.
#[wasm_bindgen]
pub struct MultiFormantPreservingPitchShifter(MultiProcessor<FormantPreservingPitchShifter>);

#[wasm_bindgen]
impl MultiFormantPreservingPitchShifter {
    #[wasm_bindgen(constructor)]
    /// Create a new multi-channel time stretcher for given number of channels.
    pub fn new(
        params: &FormantPreservingPitchShifterParams,
        num_channels: usize,
    ) -> std::result::Result<Self, WrapAnyhowError> {
        let inner =
            MultiProcessor::<FormantPreservingPitchShifter>::new(params, num_channels).map_err(WrapAnyhowError)?;
        Ok(Self(inner))
    }

    /// Process a chunk of interleaved audio samples through the time stretcher. Output is NOT cleared.
    ///
    /// Note: you need to call `finish()` to receive the last output chunks after you call `process()` for all input
    /// samples.
    #[wasm_bindgen]
    pub fn process(&mut self, input: &Float32Vec, output: &mut Float32Vec) {
        self.0.process(&input.0, &mut output.0);
    }

    /// Finish processing any remaining audio data in the internal buffers. This method must be called after all input
    /// has been processed via `process()`. Output is NOT cleared.
    ///
    /// Note: after calling `finish()`, the time stretcher is reset and ready to process new audio data.
    #[wasm_bindgen]
    pub fn finish(&mut self, output: &mut Float32Vec) {
        self.0.finish(&mut output.0);
    }

    /// Reset all internal time stretchers.
    #[wasm_bindgen]
    pub fn reset(&mut self) {
        self.0.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generate_sine_wave;
    use crate::util::compute_dominant_frequency;
    use anyhow::Result;

    fn process_all(shifter: &mut FormantPreservingPitchShifter, input: &[f32]) -> Vec<f32> {
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

    fn process_all_small_chunks(shifter: &mut FormantPreservingPitchShifter, input: &[f32]) -> Vec<f32> {
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
    fn test_formant_preserving_pitch_shifter_no_crash() {
        use rand::Rng;

        let mut rng = rand::rng();
        const ITERATIONS: usize = 50;

        for _ in 0..ITERATIONS {
            let sample_rate = rng.random_range(10000..=100000);
            let overlap = 1 << rng.random_range(2..=5); // 4, 8, 16, 32
            // pitch shift must be within 0.5..=2.0 and also <= overlap/3 (TimeStretcher constraint)
            let max_pitch = (overlap as f32 / 3.0).min(2.0);
            let pitch_shift = rng.random_range(0.5..=max_pitch);
            let fft_size = 1 << rng.random_range(9..=12); // 512, 1024, 2048, 4096

            let mut params = FormantPreservingPitchShifterParams::new(sample_rate, pitch_shift);
            params.fft_size = fft_size;
            params.overlap = overlap;
            params.cepstrum_cutoff_ms = rng.random_range(0.5..=10.0);

            let len = rng.random_range(0..=4 * fft_size);
            let audio_data: Vec<f32> = (0..len).map(|_| rng.random_range(-1.0..1.0)).collect();

            let mut shifter = FormantPreservingPitchShifter::new(&params).unwrap();
            let _output = process_all(&mut shifter, &audio_data);
        }
    }

    #[test]
    fn test_formant_preserving_pitch_shift_identity() -> Result<()> {
        const FREQ: f32 = 440.0;
        const MAGNITUDE: f32 = 1.0;
        const DURATION: f32 = 0.5;

        for sample_rate in [44100.0, 96000.0] {
            for fft_size in [1024, 4096] {
                for overlap in [8, 16] {
                    for window_type in [WindowType::Hann, WindowType::SqrtBlackman, WindowType::SqrtHann] {
                        let input = generate_sine_wave(FREQ, sample_rate, MAGNITUDE, DURATION);

                        let mut params = FormantPreservingPitchShifterParams::new(sample_rate as u32, 1.0);
                        params.fft_size = fft_size;
                        params.overlap = overlap;
                        params.window_type = window_type;
                        let mut shifter = FormantPreservingPitchShifter::new(&params).unwrap();
                        let output = process_all_small_chunks(&mut shifter, &input);

                        // Skip transient at start and end, compare the middle
                        let offset = fft_size * 2;
                        if output.len() < offset * 3 || input.len() < offset * 3 {
                            continue;
                        }
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
                            max_diff < 0.15,
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

    #[test]
    fn test_formant_preserving_pitch_shift_frequency() -> Result<()> {
        const DURATION: f32 = 0.5;
        const MAGNITUDE: f32 = 0.37;
        const INPUT_FREQ: f32 = 400.0;

        for sample_rate in [44100.0, 96000.0] {
            for fft_size in [1024, 4096] {
                for overlap in [8, 16] {
                    for pitch_shift in [0.5, 0.75, 1.25, 1.5, 2.0] {
                        let input = generate_sine_wave(INPUT_FREQ, sample_rate, MAGNITUDE, DURATION);

                        let mut params = FormantPreservingPitchShifterParams::new(sample_rate as u32, pitch_shift);
                        params.fft_size = fft_size;
                        params.overlap = overlap;
                        let mut shifter = FormantPreservingPitchShifter::new(&params).unwrap();
                        let output = process_all(&mut shifter, &input);

                        let expected_freq = INPUT_FREQ * pitch_shift;
                        let output_freq = compute_dominant_frequency(&output, sample_rate);

                        let bin_width = sample_rate as f32 / params.fft_size as f32;
                        let tolerance = bin_width * 2.0;
                        println!(
                            "Formant-preserving: sample rate {}, fft size {}, overlap {}, pitch shift {}, input {} Hz, output {} Hz, expected {} Hz, input length {}, output length {}",
                            sample_rate,
                            fft_size,
                            overlap,
                            pitch_shift,
                            INPUT_FREQ,
                            output_freq,
                            expected_freq,
                            input.len(),
                            output.len()
                        );

                        assert!(
                            (output_freq - expected_freq).abs() < tolerance,
                            "expected {} Hz, got {} Hz for sample rate {}, fft size {}, overlap {}, pitch shift {}",
                            expected_freq,
                            output_freq,
                            sample_rate,
                            fft_size,
                            overlap,
                            pitch_shift,
                        );

                        // Output length should be roughly the same as input (no time stretching)
                        assert!(
                            output.len().abs_diff(input.len()) < input.len() / 5,
                            "expected output length ~{}, got {} for sample rate {}, fft size {}, overlap {}, pitch shift {}",
                            input.len(),
                            output.len(),
                            sample_rate,
                            fft_size,
                            overlap,
                            pitch_shift,
                        );
                    }
                }
            }
        }

        Ok(())
    }
}
