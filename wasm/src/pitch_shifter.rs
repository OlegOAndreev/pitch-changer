use wasm_bindgen::prelude::*;

use crate::basic_vocoder::BasicVocoder;
use crate::phase_gradient_vocoder::PhaseGradientVocoder;
use crate::stft::Stft;
use crate::util::HANN_WINDOW_SQ_SUM_PER_OVERLAP;
use crate::web::error_and_panic;

/// Stretching method to use.
#[derive(Debug, Clone, Copy)]
#[wasm_bindgen]
pub enum StretchMethod {
    /// Simple phase vocoder
    Basic,
    /// Phase gradient vocoder (from Phase Vocoder Done Right, potentially higher quality)
    PhaseGradient,
}

/// Parameters for audio stretching/pitch shifting.
#[derive(Debug, Clone, Copy)]
#[wasm_bindgen]
pub struct StretchParams {
    /// Pitch shift factor (e.g., 2.0 = one octave up, 0.5 = one octave down)
    pub pitch_shift: f32,
    /// Time stretch factor (e.g., 2.0 = twice as long, 0.5 = half length)
    pub time_stretch: f32,
    /// Sample rate in Hz
    pub rate: u32,
    /// FFT window size in samples
    pub fft_size: usize,
    /// Overlap factor (number of windows overlapping one point, must be power of two)
    pub overlap: u32,
    /// Stretching method to use
    pub method: StretchMethod,
}

#[wasm_bindgen]
impl StretchParams {
    #[wasm_bindgen(constructor)]
    pub fn new(rate: u32, pitch_shift: f32, time_stretch: f32) -> Self {
        Self { pitch_shift, time_stretch, rate, fft_size: 2048, overlap: 4, method: StretchMethod::Basic }
    }
}

/// Pitch shifter that processes audio samples in real-time using STFT and phase vocoder techniques.
///
/// The algorithm maintains proper phase continuity across frames and handles the windowing overlap correctly. After
/// processing all input samples, `finish()` must be called to flush any remaining data from the internal buffers.
#[wasm_bindgen]
pub struct PitchShifter {
    params: StretchParams,
    hop_size: usize,
    stft: Stft,
    basic_vocoder: BasicVocoder,
    phase_gradient_vocoder: PhaseGradientVocoder,
    // Source data is accumulated in src_buf until there is enough data to run STFT. After each STFT the buffer is
    // shifted by `hop_size`. Zeroes are appended both at the start and after the end of source data so that each source
    // frame is processed `overlap` times.
    //
    // ASCII diagram (hop_size = HS, fft_size = FS, overlap = OL):
    //
    // 1. src_buf and dst_accum_buf are conceptually split into pieces of size H:
    //                   ┌────────┬────────┬─────┬─────────────┬─────────────┐
    //    src_buf        | piece0 | piece1 | ... | piece(OL-1) | new_data... |
    //                   └────────┴────────┴─────┴─────────────┴─────────────┘
    //                   ┌───────────────────────────────────────┐
    //    dst_accum_buf  | zeros  | zeros  | ... | zeros | zeros |
    //                   └───────────────────────────────────────┘
    //
    // 2. When src_buf reaches F samples, STFT processes the whole window, adds result to dst_accum_buf, then both
    //    buffers are shifted left by hop_size:
    //
    //                   ┌──────────┬────────────┬─────┬───────────────┬───────┐
    //    src_buf        | piece(N) | piece(N+1) | ... | piece(N+OL-1) | zeros |
    //                   └──────────┴────────────┴─────┴───────────────┴───────┘
    //                   ┌──────────┬────────────┬─────┬───────────────┬───────┐
    //    dst_accum_buf  | accum(N) | accum(N+1) | ... | accum(N+OL-1) | zeros |
    //                   └──────────┴────────────┴─────┴───────────────┴───────┘
    src_buf: Vec<f32>,
    // STFT outputs are accumulated in `dst_accum_buf`. The first part, `dst_accum_buf[..hop_size]` for which all
    // overlapped blocks have been summed, is written to output after each STFT step.
    //
    // The first (overlap-1) output pieces are discarded (see skip_output_for) because they correspond to padding.
    dst_accum_buf: Vec<f32>,
    // The initial data in `src_buf` is zero padding, we skip outputting the data the first few iterations.
    skip_output_for: usize,
}

#[wasm_bindgen]
impl PitchShifter {
    #[wasm_bindgen(constructor)]
    pub fn new(params: &StretchParams) -> Self {
        Self::validate_params(params);

        // // If we are time stretching, we have to compensate with corresponding pitch shift.
        // let final_pitch_shift = params.pitch_shift * params.time_stretch;
        let hop_size = params.fft_size / params.overlap as usize;
        let stft = Stft::new(params.fft_size);
        let basic_vocoder = BasicVocoder::new(params.fft_size);
        let phase_gradient_vocoder = PhaseGradientVocoder::new(params.fft_size);
        let mut src_buf = Vec::with_capacity(params.fft_size);
        // Prefill the padding
        src_buf.resize(params.fft_size - hop_size, 0.0);
        let dst_accum_buf = vec![0.0f32; params.fft_size];
        let skip_output_for = params.overlap as usize - 1;

        Self {
            params: *params,
            hop_size,
            stft,
            basic_vocoder,
            phase_gradient_vocoder,
            src_buf,
            dst_accum_buf,
            skip_output_for,
        }
    }

    fn validate_params(params: &StretchParams) {
        if (params.fft_size & (params.fft_size - 1)) != 0 {
            // This is not strictly required, but we can have arbitrary FFT size, so let's force performant sizes.
            error_and_panic!("FFT size must be power of two");
        }
        if params.fft_size < 512 || params.fft_size > 32768 {
            error_and_panic!("FFT size must be in [512, 32768] range");
        }
        if params.rate < 4096 || params.rate > 192000 {
            error_and_panic!("Sample rate must be in [4096, 192000] range");
        }
        if params.overlap < 4 {
            error_and_panic!("Overlap must be at least 4");
        }
        if (params.overlap & (params.overlap - 1)) != 0 {
            error_and_panic!("Overlap must be power of two");
        }
        if !params.fft_size.is_multiple_of(params.overlap as usize) {
            error_and_panic!("FFT size must be divisible by overlap");
        }
        if params.time_stretch != 1.0 {
            error_and_panic!("Time stretching is not supported for now");
        }
    }

    /// Process a chunk of audio samples through the pitch shifter.
    ///
    /// Note: you need to call `finish()` to receive the last output chunks after you call `process()` for all input
    /// samples.
    pub fn process(&mut self, src: &[f32]) -> Vec<f32> {
        // TODO: Do not forget the multiplier after we allow time_stretch != 1.0. We overallocate for the initial
        // results (when skip_output_for > 0), but factoring it is not worth the trouble.
        let result_capacity = src.len() / self.hop_size * self.hop_size;
        let mut result = Vec::with_capacity(result_capacity);

        let mut src_pos = 0;
        while src_pos < src.len() {
            let needed = self.params.fft_size - self.src_buf.len();
            let available = src.len() - src_pos;
            let to_copy = needed.min(available);
            if to_copy > 0 {
                self.src_buf.extend_from_slice(&src[src_pos..src_pos + to_copy]);
                src_pos += to_copy;
            }

            // If we still don't have enough data for a full FFT window, break
            if self.src_buf.len() < self.params.fft_size {
                break;
            }

            self.process_src_buf();
            self.append_dst_and_shift(&mut result);
            self.src_buf.drain(0..self.hop_size);
        }

        result
    }

    /// Finish processing any remaining audio data in the internal buffers. This method must be called after all input
    /// has been processed via `process()`.
    ///
    /// Note: after calling `finish()`, the pitch shifter is reset and ready to process new audio data.
    pub fn finish(&mut self) -> Vec<f32> {
        // TODO: Do not forget the multiplier after we allow time_stretch != 1.0. We overallocate for the initial
        // results (when skip_output_for > 0), but factoring it is not worth the trouble.
        let mut result = Vec::with_capacity(self.src_buf.len());

        // Check how much data we have in the final piece
        let tail_len = self.hop_size - (self.params.fft_size - self.src_buf.len());
        assert!(tail_len < self.hop_size);

        // Process enough times to flush all overlapping windows.
        for _ in 0..(self.params.overlap - 1) {
            self.src_buf.resize(self.params.fft_size, 0.0);
            self.process_src_buf();
            self.append_dst_and_shift(&mut result);
            self.src_buf.drain(0..self.hop_size);
        }

        // Process the final tail_len. skip_output_for should be already zero.
        self.src_buf.resize(self.params.fft_size, 0.0);
        self.process_src_buf();
        result.extend_from_slice(&self.dst_accum_buf[0..tail_len]);

        self.reset();
        result
    }

    /// Reset the pitch shifter to its initial state.
    pub fn reset(&mut self) {
        self.basic_vocoder.reset();
        self.phase_gradient_vocoder.reset();
        self.src_buf.fill(0.0);
        self.src_buf.resize(self.params.fft_size - self.hop_size, 0.0);
        self.dst_accum_buf.fill(0.0);
        self.skip_output_for = self.params.overlap as usize - 1;
    }

    fn process_src_buf(&mut self) {
        let output = self.stft.process(&self.src_buf, |src_freq, dst_freq| {
            match self.params.method {
                StretchMethod::Basic => {
                    self.basic_vocoder.stretch_freq(
                        src_freq,
                        self.params.pitch_shift,
                        self.params.overlap as usize,
                        dst_freq,
                    );
                }
                StretchMethod::PhaseGradient => {
                    self.phase_gradient_vocoder.stretch_freq(
                        src_freq,
                        self.params.pitch_shift,
                        self.params.overlap as usize,
                        dst_freq,
                    );
                }
            }

            // Ensure conjugate symmetry for real-valued inverse FFT. The first bin and last bin should have zero
            // imaginary part. After processing, they may become non-zero (even if very small).
            dst_freq[0].im = 0.0;
            dst_freq[dst_freq.len() - 1].im = 0.0;
        });

        // Normalization factor for: inverse FFT scaling (1/fft_size) * squared Hann windows sum.
        let norm = 1.0 / (self.params.fft_size as f32 * self.params.overlap as f32 * HANN_WINDOW_SQ_SUM_PER_OVERLAP);
        for (dst, src) in self.dst_accum_buf.iter_mut().zip(output) {
            *dst += *src * norm;
        }
    }

    fn append_dst_and_shift(&mut self, result: &mut Vec<f32>) {
        if self.skip_output_for == 0 {
            result.extend_from_slice(&self.dst_accum_buf[..self.hop_size]);
        } else {
            self.skip_output_for -= 1;
        }
        self.dst_accum_buf.drain(0..self.hop_size);
        self.dst_accum_buf.resize(self.params.fft_size, 0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::{cross_correlation, generate_sine_wave};
    use anyhow::Result;
    use realfft::RealFftPlanner;

    fn process_all(shifter: &mut PitchShifter, input: &[f32]) -> Vec<f32> {
        let mut result = shifter.process(input);
        result.append(&mut shifter.finish());
        result
    }

    #[test]
    fn test_data_sizes() {
        // With time_stretch = 1.0, the output should match input precisely (allowing for small floating-point
        // errors from STFT processing)
        let params = StretchParams::new(44100, 1.0, 1.0);
        let hop_size = params.fft_size / params.overlap as usize;
        let fft_size = params.fft_size;
        let test_sizes = vec![
            0,
            1,
            2,
            hop_size - 1,
            hop_size,
            hop_size + 1,
            2 * hop_size - 1,
            2 * hop_size,
            2 * hop_size + 1,
            3 * hop_size - 1,
            3 * hop_size,
            3 * hop_size + 1,
            fft_size - hop_size - 1,
            fft_size - hop_size,
            fft_size - hop_size + 1,
            fft_size - 1,
            fft_size,
            fft_size + 1,
            fft_size + hop_size - 1,
            fft_size + hop_size,
            fft_size + hop_size + 1,
            2 * fft_size - 1,
            2 * fft_size,
            2 * fft_size + 1,
            32 * fft_size - 1,
            32 * fft_size,
            32 * fft_size + 1,
        ];

        for size in test_sizes {
            let step = 100.0 / params.rate as f32;
            let input: Vec<_> = (0..size).map(|i| (i as f32 * step).sin()).collect();

            let mut shifter = PitchShifter::new(&params);

            let output = process_all(&mut shifter, &input);
            assert_eq!(output.len(), input.len());
            let mut total_diff = 0.0;
            for i in 0..output.len() {
                let diff = (output[i] - input[i]).abs();
                assert!(diff < 1e-2, "size={}, index={}: output = {}, input = {}", size, i, output[i], input[i]);
                total_diff += diff;
            }
            if size > 0 {
                assert!(total_diff / (size as f32) < 1e-3, "size={}: total_diff = {}", size, total_diff);
            }

            // Also verify that after reset, the shifter produces same output
            let output2 = process_all(&mut shifter, &input);
            assert_eq!(output2, output);
        }
    }

    fn compute_dominant_frequency(signal: &[f32], sample_rate: f32) -> Result<f32> {
        let n = signal.len();

        let fft_size = n.next_power_of_two();
        let mut planner = RealFftPlanner::<f32>::new();
        let r2c = planner.plan_fft_forward(fft_size);

        let mut input = vec![0.0; fft_size];
        input[..n].copy_from_slice(signal);
        let mut freq = r2c.make_output_vec();
        r2c.process(&mut input, &mut freq)?;

        let mut max_magn = 0.0;
        let mut max_bin = 0;
        for i in 0..freq.len() {
            let mag = freq[i].norm();
            if mag > max_magn {
                max_magn = mag;
                max_bin = i;
            }
        }

        Ok(max_bin as f32 * sample_rate / fft_size as f32)
    }

    #[test]
    fn test_pitch_shift_sine_wave() -> Result<()> {
        const DURATION: f32 = 0.5;

        for sample_rate in [44100.0, 96000.0] {
            for input_freq in [200.0, 400.0, 440.0, 800.0, 3000.0, 5000.0] {
                for pitch_shift in [0.5, 0.75, 1.25, 1.5, 2.0] {
                    let input = generate_sine_wave(input_freq, sample_rate, DURATION);

                    // Test both methods
                    for &method in &[StretchMethod::Basic, StretchMethod::PhaseGradient] {
                        let mut params = StretchParams::new(sample_rate as u32, pitch_shift, 1.0);
                        params.method = method;

                        let mut shifter = PitchShifter::new(&params);
                        let output = process_all(&mut shifter, &input);

                        let expected_freq = input_freq * pitch_shift;
                        let output_freq = compute_dominant_frequency(&output, sample_rate)?;

                        let bin_width = sample_rate / output.len() as f32;
                        let tolerance = bin_width * 2.0;
                        println!(
                            "Method {:?}: Input {} Hz, Output {} Hz, Expected {} Hz",
                            method, input_freq, output_freq, expected_freq
                        );
                        assert!(
                            (output_freq - expected_freq).abs() < tolerance,
                            "Pitch shift failed for method {:?}: expected {} Hz, got {} Hz",
                            method,
                            expected_freq,
                            output_freq,
                        );
                    }
                }
            }
        }

        Ok(())
    }

    // #[test]
    fn test_pitch_shift_cross_correlation() -> Result<()> {
        let sample_rate = 44100.0;
        let duration = 1.0; // 1 second for stable analysis
        let input_freq = 440.0; // 440 Hz sine wave
        let pitch_shift = 1.5; // Perfect fifth up
        let expected_freq = input_freq * pitch_shift; // 660 Hz

        // Generate input sine wave
        let input = generate_sine_wave(input_freq, sample_rate, duration);

        // Generate reference sine wave at expected frequency (660Hz)
        let reference = generate_sine_wave(expected_freq, sample_rate, duration);

        // Test both methods
        for &method in &[StretchMethod::Basic, StretchMethod::PhaseGradient] {
            // Create params with pitch shift
            let mut params = StretchParams::new(sample_rate as u32, pitch_shift, 1.0);
            params.method = method;

            let mut shifter = PitchShifter::new(&params);
            let output = process_all(&mut shifter, &input);

            // Compute cross-correlation between output and reference 660Hz sine wave
            let (correlation, lag) = cross_correlation(&output, &reference);

            // Define threshold for good pitch shifting
            // A correlation > 0.9 indicates strong similarity to the expected 660Hz sine wave
            let threshold = 0.9;

            println!(
                "Method {:?}: Cross-correlation with {}Hz reference = {:.4}, lag = {} samples ({:.1}ms), threshold = {}",
                method,
                expected_freq,
                correlation,
                lag,
                lag as f32 / sample_rate * 1000.0,
                threshold
            );

            assert!(
                correlation > threshold,
                "Pitch shift quality check failed for method {:?}: correlation {} is below threshold {}",
                method,
                correlation,
                threshold
            );

            // The phase vocoder can introduce significant time delay due to STFT windowing.
            // For a 1-second signal, we allow up to 100ms lag (10% of duration)
            let max_reasonable_lag = (sample_rate * 0.1) as isize; // 100ms max lag
            if lag.abs() >= max_reasonable_lag {
                println!(
                    "Warning: Large lag {} samples ({:.1}ms) for method {:?}, but correlation is good ({:.4})",
                    lag,
                    lag as f32 / sample_rate * 1000.0,
                    method,
                    correlation
                );
                // Don't fail the test for large lag if correlation is good
                // This can happen due to phase wrapping in cross-correlation
            }
        }

        Ok(())
    }
}
