use std::mem;

use audioadapter_buffers::direct::InterleavedSlice;
use rubato::{Fft, FixedSync, Resampler};

use crate::web::WrapAnyhowError;

const FFT_SIZE: u32 = 512;

/// A resampler that uses rubato's FFT-based resampling with buffering for arbitrary input chunks.
pub struct StreamingResampler {
    resampler: Fft<f32>,
    // Buffer for leftover input samples that do not form a full chunk
    previous_chunk: Vec<f32>,
    // See Fft::output_delay()
    should_delay_output: bool,
}

// Rubato streaming interface is really really horrible. Thanks to Deepseek and Resampler::process_all_into_buffer
// giving inspiration on how to actually use it.
impl StreamingResampler {
    /// Create a new resampler with the given sample rate ratio. The ratio is output sample rate / input sample rate.
    pub fn new(sample_rate: u32, sample_rate_ratio: f32) -> std::result::Result<Self, WrapAnyhowError> {
        let resampler = Fft::<f32>::new(
            sample_rate as usize,
            // The params rate is likely smth like 44100 or 48000, so we do not bother too much with this rounding of
            // fractional sample rate.
            (sample_rate as f32 * sample_rate_ratio) as usize,
            FFT_SIZE as usize,
            1,
            1,
            FixedSync::Input,
        )
        .map_err(|e| WrapAnyhowError(anyhow::anyhow!("Failed to create resampler: {:?}", e)))?;

        let previous_chunk = Vec::with_capacity(resampler.input_frames_next());

        Ok(Self { resampler, previous_chunk, should_delay_output: true })
    }

    /// Resample part of the input audio. The remainder will be buffered and used during next `resample()` or `finish()`
    /// calls.
    ///
    /// Note: you need to call `finish()` to receive the last output chunks after you call `resample()` for all input
    /// samples.
    pub fn resample(&mut self, input: &[f32]) -> Vec<f32> {
        if input.is_empty() {
            return vec![];
        }

        // input_frames_next() and output_frames_next() return constant values.
        let input_chunk_size = self.resampler.input_frames_next();
        let output_chunk_size = self.resampler.output_frames_max();

        let output_capacity = (self.previous_chunk.len() + input.len()) / input_chunk_size * output_chunk_size;
        let mut output = Vec::with_capacity(output_capacity);

        let mut input_pos = 0;

        // If previous chunk is not empty, fill it and process the full chunk.
        if !self.previous_chunk.is_empty() {
            assert!(
                self.previous_chunk.len() < input_chunk_size,
                "self.previous_chunk.len() = {}, input_chunk_size = {}",
                self.previous_chunk.len(),
                input_chunk_size
            );
            let to_copy = input_chunk_size - self.previous_chunk.len();
            if input.len() < to_copy {
                self.previous_chunk.extend_from_slice(input);
                return vec![];
            }

            self.previous_chunk.extend_from_slice(&input[..to_copy]);
            // This is an annoying dance around Rust borrowck.
            let mut previous_chunk = mem::take(&mut self.previous_chunk);
            self.resample_chunk(&previous_chunk, &mut output);
            mem::swap(&mut self.previous_chunk, &mut previous_chunk);
            input_pos += to_copy;
        }

        // Now run the main loop
        while input_pos + input_chunk_size <= input.len() {
            self.resample_chunk(&input[input_pos..], &mut output);
            input_pos += input_chunk_size;
        }

        // Store the remainder in the previous_chunk
        self.previous_chunk.clear();
        self.previous_chunk.extend_from_slice(&input[input_pos..]);
        assert!(
            self.previous_chunk.len() < input_chunk_size,
            "self.previous_chunk.len() = {}, input_chunk_size = {}",
            self.previous_chunk.len(),
            input_chunk_size
        );

        output
    }

    /// Finish processing any remaining audio data in the internal buffers.
    ///
    /// Note: after calling `finish()`, the resampler is reset and ready to process new audio data.
    pub fn finish(&mut self, output: &mut Vec<f32>) {
        if self.previous_chunk.is_empty() {
            return;
        }

        let input_chunk_size = self.resampler.input_frames_next();
        self.previous_chunk.resize(input_chunk_size, 0.0);
        // This is an annoying dance around Rust borrowck.
        let previous_chunk = mem::take(&mut self.previous_chunk);
        self.resample_chunk(&previous_chunk, output);

        self.reset();
    }

    /// Reset the resampler to its initial state, clearing any internal buffers.
    pub fn reset(&mut self) {
        self.resampler.reset();
        self.previous_chunk.clear();
    }

    fn resample_chunk(&mut self, input: &[f32], output: &mut Vec<f32>) {
        let input_chunk_size = self.resampler.input_frames_next();
        let output_chunk_size = self.resampler.output_frames_next();
        let input_slice = InterleavedSlice::new(input, 1, input_chunk_size).expect("InterleavedSlice::new");
        let output_pos = output.len();
        output.resize(output_pos + output_chunk_size, 0.0);
        let mut output_slice = InterleavedSlice::new_mut(&mut output[output_pos..], 1, output_chunk_size)
            .expect("InterleavedSlice::new_mut");
        let (input_consumed, output_written) = self
            .resampler
            .process_into_buffer(&input_slice, &mut output_slice, None)
            .expect("process_into_buffer");
        assert_eq!(input_consumed, input_chunk_size);
        output.truncate(output_pos + output_written);

        if self.should_delay_output && output_written > 0 {
            assert!(
                self.resampler.output_delay() <= output_written,
                "self.resampler.output_delay() = {}, output_written = {}",
                self.resampler.output_delay(),
                output_written
            );
            let truncated_len = output.len() - self.resampler.output_delay();
            output.truncate(truncated_len);
            self.should_delay_output = false;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::generate_sine_wave;
    use crate::{compute_dominant_frequency, compute_magnitude};
    use anyhow::Result;

    fn resample_all_small_chunks(resampler: &mut StreamingResampler, input: &[f32]) -> Vec<f32> {
        const CHUNK_SIZE: usize = 256;

        let mut output = Vec::new();
        for chunk in input.chunks(CHUNK_SIZE) {
            output.extend_from_slice(&resampler.resample(chunk));
        }
        resampler.finish(&mut output);
        output
    }

    #[test]
    fn test_resampler_different_ratios() -> Result<()> {
        const INPUT_FREQ: f32 = 550.0;
        const MAGNITUDE: f32 = 1.0;
        const DURATION: f32 = 0.5;
        const SAMPLE_RATE: f32 = 48000.0;

        let input = generate_sine_wave(INPUT_FREQ, SAMPLE_RATE, MAGNITUDE, DURATION);

        // Test various ratios
        for &ratio in &[0.5, 0.75, 1.25, 1.5] {
            let mut resampler = StreamingResampler::new(SAMPLE_RATE as u32, ratio)?;
            let output = resample_all_small_chunks(&mut resampler, &input);

            let expected_freq = INPUT_FREQ / ratio;
            let output_freq = compute_dominant_frequency(&output, SAMPLE_RATE);
            let output_magn = compute_magnitude(&output);

            let expected_len = (input.len() as f32 * ratio) as usize;

            let bin_width = SAMPLE_RATE as f32 / FFT_SIZE as f32;
            let tolerance = bin_width * 2.0;
            println!(
                "Sample rate {}, ratio {}, output {} Hz, expected {} Hz, magnitude {}, input length {}, output length {}",
                SAMPLE_RATE,
                ratio,
                output_freq,
                expected_freq,
                output_magn,
                input.len(),
                output.len()
            );
            // Check that frequency, magnitude and output length are roughtly what is expected.
            assert!(
                (output_freq - expected_freq).abs() < tolerance,
                "expected {} Hz, got {} Hz for ratio {}",
                expected_freq,
                output_freq,
                ratio
            );
            assert!(
                (output_magn - MAGNITUDE).abs() < MAGNITUDE * 0.1,
                "expected magnitude {}, got {} for ratio {}",
                MAGNITUDE,
                output_magn,
                ratio
            );
            assert!(
                output.len().abs_diff(expected_len) < expected_len / 20,
                "expected output length {}, got {} for ratio {}",
                expected_len,
                output.len(),
                ratio
            );
        }

        Ok(())
    }
}
