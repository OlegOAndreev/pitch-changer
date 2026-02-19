use std::mem;

use audioadapter_buffers::direct::InterleavedSlice;
use rubato::{Async, Resampler};

use crate::web::WrapAnyhowError;

/// A resampler that uses rubato's FFT-based resampling with buffering for arbitrary input chunks.
pub struct StreamingResampler {
    resampler: Async<f32>,
    // Buffer for leftover input samples that do not form a full chunk
    previous_input: Vec<f32>,
}

// Rubato streaming interface is really really horrible. Thanks to Deepseek and Resampler::process_all_into_buffer
// giving inspiration on how to actually use it.
impl StreamingResampler {
    #[allow(unused)]
    pub const LATENCY: usize = 1;

    /// Create a new resampler with the given sample rate ratio. The ratio is output sample rate / input sample rate.
    pub fn new(sample_rate_ratio: f32) -> std::result::Result<Self, WrapAnyhowError> {
        // We do not care too much about quality in case of pitch shifting.
        let resampler = Async::<f32>::new_poly(
            sample_rate_ratio as f64,
            4.0,
            rubato::PolynomialDegree::Cubic,
            512,
            1,
            rubato::FixedAsync::Input,
        )
        .map_err(|e| WrapAnyhowError(anyhow::anyhow!("Failed to create resampler: {:?}", e)))?;

        let previous_input = Vec::with_capacity(resampler.input_frames_next());

        Ok(Self { resampler, previous_input })
    }

    /// Resample part of the input audio. The remainder will be buffered and used during next `resample()` or `finish()`
    /// calls. Output is NOT cleared.
    ///
    /// Note: you need to call `finish()` to receive the last output chunks after you call `resample()` for all input
    /// samples.
    pub fn resample(&mut self, input: &[f32], output: &mut Vec<f32>) {
        if input.is_empty() {
            return;
        }

        // input_frames_next() and output_frames_next() return constant values.
        let input_chunk_size = self.resampler.input_frames_next();
        let output_chunk_size = self.resampler.output_frames_max();

        let output_capacity = (self.previous_input.len() + input.len()) / input_chunk_size * output_chunk_size;
        output.reserve(output_capacity);

        let mut input_pos = 0;

        // If previous chunk is not empty, fill it and process the full chunk.
        if !self.previous_input.is_empty() {
            assert!(
                self.previous_input.len() < input_chunk_size,
                "self.previous_chunk.len() = {}, input_chunk_size = {}",
                self.previous_input.len(),
                input_chunk_size
            );
            let to_copy = input_chunk_size - self.previous_input.len();
            if input.len() < to_copy {
                self.previous_input.extend_from_slice(input);
                return;
            }

            self.previous_input.extend_from_slice(&input[..to_copy]);
            self.resample_previous_chunk(output);
            input_pos += to_copy;
        }

        // Now run the main loop
        while input_pos + input_chunk_size <= input.len() {
            self.resample_chunk(&input[input_pos..input_pos + input_chunk_size], output);
            input_pos += input_chunk_size;
        }

        // Store the remainder in the previous_chunk
        self.previous_input.clear();
        self.previous_input.extend_from_slice(&input[input_pos..]);
        assert!(
            self.previous_input.len() < input_chunk_size,
            "self.previous_chunk.len() = {}, input_chunk_size = {}",
            self.previous_input.len(),
            input_chunk_size
        );
    }

    /// Finish processing any remaining audio data in the internal buffers. Output is NOT cleared.
    ///
    /// Note: after calling `finish()`, the resampler is reset and ready to process new audio data.
    pub fn finish(&mut self, output: &mut Vec<f32>) {
        if self.previous_input.is_empty() {
            return;
        }
        self.resample_previous_chunk(output);

        self.reset();
    }

    /// Reset the resampler to its initial state, clearing any internal buffers.
    pub fn reset(&mut self) {
        self.resampler.reset();
        self.previous_input.clear();
    }

    /// Update the resampling ratio.
    pub fn set_ratio(&mut self, sample_rate_ratio: f32) {
        self.resampler
            .set_resample_ratio(sample_rate_ratio as f64, true)
            .expect("Failed to change resampling ratio");
    }

    fn resample_chunk(&mut self, input: &[f32], output: &mut Vec<f32>) {
        let input_chunk_size = self.resampler.input_frames_next();
        let output_chunk_size = self.resampler.output_frames_next();
        let input_slice = InterleavedSlice::new(input, 1, input.len()).expect("InterleavedSlice::new");
        let output_pos = output.len();
        output.resize(output_pos + output_chunk_size, 0.0);
        let mut output_slice = InterleavedSlice::new_mut(&mut output[output_pos..], 1, output_chunk_size)
            .expect("InterleavedSlice::new_mut");
        // I am really confused by rubato API and its combination of adapters and parameters.
        let indexing = rubato::Indexing {
            input_offset: 0,
            output_offset: 0,
            partial_len: Some(input.len()),
            active_channels_mask: None,
        };
        // We ignore the output delay because it is very small for async resamplers.
        let (input_consumed, output_written) = self
            .resampler
            .process_into_buffer(&input_slice, &mut output_slice, Some(&indexing))
            .expect("process_into_buffer");
        assert_eq!(input_consumed, input_chunk_size);
        output.truncate(output_pos + output_written);
    }

    fn resample_previous_chunk(&mut self, output: &mut Vec<f32>) {
        // This is an annoying dance around Rust borrowck: temporarily move the previous_chunk into local var and move
        // local var back after processing. The alternative would be creating a RefCell, but I do not like how
        // cumbersome it is.
        let mut previous_chunk = mem::take(&mut self.previous_input);
        self.resample_chunk(&previous_chunk, output);
        mem::swap(&mut self.previous_input, &mut previous_chunk);
        // Sanity check that nothing was added to self.previous_chunk during resample_chunk.
        assert!(previous_chunk.is_empty());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::{compute_dominant_frequency, compute_magnitude, generate_sine_wave};
    use anyhow::Result;

    fn resample_all_small_chunks(resampler: &mut StreamingResampler, input: &[f32]) -> Vec<f32> {
        const PREFIX_SIZE: usize = 1000;
        const PREFIX: f32 = -1234.0;
        const CHUNK_SIZE: usize = 256;

        let mut output = vec![PREFIX; PREFIX_SIZE];
        for chunk in input.chunks(CHUNK_SIZE) {
            resampler.resample(chunk, &mut output);
        }
        resampler.finish(&mut output);
        for i in 0..PREFIX_SIZE {
            assert_eq!(output[i], PREFIX);
        }
        output.drain(..PREFIX_SIZE);
        output
    }

    #[test]
    fn test_resampler_different_ratios() -> Result<()> {
        const INPUT_FREQ: f32 = 550.0;
        const MAGNITUDE: f32 = 1.0;
        const DURATION: f32 = 1.5;
        const SAMPLE_RATE: f32 = 48000.0;

        let input = generate_sine_wave(INPUT_FREQ, SAMPLE_RATE, MAGNITUDE, DURATION);

        // Test various ratios
        for &ratio in &[0.5, 0.75, 1.25, 1.5] {
            let mut resampler = StreamingResampler::new(ratio)?;
            let output = resample_all_small_chunks(&mut resampler, &input);

            let expected_freq = INPUT_FREQ / ratio;
            let output_freq = compute_dominant_frequency(&output, SAMPLE_RATE);
            let output_magn = compute_magnitude(&output);

            let expected_len = (input.len() as f32 * ratio) as usize;

            let bin_width = SAMPLE_RATE as f32 / input.len() as f32;
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
