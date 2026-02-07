use audioadapter_buffers::direct::InterleavedSlice;
use rubato::{Fft, FixedSync, Resampler};

use crate::web::WrapAnyhowError;

const FFT_SIZE: u32 = 512;

/// A resampler that uses rubato's FFT-based resampling with buffering for arbitrary input chunks.
pub struct StreamingResampler {
    resampler: Fft<f32>,
    /// Buffer for leftover input samples that are not a full chunk.
    leftover_buffer: Vec<f32>,
}

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

        Ok(Self { resampler, leftover_buffer: Vec::new() })
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

        // Combine leftover buffer with new input
        let mut combined_input = Vec::with_capacity(self.leftover_buffer.len() + input.len());
        combined_input.extend_from_slice(&self.leftover_buffer);
        combined_input.extend_from_slice(input);

        // Clear leftover buffer for this call
        self.leftover_buffer.clear();

        let chunk_size = self.resampler.input_frames_next();
        let total_samples = combined_input.len();

        // If we don't have enough for even one chunk, store everything in leftover buffer
        if total_samples < chunk_size {
            self.leftover_buffer = combined_input;
            return Vec::new();
        }

        // Calculate how many complete chunks we can process
        let complete_chunks = total_samples / chunk_size;
        let leftover = total_samples % chunk_size;

        // Process complete chunks using process_into_buffer
        let mut output = Vec::new();
        for chunk_idx in 0..complete_chunks {
            let start = chunk_idx * chunk_size;
            let end = start + chunk_size;
            let chunk = &combined_input[start..end];

            let input_slice = InterleavedSlice::new(chunk, 1, chunk.len()).expect("InterleavedSlice::new error");

            // Output size for this chunk
            let output_len = self.resampler.output_frames_next();
            let mut output_chunk = vec![0.0f32; output_len];
            let mut output_slice =
                InterleavedSlice::new_mut(&mut output_chunk, 1, output_len).expect("InterleavedSlice::new_mut error");

            // Process the chunk
            let (input_consumed, output_written) = self
                .resampler
                .process_into_buffer(&input_slice, &mut output_slice, None)
                .expect("process_into_buffer error");

            assert_eq!(input_consumed, chunk_size);
            output_chunk.truncate(output_written);
            output.extend_from_slice(&output_chunk);
        }

        // Store leftover samples for next call
        if leftover > 0 {
            let start = complete_chunks * chunk_size;
            self.leftover_buffer.extend_from_slice(&combined_input[start..]);
        }

        output
    }

    /// Finish processing any remaining audio data in the internal buffers.
    ///
    /// Note: after calling `finish()`, the resampler is reset and ready to process new audio data.
    pub fn finish(&mut self, output: &mut Vec<f32>)  {
        use audioadapter_buffers::direct::InterleavedSlice;

        if self.leftover_buffer.is_empty() {
            return;
        }

        let chunk_size = self.resampler.input_frames_next();
        let leftover_len = self.leftover_buffer.len();

        // If we have leftover samples, pad with zeros to make a complete chunk
        if leftover_len > 0 {
            let mut padded_input = self.leftover_buffer.clone();
            padded_input.resize(chunk_size, 0.0);

            let input_slice =
                InterleavedSlice::new(&padded_input, 1, padded_input.len()).expect("InterleavedSlice::new error");

            let output_len = self.resampler.output_frames_next();
            let offset = output.len();
            output.resize(offset + output_len, 0.0);
            let mut output_slice = InterleavedSlice::new_mut(&mut output[offset..], 1, output_len)
                .expect("InterleavedSlice::new_mut error");

            let (input_consumed, output_written) = self
                .resampler
                .process_into_buffer(&input_slice, &mut output_slice, None)
                .expect("process_into_buffer error");

            assert_eq!(input_consumed, chunk_size);
            output.truncate(offset + output_written);
        }
        self.reset();
    }

    /// Reset the resampler to its initial state, clearing any internal buffers.
    pub fn reset(&mut self) {
        self.resampler.reset();
        self.leftover_buffer.clear();
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
