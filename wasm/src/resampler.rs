use anyhow::anyhow;

use crate::web::WrapAnyhowError;

/// A cubic-spline based resampler which has a streaming interface. Originally it was a wrapper around Rubato, but
/// Rubato interface is way too convoluted and I decided to remove the dependency. Cubic spline is enough for our
/// use case anyway.
pub struct StreamingResampler {
    // Step is split into integer and fractional part, step_fract is in [0, 1)
    step_int: usize,
    step_fract: f64,

    // Absolute position of start of the next window (4 samples) in the input stream. It originally is -1 because we pad
    // the input with 0.0 at the start, so that the first input element is not skipped.
    next_pos_int: isize,
    next_pos_fract: f64,
    // Total length of processed input chunks.
    total_input_len: usize,
    // Leftover input samples from previous chunks.
    prev_input: Vec<f32>,
}

// 4-point, 3rd-order Lagrange polynomial interpolation. Boundary conditions: interpolate_cubic(0.0) == y1,
// interpolate_cubic(1.0) == y2 See https://yehar.com/blog/wp-content/uploads/2009/08/deip-original.pdf for details:
// while B-spline has lower SNR, it does not pass through any of the points at t = 0, which makes testing kinda annoying
// :-)
//
// Taken from https://github.com/HEnquist/rubato/blob/master/src/asynchro_fast.rs
#[inline(always)]
fn interpolate_cubic(t: f32, y0: f32, y1: f32, y2: f32, y3: f32) -> f32 {
    let x2 = t * t;
    let x3 = x2 * t;
    let a0 = y1;
    let a1 = -(1.0 / 3.0) * y0 - 0.5 * y1 + y2 - (1.0 / 6.0) * y3;
    let a2 = 0.5 * (y0 + y2) - y1;
    let a3 = 0.5 * (y1 - y2) + (1.0 / 6.0) * (y3 - y0);
    a0 + a1 * t + a2 * x2 + a3 * x3
}

impl StreamingResampler {
    /// Create a new resampler with the given sample rate ratio. The ratio is output sample rate / input sample rate.
    pub fn new(sample_rate_ratio: f64) -> std::result::Result<Self, WrapAnyhowError> {
        if sample_rate_ratio < 1e-2 {
            return Err(WrapAnyhowError(anyhow!(
                "sample_rate_ratio must be greater than 0.01, is {}",
                sample_rate_ratio
            )));
        }

        let step_int = (1.0 / sample_rate_ratio).floor() as usize;
        let step_fract = (1.0 / sample_rate_ratio).fract();

        Ok(Self {
            step_int,
            step_fract,
            next_pos_int: -1,
            next_pos_fract: 0.0,
            total_input_len: 0,
            prev_input: vec![0.0],
        })
    }

    /// Resample part of the input audio. The remainder will be buffered and used during next `resample()` or `finish()`
    /// calls. Output is NOT cleared.
    ///
    /// Note: you need to call `finish()` to receive the last output chunks after you call `resample()` for all input
    /// samples.
    pub fn resample(&mut self, input: &[f32], output: &mut Vec<f32>) {
        let input_start_idx = self.total_input_len;
        let input_end_idx = self.total_input_len + input.len();
        // Available input samples: previous_input are input samples up to start_idx-1, input are input samples from
        // start_idx and up.
        let get_input_slow = |idx: isize| -> f32 {
            if idx < input_start_idx as isize {
                let from_end = (input_start_idx as isize - idx) as usize;
                self.prev_input[self.prev_input.len() - from_end]
            } else {
                input[idx as usize - input_start_idx]
            }
        };

        let mut pos_int = self.next_pos_int;
        let mut pos_fract = self.next_pos_fract;
        while pos_int < input_start_idx as isize && pos_int + 3 < input_end_idx as isize {
            let y0 = get_input_slow(pos_int);
            let y1 = get_input_slow(pos_int + 1);
            let y2 = get_input_slow(pos_int + 2);
            let y3 = get_input_slow(pos_int + 3);
            output.push(interpolate_cubic(pos_fract as f32, y0, y1, y2, y3));
            (pos_int, pos_fract) = self.step(pos_int, pos_fract);
        }
        // Fast path: pos_int is inside input slice, no need to call get_input_slow()
        while pos_int + 3 < input_end_idx as isize {
            let idx = pos_int as usize - input_start_idx;
            let buf = &input[idx..idx + 4];
            output.push(interpolate_cubic(pos_fract as f32, buf[0], buf[1], buf[2], buf[3]));
            (pos_int, pos_fract) = self.step(pos_int, pos_fract);
        }

        // Store the at most 3 elements from input tail into self.prev_input. If input contains fewer than 3 elements,
        // append them to current prev_input.
        let save_to_prev_input = input.len().min(3);
        let prev_input_to_drain = if self.prev_input.len() + save_to_prev_input > 3 {
            self.prev_input.len() + save_to_prev_input - 3
        } else {
            0
        };
        self.prev_input.drain(..prev_input_to_drain);
        self.prev_input.extend_from_slice(&input[input.len() - save_to_prev_input..]);

        self.next_pos_int = pos_int;
        self.next_pos_fract = pos_fract;
        self.total_input_len = input_end_idx;
    }

    /// Finish processing any remaining audio data in the internal buffers. Output is NOT cleared.
    ///
    /// Note: after calling `finish()`, the resampler is reset and ready to process new audio data.
    pub fn finish(&mut self, output: &mut Vec<f32>) {
        // Append two zeros to input to process all input values
        self.resample(&[0.0, 0.0], output);
        self.reset();
    }

    /// Reset the resampler to its initial state, clearing any internal buffers.
    pub fn reset(&mut self) {
        self.next_pos_int = -1;
        self.next_pos_fract = 0.0;
        self.total_input_len = 0;
        self.prev_input.clear();
        self.prev_input.push(0.0);
    }

    /// Update the resampling ratio.
    pub fn set_ratio(&mut self, sample_rate_ratio: f64) {
        if sample_rate_ratio < 1e-2 {
            panic!("sample_rate_ratio must be greater than 0.01, is {}", sample_rate_ratio);
        }
        self.step_int = (1.0 / sample_rate_ratio).floor() as usize;
        self.step_fract = (1.0 / sample_rate_ratio).fract();
    }

    #[inline(always)]
    fn step(&self, mut pos_int: isize, mut pos_fract: f64) -> (isize, f64) {
        pos_int += self.step_int as isize;
        pos_fract += self.step_fract;
        if pos_fract >= 1.0 {
            pos_int += 1;
            pos_fract -= 1.0;
        }
        (pos_int, pos_fract)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::{compute_dominant_frequency, compute_magnitude, generate_sine_wave};
    use anyhow::Result;

    // Test cases where the output should be exact.
    #[test]
    fn test_resampler_exact() -> Result<()> {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0];

        fn compare(resampler: &mut StreamingResampler, input: &[f32], expected: &[f32]) {
            let mut output = vec![];
            for i in 0..input.len() {
                resampler.resample(&input[i..i + 1], &mut output);
            }
            resampler.finish(&mut output);
            assert_eq!(output, expected);

            let mut output = vec![];
            resampler.reset();
            for i in (0..input.len()).step_by(2) {
                resampler.resample(&input[i..i + 2], &mut output);
            }
            resampler.finish(&mut output);
            assert_eq!(output, expected);

            let mut output = vec![];
            resampler.reset();
            resampler.resample(&input, &mut output);
            resampler.finish(&mut output);
            assert_eq!(output, expected);
        }

        let mut resampler = StreamingResampler::new(1.0)?;
        compare(&mut resampler, &input, &input);

        let mut resampler = StreamingResampler::new(1.0 / 2.0)?;
        compare(&mut resampler, &input, &[1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0]);

        let mut resampler = StreamingResampler::new(1.0 / 3.0)?;
        compare(&mut resampler, &input, &[1.0, 4.0, 7.0, 10.0, 13.0, 16.0]);

        let mut resampler = StreamingResampler::new(1.0 / 4.0)?;
        compare(&mut resampler, &input, &[1.0, 5.0, 9.0, 13.0, 17.0]);

        let mut resampler = StreamingResampler::new(1.0 / 5.0)?;
        compare(&mut resampler, &input, &[1.0, 6.0, 11.0, 16.0]);

        let mut resampler = StreamingResampler::new(1.0 / 6.0)?;
        compare(&mut resampler, &input, &[1.0, 7.0, 13.0]);

        let mut resampler = StreamingResampler::new(1.0 / 7.0)?;
        compare(&mut resampler, &input, &[1.0, 8.0, 15.0]);

        let mut resampler = StreamingResampler::new(1.0 / 8.0)?;
        compare(&mut resampler, &input, &[1.0, 9.0, 17.0]);

        let mut resampler = StreamingResampler::new(1.0 / 9.0)?;
        compare(&mut resampler, &input, &[1.0, 10.0]);

        let mut resampler = StreamingResampler::new(2.0)?;
        let mut output = vec![];
        resampler.resample(&input, &mut output);
        resampler.finish(&mut output);
        // Take every second element
        for i in 0..input.len() {
            assert!(
                (input[i] - output[i * 2]).abs() < 1e-5,
                "expected output {}, got {} for index {}",
                input[i],
                output[i * 2],
                i
            );
        }

        let mut resampler = StreamingResampler::new(10.0)?;
        let mut output = vec![];
        resampler.resample(&input, &mut output);
        resampler.finish(&mut output);
        // Take every 10th element
        for i in 0..input.len() {
            assert!(
                (input[i] - output[i * 10]).abs() < 1e-5,
                "expected output {}, got {} for index {}",
                input[i],
                output[i * 10],
                i
            );
        }

        Ok(())
    }

    fn resample_all_small_chunks(resampler: &mut StreamingResampler, input: &[f32]) -> Vec<f32> {
        const PREFIX_SIZE: usize = 1000;
        const PREFIX: f32 = -1234.0;
        const CHUNK_SIZE: usize = 199;

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
        for &ratio in &[0.1, 0.3, 0.5, 0.75, 1.25, 1.3, 1.5, 15.0] {
            let mut resampler = StreamingResampler::new(ratio)?;
            let output = resample_all_small_chunks(&mut resampler, &input);

            let expected_freq = INPUT_FREQ / ratio as f32;
            let output_freq = compute_dominant_frequency(&output, SAMPLE_RATE);
            let output_magn = compute_magnitude(&output);

            let expected_len = (input.len() as f64 * ratio) as usize;

            let bin_width = SAMPLE_RATE as f32 / input.len() as f32;
            let tolerance = bin_width * 4.0;
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
            // Check that frequency, magnitude and output length are roughly what is expected.
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
                output.len().abs_diff(expected_len) < expected_len / 100,
                "expected output length {}, got {} for ratio {}",
                expected_len,
                output.len(),
                ratio
            );
        }

        Ok(())
    }
}
