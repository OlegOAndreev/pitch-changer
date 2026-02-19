use std::mem;

use anyhow::{Result, bail};

/// Peak correction processing.
pub struct PeakCorrector {
    block_len: usize,
    recovery_rate: f32,
    num_channels: usize,

    // We always have two buffers: one contains the buffer which the gain will be applied to, the other contains the
    // samples which are used to compute the target gain. The output buffer is filled first.
    output_buf: Vec<f32>,
    analysis_buf: Vec<f32>,
    // There are multiple gains: the first is the current gain we arrived at, the second is the gain which is required
    // by the output_buf block to not clip (current_gain must be not greater than required_gain).
    current_gain: f32,
    required_gain: f32,
}

impl PeakCorrector {
    /// Create a new peak correction state
    pub fn new(block_len: usize, recovery_rate: f32, num_channels: usize) -> Result<Self> {
        // Validate parameters
        if block_len == 0 {
            bail!("Block size must be at least 1");
        }
        if recovery_rate <= 0.0 || recovery_rate > 1.0 {
            bail!("Recovery rate must be in (0.0, 1.0] range");
        }

        Ok(Self {
            block_len,
            recovery_rate,
            num_channels,
            output_buf: Vec::with_capacity(block_len * num_channels),
            analysis_buf: Vec::with_capacity(block_len * num_channels),
            current_gain: 1.0,
            required_gain: 1.0,
        })
    }

    /// Reset the state.
    pub fn reset(&mut self) {
        self.output_buf.clear();
        self.analysis_buf.clear();
        self.current_gain = 1.0;
        self.required_gain = 1.0;
    }

    /// Process interleaved audio samples with peak correction. One block_len of latency is used: current block is
    /// altered based on the target gain computed on the next block.
    pub fn process(&mut self, input: &[f32], output: &mut Vec<f32>) {
        let mut input_pos = 0;
        output.reserve(input.len());

        let block_size = self.block_len * self.num_channels;
        // This happens only at the start of processing.
        if self.output_buf.len() < block_size {
            let to_fill = block_size - self.output_buf.len();
            if to_fill > input.len() {
                self.output_buf.extend_from_slice(input);
                return;
            }
            self.output_buf.extend_from_slice(&input[..to_fill]);
            // Let's set current and required gain based on the contents of output_buf (this does not matter in
            // practice, but makes testing easier).
            self.current_gain = compute_required_gain(&self.output_buf);
            self.required_gain = self.current_gain;
            input_pos = to_fill;
        }

        while input_pos < input.len() {
            // Try to fill analysis buffer.
            let to_fill = block_size - self.analysis_buf.len();
            if input_pos + to_fill >= input.len() {
                self.analysis_buf.extend_from_slice(&input[input_pos..]);
                return;
            }
            self.analysis_buf.extend_from_slice(&input[input_pos..input_pos + to_fill]);
            input_pos += to_fill;

            self.calculate_and_apply_gain(output);
        }
    }

    /// Finish processing remaining samples.
    pub fn finish(&mut self, output: &mut Vec<f32>) {
        // Simply drain the current buffer with current gain.
        output.reserve(self.output_buf.len() + self.analysis_buf.len());
        for s in &self.output_buf {
            output.push(s * self.current_gain);
        }
        for s in &self.analysis_buf {
            output.push(s * self.current_gain);
        }
        self.reset();
    }

    /// Calculate gain from analysis block, write output block into output with gain.
    fn calculate_and_apply_gain(&mut self, output: &mut Vec<f32>) {
        assert!(
            self.current_gain <= self.required_gain,
            "got current_gain {} and required_gain {}",
            self.current_gain,
            self.required_gain
        );
        assert_eq!(self.output_buf.len(), self.block_len * self.num_channels);
        assert_eq!(self.analysis_buf.len(), self.block_len * self.num_channels);

        let analysis_gain = compute_required_gain(&self.analysis_buf);
        // The gain for current output block must be not greater than required_gain and we can recover gain not more
        // than recovery_rate per block.
        let target_gain = analysis_gain.min(self.required_gain).min(self.current_gain + self.recovery_rate);

        let step = (target_gain - self.current_gain) / self.block_len as f32;
        let mut gain = self.current_gain;
        for i in 0..self.block_len {
            for ch in 0..self.num_channels {
                output.push(self.output_buf[i * self.num_channels + ch] * gain);
            }
            gain += step;
        }
        self.current_gain = target_gain;
        self.required_gain = analysis_gain;

        // Move analysis_buf to output_buf.
        mem::swap(&mut self.analysis_buf, &mut self.output_buf);
        self.analysis_buf.clear();
    }
}

/// Compute required gain for signal (either mono or multichannel interleaved).
pub fn compute_required_gain(signal: &[f32]) -> f32 {
    let mut max_peak = 0.0;
    for &s in signal {
        let abs = s.abs();
        if abs > max_peak {
            max_peak = abs;
        }
    }
    if max_peak > 1.0 { 1.0 / max_peak } else { 1.0 }
}

#[cfg(test)]
mod tests {
    use crate::{compute_magnitude, generate_sine_wave, interleave_samples};

    use super::*;

    #[test]
    fn test_peak_correction_no_clipping() {
        let mut corrector = PeakCorrector::new(256, 0.01, 1).unwrap();

        let input = generate_sine_wave(440.0, 44100.0, 0.5, 1.0);
        let mut output = vec![];
        for chunk in input.chunks(3) {
            corrector.process(chunk, &mut output);
        }
        corrector.finish(&mut output);

        assert_eq!(input, output);
    }

    #[test]
    fn test_peak_correction_clipping() {
        let mut corrector = PeakCorrector::new(256, 0.01, 1).unwrap();

        // Create input that clips (peak = 1.5)
        let input = generate_sine_wave(44100.0, 440.0, 1.5, 1.0);

        let mut output = vec![];
        for chunk in input.chunks(3) {
            corrector.process(chunk, &mut output);
        }
        corrector.finish(&mut output);

        let max_output = compute_magnitude(&output);
        assert!(max_output <= 1.001, "Output should not clip, max was {}", max_output);
    }

    #[test]
    fn test_multi_channel_peak_detection() {
        let mut corrector = PeakCorrector::new(256, 0.01, 1).unwrap();

        // Create input that clips (peak = 1.5)
        let mut input = generate_sine_wave(44100.0, 440.0, 1.5, 1.0);
        input.extend_from_slice(&generate_sine_wave(44100.0, 440.0, 0.5, 1.0));
        let mut interleaved = vec![];
        interleave_samples(&input, 2, &mut interleaved);

        let mut output = vec![];
        for chunk in interleaved.chunks(3) {
            corrector.process(chunk, &mut output);
        }
        corrector.finish(&mut output);

        let max_output = compute_magnitude(&output);
        assert!(max_output <= 1.0001, "Output should not clip, max was {}", max_output);
    }

    #[test]
    fn test_gain_recovery() {
        let mut corrector = PeakCorrector::new(128, 0.5, 1).unwrap();
        // The input consists of 1 second of clipping and 1 second of not clipping.
        let mut clipping_input = generate_sine_wave(440.0, 44100.0, 1.5, 1.0);
        clipping_input.extend_from_slice(&generate_sine_wave(440.0, 44100.0, 1.0, 1.0));
        let mut output = vec![];
        corrector.process(&clipping_input, &mut output);
        corrector.finish(&mut output);
        // Gain should have recovered
        assert!((corrector.current_gain - 1.0).abs() < 1e-3, "Gain should have recovered: {}", corrector.current_gain);
    }
}
