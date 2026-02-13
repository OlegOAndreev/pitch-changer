use anyhow::{Result, bail};

use crate::{deinterleave_samples, interleave_samples};

/// Internal trait used for MultiProcessor.
pub trait MonoProcessor : Sized {
    type Params;

    fn new(params: &Self::Params) -> Result<Self>;

    /// Process a chunk of audio samples through the processor. Output is NOT cleared.
    ///
    /// Note: you need to call `finish()` to receive the last output chunks after you call `process()` for all input
    /// samples.
    fn process(&mut self, input: &[f32], output: &mut Vec<f32>);

    /// Finish processing any remaining audio data in the internal buffers. This method must be called after all input
    /// has been processed via `process()`. Output is NOT cleared.
    ///
    /// Note: after calling `finish()`, the processor is reset and ready to process new audio data.
    fn finish(&mut self, output: &mut Vec<f32>);

    /// Reset the processor to its initial state.
    fn reset(&mut self);
}

/// Wrapper for mono audio processors which de-interleaves the audio and then calls separate processors per channel.
pub struct MultiProcessor<T: MonoProcessor> {
    processors: Vec<T>,
    num_channels: usize,
    // Scratch buffers
    deinterleaved_buf: Vec<f32>,
    output_buf: Vec<f32>,
}

impl<P: MonoProcessor> MultiProcessor<P> {
    /// Create a new multi-channel processor for given number of channels.
    pub fn new(params: &P::Params, num_channels: usize) -> Result<Self> {
        if num_channels == 0 {
            bail!("Number of channels must be at least 1");
        }

        let mut processors = vec![];
        for _ in 0..num_channels {
            processors.push(P::new(params)?);
        }

        Ok(Self { processors, num_channels, deinterleaved_buf: vec![], output_buf: vec![] })
    }

    /// Process interleaved multi-channel audio data.
    pub fn process(&mut self, input: &[f32], output: &mut Vec<f32>) {
        if self.num_channels == 1 {
            self.processors[0].process(input, output);
            return;
        }

        assert!(input.len().is_multiple_of(self.num_channels));

        self.deinterleaved_buf.clear();
        self.output_buf.clear();

        let samples_per_channel = input.len() / self.num_channels;
        deinterleave_samples(input, self.num_channels, &mut self.deinterleaved_buf);

        for (ch, processor) in self.processors.iter_mut().enumerate() {
            let channel_start = ch * samples_per_channel;
            let channel_end = (ch + 1) * samples_per_channel;
            processor.process(&self.deinterleaved_buf[channel_start..channel_end], &mut self.output_buf);
        }

        interleave_samples(&self.output_buf, self.num_channels, output);
    }

    /// Finish processing for all channels.
    pub fn finish(&mut self, output: &mut Vec<f32>) {
        if self.num_channels == 1 {
            self.processors[0].finish(output);
            return;
        }

        self.output_buf.clear();
        for processor in &mut self.processors {
            processor.finish(&mut self.output_buf);
        }

        interleave_samples(&self.output_buf, self.num_channels, output);
    }

    /// Reset all processors.
    pub fn reset(&mut self) {
        for processor in &mut self.processors {
            processor.reset();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A simple MonoProcessor that multiplies each sample by a constant factor.
    struct MultiplyProcessor {
        factor: f32,
    }

    impl MonoProcessor for MultiplyProcessor {
        type Params = f32;

        fn new(params: &Self::Params) -> Result<Self> {
            Ok(MultiplyProcessor { factor: *params })
        }

        fn process(&mut self, input: &[f32], output: &mut Vec<f32>) {
            for &sample in input {
                output.push(sample * self.factor);
            }
        }

        fn finish(&mut self, _output: &mut Vec<f32>) {
            // No internal buffers to flush
        }

        fn reset(&mut self) {
            // No state to reset
        }
    }

    #[test]
    fn test_multiply_processor_single_channel() {
        let factor = 2.5;
        let mut mp = MultiProcessor::<MultiplyProcessor>::new(&factor, 1).unwrap();
        
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let mut output = Vec::new();
        mp.process(&input, &mut output);
        
        assert_eq!(output, vec![2.5, 5.0, 7.5, 10.0]);
        
        // Test finish (should produce nothing extra)
        let mut finish_output = Vec::new();
        mp.finish(&mut finish_output);
        assert!(finish_output.is_empty());
        
        // Reset and process again
        mp.reset();
        let mut output2 = Vec::new();
        mp.process(&input, &mut output2);
        assert_eq!(output2, vec![2.5, 5.0, 7.5, 10.0]);
    }

    #[test]
    fn test_multiply_processor_multi_channel() {
        let factor = 3.0;
        let num_channels = 2;
        let mut mp = MultiProcessor::<MultiplyProcessor>::new(&factor, num_channels).unwrap();
        
        // Interleaved stereo: [L1, R1, L2, R2, L3, R3]
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut output = Vec::new();
        mp.process(&input, &mut output);
        
        // Expected: each channel multiplied by factor, interleaved preserved
        assert_eq!(output, vec![3.0, 6.0, 9.0, 12.0, 15.0, 18.0]);
        
        // Test finish
        let mut finish_output = Vec::new();
        mp.finish(&mut finish_output);
        assert!(finish_output.is_empty());
        
        // Reset and process again
        mp.reset();
        let mut output2 = Vec::new();
        mp.process(&input, &mut output2);
        assert_eq!(output2, vec![3.0, 6.0, 9.0, 12.0, 15.0, 18.0]);
    }

    #[test]
    fn test_multiply_processor_empty_input() {
        let factor = 2.0;
        let mut mp = MultiProcessor::<MultiplyProcessor>::new(&factor, 1).unwrap();
        
        let input = vec![];
        let mut output = Vec::new();
        mp.process(&input, &mut output);
        assert!(output.is_empty());
        
        let mut finish_output = Vec::new();
        mp.finish(&mut finish_output);
        assert!(finish_output.is_empty());
    }

    #[test]
    fn test_multiply_processor_zero_channels_error() {
        let factor = 1.0;
        let result = MultiProcessor::<MultiplyProcessor>::new(&factor, 0);
        assert!(result.is_err());
    }
}
