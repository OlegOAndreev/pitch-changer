use std::f32::consts::PI;

/// Normalize a phase value to the range [-PI, PI).
pub fn normalize_phase(phase: f32) -> f32 {
    let shifted = phase + PI;
    shifted - (shifted / (2.0 * PI)).floor() * 2.0 * PI - PI
}

/// Compute the dominant frequency of a signal using FFT. Returns the frequency in Hz.
pub fn compute_dominant_frequency(signal: &[f32], sample_rate: f32) -> f32 {
    use realfft::RealFftPlanner;

    let n = signal.len();

    let fft_size = n.next_power_of_two();
    let mut planner = RealFftPlanner::<f32>::new();
    let r2c = planner.plan_fft_forward(fft_size);

    let mut input = vec![0.0; fft_size];
    input[..n].copy_from_slice(signal);
    let mut freq = r2c.make_output_vec();
    r2c.process(&mut input, &mut freq).expect("failed FFT");

    let mut max_magn = 0.0;
    let mut max_bin = 0;
    for (i, f) in freq.iter().enumerate() {
        let mag = f.norm();
        if mag > max_magn {
            max_magn = mag;
            max_bin = i;
        }
    }
    max_bin as f32 * sample_rate / fft_size as f32
}

/// Compute the magnitude of a signal by removing top-5 and bottom-5 outliers and taking the difference between
/// remaining min and max.
pub fn compute_magnitude(signal: &[f32]) -> f32 {
    let mut input = Vec::from(signal);
    input.sort_by(|a, b| a.total_cmp(b));
    if input.len() < 12 {
        return 0.0;
    }
    // Remove 5 bottom and top values and compute the difference between remaining min and max.
    (input[input.len() - 6] - input[5]) * 0.5
}

/// Generate a sine wave at the given frequency, sample rate, magnitude and duration.
pub fn generate_sine_wave(freq: f32, sample_rate: f32, magnitude: f32, duration: f32) -> Vec<f32> {
    let num_samples = (sample_rate * duration) as usize;
    let step = 2.0 * PI * freq / sample_rate;
    let mut result = Vec::with_capacity(num_samples);
    let mut phase = 0.0f32;
    for _ in 0..num_samples {
        result.push(magnitude * phase.sin());
        phase = normalize_phase(phase + step);
    }
    result
}

/// Deinterleave multi-channel audio samples:
///   [ch0s0, ch1s0, ch2s0, ch0s1, ch1s1, ...] -> [ch0s0, ch0s1, ..., ch1s0, ch1s1, ...]
///
/// The output is NOT cleared.
pub fn deinterleave_samples(input: &[f32], num_channels: usize, output: &mut Vec<f32>) {
    assert!(input.len().is_multiple_of(num_channels));

    let num_samples = input.len() / num_channels;
    output.reserve(input.len());

    // Fast-path for mono
    if num_channels == 1 {
        output.extend_from_slice(input);
        return;
    }

    let output_base = output.len();
    output.resize(output.len() + input.len(), 0.0);
    for ch in 0..num_channels {
        for sample_idx in 0..num_samples {
            unsafe {
                *output.get_unchecked_mut(output_base + ch * num_samples + sample_idx) =
                    *input.get_unchecked(sample_idx * num_channels + ch);
            }
        }
    }
}

/// Interleave multi-channel audio samples:
///    [ch0s0, ch0s1, ..., ch1s0, ch1s1, ...] -> [ch0s0, ch1s0, ch2s0, ch0s1, ch1s1, ...]
///
/// The output is NOT cleared.
pub fn interleave_samples(input: &[f32], num_channels: usize, output: &mut Vec<f32>) {
    assert!(input.len().is_multiple_of(num_channels));

    let num_samples = input.len() / num_channels;
    output.reserve(input.len());

    if num_channels == 1 {
        output.extend_from_slice(input);
        return;
    }

    let output_base = output.len();
    output.resize(output.len() + input.len(), 0.0);
    for sample_idx in 0..num_samples {
        for ch in 0..num_channels {
            unsafe {
                *output.get_unchecked_mut(output_base + sample_idx * num_channels + ch) =
                    *input.get_unchecked(ch * num_samples + sample_idx);
            };
        }
    }
}

/// Return linearly interpolated sample at given position.
pub fn linear_sample(input: &[f32], pos: f32) -> f32 {
    let index = pos as usize;
    if index >= input.len() - 1 {
        input[input.len() - 1]
    } else {
        let frac = pos.fract();
        input[index] * (1.0 - frac) + input[index + 1] * frac
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_phase() {
        // Values already in range. The PI gets normalized to -PI
        assert_eq!(normalize_phase(0.0), 0.0);
        assert_eq!(normalize_phase(PI), -PI);
        assert_eq!(normalize_phase(-PI), -PI);

        assert!((normalize_phase(PI / 2.0) - PI / 2.0).abs() < 1e-6);
        assert!((normalize_phase(-PI / 3.0) + PI / 3.0).abs() < 1e-6);
        assert!((normalize_phase(2.5 * PI) - (0.5 * PI)).abs() < 1e-6);
        assert!((normalize_phase(4.0 * PI) - 0.0).abs() < 1e-6);
        assert!((normalize_phase(-2.5 * PI) - (-0.5 * PI)).abs() < 1e-6);
        assert!((normalize_phase(-4.0 * PI) - 0.0).abs() < 1e-6);

        assert!(normalize_phase(1000.0 * PI).abs() < 1e-4);
        assert!(normalize_phase(-1000.0 * PI).abs() < 1e-4);
        assert!((normalize_phase(1000.0 * PI + 0.5) - 0.5).abs() < 1e-4);
        assert!((normalize_phase(-1000.0 * PI + 0.5) - 0.5).abs() < 1e-4);

        let epsilon = 1e-6;
        assert!((normalize_phase(PI + epsilon) - (-PI + epsilon)).abs() < 1e-5);
        assert!((normalize_phase(-PI - epsilon) - (PI - epsilon)).abs() < 1e-5);
    }

    #[test]
    fn test_compute_dominant_frequency() {
        for freq in [500.0, 1000.0] {
            for magnitude in [1.0, 5.0] {
                for sample_rate in [44100.0, 96000.0] {
                    let signal = generate_sine_wave(freq, sample_rate, magnitude, 1.0);
                    let computed_freq = compute_dominant_frequency(&signal, sample_rate);
                    let error = (computed_freq - freq).abs();
                    assert!(
                        error < 50.0,
                        "Frequency error too large: {} Hz (expected {} Hz, got {} Hz)",
                        error,
                        freq,
                        computed_freq
                    );
                }
            }
        }
    }

    #[test]
    fn test_compute_magnitude() {
        let magnitude = 2.5;
        let sine_wave = generate_sine_wave(400.0, 44100.0, magnitude, 1.0);
        let computed_magnitude = compute_magnitude(&sine_wave);
        assert!(
            (magnitude - computed_magnitude).abs() < 1e-1,
            "Sine wave magnitude out of expected range: {} (expected ~{})",
            computed_magnitude,
            magnitude
        );
    }

    #[test]
    fn test_deinterleave_samples() {
        let input_mono = vec![1.0, 2.0, 3.0, 4.0];
        let mut output = vec![];
        deinterleave_samples(&input_mono, 1, &mut output);
        assert_eq!(output, vec![1.0, 2.0, 3.0, 4.0]);

        let input_3ch = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        output.clear();
        deinterleave_samples(&input_3ch, 3, &mut output);
        // Expected: [C0_S0, C0_S1, C1_S0, C1_S1, C2_S0, C2_S1]
        assert_eq!(output, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_interleave_samples() {
        let input_mono = vec![1.0, 2.0, 3.0, 4.0];
        let mut output = vec![];
        interleave_samples(&input_mono, 1, &mut output);
        assert_eq!(output, vec![1.0, 2.0, 3.0, 4.0]);

        let input_3ch = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        output.clear();
        interleave_samples(&input_3ch, 3, &mut output);
        assert_eq!(output, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_deinterleave_interleave_roundtrip() {
        // Test roundtrip for various channel counts
        for num_channels in 1..=5 {
            let num_samples = 10;
            let total_samples = num_channels * num_samples;
            let original: Vec<f32> = (0..total_samples).map(|i| i as f32).collect();

            // Deinterleave
            let mut deinterleaved = vec![];
            deinterleave_samples(&original, num_channels, &mut deinterleaved);

            // Interleave back
            let mut interleaved = vec![];
            interleave_samples(&deinterleaved, num_channels, &mut interleaved);

            assert_eq!(interleaved, original);
        }
    }

    #[test]
    fn test_linear_sample() {
        let data = [3.0, 4.0, 5.0];
        // Last element
        assert_eq!(linear_sample(&data, -1.0), 3.0);
        assert_eq!(linear_sample(&data, 3.0), 5.0);
        assert_eq!(linear_sample(&data, 2.0), 5.0);
        assert_eq!(linear_sample(&data, 1.0), 4.0);
        assert_eq!(linear_sample(&data, 0.0), 3.0);
        assert_eq!(linear_sample(&data, 0.2), 3.2);
        assert_eq!(linear_sample(&data, 1.5), 4.5);
    }
}
