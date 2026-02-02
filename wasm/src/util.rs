use std::f32::consts::PI;

/// Normalize a phase value to the range [-PI, PI)
pub fn normalize_phase(phase: f32) -> f32 {
    (phase + PI).rem_euclid(2.0 * PI) - PI
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

        assert!(normalize_phase(100.0 * PI).abs() < 1e-4);
        assert!(normalize_phase(-100.0 * PI).abs() < 1e-4);
        assert!((normalize_phase(100.5 * PI) - (0.5 * PI)).abs() < 1e-4);
        assert!((normalize_phase(-100.5 * PI) - (-0.5 * PI)).abs() < 1e-4);

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
}
