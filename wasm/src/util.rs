use std::f32::consts::PI;

/// Normalize a phase value to the range [-PI, PI)
pub fn normalize_phase(phase: f32) -> f32 {
    (phase + PI).rem_euclid(2.0 * PI) - PI
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
}
