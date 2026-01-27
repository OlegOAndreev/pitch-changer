use std::f32::consts::PI;

/// Sum of squared Hann window values per overlap offset (overlap must be >= 4).
pub const HANN_WINDOW_SQ_SUM_PER_OVERLAP: f32 = 0.375;

/// Normalize a phase value to the range [-π, π].
pub fn normalize_phase(mut phase: f32) -> f32 {
    let two_pi = 2.0 * PI;
    // This is efficient when the phase is not too far from the [-π, π] range
    if phase < -PI {
        while phase < -PI {
            phase += two_pi;
        }
    } else if phase > PI {
        while phase > PI {
            phase -= two_pi;
        }
    }
    phase
}

/// Fill a Hann window.
pub fn fill_hann_window(window: &mut [f32]) {
    let size = window.len() as f32;
    for (k, w) in window.iter_mut().enumerate() {
        *w = -0.5 * (2.0 * PI * k as f32 / size).cos() + 0.5;
    }
}

/// Generate a sine wave at the given frequency, sample rate, and duration.
pub fn generate_sine_wave(freq: f32, sample_rate: f32, duration: f32) -> Vec<f32> {
    let num_samples = (sample_rate * duration) as usize;
    let step = 2.0 * PI * freq / sample_rate;
    let mut result = Vec::with_capacity(num_samples);
    let mut phase = 0.0f32;
    for _ in 0..num_samples {
        result.push(phase.sin());
        phase = normalize_phase(phase + step);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_phase() {
        // Values already in range
        assert_eq!(normalize_phase(0.0), 0.0);
        assert_eq!(normalize_phase(PI), PI);
        assert_eq!(normalize_phase(-PI), -PI);
        assert_eq!(normalize_phase(PI / 2.0), PI / 2.0);
        assert_eq!(normalize_phase(-PI / 3.0), -PI / 3.0);

        assert!((normalize_phase(3.0 * PI) - PI).abs() < 1e-6);
        assert!((normalize_phase(2.5 * PI) - (0.5 * PI)).abs() < 1e-6);
        assert!((normalize_phase(4.0 * PI) - 0.0).abs() < 1e-6);
        assert!((normalize_phase(7.0 * PI) - PI).abs() < 1e-6);
        assert!((normalize_phase(-3.0 * PI) - (-PI)).abs() < 1e-6);
        assert!((normalize_phase(-2.5 * PI) - (-0.5 * PI)).abs() < 1e-6);
        assert!((normalize_phase(-4.0 * PI) - 0.0).abs() < 1e-6);
        assert!((normalize_phase(-7.0 * PI) - (-PI)).abs() < 1e-6);

        assert!(normalize_phase(100.0 * PI).abs() < 1e-4);
        assert!(normalize_phase(-100.0 * PI).abs() < 1e-4);
        assert!((normalize_phase(100.5 * PI) - (0.5 * PI)).abs() < 1e-4);
        assert!((normalize_phase(-100.5 * PI) - (-0.5 * PI)).abs() < 1e-4);

        let epsilon = 1e-6;
        assert!((normalize_phase(PI + epsilon) - (-PI + epsilon)).abs() < 1e-5);
        assert!((normalize_phase(-PI - epsilon) - (PI - epsilon)).abs() < 1e-5);
    }

    #[test]
    fn test_window_bounds() {
        let mut window = vec![0.0; 256];
        fill_hann_window(&mut window);
        // Do basic test from https://www.cs.princeton.edu/courses/archive/spr09/cos325/Bernardini.pdf
        //
        // In an STFT, it is important to ensure that the periodicity of the framing window is correct: the periodicity
        // of the framing window should be equal to the declared argument of its function definition. A hanning window,
        // for example, must begin by a zero-valued sample and end by a non-zero valued sample (whose value must be the
        // same as the second sample)
        assert_eq!(window[1], window[window.len() - 1]);
    }

    #[test]
    fn test_hann_squared_window_sum() {
        for window_size in [256, 512, 1024, 2048] {
            let mut window = vec![0.0; window_size];
            fill_hann_window(&mut window);
            for overlap in [4, 8, 16, 32] {
                assert_eq!(window_size % overlap, 0, "window_size must be divisible by overlap");
                let hop_size = window_size / overlap;
                let expected = super::HANN_WINDOW_SQ_SUM_PER_OVERLAP * overlap as f32;

                for pos in 0..hop_size {
                    let mut sum = 0.0;
                    for k in 0..overlap {
                        let idx = k * hop_size + pos;
                        sum += window[idx] * window[idx];
                    }
                    assert!(
                        (sum - expected).abs() < 1e-5,
                        "n={}, r={}, pos={}: sum = {}, expected = {}",
                        window_size,
                        overlap,
                        pos,
                        sum,
                        expected
                    );
                }
            }
        }
    }
}
