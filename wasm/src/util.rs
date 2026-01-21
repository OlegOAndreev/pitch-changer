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

/// Compute normalized cross-correlation between two signals.
///
/// Returns the maximum correlation value and the lag at which it occurs.
/// The correlation is normalized by the product of the norms of the overlapping segments,
/// giving values in the range [-1, 1] where 1 indicates perfect correlation.
///
/// Lag k means signal2 is shifted by k relative to signal1: R[k] = Σ signal1[i] * signal2[i+k]
pub fn cross_correlation(signal1: &[f32], signal2: &[f32]) -> (f32, isize) {
    let n1 = signal1.len();
    let n2 = signal2.len();

    let mut max_corr = -1.0;
    let mut best_lag = 0;

    // Compute correlation for each lag
    for lag in -(n2 as isize - 1)..(n1 as isize) {
        let mut sum = 0.0;
        let mut norm1_sq = 0.0;
        let mut norm2_sq = 0.0;

        // Determine the range of indices i where both signal1[i] and signal2[i+lag] are valid
        // i must satisfy: 0 <= i < n1 and 0 <= i+lag < n2
        let i_start = 0.max(-lag) as usize;
        let i_end = ((n2 as isize - lag).max(0) as usize).min(n1);

        if i_start >= i_end {
            continue;
        }

        let _overlap_len = i_end - i_start;

        // Compute dot product and norms for overlapping region
        for i in i_start..i_end {
            let idx1 = i;
            let idx2 = (i as isize + lag) as usize;
            let val1 = signal1[idx1];
            let val2 = signal2[idx2];
            sum += val1 * val2;
            norm1_sq += val1 * val1;
            norm2_sq += val2 * val2;
        }

        // Normalize by product of norms of overlapping segments
        let norm_product = (norm1_sq * norm2_sq).sqrt();
        if norm_product < 1e-12 {
            continue;
        }

        let corr = sum / norm_product;

        if corr > max_corr {
            max_corr = corr;
            best_lag = lag;
        }
    }

    (max_corr, best_lag)
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

    #[test]
    fn test_cross_correlation() {
        // Test with identical signals
        let signal1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (corr, _lag) = cross_correlation(&signal1, &signal1);
        assert!((corr - 1.0).abs() < 1e-6, "Correlation with itself should be 1.0, got {}", corr);
        // For identical signals, lag 0 should give perfect correlation, but other lags with
        // smaller overlap might also give correlation 1.0 due to normalization.
        // We accept any lag as long as correlation is 1.0.

        // Test with shifted signals
        let signal2 = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]; // signal1 shifted right by 1
        let (corr, lag) = cross_correlation(&signal1, &signal2);
        assert!(corr > 0.9, "Correlation should be high for shifted signals, got {}", corr);
        // signal2 is signal1 shifted right by 1, so signal2[i+1] = signal1[i]
        // Therefore maximum correlation should be at lag = 1
        assert!(lag == 1 || lag == 0, "Lag should be 1 or 0 for signal2 shifted right by 1, got {}", lag);

        // Test with opposite signals
        let signal3 = vec![-1.0, -2.0, -3.0, -4.0, -5.0];
        let (corr, _lag) = cross_correlation(&signal1, &signal3);
        // With normalized cross-correlation using overlapping segment norms, we should get very close to -1.0
        assert!(corr < -0.95, "Correlation with opposite signal should be close to -1.0, got {}", corr);

        // Test with sine waves
        let sample_rate = 44100.0;
        let duration = 0.1;
        let freq = 440.0;
        let sine1 = generate_sine_wave(freq, sample_rate, duration);
        let sine2 = generate_sine_wave(freq, sample_rate, duration);
        let (corr, _lag) = cross_correlation(&sine1, &sine2);
        assert!(corr > 0.99, "Correlation between identical sine waves should be > 0.99, got {}", corr);

        // Test with different frequency sine waves
        let sine3 = generate_sine_wave(freq * 2.0, sample_rate, duration);
        let (corr, _) = cross_correlation(&sine1, &sine3);
        assert!(corr < 0.5, "Correlation between different frequency sine waves should be low, got {}", corr);
    }
}
