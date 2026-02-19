use wasm_bindgen::prelude::*;

use std::f32::consts::PI;

/// Window for STFT.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[wasm_bindgen]
pub enum WindowType {
    Hann,
    SqrtHann,
    SqrtBlackman,
}

/// Generate a window of given type and size. The window is periodic: first element is zero, last element is non-zero
/// and equals the second element.
pub fn generate_window(window_type: WindowType, size: usize) -> Vec<f32> {
    let mut window = vec![0.0; size];

    match window_type {
        WindowType::Hann => {
            for (k, w) in window.iter_mut().enumerate() {
                *w = 0.5 * (1.0 - (2.0 * PI * k as f32 / size as f32).cos());
            }
        }
        WindowType::SqrtHann => {
            for (k, w) in window.iter_mut().enumerate() {
                let hann = 0.5 * (1.0 - (2.0 * PI * k as f32 / size as f32).cos());
                *w = hann.sqrt();
            }
        }
        WindowType::SqrtBlackman => {
            let n = size as f32;
            for (k, w) in window.iter_mut().enumerate() {
                let k_f32 = k as f32;
                let blackman = 0.42 - 0.5 * (2.0 * PI * k_f32 / n).cos() + 0.08 * (4.0 * PI * k_f32 / n).cos();
                // Ensure non-negative float
                *w = blackman.max(0.0).sqrt();
            }
        }
    }

    window
}

/// Generate the half window of given type and size, that is generate_tail_window(size) ==
/// generate_window(size * 2)[size..].
pub fn generate_tail_window(window_type: WindowType, size: usize) -> Vec<f32> {
    let mut window = vec![0.0; size];

    match window_type {
        WindowType::Hann => {
            for (k, w) in window.iter_mut().enumerate() {
                *w = 0.5 * (1.0 - (PI * (k + size) as f32 / size as f32).cos());
            }
        }
        WindowType::SqrtHann => {
            for (k, w) in window.iter_mut().enumerate() {
                let hann = 0.5 * (1.0 - (PI * (k + size) as f32 / size as f32).cos());
                *w = hann.sqrt();
            }
        }
        WindowType::SqrtBlackman => {
            let n = size as f32;
            for (k, w) in window.iter_mut().enumerate() {
                let k_f32 = (k + size) as f32;
                let blackman = 0.42 - 0.5 * (PI * k_f32 / n).cos() + 0.08 * (2.0 * PI * k_f32 / n).cos();
                // Ensure non-negative float
                *w = blackman.max(0.0).sqrt();
            }
        }
    }

    window
}

/// Calculate the sum of squared window values for a given hop size.
pub fn get_window_squared_sum(window_type: WindowType, size: usize, hop_size: usize) -> f32 {
    let overlap = size / hop_size;

    let per_overlap = match window_type {
        WindowType::Hann => 0.375,
        WindowType::SqrtHann => 0.5,
        WindowType::SqrtBlackman => 0.42,
    };

    per_overlap * overlap as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_window_bounds(window_type: WindowType) {
        for size in [256, 512, 1024] {
            let window = generate_window(window_type, size);
            assert_eq!(window[0], 0.0, "first element must be zero");
            assert!(window[size - 1] > 0.0, "last element must be non-zero");
            assert!(
                (window[1] - window[size - 1]).abs() < 1e-6,
                "second element ({}) must equal last element ({})",
                window[1],
                window[size - 1]
            );
        }
    }

    #[test]
    fn test_all_window_bounds() {
        test_window_bounds(WindowType::Hann);
        test_window_bounds(WindowType::SqrtHann);
        test_window_bounds(WindowType::SqrtBlackman);
    }

    fn test_squared_sum(window_type: WindowType) {
        for size in [256usize, 512, 1024] {
            for overlap in [4, 8, 16] {
                assert!(size.is_multiple_of(overlap), "size must be divisible by overlap");
                let window = generate_window(window_type, size);
                let hop_size = size / overlap;
                let expected = get_window_squared_sum(window_type, size, hop_size);

                // For each position within one hop, sum squared values across all overlapping windows
                for pos in 0..hop_size {
                    let mut sum = 0.0;
                    for i in 0..overlap {
                        let idx = i * hop_size + pos;
                        sum += window[idx] * window[idx];
                    }
                    assert!(
                        (sum - expected).abs() < 1e-5,
                        "window_type={:?}, size={}, overlap={}, pos={}: sum = {}, expected = {}",
                        window_type,
                        size,
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
    fn test_all_squared_sum() {
        test_squared_sum(WindowType::Hann);
        test_squared_sum(WindowType::SqrtHann);
        test_squared_sum(WindowType::SqrtBlackman);
    }

    fn test_last_half_window(window_type: WindowType) {
        for size in [256, 512, 1024] {
            let full_window = generate_window(window_type, size * 2);
            let half_window = generate_tail_window(window_type, size);
            for i in 0..size {
                let actual = half_window[i];
                let expected = full_window[i + size];
                assert!(
                    (actual - expected).abs() < 1e-6,
                    "window_type={:?}, size={}, k={}: actual={}, expected={}",
                    window_type,
                    size,
                    i,
                    actual,
                    expected
                );
            }
        }
    }

    #[test]
    fn test_all_last_half_window() {
        test_last_half_window(WindowType::Hann);
        test_last_half_window(WindowType::SqrtHann);
        test_last_half_window(WindowType::SqrtBlackman);
    }
}
