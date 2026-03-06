use std::f32::consts::{FRAC_PI_2, PI, TAU};

/// Get mutable references to two elements in a slice at different indices. Returns (&mut slice[idx1], &mut slice[idx2])
/// where idx1 != idx2.
pub fn get_disjoint_two_mut<T>(slice: &mut [T], idx1: usize, idx2: usize) -> (&mut T, &mut T) {
    assert!(idx1 != idx2, "indices must be different");

    if idx1 < idx2 {
        assert!(idx2 < slice.len(), "Index {} out of bounds {}", idx2, slice.len());
    } else {
        assert!(idx1 < slice.len(), "Index {} out of bounds {}", idx1, slice.len());
    }
    unsafe {
        let ptr = slice.as_mut_ptr();
        (ptr.add(idx1).as_mut().unwrap(), ptr.add(idx2).as_mut().unwrap())
    }
}

/// Normalize a phase value to the range [-PI, PI).
pub fn normalize_phase(phase: f32) -> f32 {
    let shifted = phase + PI;
    shifted - (shifted / TAU).floor() * TAU - PI
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

    // Fast-path for mono and stereo
    match num_channels {
        1 => {
            output.extend_from_slice(input);
        }
        2 => {
            let output_base = output.len();
            output.resize(output.len() + input.len(), 0.0);
            for sample_idx in 0..num_samples {
                unsafe {
                    *output.get_unchecked_mut(output_base + sample_idx) = *input.get_unchecked(sample_idx * 2);
                    *output.get_unchecked_mut(output_base + num_samples + sample_idx) =
                        *input.get_unchecked(sample_idx * 2 + 1);
                }
            }
        }
        _ => {
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

    match num_channels {
        1 => {
            output.extend_from_slice(input);
        }
        2 => {
            let output_base = output.len();
            output.resize(output.len() + input.len(), 0.0);
            for sample_idx in 0..num_samples {
                unsafe {
                    *output.get_unchecked_mut(output_base + sample_idx * num_channels) =
                        *input.get_unchecked(sample_idx);
                    *output.get_unchecked_mut(output_base + sample_idx * num_channels + 1) =
                        *input.get_unchecked(num_samples + sample_idx);
                };
            }
        }
        _ => {
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
    }
}

/// LinearSample allow sampling array with a fixed linear step.
pub struct LinearSample {
    pos: usize,
    fract: f32,

    step_i: usize,
    step_f: f32,
}

impl LinearSample {
    /// Initializes LinearSample at position 0.0 with given step.
    pub fn new(step: f32) -> Self {
        assert!(step >= 0.0);
        Self { pos: 0, fract: 0.0, step_i: step as usize, step_f: step.fract() }
    }

    /// Return the sample of data with current position.
    pub fn sample(&self, data: &[f32]) -> f32 {
        if self.pos >= data.len() - 1 {
            unsafe { *data.get_unchecked(data.len() - 1) }
        } else {
            #[cfg(test)]
            assert!(self.fract >= 0.0 && self.fract < 1.0, "Got fract {}", self.fract);
            unsafe {
                *data.get_unchecked(self.pos) * (1.0 - self.fract) + *data.get_unchecked(self.pos + 1) * self.fract
            }
        }
    }

    /// Increments the current sampling position.
    pub fn step(&mut self) {
        self.pos += self.step_i;
        self.fract += self.step_f;
        if self.fract >= 1.0 {
            self.pos += 1;
            self.fract -= 1.0;
        }
    }
}

/// Approximate atan2 implementation, taken from
/// https://math.stackexchange.com/questions/1098487/atan2-faster-approximation
pub fn approx_atan2(y: f32, x: f32) -> f32 {
    let ax = x.abs();
    let ay = y.abs();
    let max = ax.max(ay);
    if max == 0.0 {
        return 0.0;
    }
    let a = ax.min(ay) / max;
    let s = a * a;
    let mut r = ((-0.046496473 * s + 0.15931422) * s - 0.32762277) * s * a + a;
    if ay > ax {
        r = FRAC_PI_2 - r;
    }
    if x < 0.0 {
        r = PI - r;
    }
    if y < 0.0 {
        r = -r;
    }
    r
}

/// Approximate sin and cos implementation for cases when the angle is in [-PI, PI]. Taken from
/// https://www.apulsoft.ch/blog/branchless-sincos/
pub fn approx_sincos(x: f32) -> (f32, f32) {
    assert!(x <= PI + 1e-1, "{}", x);
    assert!(x >= -PI - 1e-1, "{}", x);
    const S0: f32 = -0.10132105; // x
    const S1: f32 = 0.0066206087; // x^3
    const S2: f32 = -0.00017335132; // x^5
    const S3: f32 = 2.4866881e-6; // x^7
    const S4: f32 = -1.9710331e-8; // x^9

    const C0: f32 = -0.4052844; // 1
    const C1: f32 = 0.038384996; // x^2
    const C2: f32 = -0.0013279879; // x^4
    const C3: f32 = 2.3744611e-5; // x^6
    const C4: f32 = -2.2398407e-7; // x^8

    let x2 = x * x;
    let x4 = x2 * x2;
    let x8 = x4 * x4;
    let poly1 = x8 * S4 + x4 * (S3 * x2 + S2) + S1 * x2 + S0;
    let poly2 = x8 * C4 + x4 * (C3 * x2 + C2) + C1 * x2 + C0;
    // NOTE: f32::mul_add is slower than plain multiply+add on WASM in Chrome. Revisit this code once FMA instruction
    // is available in WASM.
    // let poly1 = x8.mul_add(S4, x4.mul_add(S3.mul_add(x2, S2), S1.mul_add(x2, S0)));
    // let poly2 = x8.mul_add(C4, x4.mul_add(C3.mul_add(x2, C2), C1.mul_add(x2, C0)));

    let si = (x - PI) * (x + PI) * x * poly1;
    let co = (x - FRAC_PI_2) * (x + FRAC_PI_2) * poly2;
    (si, co)
}

/// Approximate log2 implementation for float32. Taken from
/// https://github.com/romeric/fastapprox/blob/master/fastapprox/src/fastlog.h
pub fn approx_log2(x: f32) -> f32 {
    let mx = f32::from_bits((x.to_bits() & 0x007FFFFF) | 0x3f000000);

    let y = (x.to_bits() as i32) as f32;
    y * 1.1920929e-7 - 124.22552 - 1.4980303 * mx - 1.72588 / (0.35208872 + mx)
}

/// Approximate exp2 implementation for float32. Taken from
/// https://github.com/romeric/fastapprox/blob/master/fastapprox/src/fastexp.h
pub fn approx_exp2(x: f32) -> f32 {
    let offset = if x < 0.0 { 1.0f32 } else { 0.0f32 };
    let clipp = if x < -126.0 { -126.0f32 } else { x };
    let z = clipp - clipp.trunc() + offset;
    let v = (1 << 23) as f32 * (clipp + 121.274055 + 27.728024 / (4.8425255 - z) - 1.4901291 * z);
    f32::from_bits(v as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_disjoint_two_mut() {
        // Test basic functionality with different indices
        let mut data = vec![1, 2, 3, 4, 5];

        // Test with idx1 < idx2
        let (a, b) = get_disjoint_two_mut(&mut data, 1, 3);
        *a = 20;
        *b = 40;
        assert_eq!(data, vec![1, 20, 3, 40, 5]);

        // Test with idx1 > idx2
        let (c, d) = get_disjoint_two_mut(&mut data, 4, 0);
        *c = 50;
        *d = 10;
        assert_eq!(data, vec![10, 20, 3, 40, 50]);

        // Test that modifications are independent
        let (e, f) = get_disjoint_two_mut(&mut data, 2, 3);
        assert_eq!(*e, 3);
        assert_eq!(*f, 40);
        *e = 30;
        *f = 400;
        assert_eq!(data, vec![10, 20, 30, 400, 50]);
    }

    #[test]
    #[should_panic(expected = "indices must be different")]
    fn test_get_disjoint_two_mut_same_indices() {
        let mut data = vec![1, 2, 3];
        get_disjoint_two_mut(&mut data, 1, 1);
    }

    #[test]
    #[should_panic(expected = "Index 5 out of bounds 5")]
    fn test_get_disjoint_two_mut_out_of_bounds_idx2() {
        let mut data = vec![1, 2, 3, 4, 5];
        get_disjoint_two_mut(&mut data, 1, 5);
    }

    #[test]
    #[should_panic(expected = "Index 5 out of bounds 5")]
    fn test_get_disjoint_two_mut_out_of_bounds_idx1() {
        let mut data = vec![1, 2, 3, 4, 5];
        get_disjoint_two_mut(&mut data, 5, 1);
    }

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

        let input_stereo = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        output.clear();
        deinterleave_samples(&input_stereo, 2, &mut output);
        assert_eq!(output, vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);

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

        let input_stereo = vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0];
        output.clear();
        interleave_samples(&input_stereo, 2, &mut output);
        assert_eq!(output, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

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
    fn test_approx_atan2() {
        let cases = [
            (0.0_f32, 1.0_f32),
            (0.0, -1.0),
            (1.0, 0.0),
            (-1.0, 0.0),
            (1.0, 1.0),
            (-1.0, 1.0),
            (1.0, -1.0),
            (-1.0, -1.0),
            (0.5, 0.866),
            (-0.866, -0.5),
            (0.0, 0.0),
        ];
        for (y, x) in cases {
            let expected = y.atan2(x);
            let got = approx_atan2(y, x);
            assert!((got - expected).abs() < 1e-3, "approx_atan2({y}, {x}) = {got}, expected {expected}");
        }
    }

    #[test]
    fn test_approx_sincos() {
        let cases =
            [0.0_f32, PI / 4.0, PI / 2.0, 3.0 * PI / 4.0, PI, -PI / 4.0, -PI / 2.0, -3.0 * PI / 4.0, -PI, 1.0, -1.0];
        for x in cases {
            let (si, co) = approx_sincos(x);
            let expected_si = x.sin();
            let expected_co = x.cos();
            assert!((si - expected_si).abs() < 1e-5, "approx_sincos({x}).0 = {si}, expected sin = {expected_si}");
            assert!((co - expected_co).abs() < 1e-5, "approx_sincos({x}).1 = {co}, expected cos = {expected_co}");
        }
    }

    #[test]
    fn test_approx_log2() {
        let cases = [1e-4, 1.5e-3, 3.5e-1, 1.23, 7.1e+2, 4.5e+3, 2.3e+4];
        for x in cases {
            let log2 = approx_log2(x);
            let expected_log2 = x.log2();
            assert!(
                (log2 - expected_log2).abs() < 1e-3,
                "approx_log2({x}).0 = {log2}, expected log2 = {expected_log2}"
            );
        }
    }

    #[test]
    fn test_approx_exp2() {
        let cases = [-1e-4, -1.5e-3, -3.5e-1, -1.23, -7.1e+2, -4.5e+3, -2.3e+4, 1e-4, 1.5e-3, 3.5e-1, 1.23];
        for x in cases {
            let exp2 = approx_exp2(x);
            let expected_exp2 = x.exp2();
            assert!(
                (exp2 - expected_exp2).abs() < 1e-4,
                "approx_exp2({x}).0 = {exp2}, expected exp2 = {expected_exp2}"
            );
        }
    }
}
