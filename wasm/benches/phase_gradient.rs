use criterion::{Criterion, criterion_group, criterion_main};
use rand::RngExt;
use realfft::RealFftPlanner;
use realfft::num_complex::Complex;
use std::f32::consts::PI;
use std::hint::black_box;

use wasm_main_module::PhaseGradientTimeStretch;

fn make_spectrum(input: &mut [f32]) -> Vec<Complex<f32>> {
    let fft_size = input.len();
    // Apply Hann window
    for i in 0..fft_size {
        let window_value = 0.5 * (1.0 - (2.0 * PI * i as f32 / (fft_size as f32 - 1.0)).cos());
        input[i] *= window_value;
    }

    let mut planner = RealFftPlanner::<f32>::new();
    let fft_plan = planner.plan_fft_forward(fft_size);
    let mut spectrum = fft_plan.make_output_vec();
    fft_plan.process(input, &mut spectrum).expect("FFT computation failed");

    spectrum
}

/// Generate FFT spectrum of random audio data
fn generate_random_fft_spectrum(fft_size: usize) -> Vec<Complex<f32>> {
    let mut rng = rand::rng();
    let mut input = vec![0.0f32; fft_size];
    for sample in &mut input {
        *sample = rng.random_range(-1.0..1.0);
    }

    make_spectrum(&mut input)
}

/// Generate FFT spectrum of 10 sine waves with different frequencies
fn generate_sine_waves_fft_spectrum(fft_size: usize) -> Vec<Complex<f32>> {
    let frequencies = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55];

    let mut input = vec![0.0f32; fft_size];
    for i in 0..fft_size {
        let mut sample = 0.0;
        for &freq_norm in &frequencies {
            let freq = 2.0 * PI * freq_norm;
            sample += (freq * i as f32).sin();
        }
        input[i] = sample;
    }

    // Normalize to [-1, 1]
    let max_magn = *input.iter().max_by(|a, b| a.total_cmp(b)).unwrap();
    for f in &mut input {
        *f /= max_magn;
    }

    make_spectrum(&mut input)
}

// PhaseGradientTimeStretch is the most time-consuming step of the whole pitch shifting pipeline (takes up to 80%)
fn benchmark_multiple_iterations(c: &mut Criterion) {
    let fft_size = 2048;
    let ana_hop_size = 8;
    let syn_hop_size = 16;

    // Test with multiple iterations to simulate continuous processing
    let mut time_stretcher = PhaseGradientTimeStretch::new(fft_size);
    let random_spectrum = generate_random_fft_spectrum(fft_size);
    let sine_spectrum = generate_sine_waves_fft_spectrum(fft_size);

    let mut group = c.benchmark_group("phase_gradient");
    group.bench_function("process_1000_iterations_sine", |b| {
        b.iter(|| {
            let mut output_spectrum = vec![Complex::new(0.0, 0.0); sine_spectrum.len()];
            for _ in 0..1000 {
                time_stretcher.process(&sine_spectrum, ana_hop_size, &mut output_spectrum, syn_hop_size);
            }
        })
    });

    group.bench_function("process_1000_iterations_random", |b| {
        b.iter(|| {
            let mut output_spectrum = vec![Complex::new(0.0, 0.0); random_spectrum.len()];
            for _ in 0..1000 {
                time_stretcher.process(&random_spectrum, ana_hop_size, &mut output_spectrum, syn_hop_size);
                let _ = black_box(&output_spectrum);
            }
        })
    });
}

criterion_group!(benches, benchmark_multiple_iterations);
criterion_main!(benches);
