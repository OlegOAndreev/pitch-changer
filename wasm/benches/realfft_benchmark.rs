use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use realfft::RealFftPlanner;
use realfft::num_complex::Complex;
use rustfft::FftPlanner;
use std::hint::black_box;
use wasm_main_module::{FftComplexToReal, FftRealToComplex};

fn generate_input(size: usize) -> Vec<f32> {
    let mut input = vec![0.0f32; size];

    for (i, sample) in input.iter_mut().enumerate() {
        let x = i as f32;
        *sample = (x * 0.013).sin() + (x * 0.071).cos() * 0.5;
    }

    input
}

fn generate_complex_input(size: usize) -> Vec<Complex<f32>> {
    let mut input = vec![Complex::ZERO; size];

    for (i, sample) in input.iter_mut().enumerate() {
        let x = i as f32;
        sample.re = (x * 0.013).sin() + (x * 0.071).cos() * 0.5;
        sample.im = (x * 0.023).sin() + (x * 0.081).cos() * 0.5;
    }

    input
}

fn bench_realfft_forward_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("realfft");

    for size in [1024usize, 2048, 4096] {
        let mut planner = RealFftPlanner::<f32>::new();
        let r2c = planner.plan_fft_forward(size);

        let input = generate_input(size);
        let mut in_buf = input.clone();
        let mut out_buf = r2c.make_output_vec();
        let mut scratch_buf = r2c.make_scratch_vec();

        group.bench_with_input(BenchmarkId::new("forward_fft", size), &size, |b, _| {
            b.iter(|| {
                in_buf.copy_from_slice(&input);
                r2c.process_with_scratch(&mut in_buf, &mut out_buf, &mut scratch_buf).unwrap();

                black_box(&out_buf);
            });
        });
    }

    group.finish();
}

fn bench_rustfft_forward_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("rustfft");

    for size in [512usize, 1024, 2048] {
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(size);

        let input = generate_complex_input(size);
        let mut in_buf = input.clone();
        let mut out_buf = vec![Complex::ZERO; size];
        let mut scratch_buf = vec![Complex::ZERO; size];

        group.bench_with_input(BenchmarkId::new("forward_fft", size), &size, |b, _| {
            b.iter(|| {
                in_buf.copy_from_slice(&input);
                fft.process_immutable_with_scratch(&mut in_buf, &mut out_buf, &mut scratch_buf);

                black_box(&out_buf);
            });
        });
    }

    group.finish();
}

fn bench_real_fft_forward_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("real_fft");

    for size in [1024usize, 2048, 4096] {
        let input = generate_input(size);
        let mut in_buf = input.clone();

        let forward = FftRealToComplex::new(size).unwrap();
        let mut spectrum = vec![Complex::ZERO; size / 2 + 1];
        let mut scratch = vec![Complex::ZERO; size / 2];

        group.bench_with_input(BenchmarkId::new("forward_fft", size), &size, |b, _| {
            b.iter(|| {
                in_buf.copy_from_slice(&input);
                forward.process(&mut in_buf, &mut spectrum, &mut scratch).unwrap();

                black_box(&spectrum);
            });
        });
    }

    group.finish();
}

#[cfg(feature = "pffft")]
fn bench_pffft_real_forward_sizes(c: &mut Criterion) {
    use realfft::num_complex::Complex;
    use wasm_main_module::PffftRealToComplex;

    let mut group = c.benchmark_group("pffft_real");

    for size in [1024usize, 2048, 4096] {
        let input = generate_input(size);

        let mut pffft = PffftRealToComplex::new(size).unwrap();
        let mut output_buf = vec![Complex::new(0.0, 0.0); size / 2 + 1];

        group.bench_with_input(BenchmarkId::new("forward_fft", size), &size, |b, _| {
            b.iter(|| {
                pffft.process(&input, &mut output_buf);
                black_box(&output_buf);
            });
        });
    }

    group.finish();
}

#[cfg(feature = "pffft")]
fn bench_pffft_complex_forward_sizes(c: &mut Criterion) {
    use realfft::num_complex::Complex;
    use wasm_main_module::PffftComplex;

    let mut group = c.benchmark_group("pffft_complex");

    for size in [512usize, 1024, 2048] {
        let input = generate_complex_input(size);

        let mut pffft = PffftComplex::new(size).unwrap();
        let mut output_buf = vec![Complex::new(0.0, 0.0); size];

        group.bench_with_input(BenchmarkId::new("forward_fft", size), &size, |b, _| {
            b.iter(|| {
                pffft.forward(&input, &mut output_buf);
                black_box(&output_buf);
            });
        });
    }

    group.finish();
}

#[cfg(feature = "pffft")]
criterion_group!(
    benches,
    bench_realfft_forward_sizes,
    bench_real_fft_forward_sizes,
    bench_pffft_real_forward_sizes,
    bench_rustfft_forward_sizes,
    bench_pffft_complex_forward_sizes,
);

#[cfg(not(feature = "pffft"))]
criterion_group!(benches, bench_realfft_forward_sizes, bench_real_fft_forward_sizes, bench_rustfft_forward_sizes);

criterion_main!(benches);
