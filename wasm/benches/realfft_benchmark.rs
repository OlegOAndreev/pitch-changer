use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use realfft::RealFftPlanner;
use std::hint::black_box;

fn generate_input(size: usize) -> Vec<f32> {
    let mut input = vec![0.0f32; size];

    for (i, sample) in input.iter_mut().enumerate() {
        let x = i as f32;
        *sample = (x * 0.013).sin() + (x * 0.071).cos() * 0.5;
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

        group.bench_with_input(BenchmarkId::new("forward_fft", size), &size, |b, &_size| {
            b.iter(|| {
                in_buf.copy_from_slice(&input);
                r2c.process_with_scratch(&mut in_buf, &mut out_buf, &mut scratch_buf).unwrap();

                black_box(&out_buf);
            });
        });
    }

    group.finish();
}

#[cfg(feature = "pffft")]
fn bench_pffft_forward_sizes(c: &mut Criterion) {
    use realfft::num_complex::Complex;
    use wasm_main_module::PffftRealToComplex;

    let mut group = c.benchmark_group("pffft");

    for size in [1024usize, 2048, 4096] {
        let input = generate_input(size);

        let mut pffft = PffftRealToComplex::new(size).unwrap();
        let mut output_buf = vec![Complex::new(0.0, 0.0); size / 2 + 1];

        group.bench_with_input(BenchmarkId::new("forward_fft", size), &size, |b, &_size| {
            b.iter(|| {
                pffft.process(&input, &mut output_buf);
                black_box(&output_buf);
            });
        });
    }

    group.finish();
}

#[cfg(feature = "pffft")]
criterion_group!(benches, bench_realfft_forward_sizes, bench_pffft_forward_sizes);

#[cfg(not(feature = "pffft"))]
criterion_group!(benches, bench_realfft_forward_sizes);

criterion_main!(benches);
