use criterion::{Criterion, criterion_group, criterion_main};
use rand::RngExt;
use std::hint::black_box;

// Test sorting two-element struct

#[derive(PartialEq, PartialOrd, Clone, Copy, Debug)]
struct Pair {
    key: f32,
    value: usize,
}

impl Eq for Pair {}

impl Ord for Pair {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

// This is a copy of PhaseGradientBin
#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Debug)]
struct CompressedPair {
    repr: i64,
}

impl CompressedPair {
    fn new(key: f32, value: usize) -> Self {
        let repr = (key.to_bits() as i64) << 32 | value as i64;
        Self { repr }
    }

    fn value(&self) -> usize {
        (self.repr & 0xFFFFFFFF) as usize
    }

    fn key(&self) -> f32 {
        f32::from_bits((self.repr >> 32) as u32)
    }
}

fn generate_random_vec(size: usize) -> Vec<Pair> {
    let mut rng = rand::rng();
    let mut arr = Vec::with_capacity(size);
    for i in 0..size {
        let key = rng.random_range(0.0..1.0);
        arr.push(Pair { key, value: i });
    }
    arr
}

fn to_compressed_pair_vec(vec: &[Pair]) -> Vec<CompressedPair> {
    vec.iter()
        .map(|x| {
            assert!(x.key >= 0.0);
            CompressedPair::new(x.key, x.value)
        })
        .collect()
}

fn from_compressed_pair_vec(arr: &[CompressedPair]) -> Vec<Pair> {
    arr.iter().map(|x| Pair { key: x.key(), value: x.value() }).collect()
}

/// Test that both sorting methods produce the same results
fn test_sorting_equivalence(original: &[Pair]) {
    let mut sorted_pairs = original.to_vec();
    sorted_pairs.sort();

    let mut compressed_pairs = to_compressed_pair_vec(original);
    compressed_pairs.sort();
    let sorted_via_compressed = from_compressed_pair_vec(&compressed_pairs);

    assert_eq!(sorted_pairs.len(), sorted_via_compressed.len());
    for (a, b) in sorted_pairs.iter().zip(sorted_via_compressed.iter()) {
        assert_eq!(a, b);
    }

    for i in 1..sorted_pairs.len() {
        assert!(sorted_pairs[i - 1].key <= sorted_pairs[i].key);
        assert!(sorted_via_compressed[i - 1].key <= sorted_via_compressed[i].key);
    }
}

fn benchmark_sort_f32(c: &mut Criterion, num: usize) {
    let vec = generate_random_vec(num);
    let vec_compressed = to_compressed_pair_vec(&vec);
    test_sorting_equivalence(&vec);

    let mut scratch = vec![Pair { key: 0.0, value: 0 }; num];
    let mut scratch_compressed = vec![CompressedPair { repr: 0 }; num];

    let mut group = c.benchmark_group(format!("sort_{}", num));
    group.bench_function("pair", |b| {
        b.iter(|| {
            scratch.copy_from_slice(&vec);
            scratch.sort_unstable();
            black_box(&scratch);
        })
    });

    group.bench_function("compressed_pair", |b| {
        b.iter(|| {
            scratch_compressed.copy_from_slice(&vec_compressed);
            scratch_compressed.sort_unstable();
            black_box(&scratch_compressed);
        })
    });
}

fn benchmark_sort_10_elements(c: &mut Criterion) {
    benchmark_sort_f32(c, 10);
}

fn benchmark_sort_100_elements(c: &mut Criterion) {
    benchmark_sort_f32(c, 100);
}

fn benchmark_sort_1000_elements(c: &mut Criterion) {
    benchmark_sort_f32(c, 1000);
}

fn benchmark_sort_10000_elements(c: &mut Criterion) {
    benchmark_sort_f32(c, 10000);
}

criterion_group!(
    benches,
    benchmark_sort_10_elements,
    benchmark_sort_100_elements,
    benchmark_sort_1000_elements,
    benchmark_sort_10000_elements
);
criterion_main!(benches);
