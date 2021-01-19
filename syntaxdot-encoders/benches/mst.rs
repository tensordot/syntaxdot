use criterion::{criterion_group, criterion_main, Criterion};
use ndarray::Array;
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use rand_xorshift::XorShiftRng;

use syntaxdot_encoders::dependency::mst::chu_liu_edmonds;

fn mst_benchmark(c: &mut Criterion) {
    let mut rng = XorShiftRng::seed_from_u64(42);

    for &dim in &[5, 10, 20, 40, 80, 160] {
        let scores = Array::random_using((dim, dim), Uniform::new(0f32, 1f32), &mut rng);
        c.bench_function(&format!("mst-{}x{}", dim, dim), |b| {
            b.iter(|| chu_liu_edmonds(scores.view(), 0))
        });
    }
}

criterion_group!(mst_benches, mst_benchmark);
criterion_main!(mst_benches);
