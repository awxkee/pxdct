/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
use criterion::measurement::WallTime;
use criterion::{BatchSize, BenchmarkGroup, Criterion, criterion_group, criterion_main};
use pxdct::Pxdct;
use rand::RngExt;
use std::time::Duration;

pub(crate) fn prime_factors(mut n: u64) -> Vec<u64> {
    let mut res = Vec::new();
    if n < 2 {
        return res;
    }

    // factor out 2s
    while (n & 1) == 0 {
        res.push(2);
        n >>= 1;
    }

    // factor out 3s
    while n.is_multiple_of(3) {
        res.push(3);
        n /= 3;
    }

    // trial divide by 6k - 1 and 6k + 1
    let mut p: u64 = 5;
    while (p as u128) * (p as u128) <= n as u128 {
        while n.is_multiple_of(p) {
            res.push(p);
            n /= p;
        }
        let q = p + 2; // p = 6k-1, q = 6k+1
        while n.is_multiple_of(q) {
            res.push(q);
            n /= q;
        }
        p += 6;
    }

    // if remaining n > 1 it's prime
    if n > 1 {
        res.push(n);
    }
    res
}

pub(crate) fn prime_factorization(n: u64) -> Vec<(u64, u32)> {
    let factors = prime_factors(n);
    let mut out = Vec::new();
    let mut iter = factors.into_iter();
    if let Some(mut cur) = iter.next() {
        let mut cnt: u32 = 1;
        for f in iter {
            if f == cur {
                cnt += 1;
            } else {
                out.push((cur, cnt));
                cur = f;
                cnt = 1;
            }
        }
        out.push((cur, cnt));
    }
    out
}

pub fn bench_rustdct_averages_no_primes(c: &mut BenchmarkGroup<WallTime>, cap: usize) {
    c.bench_function(format!("rustdct dct2 no primes 1..={cap} float"), |b| {
        b.iter_batched(
            || {
                let mut plan = rustdct::DctPlanner::new();
                // Prepare all inputs and FFT plans
                let mut plans = Vec::new();
                for i in 1..cap {
                    let factors = prime_factorization(i as u64);
                    if factors
                        .iter()
                        .all(|x| x.0 <= 31 && (if x.0 >= 11 { x.1 == 1 } else { true }))
                    {
                        let input: Vec<f32> = (0..i).map(|i| i as f32).collect();
                        let fft = plan.plan_dct2(input.len());
                        plans.push((input, fft))
                    }
                }
                plans
            },
            |mut plans_and_inputs| {
                // Execute FFTs for all sizes
                for (input, fft) in plans_and_inputs.iter() {
                    let mut c = input.to_vec();
                    fft.process_dct2(&mut c);
                }
            },
            BatchSize::LargeInput,
        );
    });
}

pub fn bench_rustdct_averages(c: &mut BenchmarkGroup<WallTime>, cap: usize) {
    c.bench_function(format!("rustdct dct2 1..={cap} float"), |b| {
        b.iter_batched(
            || {
                let mut plan = rustdct::DctPlanner::new();
                // Prepare all inputs and FFT plans
                (1..=cap)
                    .map(|n| {
                        let input: Vec<f32> = (0..n).map(|i| i as f32).collect();
                        let fft = plan.plan_dct2(input.len());
                        (input, fft)
                    })
                    .collect::<Vec<_>>()
            },
            |mut plans_and_inputs| {
                // Execute FFTs for all sizes
                for (input, fft) in plans_and_inputs.iter() {
                    let mut c = input.to_vec();
                    fft.process_dct2(&mut c);
                }
            },
            BatchSize::LargeInput,
        );
    });
}

pub fn bench_pxdct_averages(c: &mut BenchmarkGroup<WallTime>, cap: usize) {
    c.bench_function(format!("pxdct dct2 1..={cap} float"), |b| {
        b.iter_batched(
            || {
                // Prepare all inputs and FFT plans
                (1..=cap)
                    .map(|n| {
                        let input: Vec<f32> = (0..n).map(|i| i as f32).collect();
                        let fft = Pxdct::make_dct2_f32(n).unwrap();
                        (input, fft)
                    })
                    .collect::<Vec<_>>()
            },
            |mut plans_and_inputs| {
                // Execute FFTs for all sizes
                for (input, fft) in plans_and_inputs.iter() {
                    let mut c = input.to_vec();
                    fft.execute(&mut c).unwrap();
                }
            },
            BatchSize::LargeInput,
        );
    });
}

pub fn bench_pxdct_averages_no_primes(c: &mut BenchmarkGroup<WallTime>, cap: usize) {
    c.bench_function(format!("pxdct dct2 no primes 1..={cap} float"), |b| {
        b.iter_batched(
            || {
                // Prepare all inputs and FFT plans
                let mut plans = Vec::new();
                for i in 1..cap {
                    let factors = prime_factorization(i as u64);
                    if factors
                        .iter()
                        .all(|x| x.0 <= 31 && (if x.0 >= 11 { x.1 == 1 } else { true }))
                    {
                        let input: Vec<f32> = (0..i).map(|i| i as f32).collect();
                        let fft = Pxdct::make_dct2_f32(i).unwrap();
                        plans.push((input, fft))
                    }
                }
                plans
            },
            |mut plans_and_inputs| {
                // Execute FFTs for all sizes
                for (input, fft) in plans_and_inputs.iter() {
                    let mut c = input.to_vec();
                    fft.execute(&mut c).unwrap();
                }
            },
            BatchSize::LargeInput,
        );
    });
}

fn check_power_group(c: &mut BenchmarkGroup<WallTime>, n: usize, group: String) {
    let mut input_power = vec![0f32; n];
    for z in input_power.iter_mut() {
        *z = rand::rng().random();
    }

    c.bench_function(format!("rustdct dct2 {group}s").as_str(), |b| {
        let mut planner = rustdct::DctPlanner::new();
        let plan = planner.plan_dct2(input_power.len());
        let mut working = input_power.to_vec();
        b.iter(|| {
            plan.process_dct2(&mut working);
        })
    });

    c.bench_function(format!("pxdct dct2 {group}s").as_str(), |b| {
        let plan = Pxdct::make_dct2_f32(input_power.len()).unwrap();
        let mut working = input_power.to_vec();
        let mut scratch = vec![0f32; plan.scratch_size()];
        b.iter(|| {
            plan.execute_with_scratch(&mut working, &mut scratch)
                .unwrap();
        })
    });

    let mut input_power = vec![0f64; n];
    for z in input_power.iter_mut() {
        *z = rand::rng().random();
    }

    c.bench_function(format!("rustdct dct2 {group}d").as_str(), |b| {
        let mut planner = rustdct::DctPlanner::new();
        let plan = planner.plan_dct2(input_power.len());
        let mut working = input_power.to_vec();
        b.iter(|| {
            plan.process_dct2(&mut working);
        })
    });

    c.bench_function(format!("pxdct dct2 {group}d").as_str(), |b| {
        let plan = Pxdct::make_dct2_f64(input_power.len()).unwrap();
        let mut working = input_power.to_vec();
        let mut scratch = vec![0.; plan.scratch_size()];
        b.iter(|| {
            plan.execute_with_scratch(&mut working, &mut scratch)
                .unwrap();
        })
    });
}

pub fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("group");
    let c = group
        .measurement_time(Duration::from_millis(400))
        .warm_up_time(Duration::from_millis(400));
    bench_rustdct_averages(c, 150);
    bench_pxdct_averages(c, 150);
    bench_rustdct_averages(c, 1800);
    bench_pxdct_averages(c, 1800);

    check_power_group(c, 4, "4".to_string());
    check_power_group(c, 8, "8".to_string());
    check_power_group(c, 16, "16".to_string());
    check_power_group(c, 32, "32".to_string());
    check_power_group(c, 64, "64".to_string());
    check_power_group(c, 128, "128".to_string());
    check_power_group(c, 256, "256".to_string());
    check_power_group(c, 512, "512".to_string());
    check_power_group(c, 1024, "1024".to_string());
    check_power_group(c, 2048, "2048".to_string());
    check_power_group(c, 4096, "4096".to_string());
    check_power_group(c, 8192, "8192".to_string());
    check_power_group(c, 16384, "16384".to_string());
    check_power_group(c, 32768, "32768".to_string());
    check_power_group(c, 65536, "65536".to_string());
    check_power_group(c, 131072, "131072".to_string());

    check_power_group(c, 1803, "1803".to_string());

    check_power_group(c, 169, "169".to_string());
    check_power_group(c, 2197, "2197".to_string());
    check_power_group(c, 28561, "28561".to_string());
    check_power_group(c, 371293, "371293".to_string());

    check_power_group(c, 121, "121".to_string());
    check_power_group(c, 1331, "1331".to_string());
    check_power_group(c, 14641, "14641".to_string());
    check_power_group(c, 161051, "161051".to_string());

    check_power_group(c, 7, "7".to_string());
    check_power_group(c, 49, "49".to_string());
    check_power_group(c, 343, "343".to_string());
    check_power_group(c, 2401, "2401".to_string());
    check_power_group(c, 16807, "16807".to_string());
    check_power_group(c, 117649, "117649".to_string());

    check_power_group(c, 9, "9".to_string());
    check_power_group(c, 27, "27".to_string());
    check_power_group(c, 81, "81".to_string());
    check_power_group(c, 243, "243".to_string());
    check_power_group(c, 729, "729".to_string());
    check_power_group(c, 2187, "2187".to_string());
    check_power_group(c, 6561, "6561".to_string());
    check_power_group(c, 19683, "19683".to_string());
    check_power_group(c, 59049, "59049".to_string());

    check_power_group(c, 5, "5".to_string());
    check_power_group(c, 25, "25".to_string());
    check_power_group(c, 125, "125".to_string());
    check_power_group(c, 3125, "3125".to_string());
    check_power_group(c, 15625, "15625".to_string());

    check_power_group(c, 6, "6".to_string());
    check_power_group(c, 216, "216".to_string());
    check_power_group(c, 1296, "1296".to_string());
    check_power_group(c, 6usize.pow(5), "6^5".to_string());
    check_power_group(c, 6usize.pow(7), "6^7".to_string());

    check_power_group(c, 10, "10".to_string());
    check_power_group(c, 12, "12".to_string());
    check_power_group(c, 14, "14".to_string());
    check_power_group(c, 15, "15".to_string());
    check_power_group(c, 18, "18".to_string());
    check_power_group(c, 20, "20".to_string());
    check_power_group(c, 21, "21".to_string());
    check_power_group(c, 24, "24".to_string());
    check_power_group(c, 35, "35".to_string());
    check_power_group(c, 30, "30".to_string());
    check_power_group(c, 36, "36".to_string());
    check_power_group(c, 48, "48".to_string());
    // check_power_group(c, 70, "70".to_string());

    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
