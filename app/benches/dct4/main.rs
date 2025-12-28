/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
use criterion::measurement::WallTime;
use criterion::{BenchmarkGroup, Criterion, criterion_group, criterion_main};
use pxdct::Pxdct;
use rand::Rng;
use std::time::Duration;

fn check_power_group(c: &mut BenchmarkGroup<WallTime>, n: usize, group: String) {
    let mut input_power = vec![0f32; n];
    for z in input_power.iter_mut() {
        *z = rand::rng().random();
    }

    c.bench_function(format!("rustdct dct4 {group}s").as_str(), |b| {
        let mut planner = rustdct::DctPlanner::new();
        let plan = planner.plan_dct4(input_power.len());
        let mut working = input_power.to_vec();
        b.iter(|| {
            plan.process_dct4(&mut working);
        })
    });

    c.bench_function(format!("pxdct dct4 {group}s").as_str(), |b| {
        let plan = Pxdct::make_dct4_f32(input_power.len()).unwrap();
        let mut working = input_power.to_vec();
        b.iter(|| {
            plan.execute(&mut working).unwrap();
        })
    });
}

pub fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("group");
    let c = group
        .measurement_time(Duration::from_millis(400))
        .warm_up_time(Duration::from_millis(400));

    check_power_group(c, 4, "4".to_string());
    check_power_group(c, 8, "8".to_string());
    check_power_group(c, 16, "16".to_string());
    check_power_group(c, 32, "32".to_string());
    check_power_group(c, 64, "64".to_string());
    check_power_group(c, 128, "128".to_string());
    check_power_group(c, 256, "256".to_string());
    check_power_group(c, 512, "512".to_string());
    check_power_group(c, 1024, "1024".to_string());

    check_power_group(c, 6, "6".to_string());
    check_power_group(c, 10, "10".to_string());
    check_power_group(c, 12, "12".to_string());
    check_power_group(c, 14, "14".to_string());
    check_power_group(c, 18, "18".to_string());
    check_power_group(c, 20, "20".to_string());
    check_power_group(c, 22, "22".to_string());
    check_power_group(c, 24, "24".to_string());
    check_power_group(c, 26, "26".to_string());
    check_power_group(c, 28, "28".to_string());
    check_power_group(c, 30, "30".to_string());

    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
