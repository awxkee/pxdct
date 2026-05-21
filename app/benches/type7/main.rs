/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
use criterion::measurement::WallTime;
use criterion::{BenchmarkGroup, Criterion, criterion_group, criterion_main};
use pxdct::Pxdct;
use rand::RngExt;
use std::time::Duration;

fn check_power_group(c: &mut BenchmarkGroup<WallTime>, n: usize, group: String) {
    let mut input_power = vec![0f32; n];
    for z in input_power.iter_mut() {
        *z = rand::rng().random();
    }

    c.bench_function(format!("rustdct type7 {group}s").as_str(), |b| {
        let mut planner = rustdct::DctPlanner::new();
        let plan = planner.plan_dst7(input_power.len());
        let mut working = input_power.to_vec();
        let mut scratch = vec![0.; plan.get_scratch_len()];
        b.iter(|| {
            plan.process_dst7_with_scratch(&mut working, &mut scratch);
        })
    });

    c.bench_function(format!("pxdct type7 {group}s").as_str(), |b| {
        let plan = Pxdct::make_dst7_f32(input_power.len()).unwrap();
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

    check_power_group(c, 2, "2".to_string());
    check_power_group(c, 3, "3".to_string());
    check_power_group(c, 4, "4".to_string());
    check_power_group(c, 5, "5".to_string());
    check_power_group(c, 7, "7".to_string());
    check_power_group(c, 8, "8".to_string());
    check_power_group(c, 16, "16".to_string());
    check_power_group(c, 32, "32".to_string());

    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
