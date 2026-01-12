/*
 * // Copyright (c) Radzivon Bartoshyk 12/2025. All rights reserved.
 * //
 * // Redistribution and use in source and binary forms, with or without modification,
 * // are permitted provided that the following conditions are met:
 * //
 * // 1.  Redistributions of source code must retain the above copyright notice, this
 * // list of conditions and the following disclaimer.
 * //
 * // 2.  Redistributions in binary form must reproduce the above copyright notice,
 * // this list of conditions and the following disclaimer in the documentation
 * // and/or other materials provided with the distribution.
 * //
 * // 3.  Neither the name of the copyright holder nor the names of its
 * // contributors may be used to endorse or promote products derived from
 * // this software without specific prior written permission.
 * //
 * // THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * // AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * // IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * // DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * // FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * // DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * // SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * // CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * // OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * // OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
mod coprime_generator;
mod dct2_gen;
mod dct2_gen_fma;
mod dct2_gen_neon;
mod dct2_radixq_codegen;
mod dct3_gen;
mod dct4_butterfly_generator;
mod dct4_gen;
mod solver;

use criterion::Criterion;
use pxdct::PxdctExecutor;
use rand::Rng;

fn naive_dct4(input: &[f32]) -> Vec<f32> {
    let mut result = Vec::new();

    for output_index in 0..input.len() {
        let mut entry = 0.0;
        for input_index in 0..input.len() {
            let cos_inner =
                (output_index as f32 + 0.5) * (input_index as f32 + 0.5) * std::f32::consts::PI
                    / (input.len() as f32);
            let twiddle = cos_inner.cos();
            entry += input[input_index] * twiddle;
        }
        result.push(entry);
    }

    result
}

fn dct4_radix2_4pt(data: &mut [f32]) {
    // 1. Define Real-Valued Rotation Coefficients (for N=4, based on pi/16)
    let n_f64 = 8.0;
    // C1 = cos(pi/16), S1 = sin(pi/16)
    let naive = naive_dct4(data);
    println!("naive = {:?}", naive);

    let y0 = data[0];
    let y1 = data[1];
    let y2 = data[2];
    let y3 = data[3];
    let y4 = data[4];
    let y5 = data[5];
    let y6 = data[6];
    let y7 = data[7];

    // Step 2: C/S rotations
    let c1 = (1.0 * std::f32::consts::PI / (4.0 * n_f64)).cos();
    let s1 = (1.0 * std::f32::consts::PI / (4.0 * n_f64)).sin();
    let c3 = (3.0 * std::f32::consts::PI / (4.0 * n_f64)).cos();
    let s3 = (3.0 * std::f32::consts::PI / (4.0 * n_f64)).sin();
    let c5 = (5.0 * std::f32::consts::PI / (4.0 * n_f64)).cos();
    let s5 = (5.0 * std::f32::consts::PI / (4.0 * n_f64)).sin();
    let c7 = (7.0 * std::f32::consts::PI / (4.0 * n_f64)).cos();
    let s7 = (7.0 * std::f32::consts::PI / (4.0 * n_f64)).sin();

    let z0 = c1 * y0 + s1 * y7;
    let z1 = c3 * y1 + s3 * y6;
    let z2 = c5 * y2 + s5 * y5;
    let z3 = c7 * y3 + s7 * y4;
    let z4 = -s7 * y3 + c7 * y4;
    let z5 = s5 * y2 - c5 * y5;
    let z6 = -s3 * y1 + c3 * y6;
    let z7 = s1 * y0 - c1 * y7;
    // 4. Recursive Step: Two 2-point DCT-IIs
    let mut row0: [f32; 4] = [z0, z1, z2, z3];
    let mut row1: [f32; 4] = [z4, z5, z6, z7];

    let process16 = pxdct::Pxdct::make_dct2_f32(4).unwrap();
    process16.execute(&mut row0).unwrap(); // Computes X_II_0 and X_II_2
    process16.execute(&mut row1).unwrap(); // Computes X_II_1 and X_II_3

    let out0 = row0[0];
    let out1 = row0[1] - row1[3];
    let out2 = row0[1] + row1[3];
    let out3 = row0[2] + row1[2];
    let out4 = row0[2] - row1[2];
    let out5 = row0[3] - row1[1];
    let out6 = row1[1] + row0[3];
    let out7 = row1[0];

    data[0] = out0;
    data[1] = out1;
    data[2] = out2;
    data[3] = out3;
    data[5] = out5;
    data[4] = out4;
    data[6] = out6;
    data[7] = out7;
}

fn dct4_radix2_6pt(data: &mut [f32]) {
    // 1. Define Real-Valued Rotation Coefficients (for N=4, based on pi/16)
    let n_f64 = 6.0;
    // C1 = cos(pi/16), S1 = sin(pi/16)
    let naive = naive_dct4(data);
    println!("naive = {:?}", naive);

    let process16 = pxdct::Pxdct::make_dct4_f32(6).unwrap();
    process16.execute(data).unwrap(); // Computes X_II_0 and X_II_2

    println!("process16 = {:?}", data);
    println!("naive = {:?}", naive);
}

fn dct4_radix2_16pt(data: &mut [f32]) {
    // 1. Define Real-Valued Rotation Coefficients (for N=4, based on pi/16)
    let n_f64 = 16.0;
    // C1 = cos(pi/16), S1 = sin(pi/16)
    let naive = naive_dct4(data);
    println!("naive = {:?}", naive);

    let process16 = pxdct::Pxdct::make_dct4_f32(16).unwrap();
    process16.execute(data).unwrap();
    println!("{:?}, {:?}", data, naive); // Computes X_II_0 and X_II_2
}

fn main() {
    // let (dc3, lanes) = generate_dct2_fma(23, "fmla".to_string());
    // println!("{}", dc3);
    // let radixq_gen = Dct2RadixqGenerator::new(5, 1);
    // radixq_gen.execute();
    // gen_coprimes(5, 4);
    // for i in 1..150 {
    //     let mut long_bf36_6 = vec![0.; i];
    //     for z in long_bf36_6.iter_mut() {
    //         *z = rand::rng().random_range(1.0..2.0);
    //     }
    //
    //     let process16 = pxdct::Pxdct::make_dct2_f32(long_bf36_6.len()).unwrap();
    //     process16.execute(&mut long_bf36_6).unwrap();
    //     println!("finished {i}");
    // }

    // let bf = generate_butterfly_dct4(30);
    // println!("{}", bf);

    let mut short_bf45 = vec![5f64, 2., 9., 4., 1., 8., 1., 8.];
    for (i, z) in short_bf45.iter_mut().enumerate() {
        *z = i as f64; // rand::rng().random_range(1.0..2.0);
    }
    // dct4_radix2_6pt(&mut short_bf45);
    let process16 = pxdct::Pxdct::make_dst2_f64(8).unwrap();
    // [190.0, -80.973434, 0.0, -8.921358, 3.5762787e-7, -3.1543207, 0.0, -1.5615854, 8.34465e-7, -0.9014206, 0.0, -0.5615945, 9.536743e-7, -4.6403565, 0.0, -0.22416973, -1.1920929e-7, 5.5045886, -14.557901, -15.471075]
    process16.execute(&mut short_bf45).unwrap();
    println!("{:?}", short_bf45);

    // let process16 = pxdct::Pxdct::make_dct2_f32(short_bf12.len()).unwrap();
    //     process16.execute(&mut short_bf12).unwrap();

    // let mut short_bf36 = vec![0.; 36];
    // for z in short_bf12.iter_mut() {
    //     *z = rand::rng().random_range(1.0..2.0);
    // }

    let mut c = Criterion::default();
    c.bench_function("length 169", |r| {
        let mut short_bf45 = vec![0.; 169];
        for z in short_bf45.iter_mut() {
            *z = rand::rng().random_range(1.0..2.0);
        }
        let process16 = pxdct::Pxdct::make_dct2_f32(169).unwrap();
        r.iter(|| {
            process16.execute(&mut short_bf45).unwrap();
        });
    });
    c.bench_function("length 2197", |r| {
        let mut short_bf45 = vec![0.; 2197];
        for z in short_bf45.iter_mut() {
            *z = rand::rng().random_range(1.0..2.0);
        }
        let process16 = pxdct::Pxdct::make_dct2_f64(2197).unwrap();
        r.iter(|| {
            process16.execute(&mut short_bf45).unwrap();
        });
    });
    c.bench_function("length 28561", |r| {
        let mut short_bf45 = vec![0.; 28561];
        for z in short_bf45.iter_mut() {
            *z = rand::rng().random_range(1.0..2.0);
        }
        let process16 = pxdct::Pxdct::make_dct2_f64(28561).unwrap();
        r.iter(|| {
            process16.execute(&mut short_bf45).unwrap();
        });
    });
    //
    // c.bench_function("length 256", |r| {
    //     let mut short_bf45 = vec![0.; 256];
    //     for z in short_bf45.iter_mut() {
    //         *z = rand::rng().random_range(1.0..2.0);
    //     }
    //     let process16 = pxdct::Pxdct::make_dct2_f64(256).unwrap();
    //     r.iter(|| {
    //         process16.execute(&mut short_bf45).unwrap();
    //     });
    // });
    // c.bench_function("length 512", |r| {
    //     let mut short_bf45 = vec![0.; 512];
    //     for z in short_bf45.iter_mut() {
    //         *z = rand::rng().random_range(1.0..2.0);
    //     }
    //     let process16 = pxdct::Pxdct::make_dct2_f64(512).unwrap();
    //     r.iter(|| {
    //         process16.execute(&mut short_bf45).unwrap();
    //     });
    // });
    // c.bench_function("length 1024", |r| {
    //     let mut short_bf45 = vec![0.; 1024];
    //     for z in short_bf45.iter_mut() {
    //         *z = rand::rng().random_range(1.0..2.0);
    //     }
    //     let process16 = pxdct::Pxdct::make_dct2_f64(1024).unwrap();
    //     r.iter(|| {
    //         process16.execute(&mut short_bf45).unwrap();
    //     });
    // });
    // c.bench_function("length 2048", |r| {
    //     let mut short_bf45 = vec![0.; 2048];
    //     for z in short_bf45.iter_mut() {
    //         *z = rand::rng().random_range(1.0..2.0);
    //     }
    //     let process16 = pxdct::Pxdct::make_dct2_f64(2048).unwrap();
    //     r.iter(|| {
    //         process16.execute(&mut short_bf45).unwrap();
    //     });
    // });
    // c.bench_function("mixed radix 9*12", |r| {
    //     let process16 = pxdct::Pxdct::make_dct2_f32(short_bf12.len()).unwrap();
    //     r.iter(|| {
    //         process16.execute(&mut short_bf12).unwrap();
    //     });
    // });
    // c.bench_function("power 6, 6", |r| {
    //     let process16 = pxdct::Pxdct::make_dct2_f32(short_bf12.len()).unwrap();
    //     r.iter(|| {
    //         process16.execute(&mut short_bf12).unwrap();
    //     });
    // });
    // c.bench_function("power 3, 81", |r| {
    //     let process16 = pxdct::Pxdct::make_dct2_f32(short_bf81.len()).unwrap();
    //     r.iter(|| {
    //         process16.execute(&mut short_bf81).unwrap();
    //     });
    // });
}
