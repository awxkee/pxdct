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
mod dct3_coprime;
mod dct3_gen;
mod dct4_butterfly_generator;
mod dct4_gen;
mod solver;
mod split_radix_dct3;

use crate::dct3_coprime::gen_dct3_coprimes;
use crate::dct3_gen::{compute_twiddle, generate_dct3};
use crate::solver::solve_expression_arr;
use crate::split_radix_dct3::split_radix_dct3;
use criterion::Criterion;
use num_complex::Complex;
use pxdct::Pxdct;
use pxfm::{f_cospi, f_sec, f_sincospi, f_sinpi};
use rustdct::num_traits::FloatConst;
use std::f64::consts::PI;
use std::fmt::format;

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

#[inline(always)]
pub fn cos3_2(x: [f64; 2]) -> [f64; 2] {
    let half_0 = x[0] * 0.5;
    let frac_1 = x[1] * f64::FRAC_1_SQRT_2();
    let v0 = half_0 + frac_1;
    let v1 = half_0 - frac_1;
    [v0, v1]
}

#[inline(always)]
pub fn cos4_2(x: [f64; 2]) -> [f64; 2] {
    let half_0 = x[0] * 0.5;
    let frac_1 = x[1] * f64::FRAC_1_SQRT_2();
    let v0 = half_0 + frac_1;
    let v1 = half_0 - frac_1;
    [v0, v1]
}

#[inline(always)]
pub fn cos3_4(x: [f64; 4]) -> [f64; 4] {
    let twiddle = compute_twiddle(1, 16).conj();

    let u0 = x[0];
    let u1 = x[1];
    let u2 = x[2] * 2.0;
    let u3 = x[3];

    let [z1_0, z1_1] = cos3_2([(u0 + u2), u0 - u2]);
    let [z2_0, z2_1] = cos3_2([(u1 + u3), u1 - u3]);

    let lw = f64::mul_add(z2_0, twiddle.re, z2_1 * twiddle.im);
    let uw = f64::mul_add(z2_0, twiddle.im, -z2_1 * twiddle.re);

    [z1_0, z1_1 + uw, z2_0 + lw, z1_0 + uw]
}

#[inline(always)]
pub fn idct5(input: [f64; 5]) -> [f64; 5] {
    // Cosine constants
    let c1 = (PI / 5.0).cos(); // ≈ 0.809017
    let c2 = (2.0 * PI / 5.0).cos(); // ≈ 0.309017
    let s1 = (PI / 5.0).sin(); // ≈ 0.587785
    let s2 = (2.0 * PI / 5.0).sin(); // ≈ 0.951057
    let a0 = input[0];
    let a1 = input[1] + input[4];
    let a2 = input[2] + input[3];
    let b1 = input[1] - input[4];
    let b2 = input[2] - input[3];

    // Stage 2: Intermediate computation
    let t1 = c1 * a1 + c2 * a2;
    let t2 = c2 * a1 - c1 * a2;
    let u1 = s1 * b1 + s2 * b2;
    let u2 = s2 * b1 - s1 * b2;

    // Stage 3: Final outputs
    [
        a0 + a1 + a2,
        a0 + t1 + u1,
        a0 + t2 + u2,
        a0 + t2 - u2,
        a0 + t1 - u1,
    ]
}

#[inline(always)]
pub fn cos2_6(x: [f64; 6]) -> [f64; 6] {
    let u0 = x[0] + x[5];
    let u1 = x[1] + x[4];
    let u2 = x[2] + x[3];
    let u3 = x[2] - x[3];
    let u4 = x[1] - x[4];
    let u5 = x[0] - x[5];
    let internal = Pxdct::make_dct2_f64(3).unwrap();
    let internal4 = Pxdct::make_dct4_f64(3).unwrap();
    let mut z1 = [u0, u1, u2];
    let mut z2 = [u5, u4, u3];
    internal.execute(&mut z1).unwrap();
    internal4.execute(&mut z2).unwrap();
    let mut output: [f64; 6] = [0.0; 6];
    for i in 0..3 {
        output[i * 2] = z1[i];
        output[i * 2 + 1] = z2[i];
    }
    output
}

#[inline(never)]
pub fn cos3_6(x: [f64; 6]) -> [f64; 6] {
    let internal = Pxdct::make_dct3_f64(3).unwrap();
    let internal4 = Pxdct::make_dct4_f64(3).unwrap();
    let mut z1 = [x[0], x[2], x[4]];
    let mut z2 = [x[1], x[3], x[5]];
    internal.execute(&mut z1).unwrap();
    internal4.execute(&mut z2).unwrap();
    let mut output: [f64; 6] = [0.0; 6];
    for i in 0..3 {
        output[i] = z1[i] + z2[i];
        output[6 - i - 1] = z1[i] - z2[i];
    }
    output
}

#[inline]
pub(crate) fn compute_twiddle_s(index: usize, fft_len: usize) -> f64 {
    let angle = index as f64 / (2. * fft_len as f64) * std::f64::consts::PI;
    f_sec(angle) * 0.5
}

#[inline(always)]
pub fn cos2_8(x: [f64; 8]) -> [f64; 8] {
    let tw0 = compute_twiddle_s(1, 8);
    let tw1 = compute_twiddle_s(3, 8);
    let tw2 = compute_twiddle_s(5, 8);
    let tw3 = compute_twiddle_s(7, 8);

    let u0 = x[0] + x[7];
    let u1 = x[1] + x[6];
    let u2 = x[2] + x[5];
    let u3 = x[3] + x[4];

    let u4 = x[3] - x[4];
    let u5 = x[2] - x[5];
    let u6 = x[1] - x[6];
    let u7 = x[0] - x[7];
    let internal = Pxdct::make_dct2_f64(4).unwrap();
    let internal4 = Pxdct::make_dct4_f64(4).unwrap();
    let mut z1 = [u0, u1, u2, u3];
    let mut z2 = [u7, u6, u5, u4];
    internal.execute(&mut z1).unwrap();
    internal4.execute(&mut z2).unwrap();
    let mut output: [f64; 8] = [0.0; 8];
    for i in 0..4 {
        output[i * 2] = z1[i];
        output[i * 2 + 1] = z2[i];
    }
    output
}

#[inline(always)]
pub fn cos2_4(x: [f64; 4]) -> [f64; 4] {
    let tw0 = compute_twiddle(1, 4 * 4) * 0.5;
    let tw1 = compute_twiddle(3, 4 * 4) * 0.5;

    let u0 = x[0] + x[3];
    let u1 = x[1] + x[2];
    let mut u2 = x[1] - x[2];
    let mut u3 = x[0] - x[3];

    let v1 = -u2 * tw0.re + u3 * tw0.im;
    let v2 = u2 * tw1.im - u3 * tw1.re;

    let internal = Pxdct::make_dct2_f64(2).unwrap();
    let internal4 = Pxdct::make_dct2_f64(2).unwrap();

    let mut z1 = [u0, u1];
    let mut z2 = [v2, v1];
    internal.execute(&mut z1).unwrap();
    internal4.execute(&mut z2).unwrap();
    let mut output: [f64; 4] = [0.0; 4];
    for i in 0..2 {
        output[i * 2] = z1[i];
        output[i * 2 + 1] = z2[i];
    }
    output
}

fn main() {
    let mut bf = [1.1, 2.1, 3.1, 4.1];
    let received = cos2_4(bf);
    let cvt = Pxdct::make_dct2_f64(4).unwrap();
    cvt.execute(&mut bf).unwrap();
    println!("received {:?}", received);
    println!("converted {:?}", bf);

    // let (dc3, lanes) = generate_dct2_fma(23, "fmla".to_string());
    // println!("{}", dc3);
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

    // let mut short_bf45 = vec![5f64, 2., 9., 4.];
    // for (i, z) in short_bf45.iter_mut().enumerate() {
    //     *z = i as f64; // rand::rng().random_range(1.0..2.0);
    // }
    // let cos3_4 = cos3_4(short_bf45.to_vec().try_into().unwrap());
    // println!("cos3_4 = {:?}", cos3_4);
    // let process16 = pxdct::Pxdct::make_dct3_f64(4).unwrap();
    // // [190.0, -80.973434, 0.0, -8.921358, 3.5762787e-7, -3.1543207, 0.0, -1.5615854, 8.34465e-7, -0.9014206, 0.0, -0.5615945, 9.536743e-7, -4.6403565, 0.0, -0.22416973, -1.1920929e-7, 5.5045886, -14.557901, -15.471075]
    // process16.execute(&mut short_bf45).unwrap();
    // println!("{:?}", short_bf45);

    // let process16 = pxdct::Pxdct::make_dct2_f32(short_bf12.len()).unwrap();
    //     process16.execute(&mut short_bf12).unwrap();

    // let mut short_bf36 = vec![0.; 36];
    // for z in short_bf12.iter_mut() {
    //     *z = rand::rng().random_range(1.0..2.0);
    // }

    // let mut c = Criterion::default();
    // c.bench_function("length 169", |r| {
    //     let mut short_bf45 = vec![0.; 169];
    //     for z in short_bf45.iter_mut() {
    //         *z = rand::rng().random_range(1.0..2.0);
    //     }
    //     let process16 = pxdct::Pxdct::make_dct2_f32(169).unwrap();
    //     r.iter(|| {
    //         process16.execute(&mut short_bf45).unwrap();
    //     });
    // });
    // c.bench_function("length 2197", |r| {
    //     let mut short_bf45 = vec![0.; 2197];
    //     for z in short_bf45.iter_mut() {
    //         *z = rand::rng().random_range(1.0..2.0);
    //     }
    //     let process16 = pxdct::Pxdct::make_dct2_f64(2197).unwrap();
    //     r.iter(|| {
    //         process16.execute(&mut short_bf45).unwrap();
    //     });
    // });
    // c.bench_function("length 28561", |r| {
    //     let mut short_bf45 = vec![0.; 28561];
    //     for z in short_bf45.iter_mut() {
    //         *z = rand::rng().random_range(1.0..2.0);
    //     }
    //     let process16 = pxdct::Pxdct::make_dct2_f64(28561).unwrap();
    //     r.iter(|| {
    //         process16.execute(&mut short_bf45).unwrap();
    //     });
    // });
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
