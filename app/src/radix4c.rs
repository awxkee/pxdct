/*
 * // Copyright (c) Radzivon Bartoshyk 5/2026. All rights reserved.
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

use std::f64::consts::{FRAC_1_SQRT_2, PI};

pub fn cos2r4(x: &[f64], m: usize) -> Vec<f64> {
    assert!(
        m >= 4 && m.is_power_of_two() && (m as f64).log2() as usize % 2 == 0,
        "m must be a power of 4 (4, 16, 64, …), got {}",
        m
    );
    assert_eq!(x.len(), m, "input length must equal m");

    if m == 4 {
        let h0 = x[0] + x[3];
        let h1 = x[1] + x[2];
        let h2 = x[0] - x[3];
        let h3 = x[1] - x[2];

        let w0 = 1.0 / (2.0 * (PI / 8.0).cos());
        let w1 = 1.0 / (2.0 * (3.0 * PI / 8.0).cos());

        let v0 = h0 + h1;
        let v1 = h0 - h1;
        let v2 = h2 * w0 + h3 * w1;
        let v3 = h2 * w0 - h3 * w1;

        let mut y = vec![0.0f64; 4];
        y[0] = v0;
        y[1] = v2 + v3 * FRAC_1_SQRT_2; // (v2·√2 + v3) / √2
        y[2] = v1 * FRAC_1_SQRT_2;
        y[3] = v3 * FRAC_1_SQRT_2;
        return y;
    }

    let m1 = m / 2;
    let m2 = m / 4; // quarter size = m1 / 2

    let mut buf = vec![0.0f64; m];
    for r in 0..m1 {
        let mirror = m - 1 - r;
        let sec = 1.0 / (2.0 * ((2 * r + 1) as f64 * PI / (2 * m) as f64).cos());
        buf[r] = x[r] + x[mirror];
        buf[m1 + r] = (x[r] - x[mirror]) * sec;
    }

    for b in 0..2usize {
        let off = b * m1;
        let block = buf[off..off + m1].to_vec();
        for r in 0..m2 {
            let mirror = m1 - 1 - r;
            let sec = 1.0 / (2.0 * ((2 * r + 1) as f64 * PI / (2 * m1) as f64).cos());
            buf[off + r] = block[r] + block[mirror];
            buf[off + m2 + r] = (block[r] - block[mirror]) * sec;
        }
    }

    let z1 = cos2r4(&buf[0..m2], m2);
    let z2 = cos2r4(&buf[m2..2 * m2], m2);
    let z3 = cos2r4(&buf[2 * m2..3 * m2], m2);
    let z4 = cos2r4(&buf[3 * m2..m], m2);
    let z = [z1, z2, z3, z4].concat();

    let mut tmp = vec![0.0f64; m];
    for b in 0..2usize {
        let off = b * m1;
        let b_off = off + m2;
        for j in 0..m2 {
            tmp[off + 2 * j] = z[off + j]; // even ← identity
            // b_mat(m2) with first-row coefficient = 1:
            let b_out = if j == 0 {
                z[b_off] + if m2 > 1 { z[b_off + 1] } else { 0.0 }
            } else if j < m2 - 1 {
                z[b_off + j] + z[b_off + j + 1]
            } else {
                z[b_off + j] // j == m2-1: last element unchanged
            };
            tmp[off + 2 * j + 1] = b_out; // odd ← b_mat
        }
    }

    let mut out = vec![0.0f64; m];
    for j in 0..m1 {
        out[2 * j] = tmp[j]; // even ← identity
        // b_mat(m1) with first-row coefficient = 1:
        let b_out = if j == 0 {
            tmp[m1] + if m1 > 1 { tmp[m1 + 1] } else { 0.0 }
        } else if j < m1 - 1 {
            tmp[m1 + j] + tmp[m1 + j + 1]
        } else {
            tmp[m1 + j] // j == m1-1
        };
        out[2 * j + 1] = b_out; // odd ← b_mat
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Unscaled DCT-II reference: Y[k] = Σ x[n]·cos(π·k·(2n+1)/2m)
    fn dct2_reference(x: &[f64]) -> Vec<f64> {
        let m = x.len();
        let fm = m as f64;
        (0..m)
            .map(|k| {
                x.iter()
                    .enumerate()
                    .map(|(n, &xn)| xn * (PI * k as f64 * (2 * n + 1) as f64 / (2.0 * fm)).cos())
                    .sum()
            })
            .collect()
    }

    fn allclose(a: &[f64], b: &[f64], tol: f64) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tol)
    }

    #[test]
    fn test_m4() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = cos2r4(&x, 4);
        let ref_ = dct2_reference(&x);
        assert!(allclose(&y, &ref_, 1e-10), "m=4\ngot: {y:?}\nref: {ref_:?}");
    }

    #[test]
    fn test_m16() {
        let x: Vec<f64> = (1..=16).map(|i| i as f64).collect();
        let y = cos2r4(&x, 16);
        let ref_ = dct2_reference(&x);
        assert!(allclose(&y, &ref_, 1e-9), "m=16\ngot: {y:?}\nref: {ref_:?}");
    }

    #[test]
    fn test_m64_random() {
        let mut s: u64 = 42;
        let x: Vec<f64> = (0..64)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (s >> 33) as f64 / u32::MAX as f64
            })
            .collect();
        assert!(allclose(&cos2r4(&x, 64), &dct2_reference(&x), 1e-8), "m=64");
    }

    #[test]
    fn test_m256_random() {
        let mut s: u64 = 12345;
        let x: Vec<f64> = (0..256)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (s >> 33) as f64 / u32::MAX as f64
            })
            .collect();
        assert!(
            allclose(&cos2r4(&x, 256), &dct2_reference(&x), 1e-7),
            "m=256"
        );
    }
}
