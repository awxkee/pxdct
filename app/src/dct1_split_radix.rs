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

use pxdct::Pxdct;
use std::f64::consts::FRAC_1_SQRT_2;

/// Computes the Split-Radix DCT-I for an input of length n + 1
pub fn split_radix_dct1(x: &mut [f64]) {
    let len = x.len();

    // Base case: No scaling factor here
    if len <= 2 {
        if len == 2 {
            let (a, b) = (x[0], x[1]);
            x[0] = a + b;
            x[1] = a - b;
        }
        return;
    }

    let n = len - 1;
    assert!(n % 2 == 0, "n must be an even integer");
    let n1 = n / 2;

    let mut v = vec![0.0; len];

    // v[n1] = x[n1];
    v[n1] = 2.0 * x[n1];

    for i in 0..n1 {
        v[i] = x[i] + x[n - i]; // no 1/√2
        v[n1 + 1 + i] = x[i] - x[n - i]; // no 1/√2
    }

    split_radix_dct1(&mut v[0..=n1]);
    dct3_stub(&mut v[n1 + 1..len]);

    for i in 0..=n1 {
        x[2 * i] = v[i];
    }
    for i in 0..n1 {
        x[2 * i + 1] = v[n1 + 1 + i] * 2.0;
    }
}

/// Placeholder for the DCT-III execution
fn dct3_stub(_x: &mut [f64]) {
    let dct3 = Pxdct::make_dct3_f64(_x.len()).unwrap();
    dct3.execute(_x).unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::FRAC_1_SQRT_2;

    /// Helper function to compare floating-point slices with a tolerance
    fn assert_slice_close(actual: &[f64], expected: &[f64], epsilon: f64) {
        assert_eq!(actual.len(), expected.len(), "Slice lengths differ");
        for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() < epsilon,
                "Mismatch at index {}: actual {}, expected {}",
                i,
                a,
                e
            );
        }
    }

    #[test]
    fn test_base_case_len_2() {
        // For length 2, the algorithm applies a simple scaled sum/difference
        let mut x = [3.0, 1.0];
        split_radix_dct1(&mut x);

        let expected_0 = (3.0 + 1.0) * FRAC_1_SQRT_2; // 4 * ~0.707 = 2.8284
        let expected_1 = (3.0 - 1.0) * FRAC_1_SQRT_2; // 2 * ~0.707 = 1.4142

        assert_slice_close(&x, &[expected_0, expected_1], 1e-6);
    }

    #[test]
    fn test_base_case_len_1() {
        // Length 1 should return unmodified
        let mut x = [42.0];
        split_radix_dct1(&mut x);
        assert_eq!(x[0], 42.0);
    }

    fn naive_dct1(input: &[f64]) -> Vec<f64> {
        let n = input.len() - 1; // N = len - 1
        let mut output = vec![0.0; input.len()];
        for k in 0..=n {
            let mut sum = input[0] + (if k % 2 == 0 { 1.0 } else { -1.0 }) * input[n];
            for j in 1..n {
                sum += 2.0
                    * input[j]
                    * (std::f64::consts::PI * (k as f64) * (j as f64) / (n as f64)).cos();
            }
            output[k] = sum;
        }
        output
    }

    fn naive_dct1_orth(input: &[f64]) -> Vec<f64> {
        let n = input.len();
        let mut output = vec![0.0; n];
        let inv_sqrt_2 = 1.0 / 2.0f64.sqrt();

        for k in 0..n {
            let mut sum = input[0] * inv_sqrt_2
                + (if (k % 2) == 0 { 1.0 } else { -1.0 }) * input[n - 1] * inv_sqrt_2;
            for n_idx in 1..n - 1 {
                sum += input[n_idx]
                    * (std::f64::consts::PI * (k as f64) * (n_idx as f64) / ((n - 1) as f64)).cos();
            }
            output[k] = sum * (2.0 / (n - 1) as f64).sqrt(); // Usually orthogonalized by sqrt(2/(N-1))
        }
        output
    }

    pub fn reference_dct1(input: &[f64]) -> Vec<f64> {
        let mut result = Vec::new();

        for output_index in 0..input.len() {
            let mut entry = 0.0;
            for input_index in 0..input.len() {
                let multiplier = if input_index == 0 || input_index == input.len() - 1 {
                    0.5
                } else {
                    1.0
                };
                let cos_inner = (output_index as f64) * (input_index as f64) * std::f64::consts::PI
                    / ((input.len() - 1) as f64);
                let twiddle = cos_inner.cos();
                entry += input[input_index] * twiddle * multiplier;
            }
            result.push(entry);
        }
        result
    }

    #[test]
    fn test_dct1_size_5() {
        let mut x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let expected = naive_dct1(&x);

        split_radix_dct1(&mut x);

        // Note: If you use the standard DCT-I definition above,
        // your fast version must also include the boundary scaling
        // to match exactly.
        assert_slice_close(&x, &expected, 1e-6);
    }

    #[test]
    fn test_dct1_size_9() {
        let x_vals = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let mut x = x_vals.clone();
        let expected = naive_dct1(&x);

        split_radix_dct1(&mut x);

        assert_slice_close(&x, &expected, 1e-6);
    }

    fn naive_dct1_unscaled(input: &[f64]) -> Vec<f64> {
        let n = input.len();
        let mut output = vec![0.0; n];
        for k in 0..n {
            // Standard definition: no 1/sqrt(2) scaling at endpoints
            let mut sum = input[0] + (if (k % 2) == 0 { 1.0 } else { -1.0 }) * input[n - 1];
            for n_idx in 1..n - 1 {
                sum += 2.0
                    * input[n_idx]
                    * (std::f64::consts::PI * (k as f64) * (n_idx as f64) / ((n - 1) as f64)).cos();
            }
            output[k] = sum;
        }
        output
    }

    #[test]
    fn test_dct1_len_3_routing_and_interleave() {
        // For n=2 (length 3), the DCT-III block is length 1, which our stub
        // naturally handles as a no-op. This allows us to fully test the
        // recursive T matrix and P^T interleave matrix.
        let mut x = [1.0, 2.0, 3.0];

        // Manual trace of the math in the function:
        // n = 2, n1 = 1.
        // v[1] = x[1] = 2.0
        // v[0] = (1.0 + 3.0) / sqrt(2) = 2.828427
        // v[2] = (1.0 - 3.0) / sqrt(2) = -1.414213
        //
        // Recurse on v[0..=1] (Length 2 base case):
        // v'[0] = (2.828427 + 2.0) / sqrt(2) = 3.414213
        // v'[1] = (2.828427 - 2.0) / sqrt(2) = 0.585786
        //
        // Stub on v[2..3]: remains -1.414213
        //
        // Interleave (P^T):
        // x[0] = v[0] = 3.414213
        // x[1] = v[2] = -1.414213
        // x[2] = v[1] = 0.585786

        let expected = naive_dct1(&x);

        split_radix_dct1(&mut x);

        assert_slice_close(&x, &expected, 1e-6);
    }

    #[test]
    #[should_panic(expected = "n must be an even integer")]
    fn test_invalid_length_panics() {
        // The paper states n must be even (so length n+1 must be odd for anything > 2)
        // Passing an array of length 4 implies n=3, which should trigger our assert.
        let mut x = [1.0, 2.0, 3.0, 4.0];
        split_radix_dct1(&mut x);
    }
}
