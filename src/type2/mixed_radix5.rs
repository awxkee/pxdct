/*
 * // Copyright (c) Radzivon Bartoshyk 1/2026. All rights reserved.
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
use crate::bidirectional::{BidirectionalStore, InPlaceStore};
use crate::mla::fmla;
use crate::type2::util::{radixq_cos_twiddle, radixq_rotation_twiddle};
use crate::util::{DctSample, try_vec, validate_scratch};
use crate::{PxdctError, PxdctExecutor};
use num_complex::Complex;
use num_traits::{AsPrimitive, Zero};
use std::sync::Arc;

pub(crate) trait MixedRadix5Sample {
    //def float_to_hex(f):
    //     packed = struct.pack('>f', float(f))
    //     return '0x' + packed.hex()
    //
    // def phase_a(q, m, j):
    //     img = (((q - 1 - 2*m)*j*R(2)*R.pi())/q).cos()
    //     return img
    //
    // even2 = phase_a(5, 0, 2)
    // print(even2)
    //
    // float_to_hex(even2)
    //
    // def double_to_hex(f):
    //         packed = struct.pack('>d', float(f))
    //         return '0x' + packed.hex()
    // double_to_hex(even2)
    const R5_COS_EVEN2_M0: Self;
    const R5_COS_EVEN4_M0: Self;
    // from sage.all import *
    // import struct
    //
    // R = RealField(90)
    //
    // def float_to_hex(f):
    //     packed = struct.pack('>f', float(f))
    //     return '0x' + packed.hex()
    //
    // def phase_b(q, m, j):
    //     theta = (q - 1 - 2*m)*j
    //     img = ((theta*R.pi())/q).sin()
    //     return img
    //
    // odd = abs(phase_b(5, 0, 3))
    // print(odd)
    //
    // print(float_to_hex(odd))
    //
    // def double_to_hex(f):
    //         packed = struct.pack('>d', float(f))
    //         return '0x' + packed.hex()
    //
    // print(double_to_hex(odd))
    const R5_SIN_ODD_M0: Self;
    // from sage.all import *
    // import struct
    //
    // R = RealField(90)
    // def tan_module(q, k, m, n):
    //     return (((q - 1 - 2*m)*k*R.pi())/(2*n)).tan()
    //
    // def float_to_hex(f):
    //     packed = struct.pack('>f', float(f))
    //     return '0x' + packed.hex()
    //
    // def phase_b(q, m, j):
    //     theta = (q - 1 - 2*m)*j
    //     img = ((theta*R.pi())/q).sin()
    //     return img
    //
    // odd = -phase_b(5, 0, 1)
    // print(odd)
    //
    // print(float_to_hex(odd))
    //
    // def double_to_hex(f):
    //         packed = struct.pack('>d', float(f))
    //         return '0x' + packed.hex()
    //
    // print(double_to_hex(odd))
    const R5_SIN_ODD1_M0: Self;
}

impl MixedRadix5Sample for f32 {
    const R5_COS_EVEN2_M0: Self = f32::from_bits(0xbf4f1bbd);
    const R5_COS_EVEN4_M0: Self = f32::from_bits(0x3e9e377a);
    const R5_SIN_ODD_M0: Self = f32::from_bits(0x3f737871);
    const R5_SIN_ODD1_M0: Self = f32::from_bits(0xbf167918);
}
impl MixedRadix5Sample for f64 {
    const R5_COS_EVEN2_M0: Self = f64::from_bits(0xbfe9e3779b97f4a8);
    const R5_COS_EVEN4_M0: Self = f64::from_bits(0x3fd3c6ef372fe950);
    const R5_SIN_ODD_M0: Self = f64::from_bits(0x3fee6f0e134454ff);
    const R5_SIN_ODD1_M0: Self = f64::from_bits(0xbfe2cf2304755a5e);
}

/// Radix-5 DCT-II implementation using direct decomposition algorithm.
///
/// This implements a fast DCT-II for lengths divisible by 5, decomposing the transform
/// into smaller sub-transforms around a center pivot element. The algorithm exploits
/// the symmetry structure: C₀ - S₁ - A - S₃ - C₄, where A is the center buffer.
pub(crate) struct Dct2MixedRadix5<T> {
    /// Precomputed rotation twiddles: tan(π(q-1-2m)k/(2N)) for combining C and S buffers
    rotation_layer: Vec<Complex<T>>,
    /// Precomputed cosine twiddles for even and odd components
    cos_twiddles: Vec<Complex<T>>,
    inner_dct: Arc<dyn PxdctExecutor<T> + Send + Sync>,
    execution_length: usize,
}

impl<T: DctSample + MixedRadix5Sample> Dct2MixedRadix5<T>
where
    f64: AsPrimitive<T>,
    usize: AsPrimitive<T>,
{
    #[allow(unused)]
    pub(crate) fn new(
        len: usize,
        inner_dct: Arc<dyn PxdctExecutor<T> + Send + Sync>,
    ) -> Result<Dct2MixedRadix5<T>, PxdctError> {
        assert!(
            len.is_multiple_of(5),
            "Mixed radix 5 should not be called on sizes no divisible by 5"
        );

        let q_modules = len / 5;

        // always 2 inner groups in Radix-5
        let inner_groups = 2;

        // Precompute rotation twiddles for k≥1
        // Format: [m0_k1, m1_k1, m0_k2, m1_k2, ...]
        let mut rotation_layer = try_vec![Complex::<T>::zero(); 2 * (q_modules - 1)];
        for (k, rotation_layer) in rotation_layer.chunks_exact_mut(2).enumerate() {
            for (m, layer) in rotation_layer.iter_mut().enumerate() {
                *layer =
                    radixq_rotation_twiddle(5, m, (k + 1).as_(), (q_modules - (k + 1)).as_(), len);
            }
        }

        // Precompute cosine twiddles for even components
        // Stored as Complex{re: even_twiddle, im: odd_twiddle} for cache efficiency
        let mut cos_twiddles = try_vec![Complex::<T>::zero(); (q_modules - 1) * inner_groups];
        for (k, k_layer) in cos_twiddles.chunks_exact_mut(inner_groups).enumerate() {
            for (m, m_layer) in k_layer.iter_mut().enumerate() {
                let k = k + 1;
                let even = radixq_cos_twiddle(5, m, k.as_(), len);
                let odd = radixq_cos_twiddle(
                    5,
                    m,
                    if k == 0 {
                        k.as_()
                    } else {
                        (q_modules - k).as_()
                    },
                    len,
                );
                *m_layer = Complex { re: even, im: odd };
            }
        }

        Ok(Dct2MixedRadix5 {
            rotation_layer,
            inner_dct,
            cos_twiddles,
            execution_length: len,
        })
    }
}

impl<T: DctSample + MixedRadix5Sample> Dct2MixedRadix5<T>
where
    f64: AsPrimitive<T>,
{
    fn execute_with_store<S: BidirectionalStore<T>>(
        &self,
        data: &mut S,
        scratch: &mut [T],
    ) -> Result<(), PxdctError> {
        let (scratch, inner_scratch) = scratch.split_at_mut(self.execution_length);
        let q_modules = self.execution_length / 5;
        let (a_buffer, c_s_buffer) = scratch.split_at_mut(q_modules);
        let (c_buffer, s_buffer) = c_s_buffer.split_at_mut(q_modules * 2);

        // Step 1: Decompose input into A (center), C (even-symmetric), S (odd-symmetric) buffers
        for (n, dst) in a_buffer.iter_mut().enumerate() {
            *dst = data[n * 5 + 2];
        }

        // Extract and combine symmetric pairs with sign alternation for S buffer
        for (m, (c_buffer, s_buffer)) in c_buffer
            .chunks_exact_mut(q_modules)
            .zip(s_buffer.chunks_exact_mut(q_modules))
            .enumerate()
        {
            let mut sign = T::one();
            for (n, (c_dst, s_dst)) in c_buffer.iter_mut().zip(s_buffer.iter_mut()).enumerate() {
                let u0 = data[5 * n + m];
                let u1 = data[5 * n + 5 - m - 1];

                *c_dst = u0 + u1;
                *s_dst = (u0 - u1).mulsign(sign);

                sign = -sign;
            }
        }

        // Step 2: Apply DCT-II to all buffers (A, C₀, C₁, S₀, S₁)
        self.inner_dct
            .execute_with_scratch(scratch, inner_scratch)?;

        let (a_buffer, c_s_buffer) = scratch.split_at_mut(q_modules);
        let (c_buffer, s_buffer) = c_s_buffer.split_at_mut(q_modules * 2);

        {
            // Step 3: Recombine transformed buffers with twiddle factors

            // Handle k=0 case (DC and low frequencies)
            let qc = c_buffer[0];
            let mut c0 = qc; // Component C₀ (position 0)
            let mut c1 = qc * T::R5_COS_EVEN2_M0; // Component C₂ (position 2, uses j=2)
            let mut c2 = qc * T::R5_COS_EVEN4_M0; // Component C₄ (position 4, uses j=4)

            let s0_twiddled = s_buffer[0];

            // Odd components: S₁ uses j=1 (abs), S₃ uses j=3 (negated)
            let mut s0 = s0_twiddled * T::R5_SIN_ODD_M0; // S₁: abs(sin(3π/5))
            let mut s1 = s0_twiddled * T::R5_SIN_ODD1_M0; // S₃: -sin(π/5)

            {
                let ci = unsafe { *c_buffer.get_unchecked(q_modules) };
                let si = unsafe { *s_buffer.get_unchecked(q_modules) };

                let twiddle_ci = ci;
                let twiddle_si = si;

                c0 = ci + c0;
                c1 = fmla(twiddle_ci, T::R5_COS_EVEN4_M0, c1);
                c2 = fmla(twiddle_ci, T::R5_COS_EVEN2_M0, c2);
                s0 = fmla(twiddle_si, -T::R5_SIN_ODD1_M0, s0);
                s1 = fmla(twiddle_si, T::R5_SIN_ODD_M0, s1);
            }

            // Write output: C₀ (pos 0), S₁ (pos q_modules), C₂ (pos 2*q_modules),
            //               S₃ (pos 3*q_modules), C₄ (pos 4*q_modules)
            let a0 = a_buffer[0];
            let dc = c0 + a0;
            data[0] = dc;

            let dc2 = c2 + a0;
            data[q_modules * 4] = dc2;
            data[q_modules * 3] = -s1;
            data[q_modules] = s0;

            let idx1 = q_modules * 2;
            let qid2 = -(c1 + a0); // negated 2j
            data[idx1] = qid2;

            // Step 4: Handle k≥1 cases with rotation twiddles
            for k in 1..q_modules {
                // Apply rotation twiddles to combine forward and inverted components
                let rotation_twiddle = unsafe { *self.rotation_layer.get_unchecked((k - 1) * 2) };

                let c_forward = unsafe { *c_buffer.get_unchecked(k) };
                let s_forward = unsafe { *s_buffer.get_unchecked(q_modules - k) };

                let rotated_dc = fmla(s_forward, rotation_twiddle.re, c_forward);

                let twiddle = unsafe { *self.cos_twiddles.get_unchecked((k - 1) * 2) };

                let twiddled_dc = rotated_dc * twiddle.re;

                let mut dc0 = twiddled_dc;
                let mut dc2 = twiddled_dc * T::R5_COS_EVEN2_M0;
                let mut dc4 = twiddled_dc * T::R5_COS_EVEN4_M0;

                let rotated_ds = fmla(c_forward, rotation_twiddle.im, s_forward);

                let twiddled_ds = rotated_ds * twiddle.im;

                let mut ds1 = twiddled_ds * T::R5_SIN_ODD_M0;
                let mut ds3 = twiddled_ds * T::R5_SIN_ODD1_M0;

                {
                    let c_forward = unsafe { *c_buffer.get_unchecked(q_modules + k) };
                    let s_forward = unsafe { *s_buffer.get_unchecked(q_modules * 2 - k) };

                    let rotation_twiddle =
                        unsafe { *self.rotation_layer.get_unchecked((k - 1) * 2 + 1) };

                    let twiddle = unsafe { *self.cos_twiddles.get_unchecked((k - 1) * 2 + 1) };

                    let rotated_dc1 = fmla(s_forward, rotation_twiddle.re, c_forward);
                    let rotated_ds2 = fmla(c_forward, rotation_twiddle.im, s_forward);

                    let twiddled_dc = twiddle.re * rotated_dc1;
                    let twiddled_ds = twiddle.im * rotated_ds2;

                    dc0 = twiddled_dc + dc0;
                    dc2 = fmla(twiddled_dc, T::R5_COS_EVEN4_M0, dc2);
                    dc4 = fmla(twiddled_dc, T::R5_COS_EVEN2_M0, dc4);

                    ds1 = fmla(twiddled_ds, -T::R5_SIN_ODD1_M0, ds1);
                    ds3 = fmla(twiddled_ds, T::R5_SIN_ODD_M0, ds3);
                }

                let a0 = unsafe { *a_buffer.get_unchecked(k) };
                let dc = dc0 + a0;
                data[k] = dc;

                let idx = q_modules * 2 - k;
                let dss1 = fmla(2f64.as_(), ds1, -dc);
                data[idx] = dss1;

                let idx1 = q_modules * 2 + k;
                dc2 = -(dc2 + a0); // negated 2j
                dc2 = fmla(2f64.as_(), dc2, -dss1);
                data[idx1] = dc2;

                let idx = q_modules * 4 - k;
                let dss3 = fmla(2f64.as_(), -ds3, -dc2);
                data[idx] = dss3;

                dc4 += a0;

                let idx1 = q_modules * 4 + k;
                data[idx1] = fmla(2f64.as_(), dc4, -dss3);
            }
        }
        Ok(())
    }
}

impl<T: DctSample + MixedRadix5Sample> PxdctExecutor<T> for Dct2MixedRadix5<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, data: &mut [T]) -> Result<(), PxdctError> {
        let mut scratch = try_vec![T::default(); self.scratch_size()];
        self.execute_with_scratch(data, &mut scratch)
    }

    fn execute_with_scratch(&self, data: &mut [T], scratch: &mut [T]) -> Result<(), PxdctError> {
        if !data.len().is_multiple_of(self.execution_length) {
            return Err(PxdctError::InvalidSizeMultiplier(
                data.len(),
                self.execution_length,
            ));
        }

        let full_scratch = validate_scratch!(scratch, self.scratch_size());

        for chunk in data.chunks_exact_mut(self.execution_length) {
            self.execute_with_store(&mut InPlaceStore::new(chunk), full_scratch)?;
        }

        Ok(())
    }

    fn execute_into(&self, input: &[T], output: &mut [T]) -> Result<(), PxdctError> {
        let mut scratch = try_vec![T::default(); self.scratch_size()];
        self.execute_into_with_scratch(input, output, &mut scratch)
    }

    fn execute_into_with_scratch(
        &self,
        input: &[T],
        output: &mut [T],
        scratch: &mut [T],
    ) -> Result<(), PxdctError> {
        use crate::util::validate_oof_sizes;
        validate_oof_sizes!(input, output, self.execution_length);

        let full_scratch = validate_scratch!(scratch, self.scratch_size());

        use crate::bidirectional::BiStore;
        for (src, dst) in input
            .chunks_exact(self.execution_length)
            .zip(output.chunks_exact_mut(self.execution_length))
        {
            self.execute_with_store(&mut BiStore::new(src, dst), full_scratch)?;
        }
        Ok(())
    }

    #[inline]
    fn length(&self) -> usize {
        self.execution_length
    }

    fn scratch_size(&self) -> usize {
        self.execution_length + self.inner_dct.scratch_size()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Pxdct;
    use crate::tests::{naive_dct2, naive_dct2_f32};
    use crate::type2::Dct2Butterfly25;

    #[test]
    fn test_radix5_dct() {
        let mut input = vec![0.; 25];
        for (i, z) in input.iter_mut().enumerate() {
            *z = i as f32 + rand::random::<f32>() * 10.0;
        }
        let mut input = vec![
            7.6871257, 1.2637726, 11.096954, 4.113755, 6.3156953, 12.343984, 9.859292, 15.516256,
            12.010594, 18.957434, 11.183157, 16.510174, 13.310775, 21.062075, 19.775341, 20.445467,
            22.57258, 25.571342, 23.987795, 19.597996, 24.935028, 21.360756, 22.820232, 27.915956,
            31.28283, 24.935028, 21.360756, 22.820232, 27.915956, 31.28283,
        ];
        let mut reference_input = input.clone();
        let reference_input2 = input.clone();
        // let rr = Pxdct::make_dct2_f32(25).unwrap();
        // rr.execute(&mut reference_input).unwrap();
        reference_input = naive_dct2_f32(&reference_input);
        let bf = Dct2MixedRadix5::new(input.len(), Pxdct::make_dct2_f32(input.len() / 5).unwrap())
            .unwrap();
        bf.execute(&mut input).unwrap();
        println!(
            "{:?}",
            input
                .iter()
                .enumerate()
                .map(|(i, x)| format!("({i}) {}", x))
                .collect::<Vec<_>>()
        );
        println!(
            "{:?}",
            reference_input
                .iter()
                .enumerate()
                .map(|(i, x)| format!("({i}) {}", x))
                .collect::<Vec<_>>()
        );
        println!(
            "{:?}",
            reference_input2
                .iter()
                .enumerate()
                .map(|(i, x)| format!("({i}) {}", x))
                .collect::<Vec<_>>()
        );
        input
            .iter()
            .zip(reference_input.iter())
            .enumerate()
            .for_each(|(i, (&src, &r0))| {
                if (src - r0).abs() > 1e-1 {
                    println!(
                        "Difference must be < {}, but it was {}, at position {i}",
                        1e-1,
                        (src - r0).abs()
                    )
                }
            });
    }

    #[test]
    fn test_radix5_dctf64() {
        let mut input = vec![0.; 25];
        for (i, z) in input.iter_mut().enumerate() {
            *z = i as f32 + rand::random::<f32>() * 10.0;
        }
        let mut input = vec![
            7.6871257f64,
            1.2637726,
            11.096954,
            4.113755,
            6.3156953,
            12.343984,
            9.859292,
            15.516256,
            12.010594,
            18.957434,
            11.183157,
            16.510174,
            13.310775,
            21.062075,
            19.775341,
            20.445467,
            22.57258,
            25.571342,
            23.987795,
            19.597996,
            24.935028,
            21.360756,
            22.820232,
            27.915956,
            31.28283,
        ];
        let mut reference_input = input.clone();
        let mut reference_input2 = input.clone();
        // let rr = Pxdct::make_dct2_f32(25).unwrap();
        // rr.execute(&mut reference_input).unwrap();
        reference_input = naive_dct2(&reference_input);
        let bf = Dct2MixedRadix5::new(input.len(), Pxdct::make_dct2_f64(input.len() / 5).unwrap())
            .unwrap();
        bf.execute(&mut input).unwrap();
        let bf = Dct2Butterfly25::default();
        bf.execute(&mut reference_input2).unwrap();
        println!(
            "{:?}",
            input
                .iter()
                .enumerate()
                .map(|(i, x)| format!("({i}) {}", x))
                .collect::<Vec<_>>()
        );
        println!(
            "{:?}",
            reference_input
                .iter()
                .enumerate()
                .map(|(i, x)| format!("({i}) {}", x))
                .collect::<Vec<_>>()
        );
        println!(
            "{:?}",
            reference_input2
                .iter()
                .enumerate()
                .map(|(i, x)| format!("({i}) {}", x))
                .collect::<Vec<_>>()
        );
        input
            .iter()
            .zip(reference_input.iter())
            .enumerate()
            .for_each(|(i, (&src, &r0))| {
                if (src - r0).abs() > 1e-1 {
                    println!(
                        "Difference must be < {}, but it was {}, at position {i}",
                        1e-1,
                        (src - r0).abs()
                    )
                }
            });
    }
}
