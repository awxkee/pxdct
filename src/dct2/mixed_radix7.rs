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
use crate::dct2::util::{radixq_cos_twiddle, radixq_rotation_twiddle};
use crate::mla::fmla;
use crate::util::{DctSample, try_vec};
use crate::{PxdctError, PxdctExecutor};
use num_complex::Complex;
use num_traits::{AsPrimitive, Zero};
use std::sync::Arc;

pub(crate) trait MixedRadix7Sample {
    // from sage.all import *
    // import struct
    //
    // R = RealField(90)
    //
    // def float_to_hex(f):
    //     packed = struct.pack('>f', float(f))
    //     return '0x' + packed.hex()
    //
    // def phase_a(q, m, j):
    //     img = (((q - 1 - 2*m)*j*R.pi())/(R(2) * q)).cos()
    //     return img
    //
    // even2 = phase_a(7, 0, 2)
    // print(even2)
    //
    // print(float_to_hex(even2))
    //
    // def double_to_hex(f):
    //         packed = struct.pack('>d', float(f))
    //         return '0x' + packed.hex()
    //
    // print(double_to_hex(even2))
    const R7_COS_EVEN2_M0: Self;
    // even2 = phase_a(7, 1, 2)
    // print(even2)
    const R7_COS_EVEN2_M1: Self;
    // even2 = phase_a(7, 2, 2)
    // print(even2)
    const R7_COS_EVEN2_M2: Self;
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
    //     theta = (q - 1 - 2*m)*(2*j+1)
    //     img = ((theta*R.pi())/(2*q)).sin()
    //     return img
    //
    // odd = phase_b(7, 0, 0)
    // print(odd)
    //
    // print(float_to_hex(odd))
    //
    // def double_to_hex(f):
    //         packed = struct.pack('>d', float(f))
    //         return '0x' + packed.hex()
    //
    // print(double_to_hex(odd))
    const R7_SIN_ODD0_M0: Self;
    // odd = phase_b(7, 1, 0)
    // print(odd)
    const R7_SIN_ODD0_M1: Self;
    // odd = phase_b(7, 2, 0)
    // print(odd)
    const R7_SIN_ODD0_M2: Self;
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
    //     theta = (q - 1 - 2*m)*(2*j+1)
    //     img = ((theta*R.pi())/(2*q)).sin()
    //     return img
    //
    // odd = phase_b(7, 0, 0)
    // print(odd)
    //
    // print(float_to_hex(odd))
    //
    // def double_to_hex(f):
    //         packed = struct.pack('>d', float(f))
    //         return '0x' + packed.hex()
    //
    // print(double_to_hex(odd))
    const R7_SIN_ODD1_M0: Self;
    // odd = phase_b(7, 1, 1)
    // print(odd)
    const R7_SIN_ODD1_M1: Self;
    // odd = phase_b(7, 1, 2)
    // print(odd)
    const R7_SIN_ODD1_M2: Self;
    // odd = phase_b(7, 0, 2)
    // print(odd)
    const R7_SIN_ODD2_M0: Self;
    // odd = phase_b(7, 1, 2)
    // print(odd)
    const R7_SIN_ODD2_M1: Self;
    // odd = phase_b(7, 2, 2)
    // print(odd)
    const R7_SIN_ODD2_M2: Self;
}

impl MixedRadix7Sample for f32 {
    const R7_COS_EVEN2_M0: Self = f32::from_bits(0xbf66a5e5);
    const R7_COS_EVEN2_M1: Self = f32::from_bits(0xbe63dc87);
    const R7_COS_EVEN2_M2: Self = f32::from_bits(0x3f1f9d07);
    const R7_SIN_ODD0_M0: Self = f32::from_bits(0x3f7994e0);
    const R7_SIN_ODD0_M1: Self = f32::from_bits(0x3f48261c);
    const R7_SIN_ODD0_M2: Self = f32::from_bits(0x3ede2602);
    const R7_SIN_ODD1_M0: Self = -Self::R7_SIN_ODD0_M1;
    const R7_SIN_ODD1_M1: Self = Self::R7_SIN_ODD0_M2;
    const R7_SIN_ODD1_M2: Self = Self::R7_SIN_ODD0_M0;
    const R7_SIN_ODD2_M0: Self = Self::R7_SIN_ODD0_M2;
    const R7_SIN_ODD2_M1: Self = -Self::R7_SIN_ODD0_M0;
    const R7_SIN_ODD2_M2: Self = Self::R7_SIN_ODD0_M1;
}
impl MixedRadix7Sample for f64 {
    const R7_COS_EVEN2_M0: Self = f64::from_bits(0xbfecd4bca9cb5c71);
    const R7_COS_EVEN2_M1: Self = f64::from_bits(0xbfcc7b90e3024582);
    const R7_COS_EVEN2_M2: Self = f64::from_bits(0x3fe3f3a0e28bedd1);
    const R7_SIN_ODD0_M0: Self = f64::from_bits(0x3fef329c0558e969);
    const R7_SIN_ODD0_M1: Self = f64::from_bits(0x3fe904c37505de4b);
    const R7_SIN_ODD0_M2: Self = f64::from_bits(0x3fdbc4c04d71abc1);
    const R7_SIN_ODD1_M0: Self = -Self::R7_SIN_ODD0_M1;
    const R7_SIN_ODD1_M1: Self = Self::R7_SIN_ODD0_M2;
    const R7_SIN_ODD1_M2: Self = Self::R7_SIN_ODD0_M0;
    const R7_SIN_ODD2_M0: Self = Self::R7_SIN_ODD0_M2;
    const R7_SIN_ODD2_M1: Self = -Self::R7_SIN_ODD0_M0;
    const R7_SIN_ODD2_M2: Self = Self::R7_SIN_ODD0_M1;
}

pub(crate) struct Dct2MixedRadix7<T> {
    rotation_layer: Vec<Complex<T>>,
    cos_twiddles: Vec<Complex<T>>,
    inner_dct: Arc<dyn PxdctExecutor<T> + Send + Sync>,
    execution_length: usize,
}

impl<T: DctSample + MixedRadix7Sample> Dct2MixedRadix7<T>
where
    f64: AsPrimitive<T>,
    usize: AsPrimitive<T>,
{
    #[allow(unused)]
    pub(crate) fn new(
        len: usize,
        inner_dct: Arc<dyn PxdctExecutor<T> + Send + Sync>,
    ) -> Result<Dct2MixedRadix7<T>, PxdctError> {
        assert!(
            len.is_multiple_of(7),
            "Mixed radix 7 should not be called on sizes no divisible by 7"
        );

        let q_modules = len / 7;

        // always 3 inner groups in Radix-7
        let inner_groups = 3;

        // Precompute rotation twiddles for k≥1
        // Format: [m0_k1, m1_k1, m0_k2, m1_k2, ...]
        let mut rotation_layer = try_vec![Complex::<T>::zero(); 3 * (q_modules - 1)];
        for (k, rotation_layer) in rotation_layer.chunks_exact_mut(3).enumerate() {
            for (m, layer) in rotation_layer.iter_mut().enumerate() {
                *layer =
                    radixq_rotation_twiddle(7, m, (k + 1).as_(), (q_modules - (k + 1)).as_(), len);
            }
        }

        // Precompute cosine twiddles for even components
        // Stored as Complex{re: even_twiddle, im: odd_twiddle} for cache efficiency
        let mut cos_twiddles = try_vec![Complex::<T>::zero(); (q_modules - 1) * inner_groups];
        for (k, k_layer) in cos_twiddles.chunks_exact_mut(inner_groups).enumerate() {
            for (m, m_layer) in k_layer.iter_mut().enumerate() {
                let k = k + 1;
                let even = radixq_cos_twiddle(7, m, k.as_(), len);
                let odd = radixq_cos_twiddle(
                    7,
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

        Ok(Dct2MixedRadix7 {
            rotation_layer,
            inner_dct,
            cos_twiddles,
            execution_length: len,
        })
    }
}

impl<T: DctSample + MixedRadix7Sample> PxdctExecutor<T> for Dct2MixedRadix7<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, data: &mut [T]) -> Result<(), PxdctError> {
        if !data.len().is_multiple_of(self.execution_length) {
            return Err(PxdctError::InvalidSizeMultiplier(
                data.len(),
                self.execution_length,
            ));
        }

        let mut scratch = try_vec![T::default(); self.execution_length];

        let q_modules = self.execution_length / 7;

        for chunk in data.chunks_exact_mut(self.execution_length) {
            let (a_buffer, c_s_buffer) = scratch.split_at_mut(q_modules);
            let (c_buffer, s_buffer) = c_s_buffer.split_at_mut(q_modules * 3);

            // Step 1: Decompose input into A (center), C (even-symmetric), S (odd-symmetric) buffers
            for (n, dst) in a_buffer.iter_mut().enumerate() {
                unsafe {
                    *dst = *chunk.get_unchecked(n * 7 + 3);
                }
            }

            // Extract and combine symmetric pairs with sign alternation for S buffer
            for (m, (c_buffer, s_buffer)) in c_buffer
                .chunks_exact_mut(q_modules)
                .zip(s_buffer.chunks_exact_mut(q_modules))
                .enumerate()
            {
                let mut sign = T::one();
                for (n, (c_dst, s_dst)) in c_buffer.iter_mut().zip(s_buffer.iter_mut()).enumerate()
                {
                    let u0 = unsafe { *chunk.get_unchecked(7 * n + m) };
                    let u1 = unsafe { *chunk.get_unchecked(7 * n + 7 - m - 1) };

                    *c_dst = u0 + u1;
                    *s_dst = (u0 - u1).mulsign(sign);

                    sign = -sign;
                }
            }

            // Step 2: Apply DCT-II to all buffers (A, C₀, C₁, S₀, S₁)
            self.inner_dct.execute(&mut scratch)?;

            let (a_buffer, c_s_buffer) = scratch.split_at_mut(q_modules);
            let (c_buffer, s_buffer) = c_s_buffer.split_at_mut(q_modules * 3);

            {
                // Step 3: Recombine transformed buffers with twiddle factors

                // Handle k=0 case (DC and low frequencies)
                let qc = c_buffer[0];
                let mut c0 = qc; // Component C₀ (position 0)
                let mut c1 = qc * T::R7_COS_EVEN2_M0; // Component C₂ (position 2, uses j=2)
                let mut c2 = qc * T::R7_COS_EVEN2_M2; // Component C₄ (position 4, uses j=4)
                let mut c3 = qc * T::R7_COS_EVEN2_M1; // Component C6 (position 6, uses j=6)

                let s0_twiddled = s_buffer[0];

                let mut s0 = s0_twiddled * T::R7_SIN_ODD0_M0;
                let mut s1 = s0_twiddled * T::R7_SIN_ODD1_M0;
                let mut s2 = s0_twiddled * T::R7_SIN_ODD2_M0;

                {
                    let ci = unsafe { *c_buffer.get_unchecked(q_modules) };
                    let si = unsafe { *s_buffer.get_unchecked(q_modules) };

                    let ci2 = unsafe { *c_buffer.get_unchecked(q_modules * 2) };
                    let si2 = unsafe { *s_buffer.get_unchecked(q_modules * 2) };

                    c0 = ci + c0 + ci2;

                    c1 = fmla(ci, T::R7_COS_EVEN2_M1, c1);
                    c1 = fmla(ci2, T::R7_COS_EVEN2_M2, c1);

                    c2 = fmla(ci, T::R7_COS_EVEN2_M0, c2);
                    c2 = fmla(ci2, T::R7_COS_EVEN2_M1, c2);

                    c3 = fmla(ci, T::R7_COS_EVEN2_M2, c3);
                    c3 = fmla(ci2, T::R7_COS_EVEN2_M0, c3);

                    s0 = fmla(si, T::R7_SIN_ODD0_M1, s0);
                    s0 = fmla(si2, T::R7_SIN_ODD0_M2, s0);

                    s1 = fmla(si, T::R7_SIN_ODD1_M1, s1);
                    s1 = fmla(si2, T::R7_SIN_ODD1_M2, s1);

                    s2 = fmla(si, T::R7_SIN_ODD2_M1, s2);
                    s2 = fmla(si2, T::R7_SIN_ODD2_M2, s2);
                }

                // Write output: C₀ (pos 0), S₁ (pos q_modules), C₂ (pos 2*q_modules),
                //               S₃ (pos 3*q_modules), C₄ (pos 4*q_modules)
                let a0 = a_buffer[0];
                let dc = c0 + a0;
                unsafe {
                    *chunk.get_unchecked_mut(0) = dc;
                }

                let dc2 = c2 + a0;
                unsafe {
                    *chunk.get_unchecked_mut(q_modules * 4) = dc2;
                }
                unsafe {
                    *chunk.get_unchecked_mut(q_modules * 3) = -s1;
                }

                unsafe {
                    *chunk.get_unchecked_mut(q_modules) = s0;
                }

                unsafe {
                    let idx1 = q_modules * 2;
                    let qid2 = -(c1 + a0); // negated 2j
                    *chunk.get_unchecked_mut(idx1) = qid2;
                }

                unsafe {
                    let dc3 = c3 + a0;
                    *chunk.get_unchecked_mut(q_modules * 6) = -dc3;
                }

                unsafe {
                    *chunk.get_unchecked_mut(q_modules * 5) = s2;
                }

                // Step 4: Handle k≥1 cases with rotation twiddles
                for k in 1..q_modules {
                    // Apply rotation twiddles to combine forward and inverted components
                    let rotation_twiddle =
                        unsafe { *self.rotation_layer.get_unchecked((k - 1) * 3) };

                    let c_forward = unsafe { *c_buffer.get_unchecked(k) };
                    let s_forward = unsafe { *s_buffer.get_unchecked(q_modules - k) };

                    let rotated_dc = fmla(s_forward, rotation_twiddle.re, c_forward);

                    let twiddle = unsafe { *self.cos_twiddles.get_unchecked((k - 1) * 3) };

                    let twiddled_dc = rotated_dc * twiddle.re;

                    let mut dc0 = twiddled_dc;
                    let mut dc2 = twiddled_dc * T::R7_COS_EVEN2_M0;
                    let mut dc4 = twiddled_dc * T::R7_COS_EVEN2_M2;
                    let mut dc6 = twiddled_dc * T::R7_COS_EVEN2_M1;

                    let rotated_ds = fmla(c_forward, rotation_twiddle.im, s_forward);

                    let twiddled_ds = rotated_ds * twiddle.im;

                    let mut ds1 = twiddled_ds * T::R7_SIN_ODD0_M0;
                    let mut ds3 = twiddled_ds * T::R7_SIN_ODD1_M0;
                    let mut ds5 = twiddled_ds * T::R7_SIN_ODD2_M0;

                    {
                        let c_forward = unsafe { *c_buffer.get_unchecked(q_modules + k) };
                        let s_forward = unsafe { *s_buffer.get_unchecked(q_modules * 2 - k) };

                        let rotation_twiddle =
                            unsafe { *self.rotation_layer.get_unchecked((k - 1) * 3 + 1) };

                        let twiddle = unsafe { *self.cos_twiddles.get_unchecked((k - 1) * 3 + 1) };

                        let rotated_dc1 = fmla(s_forward, rotation_twiddle.re, c_forward);
                        let rotated_ds2 = fmla(c_forward, rotation_twiddle.im, s_forward);

                        let twiddled_dc = twiddle.re * rotated_dc1;
                        let twiddled_ds = twiddle.im * rotated_ds2;

                        dc0 = twiddled_dc + dc0;
                        dc2 = fmla(twiddled_dc, T::R7_COS_EVEN2_M1, dc2);
                        dc4 = fmla(twiddled_dc, T::R7_COS_EVEN2_M0, dc4);
                        dc6 = fmla(twiddled_dc, T::R7_COS_EVEN2_M2, dc6);

                        ds1 = fmla(twiddled_ds, T::R7_SIN_ODD0_M1, ds1);
                        ds3 = fmla(twiddled_ds, T::R7_SIN_ODD1_M1, ds3);
                        ds5 = fmla(twiddled_ds, T::R7_SIN_ODD2_M1, ds5);
                    }

                    {
                        let c_forward = unsafe { *c_buffer.get_unchecked(q_modules * 2 + k) };
                        let s_forward = unsafe { *s_buffer.get_unchecked(q_modules * 3 - k) };

                        let rotation_twiddle =
                            unsafe { *self.rotation_layer.get_unchecked((k - 1) * 3 + 2) };

                        let twiddle = unsafe { *self.cos_twiddles.get_unchecked((k - 1) * 3 + 2) };

                        let rotated_dc1 = fmla(s_forward, rotation_twiddle.re, c_forward);
                        let rotated_ds2 = fmla(c_forward, rotation_twiddle.im, s_forward);

                        let twiddled_dc = twiddle.re * rotated_dc1;
                        let twiddled_ds = twiddle.im * rotated_ds2;

                        dc0 = twiddled_dc + dc0;
                        dc2 = fmla(twiddled_dc, T::R7_COS_EVEN2_M2, dc2);
                        dc4 = fmla(twiddled_dc, T::R7_COS_EVEN2_M1, dc4);
                        dc6 = fmla(twiddled_dc, T::R7_COS_EVEN2_M0, dc6);

                        ds1 = fmla(twiddled_ds, T::R7_SIN_ODD0_M2, ds1);
                        ds3 = fmla(twiddled_ds, T::R7_SIN_ODD1_M2, ds3);
                        ds5 = fmla(twiddled_ds, T::R7_SIN_ODD2_M2, ds5);
                    }

                    let a0 = unsafe { *a_buffer.get_unchecked(k) };
                    let dc = dc0 + a0;
                    unsafe {
                        *chunk.get_unchecked_mut(k) = dc;
                    }

                    let idx = q_modules * 2 - k;
                    let dss1 = fmla(2f64.as_(), ds1, -dc);
                    unsafe {
                        *chunk.get_unchecked_mut(idx) = dss1;
                    }

                    let idx1 = q_modules * 2 + k;
                    dc2 = -(dc2 + a0); // negated 2j
                    dc2 = fmla(2f64.as_(), dc2, -dss1);
                    unsafe {
                        *chunk.get_unchecked_mut(idx1) = dc2;
                    }

                    let idx = q_modules * 4 - k;
                    let dss3 = fmla(2f64.as_(), -ds3, -dc2);
                    unsafe {
                        *chunk.get_unchecked_mut(idx) = dss3;
                    }

                    dc4 += a0;

                    let idx1 = q_modules * 4 + k;
                    let mdc4 = fmla(2f64.as_(), dc4, -dss3);
                    unsafe {
                        *chunk.get_unchecked_mut(idx1) = mdc4;
                    }

                    let dss5 = fmla(2f64.as_(), ds5, -mdc4);
                    unsafe {
                        let idx = q_modules * 6 - k;
                        *chunk.get_unchecked_mut(idx) = dss5;
                    }

                    dc6 += a0;
                    dc6 = fmla(2f64.as_(), -dc6, -dss5);

                    unsafe {
                        let idx = q_modules * 6 + k;
                        *chunk.get_unchecked_mut(idx) = dc6;
                    }
                }
            }
        }

        Ok(())
    }

    #[inline]
    fn length(&self) -> usize {
        self.execution_length
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Pxdct;
    use crate::tests::naive_dct2_f32;

    #[test]
    fn test_radix7_dct() {
        let mut input = vec![0.; 7 * 5];
        for (i, z) in input.iter_mut().enumerate() {
            *z = i as f32 + rand::random::<f32>() * 10.0;
        }
        // let mut input = vec![
        //     7.6871257, 1.2637726, 11.096954, 4.113755, 6.3156953, 12.343984, 9.859292, 15.516256,
        //     12.010594, 18.957434, 11.183157, 16.510174, 13.310775, 21.062075, 19.775341, 20.445467,
        //     22.57258, 25.571342, 23.987795, 19.597996, 24.935028, 21.360756, 22.820232, 27.915956,
        //     31.28283, 27.915956, 31.28283, 7.6871257, 1.2637726, 11.096954, 4.113755, 6.3156953,
        //     12.343984, 9.859292, 15.516256, 12.010594, 18.957434, 11.183157, 16.510174, 13.310775,
        //     21.062075, 19.775341, 20.445467, 22.57258, 25.571342, 23.987795, 19.597996, 24.935028,
        //     21.360756,
        // ];
        let mut reference_input = input.clone();
        // let rr = Pxdct::make_dct2_f32(25).unwrap();
        // rr.execute(&mut reference_input).unwrap();
        reference_input = naive_dct2_f32(&reference_input);
        let bf = Dct2MixedRadix7::new(input.len(), Pxdct::make_dct2_f32(input.len() / 7).unwrap())
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
