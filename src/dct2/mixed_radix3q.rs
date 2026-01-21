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
use crate::dct2::{radixq_cos_twiddle, radixq_rotation_twiddle};
use crate::mla::fmla;
use crate::util::{DctSample, try_vec};
use crate::{PxdctError, PxdctExecutor};
use num_complex::Complex;
use num_traits::{AsPrimitive, Zero};
use std::sync::Arc;

pub(crate) trait MixedRadix3Sample {
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
    // odd = phase_b(3, 0, 0)
    // print(odd)
    //
    // print(float_to_hex(odd))
    //
    // def double_to_hex(f):
    //         packed = struct.pack('>d', float(f))
    //         return '0x' + packed.hex()
    //
    // print(double_to_hex(odd))
    const SIN2PI_OVER_3: Self;
}

impl MixedRadix3Sample for f32 {
    const SIN2PI_OVER_3: f32 = f32::from_bits(0x3f5db3d7);
}

impl MixedRadix3Sample for f64 {
    const SIN2PI_OVER_3: Self = f64::from_bits(0x3febb67ae8584caa);
}

pub(crate) struct Dct2MixedRadix3q<T> {
    twiddles: Vec<Complex<T>>,
    rotation_twiddles: Vec<Complex<T>>,
    inner_dct: Arc<dyn PxdctExecutor<T> + Send + Sync>,
    execution_length: usize,
}

impl<T: DctSample> Dct2MixedRadix3q<T>
where
    f64: AsPrimitive<T>,
    usize: AsPrimitive<T>,
{
    #[allow(unused)]
    pub(crate) fn new(
        len: usize,
        third_dct: Arc<dyn PxdctExecutor<T> + Send + Sync>,
    ) -> Result<Dct2MixedRadix3q<T>, PxdctError> {
        assert!(
            len.is_multiple_of(3),
            "Mixed radix 5 should not be called on sizes no divisible by 5"
        );
        let q_modules = len / 3;

        // always 1 inner groups in Radix-3
        let inner_groups = 1;

        let mut rotation_layer = try_vec![Complex::<T>::zero();  q_modules - 1];
        for (k, rotation_layer) in rotation_layer.chunks_exact_mut(1).enumerate() {
            for (m, layer) in rotation_layer.iter_mut().enumerate() {
                *layer =
                    radixq_rotation_twiddle(3, m, (k + 1).as_(), (q_modules - (k + 1)).as_(), len);
            }
        }

        // Precompute cosine twiddles for even components
        // Stored as Complex{re: even_twiddle, im: odd_twiddle} for cache efficiency
        let mut cos_twiddles = try_vec![Complex::<T>::zero(); (q_modules - 1) * inner_groups];
        for (k, k_layer) in cos_twiddles.chunks_exact_mut(inner_groups).enumerate() {
            for (m, m_layer) in k_layer.iter_mut().enumerate() {
                let k = k + 1;
                let even = radixq_cos_twiddle(3, m, k.as_(), len);
                let odd = radixq_cos_twiddle(
                    3,
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

        Ok(Dct2MixedRadix3q {
            twiddles: cos_twiddles,
            rotation_twiddles: rotation_layer,
            inner_dct: third_dct,
            execution_length: len,
        })
    }
}

impl<T: DctSample + MixedRadix3Sample> PxdctExecutor<T> for Dct2MixedRadix3q<T>
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

        assert!(!self.twiddles.is_empty());

        let q_modules = self.execution_length / 3;

        let mut scratch = try_vec![T::default(); self.execution_length];

        for chunk in data.chunks_exact_mut(self.execution_length) {
            let (a_buffer, c_s_buffer) = scratch.split_at_mut(q_modules);
            let (c_buffer, s_buffer) = c_s_buffer.split_at_mut(q_modules);

            // Step 1: Decompose input into A (center), C (even-symmetric), S (odd-symmetric) buffers
            for (n, dst) in a_buffer.iter_mut().enumerate() {
                unsafe {
                    *dst = *chunk.get_unchecked(n * 3 + 1);
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
                    let u0 = unsafe { *chunk.get_unchecked(3 * n + m) };
                    let u1 = unsafe { *chunk.get_unchecked(3 * n + 3 - m - 1) };

                    *c_dst = u0 + u1;
                    *s_dst = (u0 - u1).mulsign(sign);

                    sign = -sign;
                }
            }

            // Step 2: Apply DCT-II to all buffers (A, C₀, C₁, S₀, S₁)
            self.inner_dct.execute(&mut scratch)?;

            let (a_buffer, c_s_buffer) = scratch.split_at_mut(q_modules);
            let (c_buffer, s_buffer) = c_s_buffer.split_at_mut(q_modules);

            {
                // Step 3: Recombine transformed buffers with twiddle factors

                // Handle k=0 case (DC and low frequencies)
                let qc = c_buffer[0];
                let c0 = qc; // Component C₀ (position 0)
                let c1 = qc * -T::HALF; // Component C₂ (position 2, uses j=2)

                let s0_twiddled = s_buffer[0];

                // Odd components: S₁ uses j=1
                let s0 = s0_twiddled * T::SIN2PI_OVER_3; // S₁: sin(2π/5)

                // Write output: C₀ (pos 0), S₁ (pos q_modules)
                let a0 = a_buffer[0];
                let dc = c0 + a0;
                unsafe {
                    *chunk.get_unchecked_mut(0) = dc;
                }

                unsafe {
                    *chunk.get_unchecked_mut(q_modules) = s0;
                }

                unsafe {
                    let idx1 = q_modules * 2;
                    let qid2 = -(c1 + a0); // negated 2j
                    *chunk.get_unchecked_mut(idx1) = qid2;
                }

                // Step 4: Handle k≥1 cases with rotation twiddles
                for k in 1..q_modules {
                    // Apply rotation twiddles to combine forward and inverted components
                    let rotation_twiddle = unsafe { *self.rotation_twiddles.get_unchecked(k - 1) };

                    let c_forward = unsafe { *c_buffer.get_unchecked(k) };
                    let s_forward = unsafe { *s_buffer.get_unchecked(q_modules - k) };

                    let rotated_dc = fmla(s_forward, rotation_twiddle.re, c_forward);

                    let twiddle = unsafe { *self.twiddles.get_unchecked(k - 1) };

                    let twiddled_dc = rotated_dc * twiddle.re;

                    let dc0 = twiddled_dc;
                    let mut dc2 = twiddled_dc * -T::HALF;

                    let rotated_ds = fmla(c_forward, rotation_twiddle.im, s_forward);

                    let twiddled_ds = rotated_ds * twiddle.im;

                    let ds1 = twiddled_ds * T::SIN2PI_OVER_3;

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
    use crate::dct2::power2_butterflies::Dct2Butterfly4;
    use crate::tests::naive_dct2;
    use rand::Rng;

    #[test]
    fn test_radix3_dct() {
        let mut input = vec![0.; 12];
        for z in input.iter_mut() {
            *z = rand::rng().random_range(1.0..2.0);
        }
        let reference_input = input.clone();
        let reference_input = naive_dct2(&reference_input);
        let bf = Dct2MixedRadix3q::new(12, Arc::new(Dct2Butterfly4::default())).unwrap();
        bf.execute(&mut input).unwrap();
        input
            .iter()
            .zip(reference_input.iter())
            .enumerate()
            .for_each(|(i, (&src, &r0))| {
                assert!(
                    (src - r0).abs() < 1e-7,
                    "Difference must be < {}, but it was {}, at position {i}",
                    1e-7,
                    (src - r0).abs()
                )
            });
    }
}
