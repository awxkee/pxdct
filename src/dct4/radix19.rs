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
use crate::dct4::prime_butterflies::Dct4MixedRadix19Sample;
use crate::dct4::utils::radixq_dct4_rotation_twiddle;
use crate::mla::fmla;
use crate::util::{DctSample, try_vec, validate_scratch};
use crate::{PxdctError, PxdctExecutor};
use num_complex::Complex;
use num_traits::AsPrimitive;
use std::sync::Arc;

pub(crate) struct Dct4MixedRadix19<T> {
    inner_dct4: Arc<dyn PxdctExecutor<T> + Send + Sync>,
    inner_dct4_scratch_size: usize,
    rotation_twiddles: Vec<Complex<T>>,
    execution_length: usize,
    q_modules: usize,
    s: usize,
}

impl<T: DctSample> Dct4MixedRadix19<T>
where
    f64: AsPrimitive<T>,
    usize: AsPrimitive<T>,
{
    #[allow(unused)]
    pub(crate) fn new(
        len: usize,
        dct4: Arc<dyn PxdctExecutor<T> + Send + Sync>,
    ) -> Result<Self, PxdctError> {
        assert_eq!(
            dct4.length(),
            len / 19,
            "DCT-IV Mixed-Radix-17 length DCTs must be one seventeenth of DCT-IV"
        );

        let mut twiddles = try_vec![Complex::<T>::default(); len / 19 * 9];
        for (k, dst) in twiddles.chunks_exact_mut(9).enumerate() {
            dst[0] = radixq_dct4_rotation_twiddle(19, 0, k, len);
            dst[1] = radixq_dct4_rotation_twiddle(19, 1, k, len);
            dst[2] = radixq_dct4_rotation_twiddle(19, 2, k, len);
            dst[3] = radixq_dct4_rotation_twiddle(19, 3, k, len);
            dst[4] = radixq_dct4_rotation_twiddle(19, 4, k, len);
            dst[5] = radixq_dct4_rotation_twiddle(19, 5, k, len);
            dst[6] = radixq_dct4_rotation_twiddle(19, 6, k, len);
            dst[7] = radixq_dct4_rotation_twiddle(19, 7, k, len);
            dst[8] = radixq_dct4_rotation_twiddle(19, 8, k, len);
        }

        let inner_dct4_scratch_size = dct4.scratch_size();

        let q_modules = len / 19;
        let s = 2 * len / 19;

        Ok(Self {
            inner_dct4: dct4,
            inner_dct4_scratch_size,
            execution_length: len,
            rotation_twiddles: twiddles,
            q_modules,
            s,
        })
    }
}

impl<T: DctSample + Dct4MixedRadix19Sample> Dct4MixedRadix19<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    fn execute_with_store<S: BidirectionalStore<T>>(
        &self,
        data: &mut S,
        scratch: &mut [T],
    ) -> Result<(), PxdctError> {
        let (scratch, inner_scratch) = scratch.split_at_mut(self.execution_length);
        let (a_buffer, c_s_buffer) = scratch.split_at_mut(self.q_modules);
        let (c_buffer, s_buffer) = c_s_buffer.split_at_mut(self.q_modules * 9);

        // Step 1: Decompose input into A (center), C (even-symmetric), S (odd-symmetric) buffers
        for (n, dst) in a_buffer.iter_mut().enumerate() {
            *dst = data[n * 19 + 9];
        }

        // Extract and combine symmetric pairs with sign alternation for S buffer
        for (m, (c_buffer, s_buffer)) in c_buffer
            .chunks_exact_mut(self.q_modules)
            .zip(s_buffer.chunks_exact_mut(self.q_modules))
            .enumerate()
        {
            let mut sign = T::one();
            for (n, (c_dst, s_dst)) in c_buffer.iter_mut().zip(s_buffer.iter_mut()).enumerate() {
                let u0 = data[19 * n + m];
                let u1 = data[19 * n + 19 - m - 1];

                *c_dst = u0 + u1;
                *s_dst = (u0 - u1).mulsign(sign);

                sign = -sign;
            }
        }

        // Step 2: Apply DCT-IV to all buffers (A, C₀, C₁, S₀, S₁)
        self.inner_dct4
            .execute_with_scratch(scratch, inner_scratch)?;

        let (a_buffer, c_s_buffer) = scratch.split_at_mut(self.q_modules);
        let (c_buffer, s_buffer) = c_s_buffer.split_at_mut(self.q_modules * 9);

        // Step 4: Handle k≥0 cases with rotation twiddles
        for k in 0..self.q_modules {
            let c_v0 = unsafe { *c_buffer.get_unchecked(k) };
            let s_v0 = unsafe { *s_buffer.get_unchecked(self.q_modules - 1 - k) };
            let a_v0 = unsafe { *a_buffer.get_unchecked(k) };

            let c_v1 = unsafe { *c_buffer.get_unchecked(self.q_modules + k) };
            let s_v1 = unsafe { *s_buffer.get_unchecked(self.q_modules * 2 - 1 - k) };

            let c_v2 = unsafe { *c_buffer.get_unchecked(self.q_modules * 2 + k) };
            let s_v2 = unsafe { *s_buffer.get_unchecked(self.q_modules * 3 - 1 - k) };

            let c_v3 = unsafe { *c_buffer.get_unchecked(self.q_modules * 3 + k) };
            let s_v3 = unsafe { *s_buffer.get_unchecked(self.q_modules * 4 - 1 - k) };

            let c_v4 = unsafe { *c_buffer.get_unchecked(self.q_modules * 4 + k) };
            let s_v4 = unsafe { *s_buffer.get_unchecked(self.q_modules * 5 - 1 - k) };

            let c_v5 = unsafe { *c_buffer.get_unchecked(self.q_modules * 5 + k) };
            let s_v5 = unsafe { *s_buffer.get_unchecked(self.q_modules * 6 - 1 - k) };

            let c_v6 = unsafe { *c_buffer.get_unchecked(self.q_modules * 6 + k) };
            let s_v6 = unsafe { *s_buffer.get_unchecked(self.q_modules * 7 - 1 - k) };

            let c_v7 = unsafe { *c_buffer.get_unchecked(self.q_modules * 7 + k) };
            let s_v7 = unsafe { *s_buffer.get_unchecked(self.q_modules * 8 - 1 - k) };

            let c_v8 = unsafe { *c_buffer.get_unchecked(self.q_modules * 8 + k) };
            let s_v8 = unsafe { *s_buffer.get_unchecked(self.q_modules * 9 - 1 - k) };

            let twiddle0 = unsafe { self.rotation_twiddles.get_unchecked(k * 9) };
            let twiddle1 = unsafe { self.rotation_twiddles.get_unchecked(k * 9 + 1) };
            let twiddle2 = unsafe { self.rotation_twiddles.get_unchecked(k * 9 + 2) };
            let twiddle3 = unsafe { self.rotation_twiddles.get_unchecked(k * 9 + 3) };
            let twiddle4 = unsafe { self.rotation_twiddles.get_unchecked(k * 9 + 4) };
            let twiddle5 = unsafe { self.rotation_twiddles.get_unchecked(k * 9 + 5) };
            let twiddle6 = unsafe { self.rotation_twiddles.get_unchecked(k * 9 + 6) };
            let twiddle7 = unsafe { self.rotation_twiddles.get_unchecked(k * 9 + 7) };
            let twiddle8 = unsafe { self.rotation_twiddles.get_unchecked(k * 9 + 8) };

            let iq0 = fmla(c_v0, twiddle0.re, s_v0 * twiddle0.im);
            let siq0 = fmla(c_v0, twiddle0.im, -s_v0 * twiddle0.re);
            let mut u0 = iq0;
            let mut u1 = u0;
            let mut v0 = siq0;

            u1 *= T::D4_R19_ROT_TWIDDLE_2;
            v0 *= T::D4_R19_ROT_TWIDDLE_3;

            let iq1 = fmla(c_v1, twiddle1.re, s_v1 * twiddle1.im);
            let siq1 = fmla(c_v1, twiddle1.im, -s_v1 * twiddle1.re);

            u1 = fmla(iq1, T::D4_R19_ROT_TWIDDLE_6, u1);
            v0 = fmla(siq1, T::D4_R19_ROT_TWIDDLE_7, v0);

            let iq2 = fmla(c_v2, twiddle2.re, s_v2 * twiddle2.im);
            let siq2 = fmla(c_v2, twiddle2.im, -s_v2 * twiddle2.re);

            u1 = fmla(iq2, T::D4_R19_ROT_TWIDDLE_1, u1);
            v0 = fmla(siq2, T::D4_R19_ROT_TWIDDLE_0, v0);

            let iq3 = fmla(c_v3, twiddle3.re, s_v3 * twiddle3.im);
            let siq3 = fmla(c_v3, twiddle3.im, -s_v3 * twiddle3.re);

            u1 = fmla(iq3, T::D4_R19_ROT_TWIDDLE_10, u1);
            v0 = fmla(siq3, T::D4_R19_ROT_TWIDDLE_11, v0);

            let iq4 = fmla(c_v4, twiddle4.re, s_v4 * twiddle4.im);
            let siq4 = fmla(c_v4, twiddle4.im, -s_v4 * twiddle4.re);

            u1 = fmla(iq4, T::D4_R19_ROT_TWIDDLE_12, u1);
            v0 = fmla(siq4, T::D4_R19_ROT_TWIDDLE_13, v0);

            let iq5 = fmla(c_v5, twiddle5.re, s_v5 * twiddle5.im);
            let siq5 = fmla(c_v5, twiddle5.im, -s_v5 * twiddle5.re);

            u1 = fmla(iq5, T::D4_R19_ROT_TWIDDLE_16, u1);
            v0 = fmla(siq5, T::D4_R19_ROT_TWIDDLE_17, v0);

            let iq6 = fmla(c_v6, twiddle6.re, s_v6 * twiddle6.im);
            let siq6 = fmla(c_v6, twiddle6.im, -s_v6 * twiddle6.re);

            u1 = fmla(iq6, -T::D4_R19_ROT_TWIDDLE_9, u1);
            v0 = fmla(siq6, T::D4_R19_ROT_TWIDDLE_8, v0);

            let iq7 = fmla(c_v7, twiddle7.re, s_v7 * twiddle7.im);
            let siq7 = fmla(c_v7, twiddle7.im, -s_v7 * twiddle7.re);

            u1 = fmla(iq7, -T::D4_R19_ROT_TWIDDLE_4, u1);
            v0 = fmla(siq7, T::D4_R19_ROT_TWIDDLE_5, v0);

            let iq8 = fmla(c_v8, twiddle8.re, s_v8 * twiddle8.im);
            let siq8 = fmla(c_v8, twiddle8.im, -s_v8 * twiddle8.re);

            u1 = fmla(iq8, -T::D4_R19_ROT_TWIDDLE_14, u1);
            v0 = fmla(siq8, T::D4_R19_ROT_TWIDDLE_15, v0);

            u0 += iq1 + iq2 + iq3 + iq4 + iq5 + iq6 + iq7 + iq8 + a_v0;
            u1 = u1 - a_v0;

            let uc0 = u1 - v0;
            let uc1 = u1 + v0;

            data[k] = u0;
            data[self.s + k] = uc1;
            data[self.s - 1 - k] = uc0;

            let mut u2 = iq0;
            let mut v2 = siq0;
            u2 *= T::D4_R19_ROT_TWIDDLE_14;
            v2 *= T::D4_R19_ROT_TWIDDLE_15;
            u2 = fmla(iq1, T::D4_R19_ROT_TWIDDLE_9, u2);
            v2 = fmla(siq1, T::D4_R19_ROT_TWIDDLE_8, v2);
            u2 = fmla(iq2, -T::D4_R19_ROT_TWIDDLE_12, u2);
            v2 = fmla(siq2, T::D4_R19_ROT_TWIDDLE_13, v2);
            u2 = fmla(iq3, -T::D4_R19_ROT_TWIDDLE_1, u2);
            v2 = fmla(siq3, T::D4_R19_ROT_TWIDDLE_0, v2);
            u2 = fmla(iq4, -T::D4_R19_ROT_TWIDDLE_2, u2);
            v2 = fmla(siq4, T::D4_R19_ROT_TWIDDLE_3, v2);
            u2 = fmla(iq5, -T::D4_R19_ROT_TWIDDLE_6, u2);
            v2 = fmla(siq5, -T::D4_R19_ROT_TWIDDLE_7, v2);
            u2 = fmla(iq6, -T::D4_R19_ROT_TWIDDLE_10, u2);
            v2 = fmla(siq6, -T::D4_R19_ROT_TWIDDLE_11, v2);
            u2 = fmla(iq7, -T::D4_R19_ROT_TWIDDLE_16, u2);
            v2 = fmla(siq7, -T::D4_R19_ROT_TWIDDLE_17, v2);
            u2 = fmla(iq8, T::D4_R19_ROT_TWIDDLE_4, u2);
            v2 = fmla(siq8, -T::D4_R19_ROT_TWIDDLE_5, v2);
            u2 += a_v0;
            let uc2 = u2 - v2;
            let uc3 = u2 + v2;

            data[2 * self.s - 1 - k] = uc2;
            data[2 * self.s + k] = uc3;

            let mut u3 = iq0;
            let mut v3 = siq0;
            u3 *= T::D4_R19_ROT_TWIDDLE_6;
            v3 *= T::D4_R19_ROT_TWIDDLE_7;
            u3 = fmla(iq1, T::D4_R19_ROT_TWIDDLE_12, u3);
            v3 = fmla(siq1, T::D4_R19_ROT_TWIDDLE_13, v3);
            u3 = fmla(iq2, -T::D4_R19_ROT_TWIDDLE_4, u3);
            v3 = fmla(siq2, T::D4_R19_ROT_TWIDDLE_5, v3);
            u3 = fmla(iq3, -T::D4_R19_ROT_TWIDDLE_14, u3);
            v3 = fmla(siq3, -T::D4_R19_ROT_TWIDDLE_15, v3);
            u3 = fmla(iq4, T::D4_R19_ROT_TWIDDLE_16, u3);
            v3 = fmla(siq4, -T::D4_R19_ROT_TWIDDLE_17, v3);
            u3 = fmla(iq5, T::D4_R19_ROT_TWIDDLE_1, u3);
            v3 = fmla(siq5, -T::D4_R19_ROT_TWIDDLE_0, v3);
            u3 = fmla(iq6, T::D4_R19_ROT_TWIDDLE_2, u3);
            v3 = fmla(siq6, T::D4_R19_ROT_TWIDDLE_3, v3);
            u3 = fmla(iq7, T::D4_R19_ROT_TWIDDLE_10, u3);
            v3 = fmla(siq7, T::D4_R19_ROT_TWIDDLE_11, v3);
            u3 = fmla(iq8, -T::D4_R19_ROT_TWIDDLE_9, u3);
            v3 = fmla(siq8, T::D4_R19_ROT_TWIDDLE_8, v3);
            u3 = u3 - a_v0;
            let uc4 = u3 - v3;
            let uc5 = u3 + v3;

            data[3 * self.s - 1 - k] = uc4;
            data[3 * self.s + k] = uc5;

            let mut u4 = iq0;
            let mut v4 = siq0;
            u4 *= T::D4_R19_ROT_TWIDDLE_4;
            v4 *= T::D4_R19_ROT_TWIDDLE_5;
            u4 = fmla(iq1, -T::D4_R19_ROT_TWIDDLE_10, u4);
            v4 = fmla(siq1, T::D4_R19_ROT_TWIDDLE_11, v4);
            u4 = fmla(iq2, -T::D4_R19_ROT_TWIDDLE_2, u4);
            v4 = fmla(siq2, -T::D4_R19_ROT_TWIDDLE_3, v4);
            u4 = fmla(iq3, -T::D4_R19_ROT_TWIDDLE_12, u4);
            v4 = fmla(siq3, -T::D4_R19_ROT_TWIDDLE_13, v4);
            u4 = fmla(iq4, T::D4_R19_ROT_TWIDDLE_14, u4);
            v4 = fmla(siq4, -T::D4_R19_ROT_TWIDDLE_15, v4);
            u4 = fmla(iq5, T::D4_R19_ROT_TWIDDLE_9, u4);
            v4 = fmla(siq5, T::D4_R19_ROT_TWIDDLE_8, v4);
            u4 = fmla(iq6, -T::D4_R19_ROT_TWIDDLE_1, u4);
            v4 = fmla(siq6, T::D4_R19_ROT_TWIDDLE_0, v4);
            u4 = fmla(iq7, -T::D4_R19_ROT_TWIDDLE_6, u4);
            v4 = fmla(siq7, -T::D4_R19_ROT_TWIDDLE_7, v4);
            u4 = fmla(iq8, -T::D4_R19_ROT_TWIDDLE_16, u4);
            v4 = fmla(siq8, -T::D4_R19_ROT_TWIDDLE_17, v4);
            u4 += a_v0;
            let uc6 = u4 - v4;
            let uc7 = u4 + v4;

            data[4 * self.s - 1 - k] = uc6;
            data[4 * self.s + k] = uc7;

            let mut u5 = iq0;
            let mut v5 = siq0;
            u5 *= T::D4_R19_ROT_TWIDDLE_1;
            v5 *= T::D4_R19_ROT_TWIDDLE_0;
            u5 = fmla(iq1, -T::D4_R19_ROT_TWIDDLE_4, u5);
            v5 = fmla(siq1, T::D4_R19_ROT_TWIDDLE_5, v5);
            u5 = fmla(iq2, -T::D4_R19_ROT_TWIDDLE_9, u5);
            v5 = fmla(siq2, -T::D4_R19_ROT_TWIDDLE_8, v5);
            u5 = fmla(iq3, T::D4_R19_ROT_TWIDDLE_6, u5);
            v5 = fmla(siq3, -T::D4_R19_ROT_TWIDDLE_7, v5);
            u5 = fmla(iq4, T::D4_R19_ROT_TWIDDLE_10, u5);
            v5 = fmla(siq4, T::D4_R19_ROT_TWIDDLE_11, v5);
            u5 = fmla(iq5, -T::D4_R19_ROT_TWIDDLE_14, u5);
            v5 = fmla(siq5, T::D4_R19_ROT_TWIDDLE_15, v5);
            u5 = fmla(iq6, T::D4_R19_ROT_TWIDDLE_16, u5);
            v5 = fmla(siq6, -T::D4_R19_ROT_TWIDDLE_17, v5);
            u5 = fmla(iq7, T::D4_R19_ROT_TWIDDLE_2, u5);
            v5 = fmla(siq7, -T::D4_R19_ROT_TWIDDLE_3, v5);
            u5 = fmla(iq8, T::D4_R19_ROT_TWIDDLE_12, u5);
            v5 = fmla(siq8, T::D4_R19_ROT_TWIDDLE_13, v5);
            u5 = u5 - a_v0;
            let uc8 = u5 - v5;
            let uc9 = u5 + v5;

            data[5 * self.s - 1 - k] = uc8;
            data[5 * self.s + k] = uc9;

            let mut u6 = iq0;
            let mut v6 = siq0;
            u6 *= T::D4_R19_ROT_TWIDDLE_9;
            v6 *= T::D4_R19_ROT_TWIDDLE_8;
            u6 = fmla(iq1, -T::D4_R19_ROT_TWIDDLE_2, u6);
            v6 = fmla(siq1, T::D4_R19_ROT_TWIDDLE_3, v6);
            u6 = fmla(iq2, -T::D4_R19_ROT_TWIDDLE_16, u6);
            v6 = fmla(siq2, -T::D4_R19_ROT_TWIDDLE_17, v6);
            u6 = fmla(iq3, T::D4_R19_ROT_TWIDDLE_4, u6);
            v6 = fmla(siq3, T::D4_R19_ROT_TWIDDLE_5, v6);
            u6 = fmla(iq4, -T::D4_R19_ROT_TWIDDLE_6, u6);
            v6 = fmla(siq4, T::D4_R19_ROT_TWIDDLE_7, v6);
            u6 = fmla(iq5, -T::D4_R19_ROT_TWIDDLE_12, u6);
            v6 = fmla(siq5, -T::D4_R19_ROT_TWIDDLE_13, v6);
            u6 = fmla(iq6, T::D4_R19_ROT_TWIDDLE_14, u6);
            v6 = fmla(siq6, T::D4_R19_ROT_TWIDDLE_15, v6);
            u6 = fmla(iq7, -T::D4_R19_ROT_TWIDDLE_1, u6);
            v6 = fmla(siq7, T::D4_R19_ROT_TWIDDLE_0, v6);
            u6 = fmla(iq8, -T::D4_R19_ROT_TWIDDLE_10, u6);
            v6 = fmla(siq8, -T::D4_R19_ROT_TWIDDLE_11, v6);
            u6 += a_v0;
            let uc10 = u6 - v6;
            let uc11 = u6 + v6;

            data[6 * self.s - 1 - k] = uc10;
            data[6 * self.s + k] = uc11;

            let mut u7 = iq0;
            let mut v7 = siq0;
            u7 *= T::D4_R19_ROT_TWIDDLE_10;
            v7 *= T::D4_R19_ROT_TWIDDLE_11;
            u7 = fmla(iq1, -T::D4_R19_ROT_TWIDDLE_14, u7);
            v7 = fmla(siq1, -T::D4_R19_ROT_TWIDDLE_15, v7);
            u7 = fmla(iq2, T::D4_R19_ROT_TWIDDLE_6, u7);
            v7 = fmla(siq2, -T::D4_R19_ROT_TWIDDLE_7, v7);
            u7 = fmla(iq3, T::D4_R19_ROT_TWIDDLE_16, u7);
            v7 = fmla(siq3, T::D4_R19_ROT_TWIDDLE_17, v7);
            u7 = fmla(iq4, -T::D4_R19_ROT_TWIDDLE_9, u7);
            v7 = fmla(siq4, -T::D4_R19_ROT_TWIDDLE_8, v7);
            u7 = fmla(iq5, T::D4_R19_ROT_TWIDDLE_2, u7);
            v7 = fmla(siq5, T::D4_R19_ROT_TWIDDLE_3, v7);
            u7 = fmla(iq6, -T::D4_R19_ROT_TWIDDLE_4, u7);
            v7 = fmla(siq6, T::D4_R19_ROT_TWIDDLE_5, v7);
            u7 = fmla(iq7, T::D4_R19_ROT_TWIDDLE_12, u7);
            v7 = fmla(siq7, -T::D4_R19_ROT_TWIDDLE_13, v7);
            u7 = fmla(iq8, T::D4_R19_ROT_TWIDDLE_1, u7);
            v7 = fmla(siq8, T::D4_R19_ROT_TWIDDLE_0, v7);
            u7 = u7 - a_v0;

            let uc12 = u7 - v7;
            let uc13 = u7 + v7;

            data[7 * self.s - 1 - k] = uc12;
            data[7 * self.s + k] = uc13;

            let mut u8 = iq0;
            let mut v8 = siq0;
            u8 *= -T::D4_R19_ROT_TWIDDLE_16;
            v8 *= T::D4_R19_ROT_TWIDDLE_17;
            u8 = fmla(iq1, -T::D4_R19_ROT_TWIDDLE_1, u8);
            v8 = fmla(siq1, -T::D4_R19_ROT_TWIDDLE_0, v8);
            u8 = fmla(iq2, T::D4_R19_ROT_TWIDDLE_14, u8);
            v8 = fmla(siq2, T::D4_R19_ROT_TWIDDLE_15, v8);
            u8 = fmla(iq3, -T::D4_R19_ROT_TWIDDLE_2, u8);
            v8 = fmla(siq3, T::D4_R19_ROT_TWIDDLE_3, v8);
            u8 = fmla(iq4, T::D4_R19_ROT_TWIDDLE_4, u8);
            v8 = fmla(siq4, -T::D4_R19_ROT_TWIDDLE_5, v8);
            u8 = fmla(iq5, -T::D4_R19_ROT_TWIDDLE_10, u8);
            v8 = fmla(siq5, T::D4_R19_ROT_TWIDDLE_11, v8);
            u8 = fmla(iq6, -T::D4_R19_ROT_TWIDDLE_12, u8);
            v8 = fmla(siq6, -T::D4_R19_ROT_TWIDDLE_13, v8);
            u8 = fmla(iq7, T::D4_R19_ROT_TWIDDLE_9, u8);
            v8 = fmla(siq7, T::D4_R19_ROT_TWIDDLE_8, v8);
            u8 = fmla(iq8, -T::D4_R19_ROT_TWIDDLE_6, u8);
            v8 = fmla(siq8, -T::D4_R19_ROT_TWIDDLE_7, v8);
            u8 += a_v0;
            let uc14 = u8 - v8;
            let uc15 = u8 + v8;

            data[8 * self.s - 1 - k] = uc14;
            data[8 * self.s + k] = uc15;

            let mut u9 = iq0;
            let mut v9 = siq0;
            u9 *= T::D4_R19_ROT_TWIDDLE_12;
            v9 *= T::D4_R19_ROT_TWIDDLE_13;
            u9 = fmla(iq1, T::D4_R19_ROT_TWIDDLE_16, u9);
            v9 = fmla(siq1, -T::D4_R19_ROT_TWIDDLE_17, v9);
            u9 = fmla(iq2, T::D4_R19_ROT_TWIDDLE_10, u9);
            v9 = fmla(siq2, T::D4_R19_ROT_TWIDDLE_11, v9);
            u9 = fmla(iq3, -T::D4_R19_ROT_TWIDDLE_9, u9);
            v9 = fmla(siq3, -T::D4_R19_ROT_TWIDDLE_8, v9);
            u9 = fmla(iq4, T::D4_R19_ROT_TWIDDLE_1, u9);
            v9 = fmla(siq4, T::D4_R19_ROT_TWIDDLE_0, v9);
            u9 = fmla(iq5, -T::D4_R19_ROT_TWIDDLE_4, u9);
            v9 = fmla(siq5, -T::D4_R19_ROT_TWIDDLE_5, v9);
            u9 = fmla(iq6, T::D4_R19_ROT_TWIDDLE_6, u9);
            v9 = fmla(siq6, T::D4_R19_ROT_TWIDDLE_7, v9);
            u9 = fmla(iq7, -T::D4_R19_ROT_TWIDDLE_14, u9);
            v9 = fmla(siq7, -T::D4_R19_ROT_TWIDDLE_15, v9);
            u9 = fmla(iq8, T::D4_R19_ROT_TWIDDLE_2, u9);
            v9 = fmla(siq8, T::D4_R19_ROT_TWIDDLE_3, v9);
            u9 = u9 - a_v0;
            let uc16 = u9 - v9;
            let uc17 = u9 + v9;

            data[9 * self.s - 1 - k] = uc16;
            data[9 * self.s + k] = uc17;
        }
        Ok(())
    }
}

impl<T: DctSample + Dct4MixedRadix19Sample> PxdctExecutor<T> for Dct4MixedRadix19<T>
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

    #[inline]
    fn scratch_size(&self) -> usize {
        self.execution_length + self.inner_dct4_scratch_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dct4::Dct4Identity;
    use crate::tests::naive_dct4;
    use rand::Rng;

    #[test]
    fn test_split_dct4() {
        let mut input = vec![0.; 19];
        for z in input.iter_mut() {
            *z = rand::rng().random_range(1.0..2.0);
        }
        let reference_input = input.clone();
        let reference_input = naive_dct4(&reference_input);
        let bf = Dct4MixedRadix19::new(input.len(), Arc::new(Dct4Identity::default())).unwrap();
        bf.execute(&mut input).unwrap();
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
