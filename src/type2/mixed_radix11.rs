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
use crate::butterflies::MixedRadix9Sample;
use crate::mla::fmla;
use crate::type2::prime_butterflies::MixedRadix11Sample;
use crate::type2::util::{radixq_cos_twiddle, radixq_rotation_twiddle};
use crate::util::{DctSample, try_vec, validate_scratch};
use crate::{PxdctError, PxdctExecutor};
use num_complex::Complex;
use num_traits::{AsPrimitive, Zero};
use std::sync::Arc;

pub(crate) struct Dct2MixedRadix11<T> {
    rotation_layer: Vec<Complex<T>>,
    cos_twiddles: Vec<Complex<T>>,
    inner_dct: Arc<dyn PxdctExecutor<T> + Send + Sync>,
    execution_length: usize,
    q_modules: usize,
    inner_dct_scratch_length: usize,
}

impl<T: DctSample + MixedRadix9Sample> Dct2MixedRadix11<T>
where
    f64: AsPrimitive<T>,
    usize: AsPrimitive<T>,
{
    #[allow(unused)]
    pub(crate) fn new(
        len: usize,
        inner_dct: Arc<dyn PxdctExecutor<T> + Send + Sync>,
    ) -> Result<Dct2MixedRadix11<T>, PxdctError> {
        assert!(
            len.is_multiple_of(11),
            "Mixed radix 9 should not be called on sizes no divisible by 9"
        );

        let q_modules = len / 11;

        // always 4 inner groups in Radix-11
        let inner_groups = 5;

        // Precompute rotation twiddles for k≥1
        // Format: [m0_k1, m1_k1, m0_k2, m1_k2, ...]
        let mut rotation_layer = try_vec![Complex::<T>::zero(); inner_groups * (q_modules - 1)];
        for (k, rotation_layer) in rotation_layer.as_chunks_mut::<5>().0.iter_mut().enumerate() {
            for (m, layer) in rotation_layer.iter_mut().enumerate() {
                *layer =
                    radixq_rotation_twiddle(11, m, (k + 1).as_(), (q_modules - (k + 1)).as_(), len);
            }
        }

        // Precompute cosine twiddles for even components
        // Stored as Complex{re: even_twiddle, im: odd_twiddle} for cache efficiency
        let mut cos_twiddles = try_vec![Complex::<T>::zero(); (q_modules - 1) * inner_groups];
        for (k, k_layer) in cos_twiddles.as_chunks_mut::<5>().0.iter_mut().enumerate() {
            for (m, m_layer) in k_layer.iter_mut().enumerate() {
                let k = k + 1;
                let even = radixq_cos_twiddle(11, m, k.as_(), len);
                let odd = radixq_cos_twiddle(
                    11,
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

        let q_modules = len / 11;
        let inner_dct_scratch_length = inner_dct.scratch_size();

        Ok(Dct2MixedRadix11 {
            rotation_layer,
            inner_dct,
            cos_twiddles,
            execution_length: len,
            q_modules,
            inner_dct_scratch_length,
        })
    }
}

impl<T: DctSample + MixedRadix11Sample> Dct2MixedRadix11<T>
where
    f64: AsPrimitive<T>,
{
    fn execute_with_store<S: BidirectionalStore<T>>(
        &self,
        data: &mut S,
        scratch: &mut [T],
    ) -> Result<(), PxdctError> {
        let (scratch, inner_scratch) = scratch.split_at_mut(self.execution_length);
        let (a_buffer, c_s_buffer) = scratch.split_at_mut(self.q_modules);
        let (c_buffer, s_buffer) = c_s_buffer.split_at_mut(self.q_modules * 5);

        // Step 1: Decompose input into A (center), C (even-symmetric), S (odd-symmetric) buffers
        for (n, dst) in a_buffer.iter_mut().enumerate() {
            *dst = data[n * 11 + 5];
        }

        // Extract and combine symmetric pairs with sign alternation for S buffer
        for (m, (c_buffer, s_buffer)) in c_buffer
            .chunks_exact_mut(self.q_modules)
            .zip(s_buffer.chunks_exact_mut(self.q_modules))
            .enumerate()
        {
            let mut sign = T::one();
            for (n, (c_dst, s_dst)) in c_buffer.iter_mut().zip(s_buffer.iter_mut()).enumerate() {
                let u0 = data[11 * n + m];
                let u1 = data[11 * n + 11 - m - 1];

                *c_dst = u0 + u1;
                *s_dst = (u0 - u1).mulsign(sign);

                sign = -sign;
            }
        }

        // Step 2: Apply DCT-II to all buffers (A, C₀, C₁, S₀, S₁)
        self.inner_dct
            .execute_with_scratch(scratch, inner_scratch)?;

        let (a_buffer, c_s_buffer) = scratch.split_at_mut(self.q_modules);
        let (c_buffer, s_buffer) = c_s_buffer.split_at_mut(self.q_modules * 5);

        {
            // Step 3: Recombine transformed buffers with twiddle factors

            // Handle k=0 case (DC and low frequencies)
            let qc = c_buffer[0];
            let mut c0 = qc; // Component C₀ (position 0)
            let mut c1 = qc * T::R11_EVEN_TWIDDLE_0; // Component C₂ (position 2, uses j=2)
            let mut c2 = qc * T::R11_EVEN_TWIDDLE_4; // Component C₄ (position 4, uses j=4)
            let mut c3 = qc * T::R11_EVEN_TWIDDLE_1; // Component C6 (position 6, uses j=6)
            let mut c4 = qc * T::R11_EVEN_TWIDDLE_3; // Component C8 (position 8, uses j=8)
            let mut c5 = qc * T::R11_EVEN_TWIDDLE_2;

            let s0_twiddled = s_buffer[0];

            let mut s0 = s0_twiddled * T::R11_ODD_TWIDDLE_0;
            let mut s1 = s0_twiddled * -T::R11_ODD_TWIDDLE_1;
            let mut s2 = s0_twiddled * T::R11_ODD_TWIDDLE_2;
            let mut s3 = s0_twiddled * -T::R11_ODD_TWIDDLE_3;
            let mut s4 = s0_twiddled * T::R11_ODD_TWIDDLE_4;

            let ci = unsafe { *c_buffer.get_unchecked(self.q_modules) };
            let si = unsafe { *s_buffer.get_unchecked(self.q_modules) };

            let ci2 = unsafe { *c_buffer.get_unchecked(self.q_modules * 2) };
            let si2 = unsafe { *s_buffer.get_unchecked(self.q_modules * 2) };

            let ci3 = unsafe { *c_buffer.get_unchecked(self.q_modules * 3) };
            let si3 = unsafe { *s_buffer.get_unchecked(self.q_modules * 3) };

            let ci4 = unsafe { *c_buffer.get_unchecked(self.q_modules * 4) };
            let si4 = unsafe { *s_buffer.get_unchecked(self.q_modules * 4) };

            c0 = ci + c0 + ci2 + ci3 + ci4;

            let a0 = a_buffer[0];

            let dc = c0 + a0;
            data[0] = dc;

            c1 = fmla(ci, T::R11_EVEN_TWIDDLE_1, c1);
            c1 = fmla(ci2, T::R11_EVEN_TWIDDLE_2, c1);
            c1 = fmla(ci3, T::R11_EVEN_TWIDDLE_3, c1);
            c1 = fmla(ci4, T::R11_EVEN_TWIDDLE_4, c1);

            c2 = fmla(ci, T::R11_EVEN_TWIDDLE_2, c2);
            c2 = fmla(ci2, T::R11_EVEN_TWIDDLE_0, c2);
            c2 = fmla(ci3, T::R11_EVEN_TWIDDLE_1, c2);
            c2 = fmla(ci4, T::R11_EVEN_TWIDDLE_3, c2);

            let dc2 = c2 + a0;
            data[self.q_modules * 4] = dc2;

            c3 = fmla(ci, T::R11_EVEN_TWIDDLE_4, c3);
            c3 = fmla(ci2, T::R11_EVEN_TWIDDLE_3, c3);
            c3 = fmla(ci3, T::R11_EVEN_TWIDDLE_0, c3);
            c3 = fmla(ci4, T::R11_EVEN_TWIDDLE_2, c3);

            let dc3 = c3 + a0;
            data[self.q_modules * 6] = -dc3;

            c4 = fmla(ci, T::R11_EVEN_TWIDDLE_0, c4);
            c4 = fmla(ci2, T::R11_EVEN_TWIDDLE_4, c4);
            c4 = fmla(ci3, T::R11_EVEN_TWIDDLE_2, c4);
            c4 = fmla(ci4, T::R11_EVEN_TWIDDLE_1, c4);

            let dc4 = c4 + a0;
            data[self.q_modules * 8] = dc4;

            c5 = fmla(ci, T::R11_EVEN_TWIDDLE_3, c5);
            c5 = fmla(ci2, T::R11_EVEN_TWIDDLE_1, c5);
            c5 = fmla(ci3, T::R11_EVEN_TWIDDLE_4, c5);
            c5 = fmla(ci4, T::R11_EVEN_TWIDDLE_0, c5);

            let dc5 = c5 + a0;
            data[self.q_modules * 10] = -dc5;

            s0 = fmla(si, T::R11_ODD_TWIDDLE_1, s0);
            s0 = fmla(si2, T::R11_ODD_TWIDDLE_2, s0);
            s0 = fmla(si3, T::R11_ODD_TWIDDLE_3, s0);
            s0 = fmla(si4, T::R11_ODD_TWIDDLE_4, s0);

            s1 = fmla(si, -T::R11_ODD_TWIDDLE_4, s1);
            s1 = fmla(si2, T::R11_ODD_TWIDDLE_3, s1);
            s1 = fmla(si3, T::R11_ODD_TWIDDLE_0, s1);
            s1 = fmla(si4, T::R11_ODD_TWIDDLE_2, s1);

            s2 = fmla(si, -T::R11_ODD_TWIDDLE_3, s2);
            s2 = fmla(si2, -T::R11_ODD_TWIDDLE_1, s2);
            s2 = fmla(si3, T::R11_ODD_TWIDDLE_4, s2);
            s2 = fmla(si4, T::R11_ODD_TWIDDLE_0, s2);

            s3 = fmla(si, T::R11_ODD_TWIDDLE_0, s3);
            s3 = fmla(si2, -T::R11_ODD_TWIDDLE_4, s3);
            s3 = fmla(si3, -T::R11_ODD_TWIDDLE_2, s3);
            s3 = fmla(si4, T::R11_ODD_TWIDDLE_1, s3);

            s4 = fmla(si, -T::R11_ODD_TWIDDLE_2, s4);
            s4 = fmla(si2, T::R11_ODD_TWIDDLE_0, s4);
            s4 = fmla(si3, -T::R11_ODD_TWIDDLE_1, s4);
            s4 = fmla(si4, T::R11_ODD_TWIDDLE_3, s4);

            data[self.q_modules * 3] = -s1;
            data[self.q_modules] = s0;

            let qid2 = -(c1 + a0); // negated 2j
            data[self.q_modules * 2] = qid2;
            data[self.q_modules * 5] = s2;
            data[self.q_modules * 7] = -s3;
            data[self.q_modules * 9] = s4;

            // Step 4: Handle k≥1 cases with rotation twiddles
            for k in 1..self.q_modules {
                // Apply rotation twiddles to combine forward and inverted components
                let rotation_twiddle = unsafe { *self.rotation_layer.get_unchecked((k - 1) * 5) };

                let c_forward = unsafe { *c_buffer.get_unchecked(k) };
                let s_forward = unsafe { *s_buffer.get_unchecked(self.q_modules - k) };

                let rotated_dc = fmla(s_forward, rotation_twiddle.re, c_forward);

                let twiddle = unsafe { *self.cos_twiddles.get_unchecked((k - 1) * 5) };

                let twiddled_dc = rotated_dc * twiddle.re;

                let mut dc0 = twiddled_dc;
                let mut dc2 = twiddled_dc * T::R11_EVEN_TWIDDLE_0;
                let mut dc4 = twiddled_dc * T::R11_EVEN_TWIDDLE_4;
                let mut dc6 = twiddled_dc * T::R11_EVEN_TWIDDLE_1;
                let mut dc8 = twiddled_dc * T::R11_EVEN_TWIDDLE_3;
                let mut dc10 = twiddled_dc * T::R11_EVEN_TWIDDLE_2;

                let rotated_ds = fmla(c_forward, rotation_twiddle.im, s_forward);

                let twiddled_ds = rotated_ds * twiddle.im;

                let mut ds1 = twiddled_ds * T::R11_ODD_TWIDDLE_0;
                let mut ds3 = twiddled_ds * -T::R11_ODD_TWIDDLE_1;
                let mut ds5 = twiddled_ds * T::R11_ODD_TWIDDLE_2;
                let mut ds7 = twiddled_ds * -T::R11_ODD_TWIDDLE_3;
                let mut ds9 = twiddled_ds * T::R11_ODD_TWIDDLE_4;

                {
                    let c_forward = unsafe { *c_buffer.get_unchecked(self.q_modules + k) };
                    let s_forward = unsafe { *s_buffer.get_unchecked(self.q_modules * 2 - k) };

                    let rotation_twiddle =
                        unsafe { *self.rotation_layer.get_unchecked((k - 1) * 5 + 1) };

                    let twiddle = unsafe { *self.cos_twiddles.get_unchecked((k - 1) * 5 + 1) };

                    let rotated_dc1 = fmla(s_forward, rotation_twiddle.re, c_forward);
                    let rotated_ds2 = fmla(c_forward, rotation_twiddle.im, s_forward);

                    let twiddled_dc = twiddle.re * rotated_dc1;
                    let twiddled_ds = twiddle.im * rotated_ds2;

                    dc0 = twiddled_dc + dc0;
                    dc2 = fmla(twiddled_dc, T::R11_EVEN_TWIDDLE_1, dc2);
                    dc4 = fmla(twiddled_dc, T::R11_EVEN_TWIDDLE_2, dc4);
                    dc6 = fmla(twiddled_dc, T::R11_EVEN_TWIDDLE_4, dc6);
                    dc8 = fmla(twiddled_dc, T::R11_EVEN_TWIDDLE_0, dc8);
                    dc10 = fmla(twiddled_dc, T::R11_EVEN_TWIDDLE_3, dc10);

                    ds1 = fmla(twiddled_ds, T::R11_ODD_TWIDDLE_1, ds1);
                    ds3 = fmla(twiddled_ds, -T::R11_ODD_TWIDDLE_4, ds3);
                    ds5 = fmla(twiddled_ds, -T::R11_ODD_TWIDDLE_3, ds5);
                    ds7 = fmla(twiddled_ds, T::R11_ODD_TWIDDLE_0, ds7);
                    ds9 = fmla(twiddled_ds, -T::R11_ODD_TWIDDLE_2, ds9);
                }

                {
                    let c_forward = unsafe { *c_buffer.get_unchecked(self.q_modules * 2 + k) };
                    let s_forward = unsafe { *s_buffer.get_unchecked(self.q_modules * 3 - k) };

                    let rotation_twiddle =
                        unsafe { *self.rotation_layer.get_unchecked((k - 1) * 5 + 2) };

                    let twiddle = unsafe { *self.cos_twiddles.get_unchecked((k - 1) * 5 + 2) };

                    let rotated_dc1 = fmla(s_forward, rotation_twiddle.re, c_forward);
                    let rotated_ds2 = fmla(c_forward, rotation_twiddle.im, s_forward);

                    let twiddled_dc = twiddle.re * rotated_dc1;
                    let twiddled_ds = twiddle.im * rotated_ds2;

                    dc0 = twiddled_dc + dc0;
                    dc2 = fmla(twiddled_dc, T::R11_EVEN_TWIDDLE_2, dc2);
                    dc4 = fmla(twiddled_dc, T::R11_EVEN_TWIDDLE_0, dc4);
                    dc6 = fmla(twiddled_dc, T::R11_EVEN_TWIDDLE_3, dc6);
                    dc8 = fmla(twiddled_dc, T::R11_EVEN_TWIDDLE_4, dc8);
                    dc10 = fmla(twiddled_dc, T::R11_EVEN_TWIDDLE_1, dc10);

                    ds1 = fmla(twiddled_ds, T::R11_ODD_TWIDDLE_2, ds1);
                    ds3 = fmla(twiddled_ds, T::R11_ODD_TWIDDLE_3, ds3);
                    ds5 = fmla(twiddled_ds, -T::R11_ODD_TWIDDLE_1, ds5);
                    ds7 = fmla(twiddled_ds, -T::R11_ODD_TWIDDLE_4, ds7);
                    ds9 = fmla(twiddled_ds, T::R11_ODD_TWIDDLE_0, ds9);
                }

                {
                    let c_forward = unsafe { *c_buffer.get_unchecked(self.q_modules * 3 + k) };
                    let s_forward = unsafe { *s_buffer.get_unchecked(self.q_modules * 4 - k) };

                    let rotation_twiddle =
                        unsafe { *self.rotation_layer.get_unchecked((k - 1) * 5 + 3) };

                    let twiddle = unsafe { *self.cos_twiddles.get_unchecked((k - 1) * 5 + 3) };

                    let rotated_dc1 = fmla(s_forward, rotation_twiddle.re, c_forward);
                    let rotated_ds2 = fmla(c_forward, rotation_twiddle.im, s_forward);

                    let twiddled_dc = twiddle.re * rotated_dc1;
                    let twiddled_ds = twiddle.im * rotated_ds2;

                    dc0 = twiddled_dc + dc0;
                    dc2 = fmla(twiddled_dc, T::R11_EVEN_TWIDDLE_3, dc2);
                    dc4 = fmla(twiddled_dc, T::R11_EVEN_TWIDDLE_1, dc4);
                    dc6 = fmla(twiddled_dc, T::R11_EVEN_TWIDDLE_0, dc6);
                    dc8 = fmla(twiddled_dc, T::R11_EVEN_TWIDDLE_2, dc8);
                    dc10 = fmla(twiddled_dc, T::R11_EVEN_TWIDDLE_4, dc10);

                    ds1 = fmla(twiddled_ds, T::R11_ODD_TWIDDLE_3, ds1);
                    ds3 = fmla(twiddled_ds, T::R11_ODD_TWIDDLE_0, ds3);
                    ds5 = fmla(twiddled_ds, T::R11_ODD_TWIDDLE_4, ds5);
                    ds7 = fmla(twiddled_ds, -T::R11_ODD_TWIDDLE_2, ds7);
                    ds9 = fmla(twiddled_ds, -T::R11_ODD_TWIDDLE_1, ds9);
                }

                {
                    let c_forward = unsafe { *c_buffer.get_unchecked(self.q_modules * 4 + k) };
                    let s_forward = unsafe { *s_buffer.get_unchecked(self.q_modules * 5 - k) };

                    let rotation_twiddle =
                        unsafe { *self.rotation_layer.get_unchecked((k - 1) * 5 + 4) };

                    let twiddle = unsafe { *self.cos_twiddles.get_unchecked((k - 1) * 5 + 4) };

                    let rotated_dc1 = fmla(s_forward, rotation_twiddle.re, c_forward);
                    let rotated_ds2 = fmla(c_forward, rotation_twiddle.im, s_forward);

                    let twiddled_dc = twiddle.re * rotated_dc1;
                    let twiddled_ds = twiddle.im * rotated_ds2;

                    dc0 = twiddled_dc + dc0;
                    dc2 = fmla(twiddled_dc, T::R11_EVEN_TWIDDLE_4, dc2);
                    dc4 = fmla(twiddled_dc, T::R11_EVEN_TWIDDLE_3, dc4);
                    dc6 = fmla(twiddled_dc, T::R11_EVEN_TWIDDLE_2, dc6);
                    dc8 = fmla(twiddled_dc, T::R11_EVEN_TWIDDLE_1, dc8);
                    dc10 = fmla(twiddled_dc, T::R11_EVEN_TWIDDLE_0, dc10);

                    ds1 = fmla(twiddled_ds, T::R11_ODD_TWIDDLE_4, ds1);
                    ds3 = fmla(twiddled_ds, T::R11_ODD_TWIDDLE_2, ds3);
                    ds5 = fmla(twiddled_ds, T::R11_ODD_TWIDDLE_0, ds5);
                    ds7 = fmla(twiddled_ds, T::R11_ODD_TWIDDLE_1, ds7);
                    ds9 = fmla(twiddled_ds, T::R11_ODD_TWIDDLE_3, ds9);
                }

                let a0 = unsafe { *a_buffer.get_unchecked(k) };
                let dc = dc0 + a0;

                dc2 = -(dc2 + a0);
                dc4 += a0;
                dc6 += a0;
                dc8 += a0;
                dc10 += a0;

                data[k] = dc;

                let dss1 = fmla(2f64.as_(), ds1, -dc);
                data[self.q_modules * 2 - k] = dss1;

                dc2 = fmla(2f64.as_(), dc2, -dss1);
                data[self.q_modules * 2 + k] = dc2;

                let dss3 = fmla(2f64.as_(), -ds3, -dc2);
                data[self.q_modules * 4 - k] = dss3;

                let mdc4 = fmla(2f64.as_(), dc4, -dss3);
                data[self.q_modules * 4 + k] = mdc4;

                let dss5 = fmla(2f64.as_(), ds5, -mdc4);
                data[self.q_modules * 6 - k] = dss5;

                dc6 = fmla(2f64.as_(), -dc6, -dss5);
                data[self.q_modules * 6 + k] = dc6;

                let dss6 = fmla(2f64.as_(), -ds7, -dc6);
                data[self.q_modules * 8 - k] = dss6;

                dc8 = fmla(2f64.as_(), dc8, -dss6);
                data[self.q_modules * 8 + k] = dc8;

                let dss9 = fmla(2f64.as_(), ds9, -dc8);
                data[self.q_modules * 10 - k] = dss9;

                dc10 = fmla(2f64.as_(), -dc10, -dss9);
                data[self.q_modules * 10 + k] = dc10;
            }
        }
        Ok(())
    }
}

impl<T: DctSample + MixedRadix11Sample> PxdctExecutor<T> for Dct2MixedRadix11<T>
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
        self.execution_length + self.inner_dct_scratch_length
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Pxdct;
    use crate::tests::naive_dct2_f32;

    #[test]
    fn test_radix9_dct() {
        let mut input = vec![0.; 11];
        for (i, z) in input.iter_mut().enumerate() {
            *z = i as f32 + rand::random::<f32>() * 10.0;
        }
        let mut input = vec![
            7.6871257, 1.2637726, 11.096954, 4.113755, 6.3156953, 12.343984, 9.859292, 15.516256,
            12.010594, 7.6871257, 1.2637726, 7.6871257, 1.2637726, 11.096954, 4.113755, 6.3156953,
            12.343984, 9.859292, 15.516256, 12.010594, 7.6871257, 1.2637726, 7.6871257, 1.2637726,
            11.096954, 4.113755, 6.3156953, 12.343984, 9.859292, 15.516256, 12.010594, 7.6871257,
            1.2637726,
        ];
        let mut reference_input = input.clone();
        // let rr = Pxdct::make_dct2_f32(25).unwrap();
        // rr.execute(&mut reference_input).unwrap();
        reference_input = naive_dct2_f32(&reference_input);
        let bf =
            Dct2MixedRadix11::new(input.len(), Pxdct::make_dct2_f32(input.len() / 11).unwrap())
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
