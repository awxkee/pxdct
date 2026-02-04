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
use crate::avx::AvxDct2Butterfly9;
use crate::avx::dct2::mixed_radix9d::{
    dct2_radix9_cos_twiddles_avxd, dct2_radix9_rotation_twiddles_avxd,
};
use crate::avx::stored::AvxStoreD;
use crate::avx::util::fma;
use crate::bidirectional::{BidirectionalStore, InPlaceStore};
use crate::butterflies::MixedRadix9Sample;
use crate::factory_dct2::Dct2Factory;
use crate::util::{DctConstants, DctSample, mixed_radix_inner_twiddle};
use crate::{PxdctError, PxdctExecutor};
use num_traits::{One, Zero};
use std::sync::Arc;

#[target_feature(enable = "avx2")]
pub(crate) fn dct2_radix3_avx_groupd(len: usize) -> Vec<AvxStoreD> {
    let inner_layer_groups = len / 3;
    let simd_groups = inner_layer_groups.div_ceil(4);
    let mut inner_layer = Vec::with_capacity(simd_groups * 4);

    for g in 0..simd_groups {
        let mut re0 = [0.; 4];
        let mut im0 = [0.; 4];
        let mut re1 = [0.; 4];
        let mut im1 = [0.; 4];

        for lane in 0..4 {
            let i = g * 4 + lane;
            if i < inner_layer_groups {
                let t0 = mixed_radix_inner_twiddle(2.0 * (i as f64) + 1.0, len);
                let t1 = mixed_radix_inner_twiddle(2.0 * (2.0 * (i as f64) + 1.0), len);

                re0[lane] = t0.re;
                im0[lane] = t0.im * f64::SQRT_3;

                re1[lane] = t1.re;
                im1[lane] = -t1.im * f64::SQRT_3;
            }
        }

        inner_layer.push(AvxStoreD::load(&re0));
        inner_layer.push(AvxStoreD::load(&im0));
        inner_layer.push(AvxStoreD::load(&re1));
        inner_layer.push(AvxStoreD::load(&im1));
    }
    inner_layer
}

#[derive(Debug, Clone)]
pub(crate) struct AvxDct2Butterfly27d {
    bf9: AvxDct2Butterfly9<f64>,
    inner_layer: [AvxStoreD; 12],
}

impl Default for AvxDct2Butterfly27d {
    fn default() -> Self {
        Self {
            bf9: AvxDct2Butterfly9::default(),
            inner_layer: unsafe { dct2_radix3_avx_groupd(27).try_into().unwrap() },
        }
    }
}

impl AvxDct2Butterfly27d {
    #[inline(always)]
    fn exec<S: BidirectionalStore<f64>>(
        &self,
        data: &mut S,
        a_buffer: &mut [f64; 9],
        b_buffer: &mut [f64; 9],
        c_buffer: &mut [f64; 9],
    ) {
        unsafe {
            for i in 0..2 {
                let j = i * 4;
                let ai = AvxStoreD::load(data.slice_from(j..));
                let bi = AvxStoreD::load(data.slice_from(18 + j..));
                let mut ci = AvxStoreD::load(data.slice_from(18 - j - 4..));
                ci = ci.reverse();

                let cos_sin_ai_re = self.inner_layer[j];
                let cos_sin_ai_im = self.inner_layer[j + 1];
                let cos_sin_2ai_re = self.inner_layer[j + 2];
                let cos_sin_2ai_im = self.inner_layer[j + 3];

                let bici = bi + ci;
                let a_comp = ai + bici;
                let second_layer_comp0 = AvxStoreD::f64_mul_add(2., ai, -bici);

                let d_ci_bi = ci - bi;

                let b0_b = fma(second_layer_comp0, cos_sin_ai_re, d_ci_bi * cos_sin_ai_im);
                let c0_b = fma(second_layer_comp0, cos_sin_2ai_re, d_ci_bi * cos_sin_2ai_im);

                a_comp.write(a_buffer.get_unchecked_mut(j..));
                b0_b.write(b_buffer.get_unchecked_mut(j..));
                c0_b.write(c_buffer.get_unchecked_mut(j..));
            }

            {
                let i = 2;
                let j = i * 4;
                let ai = AvxStoreD::load1(data.slice_from(j..));
                let bi = AvxStoreD::load1(data.slice_from(18 + j..));
                let ci = AvxStoreD::load1(data.slice_from(18 - j - 1..));

                let cos_sin_ai_re = self.inner_layer[j];
                let cos_sin_ai_im = self.inner_layer[j + 1];
                let cos_sin_2ai_re = self.inner_layer[j + 2];
                let cos_sin_2ai_im = self.inner_layer[j + 3];

                let bici = bi + ci;
                let a_comp = ai + bici;
                let second_layer_comp0 = AvxStoreD::f64_mul_add(2., ai, -bici);

                let d_ci_bi = ci - bi;

                let b0_b = fma(second_layer_comp0, cos_sin_ai_re, d_ci_bi * cos_sin_ai_im);
                let c0_b = fma(second_layer_comp0, cos_sin_2ai_re, d_ci_bi * cos_sin_2ai_im);

                a_comp.write1(a_buffer.get_unchecked_mut(j..));
                b0_b.write1(b_buffer.get_unchecked_mut(j..));
                c0_b.write1(c_buffer.get_unchecked_mut(j..));
            }

            self.bf9.exec(&mut InPlaceStore::new(a_buffer));
            self.bf9.exec(&mut InPlaceStore::new(b_buffer));
            self.bf9.exec(&mut InPlaceStore::new(c_buffer));

            data[0] = a_buffer[0];
            let b0 = b_buffer[0] * f64::HALF;
            data[1] = b0;
            let c0 = c_buffer[0] * f64::HALF;
            data[2] = c0;

            let mut last_b = c0;
            let mut last_c = b0;

            for k in 1..9 {
                data[3 * k] = a_buffer[k];

                let deferred_c = b_buffer[k] - last_b;
                data[3 * k + 1] = deferred_c;

                last_b = c_buffer[k] - last_c;
                data[3 * k + 2] = last_b;
                last_c = deferred_c;
            }
        }
    }
}

impl AvxDct2Butterfly27d {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_impl(&self, data: &mut [f64]) -> Result<(), PxdctError> {
        if !data.len().is_multiple_of(27) {
            return Err(PxdctError::InvalidSizeMultiplier(data.len(), self.length()));
        }

        let mut a_buffer = [0f64; 9];
        let mut b_buffer = [0f64; 9];
        let mut c_buffer = [0f64; 9];

        for chunk in data.chunks_exact_mut(27) {
            self.exec(
                &mut InPlaceStore::new(chunk),
                &mut a_buffer,
                &mut b_buffer,
                &mut c_buffer,
            );
        }
        Ok(())
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_into_impl(&self, input: &[f64], output: &mut [f64]) -> Result<(), PxdctError> {
        use crate::util::validate_oof_sizes;
        validate_oof_sizes!(input, output, 27);

        let mut a_buffer = [0f64; 9];
        let mut b_buffer = [0f64; 9];
        let mut c_buffer = [0f64; 9];

        use crate::bidirectional::BiStore;
        for (src, dst) in input.chunks_exact(27).zip(output.chunks_exact_mut(27)) {
            self.exec(
                &mut BiStore::new(src, dst),
                &mut a_buffer,
                &mut b_buffer,
                &mut c_buffer,
            );
        }
        Ok(())
    }
}

impl PxdctExecutor<f64> for AvxDct2Butterfly27d {
    fn execute(&self, data: &mut [f64]) -> Result<(), PxdctError> {
        unsafe { self.execute_impl(data) }
    }

    fn execute_with_scratch(&self, data: &mut [f64], _: &mut [f64]) -> Result<(), PxdctError> {
        unsafe { self.execute_impl(data) }
    }

    fn execute_into(&self, input: &[f64], output: &mut [f64]) -> Result<(), PxdctError> {
        unsafe { self.execute_into_impl(input, output) }
    }

    fn execute_into_with_scratch(
        &self,
        input: &[f64],
        output: &mut [f64],
        _: &mut [f64],
    ) -> Result<(), PxdctError> {
        unsafe { self.execute_into_impl(input, output) }
    }

    fn length(&self) -> usize {
        27
    }

    fn scratch_size(&self) -> usize {
        0
    }
}

#[derive(Clone)]
pub(crate) struct AvxDct2Butterfly81d {
    bf27: Arc<dyn PxdctExecutor<f64> + Send + Sync>,
    inner_layer: [AvxStoreD; 28],
}

impl Default for AvxDct2Butterfly81d {
    fn default() -> Self {
        Self {
            bf27: f64::dct2_butterfly27(),
            inner_layer: unsafe { dct2_radix3_avx_groupd(81).try_into().unwrap() },
        }
    }
}

impl AvxDct2Butterfly81d {
    #[inline(always)]
    fn exec<S: BidirectionalStore<f64>>(
        &self,
        data: &mut S,
        a_buffer: &mut [f64; 27],
        b_buffer: &mut [f64; 27],
        c_buffer: &mut [f64; 27],
    ) {
        unsafe {
            for i in 0..6 {
                let j = i * 4;
                let ai = AvxStoreD::load(data.slice_from(j..));
                let bi = AvxStoreD::load(data.slice_from(54 + j..));
                let mut ci = AvxStoreD::load(data.slice_from(54 - j - 4..));
                ci = ci.reverse();

                let cos_sin_ai_re = self.inner_layer[j];
                let cos_sin_ai_im = self.inner_layer[j + 1];
                let cos_sin_2ai_re = self.inner_layer[j + 2];
                let cos_sin_2ai_im = self.inner_layer[j + 3];

                let bici = bi + ci;
                let a_comp = ai + bici;
                let second_layer_comp0 = AvxStoreD::f64_mul_add(2., ai, -bici);

                let d_ci_bi = ci - bi;

                let b0_b = fma(second_layer_comp0, cos_sin_ai_re, d_ci_bi * cos_sin_ai_im);
                let c0_b = fma(second_layer_comp0, cos_sin_2ai_re, d_ci_bi * cos_sin_2ai_im);

                a_comp.write(a_buffer.get_unchecked_mut(j..));
                b0_b.write(b_buffer.get_unchecked_mut(j..));
                c0_b.write(c_buffer.get_unchecked_mut(j..));
            }

            {
                let i = 6;
                let j = i * 4;
                let ai = AvxStoreD::load3(data.slice_from(j..));
                let bi = AvxStoreD::load3(data.slice_from(54 + j..));
                let mut ci = AvxStoreD::load3(data.slice_from(54 - j - 3..));
                ci = ci.reverse3();

                let cos_sin_ai_re = self.inner_layer[j];
                let cos_sin_ai_im = self.inner_layer[j + 1];
                let cos_sin_2ai_re = self.inner_layer[j + 2];
                let cos_sin_2ai_im = self.inner_layer[j + 3];

                let bici = bi + ci;
                let a_comp = ai + bici;
                let second_layer_comp0 = AvxStoreD::f64_mul_add(2., ai, -bici);

                let d_ci_bi = ci - bi;

                let b0_b = fma(second_layer_comp0, cos_sin_ai_re, d_ci_bi * cos_sin_ai_im);
                let c0_b = fma(second_layer_comp0, cos_sin_2ai_re, d_ci_bi * cos_sin_2ai_im);

                a_comp.write3(a_buffer.get_unchecked_mut(j..));
                b0_b.write3(b_buffer.get_unchecked_mut(j..));
                c0_b.write3(c_buffer.get_unchecked_mut(j..));
            }

            _ = self.bf27.execute(a_buffer);
            _ = self.bf27.execute(b_buffer);
            _ = self.bf27.execute(c_buffer);

            data[0] = a_buffer[0];
            let b0 = b_buffer[0] * f64::HALF;
            data[1] = b0;
            let c0 = c_buffer[0] * f64::HALF;
            data[2] = c0;

            let mut last_b = c0;
            let mut last_c = b0;

            for k in 1..27 {
                data[3 * k] = a_buffer[k];

                let deferred_c = b_buffer[k] - last_b;
                data[3 * k + 1] = deferred_c;

                last_b = c_buffer[k] - last_c;
                data[3 * k + 2] = last_b;
                last_c = deferred_c;
            }
        }
    }
}

impl AvxDct2Butterfly81d {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_impl(&self, data: &mut [f64]) -> Result<(), PxdctError> {
        if !data.len().is_multiple_of(81) {
            return Err(PxdctError::InvalidSizeMultiplier(data.len(), self.length()));
        }

        let mut a_buffer = [0f64; 27];
        let mut b_buffer = [0f64; 27];
        let mut c_buffer = [0f64; 27];

        for chunk in data.chunks_exact_mut(81) {
            self.exec(
                &mut InPlaceStore::new(chunk),
                &mut a_buffer,
                &mut b_buffer,
                &mut c_buffer,
            );
        }
        Ok(())
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_into_impl(&self, input: &[f64], output: &mut [f64]) -> Result<(), PxdctError> {
        use crate::util::validate_oof_sizes;
        validate_oof_sizes!(input, output, 81);

        let mut a_buffer = [0f64; 27];
        let mut b_buffer = [0f64; 27];
        let mut c_buffer = [0f64; 27];

        use crate::bidirectional::BiStore;
        for (src, dst) in input.chunks_exact(81).zip(output.chunks_exact_mut(81)) {
            self.exec(
                &mut BiStore::new(src, dst),
                &mut a_buffer,
                &mut b_buffer,
                &mut c_buffer,
            );
        }
        Ok(())
    }
}

impl PxdctExecutor<f64> for AvxDct2Butterfly81d {
    fn execute(&self, data: &mut [f64]) -> Result<(), PxdctError> {
        unsafe { self.execute_impl(data) }
    }

    fn execute_with_scratch(&self, data: &mut [f64], _: &mut [f64]) -> Result<(), PxdctError> {
        unsafe { self.execute_impl(data) }
    }

    fn execute_into(&self, input: &[f64], output: &mut [f64]) -> Result<(), PxdctError> {
        unsafe { self.execute_into_impl(input, output) }
    }

    fn execute_into_with_scratch(
        &self,
        input: &[f64],
        output: &mut [f64],
        _: &mut [f64],
    ) -> Result<(), PxdctError> {
        unsafe { self.execute_into_impl(input, output) }
    }

    fn length(&self) -> usize {
        81
    }

    fn scratch_size(&self) -> usize {
        0
    }
}

#[derive(Clone)]
pub(crate) struct AvxDct2Butterfly243d {
    bf27: Arc<dyn PxdctExecutor<f64> + Send + Sync>,
    rotation_layer: [AvxStoreD; 56],
    cos_twiddles: [AvxStoreD; 56],
}

impl Default for AvxDct2Butterfly243d {
    fn default() -> Self {
        // always 4 inner groups in Radix-9

        // Precompute rotation twiddles for k≥1
        // Format: [m0_k1, m1_k1, m0_k2, m1_k2, ...]
        let rotation_layer = unsafe { dct2_radix9_rotation_twiddles_avxd(243 / 9, 243) };

        // Precompute cosine twiddles for even components
        // Stored as Complex{re: even_twiddle, im: odd_twiddle} for cache efficiency
        let cos_twiddles = unsafe { dct2_radix9_cos_twiddles_avxd(243 / 9, 243) };

        Self {
            bf27: f64::dct2_butterfly27(),
            rotation_layer: rotation_layer.try_into().unwrap(),
            cos_twiddles: cos_twiddles.try_into().unwrap(),
        }
    }
}

impl AvxDct2Butterfly243d {
    #[inline(always)]
    fn exec<S: BidirectionalStore<f64>>(
        &self,
        data: &mut S,
        a_buffer: &mut [f64; 27],
        c_buffer: &mut [f64; 108],
        s_buffer: &mut [f64; 108],
    ) {
        unsafe {
            for n in 0..27 {
                a_buffer[n] = data[n * 9 + 4];
            }

            _ = self.bf27.execute(a_buffer);

            for (m, (c_buffer, s_buffer)) in c_buffer
                .chunks_exact_mut(27)
                .zip(s_buffer.chunks_exact_mut(27))
                .enumerate()
            {
                let mut sign = f64::one();
                for (n, (c_dst, s_dst)) in c_buffer.iter_mut().zip(s_buffer.iter_mut()).enumerate()
                {
                    let u0 = data[9 * n + m];
                    let u1 = data[9 * n + 9 - m - 1];

                    *c_dst = u0 + u1;
                    *s_dst = (u0 - u1).mulsign(sign);

                    sign = -sign;
                }

                _ = self.bf27.execute(c_buffer);
                _ = self.bf27.execute(s_buffer);
            }

            let q_modules = 27;

            // Step 3: Recombine transformed buffers with twiddle factors

            // Handle k=0 case (DC and low frequencies)
            let qc = c_buffer[0];
            let mut c0 = qc; // Component C₀ (position 0)
            let mut c1 = qc * f64::R9_EVEN_TWIDDLE_0; // Component C₂ (position 2, uses j=2)
            let mut c2 = qc * f64::R9_EVEN_TWIDDLE_2; // Component C₄ (position 4, uses j=4)
            let mut c3 = qc * -f64::HALF; // Component C6 (position 6, uses j=6)
            let mut c4 = qc * f64::R9_EVEN_TWIDDLE_1; // Component C8 (position 8, uses j=8)

            let s0_twiddled = s_buffer[0];

            let mut s0 = s0_twiddled * f64::R9_ODD_TWIDDLE_0;
            let mut s1 = s0_twiddled * -f64::R9_ODD_TWIDDLE_1;
            let mut s2 = s0_twiddled * f64::R9_ODD_TWIDDLE_2;
            let mut s3 = s0_twiddled * -f64::R9_ODD_TWIDDLE_3;

            let ci = c_buffer[q_modules];
            let si = s_buffer[q_modules];

            let ci2 = c_buffer[q_modules * 2];
            let si2 = s_buffer[q_modules * 2];

            let ci3 = c_buffer[q_modules * 3];
            let si3 = s_buffer[q_modules * 3];

            c0 = ci + c0 + ci2 + ci3;

            let a0 = a_buffer[0];

            let dc = c0 + a0;
            data[0] = dc;

            c1 = fma(ci, -f64::HALF, c1);
            c1 = fma(ci2, f64::R9_EVEN_TWIDDLE_1, c1);
            c1 = fma(ci3, f64::R9_EVEN_TWIDDLE_2, c1);

            c2 = fma(ci, -f64::HALF, c2);
            c2 = fma(ci2, f64::R9_EVEN_TWIDDLE_0, c2);
            c2 = fma(ci3, f64::R9_EVEN_TWIDDLE_1, c2);

            let dc2 = c2 + a0;
            data[q_modules * 4] = dc2;

            c3 += ci;
            c3 = fma(ci2, -f64::HALF, c3);
            c3 = fma(ci3, -f64::HALF, c3);

            let dc3 = c3 + a0;
            data[q_modules * 6] = -dc3;

            c4 = fma(ci, -f64::HALF, c4);
            c4 = fma(ci2, f64::R9_EVEN_TWIDDLE_2, c4);
            c4 = fma(ci3, f64::R9_EVEN_TWIDDLE_0, c4);

            data[q_modules * 8] = c4 + a0;

            s0 = fma(si, f64::R9_ODD_TWIDDLE_1, s0);
            s0 = fma(si2, f64::R9_ODD_TWIDDLE_2, s0);
            s0 = fma(si3, f64::R9_ODD_TWIDDLE_3, s0);

            s1 = fma(si2, f64::R9_ODD_TWIDDLE_1, s1);
            s1 = fma(si3, f64::R9_ODD_TWIDDLE_1, s1);

            s2 = fma(si, -f64::R9_ODD_TWIDDLE_1, s2);
            s2 = fma(si2, -f64::R9_ODD_TWIDDLE_3, s2);
            s2 = fma(si3, f64::R9_ODD_TWIDDLE_0, s2);

            s3 = fma(si, f64::R9_ODD_TWIDDLE_1, s3);
            s3 = fma(si2, -f64::R9_ODD_TWIDDLE_0, s3);
            s3 = fma(si3, f64::R9_ODD_TWIDDLE_2, s3);

            data[q_modules * 3] = -s1;
            data[q_modules] = s0;
            data[q_modules * 2] = -(c1 + a0);
            data[q_modules * 5] = s2;
            data[q_modules * 7] = -s3;

            let mut k = 1usize;
            let mut uk = 0usize;
            while k + 4 <= q_modules {
                // Apply rotation twiddles to combine forward and inverted components
                let rotation_twiddle_re = *self.rotation_layer.get_unchecked(uk);
                let rotation_twiddle_im = *self.rotation_layer.get_unchecked(uk + 1);

                let c_forward = AvxStoreD::load(c_buffer.get_unchecked(k..));
                let s_forward =
                    AvxStoreD::load(s_buffer.get_unchecked(q_modules - k - 3..)).reverse();

                let rotated_dc = fma(s_forward, rotation_twiddle_re, c_forward);

                let twiddle_re = *self.cos_twiddles.get_unchecked(uk);
                let twiddle_im = *self.cos_twiddles.get_unchecked(uk + 1);

                let twiddled_dc = rotated_dc * twiddle_re;

                let mut dc0 = twiddled_dc;
                let mut dc2 = twiddled_dc * f64::R9_EVEN_TWIDDLE_0;
                let mut dc4 = twiddled_dc * f64::R9_EVEN_TWIDDLE_2;
                let mut dc6 = twiddled_dc * -f64::HALF;
                let mut dc8 = twiddled_dc * f64::R9_EVEN_TWIDDLE_1;

                let rotated_ds = fma(c_forward, rotation_twiddle_im, s_forward);

                let twiddled_ds = rotated_ds * twiddle_im;

                let mut ds1 = twiddled_ds * f64::R9_ODD_TWIDDLE_0;
                let mut ds3 = twiddled_ds * -f64::R9_ODD_TWIDDLE_1;
                let mut ds5 = twiddled_ds * f64::R9_ODD_TWIDDLE_2;
                let mut ds7 = twiddled_ds * -f64::R9_ODD_TWIDDLE_3;

                {
                    let c_forward = AvxStoreD::load(c_buffer.get_unchecked(q_modules + k..));
                    let s_forward =
                        AvxStoreD::load(s_buffer.get_unchecked(q_modules * 2 - k - 3..)).reverse();

                    let rotation_twiddle_re = *self.rotation_layer.get_unchecked(uk + 2);
                    let rotation_twiddle_im = *self.rotation_layer.get_unchecked(uk + 3);

                    let twiddle_re = *self.cos_twiddles.get_unchecked(uk + 2);
                    let twiddle_im = *self.cos_twiddles.get_unchecked(uk + 3);

                    let rotated_dc1 = fma(s_forward, rotation_twiddle_re, c_forward);
                    let rotated_ds2 = fma(c_forward, rotation_twiddle_im, s_forward);

                    let twiddled_dc = twiddle_re * rotated_dc1;
                    let twiddled_ds = twiddle_im * rotated_ds2;

                    dc0 = twiddled_dc + dc0;
                    dc2 = AvxStoreD::f64_mul_add(-f64::HALF, twiddled_dc, dc2);
                    dc4 = AvxStoreD::f64_mul_add(-f64::HALF, twiddled_dc, dc4);
                    dc6 = twiddled_dc + dc6;
                    dc8 = AvxStoreD::f64_mul_add(-f64::HALF, twiddled_dc, dc8);

                    ds1 = AvxStoreD::f64_mul_add(f64::R9_ODD_TWIDDLE_1, twiddled_ds, ds1);
                    ds5 = AvxStoreD::f64_mul_add(-f64::R9_ODD_TWIDDLE_1, twiddled_ds, ds5);
                    ds7 = AvxStoreD::f64_mul_add(f64::R9_ODD_TWIDDLE_1, twiddled_ds, ds7);
                }

                {
                    let c_forward = AvxStoreD::load(c_buffer.get_unchecked(q_modules * 2 + k..));
                    let s_forward =
                        AvxStoreD::load(s_buffer.get_unchecked(q_modules * 3 - k - 3..)).reverse();

                    let rotation_twiddle_re = *self.rotation_layer.get_unchecked(uk + 4);
                    let rotation_twiddle_im = *self.rotation_layer.get_unchecked(uk + 5);

                    let twiddle_re = *self.cos_twiddles.get_unchecked(uk + 4);
                    let twiddle_im = *self.cos_twiddles.get_unchecked(uk + 5);

                    let rotated_dc1 = fma(s_forward, rotation_twiddle_re, c_forward);
                    let rotated_ds2 = fma(c_forward, rotation_twiddle_im, s_forward);

                    let twiddled_dc = twiddle_re * rotated_dc1;
                    let twiddled_ds = twiddle_im * rotated_ds2;

                    dc0 = twiddled_dc + dc0;
                    dc2 = AvxStoreD::f64_mul_add(f64::R9_EVEN_TWIDDLE_1, twiddled_dc, dc2);
                    dc4 = AvxStoreD::f64_mul_add(f64::R9_EVEN_TWIDDLE_0, twiddled_dc, dc4);
                    dc6 = AvxStoreD::f64_mul_add(-f64::HALF, twiddled_dc, dc6);
                    dc8 = AvxStoreD::f64_mul_add(f64::R9_EVEN_TWIDDLE_2, twiddled_dc, dc8);

                    ds1 = AvxStoreD::f64_mul_add(f64::R9_ODD_TWIDDLE_2, twiddled_ds, ds1);
                    ds3 = AvxStoreD::f64_mul_add(f64::R9_ODD_TWIDDLE_1, twiddled_ds, ds3);
                    ds5 = AvxStoreD::f64_mul_add(-f64::R9_ODD_TWIDDLE_3, twiddled_ds, ds5);
                    ds7 = AvxStoreD::f64_mul_add(-f64::R9_ODD_TWIDDLE_0, twiddled_ds, ds7);
                }

                {
                    let c_forward = AvxStoreD::load(c_buffer.get_unchecked(q_modules * 3 + k..));
                    let s_forward =
                        AvxStoreD::load(s_buffer.get_unchecked(q_modules * 4 - k - 3..)).reverse();

                    let rotation_twiddle_re = *self.rotation_layer.get_unchecked(uk + 6);
                    let rotation_twiddle_im = *self.rotation_layer.get_unchecked(uk + 7);

                    let twiddle_re = *self.cos_twiddles.get_unchecked(uk + 6);
                    let twiddle_im = *self.cos_twiddles.get_unchecked(uk + 7);

                    let rotated_dc1 = fma(s_forward, rotation_twiddle_re, c_forward);
                    let rotated_ds2 = fma(c_forward, rotation_twiddle_im, s_forward);

                    let twiddled_dc = twiddle_re * rotated_dc1;
                    let twiddled_ds = twiddle_im * rotated_ds2;

                    dc0 = twiddled_dc + dc0;
                    dc2 = AvxStoreD::f64_mul_add(f64::R9_EVEN_TWIDDLE_2, twiddled_dc, dc2);
                    dc4 = AvxStoreD::f64_mul_add(f64::R9_EVEN_TWIDDLE_1, twiddled_dc, dc4);
                    dc6 = AvxStoreD::f64_mul_add(-f64::HALF, twiddled_dc, dc6);
                    dc8 = AvxStoreD::f64_mul_add(f64::R9_EVEN_TWIDDLE_0, twiddled_dc, dc8);

                    ds1 = AvxStoreD::f64_mul_add(f64::R9_ODD_TWIDDLE_3, twiddled_ds, ds1);
                    ds3 = AvxStoreD::f64_mul_add(f64::R9_ODD_TWIDDLE_1, twiddled_ds, ds3);
                    ds5 = AvxStoreD::f64_mul_add(f64::R9_ODD_TWIDDLE_0, twiddled_ds, ds5);
                    ds7 = AvxStoreD::f64_mul_add(f64::R9_ODD_TWIDDLE_2, twiddled_ds, ds7);
                }

                let a0 = AvxStoreD::load(a_buffer.get_unchecked(k..));
                let dc = dc0 + a0;

                dc2 = -(dc2 + a0);
                dc4 += a0;
                dc6 += a0;
                dc8 += a0;

                dc.write(data.slice_from_mut(k..));

                let dss1 = AvxStoreD::f64_mul_add(2., ds1, -dc);
                dss1.reverse()
                    .write(data.slice_from_mut(q_modules * 2 - k - 3..));

                dc2 = AvxStoreD::f64_mul_add(2., dc2, -dss1);
                dc2.write(data.slice_from_mut(q_modules * 2 + k..));

                let dss3 = AvxStoreD::f64_mul_add(2., -ds3, -dc2);
                dss3.reverse()
                    .write(data.slice_from_mut(q_modules * 4 - k - 3..));

                let mdc4 = AvxStoreD::f64_mul_add(2., dc4, -dss3);
                mdc4.write(data.slice_from_mut(q_modules * 4 + k..));

                let dss5 = AvxStoreD::f64_mul_add(2., ds5, -mdc4);
                dss5.reverse()
                    .write(data.slice_from_mut(q_modules * 6 - k - 3..));

                dc6 = AvxStoreD::f64_mul_add(2., -dc6, -dss5);

                dc6.write(data.slice_from_mut(q_modules * 6 + k..));

                let dss6 = AvxStoreD::f64_mul_add(2., -ds7, -dc6);
                dss6.reverse()
                    .write(data.slice_from_mut(q_modules * 8 - k - 3..));

                dc8 = AvxStoreD::f64_mul_add(2., dc8, -dss6);
                dc8.write(data.slice_from_mut(q_modules * 8 + k..));

                k += 4;
                uk += 8;
            }

            // Apply rotation twiddles to combine forward and inverted components
            let rotation_twiddle_re = *self.rotation_layer.get_unchecked(uk);
            let rotation_twiddle_im = *self.rotation_layer.get_unchecked(uk + 1);

            let c_forward = AvxStoreD::load2(c_buffer.get_unchecked(k..));
            let s_forward =
                AvxStoreD::load2(s_buffer.get_unchecked(q_modules - k - 1..)).reverse2();

            let rotated_dc = fma(s_forward, rotation_twiddle_re, c_forward);

            let twiddle_re = *self.cos_twiddles.get_unchecked(uk);
            let twiddle_im = *self.cos_twiddles.get_unchecked(uk + 1);

            let twiddled_dc = rotated_dc * twiddle_re;

            let mut dc0 = twiddled_dc;
            let mut dc2 = twiddled_dc * f64::R9_EVEN_TWIDDLE_0;
            let mut dc4 = twiddled_dc * f64::R9_EVEN_TWIDDLE_2;
            let mut dc6 = twiddled_dc * -f64::HALF;
            let mut dc8 = twiddled_dc * f64::R9_EVEN_TWIDDLE_1;

            let rotated_ds = fma(c_forward, rotation_twiddle_im, s_forward);

            let twiddled_ds = rotated_ds * twiddle_im;

            let mut ds1 = twiddled_ds * f64::R9_ODD_TWIDDLE_0;
            let mut ds3 = twiddled_ds * -f64::R9_ODD_TWIDDLE_1;
            let mut ds5 = twiddled_ds * f64::R9_ODD_TWIDDLE_2;
            let mut ds7 = twiddled_ds * -f64::R9_ODD_TWIDDLE_3;

            {
                let c_forward = AvxStoreD::load2(c_buffer.get_unchecked(q_modules + k..));
                let s_forward =
                    AvxStoreD::load2(s_buffer.get_unchecked(q_modules * 2 - k - 1..)).reverse2();

                let rotation_twiddle_re = *self.rotation_layer.get_unchecked(uk + 2);
                let rotation_twiddle_im = *self.rotation_layer.get_unchecked(uk + 3);

                let twiddle_re = *self.cos_twiddles.get_unchecked(uk + 2);
                let twiddle_im = *self.cos_twiddles.get_unchecked(uk + 3);

                let rotated_dc1 = fma(s_forward, rotation_twiddle_re, c_forward);
                let rotated_ds2 = fma(c_forward, rotation_twiddle_im, s_forward);

                let twiddled_dc = twiddle_re * rotated_dc1;
                let twiddled_ds = twiddle_im * rotated_ds2;

                dc0 = twiddled_dc + dc0;
                dc2 = AvxStoreD::f64_mul_add(-f64::HALF, twiddled_dc, dc2);
                dc4 = AvxStoreD::f64_mul_add(-f64::HALF, twiddled_dc, dc4);
                dc6 = twiddled_dc + dc6;
                dc8 = AvxStoreD::f64_mul_add(-f64::HALF, twiddled_dc, dc8);

                ds1 = AvxStoreD::f64_mul_add(f64::R9_ODD_TWIDDLE_1, twiddled_ds, ds1);
                ds5 = AvxStoreD::f64_mul_add(-f64::R9_ODD_TWIDDLE_1, twiddled_ds, ds5);
                ds7 = AvxStoreD::f64_mul_add(f64::R9_ODD_TWIDDLE_1, twiddled_ds, ds7);
            }

            {
                let c_forward = AvxStoreD::load2(c_buffer.get_unchecked(q_modules * 2 + k..));
                let s_forward =
                    AvxStoreD::load2(s_buffer.get_unchecked(q_modules * 3 - k - 1..)).reverse2();

                let rotation_twiddle_re = *self.rotation_layer.get_unchecked(uk + 4);
                let rotation_twiddle_im = *self.rotation_layer.get_unchecked(uk + 5);

                let twiddle_re = *self.cos_twiddles.get_unchecked(uk + 4);
                let twiddle_im = *self.cos_twiddles.get_unchecked(uk + 5);

                let rotated_dc1 = fma(s_forward, rotation_twiddle_re, c_forward);
                let rotated_ds2 = fma(c_forward, rotation_twiddle_im, s_forward);

                let twiddled_dc = twiddle_re * rotated_dc1;
                let twiddled_ds = twiddle_im * rotated_ds2;

                dc0 = twiddled_dc + dc0;
                dc2 = AvxStoreD::f64_mul_add(f64::R9_EVEN_TWIDDLE_1, twiddled_dc, dc2);
                dc4 = AvxStoreD::f64_mul_add(f64::R9_EVEN_TWIDDLE_0, twiddled_dc, dc4);
                dc6 = AvxStoreD::f64_mul_add(-f64::HALF, twiddled_dc, dc6);
                dc8 = AvxStoreD::f64_mul_add(f64::R9_EVEN_TWIDDLE_2, twiddled_dc, dc8);

                ds1 = AvxStoreD::f64_mul_add(f64::R9_ODD_TWIDDLE_2, twiddled_ds, ds1);
                ds3 = AvxStoreD::f64_mul_add(f64::R9_ODD_TWIDDLE_1, twiddled_ds, ds3);
                ds5 = AvxStoreD::f64_mul_add(-f64::R9_ODD_TWIDDLE_3, twiddled_ds, ds5);
                ds7 = AvxStoreD::f64_mul_add(-f64::R9_ODD_TWIDDLE_0, twiddled_ds, ds7);
            }

            {
                let c_forward = AvxStoreD::load2(c_buffer.get_unchecked(q_modules * 3 + k..));
                let s_forward =
                    AvxStoreD::load2(s_buffer.get_unchecked(q_modules * 4 - k - 1..)).reverse2();

                let rotation_twiddle_re = *self.rotation_layer.get_unchecked(uk + 6);
                let rotation_twiddle_im = *self.rotation_layer.get_unchecked(uk + 7);

                let twiddle_re = *self.cos_twiddles.get_unchecked(uk + 6);
                let twiddle_im = *self.cos_twiddles.get_unchecked(uk + 7);

                let rotated_dc1 = fma(s_forward, rotation_twiddle_re, c_forward);
                let rotated_ds2 = fma(c_forward, rotation_twiddle_im, s_forward);

                let twiddled_dc = twiddle_re * rotated_dc1;
                let twiddled_ds = twiddle_im * rotated_ds2;

                dc0 = twiddled_dc + dc0;
                dc2 = AvxStoreD::f64_mul_add(f64::R9_EVEN_TWIDDLE_2, twiddled_dc, dc2);
                dc4 = AvxStoreD::f64_mul_add(f64::R9_EVEN_TWIDDLE_1, twiddled_dc, dc4);
                dc6 = AvxStoreD::f64_mul_add(-f64::HALF, twiddled_dc, dc6);
                dc8 = AvxStoreD::f64_mul_add(f64::R9_EVEN_TWIDDLE_0, twiddled_dc, dc8);

                ds1 = AvxStoreD::f64_mul_add(f64::R9_ODD_TWIDDLE_3, twiddled_ds, ds1);
                ds3 = AvxStoreD::f64_mul_add(f64::R9_ODD_TWIDDLE_1, twiddled_ds, ds3);
                ds5 = AvxStoreD::f64_mul_add(f64::R9_ODD_TWIDDLE_0, twiddled_ds, ds5);
                ds7 = AvxStoreD::f64_mul_add(f64::R9_ODD_TWIDDLE_2, twiddled_ds, ds7);
            }

            let a0 = AvxStoreD::load2(a_buffer.get_unchecked(k..));
            let dc = dc0 + a0;

            dc2 = -(dc2 + a0);
            dc4 += a0;
            dc6 += a0;
            dc8 += a0;

            dc.write2(data.slice_from_mut(k..));

            let dss1 = AvxStoreD::f64_mul_add(2., ds1, -dc);
            dss1.reverse2()
                .write2(data.slice_from_mut(q_modules * 2 - k - 1..));

            dc2 = AvxStoreD::f64_mul_add(2., dc2, -dss1);
            dc2.write2(data.slice_from_mut(q_modules * 2 + k..));

            let dss3 = AvxStoreD::f64_mul_add(2., -ds3, -dc2);

            dss3.reverse2()
                .write2(data.slice_from_mut(q_modules * 4 - k - 1..));

            let mdc4 = AvxStoreD::f64_mul_add(2., dc4, -dss3);
            mdc4.write2(data.slice_from_mut(q_modules * 4 + k..));

            let dss5 = AvxStoreD::f64_mul_add(2., ds5, -mdc4);
            dss5.reverse2()
                .write2(data.slice_from_mut(q_modules * 6 - k - 1..));

            dc6 = AvxStoreD::f64_mul_add(2., -dc6, -dss5);
            dc6.write2(data.slice_from_mut(q_modules * 6 + k..));

            let dss6 = AvxStoreD::f64_mul_add(2., -ds7, -dc6);
            dss6.reverse2()
                .write2(data.slice_from_mut(q_modules * 8 - k - 1..));

            dc8 = AvxStoreD::f64_mul_add(2., dc8, -dss6);
            dc8.write2(data.slice_from_mut(q_modules * 8 + k..));
        }
    }
}

impl AvxDct2Butterfly243d {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_impl(&self, data: &mut [f64]) -> Result<(), PxdctError> {
        if !data.len().is_multiple_of(243) {
            return Err(PxdctError::InvalidSizeMultiplier(data.len(), self.length()));
        }

        let mut a_buffer = [f64::zero(); 27];
        let mut c_buffer = [f64::zero(); 108];
        let mut s_buffer = [f64::zero(); 108];

        for chunk in data.chunks_exact_mut(243) {
            self.exec(
                &mut InPlaceStore::new(chunk),
                &mut a_buffer,
                &mut c_buffer,
                &mut s_buffer,
            );
        }
        Ok(())
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_into_impl(&self, input: &[f64], output: &mut [f64]) -> Result<(), PxdctError> {
        use crate::util::validate_oof_sizes;
        validate_oof_sizes!(input, output, 243);

        let mut a_buffer = [f64::zero(); 27];
        let mut c_buffer = [f64::zero(); 108];
        let mut s_buffer = [f64::zero(); 108];

        use crate::bidirectional::BiStore;
        for (src, dst) in input.chunks_exact(243).zip(output.chunks_exact_mut(243)) {
            self.exec(
                &mut BiStore::new(src, dst),
                &mut a_buffer,
                &mut c_buffer,
                &mut s_buffer,
            );
        }
        Ok(())
    }
}

impl PxdctExecutor<f64> for AvxDct2Butterfly243d {
    fn execute(&self, data: &mut [f64]) -> Result<(), PxdctError> {
        unsafe { self.execute_impl(data) }
    }

    fn execute_with_scratch(&self, data: &mut [f64], _: &mut [f64]) -> Result<(), PxdctError> {
        unsafe { self.execute_impl(data) }
    }

    fn execute_into(&self, input: &[f64], output: &mut [f64]) -> Result<(), PxdctError> {
        unsafe { self.execute_into_impl(input, output) }
    }

    fn execute_into_with_scratch(
        &self,
        input: &[f64],
        output: &mut [f64],
        _: &mut [f64],
    ) -> Result<(), PxdctError> {
        unsafe { self.execute_into_impl(input, output) }
    }

    fn length(&self) -> usize {
        243
    }

    fn scratch_size(&self) -> usize {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PxdctExecutor;
    use crate::avx::dct2_bf_power2::gen_test_avx_butterfly;
    use crate::tests::naive_dct2;

    gen_test_avx_butterfly!(test_bf27_f64, AvxDct2Butterfly27d, 27, 1e-3, naive_dct2);
    gen_test_avx_butterfly!(test_bf81_f64, AvxDct2Butterfly81d, 81, 1e-3, naive_dct2);
    gen_test_avx_butterfly!(test_bf243_f64, AvxDct2Butterfly243d, 243, 1e-3, naive_dct2);
}
