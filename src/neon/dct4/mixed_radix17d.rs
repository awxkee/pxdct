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
use crate::bidirectional::BidirectionalStore;
use crate::dct4::Dct4MixedRadix17Sample;
use crate::mla::fmla;
use crate::neon::dct4::mixed_radix3d::dct4_radix_n_rotation_twiddles_neond;
use crate::neon::store_d::NeonStoreD;
use crate::neon::util::boring_neon_mixed_radix;
use crate::util::{DctSample, try_vec, validate_scratch};
use crate::{PxdctError, PxdctExecutor};
use num_traits::One;
use std::sync::Arc;

pub(crate) struct NeonDct4MixedRadix17d {
    inner_dct4: Arc<dyn PxdctExecutor<f64> + Send + Sync>,
    inner_dct_scratch_size: usize,
    rotation_twiddles: Vec<NeonStoreD>,
    execution_length: usize,
    q_modules: usize,
    s: usize,
}

impl NeonDct4MixedRadix17d {
    #[allow(unused)]
    pub(crate) fn new(
        len: usize,
        dct4: Arc<dyn PxdctExecutor<f64> + Send + Sync>,
    ) -> Result<Self, PxdctError> {
        assert_eq!(
            dct4.length(),
            len / 17,
            "DCT-IV Mixed-Radix-17 length DCTs must be one seventeenth of DCT-IV"
        );

        let inner_dct4_scratch_size = dct4.scratch_size();

        Ok(Self {
            inner_dct4: dct4,
            inner_dct_scratch_size: inner_dct4_scratch_size,
            execution_length: len,
            rotation_twiddles: dct4_radix_n_rotation_twiddles_neond(17, len / 17, len),
            q_modules: len / 17,
            s: 2 * len / 17,
        })
    }
}

impl NeonDct4MixedRadix17d {
    #[inline(always)]
    fn exec_block<S: BidirectionalStore<f64>, const N: usize>(
        &self,
        data: &mut S,
        a_buffer: &[f64],
        s_buffer: &[f64],
        c_buffer: &[f64],
        uk: usize,
        k: usize,
    ) {
        let c_v0 = NeonStoreD::load_n::<N>(unsafe { c_buffer.get_unchecked(k..) });
        let s_v0 =
            NeonStoreD::load_n::<N>(unsafe { s_buffer.get_unchecked(self.q_modules - N - k..) })
                .reverse_n::<N>();
        let a_v0 = NeonStoreD::load_n::<N>(unsafe { a_buffer.get_unchecked(k..) });

        let c_v1 = NeonStoreD::load_n::<N>(unsafe { c_buffer.get_unchecked(self.q_modules + k..) });
        let s_v1 = NeonStoreD::load_n::<N>(unsafe {
            s_buffer.get_unchecked(self.q_modules * 2 - N - k..)
        })
        .reverse_n::<N>();

        let c_v2 =
            NeonStoreD::load_n::<N>(unsafe { c_buffer.get_unchecked(self.q_modules * 2 + k..) });
        let s_v2 = NeonStoreD::load_n::<N>(unsafe {
            s_buffer.get_unchecked(self.q_modules * 3 - N - k..)
        })
        .reverse_n::<N>();

        let c_v3 =
            NeonStoreD::load_n::<N>(unsafe { c_buffer.get_unchecked(self.q_modules * 3 + k..) });
        let s_v3 = NeonStoreD::load_n::<N>(unsafe {
            s_buffer.get_unchecked(self.q_modules * 4 - N - k..)
        })
        .reverse_n::<N>();

        let c_v4 =
            NeonStoreD::load_n::<N>(unsafe { c_buffer.get_unchecked(self.q_modules * 4 + k..) });
        let s_v4 = NeonStoreD::load_n::<N>(unsafe {
            s_buffer.get_unchecked(self.q_modules * 5 - N - k..)
        })
        .reverse_n::<N>();

        let c_v5 =
            NeonStoreD::load_n::<N>(unsafe { c_buffer.get_unchecked(self.q_modules * 5 + k..) });
        let s_v5 = NeonStoreD::load_n::<N>(unsafe {
            s_buffer.get_unchecked(self.q_modules * 6 - N - k..)
        })
        .reverse_n::<N>();

        let c_v6 =
            NeonStoreD::load_n::<N>(unsafe { c_buffer.get_unchecked(self.q_modules * 6 + k..) });
        let s_v6 = NeonStoreD::load_n::<N>(unsafe {
            s_buffer.get_unchecked(self.q_modules * 7 - N - k..)
        })
        .reverse_n::<N>();

        let c_v7 =
            NeonStoreD::load_n::<N>(unsafe { c_buffer.get_unchecked(self.q_modules * 7 + k..) });
        let s_v7 = NeonStoreD::load_n::<N>(unsafe {
            s_buffer.get_unchecked(self.q_modules * 8 - N - k..)
        })
        .reverse_n::<N>();

        let twiddle0_re = unsafe { *self.rotation_twiddles.get_unchecked(uk) };
        let twiddle0_im = unsafe { *self.rotation_twiddles.get_unchecked(uk + 1) };
        let twiddle1_re = unsafe { *self.rotation_twiddles.get_unchecked(uk + 2) };
        let twiddle1_im = unsafe { *self.rotation_twiddles.get_unchecked(uk + 3) };
        let twiddle2_re = unsafe { *self.rotation_twiddles.get_unchecked(uk + 4) };
        let twiddle2_im = unsafe { *self.rotation_twiddles.get_unchecked(uk + 5) };
        let twiddle3_re = unsafe { *self.rotation_twiddles.get_unchecked(uk + 6) };
        let twiddle3_im = unsafe { *self.rotation_twiddles.get_unchecked(uk + 7) };
        let twiddle4_re = unsafe { *self.rotation_twiddles.get_unchecked(uk + 8) };
        let twiddle4_im = unsafe { *self.rotation_twiddles.get_unchecked(uk + 9) };
        let twiddle5_re = unsafe { *self.rotation_twiddles.get_unchecked(uk + 10) };
        let twiddle5_im = unsafe { *self.rotation_twiddles.get_unchecked(uk + 11) };
        let twiddle6_re = unsafe { *self.rotation_twiddles.get_unchecked(uk + 12) };
        let twiddle6_im = unsafe { *self.rotation_twiddles.get_unchecked(uk + 13) };
        let twiddle7_re = unsafe { *self.rotation_twiddles.get_unchecked(uk + 14) };
        let twiddle7_im = unsafe { *self.rotation_twiddles.get_unchecked(uk + 15) };

        let iq0 = fmla(c_v0, twiddle0_re, s_v0 * twiddle0_im);
        let siq0 = fmla(c_v0, twiddle0_im, -s_v0 * twiddle0_re);
        let mut u0 = iq0;
        let mut u1 = u0;
        let mut v0 = siq0;

        u1 *= f64::D4_R17_ROT_TWIDDLE_2;
        v0 *= f64::D4_R17_ROT_TWIDDLE_3;

        let iq1 = fmla(c_v1, twiddle1_re, s_v1 * twiddle1_im);
        let siq1 = fmla(c_v1, twiddle1_im, -s_v1 * twiddle1_re);

        u1 = NeonStoreD::mul_f64_add(iq1, f64::D4_R17_ROT_TWIDDLE_6, u1);
        v0 = NeonStoreD::mul_f64_add(siq1, f64::D4_R17_ROT_TWIDDLE_7, v0);

        let iq2 = fmla(c_v2, twiddle2_re, s_v2 * twiddle2_im);
        let siq2 = fmla(c_v2, twiddle2_im, -s_v2 * twiddle2_re);

        u1 = NeonStoreD::mul_f64_add(iq2, f64::D4_R17_ROT_TWIDDLE_5, u1);
        v0 = NeonStoreD::mul_f64_add(siq2, f64::D4_R17_ROT_TWIDDLE_4, v0);

        let iq3 = fmla(c_v3, twiddle3_re, s_v3 * twiddle3_im);
        let siq3 = fmla(c_v3, twiddle3_im, -s_v3 * twiddle3_re);

        u1 = NeonStoreD::mul_f64_add(iq3, f64::D4_R17_ROT_TWIDDLE_10, u1);
        v0 = NeonStoreD::mul_f64_add(siq3, f64::D4_R17_ROT_TWIDDLE_11, v0);

        let iq4 = fmla(c_v4, twiddle4_re, s_v4 * twiddle4_im);
        let siq4 = fmla(c_v4, twiddle4_im, -s_v4 * twiddle4_re);

        u1 = NeonStoreD::mul_f64_add(iq4, f64::D4_R17_ROT_TWIDDLE_14, u1);
        v0 = NeonStoreD::mul_f64_add(siq4, f64::D4_R17_ROT_TWIDDLE_15, v0);

        let iq5 = fmla(c_v5, twiddle5_re, s_v5 * twiddle5_im);
        let siq5 = fmla(c_v5, twiddle5_im, -s_v5 * twiddle5_re);

        u1 = NeonStoreD::mul_f64_add(iq5, -f64::D4_R17_ROT_TWIDDLE_9, u1);
        v0 = NeonStoreD::mul_f64_add(siq5, f64::D4_R17_ROT_TWIDDLE_8, v0);

        let iq6 = fmla(c_v6, twiddle6_re, s_v6 * twiddle6_im);
        let siq6 = fmla(c_v6, twiddle6_im, -s_v6 * twiddle6_re);

        u1 = NeonStoreD::mul_f64_add(iq6, -f64::D4_R17_ROT_TWIDDLE_0, u1);
        v0 = NeonStoreD::mul_f64_add(siq6, f64::D4_R17_ROT_TWIDDLE_1, v0);

        let iq7 = fmla(c_v7, twiddle7_re, s_v7 * twiddle7_im);
        let siq7 = fmla(c_v7, twiddle7_im, -s_v7 * twiddle7_re);

        u1 = NeonStoreD::mul_f64_add(iq7, -f64::D4_R17_ROT_TWIDDLE_12, u1);
        v0 = NeonStoreD::mul_f64_add(siq7, f64::D4_R17_ROT_TWIDDLE_13, v0);

        u0 += iq1 + iq2 + iq3 + iq4 + iq5 + iq6 + iq7 + a_v0;
        u1 = u1 - a_v0;

        let uc0 = u1 - v0;
        let uc1 = u1 + v0;

        u0.write_n::<N>(data.slice_from_mut(k..));
        uc1.write_n::<N>(data.slice_from_mut(self.s + k..));
        uc0.reverse_n::<N>()
            .write_n::<N>(data.slice_from_mut(self.s - N - k..));

        let mut u2 = iq0;
        let mut v2 = siq0;
        u2 *= f64::D4_R17_ROT_TWIDDLE_12;
        v2 *= f64::D4_R17_ROT_TWIDDLE_13;
        u2 = NeonStoreD::mul_f64_add(iq1, f64::D4_R17_ROT_TWIDDLE_9, u2);
        v2 = NeonStoreD::mul_f64_add(siq1, f64::D4_R17_ROT_TWIDDLE_8, v2);
        u2 = NeonStoreD::mul_f64_add(iq2, -f64::D4_R17_ROT_TWIDDLE_10, u2);
        v2 = NeonStoreD::mul_f64_add(siq2, f64::D4_R17_ROT_TWIDDLE_11, v2);
        u2 = NeonStoreD::mul_f64_add(iq3, -f64::D4_R17_ROT_TWIDDLE_6, u2);
        v2 = NeonStoreD::mul_f64_add(siq3, f64::D4_R17_ROT_TWIDDLE_7, v2);
        u2 = NeonStoreD::mul_f64_add(iq4, -f64::D4_R17_ROT_TWIDDLE_2, u2);
        v2 = NeonStoreD::mul_f64_add(siq4, -f64::D4_R17_ROT_TWIDDLE_3, v2);
        u2 = NeonStoreD::mul_f64_add(iq5, -f64::D4_R17_ROT_TWIDDLE_5, u2);
        v2 = NeonStoreD::mul_f64_add(siq5, -f64::D4_R17_ROT_TWIDDLE_4, v2);
        u2 = NeonStoreD::mul_f64_add(iq6, -f64::D4_R17_ROT_TWIDDLE_14, u2);
        v2 = NeonStoreD::mul_f64_add(siq6, -f64::D4_R17_ROT_TWIDDLE_15, v2);
        u2 = NeonStoreD::mul_f64_add(iq7, f64::D4_R17_ROT_TWIDDLE_0, u2);
        v2 = NeonStoreD::mul_f64_add(siq7, -f64::D4_R17_ROT_TWIDDLE_1, v2);
        u2 += a_v0;
        let uc2 = u2 - v2;
        let uc3 = u2 + v2;

        uc2.reverse_n::<N>()
            .write_n::<N>(data.slice_from_mut(2 * self.s - N - k..));
        uc3.write_n::<N>(data.slice_from_mut(2 * self.s + k..));

        let mut u3 = iq0;
        let mut v3 = siq0;
        u3 *= f64::D4_R17_ROT_TWIDDLE_6;
        v3 *= f64::D4_R17_ROT_TWIDDLE_7;
        u3 = NeonStoreD::mul_f64_add(iq1, f64::D4_R17_ROT_TWIDDLE_14, u3);
        v3 = NeonStoreD::mul_f64_add(siq1, f64::D4_R17_ROT_TWIDDLE_15, v3);
        u3 = NeonStoreD::mul_f64_add(iq2, -f64::D4_R17_ROT_TWIDDLE_12, u3);
        v3 = NeonStoreD::mul_f64_add(siq2, f64::D4_R17_ROT_TWIDDLE_13, v3);
        u3 = NeonStoreD::mul_f64_add(iq3, -f64::D4_R17_ROT_TWIDDLE_0, u3);
        v3 = NeonStoreD::mul_f64_add(siq3, -f64::D4_R17_ROT_TWIDDLE_1, v3);
        u3 = NeonStoreD::mul_f64_add(iq4, f64::D4_R17_ROT_TWIDDLE_10, u3);
        v3 = NeonStoreD::mul_f64_add(siq4, -f64::D4_R17_ROT_TWIDDLE_11, v3);
        u3 = NeonStoreD::mul_f64_add(iq5, f64::D4_R17_ROT_TWIDDLE_2, u3);
        v3 = NeonStoreD::mul_f64_add(siq5, -f64::D4_R17_ROT_TWIDDLE_3, v3);
        u3 = NeonStoreD::mul_f64_add(iq6, f64::D4_R17_ROT_TWIDDLE_5, u3);
        v3 = NeonStoreD::mul_f64_add(siq6, f64::D4_R17_ROT_TWIDDLE_4, v3);
        u3 = NeonStoreD::mul_f64_add(iq7, -f64::D4_R17_ROT_TWIDDLE_9, u3);
        v3 = NeonStoreD::mul_f64_add(siq7, f64::D4_R17_ROT_TWIDDLE_8, v3);
        u3 = u3 - a_v0;
        let uc4 = u3 - v3;
        let uc5 = u3 + v3;

        uc4.reverse_n::<N>()
            .write_n::<N>(data.slice_from_mut(3 * self.s - N - k..));
        uc5.write_n::<N>(data.slice_from_mut(3 * self.s + k..));

        let mut u4 = iq0;
        let mut v4 = siq0;
        u4 *= f64::D4_R17_ROT_TWIDDLE_0;
        v4 *= f64::D4_R17_ROT_TWIDDLE_1;
        u4 = NeonStoreD::mul_f64_add(iq1, -f64::D4_R17_ROT_TWIDDLE_5, u4);
        v4 = NeonStoreD::mul_f64_add(siq1, f64::D4_R17_ROT_TWIDDLE_4, v4);
        u4 = NeonStoreD::mul_f64_add(iq2, -f64::D4_R17_ROT_TWIDDLE_6, u4);
        v4 = NeonStoreD::mul_f64_add(siq2, -f64::D4_R17_ROT_TWIDDLE_7, v4);
        u4 = NeonStoreD::mul_f64_add(iq3, f64::D4_R17_ROT_TWIDDLE_9, u4);
        v4 = NeonStoreD::mul_f64_add(siq3, -f64::D4_R17_ROT_TWIDDLE_8, v4);
        u4 = NeonStoreD::mul_f64_add(iq4, f64::D4_R17_ROT_TWIDDLE_12, u4);
        v4 = NeonStoreD::mul_f64_add(siq4, f64::D4_R17_ROT_TWIDDLE_13, v4);
        u4 = NeonStoreD::mul_f64_add(iq5, -f64::D4_R17_ROT_TWIDDLE_10, u4);
        v4 = NeonStoreD::mul_f64_add(siq5, f64::D4_R17_ROT_TWIDDLE_11, v4);
        u4 = NeonStoreD::mul_f64_add(iq6, -f64::D4_R17_ROT_TWIDDLE_2, u4);
        v4 = NeonStoreD::mul_f64_add(siq6, -f64::D4_R17_ROT_TWIDDLE_3, v4);
        u4 = NeonStoreD::mul_f64_add(iq7, -f64::D4_R17_ROT_TWIDDLE_14, u4);
        v4 = NeonStoreD::mul_f64_add(siq7, -f64::D4_R17_ROT_TWIDDLE_15, v4);
        u4 += a_v0;
        let uc6 = u4 - v4;
        let uc7 = u4 + v4;

        uc6.reverse_n::<N>()
            .write_n::<N>(data.slice_from_mut(4 * self.s - N - k..));
        uc7.write_n::<N>(data.slice_from_mut(4 * self.s + k..));

        let mut u5 = iq0;
        let mut v5 = siq0;
        u5 *= f64::D4_R17_ROT_TWIDDLE_5;
        v5 *= f64::D4_R17_ROT_TWIDDLE_4;
        u5 = NeonStoreD::mul_f64_add(iq1, -f64::D4_R17_ROT_TWIDDLE_12, u5);
        v5 = NeonStoreD::mul_f64_add(siq1, f64::D4_R17_ROT_TWIDDLE_13, v5);
        u5 = NeonStoreD::mul_f64_add(iq2, f64::D4_R17_ROT_TWIDDLE_14, u5);
        v5 = NeonStoreD::mul_f64_add(siq2, -f64::D4_R17_ROT_TWIDDLE_15, v5);
        u5 = NeonStoreD::mul_f64_add(iq3, f64::D4_R17_ROT_TWIDDLE_2, u5);
        v5 = NeonStoreD::mul_f64_add(siq3, f64::D4_R17_ROT_TWIDDLE_3, v5);
        u5 = NeonStoreD::mul_f64_add(iq4, -f64::D4_R17_ROT_TWIDDLE_9, u5);
        v5 = NeonStoreD::mul_f64_add(siq4, f64::D4_R17_ROT_TWIDDLE_8, v5);
        u5 = NeonStoreD::mul_f64_add(iq5, -f64::D4_R17_ROT_TWIDDLE_0, u5);
        v5 = NeonStoreD::mul_f64_add(siq5, -f64::D4_R17_ROT_TWIDDLE_1, v5);
        u5 = NeonStoreD::mul_f64_add(iq6, f64::D4_R17_ROT_TWIDDLE_6, u5);
        v5 = NeonStoreD::mul_f64_add(siq6, -f64::D4_R17_ROT_TWIDDLE_7, v5);
        u5 = NeonStoreD::mul_f64_add(iq7, f64::D4_R17_ROT_TWIDDLE_10, u5);
        v5 = NeonStoreD::mul_f64_add(siq7, f64::D4_R17_ROT_TWIDDLE_11, v5);
        u5 = u5 - a_v0;
        let uc8 = u5 - v5;
        let uc9 = u5 + v5;

        uc8.reverse_n::<N>()
            .write_n::<N>(data.slice_from_mut(5 * self.s - N - k..));
        uc9.write_n::<N>(data.slice_from_mut(5 * self.s + k..));

        let mut u6 = iq0;
        let mut v6 = siq0;
        u6 *= f64::D4_R17_ROT_TWIDDLE_9;
        v6 *= f64::D4_R17_ROT_TWIDDLE_8;
        u6 = NeonStoreD::mul_f64_add(iq1, -f64::D4_R17_ROT_TWIDDLE_2, u6);
        v6 = NeonStoreD::mul_f64_add(siq1, -f64::D4_R17_ROT_TWIDDLE_3, v6);
        u6 = NeonStoreD::mul_f64_add(iq2, f64::D4_R17_ROT_TWIDDLE_0, u6);
        v6 = NeonStoreD::mul_f64_add(siq2, -f64::D4_R17_ROT_TWIDDLE_1, v6);
        u6 = NeonStoreD::mul_f64_add(iq3, -f64::D4_R17_ROT_TWIDDLE_14, u6);
        v6 = NeonStoreD::mul_f64_add(siq3, f64::D4_R17_ROT_TWIDDLE_15, v6);
        u6 = NeonStoreD::mul_f64_add(iq4, -f64::D4_R17_ROT_TWIDDLE_6, u6);
        v6 = NeonStoreD::mul_f64_add(siq4, -f64::D4_R17_ROT_TWIDDLE_7, v6);
        u6 = NeonStoreD::mul_f64_add(iq5, f64::D4_R17_ROT_TWIDDLE_12, u6);
        v6 = NeonStoreD::mul_f64_add(siq5, -f64::D4_R17_ROT_TWIDDLE_13, v6);
        u6 = NeonStoreD::mul_f64_add(iq6, -f64::D4_R17_ROT_TWIDDLE_10, u6);
        v6 = NeonStoreD::mul_f64_add(siq6, f64::D4_R17_ROT_TWIDDLE_11, v6);
        u6 = NeonStoreD::mul_f64_add(iq7, -f64::D4_R17_ROT_TWIDDLE_5, u6);
        v6 = NeonStoreD::mul_f64_add(siq7, -f64::D4_R17_ROT_TWIDDLE_4, v6);
        u6 += a_v0;
        let uc10 = u6 - v6;
        let uc11 = u6 + v6;

        uc10.reverse_n::<N>()
            .write_n::<N>(data.slice_from_mut(6 * self.s - N - k..));
        uc11.write_n::<N>(data.slice_from_mut(6 * self.s + k..));

        let mut u7 = iq0;
        let mut v7 = siq0;
        u7 *= f64::D4_R17_ROT_TWIDDLE_10;
        v7 *= f64::D4_R17_ROT_TWIDDLE_11;
        u7 = NeonStoreD::mul_f64_add(iq1, -f64::D4_R17_ROT_TWIDDLE_0, u7);
        v7 = NeonStoreD::mul_f64_add(siq1, -f64::D4_R17_ROT_TWIDDLE_1, v7);
        u7 = NeonStoreD::mul_f64_add(iq2, f64::D4_R17_ROT_TWIDDLE_2, u7);
        v7 = NeonStoreD::mul_f64_add(siq2, f64::D4_R17_ROT_TWIDDLE_3, v7);
        u7 = NeonStoreD::mul_f64_add(iq3, -f64::D4_R17_ROT_TWIDDLE_12, u7);
        v7 = NeonStoreD::mul_f64_add(siq3, f64::D4_R17_ROT_TWIDDLE_13, v7);
        u7 = NeonStoreD::mul_f64_add(iq4, f64::D4_R17_ROT_TWIDDLE_5, u7);
        v7 = NeonStoreD::mul_f64_add(siq4, -f64::D4_R17_ROT_TWIDDLE_4, v7);
        u7 = NeonStoreD::mul_f64_add(iq5, f64::D4_R17_ROT_TWIDDLE_14, u7);
        v7 = NeonStoreD::mul_f64_add(siq5, f64::D4_R17_ROT_TWIDDLE_15, v7);
        u7 = NeonStoreD::mul_f64_add(iq6, -f64::D4_R17_ROT_TWIDDLE_9, u7);
        v7 = NeonStoreD::mul_f64_add(siq6, -f64::D4_R17_ROT_TWIDDLE_8, v7);
        u7 = NeonStoreD::mul_f64_add(iq7, f64::D4_R17_ROT_TWIDDLE_6, u7);
        v7 = NeonStoreD::mul_f64_add(siq7, f64::D4_R17_ROT_TWIDDLE_7, v7);
        u7 = u7 - a_v0;

        let uc12 = u7 - v7;
        let uc13 = u7 + v7;

        uc12.reverse_n::<N>()
            .write_n::<N>(data.slice_from_mut(7 * self.s - N - k..));
        uc13.write_n::<N>(data.slice_from_mut(7 * self.s + k..));

        let mut u8 = iq0;
        let mut v8 = siq0;
        u8 *= -f64::D4_R17_ROT_TWIDDLE_14;
        v8 *= f64::D4_R17_ROT_TWIDDLE_15;
        u8 = NeonStoreD::mul_f64_add(iq1, -f64::D4_R17_ROT_TWIDDLE_10, u8);
        v8 = NeonStoreD::mul_f64_add(siq1, -f64::D4_R17_ROT_TWIDDLE_11, v8);
        u8 = NeonStoreD::mul_f64_add(iq2, f64::D4_R17_ROT_TWIDDLE_9, u8);
        v8 = NeonStoreD::mul_f64_add(siq2, f64::D4_R17_ROT_TWIDDLE_8, v8);
        u8 = NeonStoreD::mul_f64_add(iq3, -f64::D4_R17_ROT_TWIDDLE_5, u8);
        v8 = NeonStoreD::mul_f64_add(siq3, -f64::D4_R17_ROT_TWIDDLE_4, v8);
        u8 = NeonStoreD::mul_f64_add(iq4, f64::D4_R17_ROT_TWIDDLE_0, u8);
        v8 = NeonStoreD::mul_f64_add(siq4, f64::D4_R17_ROT_TWIDDLE_1, v8);
        u8 = NeonStoreD::mul_f64_add(iq5, -f64::D4_R17_ROT_TWIDDLE_6, u8);
        v8 = NeonStoreD::mul_f64_add(siq5, -f64::D4_R17_ROT_TWIDDLE_7, v8);
        u8 = NeonStoreD::mul_f64_add(iq6, f64::D4_R17_ROT_TWIDDLE_12, u8);
        v8 = NeonStoreD::mul_f64_add(siq6, f64::D4_R17_ROT_TWIDDLE_13, v8);
        u8 = NeonStoreD::mul_f64_add(iq7, -f64::D4_R17_ROT_TWIDDLE_2, u8);
        v8 = NeonStoreD::mul_f64_add(siq7, -f64::D4_R17_ROT_TWIDDLE_3, v8);
        u8 += a_v0;
        let uc14 = u8 - v8;
        let uc15 = u8 + v8;

        uc14.reverse_n::<N>()
            .write_n::<N>(data.slice_from_mut(8 * self.s - N - k..));
        uc15.write_n::<N>(data.slice_from_mut(8 * self.s + k..));
    }

    #[inline(always)]
    fn execute_with_store<S: BidirectionalStore<f64>>(
        &self,
        data: &mut S,
        scratch: &mut [f64],
    ) -> Result<(), PxdctError> {
        let (scratch, inner_scratch) = scratch.split_at_mut(self.execution_length);
        let q_modules = self.execution_length / 17;
        let (a_buffer, c_s_buffer) = scratch.split_at_mut(q_modules);
        let (c_buffer, s_buffer) = c_s_buffer.split_at_mut(q_modules * 8);

        // Step 1: Decompose input into A (center), C (even-symmetric), S (odd-symmetric) buffers
        for (n, dst) in a_buffer.iter_mut().enumerate() {
            *dst = data[n * 17 + 8];
        }

        // Extract and combine symmetric pairs with sign alternation for S buffer
        for (m, (c_buffer, s_buffer)) in c_buffer
            .chunks_exact_mut(q_modules)
            .zip(s_buffer.chunks_exact_mut(q_modules))
            .enumerate()
        {
            let mut sign = f64::one();
            for (n, (c_dst, s_dst)) in c_buffer.iter_mut().zip(s_buffer.iter_mut()).enumerate() {
                let u0 = data[17 * n + m];
                let u1 = data[17 * n + 17 - m - 1];

                *c_dst = u0 + u1;
                *s_dst = (u0 - u1).mulsign(sign);

                sign = -sign;
            }
        }

        // Step 2: Apply DCT-IV to all buffers (A, C₀, C₁, S₀, S₁)
        self.inner_dct4
            .execute_with_scratch(scratch, inner_scratch)?;

        let (a_buffer, c_s_buffer) = scratch.split_at_mut(q_modules);
        let (c_buffer, s_buffer) = c_s_buffer.split_at_mut(q_modules * 8);

        let mut uk = 0usize;
        let mut k = 0usize;
        // Step 4: Handle k≥0 cases with rotation twiddles
        while k + 2 <= q_modules {
            self.exec_block::<S, 2>(data, a_buffer, s_buffer, c_buffer, uk, k);

            k += 2;
            uk += 16;
        }

        let rem = q_modules - k;
        if rem == 1 {
            self.exec_block::<S, 1>(data, a_buffer, s_buffer, c_buffer, uk, k);
        }
        Ok(())
    }
}

boring_neon_mixed_radix!(NeonDct4MixedRadix17d, f64);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dct4::Dct4Butterfly4;
    use crate::tests::naive_dct4;
    use rand::RngExt;

    #[test]
    fn test_split_dct4() {
        let mut input = vec![0.; 68];
        for z in input.iter_mut() {
            *z = rand::rng().random_range(1.0..2.0);
        }
        let reference_input = input.clone();
        let reference_input = naive_dct4(&reference_input);
        let bf =
            NeonDct4MixedRadix17d::new(input.len(), Arc::new(Dct4Butterfly4::default())).unwrap();
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
