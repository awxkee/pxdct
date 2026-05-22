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
use crate::avx::storef::AvxStoreF;
use crate::avx::type4::mixed_radix3f::dct4_radix_n_rotation_twiddles_avxf;
use crate::avx::util::{boring_avx_mixed_radix, fma};
use crate::bidirectional::BidirectionalStore;
use crate::type4::Dct4MixedRadix19Sample;
use crate::util::{DctSample, try_vec, validate_scratch};
use crate::{PxdctError, PxdctExecutor};
use num_traits::One;
use std::sync::Arc;

pub(crate) struct AvxDct4MixedRadix19f {
    inner_dct4: Arc<dyn PxdctExecutor<f32> + Send + Sync>,
    inner_dct_scratch_size: usize,
    rotation_twiddles: Vec<AvxStoreF>,
    execution_length: usize,
    q_modules: usize,
    s: usize,
}

impl AvxDct4MixedRadix19f {
    #[allow(unused)]
    pub(crate) fn new(
        len: usize,
        dct4: Arc<dyn PxdctExecutor<f32> + Send + Sync>,
    ) -> Result<Self, PxdctError> {
        assert_eq!(
            dct4.length(),
            len / 19,
            "DCT-IV Mixed-Radix-19 length DCTs must be one nineteenth of DCT-IV"
        );

        let inner_dct4_scratch_size = dct4.scratch_size();

        Ok(Self {
            inner_dct4: dct4,
            inner_dct_scratch_size: inner_dct4_scratch_size,
            execution_length: len,
            rotation_twiddles: unsafe { dct4_radix_n_rotation_twiddles_avxf(19, len / 19, len) },
            q_modules: len / 19,
            s: 2 * len / 19,
        })
    }
}

boring_avx_mixed_radix!(AvxDct4MixedRadix19f, f32);

impl AvxDct4MixedRadix19f {
    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    fn exec_block<S: BidirectionalStore<f32>, const N: usize>(
        &self,
        data: &mut S,
        a_buffer: &[f32],
        s_buffer: &[f32],
        c_buffer: &[f32],
        uk: usize,
        k: usize,
    ) {
        let c_v0 = AvxStoreF::load_n::<N>(unsafe { c_buffer.get_unchecked(k..) });
        let s_v0 =
            AvxStoreF::load_n::<N>(unsafe { s_buffer.get_unchecked(self.q_modules - N - k..) })
                .reverse_n::<N>();
        let a_v0 = AvxStoreF::load_n::<N>(unsafe { a_buffer.get_unchecked(k..) });

        let c_v1 = AvxStoreF::load_n::<N>(unsafe { c_buffer.get_unchecked(self.q_modules + k..) });
        let s_v1 =
            AvxStoreF::load_n::<N>(unsafe { s_buffer.get_unchecked(self.q_modules * 2 - N - k..) })
                .reverse_n::<N>();

        let c_v2 =
            AvxStoreF::load_n::<N>(unsafe { c_buffer.get_unchecked(self.q_modules * 2 + k..) });
        let s_v2 =
            AvxStoreF::load_n::<N>(unsafe { s_buffer.get_unchecked(self.q_modules * 3 - N - k..) })
                .reverse_n::<N>();

        let c_v3 =
            AvxStoreF::load_n::<N>(unsafe { c_buffer.get_unchecked(self.q_modules * 3 + k..) });
        let s_v3 =
            AvxStoreF::load_n::<N>(unsafe { s_buffer.get_unchecked(self.q_modules * 4 - N - k..) })
                .reverse_n::<N>();

        let c_v4 =
            AvxStoreF::load_n::<N>(unsafe { c_buffer.get_unchecked(self.q_modules * 4 + k..) });
        let s_v4 =
            AvxStoreF::load_n::<N>(unsafe { s_buffer.get_unchecked(self.q_modules * 5 - N - k..) })
                .reverse_n::<N>();

        let c_v5 =
            AvxStoreF::load_n::<N>(unsafe { c_buffer.get_unchecked(self.q_modules * 5 + k..) });
        let s_v5 =
            AvxStoreF::load_n::<N>(unsafe { s_buffer.get_unchecked(self.q_modules * 6 - N - k..) })
                .reverse_n::<N>();

        let c_v6 =
            AvxStoreF::load_n::<N>(unsafe { c_buffer.get_unchecked(self.q_modules * 6 + k..) });
        let s_v6 =
            AvxStoreF::load_n::<N>(unsafe { s_buffer.get_unchecked(self.q_modules * 7 - N - k..) })
                .reverse_n::<N>();

        let c_v7 =
            AvxStoreF::load_n::<N>(unsafe { c_buffer.get_unchecked(self.q_modules * 7 + k..) });
        let s_v7 =
            AvxStoreF::load_n::<N>(unsafe { s_buffer.get_unchecked(self.q_modules * 8 - N - k..) })
                .reverse_n::<N>();

        let c_v8 =
            AvxStoreF::load_n::<N>(unsafe { c_buffer.get_unchecked(self.q_modules * 8 + k..) });
        let s_v8 =
            AvxStoreF::load_n::<N>(unsafe { s_buffer.get_unchecked(self.q_modules * 9 - N - k..) })
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
        let twiddle8_re = unsafe { *self.rotation_twiddles.get_unchecked(uk + 16) };
        let twiddle8_im = unsafe { *self.rotation_twiddles.get_unchecked(uk + 17) };

        let iq0 = fma(c_v0, twiddle0_re, s_v0 * twiddle0_im);
        let siq0 = fma(c_v0, twiddle0_im, -s_v0 * twiddle0_re);
        let mut u0 = iq0;
        let mut u1 = u0;
        let mut v0 = siq0;

        u1 *= f32::D4_R19_ROT_TWIDDLE_2;
        v0 *= f32::D4_R19_ROT_TWIDDLE_3;

        let iq1 = fma(c_v1, twiddle1_re, s_v1 * twiddle1_im);
        let siq1 = fma(c_v1, twiddle1_im, -s_v1 * twiddle1_re);

        u1 = AvxStoreF::mul_f32_add(iq1, f32::D4_R19_ROT_TWIDDLE_6, u1);
        v0 = AvxStoreF::mul_f32_add(siq1, f32::D4_R19_ROT_TWIDDLE_7, v0);

        let iq2 = fma(c_v2, twiddle2_re, s_v2 * twiddle2_im);
        let siq2 = fma(c_v2, twiddle2_im, -s_v2 * twiddle2_re);

        u1 = AvxStoreF::mul_f32_add(iq2, f32::D4_R19_ROT_TWIDDLE_1, u1);
        v0 = AvxStoreF::mul_f32_add(siq2, f32::D4_R19_ROT_TWIDDLE_0, v0);

        let iq3 = fma(c_v3, twiddle3_re, s_v3 * twiddle3_im);
        let siq3 = fma(c_v3, twiddle3_im, -s_v3 * twiddle3_re);

        u1 = AvxStoreF::mul_f32_add(iq3, f32::D4_R19_ROT_TWIDDLE_10, u1);
        v0 = AvxStoreF::mul_f32_add(siq3, f32::D4_R19_ROT_TWIDDLE_11, v0);

        let iq4 = fma(c_v4, twiddle4_re, s_v4 * twiddle4_im);
        let siq4 = fma(c_v4, twiddle4_im, -s_v4 * twiddle4_re);

        u1 = AvxStoreF::mul_f32_add(iq4, f32::D4_R19_ROT_TWIDDLE_12, u1);
        v0 = AvxStoreF::mul_f32_add(siq4, f32::D4_R19_ROT_TWIDDLE_13, v0);

        let iq5 = fma(c_v5, twiddle5_re, s_v5 * twiddle5_im);
        let siq5 = fma(c_v5, twiddle5_im, -s_v5 * twiddle5_re);

        u1 = AvxStoreF::mul_f32_add(iq5, f32::D4_R19_ROT_TWIDDLE_16, u1);
        v0 = AvxStoreF::mul_f32_add(siq5, f32::D4_R19_ROT_TWIDDLE_17, v0);

        let iq6 = fma(c_v6, twiddle6_re, s_v6 * twiddle6_im);
        let siq6 = fma(c_v6, twiddle6_im, -s_v6 * twiddle6_re);

        u1 = AvxStoreF::mul_f32_add(iq6, -f32::D4_R19_ROT_TWIDDLE_9, u1);
        v0 = AvxStoreF::mul_f32_add(siq6, f32::D4_R19_ROT_TWIDDLE_8, v0);

        let iq7 = fma(c_v7, twiddle7_re, s_v7 * twiddle7_im);
        let siq7 = fma(c_v7, twiddle7_im, -s_v7 * twiddle7_re);

        u1 = AvxStoreF::mul_f32_add(iq7, -f32::D4_R19_ROT_TWIDDLE_4, u1);
        v0 = AvxStoreF::mul_f32_add(siq7, f32::D4_R19_ROT_TWIDDLE_5, v0);

        let iq8 = fma(c_v8, twiddle8_re, s_v8 * twiddle8_im);
        let siq8 = fma(c_v8, twiddle8_im, -s_v8 * twiddle8_re);

        u1 = AvxStoreF::mul_f32_add(iq8, -f32::D4_R19_ROT_TWIDDLE_14, u1);
        v0 = AvxStoreF::mul_f32_add(siq8, f32::D4_R19_ROT_TWIDDLE_15, v0);

        u0 += iq1 + iq2 + iq3 + iq4 + iq5 + iq6 + iq7 + iq8 + a_v0;
        u1 = u1 - a_v0;

        let uc0 = u1 - v0;
        let uc1 = u1 + v0;

        u0.write_n::<N>(data.slice_from_mut(k..));
        uc1.write_n::<N>(data.slice_from_mut(self.s + k..));
        uc0.reverse_n::<N>()
            .write_n::<N>(data.slice_from_mut(self.s - N - k..));

        let mut u2 = iq0;
        let mut v2 = siq0;
        u2 *= f32::D4_R19_ROT_TWIDDLE_14;
        v2 *= f32::D4_R19_ROT_TWIDDLE_15;
        u2 = AvxStoreF::mul_f32_add(iq1, f32::D4_R19_ROT_TWIDDLE_9, u2);
        v2 = AvxStoreF::mul_f32_add(siq1, f32::D4_R19_ROT_TWIDDLE_8, v2);
        u2 = AvxStoreF::mul_f32_add(iq2, -f32::D4_R19_ROT_TWIDDLE_12, u2);
        v2 = AvxStoreF::mul_f32_add(siq2, f32::D4_R19_ROT_TWIDDLE_13, v2);
        u2 = AvxStoreF::mul_f32_add(iq3, -f32::D4_R19_ROT_TWIDDLE_1, u2);
        v2 = AvxStoreF::mul_f32_add(siq3, f32::D4_R19_ROT_TWIDDLE_0, v2);
        u2 = AvxStoreF::mul_f32_add(iq4, -f32::D4_R19_ROT_TWIDDLE_2, u2);
        v2 = AvxStoreF::mul_f32_add(siq4, f32::D4_R19_ROT_TWIDDLE_3, v2);
        u2 = AvxStoreF::mul_f32_add(iq5, -f32::D4_R19_ROT_TWIDDLE_6, u2);
        v2 = AvxStoreF::mul_f32_add(siq5, -f32::D4_R19_ROT_TWIDDLE_7, v2);
        u2 = AvxStoreF::mul_f32_add(iq6, -f32::D4_R19_ROT_TWIDDLE_10, u2);
        v2 = AvxStoreF::mul_f32_add(siq6, -f32::D4_R19_ROT_TWIDDLE_11, v2);
        u2 = AvxStoreF::mul_f32_add(iq7, -f32::D4_R19_ROT_TWIDDLE_16, u2);
        v2 = AvxStoreF::mul_f32_add(siq7, -f32::D4_R19_ROT_TWIDDLE_17, v2);
        u2 = AvxStoreF::mul_f32_add(iq8, f32::D4_R19_ROT_TWIDDLE_4, u2);
        v2 = AvxStoreF::mul_f32_add(siq8, -f32::D4_R19_ROT_TWIDDLE_5, v2);
        u2 += a_v0;
        let uc2 = u2 - v2;
        let uc3 = u2 + v2;

        uc2.reverse_n::<N>()
            .write_n::<N>(data.slice_from_mut(2 * self.s - N - k..));
        uc3.write_n::<N>(data.slice_from_mut(2 * self.s + k..));

        let mut u3 = iq0;
        let mut v3 = siq0;
        u3 *= f32::D4_R19_ROT_TWIDDLE_6;
        v3 *= f32::D4_R19_ROT_TWIDDLE_7;
        u3 = AvxStoreF::mul_f32_add(iq1, f32::D4_R19_ROT_TWIDDLE_12, u3);
        v3 = AvxStoreF::mul_f32_add(siq1, f32::D4_R19_ROT_TWIDDLE_13, v3);
        u3 = AvxStoreF::mul_f32_add(iq2, -f32::D4_R19_ROT_TWIDDLE_4, u3);
        v3 = AvxStoreF::mul_f32_add(siq2, f32::D4_R19_ROT_TWIDDLE_5, v3);
        u3 = AvxStoreF::mul_f32_add(iq3, -f32::D4_R19_ROT_TWIDDLE_14, u3);
        v3 = AvxStoreF::mul_f32_add(siq3, -f32::D4_R19_ROT_TWIDDLE_15, v3);
        u3 = AvxStoreF::mul_f32_add(iq4, f32::D4_R19_ROT_TWIDDLE_16, u3);
        v3 = AvxStoreF::mul_f32_add(siq4, -f32::D4_R19_ROT_TWIDDLE_17, v3);
        u3 = AvxStoreF::mul_f32_add(iq5, f32::D4_R19_ROT_TWIDDLE_1, u3);
        v3 = AvxStoreF::mul_f32_add(siq5, -f32::D4_R19_ROT_TWIDDLE_0, v3);
        u3 = AvxStoreF::mul_f32_add(iq6, f32::D4_R19_ROT_TWIDDLE_2, u3);
        v3 = AvxStoreF::mul_f32_add(siq6, f32::D4_R19_ROT_TWIDDLE_3, v3);
        u3 = AvxStoreF::mul_f32_add(iq7, f32::D4_R19_ROT_TWIDDLE_10, u3);
        v3 = AvxStoreF::mul_f32_add(siq7, f32::D4_R19_ROT_TWIDDLE_11, v3);
        u3 = AvxStoreF::mul_f32_add(iq8, -f32::D4_R19_ROT_TWIDDLE_9, u3);
        v3 = AvxStoreF::mul_f32_add(siq8, f32::D4_R19_ROT_TWIDDLE_8, v3);
        u3 = u3 - a_v0;
        let uc4 = u3 - v3;
        let uc5 = u3 + v3;

        uc4.reverse_n::<N>()
            .write_n::<N>(data.slice_from_mut(3 * self.s - N - k..));
        uc5.write_n::<N>(data.slice_from_mut(3 * self.s + k..));

        let mut u4 = iq0;
        let mut v4 = siq0;
        u4 *= f32::D4_R19_ROT_TWIDDLE_4;
        v4 *= f32::D4_R19_ROT_TWIDDLE_5;
        u4 = AvxStoreF::mul_f32_add(iq1, -f32::D4_R19_ROT_TWIDDLE_10, u4);
        v4 = AvxStoreF::mul_f32_add(siq1, f32::D4_R19_ROT_TWIDDLE_11, v4);
        u4 = AvxStoreF::mul_f32_add(iq2, -f32::D4_R19_ROT_TWIDDLE_2, u4);
        v4 = AvxStoreF::mul_f32_add(siq2, -f32::D4_R19_ROT_TWIDDLE_3, v4);
        u4 = AvxStoreF::mul_f32_add(iq3, -f32::D4_R19_ROT_TWIDDLE_12, u4);
        v4 = AvxStoreF::mul_f32_add(siq3, -f32::D4_R19_ROT_TWIDDLE_13, v4);
        u4 = AvxStoreF::mul_f32_add(iq4, f32::D4_R19_ROT_TWIDDLE_14, u4);
        v4 = AvxStoreF::mul_f32_add(siq4, -f32::D4_R19_ROT_TWIDDLE_15, v4);
        u4 = AvxStoreF::mul_f32_add(iq5, f32::D4_R19_ROT_TWIDDLE_9, u4);
        v4 = AvxStoreF::mul_f32_add(siq5, f32::D4_R19_ROT_TWIDDLE_8, v4);
        u4 = AvxStoreF::mul_f32_add(iq6, -f32::D4_R19_ROT_TWIDDLE_1, u4);
        v4 = AvxStoreF::mul_f32_add(siq6, f32::D4_R19_ROT_TWIDDLE_0, v4);
        u4 = AvxStoreF::mul_f32_add(iq7, -f32::D4_R19_ROT_TWIDDLE_6, u4);
        v4 = AvxStoreF::mul_f32_add(siq7, -f32::D4_R19_ROT_TWIDDLE_7, v4);
        u4 = AvxStoreF::mul_f32_add(iq8, -f32::D4_R19_ROT_TWIDDLE_16, u4);
        v4 = AvxStoreF::mul_f32_add(siq8, -f32::D4_R19_ROT_TWIDDLE_17, v4);
        u4 += a_v0;
        let uc6 = u4 - v4;
        let uc7 = u4 + v4;

        uc6.reverse_n::<N>()
            .write_n::<N>(data.slice_from_mut(4 * self.s - N - k..));
        uc7.write_n::<N>(data.slice_from_mut(4 * self.s + k..));

        let mut u5 = iq0;
        let mut v5 = siq0;
        u5 *= f32::D4_R19_ROT_TWIDDLE_1;
        v5 *= f32::D4_R19_ROT_TWIDDLE_0;
        u5 = AvxStoreF::mul_f32_add(iq1, -f32::D4_R19_ROT_TWIDDLE_4, u5);
        v5 = AvxStoreF::mul_f32_add(siq1, f32::D4_R19_ROT_TWIDDLE_5, v5);
        u5 = AvxStoreF::mul_f32_add(iq2, -f32::D4_R19_ROT_TWIDDLE_9, u5);
        v5 = AvxStoreF::mul_f32_add(siq2, -f32::D4_R19_ROT_TWIDDLE_8, v5);
        u5 = AvxStoreF::mul_f32_add(iq3, f32::D4_R19_ROT_TWIDDLE_6, u5);
        v5 = AvxStoreF::mul_f32_add(siq3, -f32::D4_R19_ROT_TWIDDLE_7, v5);
        u5 = AvxStoreF::mul_f32_add(iq4, f32::D4_R19_ROT_TWIDDLE_10, u5);
        v5 = AvxStoreF::mul_f32_add(siq4, f32::D4_R19_ROT_TWIDDLE_11, v5);
        u5 = AvxStoreF::mul_f32_add(iq5, -f32::D4_R19_ROT_TWIDDLE_14, u5);
        v5 = AvxStoreF::mul_f32_add(siq5, f32::D4_R19_ROT_TWIDDLE_15, v5);
        u5 = AvxStoreF::mul_f32_add(iq6, f32::D4_R19_ROT_TWIDDLE_16, u5);
        v5 = AvxStoreF::mul_f32_add(siq6, -f32::D4_R19_ROT_TWIDDLE_17, v5);
        u5 = AvxStoreF::mul_f32_add(iq7, f32::D4_R19_ROT_TWIDDLE_2, u5);
        v5 = AvxStoreF::mul_f32_add(siq7, -f32::D4_R19_ROT_TWIDDLE_3, v5);
        u5 = AvxStoreF::mul_f32_add(iq8, f32::D4_R19_ROT_TWIDDLE_12, u5);
        v5 = AvxStoreF::mul_f32_add(siq8, f32::D4_R19_ROT_TWIDDLE_13, v5);
        u5 = u5 - a_v0;
        let uc8 = u5 - v5;
        let uc9 = u5 + v5;

        uc8.reverse_n::<N>()
            .write_n::<N>(data.slice_from_mut(5 * self.s - N - k..));
        uc9.write_n::<N>(data.slice_from_mut(5 * self.s + k..));

        let mut u6 = iq0;
        let mut v6 = siq0;
        u6 *= f32::D4_R19_ROT_TWIDDLE_9;
        v6 *= f32::D4_R19_ROT_TWIDDLE_8;
        u6 = AvxStoreF::mul_f32_add(iq1, -f32::D4_R19_ROT_TWIDDLE_2, u6);
        v6 = AvxStoreF::mul_f32_add(siq1, f32::D4_R19_ROT_TWIDDLE_3, v6);
        u6 = AvxStoreF::mul_f32_add(iq2, -f32::D4_R19_ROT_TWIDDLE_16, u6);
        v6 = AvxStoreF::mul_f32_add(siq2, -f32::D4_R19_ROT_TWIDDLE_17, v6);
        u6 = AvxStoreF::mul_f32_add(iq3, f32::D4_R19_ROT_TWIDDLE_4, u6);
        v6 = AvxStoreF::mul_f32_add(siq3, f32::D4_R19_ROT_TWIDDLE_5, v6);
        u6 = AvxStoreF::mul_f32_add(iq4, -f32::D4_R19_ROT_TWIDDLE_6, u6);
        v6 = AvxStoreF::mul_f32_add(siq4, f32::D4_R19_ROT_TWIDDLE_7, v6);
        u6 = AvxStoreF::mul_f32_add(iq5, -f32::D4_R19_ROT_TWIDDLE_12, u6);
        v6 = AvxStoreF::mul_f32_add(siq5, -f32::D4_R19_ROT_TWIDDLE_13, v6);
        u6 = AvxStoreF::mul_f32_add(iq6, f32::D4_R19_ROT_TWIDDLE_14, u6);
        v6 = AvxStoreF::mul_f32_add(siq6, f32::D4_R19_ROT_TWIDDLE_15, v6);
        u6 = AvxStoreF::mul_f32_add(iq7, -f32::D4_R19_ROT_TWIDDLE_1, u6);
        v6 = AvxStoreF::mul_f32_add(siq7, f32::D4_R19_ROT_TWIDDLE_0, v6);
        u6 = AvxStoreF::mul_f32_add(iq8, -f32::D4_R19_ROT_TWIDDLE_10, u6);
        v6 = AvxStoreF::mul_f32_add(siq8, -f32::D4_R19_ROT_TWIDDLE_11, v6);
        u6 += a_v0;
        let uc10 = u6 - v6;
        let uc11 = u6 + v6;

        uc10.reverse_n::<N>()
            .write_n::<N>(data.slice_from_mut(6 * self.s - N - k..));
        uc11.write_n::<N>(data.slice_from_mut(6 * self.s + k..));

        let mut u7 = iq0;
        let mut v7 = siq0;
        u7 *= f32::D4_R19_ROT_TWIDDLE_10;
        v7 *= f32::D4_R19_ROT_TWIDDLE_11;
        u7 = AvxStoreF::mul_f32_add(iq1, -f32::D4_R19_ROT_TWIDDLE_14, u7);
        v7 = AvxStoreF::mul_f32_add(siq1, -f32::D4_R19_ROT_TWIDDLE_15, v7);
        u7 = AvxStoreF::mul_f32_add(iq2, f32::D4_R19_ROT_TWIDDLE_6, u7);
        v7 = AvxStoreF::mul_f32_add(siq2, -f32::D4_R19_ROT_TWIDDLE_7, v7);
        u7 = AvxStoreF::mul_f32_add(iq3, f32::D4_R19_ROT_TWIDDLE_16, u7);
        v7 = AvxStoreF::mul_f32_add(siq3, f32::D4_R19_ROT_TWIDDLE_17, v7);
        u7 = AvxStoreF::mul_f32_add(iq4, -f32::D4_R19_ROT_TWIDDLE_9, u7);
        v7 = AvxStoreF::mul_f32_add(siq4, -f32::D4_R19_ROT_TWIDDLE_8, v7);
        u7 = AvxStoreF::mul_f32_add(iq5, f32::D4_R19_ROT_TWIDDLE_2, u7);
        v7 = AvxStoreF::mul_f32_add(siq5, f32::D4_R19_ROT_TWIDDLE_3, v7);
        u7 = AvxStoreF::mul_f32_add(iq6, -f32::D4_R19_ROT_TWIDDLE_4, u7);
        v7 = AvxStoreF::mul_f32_add(siq6, f32::D4_R19_ROT_TWIDDLE_5, v7);
        u7 = AvxStoreF::mul_f32_add(iq7, f32::D4_R19_ROT_TWIDDLE_12, u7);
        v7 = AvxStoreF::mul_f32_add(siq7, -f32::D4_R19_ROT_TWIDDLE_13, v7);
        u7 = AvxStoreF::mul_f32_add(iq8, f32::D4_R19_ROT_TWIDDLE_1, u7);
        v7 = AvxStoreF::mul_f32_add(siq8, f32::D4_R19_ROT_TWIDDLE_0, v7);
        u7 = u7 - a_v0;

        let uc12 = u7 - v7;
        let uc13 = u7 + v7;

        uc12.reverse_n::<N>()
            .write_n::<N>(data.slice_from_mut(7 * self.s - N - k..));
        uc13.write_n::<N>(data.slice_from_mut(7 * self.s + k..));

        let mut u8 = iq0;
        let mut v8 = siq0;
        u8 *= -f32::D4_R19_ROT_TWIDDLE_16;
        v8 *= f32::D4_R19_ROT_TWIDDLE_17;
        u8 = AvxStoreF::mul_f32_add(iq1, -f32::D4_R19_ROT_TWIDDLE_1, u8);
        v8 = AvxStoreF::mul_f32_add(siq1, -f32::D4_R19_ROT_TWIDDLE_0, v8);
        u8 = AvxStoreF::mul_f32_add(iq2, f32::D4_R19_ROT_TWIDDLE_14, u8);
        v8 = AvxStoreF::mul_f32_add(siq2, f32::D4_R19_ROT_TWIDDLE_15, v8);
        u8 = AvxStoreF::mul_f32_add(iq3, -f32::D4_R19_ROT_TWIDDLE_2, u8);
        v8 = AvxStoreF::mul_f32_add(siq3, f32::D4_R19_ROT_TWIDDLE_3, v8);
        u8 = AvxStoreF::mul_f32_add(iq4, f32::D4_R19_ROT_TWIDDLE_4, u8);
        v8 = AvxStoreF::mul_f32_add(siq4, -f32::D4_R19_ROT_TWIDDLE_5, v8);
        u8 = AvxStoreF::mul_f32_add(iq5, -f32::D4_R19_ROT_TWIDDLE_10, u8);
        v8 = AvxStoreF::mul_f32_add(siq5, f32::D4_R19_ROT_TWIDDLE_11, v8);
        u8 = AvxStoreF::mul_f32_add(iq6, -f32::D4_R19_ROT_TWIDDLE_12, u8);
        v8 = AvxStoreF::mul_f32_add(siq6, -f32::D4_R19_ROT_TWIDDLE_13, v8);
        u8 = AvxStoreF::mul_f32_add(iq7, f32::D4_R19_ROT_TWIDDLE_9, u8);
        v8 = AvxStoreF::mul_f32_add(siq7, f32::D4_R19_ROT_TWIDDLE_8, v8);
        u8 = AvxStoreF::mul_f32_add(iq8, -f32::D4_R19_ROT_TWIDDLE_6, u8);
        v8 = AvxStoreF::mul_f32_add(siq8, -f32::D4_R19_ROT_TWIDDLE_7, v8);
        u8 += a_v0;
        let uc14 = u8 - v8;
        let uc15 = u8 + v8;

        uc14.reverse_n::<N>()
            .write_n::<N>(data.slice_from_mut(8 * self.s - N - k..));
        uc15.write_n::<N>(data.slice_from_mut(8 * self.s + k..));

        let mut u9 = iq0;
        let mut v9 = siq0;
        u9 *= f32::D4_R19_ROT_TWIDDLE_12;
        v9 *= f32::D4_R19_ROT_TWIDDLE_13;
        u9 = AvxStoreF::mul_f32_add(iq1, f32::D4_R19_ROT_TWIDDLE_16, u9);
        v9 = AvxStoreF::mul_f32_add(siq1, -f32::D4_R19_ROT_TWIDDLE_17, v9);
        u9 = AvxStoreF::mul_f32_add(iq2, f32::D4_R19_ROT_TWIDDLE_10, u9);
        v9 = AvxStoreF::mul_f32_add(siq2, f32::D4_R19_ROT_TWIDDLE_11, v9);
        u9 = AvxStoreF::mul_f32_add(iq3, -f32::D4_R19_ROT_TWIDDLE_9, u9);
        v9 = AvxStoreF::mul_f32_add(siq3, -f32::D4_R19_ROT_TWIDDLE_8, v9);
        u9 = AvxStoreF::mul_f32_add(iq4, f32::D4_R19_ROT_TWIDDLE_1, u9);
        v9 = AvxStoreF::mul_f32_add(siq4, f32::D4_R19_ROT_TWIDDLE_0, v9);
        u9 = AvxStoreF::mul_f32_add(iq5, -f32::D4_R19_ROT_TWIDDLE_4, u9);
        v9 = AvxStoreF::mul_f32_add(siq5, -f32::D4_R19_ROT_TWIDDLE_5, v9);
        u9 = AvxStoreF::mul_f32_add(iq6, f32::D4_R19_ROT_TWIDDLE_6, u9);
        v9 = AvxStoreF::mul_f32_add(siq6, f32::D4_R19_ROT_TWIDDLE_7, v9);
        u9 = AvxStoreF::mul_f32_add(iq7, -f32::D4_R19_ROT_TWIDDLE_14, u9);
        v9 = AvxStoreF::mul_f32_add(siq7, -f32::D4_R19_ROT_TWIDDLE_15, v9);
        u9 = AvxStoreF::mul_f32_add(iq8, f32::D4_R19_ROT_TWIDDLE_2, u9);
        v9 = AvxStoreF::mul_f32_add(siq8, f32::D4_R19_ROT_TWIDDLE_3, v9);
        u9 = u9 - a_v0;
        let uc16 = u9 - v9;
        let uc17 = u9 + v9;

        uc16.reverse_n::<N>()
            .write_n::<N>(data.slice_from_mut(9 * self.s - N - k..));
        uc17.write_n::<N>(data.slice_from_mut(9 * self.s + k..));
    }

    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_store<S: BidirectionalStore<f32>>(
        &self,
        data: &mut S,
        scratch: &mut [f32],
    ) -> Result<(), PxdctError> {
        let (scratch, inner_scratch) = scratch.split_at_mut(self.execution_length);
        let q_modules = self.execution_length / 19;
        let (a_buffer, c_s_buffer) = scratch.split_at_mut(q_modules);
        let (c_buffer, s_buffer) = c_s_buffer.split_at_mut(q_modules * 9);

        // Step 1: Decompose input into A (center), C (even-symmetric), S (odd-symmetric) buffers
        for (n, dst) in a_buffer.iter_mut().enumerate() {
            *dst = data[n * 19 + 9];
        }

        // Extract and combine symmetric pairs with sign alternation for S buffer
        for (m, (c_buffer, s_buffer)) in c_buffer
            .chunks_exact_mut(q_modules)
            .zip(s_buffer.chunks_exact_mut(q_modules))
            .enumerate()
        {
            let mut sign = f32::one();
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

        let (a_buffer, c_s_buffer) = scratch.split_at_mut(q_modules);
        let (c_buffer, s_buffer) = c_s_buffer.split_at_mut(q_modules * 9);

        let mut k = 0usize;
        let mut uk = 0usize;

        // Step 4: Handle k≥0 cases with rotation twiddles
        while k + 8 <= q_modules {
            self.exec_block::<S, 8>(data, a_buffer, s_buffer, c_buffer, uk, k);

            k += 8;
            uk += 18;
        }

        let remainder = q_modules - k;
        if remainder == 7 {
            self.exec_block::<S, 7>(data, a_buffer, s_buffer, c_buffer, uk, k);
        } else if remainder == 6 {
            self.exec_block::<S, 6>(data, a_buffer, s_buffer, c_buffer, uk, k);
        } else if remainder == 5 {
            self.exec_block::<S, 5>(data, a_buffer, s_buffer, c_buffer, uk, k);
        } else if remainder == 4 {
            self.exec_block::<S, 4>(data, a_buffer, s_buffer, c_buffer, uk, k);
        } else if remainder == 3 {
            self.exec_block::<S, 3>(data, a_buffer, s_buffer, c_buffer, uk, k);
        } else if remainder == 2 {
            self.exec_block::<S, 2>(data, a_buffer, s_buffer, c_buffer, uk, k);
        } else if remainder == 1 {
            self.exec_block::<S, 1>(data, a_buffer, s_buffer, c_buffer, uk, k);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::naive_dct4_f32;
    use crate::type4::Dct4Identity;
    use crate::util::has_valid_avx;
    use rand::RngExt;

    #[test]
    fn test_split_dct4() {
        if !has_valid_avx() {
            return;
        }
        let mut input = vec![0.; 19];
        for z in input.iter_mut() {
            *z = rand::rng().random_range(1.0..2.0);
        }
        let reference_input = input.clone();
        let reference_input = naive_dct4_f32(&reference_input);
        let bf = AvxDct4MixedRadix19f::new(input.len(), Arc::new(Dct4Identity::default())).unwrap();
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
