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
use crate::avx::dct4::mixed_radix3d::dct4_radix_n_rotation_twiddles_avxd;
use crate::avx::stored::AvxStoreD;
use crate::avx::util::{boring_avx_mixed_radix, fma};
use crate::bidirectional::BidirectionalStore;
use crate::type4::Dct4MixedRadix13Sample;
use crate::util::{DctSample, try_vec, validate_scratch};
use crate::{PxdctError, PxdctExecutor};
use num_traits::One;
use std::sync::Arc;

pub(crate) struct AvxDct4MixedRadix13d {
    inner_dct4: Arc<dyn PxdctExecutor<f64> + Send + Sync>,
    inner_dct_scratch_size: usize,
    rotation_twiddles: Vec<AvxStoreD>,
    execution_length: usize,
    q_modules: usize,
    s: usize,
}

impl AvxDct4MixedRadix13d {
    pub(crate) fn new(
        len: usize,
        dct4: Arc<dyn PxdctExecutor<f64> + Send + Sync>,
    ) -> Result<Self, PxdctError> {
        assert_eq!(
            dct4.length(),
            len / 13,
            "DCT-IV Mixed-Radix-13 length DCTs must be one eleventh of DCT-IV"
        );

        let inner_dct4_scratch_size = dct4.scratch_size();

        Ok(Self {
            inner_dct4: dct4,
            inner_dct_scratch_size: inner_dct4_scratch_size,
            execution_length: len,
            rotation_twiddles: unsafe { dct4_radix_n_rotation_twiddles_avxd(13, len / 13, len) },
            q_modules: len / 13,
            s: 2 * len / 13,
        })
    }
}

boring_avx_mixed_radix!(AvxDct4MixedRadix13d, f64);

impl AvxDct4MixedRadix13d {
    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    fn exec_block<S: BidirectionalStore<f64>, const N: usize>(
        &self,
        data: &mut S,
        a_buffer: &[f64],
        s_buffer: &[f64],
        c_buffer: &[f64],
        uk: usize,
        k: usize,
    ) {
        let c_v0 = AvxStoreD::load_n::<N>(unsafe { c_buffer.get_unchecked(k..) });
        let s_v0 =
            AvxStoreD::load_n::<N>(unsafe { s_buffer.get_unchecked(self.q_modules - N - k..) })
                .reverse_n::<N>();
        let a_v0 = AvxStoreD::load_n::<N>(unsafe { a_buffer.get_unchecked(k..) });

        let c_v1 = AvxStoreD::load_n::<N>(unsafe { c_buffer.get_unchecked(self.q_modules + k..) });
        let s_v1 =
            AvxStoreD::load_n::<N>(unsafe { s_buffer.get_unchecked(self.q_modules * 2 - N - k..) })
                .reverse_n::<N>();

        let c_v2 =
            AvxStoreD::load_n::<N>(unsafe { c_buffer.get_unchecked(self.q_modules * 2 + k..) });
        let s_v2 =
            AvxStoreD::load_n::<N>(unsafe { s_buffer.get_unchecked(self.q_modules * 3 - N - k..) })
                .reverse_n::<N>();

        let c_v3 =
            AvxStoreD::load_n::<N>(unsafe { c_buffer.get_unchecked(self.q_modules * 3 + k..) });
        let s_v3 =
            AvxStoreD::load_n::<N>(unsafe { s_buffer.get_unchecked(self.q_modules * 4 - N - k..) })
                .reverse_n::<N>();

        let c_v4 =
            AvxStoreD::load_n::<N>(unsafe { c_buffer.get_unchecked(self.q_modules * 4 + k..) });
        let s_v4 =
            AvxStoreD::load_n::<N>(unsafe { s_buffer.get_unchecked(self.q_modules * 5 - N - k..) })
                .reverse_n::<N>();

        let c_v5 =
            AvxStoreD::load_n::<N>(unsafe { c_buffer.get_unchecked(self.q_modules * 5 + k..) });
        let s_v5 =
            AvxStoreD::load_n::<N>(unsafe { s_buffer.get_unchecked(self.q_modules * 6 - N - k..) })
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

        let iq0 = fma(c_v0, twiddle0_re, s_v0 * twiddle0_im);
        let siq0 = fma(c_v0, twiddle0_im, -s_v0 * twiddle0_re);
        let mut u0 = iq0;
        let mut u1 = u0;
        let mut v0 = siq0;

        u1 *= f64::D4_R13_ROT_TWIDDLE_2;
        v0 *= f64::D4_R13_ROT_TWIDDLE_3;

        let iq1 = fma(c_v1, twiddle1_re, s_v1 * twiddle1_im);
        let siq1 = fma(c_v1, twiddle1_im, -s_v1 * twiddle1_re);

        u1 = AvxStoreD::mul_f64_add(iq1, f64::D4_R13_ROT_TWIDDLE_0, u1);
        v0 = AvxStoreD::mul_f64_add(siq1, f64::D4_R13_ROT_TWIDDLE_1, v0);

        let iq2 = fma(c_v2, twiddle2_re, s_v2 * twiddle2_im);
        let siq2 = fma(c_v2, twiddle2_im, -s_v2 * twiddle2_re);

        u1 = AvxStoreD::mul_f64_add(iq2, f64::D4_R13_ROT_TWIDDLE_8, u1);
        v0 = AvxStoreD::mul_f64_add(siq2, f64::D4_R13_ROT_TWIDDLE_9, v0);

        let iq3 = fma(c_v3, twiddle3_re, s_v3 * twiddle3_im);
        let siq3 = fma(c_v3, twiddle3_im, -s_v3 * twiddle3_re);

        u1 = AvxStoreD::mul_f64_add(iq3, f64::D4_R13_ROT_TWIDDLE_10, u1);
        v0 = AvxStoreD::mul_f64_add(siq3, f64::D4_R13_ROT_TWIDDLE_11, v0);

        let iq4 = fma(c_v4, twiddle4_re, s_v4 * twiddle4_im);
        let siq4 = fma(c_v4, twiddle4_im, -s_v4 * twiddle4_re);

        u1 = AvxStoreD::mul_f64_add(iq4, -f64::D4_R13_ROT_TWIDDLE_5, u1);
        v0 = AvxStoreD::mul_f64_add(siq4, f64::D4_R13_ROT_TWIDDLE_4, v0);

        let iq5 = fma(c_v5, twiddle5_re, s_v5 * twiddle5_im);
        let siq5 = fma(c_v5, twiddle5_im, -s_v5 * twiddle5_re);

        u1 = AvxStoreD::mul_f64_add(iq5, -f64::D4_R13_ROT_TWIDDLE_6, u1);
        v0 = AvxStoreD::mul_f64_add(siq5, f64::D4_R13_ROT_TWIDDLE_7, v0);

        u0 += iq1 + iq2 + iq3 + iq4 + iq5 + a_v0;
        u1 = u1 - a_v0;

        let uc0 = u1 - v0;
        let uc1 = u1 + v0;

        u0.write_n::<N>(data.slice_from_mut(k..));
        uc1.write_n::<N>(data.slice_from_mut(self.s + k..));
        uc0.reverse_n::<N>()
            .write_n::<N>(data.slice_from_mut(self.s - N - k..));

        let mut u2 = iq0;
        let mut v2 = siq0;
        u2 *= f64::D4_R13_ROT_TWIDDLE_6;
        v2 *= f64::D4_R13_ROT_TWIDDLE_7;
        u2 = AvxStoreD::mul_f64_add(iq1, -f64::D4_R13_ROT_TWIDDLE_10, u2);
        v2 = AvxStoreD::mul_f64_add(siq1, f64::D4_R13_ROT_TWIDDLE_11, v2);
        u2 = AvxStoreD::mul_f64_add(iq2, -f64::D4_R13_ROT_TWIDDLE_0, u2);
        v2 = AvxStoreD::mul_f64_add(siq2, f64::D4_R13_ROT_TWIDDLE_1, v2);
        u2 = AvxStoreD::mul_f64_add(iq3, -f64::D4_R13_ROT_TWIDDLE_2, u2);
        v2 = AvxStoreD::mul_f64_add(siq3, -f64::D4_R13_ROT_TWIDDLE_3, v2);
        u2 = AvxStoreD::mul_f64_add(iq4, -f64::D4_R13_ROT_TWIDDLE_8, u2);
        v2 = AvxStoreD::mul_f64_add(siq4, -f64::D4_R13_ROT_TWIDDLE_9, v2);
        u2 = AvxStoreD::mul_f64_add(iq5, f64::D4_R13_ROT_TWIDDLE_5, u2);
        v2 = AvxStoreD::mul_f64_add(siq5, -f64::D4_R13_ROT_TWIDDLE_4, v2);
        u2 += a_v0;
        let uc2 = u2 - v2;
        let uc3 = u2 + v2;

        uc2.reverse_n::<N>()
            .write_n::<N>(data.slice_from_mut(2 * self.s - N - k..));
        uc3.write_n::<N>(data.slice_from_mut(2 * self.s + k..));

        let mut u3 = iq0;
        let mut v3 = siq0;
        u3 *= f64::D4_R13_ROT_TWIDDLE_0;
        v3 *= f64::D4_R13_ROT_TWIDDLE_1;
        u3 = AvxStoreD::mul_f64_add(iq1, -f64::D4_R13_ROT_TWIDDLE_5, u3);
        v3 = AvxStoreD::mul_f64_add(siq1, f64::D4_R13_ROT_TWIDDLE_4, v3);
        u3 = AvxStoreD::mul_f64_add(iq2, -f64::D4_R13_ROT_TWIDDLE_6, u3);
        v3 = AvxStoreD::mul_f64_add(siq2, -f64::D4_R13_ROT_TWIDDLE_7, v3);
        u3 = AvxStoreD::mul_f64_add(iq3, f64::D4_R13_ROT_TWIDDLE_8, u3);
        v3 = AvxStoreD::mul_f64_add(siq3, -f64::D4_R13_ROT_TWIDDLE_9, v3);
        u3 = AvxStoreD::mul_f64_add(iq4, f64::D4_R13_ROT_TWIDDLE_2, u3);
        v3 = AvxStoreD::mul_f64_add(siq4, f64::D4_R13_ROT_TWIDDLE_3, v3);
        u3 = AvxStoreD::mul_f64_add(iq5, f64::D4_R13_ROT_TWIDDLE_10, u3);
        v3 = AvxStoreD::mul_f64_add(siq5, f64::D4_R13_ROT_TWIDDLE_11, v3);
        u3 = u3 - a_v0;
        let uc4 = u3 - v3;
        let uc5 = u3 + v3;

        uc4.reverse_n::<N>()
            .write_n::<N>(data.slice_from_mut(3 * self.s - N - k..));
        uc5.write_n::<N>(data.slice_from_mut(3 * self.s + k..));

        let mut u4 = iq0;
        let mut v4 = siq0;
        u4 *= f64::D4_R13_ROT_TWIDDLE_5;
        v4 *= f64::D4_R13_ROT_TWIDDLE_4;
        u4 = AvxStoreD::mul_f64_add(iq1, -f64::D4_R13_ROT_TWIDDLE_2, u4);
        v4 = AvxStoreD::mul_f64_add(siq1, f64::D4_R13_ROT_TWIDDLE_3, v4);
        u4 = AvxStoreD::mul_f64_add(iq2, -f64::D4_R13_ROT_TWIDDLE_10, u4);
        v4 = AvxStoreD::mul_f64_add(siq2, -f64::D4_R13_ROT_TWIDDLE_11, v4);
        u4 = AvxStoreD::mul_f64_add(iq3, f64::D4_R13_ROT_TWIDDLE_6, u4);
        v4 = AvxStoreD::mul_f64_add(siq3, f64::D4_R13_ROT_TWIDDLE_7, v4);
        u4 = AvxStoreD::mul_f64_add(iq4, -f64::D4_R13_ROT_TWIDDLE_0, u4);
        v4 = AvxStoreD::mul_f64_add(siq4, f64::D4_R13_ROT_TWIDDLE_1, v4);
        u4 = AvxStoreD::mul_f64_add(iq5, -f64::D4_R13_ROT_TWIDDLE_8, u4);
        v4 = AvxStoreD::mul_f64_add(siq5, -f64::D4_R13_ROT_TWIDDLE_9, v4);
        u4 += a_v0;
        let uc6 = u4 - v4;
        let uc7 = u4 + v4;

        uc6.reverse_n::<N>()
            .write_n::<N>(data.slice_from_mut(4 * self.s - N - k..));
        uc7.write_n::<N>(data.slice_from_mut(4 * self.s + k..));

        let mut u5 = iq0;
        let mut v5 = siq0;
        u5 *= f64::D4_R13_ROT_TWIDDLE_8;
        v5 *= f64::D4_R13_ROT_TWIDDLE_9;
        u5 = AvxStoreD::mul_f64_add(iq1, -f64::D4_R13_ROT_TWIDDLE_6, u5);
        v5 = AvxStoreD::mul_f64_add(siq1, -f64::D4_R13_ROT_TWIDDLE_7, v5);
        u5 = AvxStoreD::mul_f64_add(iq2, f64::D4_R13_ROT_TWIDDLE_2, u5);
        v5 = AvxStoreD::mul_f64_add(siq2, -f64::D4_R13_ROT_TWIDDLE_3, v5);
        u5 = AvxStoreD::mul_f64_add(iq3, -f64::D4_R13_ROT_TWIDDLE_5, u5);
        v5 = AvxStoreD::mul_f64_add(siq3, f64::D4_R13_ROT_TWIDDLE_4, v5);
        u5 = AvxStoreD::mul_f64_add(iq4, f64::D4_R13_ROT_TWIDDLE_10, u5);
        v5 = AvxStoreD::mul_f64_add(siq4, -f64::D4_R13_ROT_TWIDDLE_11, v5);
        u5 = AvxStoreD::mul_f64_add(iq5, f64::D4_R13_ROT_TWIDDLE_0, u5);
        v5 = AvxStoreD::mul_f64_add(siq5, f64::D4_R13_ROT_TWIDDLE_1, v5);
        u5 = u5 - a_v0;
        let uc8 = u5 - v5;
        let uc9 = u5 + v5;

        uc8.reverse_n::<N>()
            .write_n::<N>(data.slice_from_mut(5 * self.s - N - k..));
        uc9.write_n::<N>(data.slice_from_mut(5 * self.s + k..));

        let mut u6 = iq0;
        let mut v6 = siq0;
        u6 *= -f64::D4_R13_ROT_TWIDDLE_10;
        v6 *= f64::D4_R13_ROT_TWIDDLE_11;
        u6 = AvxStoreD::mul_f64_add(iq1, -f64::D4_R13_ROT_TWIDDLE_8, u6);
        v6 = AvxStoreD::mul_f64_add(siq1, -f64::D4_R13_ROT_TWIDDLE_9, v6);
        u6 = AvxStoreD::mul_f64_add(iq2, f64::D4_R13_ROT_TWIDDLE_5, u6);
        v6 = AvxStoreD::mul_f64_add(siq2, f64::D4_R13_ROT_TWIDDLE_4, v6);
        u6 = AvxStoreD::mul_f64_add(iq3, -f64::D4_R13_ROT_TWIDDLE_0, u6);
        v6 = AvxStoreD::mul_f64_add(siq3, -f64::D4_R13_ROT_TWIDDLE_1, v6);
        u6 = AvxStoreD::mul_f64_add(iq4, f64::D4_R13_ROT_TWIDDLE_6, u6);
        v6 = AvxStoreD::mul_f64_add(siq4, f64::D4_R13_ROT_TWIDDLE_7, v6);
        u6 = AvxStoreD::mul_f64_add(iq5, -f64::D4_R13_ROT_TWIDDLE_2, u6);
        v6 = AvxStoreD::mul_f64_add(siq5, -f64::D4_R13_ROT_TWIDDLE_3, v6);
        u6 += a_v0;
        let uc10 = u6 - v6;
        let uc11 = u6 + v6;

        uc10.reverse_n::<N>()
            .write_n::<N>(data.slice_from_mut(6 * self.s - N - k..));
        uc11.write_n::<N>(data.slice_from_mut(6 * self.s + k..));
    }

    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_store<S: BidirectionalStore<f64>>(
        &self,
        data: &mut S,
        scratch: &mut [f64],
    ) -> Result<(), PxdctError> {
        let (scratch, inner_scratch) = scratch.split_at_mut(self.execution_length);
        let q_modules = self.execution_length / 13;
        let (a_buffer, c_s_buffer) = scratch.split_at_mut(q_modules);
        let (c_buffer, s_buffer) = c_s_buffer.split_at_mut(q_modules * 6);

        // Step 1: Decompose input into A (center), C (even-symmetric), S (odd-symmetric) buffers
        for (n, dst) in a_buffer.iter_mut().enumerate() {
            *dst = data[n * 13 + 6];
        }

        // Extract and combine symmetric pairs with sign alternation for S buffer
        for (m, (c_buffer, s_buffer)) in c_buffer
            .chunks_exact_mut(q_modules)
            .zip(s_buffer.chunks_exact_mut(q_modules))
            .enumerate()
        {
            let mut sign = f64::one();
            for (n, (c_dst, s_dst)) in c_buffer.iter_mut().zip(s_buffer.iter_mut()).enumerate() {
                let u0 = data[13 * n + m];
                let u1 = data[13 * n + 13 - m - 1];

                *c_dst = u0 + u1;
                *s_dst = (u0 - u1).mulsign(sign);

                sign = -sign;
            }
        }

        // Step 2: Apply DCT-IV to all buffers (A, C₀, C₁, S₀, S₁)
        self.inner_dct4
            .execute_with_scratch(scratch, inner_scratch)?;

        let (a_buffer, c_s_buffer) = scratch.split_at_mut(q_modules);
        let (c_buffer, s_buffer) = c_s_buffer.split_at_mut(q_modules * 6);

        let mut k = 0usize;
        let mut uk = 0usize;
        // Step 4: Handle k≥0 cases with rotation twiddles
        while k + 4 <= q_modules {
            self.exec_block::<S, 4>(data, a_buffer, s_buffer, c_buffer, uk, k);

            uk += 12;
            k += 4;
        }

        let rem = q_modules - k;
        if rem == 3 {
            self.exec_block::<S, 3>(data, a_buffer, s_buffer, c_buffer, uk, k);
        } else if rem == 2 {
            self.exec_block::<S, 2>(data, a_buffer, s_buffer, c_buffer, uk, k);
        } else if rem == 1 {
            self.exec_block::<S, 1>(data, a_buffer, s_buffer, c_buffer, uk, k);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::naive_dct4;
    use crate::type4::Dct4Butterfly2;
    use crate::util::has_valid_avx;
    use rand::RngExt;

    #[test]
    fn test_split_dct4() {
        if !has_valid_avx() {
            return;
        }
        let mut input = vec![0.; 26];
        for z in input.iter_mut() {
            *z = rand::rng().random_range(1.0..2.0);
        }
        let reference_input = input.clone();
        let reference_input = naive_dct4(&reference_input);
        let bf =
            AvxDct4MixedRadix13d::new(input.len(), Arc::new(Dct4Butterfly2::default())).unwrap();
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
