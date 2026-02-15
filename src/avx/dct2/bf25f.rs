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
use crate::avx::AvxDct2Butterfly5;
use crate::avx::storef::AvxStoreF;
use crate::avx::util::fma;
use crate::bidirectional::{BidirectionalStore, InPlaceStore};
use crate::dct2::MixedRadix5Sample;
use crate::util::DctSample;
use crate::{PxdctError, PxdctExecutor};
use num_traits::{One, Zero};

pub(crate) struct AvxDct2Butterfly25f {
    rotation_layer: [AvxStoreF; 4],
    cos_twiddles: [AvxStoreF; 4],
    bf5: AvxDct2Butterfly5<f32>,
}

impl Default for AvxDct2Butterfly25f {
    fn default() -> Self {
        unsafe { AvxDct2Butterfly25f::new() }
    }
}

impl AvxDct2Butterfly25f {
    #[target_feature(enable = "avx2")]
    pub(crate) fn new() -> Self {
        let twiddles = crate::dct2::Dct2Butterfly25Twiddles::default();
        let rotation = twiddles.rotation_layer;
        let cos_twiddles = twiddles.cos_twiddles;
        AvxDct2Butterfly25f {
            rotation_layer: [
                AvxStoreF::set_values4(
                    rotation[0].re,
                    rotation[2].re,
                    rotation[4].re,
                    rotation[6].re,
                ),
                AvxStoreF::set_values4(
                    rotation[0].im,
                    rotation[2].im,
                    rotation[4].im,
                    rotation[6].im,
                ),
                AvxStoreF::set_values4(
                    rotation[1].re,
                    rotation[3].re,
                    rotation[5].re,
                    rotation[7].re,
                ),
                AvxStoreF::set_values4(
                    rotation[1].im,
                    rotation[3].im,
                    rotation[5].im,
                    rotation[7].im,
                ),
            ],
            cos_twiddles: [
                AvxStoreF::set_values4(
                    cos_twiddles[0].re,
                    cos_twiddles[2].re,
                    cos_twiddles[4].re,
                    cos_twiddles[6].re,
                ),
                AvxStoreF::set_values4(
                    cos_twiddles[0].im,
                    cos_twiddles[2].im,
                    cos_twiddles[4].im,
                    cos_twiddles[6].im,
                ),
                AvxStoreF::set_values4(
                    cos_twiddles[1].re,
                    cos_twiddles[3].re,
                    cos_twiddles[5].re,
                    cos_twiddles[7].re,
                ),
                AvxStoreF::set_values4(
                    cos_twiddles[1].im,
                    cos_twiddles[3].im,
                    cos_twiddles[5].im,
                    cos_twiddles[7].im,
                ),
            ],
            bf5: AvxDct2Butterfly5::default(),
        }
    }
}

impl AvxDct2Butterfly25f {
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<f32>>(
        &self,
        data: &mut S,
        a_buffer: &mut [f32; 5],
        c_buffer: &mut [f32; 10],
        s_buffer: &mut [f32; 10],
    ) {
        for n in 0..5 {
            a_buffer[n] = data[n * 5 + 2];
        }

        self.bf5.exec(&mut InPlaceStore::new(a_buffer));

        for m in 0..2 {
            let mut sign = f32::one();
            for n in 0..5 {
                let u0 = data[5 * n + m];
                let u1 = data[5 * n + 5 - m - 1];

                c_buffer[m * 5 + n] = u0 + u1;
                s_buffer[m * 5 + n] = (u0 - u1).mulsign(sign);

                sign = -sign;
            }

            self.bf5
                .exec(&mut InPlaceStore::new(&mut c_buffer[m * 5..(m + 1) * 5]));
            self.bf5
                .exec(&mut InPlaceStore::new(&mut s_buffer[m * 5..(m + 1) * 5]));
        }

        unsafe {
            // first blocks
            let qc = c_buffer[0];
            let mut c0 = qc;
            let mut c1 = qc * f32::R5_COS_EVEN2_M0;
            let mut c2 = qc * f32::R5_COS_EVEN4_M0;

            let s0_twiddled = s_buffer[0];

            let mut s0 = s0_twiddled * f32::R5_SIN_ODD_M0;
            let mut s1 = s0_twiddled * f32::R5_SIN_ODD1_M0;

            {
                let ci = c_buffer[5];
                let si = s_buffer[5];
                let twiddle_ci = ci;
                let twiddle_si = si;

                c0 += ci;
                c1 = fma(twiddle_ci, f32::R5_COS_EVEN4_M0, c1);
                c2 = fma(twiddle_ci, f32::R5_COS_EVEN2_M0, c2);
                s0 = fma(twiddle_si, -f32::R5_SIN_ODD1_M0, s0);
                s1 = fma(twiddle_si, f32::R5_SIN_ODD_M0, s1);
            }

            let a0 = a_buffer[0];
            let dc = c0 + a0;
            data[0] = dc;

            let dc2 = c2 + a0;
            data[20] = dc2;
            data[15] = -s1;
            data[5] = s0;
            data[10] = -(c1 + a0);

            {
                let rotation_twiddle_re = self.rotation_layer[0];
                let rotation_twiddle_im = self.rotation_layer[1];

                let c_forward = AvxStoreF::load4(&c_buffer[1..]);
                let s_forward = AvxStoreF::load4(&s_buffer[1..]).reverse4();

                let twiddle_even = self.cos_twiddles[0];
                let twiddle_odd = self.cos_twiddles[1];

                let rotated_dc = fma(s_forward, rotation_twiddle_re, c_forward);

                let twiddled_dc = rotated_dc * twiddle_even;

                let mut dc0 = twiddled_dc;
                let mut dc2 = twiddled_dc * f32::R5_COS_EVEN2_M0;
                let mut dc4 = twiddled_dc * f32::R5_COS_EVEN4_M0;

                let rotated_ds = fma(c_forward, rotation_twiddle_im, s_forward);

                let twiddled_ds = rotated_ds * twiddle_odd;

                let mut ds1 = twiddled_ds * f32::R5_SIN_ODD_M0;
                let mut ds3 = twiddled_ds * f32::R5_SIN_ODD1_M0;

                {
                    let c_forward = AvxStoreF::load4(&c_buffer[6..]);
                    let s_forward = AvxStoreF::load4(&s_buffer[6..]).reverse4();

                    let rotation_twiddle_re = self.rotation_layer[2];
                    let rotation_twiddle_im = self.rotation_layer[3];

                    let rotated_dc1 = fma(s_forward, rotation_twiddle_re, c_forward);
                    let rotated_ds2 = fma(c_forward, rotation_twiddle_im, s_forward);

                    let twiddle_even = self.cos_twiddles[2];
                    let twiddle_odd = self.cos_twiddles[3];

                    let twiddled_dc = twiddle_even * rotated_dc1;
                    let twiddled_ds = twiddle_odd * rotated_ds2;

                    dc0 = twiddled_dc + dc0;
                    dc2 = AvxStoreF::f32_mul_add(f32::R5_COS_EVEN4_M0, twiddled_dc, dc2);
                    dc4 = AvxStoreF::f32_mul_add(f32::R5_COS_EVEN2_M0, twiddled_dc, dc4);

                    ds1 = AvxStoreF::f32_mul_nadd(f32::R5_SIN_ODD1_M0, twiddled_ds, ds1);
                    ds3 = AvxStoreF::f32_mul_add(f32::R5_SIN_ODD_M0, twiddled_ds, ds3);
                }

                let a0 = AvxStoreF::load4(&a_buffer[1..]);
                let dc = dc0 + a0;
                dc.write4(data.slice_from_mut(1..));

                let dss1 = AvxStoreF::f32_mul_add(2., ds1, -dc);
                let dss_reverse = dss1.reverse4();
                dss_reverse.write4(data.slice_from_mut(6..));

                dc2 = -(dc2 + a0); // negated 2j
                dc2 = AvxStoreF::f32_mul_add(2., dc2, -dss1);
                dc2.write4(data.slice_from_mut(11..));

                let dss3 = AvxStoreF::f32_mul_add(2., -ds3, -dc2);
                let dss3_reversed = dss3.reverse4();
                dss3_reversed.write4(data.slice_from_mut(16..));

                dc4 += a0;
                dc4 = AvxStoreF::f32_mul_add(2., dc4, -dss3);
                dc4.write4(data.slice_from_mut(21..));
            }
        }
    }
}

impl AvxDct2Butterfly25f {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_impl(&self, data: &mut [f32]) -> Result<(), PxdctError> {
        if !data.len().is_multiple_of(25) {
            return Err(PxdctError::InvalidSizeMultiplier(data.len(), self.length()));
        }

        let mut a_buffer = [f32::zero(); 5];
        let mut c_buffer = [f32::zero(); 10];
        let mut s_buffer = [f32::zero(); 10];

        for chunk in data.chunks_exact_mut(25) {
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
    fn execute_into_impl(&self, input: &[f32], output: &mut [f32]) -> Result<(), PxdctError> {
        use crate::util::validate_oof_sizes;
        validate_oof_sizes!(input, output, 25);

        let mut a_buffer = [f32::zero(); 5];
        let mut c_buffer = [f32::zero(); 10];
        let mut s_buffer = [f32::zero(); 10];

        use crate::bidirectional::BiStore;
        for (src, dst) in input.chunks_exact(25).zip(output.chunks_exact_mut(25)) {
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

impl PxdctExecutor<f32> for AvxDct2Butterfly25f {
    fn execute(&self, data: &mut [f32]) -> Result<(), PxdctError> {
        unsafe { self.execute_impl(data) }
    }

    fn execute_with_scratch(&self, data: &mut [f32], _: &mut [f32]) -> Result<(), PxdctError> {
        unsafe { self.execute_impl(data) }
    }

    fn execute_into(&self, input: &[f32], output: &mut [f32]) -> Result<(), PxdctError> {
        unsafe { self.execute_into_impl(input, output) }
    }

    fn execute_into_with_scratch(
        &self,
        input: &[f32],
        output: &mut [f32],
        _: &mut [f32],
    ) -> Result<(), PxdctError> {
        unsafe { self.execute_into_impl(input, output) }
    }

    fn length(&self) -> usize {
        25
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
    use crate::tests::naive_dct2_f32;

    gen_test_avx_butterfly!(test_bf25, AvxDct2Butterfly25f, 25, 1e-3, naive_dct2_f32);
}
