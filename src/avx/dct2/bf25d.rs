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
use crate::avx::stored::AvxStoreD;
use crate::avx::util::fma;
use crate::dct2::MixedRadix5Sample;
use crate::util::DctSample;
use crate::{PxdctError, PxdctExecutor};
use num_traits::One;

pub(crate) struct AvxDct2Butterfly25d {
    rotation_layer: [AvxStoreD; 4],
    cos_twiddles: [AvxStoreD; 4],
    bf5: AvxDct2Butterfly5<f64>,
}

impl Default for AvxDct2Butterfly25d {
    fn default() -> Self {
        unsafe { AvxDct2Butterfly25d::new() }
    }
}

impl AvxDct2Butterfly25d {
    #[target_feature(enable = "avx2")]
    pub(crate) fn new() -> Self {
        let twiddles = crate::dct2::Dct2Butterfly25Twiddles::default();
        let rotation = twiddles.rotation_layer;
        let cos_twiddles = twiddles.cos_twiddles;
        AvxDct2Butterfly25d {
            rotation_layer: [
                AvxStoreD::set_values(
                    rotation[0].re,
                    rotation[2].re,
                    rotation[4].re,
                    rotation[6].re,
                ),
                AvxStoreD::set_values(
                    rotation[0].im,
                    rotation[2].im,
                    rotation[4].im,
                    rotation[6].im,
                ),
                AvxStoreD::set_values(
                    rotation[1].re,
                    rotation[3].re,
                    rotation[5].re,
                    rotation[7].re,
                ),
                AvxStoreD::set_values(
                    rotation[1].im,
                    rotation[3].im,
                    rotation[5].im,
                    rotation[7].im,
                ),
            ],
            cos_twiddles: [
                AvxStoreD::set_values(
                    cos_twiddles[0].re,
                    cos_twiddles[2].re,
                    cos_twiddles[4].re,
                    cos_twiddles[6].re,
                ),
                AvxStoreD::set_values(
                    cos_twiddles[0].im,
                    cos_twiddles[2].im,
                    cos_twiddles[4].im,
                    cos_twiddles[6].im,
                ),
                AvxStoreD::set_values(
                    cos_twiddles[1].re,
                    cos_twiddles[3].re,
                    cos_twiddles[5].re,
                    cos_twiddles[7].re,
                ),
                AvxStoreD::set_values(
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

impl AvxDct2Butterfly25d {
    #[inline(always)]
    pub(crate) fn exec(
        &self,
        data: &mut [f64; 25],
        a_buffer: &mut [f64; 5],
        c_buffer: &mut [f64; 10],
        s_buffer: &mut [f64; 10],
    ) {
        for n in 0..5 {
            a_buffer[n] = data[n * 5 + 2];
        }

        self.bf5.exec(a_buffer);

        for m in 0..2 {
            let mut sign = f64::one();
            for n in 0..5 {
                let u0 = data[5 * n + m];
                let u1 = data[5 * n + 5 - m - 1];

                c_buffer[m * 5 + n] = u0 + u1;
                s_buffer[m * 5 + n] = (u0 - u1).mulsign(sign);

                sign = -sign;
            }

            self.bf5
                .exec((&mut c_buffer[m * 5..(m + 1) * 5]).try_into().unwrap());
            self.bf5
                .exec((&mut s_buffer[m * 5..(m + 1) * 5]).try_into().unwrap());
        }

        unsafe {
            // first blocks
            let qc = c_buffer[0];
            let mut c0 = qc;
            let mut c1 = qc * f64::R5_COS_EVEN2_M0;
            let mut c2 = qc * f64::R5_COS_EVEN4_M0;

            let s0_twiddled = s_buffer[0];

            let mut s0 = s0_twiddled * f64::R5_SIN_ODD_M0;
            let mut s1 = s0_twiddled * f64::R5_SIN_ODD1_M0;

            {
                let ci = c_buffer[5];
                let si = s_buffer[5];
                let twiddle_ci = ci;
                let twiddle_si = si;

                c0 += ci;
                c1 = fma(twiddle_ci, f64::R5_COS_EVEN4_M0, c1);
                c2 = fma(twiddle_ci, f64::R5_COS_EVEN2_M0, c2);
                s0 = fma(twiddle_si, -f64::R5_SIN_ODD1_M0, s0);
                s1 = fma(twiddle_si, f64::R5_SIN_ODD_M0, s1);
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

                let c_forward = AvxStoreD::load(&c_buffer[1..]);
                let s_forward = AvxStoreD::load(&s_buffer[1..]).reverse();

                let twiddle_even = self.cos_twiddles[0];
                let twiddle_odd = self.cos_twiddles[1];

                let rotated_dc = fma(s_forward, rotation_twiddle_re, c_forward);

                let twiddled_dc = rotated_dc * twiddle_even;

                let mut dc0 = twiddled_dc;
                let mut dc2 = twiddled_dc * f64::R5_COS_EVEN2_M0;
                let mut dc4 = twiddled_dc * f64::R5_COS_EVEN4_M0;

                let rotated_ds = fma(c_forward, rotation_twiddle_im, s_forward);

                let twiddled_ds = rotated_ds * twiddle_odd;

                let mut ds1 = twiddled_ds * f64::R5_SIN_ODD_M0;
                let mut ds3 = twiddled_ds * f64::R5_SIN_ODD1_M0;

                {
                    let c_forward = AvxStoreD::load(&c_buffer[6..]);
                    let s_forward = AvxStoreD::load(&s_buffer[6..]).reverse();

                    let rotation_twiddle_re = self.rotation_layer[2];
                    let rotation_twiddle_im = self.rotation_layer[3];

                    let rotated_dc1 = fma(s_forward, rotation_twiddle_re, c_forward);
                    let rotated_ds2 = fma(c_forward, rotation_twiddle_im, s_forward);

                    let twiddle_even = self.cos_twiddles[2];
                    let twiddle_odd = self.cos_twiddles[3];

                    let twiddled_dc = twiddle_even * rotated_dc1;
                    let twiddled_ds = twiddle_odd * rotated_ds2;

                    dc0 = twiddled_dc + dc0;
                    dc2 = AvxStoreD::f64_mul_add(f64::R5_COS_EVEN4_M0, twiddled_dc, dc2);
                    dc4 = AvxStoreD::f64_mul_add(f64::R5_COS_EVEN2_M0, twiddled_dc, dc4);

                    ds1 = AvxStoreD::f64_mul_nadd(f64::R5_SIN_ODD1_M0, twiddled_ds, ds1);
                    ds3 = AvxStoreD::f64_mul_add(f64::R5_SIN_ODD_M0, twiddled_ds, ds3);
                }

                let a0 = AvxStoreD::load(&a_buffer[1..]);
                let dc = dc0 + a0;
                dc.write(&mut data[1..]);

                let dss1 = AvxStoreD::f64_mul_add(2., ds1, -dc);
                let dss_reverse = dss1.reverse();
                dss_reverse.write(&mut data[6..]);

                dc2 = -(dc2 + a0); // negated 2j
                dc2 = AvxStoreD::f64_mul_add(2., dc2, -dss1);
                dc2.write(&mut data[11..]);

                let dss3 = AvxStoreD::f64_mul_add(2., -ds3, -dc2);
                let dss3_reversed = dss3.reverse();
                dss3_reversed.write(&mut data[16..]);

                dc4 += a0;
                dc4 = AvxStoreD::f64_mul_add(2., dc4, -dss3);
                dc4.write(&mut data[21..]);
            }
        }
    }
}

impl AvxDct2Butterfly25d {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_impl(&self, data: &mut [f64]) -> Result<(), PxdctError> {
        if !data.len().is_multiple_of(25) {
            return Err(PxdctError::InvalidSizeMultiplier(data.len(), self.length()));
        }

        let mut a_buffer = [f64::default(); 5];
        let mut c_buffer = [f64::default(); 10];
        let mut s_buffer = [f64::default(); 10];

        for chunk in data.chunks_exact_mut(25) {
            self.exec(
                (&mut chunk[..25]).try_into().unwrap(),
                &mut a_buffer,
                &mut c_buffer,
                &mut s_buffer,
            );
        }
        Ok(())
    }
}

impl PxdctExecutor<f64> for AvxDct2Butterfly25d {
    fn execute(&self, data: &mut [f64]) -> Result<(), PxdctError> {
        unsafe { self.execute_impl(data) }
    }

    fn length(&self) -> usize {
        25
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PxdctExecutor;
    use crate::avx::dct2_bf_power2::gen_test_avx_butterfly;
    use crate::tests::naive_dct2;

    gen_test_avx_butterfly!(test_bf25, AvxDct2Butterfly25d, 25, 1e-7, naive_dct2);
}
