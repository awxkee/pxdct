/*
 * // Copyright (c) Radzivon Bartoshyk 12/2025. All rights reserved.
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
use crate::avx::stored::AvxStoreD;
use crate::avx::util::{boring_avx_mixed_radix, fma};
use crate::bidirectional::BidirectionalStore;
use crate::util::{DctConstants, mixed_radix_inner_twiddle, try_vec, validate_scratch};
use crate::{PxdctError, PxdctExecutor};
use num_traits::AsPrimitive;
use std::sync::Arc;

pub(crate) struct AvxDct2MixedRadix6d {
    inner_layer: Vec<AvxStoreD>,
    sixth_dct: Arc<dyn PxdctExecutor<f64> + Send + Sync>,
    inner_dct_scratch_size: usize,
    execution_length: usize,
}

#[target_feature(enable = "avx2")]
pub(crate) fn dct2_radix6_avx_groupd(len: usize) -> Vec<AvxStoreD> {
    let inner_layer_groups = len / 6;
    let simd_groups = inner_layer_groups.div_ceil(4);
    let mut inner_layer = Vec::with_capacity(simd_groups * 7);

    for g in 0..simd_groups {
        let mut re0 = [0f64; 4];
        let mut im0 = [0f64; 4];
        let mut re1 = [0f64; 4];
        let mut im1 = [0f64; 4];
        let mut re2 = [0f64; 4];
        let mut re3 = [0f64; 4];
        let mut im3 = [0f64; 4];

        for lane in 0..4 {
            let i = g * 4 + lane;
            if i < inner_layer_groups {
                let angle = (2. * i as f64 + 1.).as_();
                let t0 = mixed_radix_inner_twiddle(angle, len);
                let t1 = mixed_radix_inner_twiddle(2.0 * angle, len);
                let t2 = mixed_radix_inner_twiddle(3.0 * angle, len);
                let t3 = mixed_radix_inner_twiddle(5.0 * angle, len);

                re0[lane] = t0.re;
                im0[lane] = t0.im * f64::SQRT_3;

                re1[lane] = t1.re;
                im1[lane] = t1.im * f64::SQRT_3;

                re2[lane] = t2.re;

                re3[lane] = t3.re;
                im3[lane] = -t3.im * f64::SQRT_3;
            }
        }

        inner_layer.push(AvxStoreD::load(&re0));
        inner_layer.push(AvxStoreD::load(&im0));
        inner_layer.push(AvxStoreD::load(&re1));
        inner_layer.push(AvxStoreD::load(&im1));
        inner_layer.push(AvxStoreD::load(&re2));
        inner_layer.push(AvxStoreD::load(&re3));
        inner_layer.push(AvxStoreD::load(&im3));
    }
    inner_layer
}

impl AvxDct2MixedRadix6d {
    pub(crate) fn new(
        len: usize,
        sixth_dct: Arc<dyn PxdctExecutor<f64> + Send + Sync>,
    ) -> Result<AvxDct2MixedRadix6d, PxdctError> {
        assert_eq!(
            len,
            sixth_dct.length() * 6,
            "Invalid DCT was received, third size is not multiple of full size"
        );

        let sixth_dct_scratch_size = sixth_dct.scratch_size();

        Ok(AvxDct2MixedRadix6d {
            inner_layer: unsafe { dct2_radix6_avx_groupd(len) },
            inner_dct_scratch_size: sixth_dct_scratch_size,
            sixth_dct,
            execution_length: len,
        })
    }
}

boring_avx_mixed_radix!(AvxDct2MixedRadix6d, f64);

impl AvxDct2MixedRadix6d {
    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_store<S: BidirectionalStore<f64>>(
        &self,
        data: &mut S,
        scratch: &mut [f64],
    ) -> Result<(), PxdctError> {
        let (scratch, inner_scratch) = scratch.split_at_mut(self.execution_length);

        let len = self.length();
        let sixth_length = self.sixth_dct.length();
        let s_n = len / 3;
        let s_2n = 2 * len / 3;
        let (a_buffer, rem) = scratch.split_at_mut(sixth_length);
        let (b_buffer, rem) = rem.split_at_mut(sixth_length);
        let (c_buffer, rem) = rem.split_at_mut(sixth_length);
        let (d_buffer, rem) = rem.split_at_mut(sixth_length);
        let (e_buffer, rem) = rem.split_at_mut(sixth_length);
        let (f_buffer, _) = rem.split_at_mut(sixth_length);

        let mut j = 0usize;

        let mut twiddle_idx = 0usize;

        while j + 4 <= sixth_length {
            let ai = AvxStoreD::load(data.slice_from(j..));
            let mut bi = AvxStoreD::load(data.slice_from(s_n - j - 4..));
            let ci = AvxStoreD::load(data.slice_from(s_n + j..));
            let mut di = AvxStoreD::load(data.slice_from(s_2n - j - 4..));
            let ei = AvxStoreD::load(data.slice_from(s_2n + j..));
            let mut fi = AvxStoreD::load(data.slice_from(len - j - 4..));

            bi = bi.reverse();
            di = di.reverse();
            fi = fi.reverse();

            let cos_sin_ai_re = unsafe { *self.inner_layer.get_unchecked(twiddle_idx) };
            let cos_sin_ai_im = unsafe { *self.inner_layer.get_unchecked(twiddle_idx + 1) };
            let cos_sin_2ai_re = unsafe { *self.inner_layer.get_unchecked(twiddle_idx + 2) };
            let cos_sin_2ai_im = unsafe { *self.inner_layer.get_unchecked(twiddle_idx + 3) };
            let cos_sin_3ai_re = unsafe { *self.inner_layer.get_unchecked(twiddle_idx + 4) };
            let cos_sin_5ai_re = unsafe { *self.inner_layer.get_unchecked(twiddle_idx + 5) };
            let cos_sin_5ai_im = unsafe { *self.inner_layer.get_unchecked(twiddle_idx + 6) };

            let s2 = bi + ei;
            let dcd = ci - di;
            let dbe = bi - ei;

            let ai2 = f64::TWO * ai;
            let fi2 = f64::TWO * fi;
            let scd = ci + di;

            let sdbedcd = dbe + dcd;
            let ai2dbedcd = ai2 + sdbedcd - fi2;

            let s2scd = s2 + scd;

            let a_comp = ai + s2scd + fi;
            let c_comp = ai2 - s2scd + fi2;
            let d_comp = f64::TWO * (ai - sdbedcd - fi);

            let dbedcd = dbe - dcd;

            let c_img = s2 - ci - di;
            let b_zet = dbedcd * cos_sin_ai_im;
            let c_zet = c_img * cos_sin_2ai_im;
            let f_zet = dbedcd * cos_sin_5ai_im;

            let e_comp = fma(
                f64::TWO * cos_sin_2ai_re,
                fma(c_comp, cos_sin_2ai_re, -c_zet),
                -c_comp,
            );

            unsafe {
                a_comp.write(a_buffer.get_unchecked_mut(j..));
                let q0 = fma(ai2dbedcd, cos_sin_ai_re, b_zet);
                q0.write(b_buffer.get_unchecked_mut(j..));
                let q1 = fma(c_comp, cos_sin_2ai_re, c_zet);
                q1.write(c_buffer.get_unchecked_mut(j..));
                let q2 = d_comp * cos_sin_3ai_re;
                q2.write(d_buffer.get_unchecked_mut(j..));
                e_comp.write(e_buffer.get_unchecked_mut(j..));
                let q3 = fma(ai2dbedcd, cos_sin_5ai_re, f_zet);
                q3.write(f_buffer.get_unchecked_mut(j..));
            }
            j += 4;
            twiddle_idx += 7;
        }

        let rem = sixth_length - j;

        match rem {
            3 => {
                let ai = AvxStoreD::load3(data.slice_from(j..));
                let mut bi = AvxStoreD::load3(data.slice_from(s_n - j - 3..));
                let ci = AvxStoreD::load3(data.slice_from(s_n + j..));
                let mut di = AvxStoreD::load3(data.slice_from(s_2n - j - 3..));
                let ei = AvxStoreD::load3(data.slice_from(s_2n + j..));
                let mut fi = AvxStoreD::load3(data.slice_from(len - j - 3..));

                bi = bi.reverse3();
                di = di.reverse3();
                fi = fi.reverse3();

                let cos_sin_ai_re = unsafe { *self.inner_layer.get_unchecked(twiddle_idx) };
                let cos_sin_ai_im = unsafe { *self.inner_layer.get_unchecked(twiddle_idx + 1) };
                let cos_sin_2ai_re = unsafe { *self.inner_layer.get_unchecked(twiddle_idx + 2) };
                let cos_sin_2ai_im = unsafe { *self.inner_layer.get_unchecked(twiddle_idx + 3) };
                let cos_sin_3ai_re = unsafe { *self.inner_layer.get_unchecked(twiddle_idx + 4) };
                let cos_sin_5ai_re = unsafe { *self.inner_layer.get_unchecked(twiddle_idx + 5) };
                let cos_sin_5ai_im = unsafe { *self.inner_layer.get_unchecked(twiddle_idx + 6) };

                let s2 = bi + ei;
                let dcd = ci - di;
                let dbe = bi - ei;

                let ai2 = f64::TWO * ai;
                let fi2 = f64::TWO * fi;
                let scd = ci + di;

                let sdbedcd = dbe + dcd;
                let ai2dbedcd = ai2 + sdbedcd - fi2;

                let s2scd = s2 + scd;

                let a_comp = ai + s2scd + fi;
                let c_comp = ai2 - s2scd + fi2;
                let d_comp = f64::TWO * (ai - sdbedcd - fi);

                let dbedcd = dbe - dcd;

                let c_img = s2 - ci - di;
                let b_zet = dbedcd * cos_sin_ai_im;
                let c_zet = c_img * cos_sin_2ai_im;
                let f_zet = dbedcd * cos_sin_5ai_im;

                let e_comp = fma(
                    f64::TWO * cos_sin_2ai_re,
                    fma(c_comp, cos_sin_2ai_re, -c_zet),
                    -c_comp,
                );

                unsafe {
                    a_comp.write3(a_buffer.get_unchecked_mut(j..));
                    let q0 = fma(ai2dbedcd, cos_sin_ai_re, b_zet);
                    q0.write3(b_buffer.get_unchecked_mut(j..));
                    let q1 = fma(c_comp, cos_sin_2ai_re, c_zet);
                    q1.write3(c_buffer.get_unchecked_mut(j..));
                    let q2 = d_comp * cos_sin_3ai_re;
                    q2.write3(d_buffer.get_unchecked_mut(j..));
                    e_comp.write3(e_buffer.get_unchecked_mut(j..));
                    let q3 = fma(ai2dbedcd, cos_sin_5ai_re, f_zet);
                    q3.write3(f_buffer.get_unchecked_mut(j..));
                }
            }
            2 => {
                let ai = AvxStoreD::load2(data.slice_from(j..));
                let mut bi = AvxStoreD::load2(data.slice_from(s_n - j - 2..));
                let ci = AvxStoreD::load2(data.slice_from(s_n + j..));
                let mut di = AvxStoreD::load2(data.slice_from(s_2n - j - 2..));
                let ei = AvxStoreD::load2(data.slice_from(s_2n + j..));
                let mut fi = AvxStoreD::load2(data.slice_from(len - j - 2..));

                bi = bi.reverse2();
                di = di.reverse2();
                fi = fi.reverse2();

                let cos_sin_ai_re = unsafe { *self.inner_layer.get_unchecked(twiddle_idx) };
                let cos_sin_ai_im = unsafe { *self.inner_layer.get_unchecked(twiddle_idx + 1) };
                let cos_sin_2ai_re = unsafe { *self.inner_layer.get_unchecked(twiddle_idx + 2) };
                let cos_sin_2ai_im = unsafe { *self.inner_layer.get_unchecked(twiddle_idx + 3) };
                let cos_sin_3ai_re = unsafe { *self.inner_layer.get_unchecked(twiddle_idx + 4) };
                let cos_sin_5ai_re = unsafe { *self.inner_layer.get_unchecked(twiddle_idx + 5) };
                let cos_sin_5ai_im = unsafe { *self.inner_layer.get_unchecked(twiddle_idx + 6) };

                let s2 = bi + ei;
                let dcd = ci - di;
                let dbe = bi - ei;

                let ai2 = f64::TWO * ai;
                let fi2 = f64::TWO * fi;
                let scd = ci + di;

                let sdbedcd = dbe + dcd;
                let ai2dbedcd = ai2 + sdbedcd - fi2;

                let s2scd = s2 + scd;

                let a_comp = ai + s2scd + fi;
                let c_comp = ai2 - s2scd + fi2;
                let d_comp = f64::TWO * (ai - sdbedcd - fi);

                let dbedcd = dbe - dcd;

                let c_img = s2 - ci - di;
                let b_zet = dbedcd * cos_sin_ai_im;
                let c_zet = c_img * cos_sin_2ai_im;
                let f_zet = dbedcd * cos_sin_5ai_im;

                let e_comp = fma(
                    f64::TWO * cos_sin_2ai_re,
                    fma(c_comp, cos_sin_2ai_re, -c_zet),
                    -c_comp,
                );

                unsafe {
                    a_comp.write2(a_buffer.get_unchecked_mut(j..));
                    let q0 = fma(ai2dbedcd, cos_sin_ai_re, b_zet);
                    q0.write2(b_buffer.get_unchecked_mut(j..));
                    let q1 = fma(c_comp, cos_sin_2ai_re, c_zet);
                    q1.write2(c_buffer.get_unchecked_mut(j..));
                    let q2 = d_comp * cos_sin_3ai_re;
                    q2.write2(d_buffer.get_unchecked_mut(j..));
                    e_comp.write2(e_buffer.get_unchecked_mut(j..));
                    let q3 = fma(ai2dbedcd, cos_sin_5ai_re, f_zet);
                    q3.write2(f_buffer.get_unchecked_mut(j..));
                }
            }
            1 => {
                let ai = AvxStoreD::load1(data.slice_from(j..));
                let bi = AvxStoreD::load1(data.slice_from(s_n - j - 1..));
                let ci = AvxStoreD::load1(data.slice_from(s_n + j..));
                let di = AvxStoreD::load1(data.slice_from(s_2n - j - 1..));
                let ei = AvxStoreD::load1(data.slice_from(s_2n + j..));
                let fi = AvxStoreD::load1(data.slice_from(len - j - 1..));

                let cos_sin_ai_re = unsafe { *self.inner_layer.get_unchecked(twiddle_idx) };
                let cos_sin_ai_im = unsafe { *self.inner_layer.get_unchecked(twiddle_idx + 1) };
                let cos_sin_2ai_re = unsafe { *self.inner_layer.get_unchecked(twiddle_idx + 2) };
                let cos_sin_2ai_im = unsafe { *self.inner_layer.get_unchecked(twiddle_idx + 3) };
                let cos_sin_3ai_re = unsafe { *self.inner_layer.get_unchecked(twiddle_idx + 4) };
                let cos_sin_5ai_re = unsafe { *self.inner_layer.get_unchecked(twiddle_idx + 5) };
                let cos_sin_5ai_im = unsafe { *self.inner_layer.get_unchecked(twiddle_idx + 6) };

                let s2 = bi + ei;
                let dcd = ci - di;
                let dbe = bi - ei;

                let ai2 = f64::TWO * ai;
                let fi2 = f64::TWO * fi;
                let scd = ci + di;

                let sdbedcd = dbe + dcd;
                let ai2dbedcd = ai2 + sdbedcd - fi2;

                let s2scd = s2 + scd;

                let a_comp = ai + s2scd + fi;
                let c_comp = ai2 - s2scd + fi2;
                let d_comp = f64::TWO * (ai - sdbedcd - fi);

                let dbedcd = dbe - dcd;

                let c_img = s2 - ci - di;
                let b_zet = dbedcd * cos_sin_ai_im;
                let c_zet = c_img * cos_sin_2ai_im;
                let f_zet = dbedcd * cos_sin_5ai_im;

                let e_comp = fma(
                    f64::TWO * cos_sin_2ai_re,
                    fma(c_comp, cos_sin_2ai_re, -c_zet),
                    -c_comp,
                );

                unsafe {
                    a_comp.write1(a_buffer.get_unchecked_mut(j..));
                    let q0 = fma(ai2dbedcd, cos_sin_ai_re, b_zet);
                    q0.write1(b_buffer.get_unchecked_mut(j..));
                    let q1 = fma(c_comp, cos_sin_2ai_re, c_zet);
                    q1.write1(c_buffer.get_unchecked_mut(j..));
                    let q2 = d_comp * cos_sin_3ai_re;
                    q2.write1(d_buffer.get_unchecked_mut(j..));
                    e_comp.write1(e_buffer.get_unchecked_mut(j..));
                    let q3 = fma(ai2dbedcd, cos_sin_5ai_re, f_zet);
                    q3.write1(f_buffer.get_unchecked_mut(j..));
                }
            }
            _ => {}
        }

        if a_buffer.len() > 1 {
            self.sixth_dct
                .execute_with_scratch(scratch, inner_scratch)?;
        }

        let (a_buffer, rem) = scratch.split_at_mut(sixth_length);
        let (b_buffer, rem) = rem.split_at_mut(sixth_length);
        let (c_buffer, rem) = rem.split_at_mut(sixth_length);
        let (d_buffer, rem) = rem.split_at_mut(sixth_length);
        let (e_buffer, rem) = rem.split_at_mut(sixth_length);
        let (f_buffer, _) = rem.split_at_mut(sixth_length);

        data[0] = a_buffer[0];
        let b0 = b_buffer[0] * f64::HALF;
        data[1] = b0;
        let c0 = c_buffer[0] * f64::HALF;
        data[2] = c0;
        let d0 = d_buffer[0] * f64::HALF;
        data[3] = d0;
        let e0 = e_buffer[0] * f64::HALF;
        data[4] = e0;
        let f0 = f_buffer[0] * f64::HALF;
        data[5] = f0;

        let mut b_diff = f0;
        let mut c_diff = e0;
        let mut e_diff = d0;
        let mut d_diff = c0;
        let mut f_diff = b0;

        for k in 1..sixth_length {
            let deferred_d_diff;
            let deferred_f_diff;
            unsafe {
                data[6 * k] = *a_buffer.get_unchecked(k);
            }
            unsafe {
                deferred_f_diff = *b_buffer.get_unchecked(k) - b_diff;
                data[6 * k + 1] = deferred_f_diff;
            }
            unsafe {
                deferred_d_diff = *c_buffer.get_unchecked(k) - c_diff;
                data[6 * k + 2] = deferred_d_diff;
            }
            unsafe {
                e_diff = *d_buffer.get_unchecked(k) - e_diff;
                data[6 * k + 3] = e_diff;
            }
            unsafe {
                let new_d = *e_buffer.get_unchecked(k) - d_diff;
                data[6 * k + 4] = new_d;
                c_diff = new_d;
                d_diff = deferred_d_diff;
            }
            unsafe {
                let new_f = *f_buffer.get_unchecked(k) - f_diff;
                b_diff = new_f;
                f_diff = deferred_f_diff;
                data[6 * k + 5] = new_f;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::Dct2Butterfly6;
    use crate::tests::naive_dct2;
    use crate::util::has_valid_avx;

    #[test]
    fn test_radix6_dct2() {
        if !has_valid_avx() {
            return;
        }
        let mut input = vec![0.; 36];
        for (i, z) in input.iter_mut().enumerate() {
            *z = i as f64;
        }
        let reference_input = input.clone();
        let reference_input = naive_dct2(&reference_input);
        let bf = AvxDct2MixedRadix6d::new(36, Arc::new(Dct2Butterfly6::default())).unwrap();
        bf.execute(&mut input).unwrap();

        input
            .iter()
            .zip(reference_input.iter())
            .enumerate()
            .for_each(|(i, (&src, &r0))| {
                assert!(
                    (src - r0).abs() < 1e-3,
                    "Difference must be < {}, but it was {}, at position {i}",
                    1e-7,
                    (src - r0).abs()
                )
            });
    }
}
