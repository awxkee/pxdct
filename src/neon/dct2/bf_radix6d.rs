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
use crate::butterflies::Dct2Butterfly6;
use crate::factory_dct2::Dct2Factory;
use crate::mla::fmla;
use crate::neon::dct2::mixed_radix6d::dct2_radix6_neon_groupd;
use crate::neon::store_d::NeonStoreD;
use crate::util::DctConstants;
use crate::{PxdctError, PxdctExecutor};
use num_traits::Zero;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub(crate) struct NeonDct2Butterfly36d {
    inner_layer: [NeonStoreD; 21],
    bf6: Dct2Butterfly6<f64>,
}

impl Default for NeonDct2Butterfly36d {
    fn default() -> Self {
        let layer = dct2_radix6_neon_groupd(36).try_into().unwrap();
        Self {
            inner_layer: layer,
            bf6: Dct2Butterfly6::default(),
        }
    }
}

impl NeonDct2Butterfly36d {
    #[inline(always)]
    fn exec(
        &self,
        data: &mut [f64; 36],
        a_buffer: &mut [f64; 6],
        b_buffer: &mut [f64; 6],
        c_buffer: &mut [f64; 6],
        d_buffer: &mut [f64; 6],
        e_buffer: &mut [f64; 6],
        f_buffer: &mut [f64; 6],
    ) {
        let s_n = 36 / 3;
        let s_2n = 2 * 36 / 3;

        let mut layer_i = 0;

        for i in (0..6).step_by(2) {
            let ai = NeonStoreD::load(unsafe { data.get_unchecked(i..) });
            let mut bi = NeonStoreD::load(unsafe { data.get_unchecked(s_n - i - 2..) });
            let ci = NeonStoreD::load(unsafe { data.get_unchecked(s_n + i..) });
            let mut di = NeonStoreD::load(unsafe { data.get_unchecked(s_2n - i - 2..) });
            let ei = NeonStoreD::load(unsafe { data.get_unchecked(s_2n + i..) });
            let mut fi = NeonStoreD::load(unsafe { data.get_unchecked(36 - i - 2..) });

            bi = bi.reverse();
            di = di.reverse();
            fi = fi.reverse();

            let cos_sin_ai_re = unsafe { *self.inner_layer.get_unchecked(layer_i) };
            let cos_sin_ai_im = unsafe { *self.inner_layer.get_unchecked(layer_i + 1) };
            let cos_sin_2ai_re = unsafe { *self.inner_layer.get_unchecked(layer_i + 2) };
            let cos_sin_2ai_im = unsafe { *self.inner_layer.get_unchecked(layer_i + 3) };
            let cos_sin_3ai_re = unsafe { *self.inner_layer.get_unchecked(layer_i + 4) };
            let cos_sin_5ai_re = unsafe { *self.inner_layer.get_unchecked(layer_i + 5) };
            let cos_sin_5ai_im = unsafe { *self.inner_layer.get_unchecked(layer_i + 6) };

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

            let e_comp = fmla(
                f64::TWO * cos_sin_2ai_re,
                fmla(c_comp, cos_sin_2ai_re, -c_zet),
                -c_comp,
            );

            unsafe {
                a_comp.write(a_buffer.get_unchecked_mut(i..));
                let q0 = fmla(ai2dbedcd, cos_sin_ai_re, b_zet);
                q0.write(b_buffer.get_unchecked_mut(i..));
                let q1 = fmla(c_comp, cos_sin_2ai_re, c_zet);
                q1.write(c_buffer.get_unchecked_mut(i..));
                let q2 = d_comp * cos_sin_3ai_re;
                q2.write(d_buffer.get_unchecked_mut(i..));
                e_comp.write(e_buffer.get_unchecked_mut(i..));
                let q3 = fmla(ai2dbedcd, cos_sin_5ai_re, f_zet);
                q3.write(f_buffer.get_unchecked_mut(i..));
            }

            layer_i += 7;
        }

        self.bf6.exec(a_buffer);
        self.bf6.exec(b_buffer);
        self.bf6.exec(c_buffer);
        self.bf6.exec(d_buffer);
        self.bf6.exec(e_buffer);
        self.bf6.exec(f_buffer);

        data[0] = a_buffer[0];
        data[1] = b_buffer[0] * f64::HALF;
        data[2] = c_buffer[0] * f64::HALF;
        data[3] = d_buffer[0] * f64::HALF;
        data[4] = e_buffer[0] * f64::HALF;
        data[5] = f_buffer[0] * f64::HALF;

        let mut b_diff = data[5];
        let mut c_diff = data[4];
        let mut e_diff = data[3];
        let mut d_diff = data[2];
        let mut f_diff = data[1];

        for k in 1..6 {
            data[6 * k] = a_buffer[k];
            let deferred_f_diff = b_buffer[k] - b_diff;
            data[6 * k + 1] = deferred_f_diff;
            let deferred_d_diff = c_buffer[k] - c_diff;
            data[6 * k + 2] = deferred_d_diff;
            e_diff = d_buffer[k] - e_diff;
            data[6 * k + 3] = e_diff;
            let new_d = e_buffer[k] - d_diff;
            data[6 * k + 4] = new_d;
            c_diff = new_d;
            d_diff = deferred_d_diff;
            let new_f = f_buffer[k] - f_diff;
            b_diff = new_f;
            f_diff = deferred_f_diff;
            data[6 * k + 5] = new_f;
        }
    }
}

impl PxdctExecutor<f64> for NeonDct2Butterfly36d {
    fn execute(&self, data: &mut [f64]) -> Result<(), PxdctError> {
        if !data.len().is_multiple_of(36) {
            return Err(PxdctError::InvalidSizeMultiplier(data.len(), self.length()));
        }

        let mut a_buffer = [f64::zero(); 6];
        let mut b_buffer = [f64::zero(); 6];
        let mut c_buffer = [f64::zero(); 6];
        let mut d_buffer = [f64::zero(); 6];
        let mut e_buffer = [f64::zero(); 6];
        let mut f_buffer = [f64::zero(); 6];

        for chunk in data.chunks_exact_mut(36) {
            self.exec(
                (&mut chunk[..36]).try_into().unwrap(),
                &mut a_buffer,
                &mut b_buffer,
                &mut c_buffer,
                &mut d_buffer,
                &mut e_buffer,
                &mut f_buffer,
            );
        }
        Ok(())
    }

    fn length(&self) -> usize {
        36
    }
}

#[derive(Clone)]
pub(crate) struct NeonDct2Butterfly216d {
    inner_layer: [NeonStoreD; 126],
    bf36: Arc<dyn PxdctExecutor<f64> + Send + Sync>,
}

impl Default for NeonDct2Butterfly216d {
    fn default() -> Self {
        Self {
            inner_layer: dct2_radix6_neon_groupd(216).try_into().unwrap(),
            bf36: f64::dct2_butterfly36(),
        }
    }
}

impl NeonDct2Butterfly216d {
    #[inline(always)]
    fn exec(&self, data: &mut [f64; 216], scratch: &mut [f64; 216]) {
        let s_n = 216 / 3;
        let s_2n = 2 * 216 / 3;

        let (a_buffer, rem) = scratch.split_at_mut(36);
        let (b_buffer, rem) = rem.split_at_mut(36);
        let (c_buffer, rem) = rem.split_at_mut(36);
        let (d_buffer, rem) = rem.split_at_mut(36);
        let (e_buffer, rem) = rem.split_at_mut(36);
        let (f_buffer, _) = rem.split_at_mut(36);

        let mut twiddle_idx = 0usize;

        for i in 0..18 {
            let j = i * 2;
            let ai = NeonStoreD::load(unsafe { data.get_unchecked(j..) });
            let mut bi = NeonStoreD::load(unsafe { data.get_unchecked(s_n - j - 2..) });
            let ci = NeonStoreD::load(unsafe { data.get_unchecked(s_n + j..) });
            let mut di = NeonStoreD::load(unsafe { data.get_unchecked(s_2n - j - 2..) });
            let ei = NeonStoreD::load(unsafe { data.get_unchecked(s_2n + j..) });
            let mut fi = NeonStoreD::load(unsafe { data.get_unchecked(216 - j - 2..) });

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

            let e_comp = fmla(
                f64::TWO * cos_sin_2ai_re,
                fmla(c_comp, cos_sin_2ai_re, -c_zet),
                -c_comp,
            );

            unsafe {
                a_comp.write(a_buffer.get_unchecked_mut(j..));
                let q0 = fmla(ai2dbedcd, cos_sin_ai_re, b_zet);
                q0.write(b_buffer.get_unchecked_mut(j..));
                let q1 = fmla(c_comp, cos_sin_2ai_re, c_zet);
                q1.write(c_buffer.get_unchecked_mut(j..));
                let q2 = d_comp * cos_sin_3ai_re;
                q2.write(d_buffer.get_unchecked_mut(j..));
                e_comp.write(e_buffer.get_unchecked_mut(j..));
                let q3 = fmla(ai2dbedcd, cos_sin_5ai_re, f_zet);
                q3.write(f_buffer.get_unchecked_mut(j..));
            }
            twiddle_idx += 7;
        }

        _ = self.bf36.execute(scratch);

        let (a_buffer, rem) = scratch.split_at_mut(36);
        let (b_buffer, rem) = rem.split_at_mut(36);
        let (c_buffer, rem) = rem.split_at_mut(36);
        let (d_buffer, rem) = rem.split_at_mut(36);
        let (e_buffer, rem) = rem.split_at_mut(36);
        let (f_buffer, _) = rem.split_at_mut(36);

        data[0] = a_buffer[0];
        data[1] = b_buffer[0] * f64::HALF;
        data[2] = c_buffer[0] * f64::HALF;
        data[3] = d_buffer[0] * f64::HALF;
        data[4] = e_buffer[0] * f64::HALF;
        data[5] = f_buffer[0] * f64::HALF;

        let mut b_diff = data[5];
        let mut c_diff = data[4];
        let mut e_diff = data[3];
        let mut d_diff = data[2];
        let mut f_diff = data[1];

        for k in 1..36 {
            data[6 * k] = a_buffer[k];
            let deferred_f_diff = b_buffer[k] - b_diff;
            data[6 * k + 1] = deferred_f_diff;
            let deferred_d_diff = c_buffer[k] - c_diff;
            data[6 * k + 2] = deferred_d_diff;
            e_diff = d_buffer[k] - e_diff;
            data[6 * k + 3] = e_diff;
            let new_d = e_buffer[k] - d_diff;
            data[6 * k + 4] = new_d;
            c_diff = new_d;
            d_diff = deferred_d_diff;
            let new_f = f_buffer[k] - f_diff;
            b_diff = new_f;
            f_diff = deferred_f_diff;
            data[6 * k + 5] = new_f;
        }
    }
}

impl PxdctExecutor<f64> for NeonDct2Butterfly216d {
    fn execute(&self, data: &mut [f64]) -> Result<(), PxdctError> {
        if !data.len().is_multiple_of(216) {
            return Err(PxdctError::InvalidSizeMultiplier(data.len(), self.length()));
        }

        let mut scratch = [f64::zero(); 216];

        for chunk in data.chunks_exact_mut(216) {
            self.exec((&mut chunk[..216]).try_into().unwrap(), &mut scratch);
        }
        Ok(())
    }

    fn length(&self) -> usize {
        216
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PxdctExecutor;
    use crate::butterflies::gen_test_butterfly_f;
    use crate::tests::naive_dct2;

    gen_test_butterfly_f!(test_bf36d, NeonDct2Butterfly36d, 36, 1e-3, naive_dct2);
    gen_test_butterfly_f!(test_bf216d, NeonDct2Butterfly216d, 216, 1e-3, naive_dct2);
}
