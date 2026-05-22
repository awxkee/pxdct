/*
 * // Copyright (c) Radzivon Bartoshyk 5/2026. All rights reserved.
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
use crate::mla::fmla;
use crate::neon::store_d::NeonStoreD;
use crate::neon::util::boring_neon_mixed_radix;
use crate::type3::radixq_dct3_n_rotation_twiddle;
use crate::util::{DctConstants, try_vec, validate_scratch};
use crate::{PxdctError, PxdctExecutor};
use num_traits::AsPrimitive;
use std::sync::Arc;

pub(crate) fn dct3_radix_n_rotation_twiddles_neond(
    q: usize,
    q_modules: usize,
    len: usize,
    k_start: usize,
) -> Vec<NeonStoreD>
where
    f64: AsPrimitive<f32>,
{
    let inner_groups = q.saturating_sub(3) / 2 + 1;

    debug_assert!(k_start <= q_modules);
    let count = q_modules - k_start;

    let main_groups = count / 2;
    let has_remainder = !count.is_multiple_of(2) as usize;
    let mut twiddles = Vec::with_capacity((main_groups + has_remainder) * 2 * inner_groups);

    let mut uk = 0usize;
    while uk + 2 <= count {
        let k = k_start + uk;

        let mut array_re = [0.0; 2];
        let mut array_im = [0.0; 2];
        for m in 0..inner_groups {
            for i in 0..2 {
                let layer = radixq_dct3_n_rotation_twiddle::<f64>(q, m, (k + i).as_(), len);
                array_re[i] = layer.re;
                array_im[i] = layer.im;
            }

            twiddles.push(NeonStoreD::load(array_re.as_ref()));
            twiddles.push(NeonStoreD::load(array_im.as_ref()));
        }

        uk += 2;
    }

    let remainder = count - (count / 2) * 2;
    if remainder > 0 {
        let k = k_start + uk;

        let mut array_re = [0.0; 2];
        let mut array_im = [0.0; 2];
        for m in 0..inner_groups {
            for i in 0..remainder {
                let layer = radixq_dct3_n_rotation_twiddle::<f64>(q, m, (k + i).as_(), len);
                array_re[i] = layer.re;
                array_im[i] = layer.im;
            }

            twiddles.push(NeonStoreD::load(array_re.as_ref()));
            twiddles.push(NeonStoreD::load(array_im.as_ref()));
        }
    }

    twiddles
}

pub(crate) struct NeonDct3MixedRadix3d {
    inner_dct3: Arc<dyn PxdctExecutor<f64> + Send + Sync>,
    inner_dct_scratch_size: usize,
    rotation_twiddles: Vec<NeonStoreD>,
    execution_length: usize,
    p: usize, // = N / 3
    s: usize, // = 2N / 3
}

impl NeonDct3MixedRadix3d {
    pub(crate) fn new(
        len: usize,
        dct3: Arc<dyn PxdctExecutor<f64> + Send + Sync>,
    ) -> Result<Self, PxdctError> {
        assert_eq!(
            dct3.length(),
            len / 3,
            "DCT-III Mixed-Radix-3 length DCTs must be one third of DCT-III"
        );

        let inner_dct3_scratch_size = dct3.scratch_size();
        let p = len / 3;

        Ok(Self {
            inner_dct3: dct3,
            inner_dct_scratch_size: inner_dct3_scratch_size,
            execution_length: len,
            rotation_twiddles: dct3_radix_n_rotation_twiddles_neond(3, p, len, 1),
            p,
            s: 2 * len / 3,
        })
    }
}

impl NeonDct3MixedRadix3d {
    #[inline(always)]
    fn exec_stage1<S: BidirectionalStore<f64>, const N: usize>(
        &self,
        data: &S,
        a_buffer: &mut [f64],
        c_buffer: &mut [f64],
        w_buffer: &mut [f64],
        uk: usize,
        k: usize,
    ) {
        let p = self.p;
        let s = self.s;

        let xk = NeonStoreD::load_n::<N>(data.slice_from(k..));
        let xp = NeonStoreD::load_n::<N>(data.slice_from(s + k..));
        let xm = NeonStoreD::load_n::<N>(data.slice_from(s - N - k + 1..)).reverse_n::<N>();

        let s1 = xp + xm;
        let t1 = xp - xm;

        let a_v = xk - s1;
        unsafe {
            a_v.write_n::<N>(a_buffer.get_unchecked_mut(k..));
        }

        let c_v = fmla(s1, NeonStoreD::dup(0.5_f64), xk);
        let s_v = t1 * NeonStoreD::dup(f64::SQRT_3_OVER_2);

        let twiddle_re = unsafe { *self.rotation_twiddles.get_unchecked(uk) };
        let twiddle_im = unsafe { *self.rotation_twiddles.get_unchecked(uk + 1) };

        // v_k  = c_v * tw.re + s_v * tw.im  (positive s_v, unlike radix-5 m=0)
        let v_v = fmla(c_v, twiddle_re, s_v * twiddle_im);
        // w_k  = c_v * tw.im - s_v * tw.re
        let w_v = fmla(c_v, twiddle_im, -s_v * twiddle_re);

        unsafe {
            v_v.write_n::<N>(c_buffer.get_unchecked_mut(k..));
            w_v.reverse_n::<N>()
                .write_n::<N>(w_buffer.get_unchecked_mut(p - N - k + 1..));
        }
    }

    #[inline(always)]
    fn exec_stage3<S: BidirectionalStore<f64>, const N: usize>(
        &self,
        data: &mut S,
        a_buffer: &[f64],
        c_buffer: &[f64],
        w_buffer: &[f64],
        dc_adjust_a_v: NeonStoreD,
        dc_adjust_fg_v: NeonStoreD,
        w0_half_v: NeonStoreD,
        sign_v: NeonStoreD,
        n: usize,
    ) {
        // Center stream (3n+1): a + dc_adjust_a
        let a_v = NeonStoreD::load_n::<N>(unsafe { a_buffer.get_unchecked(n..) });
        let center = a_v + dc_adjust_a_v;

        // Outer streams (3n+0, 3n+2)
        let f_v = NeonStoreD::load_n::<N>(unsafe { c_buffer.get_unchecked(n..) });
        let g_raw = NeonStoreD::load_n::<N>(unsafe { w_buffer.get_unchecked(n..) });
        // g_v = g_raw * sign + w0_half * sign = (g_raw + w0_half) * sign
        let g_v = fmla(g_raw, sign_v, w0_half_v * sign_v);
        let f_dc = f_v + dc_adjust_fg_v;
        let out0 = f_dc + g_v;
        let out2 = f_dc - g_v;

        let center_a = center.to_array();
        let out0_a = out0.to_array();
        let out2_a = out2.to_array();
        for i in 0..N {
            let base = 3 * (n + i);
            data[base] = out0_a[i];
            data[base + 1] = center_a[i];
            data[base + 2] = out2_a[i];
        }
    }

    #[inline(always)]
    fn execute_with_store<S: BidirectionalStore<f64>>(
        &self,
        data: &mut S,
        scratch: &mut [f64],
    ) -> Result<(), PxdctError> {
        let p = self.p;
        let s = self.s;

        let (scratch, inner_scratch) = scratch.split_at_mut(self.execution_length);
        // Buffer layout: [A: p][C (=V_0): p][W' (=W_0): p]  — total = 3p = N.
        let (a_buffer, cw_buffer) = scratch.split_at_mut(p);
        let (c_buffer, w_buffer) = cw_buffer.split_at_mut(p);

        let x0 = data[0];
        let x_s = data[s]; // X(2N/3)
        let x_p = data[p]; // X(N/3)

        a_buffer[0] = x0 - x_s;
        c_buffer[0] = fmla(0.5_f64, x_s, x0); // x0 + 0.5 * x_s
        w_buffer[0] = x_p * f64::SQRT_3_OVER_2;

        let mut uk = 0usize;
        let mut k = 1usize;
        while k + 2 <= p {
            self.exec_stage1::<S, 2>(data, a_buffer, c_buffer, w_buffer, uk, k);
            uk += 2; // 1 inner group * 2 (re + im)
            k += 2;
        }

        let rem = p - k;
        if rem == 1 {
            self.exec_stage1::<S, 1>(data, a_buffer, c_buffer, w_buffer, uk, k);
        }

        let x0_half = x0 * 0.5;
        let u0_half = a_buffer[0] * 0.5;
        let v0_half = c_buffer[0] * 0.5;
        let w0_half = w_buffer[0] * 0.5;

        self.inner_dct3
            .execute_with_scratch(scratch, inner_scratch)?;

        let (a_buffer, cw_buffer) = scratch.split_at_mut(p);
        let (c_buffer, w_buffer) = cw_buffer.split_at_mut(p);

        let dc_adjust_a_v = NeonStoreD::dup(u0_half - x0_half);
        let dc_adjust_fg_v = NeonStoreD::dup(v0_half - x0_half);
        let w0_half_v = NeonStoreD::dup(w0_half);
        let sign_even = NeonStoreD::load(&[1.0_f64, -1.0, 1.0, -1.0]);

        let mut n = 0usize;
        while n + 2 <= p {
            self.exec_stage3::<S, 2>(
                data,
                a_buffer,
                c_buffer,
                w_buffer,
                dc_adjust_a_v,
                dc_adjust_fg_v,
                w0_half_v,
                sign_even,
                n,
            );
            n += 2;
        }

        let rem = p - n;
        if rem == 1 {
            let sign_v = NeonStoreD::load(&[1.0_f64, 0.0, 0.0, 0.0]);
            self.exec_stage3::<S, 1>(
                data,
                a_buffer,
                c_buffer,
                w_buffer,
                dc_adjust_a_v,
                dc_adjust_fg_v,
                w0_half_v,
                sign_v,
                n,
            );
        }

        Ok(())
    }
}

boring_neon_mixed_radix!(NeonDct3MixedRadix3d, f64);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Pxdct;
    use crate::tests::naive_dct3;
    use crate::type3::Dct3Butterfly3;
    use rand::RngExt;

    #[test]
    fn test_neon_dct3_radix3() {
        const N: usize = 3 * 3;
        let mut input = vec![0.0_f64; N];
        for z in input.iter_mut() {
            *z = rand::rng().random_range(1.0..2.0);
        }
        let reference = naive_dct3(&input);

        let bf = NeonDct3MixedRadix3d::new(N, Arc::new(Dct3Butterfly3::default())).unwrap();
        bf.execute(&mut input).unwrap();
        for (i, (&a, &b)) in input.iter().zip(reference.iter()).enumerate() {
            assert!((a - b).abs() < 1e-1, "mismatch at {i}: {a} vs {b}");
        }
    }

    #[test]
    fn test_neon_dct3_radix3_large() {
        const N: usize = 3 * 9;
        let mut input = vec![0.0_f64; N];
        for z in input.iter_mut() {
            *z = rand::rng().random_range(1.0..2.0);
        }
        let reference = naive_dct3(&input);

        let inner: Arc<dyn PxdctExecutor<f64> + Send + Sync> = Pxdct::make_dct3_f64(9).unwrap();
        let bf = NeonDct3MixedRadix3d::new(N, inner).unwrap();
        bf.execute(&mut input).unwrap();
        for (i, (&a, &b)) in input.iter().zip(reference.iter()).enumerate() {
            assert!((a - b).abs() < 1e-1, "mismatch at {i}: {a} vs {b}");
        }
    }
}
