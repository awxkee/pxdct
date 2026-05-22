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
use crate::neon::type3::mixed_radix3d::dct3_radix_n_rotation_twiddles_neond;
use crate::neon::util::boring_neon_mixed_radix;
use crate::type3::Dct3MixedRadix5Sample;
use crate::util::{try_vec, validate_scratch};
use crate::{PxdctError, PxdctExecutor};
use std::sync::Arc;

pub(crate) struct NeonDct3MixedRadix5d {
    inner_dct3: Arc<dyn PxdctExecutor<f64> + Send + Sync>,
    inner_dct_scratch_size: usize,
    rotation_twiddles: Vec<NeonStoreD>,
    execution_length: usize,
    p: usize,
}

impl NeonDct3MixedRadix5d {
    pub(crate) fn new(
        len: usize,
        dct3: Arc<dyn PxdctExecutor<f64> + Send + Sync>,
    ) -> Result<Self, PxdctError> {
        assert_eq!(
            dct3.length(),
            len / 5,
            "DCT-III Mixed-Radix-5 length DCTs must be one fifth of DCT-III"
        );

        let inner_dct3_scratch_size = dct3.scratch_size();

        Ok(Self {
            inner_dct3: dct3,
            inner_dct_scratch_size: inner_dct3_scratch_size,
            execution_length: len,
            rotation_twiddles: dct3_radix_n_rotation_twiddles_neond(5, len / 5, len, 1),
            p: len / 5,
        })
    }
}

impl NeonDct3MixedRadix5d {
    #[inline(always)]
    fn exec_stage1<S: BidirectionalStore<f64>, const N: usize>(
        &self,
        data: &S,
        a_buffer: &mut [f64],
        v_buffer: &mut [f64],
        w_buffer: &mut [f64],
        uk: usize,
        k: usize,
    ) {
        let p = self.p;

        let xk = NeonStoreD::load_n::<N>(data.slice_from(k..));
        let xp_1 = NeonStoreD::load_n::<N>(data.slice_from(2 * p + k..));
        let xp_2 = NeonStoreD::load_n::<N>(data.slice_from(4 * p + k..));

        let xm_1 = NeonStoreD::load_n::<N>(data.slice_from(2 * p - N - k + 1..)).reverse_n::<N>();
        let xm_2 = NeonStoreD::load_n::<N>(data.slice_from(4 * p - N - k + 1..)).reverse_n::<N>();

        let s_1 = xp_1 + xm_1;
        let t_1 = xp_1 - xm_1;
        let s_2 = xp_2 + xm_2;
        let t_2 = xp_2 - xm_2;

        let a_v = xk - s_1 + s_2;
        unsafe {
            a_v.write_n::<N>(a_buffer.get_unchecked_mut(k..));
        }

        let twiddle0_re = unsafe { *self.rotation_twiddles.get_unchecked(uk) };
        let twiddle0_im = unsafe { *self.rotation_twiddles.get_unchecked(uk + 1) };
        let twiddle1_re = unsafe { *self.rotation_twiddles.get_unchecked(uk + 2) };
        let twiddle1_im = unsafe { *self.rotation_twiddles.get_unchecked(uk + 3) };

        let mut c_acc0 = xk;
        c_acc0 = fmla(s_1, NeonStoreD::dup(f64::D3_R5_T1), c_acc0);
        c_acc0 = fmla(s_2, NeonStoreD::dup(f64::D3_R5_T3), c_acc0);
        let mut s_acc0 = -t_1 * NeonStoreD::dup(f64::D3_R5_T2);
        s_acc0 = fmla(t_2, NeonStoreD::dup(-f64::D3_R5_T0), s_acc0);

        let v_val0 = fmla(c_acc0, twiddle0_re, -s_acc0 * twiddle0_im);
        let w_val0 = fmla(c_acc0, twiddle0_im, s_acc0 * twiddle0_re);

        unsafe {
            v_val0.write_n::<N>(v_buffer.get_unchecked_mut(k..));
            w_val0
                .reverse_n::<N>()
                .write_n::<N>(w_buffer.get_unchecked_mut(p - N - k + 1..));
        }

        let mut c_acc1 = xk;
        c_acc1 = fmla(s_1, NeonStoreD::dup(-f64::D3_R5_T3), c_acc1);
        c_acc1 = fmla(s_2, NeonStoreD::dup(-f64::D3_R5_T1), c_acc1);
        let mut s_acc1 = -t_1 * NeonStoreD::dup(f64::D3_R5_T0);
        s_acc1 = fmla(t_2, NeonStoreD::dup(f64::D3_R5_T2), s_acc1);

        let v_val1 = fmla(c_acc1, twiddle1_re, -s_acc1 * twiddle1_im);
        let w_val1 = fmla(c_acc1, twiddle1_im, s_acc1 * twiddle1_re);

        unsafe {
            v_val1.write_n::<N>(v_buffer.get_unchecked_mut(p + k..));
            w_val1
                .reverse_n::<N>()
                .write_n::<N>(w_buffer.get_unchecked_mut(p + (p - N - k + 1)..));
        }
    }

    #[inline(always)]
    fn exec_stage3<S: BidirectionalStore<f64>, const N: usize>(
        &self,
        data: &mut S,
        a_buffer: &[f64],
        v_buffer: &[f64],
        w_buffer: &[f64],
        dc_adjust_a_v: NeonStoreD,
        dc_adjust_v0_m0_v: NeonStoreD,
        dc_adjust_v0_m1_v: NeonStoreD,
        w0_m0_half_v: NeonStoreD,
        w0_m1_half_v: NeonStoreD,
        sign_v: NeonStoreD,
        n: usize,
    ) {
        let p = self.p;

        let a_v = NeonStoreD::load_n::<N>(unsafe { a_buffer.get_unchecked(n..) });
        let center = a_v + dc_adjust_a_v;

        let f_v0 = NeonStoreD::load_n::<N>(unsafe { v_buffer.get_unchecked(n..) });
        let g_raw0 = NeonStoreD::load_n::<N>(unsafe { w_buffer.get_unchecked(n..) });
        let g_v0 = fmla(g_raw0, sign_v, w0_m0_half_v * sign_v);
        let f_dc0 = f_v0 + dc_adjust_v0_m0_v;
        let out0 = f_dc0 + g_v0;
        let out4 = f_dc0 - g_v0;

        let f_v1 = NeonStoreD::load_n::<N>(unsafe { v_buffer.get_unchecked(p + n..) });
        let g_raw1 = NeonStoreD::load_n::<N>(unsafe { w_buffer.get_unchecked(p + n..) });
        let g_v1 = fmla(g_raw1, sign_v, w0_m1_half_v * sign_v);
        let f_dc1 = f_v1 + dc_adjust_v0_m1_v;
        let out1 = f_dc1 + g_v1;
        let out3 = f_dc1 - g_v1;

        let center_a = center.to_array();
        let out0_a = out0.to_array();
        let out1_a = out1.to_array();
        let out3_a = out3.to_array();
        let out4_a = out4.to_array();
        for i in 0..N {
            let base = 5 * (n + i);
            data[base] = out0_a[i];
            data[base + 1] = out1_a[i];
            data[base + 2] = center_a[i];
            data[base + 3] = out3_a[i];
            data[base + 4] = out4_a[i];
        }
    }

    #[inline(always)]
    fn execute_with_store<S: BidirectionalStore<f64>>(
        &self,
        data: &mut S,
        scratch: &mut [f64],
    ) -> Result<(), PxdctError> {
        let p = self.p;

        let (scratch, inner_scratch) = scratch.split_at_mut(self.execution_length);
        let (a_buffer, vw_buffer) = scratch.split_at_mut(p);
        let (v_buffer, w_buffer) = vw_buffer.split_at_mut(2 * p);

        a_buffer[0] = data[0] - data[2 * p] + data[4 * p];

        let mut v0_m0 = data[0];
        v0_m0 = fmla(data[2 * p], f64::D3_R5_T1, v0_m0);
        v0_m0 = fmla(data[4 * p], f64::D3_R5_T3, v0_m0);
        v_buffer[0] = v0_m0;
        let mut v0_m1 = data[0];
        v0_m1 = fmla(data[2 * p], -f64::D3_R5_T3, v0_m1);
        v0_m1 = fmla(data[4 * p], -f64::D3_R5_T1, v0_m1);
        v_buffer[p] = v0_m1;

        let mut w0_m0 = data[p] * f64::D3_R5_T0;
        w0_m0 = fmla(data[3 * p], f64::D3_R5_T2, w0_m0);
        w_buffer[0] = w0_m0;
        let mut w0_m1 = data[p] * f64::D3_R5_T2;
        w0_m1 = fmla(data[3 * p], -f64::D3_R5_T0, w0_m1);
        w_buffer[p] = w0_m1;

        let mut uk = 0usize;
        let mut k = 1usize;
        while k + 2 <= p {
            self.exec_stage1::<S, 2>(data, a_buffer, v_buffer, w_buffer, uk, k);
            uk += 4;
            k += 2;
        }

        let rem = p - k;
        if rem == 1 {
            self.exec_stage1::<S, 1>(data, a_buffer, v_buffer, w_buffer, uk, k);
        }

        let x0_half = data[0] * 0.5;
        let u0_half = a_buffer[0] * 0.5;
        let v0_m0_half = v_buffer[0] * 0.5;
        let v0_m1_half = v_buffer[p] * 0.5;
        let w0_m0_half = w_buffer[0] * 0.5;
        let w0_m1_half = w_buffer[p] * 0.5;

        self.inner_dct3
            .execute_with_scratch(scratch, inner_scratch)?;

        let (a_buffer, vw_buffer) = scratch.split_at_mut(p);
        let (v_buffer, w_buffer) = vw_buffer.split_at_mut(2 * p);

        let dc_adjust_a_v = NeonStoreD::dup(u0_half - x0_half);
        let dc_adjust_v0_m0_v = NeonStoreD::dup(v0_m0_half - x0_half);
        let dc_adjust_v0_m1_v = NeonStoreD::dup(v0_m1_half - x0_half);
        let w0_m0_half_v = NeonStoreD::dup(w0_m0_half);
        let w0_m1_half_v = NeonStoreD::dup(w0_m1_half);

        let sign_even = NeonStoreD::load(&[1.0_f64, -1.0, 1.0, -1.0]);

        let mut n = 0usize;
        while n + 2 <= p {
            self.exec_stage3::<S, 2>(
                data,
                a_buffer,
                v_buffer,
                w_buffer,
                dc_adjust_a_v,
                dc_adjust_v0_m0_v,
                dc_adjust_v0_m1_v,
                w0_m0_half_v,
                w0_m1_half_v,
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
                v_buffer,
                w_buffer,
                dc_adjust_a_v,
                dc_adjust_v0_m0_v,
                dc_adjust_v0_m1_v,
                w0_m0_half_v,
                w0_m1_half_v,
                sign_v,
                n,
            );
        }

        Ok(())
    }
}

boring_neon_mixed_radix!(NeonDct3MixedRadix5d, f64);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::naive_dct3;
    use crate::type3::Dct3Butterfly5;

    use rand::RngExt;

    #[test]
    fn test_split_dct3_radix5_neon() {
        const N: usize = 5 * 5;
        let mut input = vec![0.0_f64; N];
        for z in input.iter_mut() {
            *z = rand::rng().random_range(1.0..2.0);
        }
        let reference = naive_dct3(&input);

        let bf = NeonDct3MixedRadix5d::new(N, Arc::new(Dct3Butterfly5::default())).unwrap();
        bf.execute(&mut input).unwrap();
        for (i, (&a, &b)) in input.iter().zip(reference.iter()).enumerate() {
            assert!((a - b).abs() < 1e-1, "mismatch at {i}: {a} vs {b}");
        }
    }
}
