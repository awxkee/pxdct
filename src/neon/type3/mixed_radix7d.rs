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
use crate::type3::Dct3MixedRadix7Sample;
use crate::util::{try_vec, validate_scratch};
use crate::{PxdctError, PxdctExecutor};
use std::sync::Arc;

pub(crate) struct NeonDct3MixedRadix7d {
    inner_dct3: Arc<dyn PxdctExecutor<f64> + Send + Sync>,
    inner_dct_scratch_size: usize,
    rotation_twiddles: Vec<NeonStoreD>,
    execution_length: usize,
    p: usize,
}

impl NeonDct3MixedRadix7d {
    pub(crate) fn new(
        len: usize,
        dct3: Arc<dyn PxdctExecutor<f64> + Send + Sync>,
    ) -> Result<Self, PxdctError> {
        assert_eq!(
            dct3.length(),
            len / 7,
            "DCT-III Mixed-Radix-7 length DCTs must be one seventh of DCT-III"
        );

        let inner_dct3_scratch_size = dct3.scratch_size();
        let p = len / 7;

        Ok(Self {
            inner_dct3: dct3,
            inner_dct_scratch_size: inner_dct3_scratch_size,
            execution_length: len,
            rotation_twiddles: dct3_radix_n_rotation_twiddles_neond(7, p, len, 1),
            p,
        })
    }
}

impl NeonDct3MixedRadix7d {
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
        let xm_1 = NeonStoreD::load_n::<N>(data.slice_from(2 * p - N - k + 1..)).reverse_n::<N>();
        let xp_2 = NeonStoreD::load_n::<N>(data.slice_from(4 * p + k..));
        let xm_2 = NeonStoreD::load_n::<N>(data.slice_from(4 * p - N - k + 1..)).reverse_n::<N>();
        let xp_3 = NeonStoreD::load_n::<N>(data.slice_from(6 * p + k..));
        let xm_3 = NeonStoreD::load_n::<N>(data.slice_from(6 * p - N - k + 1..)).reverse_n::<N>();

        let s_1 = xp_1 + xm_1;
        let t_1 = xp_1 - xm_1;
        let s_2 = xp_2 + xm_2;
        let t_2 = xp_2 - xm_2;
        let s_3 = xp_3 + xm_3;
        let t_3 = xp_3 - xm_3;

        // U(k) = X(k) - S1 + S2 - S3
        let a_v = xk - s_1 + s_2 - s_3;
        unsafe {
            a_v.write_n::<N>(a_buffer.get_unchecked_mut(k..));
        }

        let tw0_re = unsafe { *self.rotation_twiddles.get_unchecked(uk) };
        let tw0_im = unsafe { *self.rotation_twiddles.get_unchecked(uk + 1) };
        let tw1_re = unsafe { *self.rotation_twiddles.get_unchecked(uk + 2) };
        let tw1_im = unsafe { *self.rotation_twiddles.get_unchecked(uk + 3) };
        let tw2_re = unsafe { *self.rotation_twiddles.get_unchecked(uk + 4) };
        let tw2_im = unsafe { *self.rotation_twiddles.get_unchecked(uk + 5) };

        let mut c_acc0 = xk;
        c_acc0 = fmla(s_1, NeonStoreD::dup(f64::D3_R7_T1), c_acc0);
        c_acc0 = fmla(s_2, NeonStoreD::dup(f64::D3_R7_T3), c_acc0);
        c_acc0 = fmla(s_3, NeonStoreD::dup(f64::D3_R7_T5), c_acc0);
        // s_acc0 = -t1*T4 - t2*T2 - t3*T0
        let mut s_acc0 = -t_1 * NeonStoreD::dup(f64::D3_R7_T4);
        s_acc0 = fmla(t_2, NeonStoreD::dup(-f64::D3_R7_T2), s_acc0);
        s_acc0 = fmla(t_3, NeonStoreD::dup(-f64::D3_R7_T0), s_acc0);
        let v_val0 = fmla(c_acc0, tw0_re, -s_acc0 * tw0_im);
        let w_val0 = fmla(c_acc0, tw0_im, s_acc0 * tw0_re);
        unsafe {
            v_val0.write_n::<N>(v_buffer.get_unchecked_mut(k..));
            w_val0
                .reverse_n::<N>()
                .write_n::<N>(w_buffer.get_unchecked_mut(p - N - k + 1..));
        }

        // ---- m = 1 ----
        // c_acc1 = xk + s1*T5 - s2*T1 - s3*T3
        let mut c_acc1 = xk;
        c_acc1 = fmla(s_1, NeonStoreD::dup(f64::D3_R7_T5), c_acc1);
        c_acc1 = fmla(s_2, NeonStoreD::dup(-f64::D3_R7_T1), c_acc1);
        c_acc1 = fmla(s_3, NeonStoreD::dup(-f64::D3_R7_T3), c_acc1);
        // s_acc1 = -t1*T0 - t2*T4 + t3*T2
        let mut s_acc1 = -t_1 * NeonStoreD::dup(f64::D3_R7_T0);
        s_acc1 = fmla(t_2, NeonStoreD::dup(-f64::D3_R7_T4), s_acc1);
        s_acc1 = fmla(t_3, NeonStoreD::dup(f64::D3_R7_T2), s_acc1);
        let v_val1 = fmla(c_acc1, tw1_re, -s_acc1 * tw1_im);
        let w_val1 = fmla(c_acc1, tw1_im, s_acc1 * tw1_re);
        v_val1.write_n::<N>(&mut v_buffer[p + k..]);
        w_val1
            .reverse_n::<N>()
            .write_n::<N>(&mut w_buffer[p + (p - N - k + 1)..]);

        // ---- m = 2 ----
        // c_acc2 = xk - s1*T3 - s2*T5 + s3*T1
        let mut c_acc2 = xk;
        c_acc2 = fmla(s_1, NeonStoreD::dup(-f64::D3_R7_T3), c_acc2);
        c_acc2 = fmla(s_2, NeonStoreD::dup(-f64::D3_R7_T5), c_acc2);
        c_acc2 = fmla(s_3, NeonStoreD::dup(f64::D3_R7_T1), c_acc2);
        // s_acc2 = -t1*T2 + t2*T0 - t3*T4
        let mut s_acc2 = -t_1 * NeonStoreD::dup(f64::D3_R7_T2);
        s_acc2 = fmla(t_2, NeonStoreD::dup(f64::D3_R7_T0), s_acc2);
        s_acc2 = fmla(t_3, NeonStoreD::dup(-f64::D3_R7_T4), s_acc2);
        let v_val2 = fmla(c_acc2, tw2_re, -s_acc2 * tw2_im);
        let w_val2 = fmla(c_acc2, tw2_im, s_acc2 * tw2_re);
        unsafe {
            v_val2.write_n::<N>(v_buffer.get_unchecked_mut(2 * p + k..));
            w_val2
                .reverse_n::<N>()
                .write_n::<N>(w_buffer.get_unchecked_mut(2 * p + (p - N - k + 1)..));
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
        dc_v0: NeonStoreD, // v0_m0_half - x0_half
        dc_v1: NeonStoreD, // v0_m1_half - x0_half
        dc_v2: NeonStoreD, // v0_m2_half - x0_half
        w0_m0_v: NeonStoreD,
        w0_m1_v: NeonStoreD,
        w0_m2_v: NeonStoreD,
        sign_v: NeonStoreD,
        n: usize,
    ) {
        let p = self.p;

        // Center stream (7n+3)
        let a_v = NeonStoreD::load_n::<N>(unsafe { a_buffer.get_unchecked(n..) });
        let center = a_v + dc_adjust_a_v;

        // m = 0  →  7n+0, 7n+6
        let f_v0 = NeonStoreD::load_n::<N>(unsafe { v_buffer.get_unchecked(n..) });
        let g_raw0 = NeonStoreD::load_n::<N>(unsafe { w_buffer.get_unchecked(n..) });
        let g_v0 = fmla(g_raw0, sign_v, w0_m0_v * sign_v);
        let f_dc0 = f_v0 + dc_v0;
        let out0 = f_dc0 + g_v0;
        let out6 = f_dc0 - g_v0;

        // m = 1  →  7n+1, 7n+5
        let f_v1 = NeonStoreD::load_n::<N>(unsafe { v_buffer.get_unchecked(p + n..) });
        let g_raw1 = NeonStoreD::load_n::<N>(unsafe { w_buffer.get_unchecked(p + n..) });
        let g_v1 = fmla(g_raw1, sign_v, w0_m1_v * sign_v);
        let f_dc1 = f_v1 + dc_v1;
        let out1 = f_dc1 + g_v1;
        let out5 = f_dc1 - g_v1;

        // m = 2  →  7n+2, 7n+4
        let f_v2 = NeonStoreD::load_n::<N>(unsafe { v_buffer.get_unchecked(2 * p + n..) });
        let g_raw2 = NeonStoreD::load_n::<N>(unsafe { w_buffer.get_unchecked(2 * p + n..) });
        let g_v2 = fmla(g_raw2, sign_v, w0_m2_v * sign_v);
        let f_dc2 = f_v2 + dc_v2;
        let out2 = f_dc2 + g_v2;
        let out4 = f_dc2 - g_v2;

        // Stride-7 scatter.
        let center_a = center.to_array();
        let out0_a = out0.to_array();
        let out1_a = out1.to_array();
        let out2_a = out2.to_array();
        let out4_a = out4.to_array();
        let out5_a = out5.to_array();
        let out6_a = out6.to_array();
        for i in 0..N {
            let base = 7 * (n + i);
            data[base] = out0_a[i];
            data[base + 1] = out1_a[i];
            data[base + 2] = out2_a[i];
            data[base + 3] = center_a[i];
            data[base + 4] = out4_a[i];
            data[base + 5] = out5_a[i];
            data[base + 6] = out6_a[i];
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
        // Buffer layout: [A: p][V_0,V_1,V_2: 3p][W_0,W_1,W_2: 3p] = 7p
        let (a_buffer, vw_buffer) = scratch.split_at_mut(p);
        let (v_buffer, w_buffer) = vw_buffer.split_at_mut(3 * p);

        a_buffer[0] = data[0] - data[2 * p] + data[4 * p] - data[6 * p];

        // V(0, m) — mirrors the scalar version exactly.
        let mut v0_m0 = data[0];
        v0_m0 = fmla(data[2 * p], f64::D3_R7_T1, v0_m0);
        v0_m0 = fmla(data[4 * p], f64::D3_R7_T3, v0_m0);
        v0_m0 = fmla(data[6 * p], f64::D3_R7_T5, v0_m0);
        v_buffer[0] = v0_m0;

        let mut v0_m1 = data[0];
        v0_m1 = fmla(data[2 * p], f64::D3_R7_T5, v0_m1);
        v0_m1 = fmla(data[4 * p], -f64::D3_R7_T1, v0_m1);
        v0_m1 = fmla(data[6 * p], -f64::D3_R7_T3, v0_m1);
        v_buffer[p] = v0_m1;

        let mut v0_m2 = data[0];
        v0_m2 = fmla(data[2 * p], -f64::D3_R7_T3, v0_m2);
        v0_m2 = fmla(data[4 * p], -f64::D3_R7_T5, v0_m2);
        v0_m2 = fmla(data[6 * p], f64::D3_R7_T1, v0_m2);
        v_buffer[2 * p] = v0_m2;

        // W'(0, m) = W(p, m).
        let mut w0_m0 = data[p] * f64::D3_R7_T0;
        w0_m0 = fmla(data[3 * p], f64::D3_R7_T2, w0_m0);
        w0_m0 = fmla(data[5 * p], f64::D3_R7_T4, w0_m0);
        w_buffer[0] = w0_m0;

        let mut w0_m1 = data[p] * f64::D3_R7_T2;
        w0_m1 = fmla(data[3 * p], -f64::D3_R7_T4, w0_m1);
        w0_m1 = fmla(data[5 * p], -f64::D3_R7_T0, w0_m1);
        w_buffer[p] = w0_m1;

        let mut w0_m2 = data[p] * f64::D3_R7_T4;
        w0_m2 = fmla(data[3 * p], -f64::D3_R7_T0, w0_m2);
        w0_m2 = fmla(data[5 * p], f64::D3_R7_T2, w0_m2);
        w_buffer[2 * p] = w0_m2;

        let mut uk = 0usize;
        let mut k = 1usize;
        while k + 2 <= p {
            self.exec_stage1::<S, 2>(data, a_buffer, v_buffer, w_buffer, uk, k);
            uk += 6; // 3 inner groups × 2 (re + im)
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
        let v0_m2_half = v_buffer[2 * p] * 0.5;
        let w0_m0_half = w_buffer[0] * 0.5;
        let w0_m1_half = w_buffer[p] * 0.5;
        let w0_m2_half = w_buffer[2 * p] * 0.5;

        self.inner_dct3
            .execute_with_scratch(scratch, inner_scratch)?;

        let (a_buffer, vw_buffer) = scratch.split_at_mut(p);
        let (v_buffer, w_buffer) = vw_buffer.split_at_mut(3 * p);

        let dc_adjust_a_v = NeonStoreD::dup(u0_half - x0_half);
        let dc_v0 = NeonStoreD::dup(v0_m0_half - x0_half);
        let dc_v1 = NeonStoreD::dup(v0_m1_half - x0_half);
        let dc_v2 = NeonStoreD::dup(v0_m2_half - x0_half);
        let w0_m0_v = NeonStoreD::dup(w0_m0_half);
        let w0_m1_v = NeonStoreD::dup(w0_m1_half);
        let w0_m2_v = NeonStoreD::dup(w0_m2_half);
        let sign_even = NeonStoreD::load(&[1.0_f64, -1.0, 1.0, -1.0]);

        let mut n = 0usize;
        while n + 2 <= p {
            self.exec_stage3::<S, 2>(
                data,
                a_buffer,
                v_buffer,
                w_buffer,
                dc_adjust_a_v,
                dc_v0,
                dc_v1,
                dc_v2,
                w0_m0_v,
                w0_m1_v,
                w0_m2_v,
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
                dc_v0,
                dc_v1,
                dc_v2,
                w0_m0_v,
                w0_m1_v,
                w0_m2_v,
                sign_v,
                n,
            );
        }

        Ok(())
    }
}

boring_neon_mixed_radix!(NeonDct3MixedRadix7d, f64);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::naive_dct3;
    use crate::type3::Dct3Butterfly7;
    use rand::RngExt;

    #[test]
    fn test_neon_dct3_radix7() {
        const N: usize = 7 * 7;
        let mut input = vec![0.0_f64; N];
        for z in input.iter_mut() {
            *z = rand::rng().random_range(1.0..2.0);
        }
        let reference = naive_dct3(&input);

        let bf = NeonDct3MixedRadix7d::new(N, Arc::new(Dct3Butterfly7::default())).unwrap();
        bf.execute(&mut input).unwrap();
        for (i, (&a, &b)) in input.iter().zip(reference.iter()).enumerate() {
            assert!((a - b).abs() < 1e-1, "mismatch at {i}: {a} vs {b}");
        }
    }
}
