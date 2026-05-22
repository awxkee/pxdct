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
use crate::neon::type3::mixed_radix5f::dct3_radix_n_rotation_twiddles_neon;
use crate::neon::util::{NeonStoreF, boring_neon_mixed_radix};
use crate::util::{DctConstants, try_vec, validate_scratch};
use crate::{PxdctError, PxdctExecutor};
use std::sync::Arc;

pub(crate) struct NeonDct3MixedRadix3f {
    inner_dct3: Arc<dyn PxdctExecutor<f32> + Send + Sync>,
    inner_dct_scratch_size: usize,
    rotation_twiddles: Vec<NeonStoreF>,
    execution_length: usize,
    p: usize, // = N / 3
    s: usize, // = 2N / 3
}

impl NeonDct3MixedRadix3f {
    pub(crate) fn new(
        len: usize,
        dct3: Arc<dyn PxdctExecutor<f32> + Send + Sync>,
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
            rotation_twiddles: dct3_radix_n_rotation_twiddles_neon(3, p, len, 1),
            p,
            s: 2 * len / 3,
        })
    }
}

impl NeonDct3MixedRadix3f {
    #[inline(always)]
    fn exec_stage1<S: BidirectionalStore<f32>, const N: usize>(
        &self,
        data: &S,
        a_buffer: &mut [f32],
        c_buffer: &mut [f32],
        w_buffer: &mut [f32],
        uk: usize,
        k: usize,
    ) {
        let p = self.p;
        let s = self.s;

        let xk = NeonStoreF::load_n::<N>(data.slice_from(k..));
        let xp = NeonStoreF::load_n::<N>(data.slice_from(s + k..));
        let xm = NeonStoreF::load_n::<N>(data.slice_from(s - N - k + 1..)).reverse_n::<N>();

        let s1 = xp + xm;
        let t1 = xp - xm;

        // U(k) = X(k) - S1(k)
        let a_v = xk - s1;
        unsafe {
            a_v.write_n::<N>(a_buffer.get_unchecked_mut(k..));
        }

        let c_v = fmla(s1, NeonStoreF::dup(0.5_f32), xk);
        let s_v = t1 * NeonStoreF::dup(f32::SQRT_3_OVER_2);

        let twiddle_re = unsafe { *self.rotation_twiddles.get_unchecked(uk) };
        let twiddle_im = unsafe { *self.rotation_twiddles.get_unchecked(uk + 1) };

        let v_v = fmla(c_v, twiddle_re, s_v * twiddle_im);
        let w_v = fmla(c_v, twiddle_im, -s_v * twiddle_re);

        unsafe {
            v_v.write_n::<N>(c_buffer.get_unchecked_mut(k..));
            w_v.reverse_n::<N>()
                .write_n::<N>(w_buffer.get_unchecked_mut(p - N - k + 1..));
        }
    }

    #[inline(always)]
    fn exec_stage3<S: BidirectionalStore<f32>, const N: usize>(
        &self,
        data: &mut S,
        a_buffer: &[f32],
        c_buffer: &[f32],
        w_buffer: &[f32],
        dc_adjust_a_v: NeonStoreF,
        dc_adjust_fg_v: NeonStoreF,
        w0_half_v: NeonStoreF,
        sign_v: NeonStoreF,
        n: usize,
    ) {
        // Center stream (3n+1): a + dc_adjust_a
        let a_v = NeonStoreF::load_n::<N>(unsafe { a_buffer.get_unchecked(n..) });
        let center = a_v + dc_adjust_a_v;

        // Outer streams (3n+0, 3n+2)
        let f_v = NeonStoreF::load_n::<N>(unsafe { c_buffer.get_unchecked(n..) });
        let g_raw = NeonStoreF::load_n::<N>(unsafe { w_buffer.get_unchecked(n..) });
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
    fn execute_with_store<S: BidirectionalStore<f32>>(
        &self,
        data: &mut S,
        scratch: &mut [f32],
    ) -> Result<(), PxdctError> {
        let p = self.p;
        let s = self.s;

        let (scratch, inner_scratch) = scratch.split_at_mut(self.execution_length);
        let (a_buffer, cw_buffer) = scratch.split_at_mut(p);
        let (c_buffer, w_buffer) = cw_buffer.split_at_mut(p);

        let x0 = data[0];
        let x_s = data[s]; // X(2N/3)
        let x_p = data[p]; // X(N/3)

        a_buffer[0] = x0 - x_s;
        c_buffer[0] = fmla(0.5_f32, x_s, x0);
        w_buffer[0] = x_p * f32::SQRT_3_OVER_2;

        let mut uk = 0usize;
        let mut k = 1usize;
        while k + 4 <= p {
            self.exec_stage1::<S, 4>(data, a_buffer, c_buffer, w_buffer, uk, k);
            uk += 2;
            k += 4;
        }

        let rem = p - k;
        if rem == 3 {
            self.exec_stage1::<S, 3>(data, a_buffer, c_buffer, w_buffer, uk, k);
        } else if rem == 2 {
            self.exec_stage1::<S, 2>(data, a_buffer, c_buffer, w_buffer, uk, k);
        } else if rem == 1 {
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

        let dc_adjust_a_v = NeonStoreF::dup(u0_half - x0_half);
        let dc_adjust_fg_v = NeonStoreF::dup(v0_half - x0_half);
        let w0_half_v = NeonStoreF::dup(w0_half);
        let sign_even = NeonStoreF::load(&[1.0_f32, -1.0, 1.0, -1.0]);

        let mut n = 0usize;
        while n + 4 <= p {
            self.exec_stage3::<S, 4>(
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
            n += 4;
        }

        let rem = p - n;
        if rem == 3 {
            let sign_v = NeonStoreF::load(&[1.0_f32, -1.0, 1.0, 0.0]);
            self.exec_stage3::<S, 3>(
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
        } else if rem == 2 {
            let sign_v = NeonStoreF::load(&[1.0_f32, -1.0, 0.0, 0.0]);
            self.exec_stage3::<S, 2>(
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
        } else if rem == 1 {
            let sign_v = NeonStoreF::load(&[1.0_f32, 0.0, 0.0, 0.0]);
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

boring_neon_mixed_radix!(NeonDct3MixedRadix3f, f32);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Pxdct;
    use crate::tests::naive_dct3_f32;
    use crate::type3::Dct3Butterfly3;
    use rand::RngExt;

    #[test]
    fn test_neon_dct3_radix3() {
        const N: usize = 3 * 3;
        let mut input = vec![0.0_f32; N];
        for z in input.iter_mut() {
            *z = rand::rng().random_range(1.0..2.0);
        }
        let reference = naive_dct3_f32(&input);

        let bf = NeonDct3MixedRadix3f::new(N, Arc::new(Dct3Butterfly3::default())).unwrap();
        bf.execute(&mut input).unwrap();
        for (i, (&a, &b)) in input.iter().zip(reference.iter()).enumerate() {
            assert!((a - b).abs() < 1e-1, "mismatch at {i}: {a} vs {b}");
        }
    }

    #[test]
    fn test_neon_dct3_radix3_large() {
        const N: usize = 3 * 9;
        let mut input = vec![0.0_f32; N];
        for z in input.iter_mut() {
            *z = rand::rng().random_range(1.0..2.0);
        }
        let reference = naive_dct3_f32(&input);

        let inner: Arc<dyn PxdctExecutor<f32> + Send + Sync> = Pxdct::make_dct3_f32(9).unwrap();
        let bf = NeonDct3MixedRadix3f::new(N, inner).unwrap();
        bf.execute(&mut input).unwrap();
        for (i, (&a, &b)) in input.iter().zip(reference.iter()).enumerate() {
            assert!((a - b).abs() < 1e-1, "mismatch at {i}: {a} vs {b}");
        }
    }
}
