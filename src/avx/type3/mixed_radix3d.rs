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
use crate::avx::stored::{AvxFullD, AvxLanesD, AvxStoreD, AvxTailD};
use crate::avx::util::{boring_avx_mixed_radix, fma};
use crate::bidirectional::BidirectionalStore;
use crate::type3::radixq_dct3_n_rotation_twiddle;
use crate::util::{DctConstants, try_vec, validate_scratch};
use crate::{PxdctError, PxdctExecutor};
use num_traits::AsPrimitive;
use std::sync::Arc;

#[target_feature(enable = "avx2")]
pub(crate) fn dct3_radix_n_rotation_twiddles_avx_d(
    q: usize,
    q_modules: usize,
    len: usize,
    k_start: usize,
) -> Vec<AvxStoreD> {
    let inner_groups = q.saturating_sub(3) / 2 + 1;

    debug_assert!(k_start <= q_modules);
    let count = q_modules - k_start;

    let main_groups = count / 4;
    let has_remainder = !count.is_multiple_of(4) as usize;
    let mut twiddles = Vec::with_capacity((main_groups + has_remainder) * 2 * inner_groups);

    let mut uk = 0usize;
    while uk + 4 <= count {
        let k = k_start + uk;

        let mut array_re = [0.0_f64; 4];
        let mut array_im = [0.0_f64; 4];
        for m in 0..inner_groups {
            for i in 0..4 {
                let layer = radixq_dct3_n_rotation_twiddle::<f64>(q, m, (k + i).as_(), len);
                array_re[i] = layer.re;
                array_im[i] = layer.im;
            }

            twiddles.push(AvxStoreD::load(array_re.as_ref()));
            twiddles.push(AvxStoreD::load(array_im.as_ref()));
        }

        uk += 4;
    }

    let remainder = count - (count / 4) * 4;
    if remainder > 0 {
        let k = k_start + uk;

        let mut array_re = [0.0_f64; 4];
        let mut array_im = [0.0_f64; 4];
        for m in 0..inner_groups {
            for i in 0..remainder {
                let layer = radixq_dct3_n_rotation_twiddle::<f64>(q, m, (k + i).as_(), len);
                array_re[i] = layer.re;
                array_im[i] = layer.im;
            }

            twiddles.push(AvxStoreD::load(array_re.as_ref()));
            twiddles.push(AvxStoreD::load(array_im.as_ref()));
        }
    }

    twiddles
}

pub(crate) struct AvxDct3MixedRadix3d {
    inner_dct3: Arc<dyn PxdctExecutor<f64> + Send + Sync>,
    inner_dct_scratch_size: usize,
    rotation_twiddles: Vec<AvxStoreD>,
    execution_length: usize,
    p: usize, // = lanes / 3
    s: usize, // = 2N / 3
}

impl AvxDct3MixedRadix3d {
    pub(crate) fn new(
        len: usize,
        dct3: Arc<dyn PxdctExecutor<f64> + Send + Sync>,
    ) -> Result<Self, PxdctError> {
        unsafe { Self::new_init(len, dct3) }
    }

    #[target_feature(enable = "avx2")]
    fn new_init(
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
            rotation_twiddles: dct3_radix_n_rotation_twiddles_avx_d(3, p, len, 1),
            p,
            s: 2 * len / 3,
        })
    }
}

impl AvxDct3MixedRadix3d {
    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    fn exec_stage1<S: BidirectionalStore<f64>, L: AvxLanesD>(
        &self,
        data: &S,
        a_buffer: &mut [f64],
        c_buffer: &mut [f64],
        w_buffer: &mut [f64],
        uk: usize,
        k: usize,
        access: L,
    ) {
        let lanes = access.len();
        let p = self.p;
        let s = self.s;

        let xk = AvxStoreD::load_lanes(access, data.slice_from(k..));
        let xp = AvxStoreD::load_lanes(access, data.slice_from(s + k..));
        let xm = AvxStoreD::load_lanes(access, data.slice_from(s - lanes - k + 1..))
            .reverse_lanes(access);

        let s1 = xp + xm;
        let t1 = xp - xm;

        // U(k) = X(k) - S1(k)
        let a_v = xk - s1;
        unsafe {
            a_v.write_lanes(access, a_buffer.get_unchecked_mut(k..));
        }

        let c_v = fma(s1, AvxStoreD::dup(0.5_f64), xk);
        let s_v = t1 * AvxStoreD::dup(f64::SQRT_3_OVER_2);

        let twiddle_re = unsafe { *self.rotation_twiddles.get_unchecked(uk) };
        let twiddle_im = unsafe { *self.rotation_twiddles.get_unchecked(uk + 1) };

        let v_v = fma(c_v, twiddle_re, s_v * twiddle_im);
        let w_v = fma(c_v, twiddle_im, -s_v * twiddle_re);

        unsafe {
            v_v.write_lanes(access, c_buffer.get_unchecked_mut(k..));
            w_v.reverse_lanes(access)
                .write_lanes(access, w_buffer.get_unchecked_mut(p - lanes - k + 1..));
        }
    }

    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    fn exec_stage3<S: BidirectionalStore<f64>, L: AvxLanesD>(
        &self,
        data: &mut S,
        a_buffer: &[f64],
        c_buffer: &[f64],
        w_buffer: &[f64],
        dc_adjust_a_v: AvxStoreD,
        dc_adjust_fg_v: AvxStoreD,
        w0_half_v: AvxStoreD,
        sign_v: AvxStoreD,
        n: usize,
        access: L,
    ) {
        let lanes = access.len();
        // Center stream (3n+1): a + dc_adjust_a
        let a_v = AvxStoreD::load_lanes(access, unsafe { a_buffer.get_unchecked(n..) });
        let center = a_v + dc_adjust_a_v;

        // Outer streams (3n+0, 3n+2)
        let f_v = AvxStoreD::load_lanes(access, unsafe { c_buffer.get_unchecked(n..) });
        let g_raw = AvxStoreD::load_lanes(access, unsafe { w_buffer.get_unchecked(n..) });
        // g_v = g_raw * sign + w0_half * sign = (g_raw + w0_half) * sign
        let g_v = fma(g_raw, sign_v, w0_half_v * sign_v);
        let f_dc = f_v + dc_adjust_fg_v;
        let out0 = f_dc + g_v;
        let out2 = f_dc - g_v;

        let center_a = center.to_array();
        let out0_a = out0.to_array();
        let out2_a = out2.to_array();
        for i in 0..lanes {
            let base = 3 * (n + i);
            data[base] = out0_a[i];
            data[base + 1] = center_a[i];
            data[base + 2] = out2_a[i];
        }
    }

    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_store<S: BidirectionalStore<f64>>(
        &self,
        data: &mut S,
        scratch: &mut [f64],
    ) -> Result<(), PxdctError> {
        let p = self.p;
        let s = self.s;

        let (scratch, inner_scratch) = scratch.split_at_mut(self.execution_length);
        let (a_buffer, cw_buffer) = scratch.split_at_mut(p);
        let (c_buffer, w_buffer) = cw_buffer.split_at_mut(p);

        let x0 = data[0];
        let x_s = data[s]; // X(2N/3)
        let x_p = data[p]; // X(lanes/3)

        a_buffer[0] = x0 - x_s;
        c_buffer[0] = fma(0.5_f64, x_s, x0);
        w_buffer[0] = x_p * f64::SQRT_3_OVER_2;

        let mut uk = 0usize;
        let mut k = 1usize;
        while k + 4 <= p {
            self.exec_stage1::<S, _>(data, a_buffer, c_buffer, w_buffer, uk, k, AvxFullD);
            uk += 2;
            k += 4;
        }

        let remainder = p - k;
        if remainder != 0 {
            let tail = AvxTailD::new(remainder);
            self.exec_stage1::<S, _>(data, a_buffer, c_buffer, w_buffer, uk, k, tail);
        }

        let x0_half = x0 * 0.5;
        let u0_half = a_buffer[0] * 0.5;
        let v0_half = c_buffer[0] * 0.5;
        let w0_half = w_buffer[0] * 0.5;

        self.inner_dct3
            .execute_with_scratch(scratch, inner_scratch)?;

        let (a_buffer, cw_buffer) = scratch.split_at_mut(p);
        let (c_buffer, w_buffer) = cw_buffer.split_at_mut(p);

        let dc_adjust_a_v = AvxStoreD::dup(u0_half - x0_half);
        let dc_adjust_fg_v = AvxStoreD::dup(v0_half - x0_half);
        let w0_half_v = AvxStoreD::dup(w0_half);
        let sign_even = AvxStoreD::load(&[1.0_f64, -1.0, 1.0, -1.0]);

        let mut n = 0usize;
        while n + 4 <= p {
            self.exec_stage3::<S, _>(
                data,
                a_buffer,
                c_buffer,
                w_buffer,
                dc_adjust_a_v,
                dc_adjust_fg_v,
                w0_half_v,
                sign_even,
                n,
                AvxFullD,
            );
            n += 4;
        }

        let remainder = p - n;
        if remainder != 0 {
            let tail = AvxTailD::new(remainder);
            self.exec_stage3::<S, _>(
                data,
                a_buffer,
                c_buffer,
                w_buffer,
                dc_adjust_a_v,
                dc_adjust_fg_v,
                w0_half_v,
                sign_even,
                n,
                tail,
            );
        }

        Ok(())
    }
}

boring_avx_mixed_radix!(AvxDct3MixedRadix3d, f64);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Pxdct;
    use crate::tests::naive_dct3;
    use crate::type3::Dct3Butterfly3;
    use crate::util::has_valid_avx;
    use rand::RngExt;

    #[test]
    fn test_neon_dct3_radix3() {
        if !has_valid_avx() {
            return;
        }
        const LENGTH: usize = 3 * 3;
        let mut input = vec![0.0_f64; LENGTH];
        for z in input.iter_mut() {
            *z = rand::rng().random_range(1.0..2.0);
        }
        let reference = naive_dct3(&input);

        let bf = AvxDct3MixedRadix3d::new(LENGTH, Arc::new(Dct3Butterfly3::default())).unwrap();
        bf.execute(&mut input).unwrap();
        for (i, (&a, &b)) in input.iter().zip(reference.iter()).enumerate() {
            assert!((a - b).abs() < 1e-1, "mismatch at {i}: {a} vs {b}");
        }
    }

    #[test]
    fn test_neon_dct3_radix3_large() {
        if !has_valid_avx() {
            return;
        }
        const LENGTH: usize = 3 * 9;
        let mut input = vec![0.0_f64; LENGTH];
        for z in input.iter_mut() {
            *z = rand::rng().random_range(1.0..2.0);
        }
        let reference = naive_dct3(&input);

        let inner: Arc<dyn PxdctExecutor<f64> + Send + Sync> = Pxdct::make_dct3_f64(9).unwrap();
        let bf = AvxDct3MixedRadix3d::new(LENGTH, inner).unwrap();
        bf.execute(&mut input).unwrap();
        for (i, (&a, &b)) in input.iter().zip(reference.iter()).enumerate() {
            assert!((a - b).abs() < 1e-1, "mismatch at {i}: {a} vs {b}");
        }
    }
}
