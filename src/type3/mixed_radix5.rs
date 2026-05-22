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

use crate::bidirectional::{BidirectionalStore, InPlaceStore};
use crate::mla::fmla;
use crate::type3::mixed_radix3::radixq_dct3_n_rotation_twiddles;
use crate::util::{DctSample, try_vec, validate_scratch};
use crate::{PxdctError, PxdctExecutor};
use num_complex::Complex;
use num_traits::AsPrimitive;
use std::sync::Arc;

pub(crate) trait Dct3MixedRadix5Sample {
    const D3_R5_T0: Self;
    const D3_R5_T1: Self;
    const D3_R5_T2: Self;
    const D3_R5_T3: Self;
}

impl Dct3MixedRadix5Sample for f32 {
    // import struct
    // from sage.all import *
    // R = RealField(256)
    // def float_to_hex(f):
    //     packed = struct.pack('>f', float(f))
    //     return '0x' + packed.hex()
    // print(float_to_hex(float(abs(sin(pi * R(4) / R(10))))))
    const D3_R5_T0: Self = f32::from_bits(0x3f737871);
    // import struct
    // from sage.all import *
    // R = RealField(256)
    // def float_to_hex(f):
    //     packed = struct.pack('>f', float(f))
    //     return '0x' + packed.hex()
    // print(float_to_hex(float(abs(cos(pi * R(4) / R(5))))))
    const D3_R5_T1: Self = f32::from_bits(0x3f4f1bbd);
    // import struct
    // from sage.all import *
    // R = RealField(256)
    // def float_to_hex(f):
    //     packed = struct.pack('>f', float(f))
    //     return '0x' + packed.hex()
    // print(float_to_hex(float(abs(sin(pi * R(4) / R(5))))))
    const D3_R5_T2: Self = f32::from_bits(0x3f167918);
    // import struct
    // from sage.all import *
    // R = RealField(256)
    // def float_to_hex(f):
    //     packed = struct.pack('>f', float(f))
    //     return '0x' + packed.hex()
    // print(float_to_hex(float(abs(cos(pi * R(8) / R(5))))))
    const D3_R5_T3: Self = f32::from_bits(0x3e9e377a);
}

impl Dct3MixedRadix5Sample for f64 {
    // import struct
    // from sage.all import *
    // R = RealField(256)
    // def double_to_hex(f):
    //     packed = struct.pack('>d', float(f))
    //     return '0x' + packed.hex()
    // print(double_to_hex(float(abs(sin(pi * R(4) / R(10))))))
    const D3_R5_T0: Self = f64::from_bits(0x3fee6f0e13445500);
    // import struct
    // from sage.all import *
    // R = RealField(256)
    // def double_to_hex(f):
    //     packed = struct.pack('>d', float(f))
    //     return '0x' + packed.hex()
    // print(double_to_hex(float(abs(cos(pi * R(4) / R(5))))))
    const D3_R5_T1: Self = f64::from_bits(0x3fe9e3779b97f4a7);
    // import struct
    // from sage.all import *
    // R = RealField(256)
    // def double_to_hex(f):
    //     packed = struct.pack('>d', float(f))
    //     return '0x' + packed.hex()
    // print(double_to_hex(float(abs(sin(pi * R(4) / R(5))))))
    const D3_R5_T2: Self = f64::from_bits(0x3fe2cf2304755a5f);
    // import struct
    // from sage.all import *
    // R = RealField(256)
    // def double_to_hex(f):
    //     packed = struct.pack('>d', float(f))
    //     return '0x' + packed.hex()
    // print(double_to_hex(float(abs(cos(pi * R(8) / R(5))))))
    const D3_R5_T3: Self = f64::from_bits(0x3fd3c6ef372fe94c);
}

pub(crate) struct Dct3MixedRadix5<T> {
    inner_dct3: Arc<dyn PxdctExecutor<T> + Send + Sync>,
    rotation_twiddles: Vec<Complex<T>>,
    execution_length: usize,
    p: usize, // = N / q
    inner_dct3_scratch_size: usize,
}

impl<T: DctSample> Dct3MixedRadix5<T>
where
    f64: AsPrimitive<T>,
    usize: AsPrimitive<T>,
{
    #[allow(unused)]
    pub(crate) fn new(
        len: usize,
        inner_dct3: Arc<dyn PxdctExecutor<T> + Send + Sync>,
    ) -> Result<Self, PxdctError> {
        assert_eq!(
            inner_dct3.length(),
            len / 5,
            "DCT-III Mixed-Radix-5 inner DCT-III length must be N / 5"
        );

        let p = len / 5;
        let rotation_twiddles = radixq_dct3_n_rotation_twiddles(5, len)?;

        let inner_dct3_scratch_size = inner_dct3.scratch_size();
        Ok(Self {
            inner_dct3,
            rotation_twiddles,
            execution_length: len,
            p,
            inner_dct3_scratch_size,
        })
    }
}

impl<T: DctSample + Dct3MixedRadix5Sample> Dct3MixedRadix5<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    fn execute_with_store<S: BidirectionalStore<T>>(
        &self,
        data: &mut S,
        scratch: &mut [T],
    ) -> Result<(), PxdctError> {
        let p = self.p;

        let (scratch, inner_scratch) = scratch.split_at_mut(self.execution_length);
        let (a_buffer, vw_buffer) = scratch.split_at_mut(p);
        let (v_buffer, w_buffer) = vw_buffer.split_at_mut(2 * p);

        a_buffer[0] = data[0] - data[2 * p] + data[4 * p];

        // V(0, m) for each m.
        let mut v0_m0 = data[0];
        v0_m0 = fmla(data[2 * p], T::D3_R5_T1, v0_m0);
        v0_m0 = fmla(data[4 * p], T::D3_R5_T3, v0_m0);
        v_buffer[0] = v0_m0;
        let mut v0_m1 = data[0];
        v0_m1 = fmla(data[2 * p], -T::D3_R5_T3, v0_m1);
        v0_m1 = fmla(data[4 * p], -T::D3_R5_T1, v0_m1);
        v_buffer[p] = v0_m1;

        // W'(0, m) = W(p, m) for each m.
        let mut w0_m0 = data[p] * T::D3_R5_T0;
        w0_m0 = fmla(data[3 * p], T::D3_R5_T2, w0_m0);
        w_buffer[0] = w0_m0;
        let mut w0_m1 = data[p] * T::D3_R5_T2;
        w0_m1 = fmla(data[3 * p], -T::D3_R5_T0, w0_m1);
        w_buffer[p] = w0_m1;

        for k in 1..p {
            let xk = data[k];

            let xp_1 = data[2 * p + k];
            let xm_1 = data[2 * p - k];
            let s_1 = xp_1 + xm_1;
            let t_1 = xp_1 - xm_1;
            let xp_2 = data[4 * p + k];
            let xm_2 = data[4 * p - k];
            let s_2 = xp_2 + xm_2;
            let t_2 = xp_2 - xm_2;

            // U(k) = X(k) + Σ_{i=1..qh} (-1)^i S_i(k)
            unsafe {
                *a_buffer.get_unchecked_mut(k) = xk - s_1 + s_2;
            }

            let mut c_acc0 = xk;
            c_acc0 = fmla(s_1, T::D3_R5_T1, c_acc0);
            c_acc0 = fmla(s_2, T::D3_R5_T3, c_acc0);
            let mut s_acc0 = -t_1 * T::D3_R5_T2;
            s_acc0 = fmla(t_2, -T::D3_R5_T0, s_acc0);
            let r0 = unsafe { self.rotation_twiddles.get_unchecked(k) };
            let v_val0 = fmla(c_acc0, r0.re, -s_acc0 * r0.im);
            let w_val0 = fmla(c_acc0, r0.im, s_acc0 * r0.re);
            unsafe {
                *v_buffer.get_unchecked_mut(k) = v_val0;
            }
            unsafe {
                *w_buffer.get_unchecked_mut(p - k) = w_val0;
            }

            let mut c_acc1 = xk;
            c_acc1 = fmla(s_1, -T::D3_R5_T3, c_acc1);
            c_acc1 = fmla(s_2, -T::D3_R5_T1, c_acc1);
            let mut s_acc1 = -t_1 * T::D3_R5_T0;
            s_acc1 = fmla(t_2, T::D3_R5_T2, s_acc1);
            let r1 = unsafe { self.rotation_twiddles.get_unchecked(p + k) };
            let v_val1 = fmla(c_acc1, r1.re, -s_acc1 * r1.im);
            let w_val1 = fmla(c_acc1, r1.im, s_acc1 * r1.re);
            unsafe {
                *v_buffer.get_unchecked_mut(p + k) = v_val1;
            }
            unsafe {
                *w_buffer.get_unchecked_mut(p + (p - k)) = w_val1;
            }
        }

        let x0_half = data[0] * T::HALF;
        let u0_half = a_buffer[0] * T::HALF;
        let v0_m0_half = v_buffer[0] * T::HALF;
        let v0_m1_half = v_buffer[p] * T::HALF;
        let w0_m0_half = w_buffer[0] * T::HALF;
        let w0_m1_half = w_buffer[p] * T::HALF;

        self.inner_dct3
            .execute_with_scratch(scratch, inner_scratch)?;

        let (a_buffer, vw_buffer) = scratch.split_at_mut(p);
        let (v_buffer, w_buffer) = vw_buffer.split_at_mut(2 * p);

        let dc_adjust_a = u0_half - x0_half;

        let mut sign = T::one();
        for n in 0..p {
            // Center stream
            data[5 * n + 2] = unsafe { *a_buffer.get_unchecked(n) } + dc_adjust_a;

            let f_v0 = unsafe { *v_buffer.get_unchecked(n) };
            let g_raw0 = unsafe { *w_buffer.get_unchecked(n) };
            let g_v0 = fmla(g_raw0, sign, w0_m0_half.mulsign(sign));
            let f_dc0 = f_v0 + (v0_m0_half - x0_half);
            data[5 * n] = f_dc0 + g_v0;
            data[5 * n + 4] = f_dc0 - g_v0;
            let f_v1 = unsafe { *v_buffer.get_unchecked(p + n) };
            let g_raw1 = unsafe { *w_buffer.get_unchecked(p + n) };
            let g_v1 = fmla(g_raw1, sign, w0_m1_half.mulsign(sign));
            let f_dc1 = f_v1 + (v0_m1_half - x0_half);
            data[5 * n + 1] = f_dc1 + g_v1;
            data[5 * n + 3] = f_dc1 - g_v1;

            sign = -sign;
        }

        Ok(())
    }
}

impl<T: DctSample + Dct3MixedRadix5Sample> PxdctExecutor<T> for Dct3MixedRadix5<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, data: &mut [T]) -> Result<(), PxdctError> {
        let mut scratch = try_vec![T::default(); self.scratch_size()];
        self.execute_with_scratch(data, &mut scratch)
    }

    fn execute_with_scratch(&self, data: &mut [T], scratch: &mut [T]) -> Result<(), PxdctError> {
        if !data.len().is_multiple_of(self.execution_length) {
            return Err(PxdctError::InvalidSizeMultiplier(
                data.len(),
                self.execution_length,
            ));
        }

        let full_scratch = validate_scratch!(scratch, self.scratch_size());

        for chunk in data.chunks_exact_mut(self.execution_length) {
            self.execute_with_store(&mut InPlaceStore::new(chunk), full_scratch)?;
        }

        Ok(())
    }

    fn execute_into(&self, input: &[T], output: &mut [T]) -> Result<(), PxdctError> {
        let mut scratch = try_vec![T::default(); self.scratch_size()];
        self.execute_into_with_scratch(input, output, &mut scratch)
    }

    fn execute_into_with_scratch(
        &self,
        input: &[T],
        output: &mut [T],
        scratch: &mut [T],
    ) -> Result<(), PxdctError> {
        use crate::util::validate_oof_sizes;
        validate_oof_sizes!(input, output, self.execution_length);

        let full_scratch = validate_scratch!(scratch, self.scratch_size());

        use crate::bidirectional::BiStore;
        for (src, dst) in input
            .chunks_exact(self.execution_length)
            .zip(output.chunks_exact_mut(self.execution_length))
        {
            self.execute_with_store(&mut BiStore::new(src, dst), full_scratch)?;
        }
        Ok(())
    }

    #[inline]
    fn length(&self) -> usize {
        self.execution_length
    }

    #[inline]
    fn scratch_size(&self) -> usize {
        self.execution_length + self.inner_dct3_scratch_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::naive_dct3;
    use crate::type3::Dct3Butterfly5;
    use rand::RngExt;

    #[test]
    fn test_split_dct3_radix5() {
        const N: usize = 5 * 5;
        let mut input = vec![0.0; N];
        for z in input.iter_mut() {
            *z = rand::rng().random_range(1.0..2.0);
        }
        let reference = naive_dct3(&input);
        let bf = Dct3MixedRadix5::new(N, Arc::new(Dct3Butterfly5::default())).unwrap();
        bf.execute(&mut input).unwrap();
        for (i, (&a, &b)) in input.iter().zip(reference.iter()).enumerate() {
            assert!((a - b).abs() < 1e-1, "mismatch at {i}: {a} vs {b}");
        }
        let _ = reference;
    }
}
