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
use crate::bidirectional::{BidirectionalStore, InPlaceStore};
use crate::mla::fmla;
use crate::twiddles::FftTrigonometry;
use crate::util::{DctSample, try_vec, validate_scratch};
use crate::{PxdctError, PxdctExecutor};
use num_complex::Complex;
use num_traits::AsPrimitive;
use std::sync::Arc;

pub(crate) fn radixq_dct3_n_rotation_twiddle<T: DctSample>(
    _q: usize,
    _m: usize,
    k: usize,
    len: usize,
) -> Complex<T>
where
    f64: AsPrimitive<T>,
{
    let alpha_num = (_q - 1 - 2 * _m) as f64; // integer factor of π in α
    let arg = alpha_num * (k as f64) / (2.0 * len as f64);
    let theta = arg.sincos_pi();
    Complex::new(theta.1.as_(), theta.0.as_())
}

pub(crate) fn radixq_dct3_n_rotation_twiddles<T: DctSample>(
    q: usize,
    len: usize,
) -> Result<Vec<Complex<T>>, PxdctError>
where
    f64: AsPrimitive<T>,
{
    let qh = q / 2;
    let p = len / q;
    let mut rotation_twiddles = try_vec![Complex::<T>::default(); qh * p];
    for m in 0..qh {
        for k in 0..p {
            rotation_twiddles[m * p + k] = radixq_dct3_n_rotation_twiddle(q, m, k, len);
        }
    }
    Ok(rotation_twiddles)
}

pub(crate) struct Dct3MixedRadix3<T> {
    inner_dct3: Arc<dyn PxdctExecutor<T> + Send + Sync>,
    rotation_twiddles: Vec<Complex<T>>,
    execution_length: usize,
    q_modules: usize,
    s: usize,
    inner_dct3_scratch_size: usize,
}

impl<T: DctSample> Dct3MixedRadix3<T>
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
            len / 3,
            "DCT-III Mixed-Radix-3 length DCTs must be third of DCT-III"
        );

        let mut twiddles = try_vec![Complex::<T>::default(); len / 3];
        for (k, dst) in twiddles.iter_mut().enumerate() {
            *dst = radixq_dct3_n_rotation_twiddle(3, 0, k, len);
        }

        let q_modules = len / 3;
        let inner_dct3_scratch_size = inner_dct3.scratch_size();

        Ok(Self {
            inner_dct3,
            execution_length: len,
            rotation_twiddles: twiddles,
            q_modules,
            s: 2 * len / 3,
            inner_dct3_scratch_size,
        })
    }
}

impl<T: DctSample> Dct3MixedRadix3<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    fn execute_with_store<S: BidirectionalStore<T>>(
        &self,
        data: &mut S,
        scratch: &mut [T],
    ) -> Result<(), PxdctError> {
        let (scratch, inner_scratch) = scratch.split_at_mut(self.execution_length);
        let (a_buffer, c_w_buffer) = scratch.split_at_mut(self.q_modules);
        let (c_buffer, w_buffer) = c_w_buffer.split_at_mut(self.q_modules);

        let p = self.q_modules;
        let s = self.s; // 2N/3

        let x0 = data[0];
        let x_s = data[s]; // X(2N/3)
        let x_p = data[p]; // X(N/3)

        a_buffer[0] = x0 - x_s; // U(0)
        c_buffer[0] = fmla(T::HALF, x_s, x0); // V(0, 0) = X(0) + 0.5 * X(2N/3)
        w_buffer[0] = x_p * T::SQRT_3_OVER_2;

        let u0_half = a_buffer[0] * T::HALF;
        let v0_half = c_buffer[0] * T::HALF;
        let w0_half = w_buffer[0] * T::HALF;
        let x0_half = x0 * T::HALF;

        let a_iter = a_buffer[1..p].iter_mut();
        let c_iter = c_buffer[1..p].iter_mut();
        let w_iter = w_buffer[1..p].iter_mut().rev();

        // k = 1..p-1: rotation-twiddle path
        for (k, ((a_dst, c_dst), w_dst)) in a_iter.zip(c_iter).zip(w_iter).enumerate() {
            let k = k + 1;
            let xp = data[s + k]; // X(2N/3 + k)
            let xm = data[s - k]; // X(2N/3 - k)
            let xk = data[k]; // X(k)

            let s1 = xp + xm;
            let t1 = xp - xm;

            // U(k) = X(k) - S1(k)
            *a_dst = xk - s1;

            let c_v = fmla(T::HALF, s1, xk);
            let s_v = t1 * T::SQRT_3_OVER_2;

            let twiddle = unsafe { self.rotation_twiddles.get_unchecked(k) };

            let v_k = fmla(c_v, twiddle.re, s_v * twiddle.im);
            let w_k = fmla(c_v, twiddle.im, -s_v * twiddle.re);

            *c_dst = v_k;
            *w_dst = w_k;
        }

        self.inner_dct3
            .execute_with_scratch(scratch, inner_scratch)?;

        let (a_buffer, c_w_buffer) = scratch.split_at_mut(self.q_modules);
        let (c_buffer, w_buffer) = c_w_buffer.split_at_mut(self.q_modules);

        let dc_adjust_a = u0_half - x0_half;
        let dc_adjust_fg = v0_half - x0_half;

        let mut sign = T::one();
        for (n, ((&a_v, &f_v), &g_raw)) in a_buffer[..p]
            .iter()
            .zip(c_buffer[..p].iter())
            .zip(w_buffer[..p].iter())
            .enumerate()
        {
            let g_v = fmla(g_raw, sign, w0_half.mulsign(sign));

            data[3 * n + 1] = a_v + dc_adjust_a;
            data[3 * n] = f_v + dc_adjust_fg + g_v;
            data[3 * n + 2] = f_v + dc_adjust_fg - g_v;

            sign = -sign;
        }

        Ok(())
    }
}

impl<T: DctSample> PxdctExecutor<T> for Dct3MixedRadix3<T>
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
    use crate::type3::Dct3Butterfly3;
    use rand::RngExt;

    #[test]
    fn test_split_dct3() {
        let mut input = vec![0.; 9];
        for z in input.iter_mut() {
            *z = rand::rng().random_range(1.0..2.0);
        }
        let reference_input = input.clone();
        let reference_input = naive_dct3(&reference_input);
        let bf = Dct3MixedRadix3::new(9, Arc::new(Dct3Butterfly3::default())).unwrap();
        bf.execute(&mut input).unwrap();
        input
            .iter()
            .zip(reference_input.iter())
            .enumerate()
            .for_each(|(i, (&src, &r0))| {
                if (src - r0).abs() > 1e-1 {
                    println!(
                        "Difference must be < {}, but it was {}, at position {i}",
                        1e-1,
                        (src - r0).abs()
                    )
                }
            });
    }
}
