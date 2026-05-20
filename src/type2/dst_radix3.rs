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
 *//*
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

use crate::bidirectional::{BiStore, BidirectionalStore, InPlaceStore};
use crate::mla::fmla;
use crate::twiddles::compute_twiddle;
use crate::util::{DctSample, try_vec};
use crate::{PxdctError, PxdctExecutor};
use num_complex::Complex;
use num_traits::AsPrimitive;
use std::sync::Arc;

/// Radix-3 DST-II Implementation
/// Mathematically corrected from Murty & Padhy (2015). Decomposes a DST-II
/// of length N into exactly THREE DST-II transforms of length N/3 (P, Q, W).
pub(crate) struct Dst2Radix3<T: DctSample> {
    dst2_subtransform: Arc<dyn PxdctExecutor<T> + Send + Sync>,
    execution_length: usize,
    /// Precomputed twiddles in linear layout: 3 Complex<T> per i, ordered
    /// [t1, t2, t3] for i = 0..n_sub. Each is stored as conj(compute_twiddle(...)) so that
    /// both components are read directly in the hot loop without negation:
    ///   t1.re = cos(angle_base),              t1.im = sin(angle_base)
    ///   t2.re = cos(angle_shift-angle_base),  t2.im = sin(angle_shift-angle_base)
    ///   t3.re = cos(angle_shift+angle_base),  t3.im = sin(angle_shift+angle_base)
    twiddles: Box<[Complex<T>]>,
}

impl<T: DctSample> Dst2Radix3<T>
where
    f64: AsPrimitive<T>,
{
    pub(crate) fn new(
        dst2_subtransform: Arc<dyn PxdctExecutor<T> + Send + Sync>,
    ) -> Result<Dst2Radix3<T>, PxdctError> {
        let sub_len = dst2_subtransform.length();
        let execution_length = sub_len * 3;
        let n_sub = sub_len;

        // Precompute all twiddles as Complex<T> in linear [t1, t2, t3] layout.
        //
        // For each i in 0..n_sub, with n = i + 1 and N = execution_length:
        //     angle_base  = (2n - 1) * pi / (2N)
        //     angle_shift = 2 * pi / 3
        //
        // compute_twiddle(index, fft_len) returns:
        //     Complex { re: cos(2*pi*index/fft_len), im: -sin(2*pi*index/fft_len) }
        //
        // Storing conj() of each result flips im to +sin(θ), so the hot loop
        // reads t.re and t.im directly with no negation at the use site.
        //
        // Choose fft_len = 12 * N so that 2*pi/fft_len = pi/(6N), the common
        // unit for all three angles:
        //     angle_base               = 3*(2n-1)        * (pi / (6N))  →  index1 = 3*(2n-1)
        //     angle_shift - angle_base = (4N-3*(2n-1))   * (pi / (6N))  →  index2 = 4N-3*(2n-1)  (mod fft_len)
        //     angle_shift + angle_base = (4N+3*(2n-1))   * (pi / (6N))  →  index3 = 4N+3*(2n-1)
        //
        // Note: t3 = conj(t2) when the two angles are symmetric around angle_shift, but that
        // only holds when angle_base = 0; in general they are independent. Store all three.

        let fft_len = 12 * execution_length;
        let four_n = 4 * execution_length;
        let mut twiddles: Vec<Complex<T>> = Vec::with_capacity(n_sub * 3);

        for i in 0..n_sub {
            let n = i + 1;
            let k = 3 * (2 * n - 1); // angle_base in units of pi/(6N)

            // t1: angle_base
            let t1 = compute_twiddle::<T>(k, fft_len);
            twiddles.push(t1.conj());

            // t2: angle_shift - angle_base.
            // index = 4N - k, clamped mod fft_len (can go negative for large n).
            let idx2 = (four_n as isize - k as isize).rem_euclid(fft_len as isize) as usize;
            let t2 = compute_twiddle::<T>(idx2, fft_len);
            twiddles.push(t2.conj());

            // t3: angle_shift + angle_base.
            let idx3 = (four_n + k) % fft_len;
            let t3 = compute_twiddle::<T>(idx3, fft_len);
            twiddles.push(t3.conj());
        }

        Ok(Dst2Radix3 {
            dst2_subtransform,
            execution_length,
            twiddles: twiddles.into_boxed_slice(),
        })
    }

    fn process_chunk<S: BidirectionalStore<T>>(
        &self,
        chunk: &mut S,
        p: &mut [T],
        q: &mut [T],
        w: &mut [T],
        sub_out: &mut [T],
        sub_scratch: &mut [T],
    ) -> Result<(), PxdctError> {
        let n_sub = self.execution_length / 3;

        // --- Butterfly: scatter src into p, q, w ---
        // tw = [t1, t2, t3]; t.re = cos(θ), t.im = sin(θ)
        p.iter_mut()
            .zip(q.iter_mut())
            .zip(w.iter_mut())
            .zip(self.twiddles.as_chunks::<3>().0.iter())
            .enumerate()
            .for_each(|(i, (((p_i, q_i), w_i), tw))| {
                let x_1 = chunk[i];
                let x_2 = chunk[2 * n_sub - 1 - i];
                let x_3 = chunk[i + 2 * n_sub];

                *p_i = x_1 - x_2 + x_3;
                *q_i = fmla(x_1, tw[0].re, fmla(-x_2, tw[1].re, x_3 * tw[2].re));
                let r_val = fmla(x_1, tw[0].im, fmla(x_2, tw[1].im, x_3 * tw[2].im));

                // W(n) = (-1)^(n+1) * R(n)
                *w_i = if i % 2 == 0 { r_val } else { -r_val };
            });

        let (y_p, rest) = sub_out.split_at_mut(n_sub);
        let (y_q, y_w) = rest.split_at_mut(n_sub);

        self.dst2_subtransform
            .execute_into_with_scratch(p, y_p, sub_scratch)?;
        self.dst2_subtransform
            .execute_into_with_scratch(q, y_q, sub_scratch)?;
        self.dst2_subtransform
            .execute_into_with_scratch(w, y_w, sub_scratch)?;

        for k in 0..n_sub - 1 {
            let w_val = unsafe { *y_w.get_unchecked(n_sub - k - 2) };
            let q_val = unsafe { *y_q.get_unchecked(n_sub - k - 2) };

            unsafe { chunk[3 * k + 2] = *y_p.get_unchecked(k) };
            unsafe { chunk[3 * k + 1] = *y_q.get_unchecked(k) - w_val };
            unsafe { chunk[self.execution_length - 3 * k - 3] = q_val + *y_w.get_unchecked(k) };
        }

        // k == n_sub - 1: w_val = 0, q_val = 0
        let k = n_sub - 1;
        unsafe { chunk[3 * k + 2] = *y_p.get_unchecked(k) };
        unsafe { chunk[3 * k + 1] = *y_q.get_unchecked(k) };
        unsafe { chunk[self.execution_length - 3 * k - 3] = *y_w.get_unchecked(k) };

        Ok(())
    }
}

impl<T: DctSample> PxdctExecutor<T> for Dst2Radix3<T>
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

        // scratch layout: [p | q | w | sub_out | sub_scratch]
        let n_sub = self.execution_length / 3;
        let (sub_buffers, rest) = scratch.split_at_mut(self.execution_length);
        let (p, rest_buffers) = sub_buffers.split_at_mut(n_sub);
        let (q, w) = rest_buffers.split_at_mut(n_sub);
        let (sub_out, sub_scratch) = rest.split_at_mut(self.execution_length);

        for chunk in data.chunks_exact_mut(self.execution_length) {
            self.process_chunk(&mut InPlaceStore::new(chunk), p, q, w, sub_out, sub_scratch)?;
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

        let n_sub = self.execution_length / 3;

        // scratch layout: [p | q | w | sub_out | sub_scratch]
        let (sub_buffers, rest) = scratch.split_at_mut(self.execution_length);
        let (p, rest_buffers) = sub_buffers.split_at_mut(n_sub);
        let (q, w) = rest_buffers.split_at_mut(n_sub);
        let (sub_out, sub_scratch) = rest.split_at_mut(self.execution_length);

        for (src, dst) in input
            .chunks_exact(self.execution_length)
            .zip(output.chunks_exact_mut(self.execution_length))
        {
            self.process_chunk(&mut BiStore::new(src, dst), p, q, w, sub_out, sub_scratch)?;
        }

        Ok(())
    }

    #[inline]
    fn length(&self) -> usize {
        self.execution_length
    }

    fn scratch_size(&self) -> usize {
        (self.execution_length * 2) + self.dst2_subtransform.scratch_size()
    }
}
#[cfg(test)]
mod tests {
    use crate::tests::naive_dst2;
    use crate::type2::dst_radix3::Dst2Radix3;
    use crate::{Pxdct, PxdctExecutor};

    #[test]
    fn test_dct7_size9() {
        let mut array = vec![
            1.0f64, 3.0, 1.0f64, 1.0f64, 3.0, 1.0f64, 1.0f64, 3.0, 1.0f64,
        ];
        let dct = Dst2Radix3::<f64>::new(Pxdct::make_dst2_f64(3).unwrap()).unwrap();
        let control = naive_dst2(&array);
        dct.execute(&mut array).unwrap();
        array
            .iter()
            .zip(control.iter())
            .enumerate()
            .for_each(|(i, (&x, &c))| {
                assert!((x - c).abs() < 1e-9, "index {i}: got {x}, expected {c}");
            });
    }

    #[test]
    fn test_dct7_size12() {
        let mut array = vec![
            1.0f64, 3.0, 1.0f64, 1.0f64, 3.0, 1.0f64, 1.0f64, 3.0, 1.0f64, 1.0f64, 3.0, 1.0f64,
        ];
        let dct = Dst2Radix3::<f64>::new(Pxdct::make_dst2_f64(4).unwrap()).unwrap();
        let control = naive_dst2(&array);
        dct.execute(&mut array).unwrap();
        array
            .iter()
            .zip(control.iter())
            .enumerate()
            .for_each(|(i, (&x, &c))| {
                assert!((x - c).abs() < 1e-9, "index {i}: got {x}, expected {c}");
            });
    }

    #[test]
    fn test_dct7_size27() {
        let mut array = vec![
            1.0f64, 3.0, 1.0f64, 1.0f64, 3.0, 1.0f64, 1.0f64, 3.0, 1.0f64, 1.0f64, 3.0, 1.0f64,
            1.0f64, 3.0, 1.0f64, 1.0f64, 3.0, 1.0f64, 1.0f64, 3.0, 1.0f64, 1.0f64, 3.0, 1.0f64,
            1.0f64, 3.0, 1.0f64,
        ];
        let dct = Dst2Radix3::<f64>::new(Pxdct::make_dst2_f64(9).unwrap()).unwrap();
        let control = naive_dst2(&array);
        dct.execute(&mut array).unwrap();
        array
            .iter()
            .zip(control.iter())
            .enumerate()
            .for_each(|(i, (&x, &c))| {
                assert!((x - c).abs() < 1e-9, "index {i}: got {x}, expected {c}");
            });
    }
}
