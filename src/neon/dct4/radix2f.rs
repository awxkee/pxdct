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
use crate::bidirectional::BidirectionalStore;
use crate::mla::fmla;
use crate::neon::util::{NeonStoreF, boring_neon_mixed_radix};
use crate::twiddles::compute_twiddle;
use crate::util::{try_vec, validate_scratch};
use crate::{PxdctError, PxdctExecutor};
use num_traits::One;
use std::sync::Arc;

pub(crate) fn dct4_radix2f_rotation_twiddles_neon(inner_len: usize, len: usize) -> Vec<NeonStoreF> {
    let simd_groups = inner_len.div_ceil(4);

    let mut twiddles = Vec::with_capacity(simd_groups * 2);

    let mut uk = 0usize;
    while uk + 4 <= inner_len {
        let k = uk;

        let mut array_re = [0.; 4];
        let mut array_im = [0.; 4];
        for i in 0..4 {
            let layer = compute_twiddle(2 * (k + i) + 1, len * 8).conj();
            array_re[i] = layer.re;
            array_im[i] = layer.im;
        }

        twiddles.push(NeonStoreF::load(array_re.as_ref()));
        twiddles.push(NeonStoreF::load(array_im.as_ref()));

        uk += 4;
    }

    let remainder = inner_len - (inner_len / 4) * 4;
    if remainder > 0 {
        let k = uk;

        let mut array_re = [0.; 4];
        let mut array_im = [0.; 4];
        for i in 0..remainder {
            let layer = compute_twiddle(2 * (k + i) + 1, len * 8).conj();
            array_re[i] = layer.re;
            array_im[i] = layer.im;
        }

        twiddles.push(NeonStoreF::load(array_re.as_ref()));
        twiddles.push(NeonStoreF::load(array_im.as_ref()));
    }

    twiddles
}

pub(crate) struct NeonDct4Radix2f {
    dct2: Arc<dyn PxdctExecutor<f32> + Send + Sync>,
    inner_dct_scratch_size: usize,
    twiddles: Vec<NeonStoreF>,
    execution_length: usize,
}

impl NeonDct4Radix2f {
    pub(crate) fn new(
        len: usize,
        dct2: Arc<dyn PxdctExecutor<f32> + Send + Sync>,
    ) -> Result<Self, PxdctError> {
        assert_eq!(
            dct2.length(),
            len / 2,
            "DCT-II even length DCTs must be half of DCT-IV"
        );
        let inner_len = dct2.length();

        let dct2_scratch_size = dct2.scratch_size();

        Ok(Self {
            dct2,
            inner_dct_scratch_size: dct2_scratch_size,
            twiddles: dct4_radix2f_rotation_twiddles_neon(inner_len, len),
            execution_length: len,
        })
    }
}

impl NeonDct4Radix2f {
    #[inline(always)]
    fn execute_with_store<S: BidirectionalStore<f32>>(
        &self,
        data: &mut S,
        scratch: &mut [f32],
    ) -> Result<(), PxdctError> {
        let (scratch, inner_scratch) = scratch.split_at_mut(self.execution_length);

        let len = self.length();
        let half_len = len / 2;
        let quarter_len = len / 4;

        // This kernel implements a radix-2 DCT-IV using one inner DCT-II.
        // It follows the classical even/odd pre-rotation → DCT-II → post-butterfly scheme.
        //
        // For each length-N block:
        //
        //   DCT4(N)  =  PostRotate · ( DCT2(N/2) ⊕ DCT2(N/2) ) · PreRotate
        //
        // where the pre/post rotations are fused with alternating sign flips to minimize
        // twiddle storage and multiplications.
        let (left, right) = scratch.split_at_mut(half_len);

        let mut k = 0usize;
        let mut tk = 0usize;
        let signs_re = NeonStoreF::set_values(-0.0, 0.0, -0.0, 0.0);
        let signs_im = NeonStoreF::set_values(0.0, -0.0, 0.0, -0.0);

        let inner_len = self.dct2.length();

        // -------- Pre-rotation / even-odd folding --------
        // Fold symmetric samples (x[i], x[N-1-i]) into two half-length sequences.
        while k + 4 <= inner_len {
            const S: usize = 4;
            let front = NeonStoreF::load(data.slice_from(k..));
            let back = NeonStoreF::load(data.slice_from(len - k - S..)).reverse();

            let twiddle_re = unsafe { *self.twiddles.get_unchecked(tk) };
            let twiddle_im = unsafe { *self.twiddles.get_unchecked(tk + 1) };

            let ll = fmla(twiddle_re, front, twiddle_im * back);
            unsafe {
                ll.write(left.get_unchecked_mut(k..));
            }
            let rr = fmla(
                twiddle_re.xor(signs_re),
                back,
                twiddle_im.xor(signs_im) * front,
            );
            unsafe {
                rr.reverse()
                    .write(right.get_unchecked_mut(half_len - k - S..));
            }

            tk += 2;
            k += 4;
        }

        let rem = inner_len - k;
        if rem == 2 {
            const S: usize = 2;
            let front = NeonStoreF::load2(data.slice_from(k..));
            let back = NeonStoreF::load2(data.slice_from(len - k - S..)).reverse2();

            let twiddle_re = unsafe { *self.twiddles.get_unchecked(tk) };
            let twiddle_im = unsafe { *self.twiddles.get_unchecked(tk + 1) };

            let ll = fmla(twiddle_re, front, twiddle_im * back);
            unsafe {
                ll.write2(left.get_unchecked_mut(k..));
            }
            let rr = fmla(
                twiddle_re.xor(signs_re),
                back,
                twiddle_im.xor(signs_im) * front,
            );
            unsafe {
                rr.reverse2()
                    .write2(right.get_unchecked_mut(half_len - k - S..));
            }
        }

        self.dct2.execute_with_scratch(scratch, inner_scratch)?;

        let (left, right) = scratch.split_at_mut(half_len);

        data[0] = left[0];
        data[len - 1] = right[0];

        // -------- Post-butterfly recombination --------
        // Interleave even and odd spectra back into full DCT-IV ordering.

        let signs_re = NeonStoreF::set_values(-1.0, 1.0, -1.0, 1.0);
        let signs_im = NeonStoreF::set_values(1.0, -1.0, 1.0, -1.0);

        let mut i = 1usize;
        while i + 4 <= quarter_len {
            let il = NeonStoreF::load(unsafe { left.get_unchecked(i..) });
            let rr = NeonStoreF::load(unsafe { right.get_unchecked(half_len - i - 3..) }).reverse();
            let rl = NeonStoreF::load(unsafe { left.get_unchecked(half_len - i - 3..) }).reverse();
            let ir = NeonStoreF::load(unsafe { right.get_unchecked(i..) });

            let q = i - 1;
            let u0 = fmla(signs_re, rr, il);
            let u1 = fmla(signs_im, rr, il);
            let interleaved = u0.zip(u1);
            interleaved[0].write(data.slice_from_mut(q * 2 + 1..));
            interleaved[1].write(data.slice_from_mut(q * 2 + 5..));

            let v0 = fmla(signs_re, ir, rl);
            let v1 = fmla(signs_im, ir, rl);
            let interleaved1 = v1.zip(v0);

            interleaved1[0]
                .reverse()
                .write(data.slice_from_mut(len - q * 2 - 5..));
            interleaved1[1]
                .reverse()
                .write(data.slice_from_mut(len - q * 2 - 9..));

            i += 4;
        }

        let mut sign_left = -f32::one();
        let mut sign_right = f32::one();

        for i in i..quarter_len {
            let il = unsafe { *left.get_unchecked(i) };
            let rr = unsafe { *right.get_unchecked(half_len - i) };
            let rl = unsafe { *left.get_unchecked(half_len - i) };
            let ir = unsafe { *right.get_unchecked(i) };

            let q = i - 1;
            data[q * 2 + 1] = fmla(sign_left, rr, il);
            data[q * 2 + 2] = fmla(sign_right, rr, il);

            data[len - q * 2 - 3] = fmla(sign_left, ir, rl);
            data[len - q * 2 - 2] = fmla(sign_right, ir, rl);

            sign_left = -sign_left;
            sign_right = -sign_right;
        }

        unsafe {
            let ir = *right.get_unchecked(quarter_len);
            let il = *left.get_unchecked(quarter_len);
            data[half_len - 1] = fmla(sign_left, ir, il);
            data[half_len] = fmla(sign_right, ir, il);
        }
        Ok(())
    }
}

boring_neon_mixed_radix!(NeonDct4Radix2f, f32);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dct2::power2_butterflies::Dct2Butterfly32;
    use crate::tests::naive_dct4_f32;
    use rand::RngExt;

    #[test]
    fn test_split_dct4() {
        let mut input = vec![0.; 64];
        for z in input.iter_mut() {
            *z = rand::rng().random_range(1.0..2.0);
        }
        let reference_input = input.clone();
        let reference_input = naive_dct4_f32(&reference_input);
        let bf = NeonDct4Radix2f::new(64, Arc::new(Dct2Butterfly32::default())).unwrap();
        bf.execute(&mut input).unwrap();
        input
            .iter()
            .zip(reference_input.iter())
            .enumerate()
            .for_each(|(i, (&src, &r0))| {
                assert!(
                    (src - r0).abs() < 1e-3,
                    "Difference must be < {}, but it was {}, at position {i}",
                    1e-3,
                    (src - r0).abs()
                )
            });
    }
}
