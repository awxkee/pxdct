/*
 * // Copyright (c) Radzivon Bartoshyk 01/2026. All rights reserved.
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
use crate::avx::dct4::radix2f::dct4_radix2f_rotation_twiddles_avx;
use crate::avx::storef::AvxStoreF;
use crate::avx::util::{boring_avx_mixed_radix, fma};
use crate::bidirectional::BidirectionalStore;
use crate::util::{try_vec, validate_scratch};
use crate::{PxdctError, PxdctExecutor};
use num_traits::One;
use std::sync::Arc;

pub(crate) struct AvxDct4MixedRadix2f {
    dct2: Arc<dyn PxdctExecutor<f32> + Send + Sync>,
    inner_dct_scratch_size: usize,
    twiddles: Vec<AvxStoreF>,
    execution_length: usize,
}

impl AvxDct4MixedRadix2f {
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
            twiddles: unsafe { dct4_radix2f_rotation_twiddles_avx(inner_len, len) },
            execution_length: len,
            inner_dct_scratch_size: dct2_scratch_size,
        })
    }
}

boring_avx_mixed_radix!(AvxDct4MixedRadix2f, f32);

impl AvxDct4MixedRadix2f {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_store<S: BidirectionalStore<f32>>(
        &self,
        data: &mut S,
        scratch: &mut [f32],
    ) -> Result<(), PxdctError> {
        let (scratch, inner_scratch) = scratch.split_at_mut(self.execution_length);

        let len = self.length();
        let half_len = len / 2;
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
        let signs_re = AvxStoreF::set_values8(-0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0);
        let signs_im = AvxStoreF::set_values8(0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0);

        let inner_len = self.dct2.length();

        // -------- Pre-rotation / even-odd folding --------
        // Fold symmetric samples (x[i], x[N-1-i]) into two half-length sequences.
        while k + 8 <= inner_len {
            const S: usize = 8;
            let front = AvxStoreF::load(data.slice_from(k..));
            let back = AvxStoreF::load(data.slice_from(len - k - S..)).reverse();

            let twiddle_re = unsafe { *self.twiddles.get_unchecked(tk) };
            let twiddle_im = unsafe { *self.twiddles.get_unchecked(tk + 1) };

            let ll = fma(twiddle_re, front, twiddle_im * back);
            unsafe {
                ll.write(left.get_unchecked_mut(k..));
            }
            let rr = fma(
                twiddle_re.xor(signs_re),
                back,
                twiddle_im.xor(signs_im) * front,
            );
            unsafe {
                rr.reverse()
                    .write(right.get_unchecked_mut(half_len - k - S..));
            }

            tk += 2;
            k += 8;
        }

        let rem = inner_len - k;
        if rem == 7 {
            const S: usize = 7;
            let front = AvxStoreF::load7(data.slice_from(k..));
            let back = AvxStoreF::load7(data.slice_from(len - k - S..)).reverse7();

            let twiddle_re = unsafe { *self.twiddles.get_unchecked(tk) };
            let twiddle_im = unsafe { *self.twiddles.get_unchecked(tk + 1) };

            let ll = fma(twiddle_re, front, twiddle_im * back);
            unsafe {
                ll.write7(left.get_unchecked_mut(k..));
            }
            let rr = fma(
                twiddle_re.xor(signs_re),
                back,
                twiddle_im.xor(signs_im) * front,
            );
            unsafe {
                rr.reverse7()
                    .write7(right.get_unchecked_mut(half_len - k - S..));
            }
        } else if rem == 6 {
            const S: usize = 6;
            let front = AvxStoreF::load6(data.slice_from(k..));
            let back = AvxStoreF::load6(data.slice_from(len - k - S..)).reverse6();

            let twiddle_re = unsafe { *self.twiddles.get_unchecked(tk) };
            let twiddle_im = unsafe { *self.twiddles.get_unchecked(tk + 1) };

            let ll = fma(twiddle_re, front, twiddle_im * back);
            unsafe {
                ll.write6(left.get_unchecked_mut(k..));
            }
            let rr = fma(
                twiddle_re.xor(signs_re),
                back,
                twiddle_im.xor(signs_im) * front,
            );
            unsafe {
                rr.reverse6()
                    .write6(right.get_unchecked_mut(half_len - k - S..));
            }
        } else if rem == 5 {
            const S: usize = 5;
            let front = AvxStoreF::load5(data.slice_from(k..));
            let back = AvxStoreF::load5(data.slice_from(len - k - S..)).reverse5();

            let twiddle_re = unsafe { *self.twiddles.get_unchecked(tk) };
            let twiddle_im = unsafe { *self.twiddles.get_unchecked(tk + 1) };

            let ll = fma(twiddle_re, front, twiddle_im * back);
            unsafe {
                ll.write5(left.get_unchecked_mut(k..));
            }
            let rr = fma(
                twiddle_re.xor(signs_re),
                back,
                twiddle_im.xor(signs_im) * front,
            );
            unsafe {
                rr.reverse5()
                    .write5(right.get_unchecked_mut(half_len - k - S..));
            }
        } else if rem == 4 {
            const S: usize = 4;
            let front = AvxStoreF::load4(data.slice_from(k..));
            let back = AvxStoreF::load4(data.slice_from(len - k - S..)).reverse4();

            let twiddle_re = unsafe { *self.twiddles.get_unchecked(tk) };
            let twiddle_im = unsafe { *self.twiddles.get_unchecked(tk + 1) };

            let ll = fma(twiddle_re, front, twiddle_im * back);
            unsafe {
                ll.write4(left.get_unchecked_mut(k..));
            }
            let rr = fma(
                twiddle_re.xor(signs_re),
                back,
                twiddle_im.xor(signs_im) * front,
            );
            unsafe {
                rr.reverse4()
                    .write4(right.get_unchecked_mut(half_len - k - S..));
            }
        } else if rem == 3 {
            const S: usize = 3;
            let front = AvxStoreF::load3(data.slice_from(k..));
            let back = AvxStoreF::load3(data.slice_from(len - k - S..)).reverse3();

            let twiddle_re = unsafe { *self.twiddles.get_unchecked(tk) };
            let twiddle_im = unsafe { *self.twiddles.get_unchecked(tk + 1) };

            let ll = fma(twiddle_re, front, twiddle_im * back);
            unsafe {
                ll.write3(left.get_unchecked_mut(k..));
            }
            let rr = fma(
                twiddle_re.xor(signs_re),
                back,
                twiddle_im.xor(signs_im) * front,
            );
            unsafe {
                rr.reverse3()
                    .write3(right.get_unchecked_mut(half_len - k - S..));
            }
        } else if rem == 2 {
            const S: usize = 2;
            let front = AvxStoreF::load2(data.slice_from(k..));
            let back = AvxStoreF::load2(data.slice_from(len - k - S..)).reverse2();

            let twiddle_re = unsafe { *self.twiddles.get_unchecked(tk) };
            let twiddle_im = unsafe { *self.twiddles.get_unchecked(tk + 1) };

            let ll = fma(twiddle_re, front, twiddle_im * back);
            unsafe {
                ll.write2(left.get_unchecked_mut(k..));
            }
            let rr = fma(
                twiddle_re.xor(signs_re),
                back,
                twiddle_im.xor(signs_im) * front,
            );
            unsafe {
                rr.reverse2()
                    .write2(right.get_unchecked_mut(half_len - k - S..));
            }
        } else if rem == 1 {
            const S: usize = 1;
            let front = AvxStoreF::load1(data.slice_from(k..));
            let back = AvxStoreF::load1(data.slice_from(len - k - S..));

            let twiddle_re = unsafe { *self.twiddles.get_unchecked(tk) };
            let twiddle_im = unsafe { *self.twiddles.get_unchecked(tk + 1) };

            let ll = fma(twiddle_re, front, twiddle_im * back);
            unsafe {
                ll.write1(left.get_unchecked_mut(k..));
            }
            let rr = fma(
                twiddle_re.xor(signs_re),
                back,
                twiddle_im.xor(signs_im) * front,
            );
            unsafe {
                rr.write1(right.get_unchecked_mut(half_len - k - S..));
            }
        }

        self.dct2.execute_with_scratch(scratch, inner_scratch)?;

        let (left, right) = scratch.split_at_mut(half_len);

        data[0] = left[0];
        data[len - 1] = right[0];

        let signs_re = if half_len.is_multiple_of(2) {
            AvxStoreF::set_values8(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
        } else {
            AvxStoreF::set_values8(1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0)
        };
        let signs_im = if half_len.is_multiple_of(2) {
            AvxStoreF::set_values8(1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0)
        } else {
            AvxStoreF::set_values8(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
        };

        let mut i = 1usize;
        while i + 8 < half_len {
            let il = AvxStoreF::load(data.slice_from(i..));
            let rr = AvxStoreF::load(data.slice_from(half_len - i - 7..)).reverse();

            let q = i - 1;
            let u0 = fma(signs_re, rr, il);
            let u1 = fma(signs_im, rr, il);
            let interleaved = u0.zip(u1);
            interleaved[0].write(data.slice_from_mut(q * 2 + 1..));
            interleaved[1].write(data.slice_from_mut(q * 2 + 9..));

            i += 8;
        }

        let mut sign_left = if half_len.is_multiple_of(2) {
            -f32::one()
        } else {
            f32::one()
        };
        let mut sign_right = if half_len.is_multiple_of(2) {
            f32::one()
        } else {
            -f32::one()
        };

        // -------- Post-butterfly recombination --------
        // Interleave even and odd spectra back into full DCT-IV ordering.
        for i in 1..half_len {
            let il = unsafe { *left.get_unchecked(i) };
            let rr = unsafe { *right.get_unchecked(half_len - i) };

            let q = i - 1;
            data[q * 2 + 1] = fma(sign_left, rr, il);
            data[q * 2 + 2] = fma(sign_right, rr, il);

            sign_left = -sign_left;
            sign_right = -sign_right;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dct2::prime_butterflies::Dct2Butterfly17;
    use crate::tests::naive_dct4_f32;
    use rand::Rng;

    #[test]
    fn test_split_dct4() {
        let mut input = vec![0.; 34];
        for z in input.iter_mut() {
            *z = rand::rng().random_range(1.0..2.0);
        }
        let reference_input = input.clone();
        let reference_input = naive_dct4_f32(&reference_input);
        let bf = AvxDct4MixedRadix2f::new(34, Arc::new(Dct2Butterfly17::default())).unwrap();
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
