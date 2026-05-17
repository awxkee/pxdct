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
use crate::dct2::{MixedRadix3Sample, radixq_cos_twiddle, radixq_rotation_twiddle};
use crate::mla::fmla;
use crate::neon::util::{NeonStoreF, boring_neon_mixed_radix};
use crate::util::{DctConstants, DctSample, try_vec, validate_scratch};
use crate::{PxdctError, PxdctExecutor};
use num_traits::{AsPrimitive, One};
use std::sync::Arc;

pub(crate) fn dct2_radix_n_rotation_twiddles_neon(
    q: usize,
    q_modules: usize,
    len: usize,
) -> Vec<NeonStoreF> {
    let main_q = q;
    let inner_groups = (main_q.saturating_sub(3)) / 2 + 1;

    let working_modules = q_modules - 1;
    let main_groups = working_modules / 4;
    let has_remainder = (working_modules % 4 != 0) as usize;
    let mut twiddles = Vec::with_capacity((main_groups + has_remainder) * 2 * inner_groups);

    let mut uk = 0usize;
    while uk + 4 <= working_modules {
        let k = uk + 1;

        let mut array_re = [0.; 4];
        let mut array_im = [0.; 4];
        for m in 0..inner_groups {
            for i in 0..4 {
                let layer = radixq_rotation_twiddle(
                    main_q,
                    m,
                    (k + i).as_(),
                    (q_modules - (k + i)).as_(),
                    len,
                );
                array_re[i] = layer.re;
                array_im[i] = layer.im;
            }
            twiddles.push(NeonStoreF::load(array_re.as_ref()));
            twiddles.push(NeonStoreF::load(array_im.as_ref()));
        }

        uk += 4;
    }

    let remainder = working_modules - (working_modules / 4) * 4;
    if remainder > 0 {
        let k = uk + 1;

        let mut array_re = [0.; 4];
        let mut array_im = [0.; 4];
        for m in 0..inner_groups {
            for i in 0..remainder {
                let layer = radixq_rotation_twiddle(
                    main_q,
                    m,
                    (k + i).as_(),
                    (q_modules - (k + i)).as_(),
                    len,
                );
                array_re[i] = layer.re;
                array_im[i] = layer.im;
            }
            twiddles.push(NeonStoreF::load(array_re.as_ref()));
            twiddles.push(NeonStoreF::load(array_im.as_ref()));
        }
    }

    twiddles
}

pub(crate) fn dct2_radix_n_cos_twiddles_neon(
    q: usize,
    q_modules: usize,
    len: usize,
) -> Vec<NeonStoreF> {
    let main_q = q;
    let inner_groups = (main_q.saturating_sub(3)) / 2 + 1;

    let working_modules = q_modules - 1;
    let main_groups = working_modules / 4;
    let has_remainder = (working_modules % 4 != 0) as usize;
    let mut twiddles = Vec::with_capacity((main_groups + has_remainder) * 2 * inner_groups);

    let mut uk = 0usize;
    while uk + 4 <= working_modules {
        let k = uk + 1;

        let mut array_re = [0.; 4];
        let mut array_im = [0.; 4];

        for m in 0..inner_groups {
            for i in 0..4 {
                array_re[i] = radixq_cos_twiddle(main_q, m, (k + i).as_(), len);
                array_im[i] = radixq_cos_twiddle(main_q, m, (q_modules - (k + i)).as_(), len);
            }

            twiddles.push(NeonStoreF::load(array_re.as_ref()));
            twiddles.push(NeonStoreF::load(array_im.as_ref()));
        }

        uk += 4;
    }

    let remainder = working_modules - (working_modules / 4) * 4;
    if remainder > 0 {
        let k = uk + 1;

        let mut array_re = [0.; 4];
        let mut array_im = [0.; 4];

        for m in 0..inner_groups {
            for i in 0..remainder {
                array_re[i] = radixq_cos_twiddle(main_q, m, (k + i).as_(), len);
                array_im[i] = radixq_cos_twiddle(main_q, m, (q_modules - (k + i)).as_(), len);
            }

            twiddles.push(NeonStoreF::load(array_re.as_ref()));
            twiddles.push(NeonStoreF::load(array_im.as_ref()));
        }
    }

    twiddles
}

pub(crate) struct NeonDct2MixedRadix3f {
    rotation_layer: Vec<NeonStoreF>,
    cos_twiddles: Vec<NeonStoreF>,
    inner_dct: Arc<dyn PxdctExecutor<f32> + Send + Sync>,
    inner_dct_scratch_size: usize,
    execution_length: usize,
    q_modules: usize,
}

impl NeonDct2MixedRadix3f {
    pub(crate) fn new(
        len: usize,
        inner_dct: Arc<dyn PxdctExecutor<f32> + Send + Sync>,
    ) -> Result<NeonDct2MixedRadix3f, PxdctError> {
        assert!(
            len.is_multiple_of(3),
            "Mixed radix 3 should not be called on sizes no divisible by 3"
        );

        let q_modules = len / 3;

        // always 1 inner groups in Radix-3

        // Precompute rotation twiddles for k≥1
        // Format: [m0_k1, m1_k1, m0_k2, m1_k2, ...]
        let rotation_layer = dct2_radix_n_rotation_twiddles_neon(3, q_modules, len);

        // Precompute cosine twiddles for even components
        // Stored as Complex{re: even_twiddle, im: odd_twiddle} for cache efficiency
        let cos_twiddles = dct2_radix_n_cos_twiddles_neon(3, q_modules, len);

        let inner_dct_scratch_size = inner_dct.scratch_size();

        Ok(NeonDct2MixedRadix3f {
            rotation_layer,
            inner_dct,
            inner_dct_scratch_size,
            cos_twiddles,
            execution_length: len,
            q_modules: len / 3,
        })
    }
}

impl NeonDct2MixedRadix3f {
    #[inline(always)]
    fn exec_block<S: BidirectionalStore<f32>, const N: usize>(
        &self,
        data: &mut S,
        a_buffer: &[f32],
        s_buffer: &[f32],
        c_buffer: &[f32],
        uk: usize,
        k: usize,
    ) {
        // Apply rotation twiddles to combine forward and inverted components
        let rotation_twiddle_re = unsafe { *self.rotation_layer.get_unchecked(uk) };
        let rotation_twiddle_im = unsafe { *self.rotation_layer.get_unchecked(uk + 1) };

        let c_forward = NeonStoreF::load_n::<N>(unsafe { c_buffer.get_unchecked(k..) });
        let s_forward = NeonStoreF::load_n::<N>(unsafe {
            s_buffer.get_unchecked(self.q_modules - k - (N - 1)..)
        })
        .reverse_n::<N>();

        let rotated_dc = fmla(s_forward, rotation_twiddle_re, c_forward);

        let twiddle_re = unsafe { *self.cos_twiddles.get_unchecked(uk) };
        let twiddle_im = unsafe { *self.cos_twiddles.get_unchecked(uk + 1) };

        let twiddled_dc = rotated_dc * twiddle_re;

        let dc0 = twiddled_dc;
        let mut dc2 = -twiddled_dc * f32::HALF;

        let rotated_ds = fmla(c_forward, rotation_twiddle_im, s_forward);

        let twiddled_ds = rotated_ds * twiddle_im;

        let ds1 = twiddled_ds * f32::SIN2PI_OVER_3;

        let a0 = NeonStoreF::load_n::<N>(unsafe { a_buffer.get_unchecked(k..) });
        let dc = dc0 + a0;
        dc.write_n::<N>(data.slice_from_mut(k..));

        let dss1 = NeonStoreF::f32_mul_add(2., ds1, -dc);
        let q = dss1.reverse_n::<N>();
        q.write_n::<N>(data.slice_from_mut(self.q_modules * 2 - k - (N - 1)..));

        dc2 = -(dc2 + a0); // negated 2j
        dc2 = NeonStoreF::f32_mul_add(2., dc2, -dss1);
        dc2.write_n::<N>(data.slice_from_mut(self.q_modules * 2 + k..));
    }

    #[inline(always)]
    fn execute_with_store<S: BidirectionalStore<f32>>(
        &self,
        data: &mut S,
        scratch: &mut [f32],
    ) -> Result<(), PxdctError> {
        let (scratch, inner_scratch) = scratch.split_at_mut(self.execution_length);

        let q_modules = self.execution_length / 3;

        let (a_buffer, c_s_buffer) = scratch.split_at_mut(q_modules);
        let (c_buffer, s_buffer) = c_s_buffer.split_at_mut(q_modules);

        // Step 1: Decompose input into A (center), C (even-symmetric), S (odd-symmetric) buffers
        for (n, dst) in a_buffer.iter_mut().enumerate() {
            *dst = data[n * 3 + 1];
        }

        // Extract and combine symmetric pairs with sign alternation for S buffer
        for (m, (c_buffer, s_buffer)) in c_buffer
            .chunks_exact_mut(q_modules)
            .zip(s_buffer.chunks_exact_mut(q_modules))
            .enumerate()
        {
            let mut sign = f32::one();
            for (n, (c_dst, s_dst)) in c_buffer.iter_mut().zip(s_buffer.iter_mut()).enumerate() {
                let u0 = data[3 * n + m];
                let u1 = data[3 * n + 3 - m - 1];

                *c_dst = u0 + u1;
                *s_dst = (u0 - u1).mulsign(sign);

                sign = -sign;
            }
        }

        // Step 2: Apply DCT-II to all buffers (A, C₀, S₀)
        self.inner_dct
            .execute_with_scratch(scratch, inner_scratch)?;

        let (a_buffer, c_s_buffer) = scratch.split_at_mut(q_modules);
        let (c_buffer, s_buffer) = c_s_buffer.split_at_mut(q_modules);

        {
            // Step 3: Recombine transformed buffers with twiddle factors

            // Handle k=0 case (DC and low frequencies)
            let qc = c_buffer[0];
            let c0 = qc; // Component C₀ (position 0)
            let c1 = qc * -f32::HALF; // Component C₂ (position 2, uses j=2)

            let s0_twiddled = s_buffer[0];

            let s0 = s0_twiddled * f32::SIN2PI_OVER_3;

            // Write output: C₀ (pos 0), S₁
            let a0 = a_buffer[0];
            let dc = c0 + a0;
            data[0] = dc;
            data[q_modules] = s0;

            let qid2 = -(c1 + a0); // negated 2j
            data[q_modules * 2] = qid2;

            let mut k = 1usize;
            let mut uk = 0usize;

            // Step 4: Handle k≥1 cases with rotation twiddles
            while k + 4 <= q_modules {
                self.exec_block::<S, 4>(data, a_buffer, s_buffer, c_buffer, uk, k);
                k += 4;
                uk += 2;
            }

            let rem = q_modules - k;

            // handle remainder
            if rem == 3 {
                self.exec_block::<S, 3>(data, a_buffer, s_buffer, c_buffer, uk, k);
            } else if rem == 2 {
                self.exec_block::<S, 2>(data, a_buffer, s_buffer, c_buffer, uk, k);
            } else if rem == 1 {
                self.exec_block::<S, 1>(data, a_buffer, s_buffer, c_buffer, uk, k);
            }
        }

        Ok(())
    }
}

boring_neon_mixed_radix!(NeonDct2MixedRadix3f, f32);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Pxdct;
    use crate::tests::naive_dct2_f32;

    #[test]
    fn test_radix7_dct() {
        let mut input = vec![0.; 3 * 11];
        for (i, z) in input.iter_mut().enumerate() {
            *z = i as f32 + rand::random::<f32>() * 10.0;
        }
        let mut reference_input = input.clone();
        // let rr = Pxdct::make_dct2_f32(25).unwrap();
        // rr.execute(&mut reference_input).unwrap();
        reference_input = naive_dct2_f32(&reference_input);
        let bf =
            NeonDct2MixedRadix3f::new(input.len(), Pxdct::make_dct2_f32(input.len() / 3).unwrap())
                .unwrap();
        bf.execute(&mut input).unwrap();
        println!(
            "{:?}",
            input
                .iter()
                .enumerate()
                .map(|(i, x)| format!("({i}) {}", x))
                .collect::<Vec<_>>()
        );
        println!(
            "{:?}",
            reference_input
                .iter()
                .enumerate()
                .map(|(i, x)| format!("({i}) {}", x))
                .collect::<Vec<_>>()
        );
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
