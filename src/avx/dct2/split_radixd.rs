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
use crate::avx::dct2::bf_split_radix2d::dct2_split_radix_rotation_twiddles_avx;
use crate::avx::stored::AvxStoreD;
use crate::avx::util::fma;
use crate::util::try_vec;
use crate::{PxdctError, PxdctExecutor};
use std::ops::Neg;
use std::sync::Arc;

pub(crate) struct AvxSplitRadixDct2d {
    twiddles: Vec<AvxStoreD>,
    half_dct: Arc<dyn PxdctExecutor<f64> + Send + Sync>,
    quarter_dct: Arc<dyn PxdctExecutor<f64> + Send + Sync>,
    execution_length: usize,
}

impl AvxSplitRadixDct2d {
    pub(crate) fn new(
        len: usize,
        half_dct: Arc<dyn PxdctExecutor<f64> + Send + Sync>,
        quarter_dct: Arc<dyn PxdctExecutor<f64> + Send + Sync>,
    ) -> Result<AvxSplitRadixDct2d, PxdctError> {
        assert_eq!(
            half_dct.length(),
            quarter_dct.length() * 2,
            "Invalid DCT was received, quarter size is not multiple of half"
        );
        let twiddles = unsafe { dct2_split_radix_rotation_twiddles_avx(len) };

        Ok(AvxSplitRadixDct2d {
            twiddles,
            half_dct,
            quarter_dct,
            execution_length: len,
        })
    }
}

impl PxdctExecutor<f64> for AvxSplitRadixDct2d {
    fn execute(&self, data: &mut [f64]) -> Result<(), PxdctError> {
        unsafe { self.execute_impl(data) }
    }

    #[inline]
    fn length(&self) -> usize {
        self.execution_length
    }
}

impl AvxSplitRadixDct2d {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_impl(&self, data: &mut [f64]) -> Result<(), PxdctError> {
        if !data.len().is_multiple_of(self.execution_length) {
            return Err(PxdctError::InvalidSizeMultiplier(
                data.len(),
                self.execution_length,
            ));
        }
        let mut scratch = try_vec![f64::default(); self.execution_length];

        let len = self.length();
        let half_len = len / 2;
        let quarter_len = len / 4;

        for chunk in data.chunks_exact_mut(self.execution_length) {
            //preprocess the data by splitting it up into vectors of size n/2, n/4, and n/4
            let (input_dct2, input_dct4) = scratch.split_at_mut(half_len);
            let (input_dct4_even, input_dct4_odd) = input_dct4.split_at_mut(quarter_len);

            let conj_odd = AvxStoreD::set_values(0.0, -0.0, 0.0, -0.0);

            for (i, twiddle_pack) in self.twiddles.chunks_exact(2).enumerate() {
                let twiddle_re = twiddle_pack[0];
                let twiddle_im = twiddle_pack[1];
                let input_bottom = AvxStoreD::load(unsafe { chunk.get_unchecked(i * 4..) });
                let input_top =
                    AvxStoreD::load(unsafe { chunk.get_unchecked(len - i * 4 - 4..) }).reverse();

                let input_half_bottom =
                    AvxStoreD::load(unsafe { chunk.get_unchecked(half_len - i * 4 - 4..) })
                        .reverse();
                let input_half_top =
                    AvxStoreD::load(unsafe { chunk.get_unchecked(half_len + i * 4..) });

                //prepare the inner DCT2
                unsafe {
                    let q = input_top + input_bottom;
                    q.write(input_dct2.get_unchecked_mut(i * 4..));
                };
                unsafe {
                    let dq = input_half_bottom + input_half_top;
                    dq.reverse()
                        .write(input_dct2.get_unchecked_mut(half_len - i * 4 - 4..));
                };

                //prepare the inner DCT4 - which consists of two DCT2s of half size
                let lower_dct4 = input_bottom - input_top;
                let upper_dct4 = input_half_bottom - input_half_top;

                let cos_input = fma(lower_dct4, twiddle_re, upper_dct4 * twiddle_im);
                let sin_input = fma(upper_dct4, twiddle_re, -lower_dct4 * twiddle_im);

                unsafe {
                    cos_input.write(input_dct4_even.get_unchecked_mut(i * 4..));
                    let conj_odd = sin_input.xor(conj_odd).reverse();
                    conj_odd.write(input_dct4_odd.get_unchecked_mut(quarter_len - i * 4 - 4..));
                };
            }

            self.half_dct.execute(input_dct2)?;
            self.quarter_dct.execute(input_dct4)?;

            let (input_dct4_even, input_dct4_odd) = input_dct4.split_at_mut(quarter_len);

            unsafe {
                //post process the 3 DCT2 outputs. the first few and the last will be done outside the loop
                chunk[0] = *input_dct2.get_unchecked(0);
                chunk[1] = *input_dct4_even.get_unchecked(0);
                chunk[2] = *input_dct2.get_unchecked(1);

                let mut i = 1usize;
                while i + 4 <= quarter_len {
                    let dct4_cos_output_v = AvxStoreD::load(input_dct4_even.get_unchecked(i..));
                    let dct4_sin_output_v =
                        AvxStoreD::load(input_dct4_odd.get_unchecked(quarter_len - i - 3..))
                            .reverse()
                            .xor(conj_odd);

                    let [dct4_cos_output0, dct4_cos_output1] =
                        dct4_cos_output_v.zip(dct4_cos_output_v);
                    let [mut dct4_sin_output0, mut dct4_sin_output1] =
                        dct4_sin_output_v.zip(dct4_sin_output_v);

                    dct4_sin_output0 = dct4_sin_output0.xor(conj_odd);
                    dct4_sin_output1 = dct4_sin_output1.xor(conj_odd);

                    let input0 = AvxStoreD::load(input_dct2.get_unchecked(i * 2..));
                    let input1 = AvxStoreD::load(input_dct2.get_unchecked(i * 2 + 4..));

                    let sums0 = dct4_cos_output0 + dct4_sin_output0;
                    let sums1 = dct4_cos_output1 + dct4_sin_output1;

                    let zipped0 = sums0.zip(input0);
                    let zipped1 = sums1.zip(input1);

                    zipped0[0].write(chunk.get_unchecked_mut(i * 4 - 1..));
                    zipped0[1].write(chunk.get_unchecked_mut(i * 4 + 3..));
                    zipped1[0].write(chunk.get_unchecked_mut(i * 4 + 7..));
                    zipped1[1].write(chunk.get_unchecked_mut(i * 4 + 11..));
                    i += 4;
                }

                while i + 2 <= quarter_len {
                    let dct4_cos_output_v = AvxStoreD::load2(input_dct4_even.get_unchecked(i..));
                    let dct4_sin_output_v =
                        AvxStoreD::load2(input_dct4_odd.get_unchecked(quarter_len - i - 1..))
                            .reverse2()
                            .xor(conj_odd);

                    let [dct4_cos_output0, _] = dct4_cos_output_v.zip(dct4_cos_output_v);
                    let [mut dct4_sin_output0, _] = dct4_sin_output_v.zip(dct4_sin_output_v);

                    dct4_sin_output0 = dct4_sin_output0.xor(conj_odd);

                    let input0 = AvxStoreD::load(input_dct2.get_unchecked(i * 2..));

                    let sums0 = dct4_cos_output0 + dct4_sin_output0;

                    let zipped0 = sums0.zip(input0);

                    zipped0[0].write(chunk.get_unchecked_mut(i * 4 - 1..));
                    zipped0[1].write(chunk.get_unchecked_mut(i * 4 + 3..));
                    i += 2;
                }

                for i in i..quarter_len {
                    let dct4_cos_output = *input_dct4_even.get_unchecked(i);
                    let dct4_sin_output = if (i + quarter_len) % 2 == 0 {
                        -*input_dct4_odd.get_unchecked(quarter_len - i)
                    } else {
                        *input_dct4_odd.get_unchecked(quarter_len - i)
                    };

                    *chunk.get_unchecked_mut(i * 4 - 1) = dct4_cos_output + dct4_sin_output;
                    *chunk.get_unchecked_mut(i * 4) = *input_dct2.get_unchecked(i * 2);

                    *chunk.get_unchecked_mut(i * 4 + 1) = dct4_cos_output - dct4_sin_output;
                    *chunk.get_unchecked_mut(i * 4 + 2) = *input_dct2.get_unchecked(i * 2 + 1);
                }

                *chunk.get_unchecked_mut(len - 1) = -*input_dct4_odd.get_unchecked(0);
            }
        }

        Ok(())
    }
}

pub(crate) struct AvxSplitRadixDst2d {
    split_radix_dct2: AvxSplitRadixDct2d,
    execution_length: usize,
}

impl AvxSplitRadixDst2d {
    pub(crate) fn new(
        len: usize,
        half_dct: Arc<dyn PxdctExecutor<f64> + Send + Sync>,
        quarter_dct: Arc<dyn PxdctExecutor<f64> + Send + Sync>,
    ) -> Result<AvxSplitRadixDst2d, PxdctError> {
        assert_eq!(
            half_dct.length(),
            quarter_dct.length() * 2,
            "Invalid DCT was received, quarter size is not multiple of half in DST-II"
        );

        Ok(AvxSplitRadixDst2d {
            split_radix_dct2: AvxSplitRadixDct2d::new(len, half_dct, quarter_dct)?,
            execution_length: len,
        })
    }
}

impl AvxSplitRadixDst2d {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_impl(&self, data: &mut [f64]) -> Result<(), PxdctError> {
        if !data.len().is_multiple_of(self.execution_length) {
            return Err(PxdctError::InvalidSizeMultiplier(
                data.len(),
                self.execution_length,
            ));
        }

        for chunk in data.chunks_exact_mut(self.execution_length) {
            for dst in chunk.chunks_exact_mut(2) {
                dst[1] = dst[1].neg();
            }

            self.split_radix_dct2.execute(chunk)?;

            chunk.reverse();
        }

        Ok(())
    }
}

impl PxdctExecutor<f64> for AvxSplitRadixDst2d {
    fn execute(&self, data: &mut [f64]) -> Result<(), PxdctError> {
        unsafe { self.execute_impl(data) }
    }

    #[inline]
    fn length(&self) -> usize {
        self.execution_length
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dct2::power2_butterflies::{Dct2Butterfly8, Dct2Butterfly16};
    use crate::tests::naive_dct2;
    use rand::Rng;

    #[test]
    fn test_split_dct2() {
        let mut input = vec![0.; 32];
        for z in input.iter_mut() {
            *z = rand::rng().random_range(1.0..2.0);
        }
        let reference_input = input.clone();
        let reference_input = naive_dct2(&reference_input);
        let bf = AvxSplitRadixDct2d::new(
            32,
            Arc::new(Dct2Butterfly16::default()),
            Arc::new(Dct2Butterfly8::default()),
        )
        .unwrap();
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
