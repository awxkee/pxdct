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
use crate::dct2::mixed_radix7::MixedRadix7Sample;
use crate::dct2::prime_butterflies::Dct2Butterfly7;
use crate::dct2::{radixq_cos_twiddle, radixq_rotation_twiddle};
use crate::mla::fmla;
use crate::util::DctSample;
use crate::{PxdctError, PxdctExecutor};
use num_complex::Complex;
use num_traits::{AsPrimitive, Zero};

pub(crate) struct Dct2Butterfly49Twiddles<T: DctSample> {
    pub(crate) rotation_layer: [Complex<T>; 18],
    pub(crate) cos_twiddles: [Complex<T>; 18],
}

impl<T: DctSample> Default for Dct2Butterfly49Twiddles<T>
where
    f64: AsPrimitive<T>,
    usize: AsPrimitive<T>,
{
    fn default() -> Self {
        let q_modules = 49 / 7;

        // always 2 inner groups in Radix-5
        let inner_groups = 3;

        let mut rotation_layer = [Complex::<T>::zero(); 18];
        for (k, rotation_layer) in rotation_layer.chunks_exact_mut(3).enumerate() {
            for (m, layer) in rotation_layer.iter_mut().enumerate() {
                *layer =
                    radixq_rotation_twiddle(7, m, (k + 1).as_(), (q_modules - (k + 1)).as_(), 49);
            }
        }

        let mut cos_twiddles = [Complex::<T>::zero(); 18];
        for (k, k_layer) in cos_twiddles.chunks_exact_mut(inner_groups).enumerate() {
            for (m, m_layer) in k_layer.iter_mut().enumerate() {
                let k = k + 1;
                let even = radixq_cos_twiddle(7, m, k.as_(), 49);
                let odd = radixq_cos_twiddle(
                    7,
                    m,
                    if k == 0 {
                        k.as_()
                    } else {
                        (q_modules - k).as_()
                    },
                    49,
                );
                *m_layer = Complex { re: even, im: odd };
            }
        }

        Dct2Butterfly49Twiddles {
            rotation_layer,
            cos_twiddles,
        }
    }
}

pub(crate) struct Dct2Butterfly49<T: DctSample> {
    rotation_layer: [Complex<T>; 18],
    cos_twiddles: [Complex<T>; 18],
    bf7: Dct2Butterfly7<T>,
}

impl<T: DctSample> Default for Dct2Butterfly49<T>
where
    f64: AsPrimitive<T>,
    usize: AsPrimitive<T>,
{
    fn default() -> Self {
        Dct2Butterfly49::new()
    }
}

impl<T: DctSample> Dct2Butterfly49<T>
where
    f64: AsPrimitive<T>,
    usize: AsPrimitive<T>,
{
    pub(crate) fn new() -> Dct2Butterfly49<T> {
        let twiddles = Dct2Butterfly49Twiddles::default();
        Dct2Butterfly49 {
            rotation_layer: twiddles.rotation_layer,
            bf7: Dct2Butterfly7::default(),
            cos_twiddles: twiddles.cos_twiddles,
        }
    }
}

impl<T: DctSample + MixedRadix7Sample> Dct2Butterfly49<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(
        &self,
        data: &mut S,
        a_buffer: &mut [T; 7],
        c_buffer: &mut [T; 21],
        s_buffer: &mut [T; 21],
    ) {
        for n in 0..7 {
            a_buffer[n] = data[n * 7 + 3];
        }

        self.bf7.exec(&mut InPlaceStore::new(a_buffer));

        let q_modules = 7;

        for m in 0..3 {
            let mut sign = T::one();
            for n in 0..7 {
                let u0 = data[7 * n + m];
                let u1 = data[7 * n + 7 - m - 1];

                c_buffer[m * 7 + n] = u0 + u1;
                s_buffer[m * 7 + n] = (u0 - u1).mulsign(sign);

                sign = -sign;
            }

            self.bf7
                .exec(&mut InPlaceStore::new(&mut c_buffer[m * 7..(m + 1) * 7]));
            self.bf7
                .exec(&mut InPlaceStore::new(&mut s_buffer[m * 7..(m + 1) * 7]));
        }

        {
            // Step 3: Recombine transformed buffers with twiddle factors

            // Handle k=0 case (DC and low frequencies)
            let qc = c_buffer[0];
            let mut c0 = qc; // Component C₀ (position 0)
            let mut c1 = qc * T::R7_COS_EVEN2_M0; // Component C₂ (position 2, uses j=2)
            let mut c2 = qc * T::R7_COS_EVEN2_M2; // Component C₄ (position 4, uses j=4)
            let mut c3 = qc * T::R7_COS_EVEN2_M1; // Component C6 (position 4, uses j=6)

            let s0_twiddled = s_buffer[0];

            let mut s0 = s0_twiddled * T::R7_SIN_ODD0_M0;
            let mut s1 = s0_twiddled * T::R7_SIN_ODD1_M0;
            let mut s2 = s0_twiddled * T::R7_SIN_ODD2_M0;

            {
                let ci = c_buffer[q_modules];
                let si = s_buffer[q_modules];

                let ci2 = c_buffer[q_modules * 2];
                let si2 = s_buffer[q_modules * 2];

                c0 = ci + c0 + ci2;

                c1 = fmla(ci, T::R7_COS_EVEN2_M1, c1);
                c1 = fmla(ci2, T::R7_COS_EVEN2_M2, c1);

                c2 = fmla(ci, T::R7_COS_EVEN2_M0, c2);
                c2 = fmla(ci2, T::R7_COS_EVEN2_M1, c2);

                c3 = fmla(ci, T::R7_COS_EVEN2_M2, c3);
                c3 = fmla(ci2, T::R7_COS_EVEN2_M0, c3);

                s0 = fmla(si, T::R7_SIN_ODD0_M1, s0);
                s0 = fmla(si2, T::R7_SIN_ODD0_M2, s0);

                s1 = fmla(si, T::R7_SIN_ODD1_M1, s1);
                s1 = fmla(si2, T::R7_SIN_ODD1_M2, s1);

                s2 = fmla(si, T::R7_SIN_ODD2_M1, s2);
                s2 = fmla(si2, T::R7_SIN_ODD2_M2, s2);
            }

            // Write output: C₀ (pos 0), S₁ (pos q_modules), C₂ (pos 2*q_modules),
            //               S₃ (pos 3*q_modules), C₄ (pos 4*q_modules)
            let a0 = a_buffer[0];
            let dc = c0 + a0;
            data[0] = dc;

            let dc2 = c2 + a0;
            data[q_modules * 4] = dc2;
            data[q_modules * 3] = -s1;
            data[q_modules] = s0;
            let qid2 = -(c1 + a0); // negated 2j
            data[q_modules * 2] = qid2;

            let dc3 = c3 + a0;
            data[q_modules * 6] = -dc3;
            data[q_modules * 5] = s2;

            // Step 4: Handle k≥1 cases with rotation twiddles
            for k in 1..7 {
                // Apply rotation twiddles to combine forward and inverted components
                let rotation_twiddle = self.rotation_layer[(k - 1) * 3];

                let c_forward = c_buffer[k];
                let s_forward = s_buffer[q_modules - k];

                let rotated_dc = fmla(s_forward, rotation_twiddle.re, c_forward);

                let twiddle = self.cos_twiddles[(k - 1) * 3];

                let twiddled_dc = rotated_dc * twiddle.re;

                let mut dc0 = twiddled_dc;
                let mut dc2 = twiddled_dc * T::R7_COS_EVEN2_M0;
                let mut dc4 = twiddled_dc * T::R7_COS_EVEN2_M2;
                let mut dc6 = twiddled_dc * T::R7_COS_EVEN2_M1;

                let rotated_ds = fmla(c_forward, rotation_twiddle.im, s_forward);

                let twiddled_ds = rotated_ds * twiddle.im;

                let mut ds1 = twiddled_ds * T::R7_SIN_ODD0_M0;
                let mut ds3 = twiddled_ds * T::R7_SIN_ODD1_M0;
                let mut ds5 = twiddled_ds * T::R7_SIN_ODD2_M0;

                {
                    let c_forward = c_buffer[q_modules + k];
                    let s_forward = s_buffer[q_modules * 2 - k];

                    let rotation_twiddle = self.rotation_layer[(k - 1) * 3 + 1];

                    let twiddle = self.cos_twiddles[(k - 1) * 3 + 1];

                    let rotated_dc1 = fmla(s_forward, rotation_twiddle.re, c_forward);
                    let rotated_ds2 = fmla(c_forward, rotation_twiddle.im, s_forward);

                    let twiddled_dc = twiddle.re * rotated_dc1;
                    let twiddled_ds = twiddle.im * rotated_ds2;

                    dc0 = twiddled_dc + dc0;
                    dc2 = fmla(twiddled_dc, T::R7_COS_EVEN2_M1, dc2);
                    dc4 = fmla(twiddled_dc, T::R7_COS_EVEN2_M0, dc4);
                    dc6 = fmla(twiddled_dc, T::R7_COS_EVEN2_M2, dc6);

                    ds1 = fmla(twiddled_ds, T::R7_SIN_ODD0_M1, ds1);
                    ds3 = fmla(twiddled_ds, T::R7_SIN_ODD1_M1, ds3);
                    ds5 = fmla(twiddled_ds, T::R7_SIN_ODD2_M1, ds5);
                }

                {
                    let c_forward = c_buffer[q_modules * 2 + k];
                    let s_forward = s_buffer[q_modules * 3 - k];

                    let rotation_twiddle = self.rotation_layer[(k - 1) * 3 + 2];

                    let twiddle = self.cos_twiddles[(k - 1) * 3 + 2];

                    let rotated_dc1 = fmla(s_forward, rotation_twiddle.re, c_forward);
                    let rotated_ds2 = fmla(c_forward, rotation_twiddle.im, s_forward);

                    let twiddled_dc = twiddle.re * rotated_dc1;
                    let twiddled_ds = twiddle.im * rotated_ds2;

                    dc0 = twiddled_dc + dc0;
                    dc2 = fmla(twiddled_dc, T::R7_COS_EVEN2_M2, dc2);
                    dc4 = fmla(twiddled_dc, T::R7_COS_EVEN2_M1, dc4);
                    dc6 = fmla(twiddled_dc, T::R7_COS_EVEN2_M0, dc6);

                    ds1 = fmla(twiddled_ds, T::R7_SIN_ODD0_M2, ds1);
                    ds3 = fmla(twiddled_ds, T::R7_SIN_ODD1_M2, ds3);
                    ds5 = fmla(twiddled_ds, T::R7_SIN_ODD2_M2, ds5);
                }

                let a0 = a_buffer[k];
                let dc = dc0 + a0;
                data[k] = dc;

                let dss1 = fmla(2f64.as_(), ds1, -dc);
                data[q_modules * 2 - k] = dss1;

                dc2 = -(dc2 + a0); // negated 2j
                dc2 = fmla(2f64.as_(), dc2, -dss1);
                data[q_modules * 2 + k] = dc2;

                let dss3 = fmla(2f64.as_(), -ds3, -dc2);
                data[q_modules * 4 - k] = dss3;

                dc4 += a0;

                let mdc4 = fmla(2f64.as_(), dc4, -dss3);
                data[q_modules * 4 + k] = mdc4;

                let dss5 = fmla(2f64.as_(), ds5, -mdc4);
                data[q_modules * 6 - k] = dss5;

                dc6 += a0;
                dc6 = fmla(2f64.as_(), -dc6, -dss5);

                data[q_modules * 6 + k] = dc6;
            }
        }
    }
}

impl<T: DctSample + MixedRadix7Sample> PxdctExecutor<T> for Dct2Butterfly49<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, data: &mut [T]) -> Result<(), PxdctError> {
        if !data.len().is_multiple_of(49) {
            return Err(PxdctError::InvalidSizeMultiplier(data.len(), self.length()));
        }

        let mut a_buffer = [T::zero(); 7];
        let mut c_buffer = [T::zero(); 21];
        let mut s_buffer = [T::zero(); 21];

        for chunk in data.chunks_exact_mut(49) {
            self.exec(
                &mut InPlaceStore::new(chunk),
                &mut a_buffer,
                &mut c_buffer,
                &mut s_buffer,
            );
        }
        Ok(())
    }

    fn execute_with_scratch(&self, data: &mut [T], _: &mut [T]) -> Result<(), PxdctError> {
        self.execute(data)
    }

    fn execute_into(&self, input: &[T], output: &mut [T]) -> Result<(), PxdctError> {
        self.execute_into_with_scratch(input, output, &mut [])
    }

    fn execute_into_with_scratch(
        &self,
        input: &[T],
        output: &mut [T],
        _: &mut [T],
    ) -> Result<(), PxdctError> {
        use crate::util::validate_oof_sizes;
        validate_oof_sizes!(input, output, 49);

        let mut a_buffer = [T::zero(); 7];
        let mut c_buffer = [T::zero(); 21];
        let mut s_buffer = [T::zero(); 21];

        use crate::bidirectional::BiStore;
        for (src, dst) in input.chunks_exact(49).zip(output.chunks_exact_mut(49)) {
            self.exec(
                &mut BiStore::new(src, dst),
                &mut a_buffer,
                &mut c_buffer,
                &mut s_buffer,
            );
        }
        Ok(())
    }

    fn length(&self) -> usize {
        49
    }

    fn scratch_size(&self) -> usize {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PxdctExecutor;
    use crate::butterflies::gen_test_butterfly;
    use crate::tests::naive_dct2;
    use rand::Rng;

    gen_test_butterfly!(test_bf49, f64, Dct2Butterfly49, 49, 1e-7, naive_dct2);
}
