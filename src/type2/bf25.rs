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
use crate::type2::mixed_radix5::MixedRadix5Sample;
use crate::type2::prime_butterflies::Dct2Butterfly5;
use crate::type2::util::{radixq_cos_twiddle, radixq_rotation_twiddle};
use crate::util::DctSample;
use crate::{PxdctError, PxdctExecutor};
use num_complex::Complex;
use num_traits::{AsPrimitive, Zero};

pub(crate) struct Dct2Butterfly25Twiddles<T: DctSample> {
    pub(crate) rotation_layer: [Complex<T>; 8],
    pub(crate) cos_twiddles: [Complex<T>; 8],
}

impl<T: DctSample> Default for Dct2Butterfly25Twiddles<T>
where
    f64: AsPrimitive<T>,
    usize: AsPrimitive<T>,
{
    fn default() -> Self {
        let q_modules = 25 / 5;

        let mut rotation_layer = [Complex::<T>::zero(); 8];
        for (k, rotation_layer) in rotation_layer.as_chunks_mut::<2>().0.iter_mut().enumerate() {
            for (m, layer) in rotation_layer.iter_mut().enumerate() {
                *layer =
                    radixq_rotation_twiddle(5, m, (k + 1).as_(), (q_modules - (k + 1)).as_(), 25);
            }
        }

        let mut cos_twiddles = [Complex::<T>::zero(); 4 * 2];
        for (k, k_layer) in cos_twiddles.as_chunks_mut::<2>().0.iter_mut().enumerate() {
            for (m, m_layer) in k_layer.iter_mut().enumerate() {
                let k = k + 1;
                let even = radixq_cos_twiddle(5, m, k.as_(), 25);
                let odd = radixq_cos_twiddle(
                    5,
                    m,
                    if k == 0 {
                        k.as_()
                    } else {
                        (q_modules - k).as_()
                    },
                    25,
                );
                *m_layer = Complex { re: even, im: odd };
            }
        }

        Dct2Butterfly25Twiddles {
            rotation_layer,
            cos_twiddles,
        }
    }
}

pub(crate) struct Dct2Butterfly25<T: DctSample> {
    rotation_layer: [Complex<T>; 8],
    cos_twiddles: [Complex<T>; 8],
    bf5: Dct2Butterfly5<T>,
}

impl<T: DctSample> Default for Dct2Butterfly25<T>
where
    f64: AsPrimitive<T>,
    usize: AsPrimitive<T>,
{
    fn default() -> Self {
        Dct2Butterfly25::new()
    }
}

impl<T: DctSample> Dct2Butterfly25<T>
where
    f64: AsPrimitive<T>,
    usize: AsPrimitive<T>,
{
    pub(crate) fn new() -> Dct2Butterfly25<T> {
        let twiddles = Dct2Butterfly25Twiddles::default();
        Dct2Butterfly25 {
            rotation_layer: twiddles.rotation_layer,
            bf5: Dct2Butterfly5::default(),
            cos_twiddles: twiddles.cos_twiddles,
        }
    }
}

impl<T: DctSample + MixedRadix5Sample> Dct2Butterfly25<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(
        &self,
        data: &mut S,
        a_buffer: &mut [T; 5],
        c_buffer: &mut [T; 10],
        s_buffer: &mut [T; 10],
    ) {
        for n in 0..5 {
            a_buffer[n] = data[n * 5 + 2];
        }

        self.bf5.exec(&mut InPlaceStore::new(a_buffer));

        for m in 0..2 {
            let mut sign = T::one();
            for n in 0..5 {
                let u0 = data[5 * n + m];
                let u1 = data[5 * n + 5 - m - 1];

                c_buffer[m * 5 + n] = u0 + u1;
                s_buffer[m * 5 + n] = (u0 - u1).mulsign(sign);

                sign = -sign;
            }

            self.bf5
                .exec(&mut InPlaceStore::new(&mut c_buffer[m * 5..(m + 1) * 5]));
            self.bf5
                .exec(&mut InPlaceStore::new(&mut s_buffer[m * 5..(m + 1) * 5]));
        }

        {
            // first blocks
            let qc = c_buffer[0];
            let mut c0 = qc;
            let mut c1 = qc * T::R5_COS_EVEN2_M0;
            let mut c2 = qc * T::R5_COS_EVEN4_M0;

            let s0_twiddled = s_buffer[0];

            let mut s0 = s0_twiddled * T::R5_SIN_ODD_M0;
            let mut s1 = s0_twiddled * T::R5_SIN_ODD1_M0;

            {
                let ci = c_buffer[5];
                let si = s_buffer[5];
                let twiddle_ci = ci;
                let twiddle_si = si;

                c0 = ci + c0;
                c1 = fmla(twiddle_ci, T::R5_COS_EVEN4_M0, c1);
                c2 = fmla(twiddle_ci, T::R5_COS_EVEN2_M0, c2);
                s0 = fmla(twiddle_si, -T::R5_SIN_ODD1_M0, s0);
                s1 = fmla(twiddle_si, T::R5_SIN_ODD_M0, s1);
            }

            let a0 = a_buffer[0];
            let dc = c0 + a0;
            data[0] = dc;

            let dc2 = c2 + a0;
            data[20] = dc2;
            data[15] = -s1;
            data[5] = s0;
            data[10] = -(c1 + a0);

            for k in 1..5 {
                let rotation_twiddle = self.rotation_layer[(k - 1) * 2];

                let c_forward = c_buffer[k];
                let s_forward = s_buffer[5 - k];

                let rotated_dc = fmla(s_forward, rotation_twiddle.re, c_forward);

                let twiddle = self.cos_twiddles[(k - 1) * 2];

                let twiddled_dc = rotated_dc * twiddle.re;

                let mut dc0 = twiddled_dc;
                let mut dc2 = twiddled_dc * T::R5_COS_EVEN2_M0;
                let mut dc4 = twiddled_dc * T::R5_COS_EVEN4_M0;

                let rotated_ds = fmla(c_forward, rotation_twiddle.im, s_forward);

                let twiddled_ds = rotated_ds * twiddle.im;

                let mut ds1 = twiddled_ds * T::R5_SIN_ODD_M0;
                let mut ds3 = twiddled_ds * T::R5_SIN_ODD1_M0;

                {
                    let c_forward = c_buffer[5 + k];
                    let s_forward = s_buffer[5 * 2 - k];

                    let rotation_twiddle = self.rotation_layer[(k - 1) * 2 + 1];
                    let twiddle = self.cos_twiddles[(k - 1) * 2 + 1];

                    let rotated_dc1 = fmla(s_forward, rotation_twiddle.re, c_forward);
                    let rotated_ds2 = fmla(c_forward, rotation_twiddle.im, s_forward);

                    let twiddled_dc = twiddle.re * rotated_dc1;
                    let twiddled_ds = twiddle.im * rotated_ds2;

                    dc0 = twiddled_dc + dc0;
                    dc2 = fmla(twiddled_dc, T::R5_COS_EVEN4_M0, dc2);
                    dc4 = fmla(twiddled_dc, T::R5_COS_EVEN2_M0, dc4);

                    ds1 = fmla(twiddled_ds, -T::R5_SIN_ODD1_M0, ds1);
                    ds3 = fmla(twiddled_ds, T::R5_SIN_ODD_M0, ds3);
                }

                let a0 = a_buffer[k];
                let dc = dc0 + a0;
                data[k] = dc;

                let dss1 = fmla(2f64.as_(), ds1, -dc);
                data[5 * 2 - k] = dss1;

                dc2 = -(dc2 + a0); // negated 2j
                dc2 = fmla(2f64.as_(), dc2, -dss1);
                data[5 * 2 + k] = dc2;

                let dss3 = fmla(2f64.as_(), -ds3, -dc2);
                data[20 - k] = dss3;

                dc4 += a0;

                data[20 + k] = fmla(2f64.as_(), dc4, -dss3);
            }
        }
    }
}

impl<T: DctSample + MixedRadix5Sample> PxdctExecutor<T> for Dct2Butterfly25<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, data: &mut [T]) -> Result<(), PxdctError> {
        if !data.len().is_multiple_of(25) {
            return Err(PxdctError::InvalidSizeMultiplier(data.len(), self.length()));
        }

        let mut a_buffer = [T::zero(); 5];
        let mut c_buffer = [T::zero(); 10];
        let mut s_buffer = [T::zero(); 10];

        for chunk in data.as_chunks_mut::<25>().0.iter_mut() {
            self.exec(
                &mut InPlaceStore::new(chunk),
                &mut a_buffer,
                &mut c_buffer,
                &mut s_buffer,
            );
        }
        Ok(())
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
        validate_oof_sizes!(input, output, 25);

        let mut a_buffer = [T::zero(); 5];
        let mut c_buffer = [T::zero(); 10];
        let mut s_buffer = [T::zero(); 10];

        use crate::bidirectional::BiStore;
        for (src, dst) in input
            .as_chunks::<25>()
            .0
            .iter()
            .zip(output.as_chunks_mut::<25>().0.iter_mut())
        {
            self.exec(
                &mut BiStore::new(src, dst),
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

    fn length(&self) -> usize {
        25
    }

    fn scratch_size(&self) -> usize {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::gen_test_butterfly;

    gen_test_butterfly!(test_bf25, f64, Dct2Butterfly25, 25, 1e-7, naive_dct2);
}
