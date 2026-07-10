/*
 * // Copyright (c) Radzivon Bartoshyk 12/2025. All rights reserved.
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
#![allow(unused)]
use crate::bidirectional::{BidirectionalStore, InPlaceStore};
use crate::factory_dct2::Dct2Factory;
use crate::mla::fmla;
use crate::twiddles::compute_twiddle;
use crate::util::{DctSample, define_in_place_butterfly, try_vec, validate_scratch};
use crate::{PxdctError, PxdctExecutor};
use num_complex::Complex;
use num_traits::AsPrimitive;
use std::marker::PhantomData;
use std::sync::Arc;

#[derive(Debug, Clone, Default)]
pub(crate) struct Dst2Butterfly2<T> {
    phantom_data: PhantomData<T>,
}

impl<T: DctSample> Dst2Butterfly2<T> {
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        let u0 = data[0];
        let u1 = data[1];
        let sum = u0 - u1;
        let v0 = (u0 + u1) * T::FRAC_1_SQRT_2;
        let v1 = sum;
        data[0] = v0;
        data[1] = v1;
    }
}

define_in_place_butterfly!(Dst2Butterfly2, 2);

#[derive(Debug, Clone, Default)]
pub(crate) struct Dct2Butterfly2<T> {
    phantom_data: PhantomData<T>,
}

impl<T: DctSample> Dct2Butterfly2<T> {
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        let sum = data[0] + data[1];
        let v1 = (data[0] - data[1]) * T::FRAC_1_SQRT_2;
        let v0 = sum;
        data[0] = v0;
        data[1] = v1;
    }
}

define_in_place_butterfly!(Dct2Butterfly2, 2);

#[derive(Debug, Clone)]
pub(crate) struct Dct2Butterfly4<T: DctSample> {
    twiddle: Complex<T>,
    bf2: Dct2Butterfly2<T>,
}

impl<T: DctSample> Default for Dct2Butterfly4<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            twiddle: compute_twiddle(1, 16).conj(),
            bf2: Dct2Butterfly2::default(),
        }
    }
}

impl<T: DctSample> Dct2Butterfly4<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        let u0 = data[0];
        let u1 = data[1];
        let u2 = data[2];
        let u3 = data[3];

        let lower_dct4 = u0 - u3;
        let upper_dct4 = u2 - u1;

        let mut dct_evens = [u0 + u3, u2 + u1];

        self.bf2.exec(&mut InPlaceStore::new(&mut dct_evens));

        let v1 = fmla(lower_dct4, self.twiddle.re, -upper_dct4 * self.twiddle.im);
        let v3 = fmla(upper_dct4, self.twiddle.re, lower_dct4 * self.twiddle.im);
        data[0] = dct_evens[0];
        data[1] = v1;
        data[2] = dct_evens[1];
        data[3] = v3;
    }
}

define_in_place_butterfly!(Dct2Butterfly4, 4);

#[derive(Debug, Clone)]
pub(crate) struct Dst2Butterfly4<T: DctSample> {
    twiddle: Complex<T>,
    bf2: Dct2Butterfly2<T>,
}

impl<T: DctSample> Default for Dst2Butterfly4<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            twiddle: compute_twiddle(1, 16).conj(),
            bf2: Dct2Butterfly2::default(),
        }
    }
}

impl<T: DctSample> Dst2Butterfly4<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        let u0 = data[0];
        let u1 = data[1];
        let u2 = data[2];
        let u3 = data[3];

        let lower_dct4 = u0 + u3;
        let upper_dct4 = u2 + u1;

        let mut dct_evens = [u0 - u3, u2 - u1];

        self.bf2.exec(&mut InPlaceStore::new(&mut dct_evens));

        let v2 = fmla(lower_dct4, self.twiddle.re, -upper_dct4 * self.twiddle.im);
        let v0 = fmla(upper_dct4, self.twiddle.re, lower_dct4 * self.twiddle.im);
        data[0] = v0;
        data[1] = dct_evens[1];
        data[2] = v2;
        data[3] = dct_evens[0];
    }
}

define_in_place_butterfly!(Dst2Butterfly4, 4);

#[derive(Debug, Clone)]
pub(crate) struct Dct2Butterfly8<T: DctSample> {
    pub(crate) bf4: Dct2Butterfly4<T>,
    bf2_dct: Dct2Butterfly2<T>,
    bf2_dst: Dst2Butterfly2<T>,
    twiddle0: Complex<T>,
    twiddle1: Complex<T>,
}

impl<T: DctSample> Default for Dct2Butterfly8<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            bf4: Dct2Butterfly4::default(),
            twiddle0: compute_twiddle(1, 32).conj(),
            twiddle1: compute_twiddle(3, 32).conj(),
            bf2_dct: Dct2Butterfly2::default(),
            bf2_dst: Dst2Butterfly2::default(),
        }
    }
}

impl<T: DctSample> Dct2Butterfly8<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        let u0 = data[0];
        let u1 = data[1];
        let u2 = data[2];
        let u3 = data[3];
        let u4 = data[4];
        let u5 = data[5];
        let u6 = data[6];
        let u7 = data[7];

        let mut dct2_buffer = [u0 + u7, u1 + u6, u2 + u5, u3 + u4];
        self.bf4.exec(&mut InPlaceStore::new(&mut dct2_buffer));

        // odds
        let differences = [u0 - u7, u3 - u4, u1 - u6, u2 - u5];

        let mut dct4_even_buffer = [
            fmla(
                differences[0],
                self.twiddle0.re,
                differences[1] * self.twiddle0.im,
            ),
            fmla(
                differences[2],
                self.twiddle1.re,
                differences[3] * self.twiddle1.im,
            ),
        ];

        self.bf2_dct
            .exec(&mut InPlaceStore::new(&mut dct4_even_buffer));
        let mut dct4_odd_buffer = [
            fmla(
                differences[3],
                self.twiddle1.re,
                -differences[2] * self.twiddle1.im,
            ),
            fmla(
                differences[1],
                self.twiddle0.re,
                -differences[0] * self.twiddle0.im,
            ),
        ];
        self.bf2_dst
            .exec(&mut InPlaceStore::new(&mut dct4_odd_buffer));

        // combine the results
        data[0] = dct2_buffer[0];
        data[1] = dct4_even_buffer[0];
        data[2] = dct2_buffer[1];
        data[3] = dct4_even_buffer[1] - dct4_odd_buffer[0];
        data[4] = dct2_buffer[2];
        data[5] = dct4_even_buffer[1] + dct4_odd_buffer[0];
        data[6] = dct2_buffer[3];
        data[7] = dct4_odd_buffer[1];
    }
}

define_in_place_butterfly!(Dct2Butterfly8, 8);

#[derive(Debug, Clone)]
pub(crate) struct Dct2Butterfly16<T: DctSample> {
    bf8: Dct2Butterfly8<T>,
    bf4_dst: Dst2Butterfly4<T>,
    twiddle0: Complex<T>,
    twiddle1: Complex<T>,
    twiddle2: Complex<T>,
    twiddle3: Complex<T>,
}

impl<T: DctSample> Default for Dct2Butterfly16<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            bf8: Dct2Butterfly8::default(),
            bf4_dst: Dst2Butterfly4::default(),
            twiddle0: compute_twiddle(1, 64).conj(),
            twiddle1: compute_twiddle(3, 64).conj(),
            twiddle2: compute_twiddle(5, 64).conj(),
            twiddle3: compute_twiddle(7, 64).conj(),
        }
    }
}

impl<T: DctSample> Dct2Butterfly16<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, chunk: &mut S) {
        let u0 = chunk[0];
        let u1 = chunk[1];
        let u2 = chunk[2];
        let u3 = chunk[3];
        let u4 = chunk[4];
        let u5 = chunk[5];
        let u6 = chunk[6];
        let u7 = chunk[7];
        let u8 = chunk[8];
        let u9 = chunk[9];
        let u10 = chunk[10];
        let u11 = chunk[11];
        let u12 = chunk[12];
        let u13 = chunk[13];
        let u14 = chunk[14];
        let u15 = chunk[15];

        //process the evens
        let mut dct2_buffer = [
            u0 + u15,
            u1 + u14,
            u2 + u13,
            u3 + u12,
            u4 + u11,
            u5 + u10,
            u6 + u9,
            u7 + u8,
        ];
        self.bf8.exec(&mut InPlaceStore::new(&mut dct2_buffer));

        //process the odds
        let differences = [
            u0 - u15,
            u7 - u8,
            u1 - u14,
            u6 - u9,
            u2 - u13,
            u5 - u10,
            u3 - u12,
            u4 - u11,
        ];

        let mut dct4_even_buffer = [
            fmla(
                differences[0],
                self.twiddle0.re,
                differences[1] * self.twiddle0.im,
            ),
            fmla(
                differences[2],
                self.twiddle1.re,
                differences[3] * self.twiddle1.im,
            ),
            fmla(
                differences[4],
                self.twiddle2.re,
                differences[5] * self.twiddle2.im,
            ),
            fmla(
                differences[6],
                self.twiddle3.re,
                differences[7] * self.twiddle3.im,
            ),
        ];
        let mut dct4_odd_buffer = [
            fmla(
                differences[7],
                self.twiddle3.re,
                -differences[6] * self.twiddle3.im,
            ),
            fmla(
                differences[5],
                self.twiddle2.re,
                -differences[4] * self.twiddle2.im,
            ),
            fmla(
                differences[3],
                self.twiddle1.re,
                -differences[2] * self.twiddle1.im,
            ),
            fmla(
                differences[1],
                self.twiddle0.re,
                -differences[0] * self.twiddle0.im,
            ),
        ];

        self.bf8
            .bf4
            .exec(&mut InPlaceStore::new(&mut dct4_even_buffer));
        self.bf4_dst
            .exec(&mut InPlaceStore::new(&mut dct4_odd_buffer));

        // combine the results
        chunk[0] = dct2_buffer[0];
        chunk[1] = dct4_even_buffer[0];
        chunk[2] = dct2_buffer[1];
        chunk[3] = dct4_even_buffer[1] - dct4_odd_buffer[0];
        chunk[4] = dct2_buffer[2];
        chunk[5] = dct4_even_buffer[1] + dct4_odd_buffer[0];
        chunk[6] = dct2_buffer[3];
        chunk[7] = dct4_even_buffer[2] + dct4_odd_buffer[1];
        chunk[8] = dct2_buffer[4];
        chunk[9] = dct4_even_buffer[2] - dct4_odd_buffer[1];
        chunk[10] = dct2_buffer[5];
        chunk[11] = dct4_even_buffer[3] - dct4_odd_buffer[2];
        chunk[12] = dct2_buffer[6];
        chunk[13] = dct4_even_buffer[3] + dct4_odd_buffer[2];
        chunk[14] = dct2_buffer[7];
        chunk[15] = dct4_odd_buffer[3];
    }
}

define_in_place_butterfly!(Dct2Butterfly16, 16);

#[derive(Debug, Clone)]
pub(crate) struct Dct2Butterfly32Impl<T: DctSample, const SCALED: bool> {
    bf8: Dct2Butterfly8<T>,
    bf16: Dct2Butterfly16<T>,
    twiddles: [Complex<T>; 8],
}

impl<T: DctSample, const SCALED: bool> Default for Dct2Butterfly32Impl<T, SCALED>
where
    f64: AsPrimitive<T>,
    usize: AsPrimitive<T>,
{
    fn default() -> Self {
        use crate::twiddles::compute_twiddle;
        let mut twiddles = [Complex::<T>::default(); 8];
        for (i, twiddle) in twiddles.iter_mut().enumerate() {
            if SCALED {
                *twiddle = compute_twiddle::<T>(2 * i + 1, 32 * 4).conj() * T::QUARTER;
            } else {
                *twiddle = compute_twiddle::<T>(2 * i + 1, 32 * 4).conj();
            }
        }
        Self {
            twiddles,
            bf8: Dct2Butterfly8::default(),
            bf16: Dct2Butterfly16::default(),
        }
    }
}

impl<T: DctSample, const SCALED: bool> Dct2Butterfly32Impl<T, SCALED>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    fn exec<S: BidirectionalStore<T>>(
        &self,
        data: &mut S,
        input_dct2: &mut [T; 16],
        dct4_even: &mut [T; 8],
        dct4_odd: &mut [T; 8],
    ) {
        //preprocess the data by splitting it up into vectors of size n/2, n/4, and n/4

        for i in 0..8 {
            let twiddle = self.twiddles[i];
            let input_bottom = data[i];
            let input_top = data[32 - i - 1];

            let input_half_bottom = data[16 - i - 1];
            let input_half_top = data[16 + i];

            //prepare the inner DCT2
            if SCALED {
                unsafe {
                    *input_dct2.get_unchecked_mut(i) = (input_top + input_bottom) * T::QUARTER
                };
                unsafe {
                    *input_dct2.get_unchecked_mut(16 - i - 1) =
                        (input_half_bottom + input_half_top) * T::QUARTER
                };
            } else {
                unsafe { *input_dct2.get_unchecked_mut(i) = (input_top + input_bottom) };
                unsafe {
                    *input_dct2.get_unchecked_mut(16 - i - 1) = (input_half_bottom + input_half_top)
                };
            }

            //prepare the inner DCT4 - which consists of two DCT2s of half size
            let lower_dct4 = input_bottom - input_top;
            let upper_dct4 = input_half_bottom - input_half_top;

            let cos_input = fmla(lower_dct4, twiddle.re, upper_dct4 * twiddle.im);
            let sin_input = fmla(upper_dct4, twiddle.re, -lower_dct4 * twiddle.im);

            unsafe { *dct4_even.get_unchecked_mut(i) = cos_input };
            unsafe {
                *dct4_odd.get_unchecked_mut(8 - i - 1) =
                    if i % 2 == 0 { sin_input } else { -sin_input }
            };
        }

        self.bf16.exec(&mut InPlaceStore::new(input_dct2));
        self.bf8.exec(&mut InPlaceStore::new(dct4_even));
        self.bf8.exec(&mut InPlaceStore::new(dct4_odd));

        unsafe {
            data[0] = *input_dct2.get_unchecked(0);
            data[1] = *dct4_even.get_unchecked(0);
            data[2] = *input_dct2.get_unchecked(1);

            for i in 1..8 {
                let dct4_cos_output = *dct4_even.get_unchecked(i);
                let dct4_sin_output = if (i + 8) % 2 == 0 {
                    -*dct4_odd.get_unchecked(8 - i)
                } else {
                    *dct4_odd.get_unchecked(8 - i)
                };

                data[i * 4 - 1] = dct4_cos_output + dct4_sin_output;
                data[i * 4] = *input_dct2.get_unchecked(i * 2);

                data[i * 4 + 1] = dct4_cos_output - dct4_sin_output;
                data[i * 4 + 2] = *input_dct2.get_unchecked(i * 2 + 1);
            }

            data[32 - 1] = -*dct4_odd.get_unchecked(0);
        }
    }
}

impl<T: DctSample, const SCALED: bool> PxdctExecutor<T> for Dct2Butterfly32Impl<T, SCALED>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, data: &mut [T]) -> Result<(), PxdctError> {
        if !data.len().is_multiple_of(32) {
            return Err(PxdctError::InvalidSizeMultiplier(data.len(), self.length()));
        }

        let mut input_dct2 = [T::zero(); 16];
        let mut dct4_even = [T::zero(); 8];
        let mut dct4_odd = [T::zero(); 8];

        for chunk in data.as_chunks_mut::<32>().0.iter_mut() {
            self.exec(
                &mut InPlaceStore::new(chunk),
                &mut input_dct2,
                &mut dct4_even,
                &mut dct4_odd,
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
        validate_oof_sizes!(input, output, 32);

        let mut input_dct2 = [T::zero(); 16];
        let mut dct4_even = [T::zero(); 8];
        let mut dct4_odd = [T::zero(); 8];

        use crate::bidirectional::BiStore;
        for (src, dst) in input
            .as_chunks::<32>()
            .0
            .iter()
            .zip(output.as_chunks_mut::<32>().0.iter_mut())
        {
            self.exec(
                &mut BiStore::new(src, dst),
                &mut input_dct2,
                &mut dct4_even,
                &mut dct4_odd,
            );
        }
        Ok(())
    }

    fn length(&self) -> usize {
        32
    }

    fn scratch_size(&self) -> usize {
        32
    }
}

pub(crate) type Dct2Butterfly32<T> = Dct2Butterfly32Impl<T, false>;
pub(crate) type ScaledDct2Butterfly32<T> = Dct2Butterfly32Impl<T, true>;

#[derive(Debug, Clone)]
pub(crate) struct Dct2Butterfly64Impl<T: DctSample, const SCALED: bool> {
    bf32: Dct2Butterfly32<T>,
    bf16: Dct2Butterfly16<T>,
    twiddles: [Complex<T>; 16],
}

impl<T: DctSample, const SCALED: bool> Default for Dct2Butterfly64Impl<T, SCALED>
where
    f64: AsPrimitive<T>,
    usize: AsPrimitive<T>,
{
    fn default() -> Self {
        use crate::twiddles::compute_twiddle;
        let mut twiddles = [Complex::<T>::default(); 16];
        for (i, twiddle) in twiddles.iter_mut().enumerate() {
            if SCALED {
                *twiddle = compute_twiddle::<T>(2 * i + 1, 64 * 4).conj() * T::SQRT_2_OVER_64;
            } else {
                *twiddle = compute_twiddle::<T>(2 * i + 1, 64 * 4).conj();
            }
        }
        Self {
            twiddles,
            bf16: Dct2Butterfly16::default(),
            bf32: Dct2Butterfly32::default(),
        }
    }
}

impl<T: DctSample, const SCALED: bool> Dct2Butterfly64Impl<T, SCALED>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    fn exec<S: BidirectionalStore<T>>(
        &self,
        data: &mut S,
        input_dct2: &mut [T; 32],
        input_dct4_even: &mut [T; 16],
        input_dct4_odd: &mut [T; 16],
        input_dct21: &mut [T; 16],
        dct4_even1: &mut [T; 8],
        dct4_odd1: &mut [T; 8],
    ) {
        //preprocess the data by splitting it up into vectors of size n/2, n/4, and n/4

        for i in 0..16 {
            let twiddle = self.twiddles[i];
            let input_bottom = data[i];
            let input_top = data[64 - i - 1];

            let input_half_bottom = data[32 - i - 1];
            let input_half_top = data[32 + i];

            //prepare the inner DCT2
            if SCALED {
                unsafe {
                    *input_dct2.get_unchecked_mut(i) =
                        (input_top + input_bottom) * T::SQRT_2_OVER_64
                };
                unsafe {
                    *input_dct2.get_unchecked_mut(32 - i - 1) =
                        (input_half_bottom + input_half_top) * T::SQRT_2_OVER_64
                };
            } else {
                unsafe { *input_dct2.get_unchecked_mut(i) = (input_top + input_bottom) };
                unsafe {
                    *input_dct2.get_unchecked_mut(32 - i - 1) = (input_half_bottom + input_half_top)
                };
            }

            //prepare the inner DCT4 - which consists of two DCT2s of half size
            let lower_dct4 = input_bottom - input_top;
            let upper_dct4 = input_half_bottom - input_half_top;

            let cos_input = fmla(lower_dct4, twiddle.re, upper_dct4 * twiddle.im);
            let sin_input = fmla(upper_dct4, twiddle.re, -lower_dct4 * twiddle.im);

            unsafe { *input_dct4_even.get_unchecked_mut(i) = cos_input };
            unsafe {
                *input_dct4_odd.get_unchecked_mut(16 - i - 1) =
                    if i % 2 == 0 { sin_input } else { -sin_input }
            };
        }

        self.bf32.exec(
            &mut InPlaceStore::new(input_dct2),
            input_dct21,
            dct4_even1,
            dct4_odd1,
        );
        self.bf16.exec(&mut InPlaceStore::new(input_dct4_even));
        self.bf16.exec(&mut InPlaceStore::new(input_dct4_odd));

        unsafe {
            data[0] = *input_dct2.get_unchecked(0);
            data[1] = *input_dct4_even.get_unchecked(0);
            data[2] = *input_dct2.get_unchecked(1);

            for i in 1..16 {
                let dct4_cos_output = *input_dct4_even.get_unchecked(i);
                let dct4_sin_output = if (i + 16) % 2 == 0 {
                    -*input_dct4_odd.get_unchecked(16 - i)
                } else {
                    *input_dct4_odd.get_unchecked(16 - i)
                };

                data[i * 4 - 1] = dct4_cos_output + dct4_sin_output;
                data[i * 4] = *input_dct2.get_unchecked(i * 2);

                data[i * 4 + 1] = dct4_cos_output - dct4_sin_output;
                data[i * 4 + 2] = *input_dct2.get_unchecked(i * 2 + 1);
            }

            data[64 - 1] = -*input_dct4_odd.get_unchecked(0);
        }
    }
}

impl<T: DctSample, const SCALED: bool> PxdctExecutor<T> for Dct2Butterfly64Impl<T, SCALED>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, data: &mut [T]) -> Result<(), PxdctError> {
        if !data.len().is_multiple_of(64) {
            return Err(PxdctError::InvalidSizeMultiplier(data.len(), self.length()));
        }

        let mut input_dct2 = [T::zero(); 32];
        let mut input_dct4_even = [T::zero(); 16];
        let mut input_dct4_odd = [T::zero(); 16];

        let mut input_dct21 = [T::zero(); 16];
        let mut dct4_even = [T::zero(); 8];
        let mut dct4_odd = [T::zero(); 8];

        for chunk in data.as_chunks_mut::<64>().0.iter_mut() {
            self.exec(
                &mut InPlaceStore::new(chunk),
                &mut input_dct2,
                &mut input_dct4_even,
                &mut input_dct4_odd,
                &mut input_dct21,
                &mut dct4_even,
                &mut dct4_odd,
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
        validate_oof_sizes!(input, output, 64);

        let mut input_dct2 = [T::zero(); 32];
        let mut input_dct4_even = [T::zero(); 16];
        let mut input_dct4_odd = [T::zero(); 16];

        let mut input_dct21 = [T::zero(); 16];
        let mut dct4_even = [T::zero(); 8];
        let mut dct4_odd = [T::zero(); 8];

        use crate::bidirectional::BiStore;
        for (src, dst) in input
            .as_chunks::<64>()
            .0
            .iter()
            .zip(output.as_chunks_mut::<64>().0.iter_mut())
        {
            self.exec(
                &mut BiStore::new(src, dst),
                &mut input_dct2,
                &mut input_dct4_even,
                &mut input_dct4_odd,
                &mut input_dct21,
                &mut dct4_even,
                &mut dct4_odd,
            );
        }
        Ok(())
    }

    fn length(&self) -> usize {
        64
    }

    fn scratch_size(&self) -> usize {
        0
    }
}

pub(crate) type Dct2Butterfly64<T> = Dct2Butterfly64Impl<T, false>;
pub(crate) type ScaledDct2Butterfly64<T> = Dct2Butterfly64Impl<T, true>;

#[derive(Clone)]
pub(crate) struct Dct2Butterfly128Impl<T: DctSample, const SCALED: bool> {
    bf64: Arc<dyn PxdctExecutor<T> + Send + Sync>,
    bf32: Arc<dyn PxdctExecutor<T> + Send + Sync>,
    twiddles: [Complex<T>; 32],
}

impl<T: DctSample + Dct2Factory, const SCALED: bool> Default for Dct2Butterfly128Impl<T, SCALED>
where
    f64: AsPrimitive<T>,
    usize: AsPrimitive<T>,
{
    fn default() -> Self {
        use crate::twiddles::compute_twiddle;
        let mut twiddles = [Complex::<T>::default(); 32];
        for (i, twiddle) in twiddles.iter_mut().enumerate() {
            if SCALED {
                *twiddle = compute_twiddle::<T>(2 * i + 1, 128 * 4).conj() * T::ONE_EIGHT;
            } else {
                *twiddle = compute_twiddle::<T>(2 * i + 1, 128 * 4).conj();
            }
        }
        Self {
            twiddles,
            bf64: T::dct2_butterfly64(),
            bf32: T::dct2_butterfly32(),
        }
    }
}

impl<T: DctSample, const SCALED: bool> Dct2Butterfly128Impl<T, SCALED>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    fn exec<S: BidirectionalStore<T>>(
        &self,
        data: &mut S,
        input_dct2: &mut [T; 64],
        input_dct4: &mut [T; 64],
    ) {
        let (input_dct4_even, input_dct4_odd) = input_dct4.split_at_mut(32);

        //preprocess the data by splitting it up into vectors of size n/2, n/4, and n/4

        for i in 0..32 {
            let twiddle = self.twiddles[i];
            let input_bottom = data[i];
            let input_top = data[128 - i - 1];

            let input_half_bottom = data[64 - i - 1];
            let input_half_top = data[64 + i];

            //prepare the inner DCT2
            if SCALED {
                unsafe {
                    *input_dct2.get_unchecked_mut(i) = (input_top + input_bottom) * T::ONE_EIGHT
                };
                unsafe {
                    *input_dct2.get_unchecked_mut(64 - i - 1) =
                        (input_half_bottom + input_half_top) * T::ONE_EIGHT
                };
            } else {
                unsafe { *input_dct2.get_unchecked_mut(i) = input_top + input_bottom };
                unsafe {
                    *input_dct2.get_unchecked_mut(64 - i - 1) = input_half_bottom + input_half_top
                };
            }

            //prepare the inner DCT4 - which consists of two DCT2s of half size
            let lower_dct4 = input_bottom - input_top;
            let upper_dct4 = input_half_bottom - input_half_top;

            let cos_input = fmla(lower_dct4, twiddle.re, upper_dct4 * twiddle.im);
            let sin_input = fmla(upper_dct4, twiddle.re, -lower_dct4 * twiddle.im);

            unsafe { *input_dct4_even.get_unchecked_mut(i) = cos_input };
            unsafe {
                *input_dct4_odd.get_unchecked_mut(32 - i - 1) =
                    if i % 2 == 0 { sin_input } else { -sin_input }
            };
        }

        _ = self.bf64.execute(input_dct2);
        _ = self.bf32.execute(input_dct4);

        let (input_dct4_even, input_dct4_odd) = input_dct4.split_at_mut(32);

        unsafe {
            data[0] = *input_dct2.get_unchecked(0);
            data[1] = *input_dct4_even.get_unchecked(0);
            data[2] = *input_dct2.get_unchecked(1);

            for i in 1..32 {
                let dct4_cos_output = *input_dct4_even.get_unchecked(i);
                let dct4_sin_output = if (i + 32) % 2 == 0 {
                    -*input_dct4_odd.get_unchecked(32 - i)
                } else {
                    *input_dct4_odd.get_unchecked(32 - i)
                };

                data[i * 4 - 1] = dct4_cos_output + dct4_sin_output;
                data[i * 4] = *input_dct2.get_unchecked(i * 2);

                data[i * 4 + 1] = dct4_cos_output - dct4_sin_output;
                data[i * 4 + 2] = *input_dct2.get_unchecked(i * 2 + 1);
            }

            data[128 - 1] = -*input_dct4_odd.get_unchecked(0);
        }
    }
}

impl<T: DctSample, const SCALED: bool> PxdctExecutor<T> for Dct2Butterfly128Impl<T, SCALED>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, data: &mut [T]) -> Result<(), PxdctError> {
        self.execute_with_scratch(data, &mut [])
    }

    fn execute_with_scratch(&self, data: &mut [T], scratch: &mut [T]) -> Result<(), PxdctError> {
        if !data.len().is_multiple_of(128) {
            return Err(PxdctError::InvalidSizeMultiplier(data.len(), self.length()));
        }

        let mut input_dct2 = [T::zero(); 64];
        let mut input_dct4 = [T::zero(); 64];

        for chunk in data.as_chunks_mut::<128>().0.iter_mut() {
            self.exec(
                &mut InPlaceStore::new(chunk),
                &mut input_dct2,
                &mut input_dct4,
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
        validate_oof_sizes!(input, output, 128);

        let mut input_dct2 = [T::zero(); 64];
        let mut input_dct4 = [T::zero(); 64];

        use crate::bidirectional::BiStore;
        for (src, dst) in input
            .as_chunks::<128>()
            .0
            .iter()
            .zip(output.as_chunks_mut::<128>().0.iter_mut())
        {
            self.exec(
                &mut BiStore::new(src, dst),
                &mut input_dct2,
                &mut input_dct4,
            );
        }
        Ok(())
    }

    fn length(&self) -> usize {
        128
    }

    fn scratch_size(&self) -> usize {
        128
    }
}

pub(crate) type Dct2Butterfly128<T> = Dct2Butterfly128Impl<T, false>;
pub(crate) type ScaledDct2Butterfly128<T> = Dct2Butterfly128Impl<T, true>;

#[derive(Clone)]
pub(crate) struct Dct2Butterfly256Impl<T: DctSample, const SCALE: bool> {
    bf128: Arc<dyn PxdctExecutor<T> + Send + Sync>,
    bf64: Arc<dyn PxdctExecutor<T> + Send + Sync>,
    twiddles: [Complex<T>; 64],
}

impl<T: DctSample + Dct2Factory, const SCALE: bool> Default for Dct2Butterfly256Impl<T, SCALE>
where
    f64: AsPrimitive<T>,
    usize: AsPrimitive<T>,
{
    fn default() -> Self {
        use crate::twiddles::compute_twiddle;
        let mut twiddles = [Complex::<T>::default(); 64];
        for (i, twiddle) in twiddles.iter_mut().enumerate() {
            if SCALE {
                *twiddle = compute_twiddle::<T>(2 * i + 1, 256 * 4).conj() * T::SQRT_2_OVER_256;
            } else {
                *twiddle = compute_twiddle::<T>(2 * i + 1, 256 * 4).conj();
            }
        }
        Self {
            twiddles,
            bf64: T::dct2_butterfly64(),
            bf128: T::dct2_butterfly128(),
        }
    }
}

impl<T: DctSample, const SCALE: bool> Dct2Butterfly256Impl<T, SCALE>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    fn exec<S: BidirectionalStore<T>>(
        &self,
        data: &mut S,
        input_dct2: &mut [T; 128],
        input_dct4: &mut [T; 128],
    ) {
        let (input_dct4_even, input_dct4_odd) = input_dct4.split_at_mut(64);

        //preprocess the data by splitting it up into vectors of size n/2, n/4, and n/4

        for i in 0..64 {
            let twiddle = self.twiddles[i];
            let input_bottom = data[i];
            let input_top = data[256 - i - 1];

            let input_half_bottom = data[128 - i - 1];
            let input_half_top = data[128 + i];

            //prepare the inner DCT2
            if SCALE {
                unsafe {
                    *input_dct2.get_unchecked_mut(i) =
                        (input_top + input_bottom) * T::SQRT_2_OVER_256
                };
                unsafe {
                    *input_dct2.get_unchecked_mut(128 - i - 1) =
                        (input_half_bottom + input_half_top) * T::SQRT_2_OVER_256
                };
            } else {
                unsafe { *input_dct2.get_unchecked_mut(i) = input_top + input_bottom };
                unsafe {
                    *input_dct2.get_unchecked_mut(128 - i - 1) = input_half_bottom + input_half_top
                };
            }

            //prepare the inner DCT4 - which consists of two DCT2s of half size
            let lower_dct4 = input_bottom - input_top;
            let upper_dct4 = input_half_bottom - input_half_top;

            let cos_input = fmla(lower_dct4, twiddle.re, upper_dct4 * twiddle.im);
            let sin_input = fmla(upper_dct4, twiddle.re, -lower_dct4 * twiddle.im);

            unsafe { *input_dct4_even.get_unchecked_mut(i) = cos_input };
            unsafe {
                *input_dct4_odd.get_unchecked_mut(64 - i - 1) =
                    if i % 2 == 0 { sin_input } else { -sin_input }
            };
        }

        _ = self.bf128.execute(input_dct2);
        _ = self.bf64.execute(input_dct4);

        let (input_dct4_even, input_dct4_odd) = input_dct4.split_at_mut(64);

        unsafe {
            data[0] = *input_dct2.get_unchecked(0);
            data[1] = *input_dct4_even.get_unchecked(0);
            data[2] = *input_dct2.get_unchecked(1);

            for i in 1..64 {
                let dct4_cos_output = *input_dct4_even.get_unchecked(i);
                let dct4_sin_output = if (i + 64) % 2 == 0 {
                    -*input_dct4_odd.get_unchecked(64 - i)
                } else {
                    *input_dct4_odd.get_unchecked(64 - i)
                };

                data[i * 4 - 1] = dct4_cos_output + dct4_sin_output;
                data[i * 4] = *input_dct2.get_unchecked(i * 2);

                data[i * 4 + 1] = dct4_cos_output - dct4_sin_output;
                data[i * 4 + 2] = *input_dct2.get_unchecked(i * 2 + 1);
            }
            data[256 - 1] = -*input_dct4_odd.get_unchecked(0);
        }
    }
}

impl<T: DctSample, const SCALE: bool> PxdctExecutor<T> for Dct2Butterfly256Impl<T, SCALE>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, data: &mut [T]) -> Result<(), PxdctError> {
        self.execute_with_scratch(data, &mut [])
    }

    fn execute_with_scratch(&self, data: &mut [T], _: &mut [T]) -> Result<(), PxdctError> {
        if !data.len().is_multiple_of(256) {
            return Err(PxdctError::InvalidSizeMultiplier(data.len(), self.length()));
        }

        let mut input_dct2 = [T::zero(); 128];
        let mut input_dct4 = [T::zero(); 128];

        for chunk in data.as_chunks_mut::<256>().0.iter_mut() {
            self.exec(
                &mut InPlaceStore::new(chunk),
                &mut input_dct2,
                &mut input_dct4,
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
        validate_oof_sizes!(input, output, 256);

        let mut input_dct2 = [T::zero(); 128];
        let mut input_dct4 = [T::zero(); 128];

        use crate::bidirectional::BiStore;
        for (src, dst) in input
            .as_chunks::<256>()
            .0
            .iter()
            .zip(output.as_chunks_mut::<256>().0.iter_mut())
        {
            self.exec(
                &mut BiStore::new(src, dst),
                &mut input_dct2,
                &mut input_dct4,
            );
        }
        Ok(())
    }

    fn length(&self) -> usize {
        256
    }

    fn scratch_size(&self) -> usize {
        0
    }
}

pub(crate) type Dct2Butterfly256<T> = Dct2Butterfly256Impl<T, false>;
pub(crate) type ScaledDct2Butterfly256<T> = Dct2Butterfly256Impl<T, true>;

#[derive(Clone)]
pub(crate) struct Dct2Butterfly512Impl<T: DctSample, const SCALE: bool> {
    bf128: Arc<dyn PxdctExecutor<T> + Send + Sync>,
    bf256: Arc<dyn PxdctExecutor<T> + Send + Sync>,
    twiddles: [Complex<T>; 128],
}

impl<T: DctSample + Dct2Factory, const SCALE: bool> Default for Dct2Butterfly512Impl<T, SCALE>
where
    f64: AsPrimitive<T>,
    usize: AsPrimitive<T>,
{
    fn default() -> Self {
        use crate::twiddles::compute_twiddle;
        let mut twiddles = [Complex::<T>::default(); 128];
        for (i, twiddle) in twiddles.iter_mut().enumerate() {
            if SCALE {
                *twiddle = compute_twiddle::<T>(2 * i + 1, 512 * 4).conj() * T::SQRT_2_OVER_512;
            } else {
                *twiddle = compute_twiddle::<T>(2 * i + 1, 512 * 4).conj();
            }
        }
        Self {
            twiddles,
            bf256: T::dct2_butterfly256(),
            bf128: T::dct2_butterfly128(),
        }
    }
}

impl<T: DctSample, const SCALED: bool> Dct2Butterfly512Impl<T, SCALED>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    fn exec<S: BidirectionalStore<T>>(
        &self,
        data: &mut S,
        input_dct2: &mut [T; 256],
        input_dct4: &mut [T; 256],
    ) {
        let (input_dct4_even, input_dct4_odd) = input_dct4.split_at_mut(128);

        //preprocess the data by splitting it up into vectors of size n/2, n/4, and n/4

        for i in 0..128 {
            let twiddle = self.twiddles[i];
            let input_bottom = data[i];
            let input_top = data[512 - i - 1];

            let input_half_bottom = data[256 - i - 1];
            let input_half_top = data[256 + i];

            //prepare the inner DCT2
            if SCALED {
                unsafe {
                    *input_dct2.get_unchecked_mut(i) =
                        (input_top + input_bottom) * T::SQRT_2_OVER_512
                };
                unsafe {
                    *input_dct2.get_unchecked_mut(256 - i - 1) =
                        (input_half_bottom + input_half_top) * T::SQRT_2_OVER_512
                };
            } else {
                unsafe { *input_dct2.get_unchecked_mut(i) = input_top + input_bottom };
                unsafe {
                    *input_dct2.get_unchecked_mut(256 - i - 1) = input_half_bottom + input_half_top
                };
            }

            //prepare the inner DCT4 - which consists of two DCT2s of half size
            let lower_dct4 = input_bottom - input_top;
            let upper_dct4 = input_half_bottom - input_half_top;

            let cos_input = fmla(lower_dct4, twiddle.re, upper_dct4 * twiddle.im);
            let sin_input = fmla(upper_dct4, twiddle.re, -lower_dct4 * twiddle.im);

            unsafe { *input_dct4_even.get_unchecked_mut(i) = cos_input };
            unsafe {
                *input_dct4_odd.get_unchecked_mut(128 - i - 1) =
                    if i % 2 == 0 { sin_input } else { -sin_input }
            };
        }

        _ = self.bf256.execute(input_dct2);
        _ = self.bf128.execute(input_dct4);

        let (input_dct4_even, input_dct4_odd) = input_dct4.split_at_mut(128);

        unsafe {
            data[0] = *input_dct2.get_unchecked(0);
            data[1] = *input_dct4_even.get_unchecked(0);
            data[2] = *input_dct2.get_unchecked(1);

            for i in 1..128 {
                let dct4_cos_output = *input_dct4_even.get_unchecked(i);
                let dct4_sin_output = if (i + 128) % 2 == 0 {
                    -*input_dct4_odd.get_unchecked(128 - i)
                } else {
                    *input_dct4_odd.get_unchecked(128 - i)
                };

                data[i * 4 - 1] = dct4_cos_output + dct4_sin_output;
                data[i * 4] = *input_dct2.get_unchecked(i * 2);

                data[i * 4 + 1] = dct4_cos_output - dct4_sin_output;
                data[i * 4 + 2] = *input_dct2.get_unchecked(i * 2 + 1);
            }

            data[512 - 1] = -*input_dct4_odd.get_unchecked(0);
        }
    }
}

impl<T: DctSample, const SCALED: bool> PxdctExecutor<T> for Dct2Butterfly512Impl<T, SCALED>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, data: &mut [T]) -> Result<(), PxdctError> {
        self.execute_with_scratch(data, &mut [])
    }

    fn execute_with_scratch(&self, data: &mut [T], scratch: &mut [T]) -> Result<(), PxdctError> {
        if !data.len().is_multiple_of(512) {
            return Err(PxdctError::InvalidSizeMultiplier(data.len(), self.length()));
        }

        let mut input_dct2 = [T::zero(); 256];
        let mut input_dct4 = [T::zero(); 256];

        for chunk in data.as_chunks_mut::<512>().0.iter_mut() {
            self.exec(
                &mut InPlaceStore::new(chunk),
                &mut input_dct2,
                &mut input_dct4,
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
        validate_oof_sizes!(input, output, 512);

        let mut input_dct2 = [T::zero(); 256];
        let mut input_dct4 = [T::zero(); 256];

        use crate::bidirectional::BiStore;
        for (src, dst) in input
            .as_chunks::<512>()
            .0
            .iter()
            .zip(output.as_chunks_mut::<512>().0.iter_mut())
        {
            self.exec(
                &mut BiStore::new(src, dst),
                &mut input_dct2,
                &mut input_dct4,
            );
        }
        Ok(())
    }

    fn length(&self) -> usize {
        512
    }

    fn scratch_size(&self) -> usize {
        0
    }
}

pub(crate) type Dct2Butterfly512<T> = Dct2Butterfly512Impl<T, false>;
pub(crate) type ScaledDct2Butterfly512<T> = Dct2Butterfly512Impl<T, true>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PxdctExecutor;
    use crate::butterflies::gen_test_butterfly;
    use crate::tests::{naive_dct2, naive_dst2, naive_scaled_dct2};
    use rand::RngExt;

    gen_test_butterfly!(test_bf_dst2, f64, Dst2Butterfly2, 2, 1e-7, naive_dst2);
    gen_test_butterfly!(test_bf_dst4, f64, Dst2Butterfly4, 4, 1e-7, naive_dst2);

    gen_test_butterfly!(test_bf2, f64, Dct2Butterfly2, 2, 1e-7, naive_dct2);
    gen_test_butterfly!(test_bf4, f64, Dct2Butterfly4, 4, 1e-7, naive_dct2);
    gen_test_butterfly!(test_bf8, f64, Dct2Butterfly8, 8, 1e-7, naive_dct2);
    gen_test_butterfly!(test_bf16, f64, Dct2Butterfly16, 16, 1e-7, naive_dct2);
    gen_test_butterfly!(test_bf32, f64, Dct2Butterfly32, 32, 1e-7, naive_dct2);
    gen_test_butterfly!(test_bf64, f64, Dct2Butterfly64, 64, 1e-7, naive_dct2);
    gen_test_butterfly!(test_bf128, f64, Dct2Butterfly128, 128, 1e-7, naive_dct2);
    gen_test_butterfly!(test_bf256, f64, Dct2Butterfly256, 256, 1e-7, naive_dct2);
    gen_test_butterfly!(test_bf512, f64, Dct2Butterfly512, 512, 1e-7, naive_dct2);

    gen_test_butterfly!(
        test_scaled_bf32,
        f64,
        ScaledDct2Butterfly32,
        32,
        1e-7,
        naive_scaled_dct2
    );
    gen_test_butterfly!(
        test_scaled_bf64,
        f64,
        ScaledDct2Butterfly64,
        64,
        1e-7,
        naive_scaled_dct2
    );
    gen_test_butterfly!(
        test_scaled_bf128,
        f64,
        ScaledDct2Butterfly128,
        128,
        1e-7,
        naive_scaled_dct2
    );
    gen_test_butterfly!(
        test_scaled_bf256,
        f64,
        ScaledDct2Butterfly256,
        256,
        1e-7,
        naive_scaled_dct2
    );
    gen_test_butterfly!(
        test_scaled_bf512,
        f64,
        ScaledDct2Butterfly512,
        512,
        1e-7,
        naive_scaled_dct2
    );
}
