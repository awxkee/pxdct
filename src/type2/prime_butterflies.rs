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
use crate::mla::fmla;
use crate::twiddles::compute_twiddle;
use crate::type2::MixedRadix7Sample;
use crate::util::{DctSample, define_in_place_butterfly};
use crate::{PxdctError, PxdctExecutor};
use num_complex::Complex;
use num_traits::AsPrimitive;

#[derive(Debug, Clone)]
pub(crate) struct Dct2Butterfly3<T: DctSample> {
    pub(crate) twiddle: T,
}

impl<T: DctSample> Default for Dct2Butterfly3<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            twiddle: compute_twiddle(1, 12).re,
        }
    }
}

impl<T: DctSample> Dct2Butterfly3<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        let u0 = data[0];
        let u1 = data[1];
        let u2 = data[2];

        let u0u2 = u0 + u2;

        data[0] = u0u2 + u1;
        data[1] = (u0 - u2) * self.twiddle;
        data[2] = fmla(u0u2, T::HALF, -u1);
    }
}

define_in_place_butterfly!(Dct2Butterfly3, 3);

#[derive(Debug, Clone)]
pub(crate) struct Dct2Butterfly5<T: DctSample> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T: DctSample> Default for Dct2Butterfly5<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: DctSample> Dct2Butterfly5<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        // Radix-q for size 5
        let mut a_buffer = [data[2]];
        let mut c_buffer = [data[0] + data[4], data[1] + data[3]];
        let mut s_buffer = [data[0] - data[4], data[1] - data[3]];

        let qc = c_buffer[0];
        let mut c0 = qc;
        let mut c1 = qc * T::R5_COS_EVEN2_M0;
        let mut c2 = qc * T::R5_COS_EVEN4_M0;

        let s0_twiddled = s_buffer[0];

        // Odd components: S₁ uses j=1 (abs), S₃ uses j=3 (negated)
        let mut s0 = s0_twiddled * T::R5_SIN_ODD_M0; // S₁: abs(sin(3π/5))
        let mut s1 = s0_twiddled * T::R5_SIN_ODD1_M0; // S₃: -sin(π/5)

        {
            let ci = c_buffer[1];
            let si = s_buffer[1];

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
        data[4] = dc2;
        data[3] = -s1;
        data[1] = s0;

        let qid2 = -(c1 + a0); // negated 2j
        data[2] = qid2;
    }
}

define_in_place_butterfly!(Dct2Butterfly5, 5);

#[derive(Debug, Clone)]
pub(crate) struct Dct2Butterfly7<T: DctSample> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T: DctSample> Default for Dct2Butterfly7<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: DctSample + MixedRadix7Sample> Dct2Butterfly7<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        // Radix-q where q = 7
        let a_buffer = [data[3]];
        let c_buffer = [data[0] + data[6], data[1] + data[5], data[2] + data[4]];
        let s_buffer = [data[0] - data[6], data[1] - data[5], data[2] - data[4]];

        let qc = c_buffer[0];
        let mut c0 = qc; // Component C₀ (position 0)
        let mut c1 = qc * T::R7_COS_EVEN2_M0; // Component C₂ (position 2, uses j=2)
        let mut c2 = qc * T::R7_COS_EVEN2_M2; // Component C₄ (position 4, uses j=4)
        let mut c3 = qc * T::R7_COS_EVEN2_M1; // Component C6 (position 6, uses j=6)

        let s0_twiddled = s_buffer[0];

        let mut s0 = s0_twiddled * T::R7_SIN_ODD0_M0;
        let mut s1 = s0_twiddled * T::R7_SIN_ODD1_M0;
        let mut s2 = s0_twiddled * T::R7_SIN_ODD2_M0;

        let ci = c_buffer[1];
        let si = s_buffer[1];

        let ci2 = c_buffer[2];
        let si2 = s_buffer[2];

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

        // Write output: C₀ (pos 0), S₁ (pos q_modules)
        let a0 = a_buffer[0];
        let dc = c0 + a0;
        data[0] = dc;

        let dc2 = c2 + a0;
        data[4] = dc2;
        data[3] = -s1;
        data[1] = s0;
        let qid2 = -(c1 + a0); // negated 2j
        data[2] = qid2;

        let dc3 = c3 + a0;
        data[6] = -dc3;
        data[5] = s2;
    }
}

define_in_place_butterfly!(Dct2Butterfly7, 7);

pub(crate) trait MixedRadix11Sample {
    const R11_EVEN_TWIDDLE_0: Self;
    const R11_EVEN_TWIDDLE_1: Self;
    const R11_EVEN_TWIDDLE_2: Self;
    const R11_EVEN_TWIDDLE_3: Self;
    const R11_EVEN_TWIDDLE_4: Self;
    const R11_ODD_TWIDDLE_0: Self;
    const R11_ODD_TWIDDLE_1: Self;
    const R11_ODD_TWIDDLE_2: Self;
    const R11_ODD_TWIDDLE_3: Self;
    const R11_ODD_TWIDDLE_4: Self;
}

impl MixedRadix11Sample for f32 {
    const R11_EVEN_TWIDDLE_0: f32 = f32::from_bits(0xbf75a155);
    const R11_EVEN_TWIDDLE_1: f32 = f32::from_bits(0xbf27a4f4);
    const R11_EVEN_TWIDDLE_2: f32 = f32::from_bits(0xbe11bafb);
    const R11_EVEN_TWIDDLE_3: f32 = f32::from_bits(0x3ed4b147);
    const R11_EVEN_TWIDDLE_4: f32 = f32::from_bits(0x3f575c64);
    const R11_ODD_TWIDDLE_0: f32 = f32::from_bits(0x3f7d64f0);
    const R11_ODD_TWIDDLE_1: f32 = f32::from_bits(0x3f68dda4);
    const R11_ODD_TWIDDLE_2: f32 = f32::from_bits(0x3f4178ce);
    const R11_ODD_TWIDDLE_3: f32 = f32::from_bits(0x3f0a6770);
    const R11_ODD_TWIDDLE_4: f32 = f32::from_bits(0x3e903f40);
}

impl MixedRadix11Sample for f64 {
    const R11_EVEN_TWIDDLE_0: f64 = f64::from_bits(0xbfeeb42a9bcd5057);
    const R11_EVEN_TWIDDLE_1: f64 = f64::from_bits(0xbfe4f49e7f775887);
    const R11_EVEN_TWIDDLE_2: f64 = f64::from_bits(0xbfc2375f640f44d6);
    const R11_EVEN_TWIDDLE_3: f64 = f64::from_bits(0x3fda9628d9c712b5);
    const R11_EVEN_TWIDDLE_4: f64 = f64::from_bits(0x3feaeb8c8764f0ba);
    const R11_ODD_TWIDDLE_0: f64 = f64::from_bits(0x3fefac9e043842ef);
    const R11_ODD_TWIDDLE_1: f64 = f64::from_bits(0x3fed1bb48eee2c13);
    const R11_ODD_TWIDDLE_2: f64 = f64::from_bits(0x3fe82f19bb3a28a1);
    const R11_ODD_TWIDDLE_3: f64 = f64::from_bits(0x3fe14cedf8bb580b);
    const R11_ODD_TWIDDLE_4: f64 = f64::from_bits(0x3fd207e7fd768dc0);
}

#[derive(Debug, Clone)]
pub(crate) struct Dct2Butterfly11<T: DctSample> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T: DctSample> Default for Dct2Butterfly11<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: DctSample + MixedRadix11Sample> Dct2Butterfly11<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        // This is autogenerated Radix-Q where Q = 11.
        let a_buffer = [data[5]];
        let c_buffer = [
            data[0] + data[10],
            data[1] + data[9],
            data[2] + data[8],
            data[3] + data[7],
            data[4] + data[6],
        ];
        let s_buffer = [
            data[0] - data[10],
            data[1] - data[9],
            data[2] - data[8],
            data[3] - data[7],
            data[4] - data[6],
        ];
        let qc = c_buffer[0];
        let mut c0 = qc;
        let mut c1 = qc * T::R11_EVEN_TWIDDLE_0;
        let mut s0 = s_buffer[0];
        s0 *= T::R11_ODD_TWIDDLE_0;
        let ci1 = c_buffer[1];
        let si1 = s_buffer[1];
        c0 = ci1 + c0;
        c1 = fmla(ci1, T::R11_EVEN_TWIDDLE_1, c1);
        s0 = fmla(si1, T::R11_ODD_TWIDDLE_1, s0);
        let ci2 = c_buffer[2];
        let si2 = s_buffer[2];
        c0 = ci2 + c0;
        c1 = fmla(ci2, T::R11_EVEN_TWIDDLE_2, c1);
        s0 = fmla(si2, T::R11_ODD_TWIDDLE_2, s0);
        let ci3 = c_buffer[3];
        let si3 = s_buffer[3];
        c0 = ci3 + c0;
        c1 = fmla(ci3, T::R11_EVEN_TWIDDLE_3, c1);
        s0 = fmla(si3, T::R11_ODD_TWIDDLE_3, s0);
        let ci4 = c_buffer[4];
        let si4 = s_buffer[4];
        c0 = ci4 + c0;
        c1 = fmla(ci4, T::R11_EVEN_TWIDDLE_4, c1);
        s0 = fmla(si4, T::R11_ODD_TWIDDLE_4, s0);
        let a0 = a_buffer[0];
        let dc = c0 + a0;
        data[0] = dc;
        data[1] = s0;
        data[2] = -(c1 + a0);
        let mut c0 = qc * T::R11_EVEN_TWIDDLE_4;
        let mut s0 = s_buffer[0];
        s0 *= -T::R11_ODD_TWIDDLE_1;
        c0 = fmla(ci1, T::R11_EVEN_TWIDDLE_2, c0);
        s0 = fmla(si1, -T::R11_ODD_TWIDDLE_4, s0);
        c0 = fmla(ci2, T::R11_EVEN_TWIDDLE_0, c0);
        s0 = fmla(si2, T::R11_ODD_TWIDDLE_3, s0);
        c0 = fmla(ci3, T::R11_EVEN_TWIDDLE_1, c0);
        s0 = fmla(si3, T::R11_ODD_TWIDDLE_0, s0);
        c0 = fmla(ci4, T::R11_EVEN_TWIDDLE_3, c0);
        s0 = fmla(si4, T::R11_ODD_TWIDDLE_2, s0);
        let dc = c0 + a0;
        data[4] = dc;
        data[3] = -s0;
        let mut c0 = qc * T::R11_EVEN_TWIDDLE_1;
        let mut s0 = s_buffer[0];
        s0 *= T::R11_ODD_TWIDDLE_2;
        c0 = fmla(ci1, T::R11_EVEN_TWIDDLE_4, c0);
        s0 = fmla(si1, -T::R11_ODD_TWIDDLE_3, s0);
        c0 = fmla(ci2, T::R11_EVEN_TWIDDLE_3, c0);
        s0 = fmla(si2, -T::R11_ODD_TWIDDLE_1, s0);
        c0 = fmla(ci3, T::R11_EVEN_TWIDDLE_0, c0);
        s0 = fmla(si3, T::R11_ODD_TWIDDLE_4, s0);
        c0 = fmla(ci4, T::R11_EVEN_TWIDDLE_2, c0);
        s0 = fmla(si4, T::R11_ODD_TWIDDLE_0, s0);
        let dc = c0 + a0;
        data[6] = -dc;
        data[5] = s0;
        let mut c0 = qc * T::R11_EVEN_TWIDDLE_3;
        let mut s0 = s_buffer[0];
        s0 *= -T::R11_ODD_TWIDDLE_3;
        c0 = fmla(ci1, T::R11_EVEN_TWIDDLE_0, c0);
        s0 = fmla(si1, T::R11_ODD_TWIDDLE_0, s0);
        c0 = fmla(ci2, T::R11_EVEN_TWIDDLE_4, c0);
        s0 = fmla(si2, -T::R11_ODD_TWIDDLE_4, s0);
        c0 = fmla(ci3, T::R11_EVEN_TWIDDLE_2, c0);
        s0 = fmla(si3, -T::R11_ODD_TWIDDLE_2, s0);
        c0 = fmla(ci4, T::R11_EVEN_TWIDDLE_1, c0);
        s0 = fmla(si4, T::R11_ODD_TWIDDLE_1, s0);
        let dc = c0 + a0;
        data[8] = dc;
        data[7] = -s0;
        let mut c0 = qc * T::R11_EVEN_TWIDDLE_2;
        let mut s0 = s_buffer[0];
        s0 *= T::R11_ODD_TWIDDLE_4;
        c0 = fmla(ci1, T::R11_EVEN_TWIDDLE_3, c0);
        s0 = fmla(si1, -T::R11_ODD_TWIDDLE_2, s0);
        c0 = fmla(ci2, T::R11_EVEN_TWIDDLE_1, c0);
        s0 = fmla(si2, T::R11_ODD_TWIDDLE_0, s0);
        c0 = fmla(ci3, T::R11_EVEN_TWIDDLE_4, c0);
        s0 = fmla(si3, -T::R11_ODD_TWIDDLE_1, s0);
        c0 = fmla(ci4, T::R11_EVEN_TWIDDLE_0, c0);
        s0 = fmla(si4, T::R11_ODD_TWIDDLE_3, s0);
        let dc = c0 + a0;
        data[10] = -dc;
        data[9] = s0;
    }
}

impl<T: DctSample + MixedRadix11Sample> PxdctExecutor<T> for Dct2Butterfly11<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, data: &mut [T]) -> Result<(), PxdctError> {
        if !data.len().is_multiple_of(11) {
            return Err(PxdctError::InvalidSizeMultiplier(data.len(), self.length()));
        }
        for chunk in data.chunks_exact_mut(11) {
            self.exec(&mut InPlaceStore::new(chunk));
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
        validate_oof_sizes!(input, output, 11);
        use crate::bidirectional::BiStore;
        for (src, dst) in input.chunks_exact(11).zip(output.chunks_exact_mut(11)) {
            self.exec(&mut BiStore::new(src, dst));
        }
        Ok(())
    }

    fn length(&self) -> usize {
        11
    }

    fn scratch_size(&self) -> usize {
        0
    }
}

pub(crate) trait MixedRadix13Sample {
    const R13_EVEN_TWIDDLE_0: Self;
    const R13_EVEN_TWIDDLE_1: Self;
    const R13_EVEN_TWIDDLE_2: Self;
    const R13_EVEN_TWIDDLE_3: Self;
    const R13_EVEN_TWIDDLE_4: Self;
    const R13_EVEN_TWIDDLE_5: Self;
    const R13_ODD_TWIDDLE_0: Self;
    const R13_ODD_TWIDDLE_1: Self;
    const R13_ODD_TWIDDLE_2: Self;
    const R13_ODD_TWIDDLE_3: Self;
    const R13_ODD_TWIDDLE_4: Self;
    const R13_ODD_TWIDDLE_5: Self;
}

impl MixedRadix13Sample for f32 {
    const R13_EVEN_TWIDDLE_0: f32 = f32::from_bits(0xbf788fa5);
    const R13_EVEN_TWIDDLE_1: f32 = f32::from_bits(0xbf3f9e67);
    const R13_EVEN_TWIDDLE_2: f32 = f32::from_bits(0xbeb58ec6);
    const R13_EVEN_TWIDDLE_3: f32 = f32::from_bits(0x3df6dbef);
    const R13_EVEN_TWIDDLE_4: f32 = f32::from_bits(0x3f116cb1);
    const R13_EVEN_TWIDDLE_5: f32 = f32::from_bits(0x3f62ad3f);
    const R13_ODD_TWIDDLE_0: f32 = f32::from_bits(0x3f7e222b);
    const R13_ODD_TWIDDLE_1: f32 = f32::from_bits(0x3f6f5d39);
    const R13_ODD_TWIDDLE_2: f32 = f32::from_bits(0x3f52af12);
    const R13_ODD_TWIDDLE_3: f32 = f32::from_bits(0x3f29c268);
    const R13_ODD_TWIDDLE_4: f32 = f32::from_bits(0x3eedf032);
    const R13_ODD_TWIDDLE_5: f32 = f32::from_bits(0x3e750f2a);
}

impl MixedRadix13Sample for f64 {
    const R13_EVEN_TWIDDLE_0: f64 = f64::from_bits(0xbfef11f493053d00);
    const R13_EVEN_TWIDDLE_1: f64 = f64::from_bits(0xbfe7f3ccd0032e0d);
    const R13_EVEN_TWIDDLE_2: f64 = f64::from_bits(0xbfd6b1d8b2365da3);
    const R13_EVEN_TWIDDLE_3: f64 = f64::from_bits(0x3fbedb7debaa3ed3);
    const R13_EVEN_TWIDDLE_4: f64 = f64::from_bits(0x3fe22d961ea71119);
    const R13_EVEN_TWIDDLE_5: f64 = f64::from_bits(0x3fec55a7e00740e9);
    const R13_ODD_TWIDDLE_0: f64 = f64::from_bits(0x3fefc44566966769);
    const R13_ODD_TWIDDLE_1: f64 = f64::from_bits(0x3fedeba72ef20147);
    const R13_ODD_TWIDDLE_2: f64 = f64::from_bits(0x3fea55e242a4c3d3);
    const R13_ODD_TWIDDLE_3: f64 = f64::from_bits(0x3fe5384d024c2f84);
    const R13_ODD_TWIDDLE_4: f64 = f64::from_bits(0x3fddbe064267c47c);
    const R13_ODD_TWIDDLE_5: f64 = f64::from_bits(0x3fcea1e54bc48dbf);
}

#[derive(Debug, Clone)]
pub(crate) struct Dct2Butterfly13<T: DctSample> {
    twiddle0: Complex<T>,
    twiddle1: Complex<T>,
    twiddle2: Complex<T>,
    twiddle3: Complex<T>,
    twiddle4: Complex<T>,
    twiddle5: Complex<T>,
}

impl<T: DctSample> Default for Dct2Butterfly13<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            twiddle0: compute_twiddle(1, 52).conj(),
            twiddle1: compute_twiddle(2, 52).conj(),
            twiddle2: compute_twiddle(3, 52).conj(),
            twiddle3: compute_twiddle(4, 52).conj(),
            twiddle4: compute_twiddle(5, 52).conj(),
            twiddle5: compute_twiddle(6, 52).conj(),
        }
    }
}

impl<T: DctSample> Dct2Butterfly13<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        // This is autogenerated Radix-Q where Q = 13.
        let a_buffer = [data[6]];
        let c_buffer = [
            data[0] + data[12],
            data[1] + data[11],
            data[2] + data[10],
            data[3] + data[9],
            data[4] + data[8],
            data[5] + data[7],
        ];
        let s_buffer = [
            data[0] - data[12],
            data[1] - data[11],
            data[2] - data[10],
            data[3] - data[9],
            data[4] - data[8],
            data[5] - data[7],
        ];
        let qc = c_buffer[0];
        let mut c0 = qc;
        let mut c1 = qc * T::R13_EVEN_TWIDDLE_0;
        let mut s0 = s_buffer[0];
        s0 *= T::R13_ODD_TWIDDLE_0;
        let ci1 = c_buffer[1];
        let si1 = s_buffer[1];
        c0 = ci1 + c0;
        c1 = fmla(ci1, T::R13_EVEN_TWIDDLE_1, c1);
        s0 = fmla(si1, T::R13_ODD_TWIDDLE_1, s0);
        let ci2 = c_buffer[2];
        let si2 = s_buffer[2];
        c0 = ci2 + c0;
        c1 = fmla(ci2, T::R13_EVEN_TWIDDLE_2, c1);
        s0 = fmla(si2, T::R13_ODD_TWIDDLE_2, s0);
        let ci3 = c_buffer[3];
        let si3 = s_buffer[3];
        c0 = ci3 + c0;
        c1 = fmla(ci3, T::R13_EVEN_TWIDDLE_3, c1);
        s0 = fmla(si3, T::R13_ODD_TWIDDLE_3, s0);
        let ci4 = c_buffer[4];
        let si4 = s_buffer[4];
        c0 = ci4 + c0;
        c1 = fmla(ci4, T::R13_EVEN_TWIDDLE_4, c1);
        s0 = fmla(si4, T::R13_ODD_TWIDDLE_4, s0);
        let ci5 = c_buffer[5];
        let si5 = s_buffer[5];
        c0 = ci5 + c0;
        c1 = fmla(ci5, T::R13_EVEN_TWIDDLE_5, c1);
        s0 = fmla(si5, T::R13_ODD_TWIDDLE_5, s0);
        let a0 = a_buffer[0];
        let dc = c0 + a0;
        data[0] = dc;
        data[1] = s0;
        data[2] = -(c1 + a0);
        let mut c0 = qc * T::R13_EVEN_TWIDDLE_5;
        let mut s0 = s_buffer[0];
        s0 *= -T::R13_ODD_TWIDDLE_1;
        c0 = fmla(ci1, T::R13_EVEN_TWIDDLE_3, c0);
        s0 = fmla(si1, -T::R13_ODD_TWIDDLE_4, s0);
        c0 = fmla(ci2, T::R13_EVEN_TWIDDLE_1, c0);
        s0 = fmla(si2, T::R13_ODD_TWIDDLE_5, s0);
        c0 = fmla(ci3, T::R13_EVEN_TWIDDLE_0, c0);
        s0 = fmla(si3, T::R13_ODD_TWIDDLE_2, s0);
        c0 = fmla(ci4, T::R13_EVEN_TWIDDLE_2, c0);
        s0 = fmla(si4, T::R13_ODD_TWIDDLE_0, s0);
        c0 = fmla(ci5, T::R13_EVEN_TWIDDLE_4, c0);
        s0 = fmla(si5, T::R13_ODD_TWIDDLE_3, s0);
        let dc = c0 + a0;
        data[4] = dc;
        data[3] = -s0;
        let mut c0 = qc * T::R13_EVEN_TWIDDLE_1;
        let mut s0 = s_buffer[0];
        s0 *= T::R13_ODD_TWIDDLE_2;
        c0 = fmla(ci1, T::R13_EVEN_TWIDDLE_4, c0);
        s0 = fmla(si1, -T::R13_ODD_TWIDDLE_5, s0);
        c0 = fmla(ci2, T::R13_EVEN_TWIDDLE_5, c0);
        s0 = fmla(si2, -T::R13_ODD_TWIDDLE_0, s0);
        c0 = fmla(ci3, T::R13_EVEN_TWIDDLE_2, c0);
        s0 = fmla(si3, -T::R13_ODD_TWIDDLE_4, s0);
        c0 = fmla(ci4, T::R13_EVEN_TWIDDLE_0, c0);
        s0 = fmla(si4, T::R13_ODD_TWIDDLE_3, s0);
        c0 = fmla(ci5, T::R13_EVEN_TWIDDLE_3, c0);
        s0 = fmla(si5, T::R13_ODD_TWIDDLE_1, s0);
        let dc = c0 + a0;
        data[6] = -dc;
        data[5] = s0;
        let mut c0 = qc * T::R13_EVEN_TWIDDLE_4;
        let mut s0 = s_buffer[0];
        s0 *= -T::R13_ODD_TWIDDLE_3;
        c0 = fmla(ci1, T::R13_EVEN_TWIDDLE_0, c0);
        s0 = fmla(si1, T::R13_ODD_TWIDDLE_2, s0);
        c0 = fmla(ci2, T::R13_EVEN_TWIDDLE_3, c0);
        s0 = fmla(si2, T::R13_ODD_TWIDDLE_4, s0);
        c0 = fmla(ci3, T::R13_EVEN_TWIDDLE_5, c0);
        s0 = fmla(si3, -T::R13_ODD_TWIDDLE_1, s0);
        c0 = fmla(ci4, T::R13_EVEN_TWIDDLE_1, c0);
        s0 = fmla(si4, -T::R13_ODD_TWIDDLE_5, s0);
        c0 = fmla(ci5, T::R13_EVEN_TWIDDLE_2, c0);
        s0 = fmla(si5, T::R13_ODD_TWIDDLE_0, s0);
        let dc = c0 + a0;
        data[8] = dc;
        data[7] = -s0;
        let mut c0 = qc * T::R13_EVEN_TWIDDLE_2;
        let mut s0 = s_buffer[0];
        s0 *= T::R13_ODD_TWIDDLE_4;
        c0 = fmla(ci1, T::R13_EVEN_TWIDDLE_5, c0);
        s0 = fmla(si1, -T::R13_ODD_TWIDDLE_0, s0);
        c0 = fmla(ci2, T::R13_EVEN_TWIDDLE_0, c0);
        s0 = fmla(si2, T::R13_ODD_TWIDDLE_3, s0);
        c0 = fmla(ci3, T::R13_EVEN_TWIDDLE_4, c0);
        s0 = fmla(si3, T::R13_ODD_TWIDDLE_5, s0);
        c0 = fmla(ci4, T::R13_EVEN_TWIDDLE_3, c0);
        s0 = fmla(si4, -T::R13_ODD_TWIDDLE_1, s0);
        c0 = fmla(ci5, T::R13_EVEN_TWIDDLE_1, c0);
        s0 = fmla(si5, T::R13_ODD_TWIDDLE_2, s0);
        let dc = c0 + a0;
        data[10] = -dc;
        data[9] = s0;
        let mut c0 = qc * T::R13_EVEN_TWIDDLE_3;
        let mut s0 = s_buffer[0];
        s0 *= -T::R13_ODD_TWIDDLE_5;
        c0 = fmla(ci1, T::R13_EVEN_TWIDDLE_2, c0);
        s0 = fmla(si1, T::R13_ODD_TWIDDLE_3, s0);
        c0 = fmla(ci2, T::R13_EVEN_TWIDDLE_4, c0);
        s0 = fmla(si2, -T::R13_ODD_TWIDDLE_1, s0);
        c0 = fmla(ci3, T::R13_EVEN_TWIDDLE_1, c0);
        s0 = fmla(si3, T::R13_ODD_TWIDDLE_0, s0);
        c0 = fmla(ci4, T::R13_EVEN_TWIDDLE_5, c0);
        s0 = fmla(si4, -T::R13_ODD_TWIDDLE_2, s0);
        c0 = fmla(ci5, T::R13_EVEN_TWIDDLE_0, c0);
        s0 = fmla(si5, T::R13_ODD_TWIDDLE_4, s0);
        let dc = c0 + a0;
        data[12] = dc;
        data[11] = -s0;
    }
}

define_in_place_butterfly!(Dct2Butterfly13, 13);

pub(crate) trait MixedRadix17Sample {
    const R17_EVEN_TWIDDLE_0: Self;
    const R17_EVEN_TWIDDLE_1: Self;
    const R17_EVEN_TWIDDLE_2: Self;
    const R17_EVEN_TWIDDLE_3: Self;
    const R17_EVEN_TWIDDLE_4: Self;
    const R17_EVEN_TWIDDLE_5: Self;
    const R17_EVEN_TWIDDLE_6: Self;
    const R17_EVEN_TWIDDLE_7: Self;
    const R17_ODD_TWIDDLE_0: Self;
    const R17_ODD_TWIDDLE_1: Self;
    const R17_ODD_TWIDDLE_2: Self;
    const R17_ODD_TWIDDLE_3: Self;
    const R17_ODD_TWIDDLE_4: Self;
    const R17_ODD_TWIDDLE_5: Self;
    const R17_ODD_TWIDDLE_6: Self;
    const R17_ODD_TWIDDLE_7: Self;
}

impl MixedRadix17Sample for f32 {
    const R17_EVEN_TWIDDLE_0: f32 = f32::from_bits(0xbf7ba420);
    const R17_EVEN_TWIDDLE_1: f32 = f32::from_bits(0xbf59a7d5);
    const R17_EVEN_TWIDDLE_2: f32 = f32::from_bits(0xbf1a4643);
    const R17_EVEN_TWIDDLE_3: f32 = f32::from_bits(0xbe8c1d8e);
    const R17_EVEN_TWIDDLE_4: f32 = f32::from_bits(0x3dbcf732);
    const R17_EVEN_TWIDDLE_5: f32 = f32::from_bits(0x3ee437d1);
    const R17_EVEN_TWIDDLE_6: f32 = f32::from_bits(0x3f3d2fb0);
    const R17_EVEN_TWIDDLE_7: f32 = f32::from_bits(0x3f6eb680);
    const R17_ODD_TWIDDLE_0: f32 = f32::from_bits(0x3f7ee86f);
    const R17_ODD_TWIDDLE_1: f32 = f32::from_bits(0x3f763a35);
    const R17_ODD_TWIDDLE_2: f32 = f32::from_bits(0x3f65296c);
    const R17_ODD_TWIDDLE_3: f32 = f32::from_bits(0x3f4c4adb);
    const R17_ODD_TWIDDLE_4: f32 = f32::from_bits(0x3f2c7751);
    const R17_ODD_TWIDDLE_5: f32 = f32::from_bits(0x3f06c442);
    const R17_ODD_TWIDDLE_6: f32 = f32::from_bits(0x3eb8f4ab);
    const R17_ODD_TWIDDLE_7: f32 = f32::from_bits(0x3e3c28d5);
}

impl MixedRadix17Sample for f64 {
    const R17_EVEN_TWIDDLE_0: f64 = f64::from_bits(0xbfef7484007faef3);
    const R17_EVEN_TWIDDLE_1: f64 = f64::from_bits(0xbfeb34fa910ea3b9);
    const R17_EVEN_TWIDDLE_2: f64 = f64::from_bits(0xbfe348c86ed5f1bc);
    const R17_EVEN_TWIDDLE_3: f64 = f64::from_bits(0xbfd183b1c61f0d02);
    const R17_EVEN_TWIDDLE_4: f64 = f64::from_bits(0x3fb79ee63259b75f);
    const R17_EVEN_TWIDDLE_5: f64 = f64::from_bits(0x3fdc86fa2b2883cc);
    const R17_EVEN_TWIDDLE_6: f64 = f64::from_bits(0x3fe7a5f6075d4884);
    const R17_EVEN_TWIDDLE_7: f64 = f64::from_bits(0x3fedd6d000370991);
    const R17_ODD_TWIDDLE_0: f64 = f64::from_bits(0x3fefdd0deb564b22);
    const R17_ODD_TWIDDLE_1: f64 = f64::from_bits(0x3feec746923c349f);
    const R17_ODD_TWIDDLE_2: f64 = f64::from_bits(0x3feca52d7c9e640b);
    const R17_ODD_TWIDDLE_3: f64 = f64::from_bits(0x3fe9895b6c9a05f6);
    const R17_ODD_TWIDDLE_4: f64 = f64::from_bits(0x3fe58eea2a9d6da3);
    const R17_ODD_TWIDDLE_5: f64 = f64::from_bits(0x3fe0d8884363dd80);
    const R17_ODD_TWIDDLE_6: f64 = f64::from_bits(0x3fd71e955d8e7cdc);
    const R17_ODD_TWIDDLE_7: f64 = f64::from_bits(0x3fc7851aacd6c6b4);
}

#[derive(Debug, Clone)]
pub(crate) struct Dct2Butterfly17<T: DctSample> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T: DctSample> Default for Dct2Butterfly17<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: DctSample> Dct2Butterfly17<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        // This is autogenerated Radix-Q where Q = 17.
        let a_buffer = [data[8]];
        let c_buffer = [
            data[0] + data[16],
            data[1] + data[15],
            data[2] + data[14],
            data[3] + data[13],
            data[4] + data[12],
            data[5] + data[11],
            data[6] + data[10],
            data[7] + data[9],
        ];
        let s_buffer = [
            data[0] - data[16],
            data[1] - data[15],
            data[2] - data[14],
            data[3] - data[13],
            data[4] - data[12],
            data[5] - data[11],
            data[6] - data[10],
            data[7] - data[9],
        ];
        let qc = c_buffer[0];
        let mut c0 = qc;
        let mut c1 = qc * T::R17_EVEN_TWIDDLE_0;
        let mut s0 = s_buffer[0];
        s0 *= T::R17_ODD_TWIDDLE_0;
        let ci1 = c_buffer[1];
        let si1 = s_buffer[1];
        c0 = ci1 + c0;
        c1 = ci1 * T::R17_EVEN_TWIDDLE_1 + c1;
        s0 = fmla(si1, T::R17_ODD_TWIDDLE_1, s0);
        let ci2 = c_buffer[2];
        let si2 = s_buffer[2];
        c0 = ci2 + c0;
        c1 = ci2 * T::R17_EVEN_TWIDDLE_2 + c1;
        s0 = fmla(si2, T::R17_ODD_TWIDDLE_2, s0);
        let ci3 = c_buffer[3];
        let si3 = s_buffer[3];
        c0 = ci3 + c0;
        c1 = ci3 * T::R17_EVEN_TWIDDLE_3 + c1;
        s0 = fmla(si3, T::R17_ODD_TWIDDLE_3, s0);
        let ci4 = c_buffer[4];
        let si4 = s_buffer[4];
        c0 = ci4 + c0;
        c1 = ci4 * T::R17_EVEN_TWIDDLE_4 + c1;
        s0 = fmla(si4, T::R17_ODD_TWIDDLE_4, s0);
        let ci5 = c_buffer[5];
        let si5 = s_buffer[5];
        c0 = ci5 + c0;
        c1 = ci5 * T::R17_EVEN_TWIDDLE_5 + c1;
        s0 = fmla(si5, T::R17_ODD_TWIDDLE_5, s0);
        let ci6 = c_buffer[6];
        let si6 = s_buffer[6];
        c0 = ci6 + c0;
        c1 = ci6 * T::R17_EVEN_TWIDDLE_6 + c1;
        s0 = fmla(si6, T::R17_ODD_TWIDDLE_6, s0);
        let ci7 = c_buffer[7];
        let si7 = s_buffer[7];
        c0 = ci7 + c0;
        c1 = ci7 * T::R17_EVEN_TWIDDLE_7 + c1;
        s0 = fmla(si7, T::R17_ODD_TWIDDLE_7, s0);
        let a0 = a_buffer[0];
        let dc = c0 + a0;
        data[0] = dc;
        data[1] = s0;
        data[2] = -(c1 + a0);
        let mut c0 = qc * T::R17_EVEN_TWIDDLE_7;
        let mut s0 = s_buffer[0];
        s0 *= -T::R17_ODD_TWIDDLE_1;
        c0 = fmla(ci1, T::R17_EVEN_TWIDDLE_5, c0);
        s0 = fmla(si1, -T::R17_ODD_TWIDDLE_4, s0);
        c0 = fmla(ci2, T::R17_EVEN_TWIDDLE_3, c0);
        s0 = fmla(si2, -T::R17_ODD_TWIDDLE_7, s0);
        c0 = fmla(ci3, T::R17_EVEN_TWIDDLE_1, c0);
        s0 = fmla(si3, T::R17_ODD_TWIDDLE_6, s0);
        c0 = fmla(ci4, T::R17_EVEN_TWIDDLE_0, c0);
        s0 = fmla(si4, T::R17_ODD_TWIDDLE_3, s0);
        c0 = fmla(ci5, T::R17_EVEN_TWIDDLE_2, c0);
        s0 = fmla(si5, T::R17_ODD_TWIDDLE_0, s0);
        c0 = fmla(ci6, T::R17_EVEN_TWIDDLE_4, c0);
        s0 = fmla(si6, T::R17_ODD_TWIDDLE_2, s0);
        c0 = fmla(ci7, T::R17_EVEN_TWIDDLE_6, c0);
        s0 = fmla(si7, T::R17_ODD_TWIDDLE_5, s0);
        let dc = c0 + a0;
        data[4] = dc;
        data[3] = -s0;
        let mut c0 = qc * T::R17_EVEN_TWIDDLE_1;
        let mut s0 = s_buffer[0];
        s0 *= T::R17_ODD_TWIDDLE_2;
        c0 = fmla(ci1, T::R17_EVEN_TWIDDLE_4, c0);
        s0 = fmla(si1, T::R17_ODD_TWIDDLE_7, s0);
        c0 = fmla(ci2, T::R17_EVEN_TWIDDLE_7, c0);
        s0 = fmla(si2, -T::R17_ODD_TWIDDLE_4, s0);
        c0 = fmla(ci3, T::R17_EVEN_TWIDDLE_6, c0);
        s0 = fmla(si3, -T::R17_ODD_TWIDDLE_0, s0);
        c0 = fmla(ci4, T::R17_EVEN_TWIDDLE_3, c0);
        s0 = fmla(si4, -T::R17_ODD_TWIDDLE_5, s0);
        c0 = fmla(ci5, T::R17_EVEN_TWIDDLE_0, c0);
        s0 = fmla(si5, T::R17_ODD_TWIDDLE_6, s0);
        c0 = fmla(ci6, T::R17_EVEN_TWIDDLE_2, c0);
        s0 = fmla(si6, T::R17_ODD_TWIDDLE_1, s0);
        c0 = fmla(ci7, T::R17_EVEN_TWIDDLE_5, c0);
        s0 = fmla(si7, T::R17_ODD_TWIDDLE_3, s0);
        let dc = c0 + a0;
        data[6] = -dc;
        data[5] = s0;
        let mut c0 = qc * T::R17_EVEN_TWIDDLE_6;
        let mut s0 = s_buffer[0];
        s0 *= -T::R17_ODD_TWIDDLE_3;
        c0 = fmla(ci1, T::R17_EVEN_TWIDDLE_2, c0);
        s0 = fmla(si1, T::R17_ODD_TWIDDLE_6, s0);
        c0 = fmla(ci2, T::R17_EVEN_TWIDDLE_1, c0);
        s0 = fmla(si2, T::R17_ODD_TWIDDLE_0, s0);
        c0 = fmla(ci3, T::R17_EVEN_TWIDDLE_5, c0);
        s0 = fmla(si3, T::R17_ODD_TWIDDLE_7, s0);
        c0 = fmla(ci4, T::R17_EVEN_TWIDDLE_7, c0);
        s0 = fmla(si4, -T::R17_ODD_TWIDDLE_2, s0);
        c0 = fmla(ci5, T::R17_EVEN_TWIDDLE_3, c0);
        s0 = fmla(si5, -T::R17_ODD_TWIDDLE_4, s0);
        c0 = fmla(ci6, T::R17_EVEN_TWIDDLE_0, c0);
        s0 = fmla(si6, T::R17_ODD_TWIDDLE_5, s0);
        c0 = fmla(ci7, T::R17_EVEN_TWIDDLE_4, c0);
        s0 = fmla(si7, T::R17_ODD_TWIDDLE_1, s0);
        let dc = c0 + a0;
        data[8] = dc;
        data[7] = -s0;
        let mut c0 = qc * T::R17_EVEN_TWIDDLE_2;
        let mut s0 = s_buffer[0];
        s0 *= T::R17_ODD_TWIDDLE_4;
        c0 = fmla(ci1, T::R17_EVEN_TWIDDLE_7, c0);
        s0 = fmla(si1, -T::R17_ODD_TWIDDLE_3, s0);
        c0 = fmla(ci2, T::R17_EVEN_TWIDDLE_4, c0);
        s0 = fmla(si2, -T::R17_ODD_TWIDDLE_5, s0);
        c0 = fmla(ci3, T::R17_EVEN_TWIDDLE_0, c0);
        s0 = fmla(si3, T::R17_ODD_TWIDDLE_2, s0);
        c0 = fmla(ci4, T::R17_EVEN_TWIDDLE_5, c0);
        s0 = fmla(si4, T::R17_ODD_TWIDDLE_6, s0);
        c0 = fmla(ci5, T::R17_EVEN_TWIDDLE_6, c0);
        s0 = fmla(si5, -T::R17_ODD_TWIDDLE_1, s0);
        c0 = fmla(ci6, T::R17_EVEN_TWIDDLE_1, c0);
        s0 = fmla(si6, -T::R17_ODD_TWIDDLE_7, s0);
        c0 = fmla(ci7, T::R17_EVEN_TWIDDLE_3, c0);
        s0 = fmla(si7, T::R17_ODD_TWIDDLE_0, s0);
        let dc = c0 + a0;
        data[10] = -dc;
        data[9] = s0;
        let mut c0 = qc * T::R17_EVEN_TWIDDLE_5;
        let mut s0 = s_buffer[0];
        s0 *= -T::R17_ODD_TWIDDLE_5;
        c0 = fmla(ci1, T::R17_EVEN_TWIDDLE_0, c0);
        s0 = fmla(si1, T::R17_ODD_TWIDDLE_0, s0);
        c0 = fmla(ci2, T::R17_EVEN_TWIDDLE_6, c0);
        s0 = fmla(si2, -T::R17_ODD_TWIDDLE_6, s0);
        c0 = fmla(ci3, T::R17_EVEN_TWIDDLE_4, c0);
        s0 = fmla(si3, -T::R17_ODD_TWIDDLE_4, s0);
        c0 = fmla(ci4, T::R17_EVEN_TWIDDLE_1, c0);
        s0 = fmla(si4, T::R17_ODD_TWIDDLE_1, s0);
        c0 = fmla(ci5, T::R17_EVEN_TWIDDLE_7, c0);
        s0 = fmla(si5, -T::R17_ODD_TWIDDLE_7, s0);
        c0 = fmla(ci6, T::R17_EVEN_TWIDDLE_3, c0);
        s0 = fmla(si6, -T::R17_ODD_TWIDDLE_3, s0);
        c0 = fmla(ci7, T::R17_EVEN_TWIDDLE_2, c0);
        s0 = fmla(si7, T::R17_ODD_TWIDDLE_2, s0);
        let dc = c0 + a0;
        data[12] = dc;
        data[11] = -s0;
        let mut c0 = qc * T::R17_EVEN_TWIDDLE_3;
        let mut s0 = s_buffer[0];
        s0 *= T::R17_ODD_TWIDDLE_6;
        c0 = fmla(ci1, T::R17_EVEN_TWIDDLE_6, c0);
        s0 = fmla(si1, -T::R17_ODD_TWIDDLE_2, s0);
        c0 = fmla(ci2, T::R17_EVEN_TWIDDLE_0, c0);
        s0 = fmla(si2, T::R17_ODD_TWIDDLE_1, s0);
        c0 = fmla(ci3, T::R17_EVEN_TWIDDLE_7, c0);
        s0 = fmla(si3, -T::R17_ODD_TWIDDLE_5, s0);
        c0 = fmla(ci4, T::R17_EVEN_TWIDDLE_2, c0);
        s0 = fmla(si4, -T::R17_ODD_TWIDDLE_7, s0);
        c0 = fmla(ci5, T::R17_EVEN_TWIDDLE_4, c0);
        s0 = fmla(si5, T::R17_ODD_TWIDDLE_3, s0);
        c0 = fmla(ci6, T::R17_EVEN_TWIDDLE_5, c0);
        s0 = fmla(si6, -T::R17_ODD_TWIDDLE_0, s0);
        c0 = fmla(ci7, T::R17_EVEN_TWIDDLE_1, c0);
        s0 = fmla(si7, T::R17_ODD_TWIDDLE_4, s0);
        let dc = c0 + a0;
        data[14] = -dc;
        data[13] = s0;
        let mut c0 = qc * T::R17_EVEN_TWIDDLE_4;
        let mut s0 = s_buffer[0];
        s0 *= -T::R17_ODD_TWIDDLE_7;
        c0 = fmla(ci1, T::R17_EVEN_TWIDDLE_3, c0);
        s0 = fmla(si1, T::R17_ODD_TWIDDLE_5, s0);
        c0 = fmla(ci2, T::R17_EVEN_TWIDDLE_5, c0);
        s0 = fmla(si2, -T::R17_ODD_TWIDDLE_3, s0);
        c0 = fmla(ci3, T::R17_EVEN_TWIDDLE_2, c0);
        s0 = fmla(si3, T::R17_ODD_TWIDDLE_1, s0);
        c0 = fmla(ci4, T::R17_EVEN_TWIDDLE_6, c0);
        s0 = fmla(si4, -T::R17_ODD_TWIDDLE_0, s0);
        c0 = fmla(ci5, T::R17_EVEN_TWIDDLE_1, c0);
        s0 = fmla(si5, T::R17_ODD_TWIDDLE_2, s0);
        c0 = fmla(ci6, T::R17_EVEN_TWIDDLE_7, c0);
        s0 = fmla(si6, -T::R17_ODD_TWIDDLE_4, s0);
        c0 = fmla(ci7, T::R17_EVEN_TWIDDLE_0, c0);
        s0 = fmla(si7, T::R17_ODD_TWIDDLE_6, s0);
        let dc = c0 + a0;
        data[16] = dc;
        data[15] = -s0;
    }
}

define_in_place_butterfly!(Dct2Butterfly17, 17);

pub(crate) trait MixedRadix19Sample {
    const R19_EVEN_TWIDDLE_0: Self;
    const R19_EVEN_TWIDDLE_1: Self;
    const R19_EVEN_TWIDDLE_2: Self;
    const R19_EVEN_TWIDDLE_3: Self;
    const R19_EVEN_TWIDDLE_4: Self;
    const R19_EVEN_TWIDDLE_5: Self;
    const R19_EVEN_TWIDDLE_6: Self;
    const R19_EVEN_TWIDDLE_7: Self;
    const R19_EVEN_TWIDDLE_8: Self;
    const R19_ODD_TWIDDLE_0: Self;
    const R19_ODD_TWIDDLE_1: Self;
    const R19_ODD_TWIDDLE_2: Self;
    const R19_ODD_TWIDDLE_3: Self;
    const R19_ODD_TWIDDLE_4: Self;
    const R19_ODD_TWIDDLE_5: Self;
    const R19_ODD_TWIDDLE_6: Self;
    const R19_ODD_TWIDDLE_7: Self;
    const R19_ODD_TWIDDLE_8: Self;
}

impl MixedRadix19Sample for f32 {
    const R19_EVEN_TWIDDLE_0: f32 = f32::from_bits(0xbf7c822d);
    const R19_EVEN_TWIDDLE_1: f32 = f32::from_bits(0xbf612531);
    const R19_EVEN_TWIDDLE_2: f32 = f32::from_bits(0xbf2d6253);
    const R19_EVEN_TWIDDLE_3: f32 = f32::from_bits(0xbecdab06);
    const R19_EVEN_TWIDDLE_4: f32 = f32::from_bits(0xbda91f5c);
    const R19_EVEN_TWIDDLE_5: f32 = f32::from_bits(0x3e7b608c);
    const R19_EVEN_TWIDDLE_6: f32 = f32::from_bits(0x3f0c04cb);
    const R19_EVEN_TWIDDLE_7: f32 = f32::from_bits(0x3f4a051d);
    const R19_EVEN_TWIDDLE_8: f32 = f32::from_bits(0x3f722114);
    const R19_ODD_TWIDDLE_0: f32 = f32::from_bits(0x3f7f2029);
    const R19_ODD_TWIDDLE_1: f32 = f32::from_bits(0x3f782a9e);
    const R19_ODD_TWIDDLE_2: f32 = f32::from_bits(0x3f6a701f);
    const R19_ODD_TWIDDLE_3: f32 = f32::from_bits(0x3f56508b);
    const R19_ODD_TWIDDLE_4: f32 = f32::from_bits(0x3f3c5867);
    const R19_ODD_TWIDDLE_5: f32 = f32::from_bits(0x3f1d3d0b);
    const R19_ODD_TWIDDLE_6: f32 = f32::from_bits(0x3ef3af60);
    const R19_ODD_TWIDDLE_7: f32 = f32::from_bits(0x3ea63f02);
    const R19_ODD_TWIDDLE_8: f32 = f32::from_bits(0x3e288b7c);
}

impl MixedRadix19Sample for f64 {
    const R19_EVEN_TWIDDLE_0: f64 = f64::from_bits(0xbfef90459484f2b2);
    const R19_EVEN_TWIDDLE_1: f64 = f64::from_bits(0xbfec24a622e3e9f8);
    const R19_EVEN_TWIDDLE_2: f64 = f64::from_bits(0xbfe5ac4a670a1cfe);
    const R19_EVEN_TWIDDLE_3: f64 = f64::from_bits(0xbfd9b560b9f596e8);
    const R19_EVEN_TWIDDLE_4: f64 = f64::from_bits(0xbfb523eb8420f5ee);
    const R19_EVEN_TWIDDLE_5: f64 = f64::from_bits(0x3fcf6c118574c840);
    const R19_EVEN_TWIDDLE_6: f64 = f64::from_bits(0x3fe180996c77c8ca);
    const R19_EVEN_TWIDDLE_7: f64 = f64::from_bits(0x3fe940a398f9cd23);
    const R19_EVEN_TWIDDLE_8: f64 = f64::from_bits(0x3fee442285231be1);
    const R19_ODD_TWIDDLE_0: f64 = f64::from_bits(0x3fefe40529a542aa);
    const R19_ODD_TWIDDLE_1: f64 = f64::from_bits(0x3fef0553b4de2e18);
    const R19_ODD_TWIDDLE_2: f64 = f64::from_bits(0x3fed4e03dd110b08);
    const R19_ODD_TWIDDLE_3: f64 = f64::from_bits(0x3feaca115aae3de4);
    const R19_ODD_TWIDDLE_4: f64 = f64::from_bits(0x3fe78b0cdee73e0f);
    const R19_ODD_TWIDDLE_5: f64 = f64::from_bits(0x3fe3a7a16b394423);
    const R19_ODD_TWIDDLE_6: f64 = f64::from_bits(0x3fde75ec0ded7bed);
    const R19_ODD_TWIDDLE_7: f64 = f64::from_bits(0x3fd4c7e04850cfa9);
    const R19_ODD_TWIDDLE_8: f64 = f64::from_bits(0x3fc5116f7f2d58c5);
}

#[derive(Debug, Clone)]
pub(crate) struct Dct2Butterfly19<T: DctSample> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T: DctSample> Default for Dct2Butterfly19<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: DctSample> Dct2Butterfly19<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        // This is autogenerated Radix-Q where Q = 19.
        let a_buffer = [data[9]];
        let c_buffer = [
            data[0] + data[18],
            data[1] + data[17],
            data[2] + data[16],
            data[3] + data[15],
            data[4] + data[14],
            data[5] + data[13],
            data[6] + data[12],
            data[7] + data[11],
            data[8] + data[10],
        ];
        let s_buffer = [
            data[0] - data[18],
            data[1] - data[17],
            data[2] - data[16],
            data[3] - data[15],
            data[4] - data[14],
            data[5] - data[13],
            data[6] - data[12],
            data[7] - data[11],
            data[8] - data[10],
        ];
        let qc = c_buffer[0];
        let mut c0 = qc;
        let mut c1 = qc * T::R19_EVEN_TWIDDLE_0;
        let mut s0 = s_buffer[0];
        s0 *= T::R19_ODD_TWIDDLE_0;
        let ci1 = c_buffer[1];
        let si1 = s_buffer[1];
        c0 = ci1 + c0;
        c1 = ci1 * T::R19_EVEN_TWIDDLE_1 + c1;
        s0 = fmla(si1, T::R19_ODD_TWIDDLE_1, s0);
        let ci2 = c_buffer[2];
        let si2 = s_buffer[2];
        c0 = ci2 + c0;
        c1 = ci2 * T::R19_EVEN_TWIDDLE_2 + c1;
        s0 = fmla(si2, T::R19_ODD_TWIDDLE_2, s0);
        let ci3 = c_buffer[3];
        let si3 = s_buffer[3];
        c0 = ci3 + c0;
        c1 = ci3 * T::R19_EVEN_TWIDDLE_3 + c1;
        s0 = fmla(si3, T::R19_ODD_TWIDDLE_3, s0);
        let ci4 = c_buffer[4];
        let si4 = s_buffer[4];
        c0 = ci4 + c0;
        c1 = ci4 * T::R19_EVEN_TWIDDLE_4 + c1;
        s0 = fmla(si4, T::R19_ODD_TWIDDLE_4, s0);
        let ci5 = c_buffer[5];
        let si5 = s_buffer[5];
        c0 = ci5 + c0;
        c1 = ci5 * T::R19_EVEN_TWIDDLE_5 + c1;
        s0 = fmla(si5, T::R19_ODD_TWIDDLE_5, s0);
        let ci6 = c_buffer[6];
        let si6 = s_buffer[6];
        c0 = ci6 + c0;
        c1 = ci6 * T::R19_EVEN_TWIDDLE_6 + c1;
        s0 = fmla(si6, T::R19_ODD_TWIDDLE_6, s0);
        let ci7 = c_buffer[7];
        let si7 = s_buffer[7];
        c0 = ci7 + c0;
        c1 = ci7 * T::R19_EVEN_TWIDDLE_7 + c1;
        s0 = fmla(si7, T::R19_ODD_TWIDDLE_7, s0);
        let ci8 = c_buffer[8];
        let si8 = s_buffer[8];
        c0 = ci8 + c0;
        c1 = ci8 * T::R19_EVEN_TWIDDLE_8 + c1;
        s0 = fmla(si8, T::R19_ODD_TWIDDLE_8, s0);
        let a0 = a_buffer[0];
        let dc = c0 + a0;
        data[0] = dc;
        data[1] = s0;
        data[2] = -(c1 + a0);
        let mut c0 = qc * T::R19_EVEN_TWIDDLE_8;
        let mut s0 = s_buffer[0];
        s0 *= -T::R19_ODD_TWIDDLE_1;
        c0 = fmla(ci1, T::R19_EVEN_TWIDDLE_6, c0);
        s0 = fmla(si1, -T::R19_ODD_TWIDDLE_4, s0);
        c0 = fmla(ci2, T::R19_EVEN_TWIDDLE_4, c0);
        s0 = fmla(si2, -T::R19_ODD_TWIDDLE_7, s0);
        c0 = fmla(ci3, T::R19_EVEN_TWIDDLE_2, c0);
        s0 = fmla(si3, T::R19_ODD_TWIDDLE_8, s0);
        c0 = fmla(ci4, T::R19_EVEN_TWIDDLE_0, c0);
        s0 = fmla(si4, T::R19_ODD_TWIDDLE_5, s0);
        c0 = fmla(ci5, T::R19_EVEN_TWIDDLE_1, c0);
        s0 = fmla(si5, T::R19_ODD_TWIDDLE_2, s0);
        c0 = fmla(ci6, T::R19_EVEN_TWIDDLE_3, c0);
        s0 = fmla(si6, T::R19_ODD_TWIDDLE_0, s0);
        c0 = fmla(ci7, T::R19_EVEN_TWIDDLE_5, c0);
        s0 = fmla(si7, T::R19_ODD_TWIDDLE_3, s0);
        c0 = fmla(ci8, T::R19_EVEN_TWIDDLE_7, c0);
        s0 = fmla(si8, T::R19_ODD_TWIDDLE_6, s0);
        let dc = c0 + a0;
        data[4] = dc;
        data[3] = -s0;
        let mut c0 = qc * T::R19_EVEN_TWIDDLE_1;
        let mut s0 = s_buffer[0];
        s0 *= T::R19_ODD_TWIDDLE_2;
        c0 = fmla(ci1, T::R19_EVEN_TWIDDLE_4, c0);
        s0 = fmla(si1, T::R19_ODD_TWIDDLE_7, s0);
        c0 = fmla(ci2, T::R19_EVEN_TWIDDLE_7, c0);
        s0 = fmla(si2, -T::R19_ODD_TWIDDLE_6, s0);
        c0 = fmla(ci3, T::R19_EVEN_TWIDDLE_8, c0);
        s0 = fmla(si3, -T::R19_ODD_TWIDDLE_1, s0);
        c0 = fmla(ci4, T::R19_EVEN_TWIDDLE_5, c0);
        s0 = fmla(si4, -T::R19_ODD_TWIDDLE_3, s0);
        c0 = fmla(ci5, T::R19_EVEN_TWIDDLE_2, c0);
        s0 = fmla(si5, -T::R19_ODD_TWIDDLE_8, s0);
        c0 = fmla(ci6, T::R19_EVEN_TWIDDLE_0, c0);
        s0 = fmla(si6, T::R19_ODD_TWIDDLE_5, s0);
        c0 = fmla(ci7, T::R19_EVEN_TWIDDLE_3, c0);
        s0 = fmla(si7, T::R19_ODD_TWIDDLE_0, s0);
        c0 = fmla(ci8, T::R19_EVEN_TWIDDLE_6, c0);
        s0 = fmla(si8, T::R19_ODD_TWIDDLE_4, s0);
        let dc = c0 + a0;
        data[6] = -dc;
        data[5] = s0;
        let mut c0 = qc * T::R19_EVEN_TWIDDLE_7;
        let mut s0 = s_buffer[0];
        s0 *= -T::R19_ODD_TWIDDLE_3;
        c0 = fmla(ci1, T::R19_EVEN_TWIDDLE_3, c0);
        s0 = fmla(si1, T::R19_ODD_TWIDDLE_8, s0);
        c0 = fmla(ci2, T::R19_EVEN_TWIDDLE_0, c0);
        s0 = fmla(si2, T::R19_ODD_TWIDDLE_1, s0);
        c0 = fmla(ci3, T::R19_EVEN_TWIDDLE_4, c0);
        s0 = fmla(si3, T::R19_ODD_TWIDDLE_5, s0);
        c0 = fmla(ci4, T::R19_EVEN_TWIDDLE_8, c0);
        s0 = fmla(si4, -T::R19_ODD_TWIDDLE_6, s0);
        c0 = fmla(ci5, T::R19_EVEN_TWIDDLE_6, c0);
        s0 = fmla(si5, -T::R19_ODD_TWIDDLE_0, s0);
        c0 = fmla(ci6, T::R19_EVEN_TWIDDLE_2, c0);
        s0 = fmla(si6, -T::R19_ODD_TWIDDLE_7, s0);
        c0 = fmla(ci7, T::R19_EVEN_TWIDDLE_1, c0);
        s0 = fmla(si7, T::R19_ODD_TWIDDLE_4, s0);
        c0 = fmla(ci8, T::R19_EVEN_TWIDDLE_5, c0);
        s0 = fmla(si8, T::R19_ODD_TWIDDLE_2, s0);
        let dc = c0 + a0;
        data[8] = dc;
        data[7] = -s0;
        let mut c0 = qc * T::R19_EVEN_TWIDDLE_2;
        let mut s0 = s_buffer[0];
        s0 *= T::R19_ODD_TWIDDLE_4;
        c0 = fmla(ci1, T::R19_EVEN_TWIDDLE_7, c0);
        s0 = fmla(si1, -T::R19_ODD_TWIDDLE_5, s0);
        c0 = fmla(ci2, T::R19_EVEN_TWIDDLE_6, c0);
        s0 = fmla(si2, -T::R19_ODD_TWIDDLE_3, s0);
        c0 = fmla(ci3, T::R19_EVEN_TWIDDLE_1, c0);
        s0 = fmla(si3, T::R19_ODD_TWIDDLE_6, s0);
        c0 = fmla(ci4, T::R19_EVEN_TWIDDLE_3, c0);
        s0 = fmla(si4, T::R19_ODD_TWIDDLE_2, s0);
        c0 = fmla(ci5, T::R19_EVEN_TWIDDLE_8, c0);
        s0 = fmla(si5, -T::R19_ODD_TWIDDLE_7, s0);
        c0 = fmla(ci6, T::R19_EVEN_TWIDDLE_5, c0);
        s0 = fmla(si6, -T::R19_ODD_TWIDDLE_1, s0);
        c0 = fmla(ci7, T::R19_EVEN_TWIDDLE_0, c0);
        s0 = fmla(si7, T::R19_ODD_TWIDDLE_8, s0);
        c0 = fmla(ci8, T::R19_EVEN_TWIDDLE_4, c0);
        s0 = fmla(si8, T::R19_ODD_TWIDDLE_0, s0);
        let dc = c0 + a0;
        data[10] = -dc;
        data[9] = s0;
        let mut c0 = qc * T::R19_EVEN_TWIDDLE_6;
        let mut s0 = s_buffer[0];
        s0 *= -T::R19_ODD_TWIDDLE_5;
        c0 = fmla(ci1, T::R19_EVEN_TWIDDLE_0, c0);
        s0 = fmla(si1, T::R19_ODD_TWIDDLE_2, s0);
        c0 = fmla(ci2, T::R19_EVEN_TWIDDLE_5, c0);
        s0 = fmla(si2, T::R19_ODD_TWIDDLE_8, s0);
        c0 = fmla(ci3, T::R19_EVEN_TWIDDLE_7, c0);
        s0 = fmla(si3, -T::R19_ODD_TWIDDLE_0, s0);
        c0 = fmla(ci4, T::R19_EVEN_TWIDDLE_1, c0);
        s0 = fmla(si4, T::R19_ODD_TWIDDLE_7, s0);
        c0 = fmla(ci5, T::R19_EVEN_TWIDDLE_4, c0);
        s0 = fmla(si5, T::R19_ODD_TWIDDLE_3, s0);
        c0 = fmla(ci6, T::R19_EVEN_TWIDDLE_8, c0);
        s0 = fmla(si6, -T::R19_ODD_TWIDDLE_4, s0);
        c0 = fmla(ci7, T::R19_EVEN_TWIDDLE_2, c0);
        s0 = fmla(si7, -T::R19_ODD_TWIDDLE_6, s0);
        c0 = fmla(ci8, T::R19_EVEN_TWIDDLE_3, c0);
        s0 = fmla(si8, T::R19_ODD_TWIDDLE_1, s0);
        let dc = c0 + a0;
        data[12] = dc;
        data[11] = -s0;
        let mut c0 = qc * T::R19_EVEN_TWIDDLE_3;
        let mut s0 = s_buffer[0];
        s0 *= T::R19_ODD_TWIDDLE_6;
        c0 = fmla(ci1, T::R19_EVEN_TWIDDLE_8, c0);
        s0 = fmla(si1, -T::R19_ODD_TWIDDLE_0, s0);
        c0 = fmla(ci2, T::R19_EVEN_TWIDDLE_1, c0);
        s0 = fmla(si2, T::R19_ODD_TWIDDLE_5, s0);
        c0 = fmla(ci3, T::R19_EVEN_TWIDDLE_5, c0);
        s0 = fmla(si3, T::R19_ODD_TWIDDLE_7, s0);
        c0 = fmla(ci4, T::R19_EVEN_TWIDDLE_6, c0);
        s0 = fmla(si4, -T::R19_ODD_TWIDDLE_1, s0);
        c0 = fmla(ci5, T::R19_EVEN_TWIDDLE_0, c0);
        s0 = fmla(si5, T::R19_ODD_TWIDDLE_4, s0);
        c0 = fmla(ci6, T::R19_EVEN_TWIDDLE_7, c0);
        s0 = fmla(si6, T::R19_ODD_TWIDDLE_8, s0);
        c0 = fmla(ci7, T::R19_EVEN_TWIDDLE_4, c0);
        s0 = fmla(si7, -T::R19_ODD_TWIDDLE_2, s0);
        c0 = fmla(ci8, T::R19_EVEN_TWIDDLE_2, c0);
        s0 = fmla(si8, T::R19_ODD_TWIDDLE_3, s0);
        let dc = c0 + a0;
        data[14] = -dc;
        data[13] = s0;
        let mut c0 = qc * T::R19_EVEN_TWIDDLE_5;
        let mut s0 = s_buffer[0];
        s0 *= -T::R19_ODD_TWIDDLE_7;
        c0 = fmla(ci1, T::R19_EVEN_TWIDDLE_2, c0);
        s0 = fmla(si1, T::R19_ODD_TWIDDLE_3, s0);
        c0 = fmla(ci2, T::R19_EVEN_TWIDDLE_8, c0);
        s0 = fmla(si2, -T::R19_ODD_TWIDDLE_0, s0);
        c0 = fmla(ci3, T::R19_EVEN_TWIDDLE_0, c0);
        s0 = fmla(si3, T::R19_ODD_TWIDDLE_4, s0);
        c0 = fmla(ci4, T::R19_EVEN_TWIDDLE_7, c0);
        s0 = fmla(si4, -T::R19_ODD_TWIDDLE_8, s0);
        c0 = fmla(ci5, T::R19_EVEN_TWIDDLE_3, c0);
        s0 = fmla(si5, -T::R19_ODD_TWIDDLE_6, s0);
        c0 = fmla(ci6, T::R19_EVEN_TWIDDLE_4, c0);
        s0 = fmla(si6, T::R19_ODD_TWIDDLE_2, s0);
        c0 = fmla(ci7, T::R19_EVEN_TWIDDLE_6, c0);
        s0 = fmla(si7, -T::R19_ODD_TWIDDLE_1, s0);
        c0 = fmla(ci8, T::R19_EVEN_TWIDDLE_1, c0);
        s0 = fmla(si8, T::R19_ODD_TWIDDLE_5, s0);
        let dc = c0 + a0;
        data[16] = dc;
        data[15] = -s0;
        let mut c0 = qc * T::R19_EVEN_TWIDDLE_4;
        let mut s0 = s_buffer[0];
        s0 *= T::R19_ODD_TWIDDLE_8;
        c0 = fmla(ci1, T::R19_EVEN_TWIDDLE_5, c0);
        s0 = fmla(si1, -T::R19_ODD_TWIDDLE_6, s0);
        c0 = fmla(ci2, T::R19_EVEN_TWIDDLE_3, c0);
        s0 = fmla(si2, T::R19_ODD_TWIDDLE_4, s0);
        c0 = fmla(ci3, T::R19_EVEN_TWIDDLE_6, c0);
        s0 = fmla(si3, -T::R19_ODD_TWIDDLE_2, s0);
        c0 = fmla(ci4, T::R19_EVEN_TWIDDLE_2, c0);
        s0 = fmla(si4, T::R19_ODD_TWIDDLE_0, s0);
        c0 = fmla(ci5, T::R19_EVEN_TWIDDLE_7, c0);
        s0 = fmla(si5, -T::R19_ODD_TWIDDLE_1, s0);
        c0 = fmla(ci6, T::R19_EVEN_TWIDDLE_1, c0);
        s0 = fmla(si6, T::R19_ODD_TWIDDLE_3, s0);
        c0 = fmla(ci7, T::R19_EVEN_TWIDDLE_8, c0);
        s0 = fmla(si7, -T::R19_ODD_TWIDDLE_5, s0);
        c0 = fmla(ci8, T::R19_EVEN_TWIDDLE_0, c0);
        s0 = fmla(si8, T::R19_ODD_TWIDDLE_7, s0);
        let dc = c0 + a0;
        data[18] = -dc;
        data[17] = s0;
    }
}

define_in_place_butterfly!(Dct2Butterfly19, 19);

pub(crate) trait MixedRadix23Sample {
    const R23_EVEN_TWIDDLE_0: Self;
    const R23_EVEN_TWIDDLE_1: Self;
    const R23_EVEN_TWIDDLE_2: Self;
    const R23_EVEN_TWIDDLE_3: Self;
    const R23_EVEN_TWIDDLE_4: Self;
    const R23_EVEN_TWIDDLE_5: Self;
    const R23_EVEN_TWIDDLE_6: Self;
    const R23_EVEN_TWIDDLE_7: Self;
    const R23_EVEN_TWIDDLE_8: Self;
    const R23_EVEN_TWIDDLE_9: Self;
    const R23_EVEN_TWIDDLE_10: Self;
    const R23_ODD_TWIDDLE_0: Self;
    const R23_ODD_TWIDDLE_1: Self;
    const R23_ODD_TWIDDLE_2: Self;
    const R23_ODD_TWIDDLE_3: Self;
    const R23_ODD_TWIDDLE_4: Self;
    const R23_ODD_TWIDDLE_5: Self;
    const R23_ODD_TWIDDLE_6: Self;
    const R23_ODD_TWIDDLE_7: Self;
    const R23_ODD_TWIDDLE_8: Self;
    const R23_ODD_TWIDDLE_9: Self;
    const R23_ODD_TWIDDLE_10: Self;
}

impl MixedRadix23Sample for f32 {
    const R23_EVEN_TWIDDLE_0: f32 = f32::from_bits(0xbf7d9d98);
    const R23_EVEN_TWIDDLE_1: f32 = f32::from_bits(0xbf6ace5c);
    const R23_EVEN_TWIDDLE_2: f32 = f32::from_bits(0xbf469504);
    const R23_EVEN_TWIDDLE_3: f32 = f32::from_bits(0xbf13a152);
    const R23_EVEN_TWIDDLE_4: f32 = f32::from_bits(0xbeab7557);
    const R23_EVEN_TWIDDLE_5: f32 = f32::from_bits(0xbd8bc2ae);
    const R23_EVEN_TWIDDLE_6: f32 = f32::from_bits(0x3e5056c6);
    const R23_EVEN_TWIDDLE_7: f32 = f32::from_bits(0x3eeb8da5);
    const R23_EVEN_TWIDDLE_8: f32 = f32::from_bits(0x3f2ebbce);
    const R23_EVEN_TWIDDLE_9: f32 = f32::from_bits(0x3f5abb3b);
    const R23_EVEN_TWIDDLE_10: f32 = f32::from_bits(0x3f7681bf);
    const R23_ODD_TWIDDLE_0: f32 = f32::from_bits(0x3f7f6738);
    const R23_ODD_TWIDDLE_1: f32 = f32::from_bits(0x3f7aa541);
    const R23_ODD_TWIDDLE_2: f32 = f32::from_bits(0x3f713803);
    const R23_ODD_TWIDDLE_3: f32 = f32::from_bits(0x3f634c72);
    const R23_ODD_TWIDDLE_4: f32 = f32::from_bits(0x3f5124f0);
    const R23_ODD_TWIDDLE_5: f32 = f32::from_bits(0x3f3b1811);
    const R23_ODD_TWIDDLE_6: f32 = f32::from_bits(0x3f218efb);
    const R23_ODD_TWIDDLE_7: f32 = f32::from_bits(0x3f050374);
    const R23_ODD_TWIDDLE_8: f32 = f32::from_bits(0x3ecbfb3a);
    const R23_ODD_TWIDDLE_9: f32 = f32::from_bits(0x3e8a22cd);
    const R23_ODD_TWIDDLE_10: f32 = f32::from_bits(0x3e0b6f45);
}

impl MixedRadix23Sample for f64 {
    const R23_EVEN_TWIDDLE_0: f64 = f64::from_bits(0xbfefb3b3035aa6cd);
    const R23_EVEN_TWIDDLE_1: f64 = f64::from_bits(0xbfed59cb83ef99bb);
    const R23_EVEN_TWIDDLE_2: f64 = f64::from_bits(0xbfe8d2a07c16d470);
    const R23_EVEN_TWIDDLE_3: f64 = f64::from_bits(0xbfe2742a4a775cfa);
    const R23_EVEN_TWIDDLE_4: f64 = f64::from_bits(0xbfd56eaae597c778);
    const R23_EVEN_TWIDDLE_5: f64 = f64::from_bits(0xbfb17855b599f3b6);
    const R23_EVEN_TWIDDLE_6: f64 = f64::from_bits(0x3fca0ad8bd1e2884);
    const R23_EVEN_TWIDDLE_7: f64 = f64::from_bits(0x3fdd71b4a0c5a6c8);
    const R23_EVEN_TWIDDLE_8: f64 = f64::from_bits(0x3fe5d779b07cfef7);
    const R23_EVEN_TWIDDLE_9: f64 = f64::from_bits(0x3feb57675cf309ee);
    const R23_EVEN_TWIDDLE_10: f64 = f64::from_bits(0x3feed037ea3d2dbb);
    const R23_ODD_TWIDDLE_0: f64 = f64::from_bits(0x3fefece70dfd3efb);
    const R23_ODD_TWIDDLE_1: f64 = f64::from_bits(0x3fef54a827142577);
    const R23_ODD_TWIDDLE_2: f64 = f64::from_bits(0x3fee270060999288);
    const R23_ODD_TWIDDLE_3: f64 = f64::from_bits(0x3fec698e42f47b09);
    const R23_ODD_TWIDDLE_4: f64 = f64::from_bits(0x3fea249e0b897caa);
    const R23_ODD_TWIDDLE_5: f64 = f64::from_bits(0x3fe763021aaa15da);
    const R23_ODD_TWIDDLE_6: f64 = f64::from_bits(0x3fe431df5838f7ef);
    const R23_ODD_TWIDDLE_7: f64 = f64::from_bits(0x3fe0a06e851db7ca);
    const R23_ODD_TWIDDLE_8: f64 = f64::from_bits(0x3fd97f6748e524b2);
    const R23_ODD_TWIDDLE_9: f64 = f64::from_bits(0x3fd14459ad2be466);
    const R23_ODD_TWIDDLE_10: f64 = f64::from_bits(0x3fc16de8a4564f0a);
}

#[derive(Debug, Clone)]
pub(crate) struct Dct2Butterfly23<T: DctSample> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T: DctSample> Default for Dct2Butterfly23<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: DctSample> Dct2Butterfly23<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        // This is autogenerated Radix-Q where Q = 23.
        let a_buffer = [data[11]];
        let c_buffer = [
            data[0] + data[22],
            data[1] + data[21],
            data[2] + data[20],
            data[3] + data[19],
            data[4] + data[18],
            data[5] + data[17],
            data[6] + data[16],
            data[7] + data[15],
            data[8] + data[14],
            data[9] + data[13],
            data[10] + data[12],
        ];
        let s_buffer = [
            data[0] - data[22],
            data[1] - data[21],
            data[2] - data[20],
            data[3] - data[19],
            data[4] - data[18],
            data[5] - data[17],
            data[6] - data[16],
            data[7] - data[15],
            data[8] - data[14],
            data[9] - data[13],
            data[10] - data[12],
        ];
        let qc = c_buffer[0];
        let mut c0 = qc;
        let mut c1 = qc * T::R23_EVEN_TWIDDLE_0;
        let mut s0 = s_buffer[0];
        s0 *= T::R23_ODD_TWIDDLE_0;
        let ci1 = c_buffer[1];
        let si1 = s_buffer[1];
        c0 = ci1 + c0;
        c1 = ci1 * T::R23_EVEN_TWIDDLE_1 + c1;
        s0 = fmla(si1, T::R23_ODD_TWIDDLE_1, s0);
        let ci2 = c_buffer[2];
        let si2 = s_buffer[2];
        c0 = ci2 + c0;
        c1 = ci2 * T::R23_EVEN_TWIDDLE_2 + c1;
        s0 = fmla(si2, T::R23_ODD_TWIDDLE_2, s0);
        let ci3 = c_buffer[3];
        let si3 = s_buffer[3];
        c0 = ci3 + c0;
        c1 = ci3 * T::R23_EVEN_TWIDDLE_3 + c1;
        s0 = fmla(si3, T::R23_ODD_TWIDDLE_3, s0);
        let ci4 = c_buffer[4];
        let si4 = s_buffer[4];
        c0 = ci4 + c0;
        c1 = ci4 * T::R23_EVEN_TWIDDLE_4 + c1;
        s0 = fmla(si4, T::R23_ODD_TWIDDLE_4, s0);
        let ci5 = c_buffer[5];
        let si5 = s_buffer[5];
        c0 = ci5 + c0;
        c1 = ci5 * T::R23_EVEN_TWIDDLE_5 + c1;
        s0 = fmla(si5, T::R23_ODD_TWIDDLE_5, s0);
        let ci6 = c_buffer[6];
        let si6 = s_buffer[6];
        c0 = ci6 + c0;
        c1 = ci6 * T::R23_EVEN_TWIDDLE_6 + c1;
        s0 = fmla(si6, T::R23_ODD_TWIDDLE_6, s0);
        let ci7 = c_buffer[7];
        let si7 = s_buffer[7];
        c0 = ci7 + c0;
        c1 = ci7 * T::R23_EVEN_TWIDDLE_7 + c1;
        s0 = fmla(si7, T::R23_ODD_TWIDDLE_7, s0);
        let ci8 = c_buffer[8];
        let si8 = s_buffer[8];
        c0 = ci8 + c0;
        c1 = ci8 * T::R23_EVEN_TWIDDLE_8 + c1;
        s0 = fmla(si8, T::R23_ODD_TWIDDLE_8, s0);
        let ci9 = c_buffer[9];
        let si9 = s_buffer[9];
        c0 = ci9 + c0;
        c1 = ci9 * T::R23_EVEN_TWIDDLE_9 + c1;
        s0 = fmla(si9, T::R23_ODD_TWIDDLE_9, s0);
        let ci10 = c_buffer[10];
        let si10 = s_buffer[10];
        c0 = ci10 + c0;
        c1 = ci10 * T::R23_EVEN_TWIDDLE_10 + c1;
        s0 = fmla(si10, T::R23_ODD_TWIDDLE_10, s0);
        let a0 = a_buffer[0];
        let dc = c0 + a0;
        data[0] = dc;
        data[1] = s0;
        data[2] = -(c1 + a0);
        let mut c0 = qc * T::R23_EVEN_TWIDDLE_10;
        let mut s0 = s_buffer[0];
        s0 *= -T::R23_ODD_TWIDDLE_1;
        c0 = fmla(ci1, T::R23_EVEN_TWIDDLE_8, c0);
        s0 = fmla(si1, -T::R23_ODD_TWIDDLE_4, s0);
        c0 = fmla(ci2, T::R23_EVEN_TWIDDLE_6, c0);
        s0 = fmla(si2, -T::R23_ODD_TWIDDLE_7, s0);
        c0 = fmla(ci3, T::R23_EVEN_TWIDDLE_4, c0);
        s0 = fmla(si3, -T::R23_ODD_TWIDDLE_10, s0);
        c0 = fmla(ci4, T::R23_EVEN_TWIDDLE_2, c0);
        s0 = fmla(si4, T::R23_ODD_TWIDDLE_9, s0);
        c0 = fmla(ci5, T::R23_EVEN_TWIDDLE_0, c0);
        s0 = fmla(si5, T::R23_ODD_TWIDDLE_6, s0);
        c0 = fmla(ci6, T::R23_EVEN_TWIDDLE_1, c0);
        s0 = fmla(si6, T::R23_ODD_TWIDDLE_3, s0);
        c0 = fmla(ci7, T::R23_EVEN_TWIDDLE_3, c0);
        s0 = fmla(si7, T::R23_ODD_TWIDDLE_0, s0);
        c0 = fmla(ci8, T::R23_EVEN_TWIDDLE_5, c0);
        s0 = fmla(si8, T::R23_ODD_TWIDDLE_2, s0);
        c0 = fmla(ci9, T::R23_EVEN_TWIDDLE_7, c0);
        s0 = fmla(si9, T::R23_ODD_TWIDDLE_5, s0);
        c0 = fmla(ci10, T::R23_EVEN_TWIDDLE_9, c0);
        s0 = fmla(si10, T::R23_ODD_TWIDDLE_8, s0);
        let dc = c0 + a0;
        data[4] = dc;
        data[3] = -s0;
        let mut c0 = qc * T::R23_EVEN_TWIDDLE_1;
        let mut s0 = s_buffer[0];
        s0 *= T::R23_ODD_TWIDDLE_2;
        c0 = fmla(ci1, T::R23_EVEN_TWIDDLE_4, c0);
        s0 = fmla(si1, T::R23_ODD_TWIDDLE_7, s0);
        c0 = fmla(ci2, T::R23_EVEN_TWIDDLE_7, c0);
        s0 = fmla(si2, -T::R23_ODD_TWIDDLE_10, s0);
        c0 = fmla(ci3, T::R23_EVEN_TWIDDLE_10, c0);
        s0 = fmla(si3, -T::R23_ODD_TWIDDLE_5, s0);
        c0 = fmla(ci4, T::R23_EVEN_TWIDDLE_9, c0);
        s0 = fmla(si4, -T::R23_ODD_TWIDDLE_0, s0);
        c0 = fmla(ci5, T::R23_EVEN_TWIDDLE_6, c0);
        s0 = fmla(si5, -T::R23_ODD_TWIDDLE_4, s0);
        c0 = fmla(ci6, T::R23_EVEN_TWIDDLE_3, c0);
        s0 = fmla(si6, -T::R23_ODD_TWIDDLE_9, s0);
        c0 = fmla(ci7, T::R23_EVEN_TWIDDLE_0, c0);
        s0 = fmla(si7, T::R23_ODD_TWIDDLE_8, s0);
        c0 = fmla(ci8, T::R23_EVEN_TWIDDLE_2, c0);
        s0 = fmla(si8, T::R23_ODD_TWIDDLE_3, s0);
        c0 = fmla(ci9, T::R23_EVEN_TWIDDLE_5, c0);
        s0 = fmla(si9, T::R23_ODD_TWIDDLE_1, s0);
        c0 = fmla(ci10, T::R23_EVEN_TWIDDLE_8, c0);
        s0 = fmla(si10, T::R23_ODD_TWIDDLE_6, s0);
        let dc = c0 + a0;
        data[6] = -dc;
        data[5] = s0;
        let mut c0 = qc * T::R23_EVEN_TWIDDLE_9;
        let mut s0 = s_buffer[0];
        s0 *= -T::R23_ODD_TWIDDLE_3;
        c0 = fmla(ci1, T::R23_EVEN_TWIDDLE_5, c0);
        s0 = fmla(si1, -T::R23_ODD_TWIDDLE_10, s0);
        c0 = fmla(ci2, T::R23_EVEN_TWIDDLE_1, c0);
        s0 = fmla(si2, T::R23_ODD_TWIDDLE_5, s0);
        c0 = fmla(ci3, T::R23_EVEN_TWIDDLE_2, c0);
        s0 = fmla(si3, T::R23_ODD_TWIDDLE_1, s0);
        c0 = fmla(ci4, T::R23_EVEN_TWIDDLE_6, c0);
        s0 = fmla(si4, T::R23_ODD_TWIDDLE_8, s0);
        c0 = fmla(ci5, T::R23_EVEN_TWIDDLE_10, c0);
        s0 = fmla(si5, -T::R23_ODD_TWIDDLE_7, s0);
        c0 = fmla(ci6, T::R23_EVEN_TWIDDLE_8, c0);
        s0 = fmla(si6, -T::R23_ODD_TWIDDLE_0, s0);
        c0 = fmla(ci7, T::R23_EVEN_TWIDDLE_4, c0);
        s0 = fmla(si7, -T::R23_ODD_TWIDDLE_6, s0);
        c0 = fmla(ci8, T::R23_EVEN_TWIDDLE_0, c0);
        s0 = fmla(si8, T::R23_ODD_TWIDDLE_9, s0);
        c0 = fmla(ci9, T::R23_EVEN_TWIDDLE_3, c0);
        s0 = fmla(si9, T::R23_ODD_TWIDDLE_2, s0);
        c0 = fmla(ci10, T::R23_EVEN_TWIDDLE_7, c0);
        s0 = fmla(si10, T::R23_ODD_TWIDDLE_4, s0);
        let dc = c0 + a0;
        data[8] = dc;
        data[7] = -s0;
        let mut c0 = qc * T::R23_EVEN_TWIDDLE_2;
        let mut s0 = s_buffer[0];
        s0 *= T::R23_ODD_TWIDDLE_4;
        c0 = fmla(ci1, T::R23_EVEN_TWIDDLE_7, c0);
        s0 = fmla(si1, -T::R23_ODD_TWIDDLE_9, s0);
        c0 = fmla(ci2, T::R23_EVEN_TWIDDLE_10, c0);
        s0 = fmla(si2, -T::R23_ODD_TWIDDLE_0, s0);
        c0 = fmla(ci3, T::R23_EVEN_TWIDDLE_5, c0);
        s0 = fmla(si3, -T::R23_ODD_TWIDDLE_8, s0);
        c0 = fmla(ci4, T::R23_EVEN_TWIDDLE_0, c0);
        s0 = fmla(si4, T::R23_ODD_TWIDDLE_5, s0);
        c0 = fmla(ci5, T::R23_EVEN_TWIDDLE_4, c0);
        s0 = fmla(si5, T::R23_ODD_TWIDDLE_3, s0);
        c0 = fmla(ci6, T::R23_EVEN_TWIDDLE_9, c0);
        s0 = fmla(si6, -T::R23_ODD_TWIDDLE_10, s0);
        c0 = fmla(ci7, T::R23_EVEN_TWIDDLE_8, c0);
        s0 = fmla(si7, -T::R23_ODD_TWIDDLE_1, s0);
        c0 = fmla(ci8, T::R23_EVEN_TWIDDLE_3, c0);
        s0 = fmla(si8, -T::R23_ODD_TWIDDLE_7, s0);
        c0 = fmla(ci9, T::R23_EVEN_TWIDDLE_1, c0);
        s0 = fmla(si9, T::R23_ODD_TWIDDLE_6, s0);
        c0 = fmla(ci10, T::R23_EVEN_TWIDDLE_6, c0);
        s0 = fmla(si10, T::R23_ODD_TWIDDLE_2, s0);
        let dc = c0 + a0;
        data[10] = -dc;
        data[9] = s0;
        let mut c0 = qc * T::R23_EVEN_TWIDDLE_8;
        let mut s0 = s_buffer[0];
        s0 *= -T::R23_ODD_TWIDDLE_5;
        c0 = fmla(ci1, T::R23_EVEN_TWIDDLE_2, c0);
        s0 = fmla(si1, T::R23_ODD_TWIDDLE_6, s0);
        c0 = fmla(ci2, T::R23_EVEN_TWIDDLE_3, c0);
        s0 = fmla(si2, T::R23_ODD_TWIDDLE_4, s0);
        c0 = fmla(ci3, T::R23_EVEN_TWIDDLE_9, c0);
        s0 = fmla(si3, -T::R23_ODD_TWIDDLE_7, s0);
        c0 = fmla(ci4, T::R23_EVEN_TWIDDLE_7, c0);
        s0 = fmla(si4, -T::R23_ODD_TWIDDLE_3, s0);
        c0 = fmla(ci5, T::R23_EVEN_TWIDDLE_1, c0);
        s0 = fmla(si5, T::R23_ODD_TWIDDLE_8, s0);
        c0 = fmla(ci6, T::R23_EVEN_TWIDDLE_4, c0);
        s0 = fmla(si6, T::R23_ODD_TWIDDLE_2, s0);
        c0 = fmla(ci7, T::R23_EVEN_TWIDDLE_10, c0);
        s0 = fmla(si7, -T::R23_ODD_TWIDDLE_9, s0);
        c0 = fmla(ci8, T::R23_EVEN_TWIDDLE_6, c0);
        s0 = fmla(si8, -T::R23_ODD_TWIDDLE_1, s0);
        c0 = fmla(ci9, T::R23_EVEN_TWIDDLE_0, c0);
        s0 = fmla(si9, T::R23_ODD_TWIDDLE_10, s0);
        c0 = fmla(ci10, T::R23_EVEN_TWIDDLE_5, c0);
        s0 = fmla(si10, T::R23_ODD_TWIDDLE_0, s0);
        let dc = c0 + a0;
        data[12] = dc;
        data[11] = -s0;
        let mut c0 = qc * T::R23_EVEN_TWIDDLE_3;
        let mut s0 = s_buffer[0];
        s0 *= T::R23_ODD_TWIDDLE_6;
        c0 = fmla(ci1, T::R23_EVEN_TWIDDLE_10, c0);
        s0 = fmla(si1, -T::R23_ODD_TWIDDLE_3, s0);
        c0 = fmla(ci2, T::R23_EVEN_TWIDDLE_5, c0);
        s0 = fmla(si2, -T::R23_ODD_TWIDDLE_9, s0);
        c0 = fmla(ci3, T::R23_EVEN_TWIDDLE_1, c0);
        s0 = fmla(si3, T::R23_ODD_TWIDDLE_0, s0);
        c0 = fmla(ci4, T::R23_EVEN_TWIDDLE_8, c0);
        s0 = fmla(si4, -T::R23_ODD_TWIDDLE_10, s0);
        c0 = fmla(ci5, T::R23_EVEN_TWIDDLE_7, c0);
        s0 = fmla(si5, -T::R23_ODD_TWIDDLE_2, s0);
        c0 = fmla(ci6, T::R23_EVEN_TWIDDLE_0, c0);
        s0 = fmla(si6, T::R23_ODD_TWIDDLE_7, s0);
        c0 = fmla(ci7, T::R23_EVEN_TWIDDLE_6, c0);
        s0 = fmla(si7, T::R23_ODD_TWIDDLE_5, s0);
        c0 = fmla(ci8, T::R23_EVEN_TWIDDLE_9, c0);
        s0 = fmla(si8, -T::R23_ODD_TWIDDLE_4, s0);
        c0 = fmla(ci9, T::R23_EVEN_TWIDDLE_2, c0);
        s0 = fmla(si9, -T::R23_ODD_TWIDDLE_8, s0);
        c0 = fmla(ci10, T::R23_EVEN_TWIDDLE_4, c0);
        s0 = fmla(si10, T::R23_ODD_TWIDDLE_1, s0);
        let dc = c0 + a0;
        data[14] = -dc;
        data[13] = s0;
        let mut c0 = qc * T::R23_EVEN_TWIDDLE_7;
        let mut s0 = s_buffer[0];
        s0 *= -T::R23_ODD_TWIDDLE_7;
        c0 = fmla(ci1, T::R23_EVEN_TWIDDLE_0, c0);
        s0 = fmla(si1, T::R23_ODD_TWIDDLE_0, s0);
        c0 = fmla(ci2, T::R23_EVEN_TWIDDLE_8, c0);
        s0 = fmla(si2, -T::R23_ODD_TWIDDLE_8, s0);
        c0 = fmla(ci3, T::R23_EVEN_TWIDDLE_6, c0);
        s0 = fmla(si3, -T::R23_ODD_TWIDDLE_6, s0);
        c0 = fmla(ci4, T::R23_EVEN_TWIDDLE_1, c0);
        s0 = fmla(si4, T::R23_ODD_TWIDDLE_1, s0);
        c0 = fmla(ci5, T::R23_EVEN_TWIDDLE_9, c0);
        s0 = fmla(si5, -T::R23_ODD_TWIDDLE_9, s0);
        c0 = fmla(ci6, T::R23_EVEN_TWIDDLE_5, c0);
        s0 = fmla(si6, -T::R23_ODD_TWIDDLE_5, s0);
        c0 = fmla(ci7, T::R23_EVEN_TWIDDLE_2, c0);
        s0 = fmla(si7, T::R23_ODD_TWIDDLE_2, s0);
        c0 = fmla(ci8, T::R23_EVEN_TWIDDLE_10, c0);
        s0 = fmla(si8, -T::R23_ODD_TWIDDLE_10, s0);
        c0 = fmla(ci9, T::R23_EVEN_TWIDDLE_4, c0);
        s0 = fmla(si9, -T::R23_ODD_TWIDDLE_4, s0);
        c0 = fmla(ci10, T::R23_EVEN_TWIDDLE_3, c0);
        s0 = fmla(si10, T::R23_ODD_TWIDDLE_3, s0);
        let dc = c0 + a0;
        data[16] = dc;
        data[15] = -s0;
        let mut c0 = qc * T::R23_EVEN_TWIDDLE_4;
        let mut s0 = s_buffer[0];
        s0 *= T::R23_ODD_TWIDDLE_8;
        c0 = fmla(ci1, T::R23_EVEN_TWIDDLE_9, c0);
        s0 = fmla(si1, -T::R23_ODD_TWIDDLE_2, s0);
        c0 = fmla(ci2, T::R23_EVEN_TWIDDLE_0, c0);
        s0 = fmla(si2, T::R23_ODD_TWIDDLE_3, s0);
        c0 = fmla(ci3, T::R23_EVEN_TWIDDLE_8, c0);
        s0 = fmla(si3, -T::R23_ODD_TWIDDLE_9, s0);
        c0 = fmla(ci4, T::R23_EVEN_TWIDDLE_5, c0);
        s0 = fmla(si4, -T::R23_ODD_TWIDDLE_7, s0);
        c0 = fmla(ci5, T::R23_EVEN_TWIDDLE_3, c0);
        s0 = fmla(si5, T::R23_ODD_TWIDDLE_1, s0);
        c0 = fmla(ci6, T::R23_EVEN_TWIDDLE_10, c0);
        s0 = fmla(si6, -T::R23_ODD_TWIDDLE_4, s0);
        c0 = fmla(ci7, T::R23_EVEN_TWIDDLE_1, c0);
        s0 = fmla(si7, T::R23_ODD_TWIDDLE_10, s0);
        c0 = fmla(ci8, T::R23_EVEN_TWIDDLE_7, c0);
        s0 = fmla(si8, T::R23_ODD_TWIDDLE_6, s0);
        c0 = fmla(ci9, T::R23_EVEN_TWIDDLE_6, c0);
        s0 = fmla(si9, -T::R23_ODD_TWIDDLE_0, s0);
        c0 = fmla(ci10, T::R23_EVEN_TWIDDLE_2, c0);
        s0 = fmla(si10, T::R23_ODD_TWIDDLE_5, s0);
        let dc = c0 + a0;
        data[18] = -dc;
        data[17] = s0;
        let mut c0 = qc * T::R23_EVEN_TWIDDLE_6;
        let mut s0 = s_buffer[0];
        s0 *= -T::R23_ODD_TWIDDLE_9;
        c0 = fmla(ci1, T::R23_EVEN_TWIDDLE_3, c0);
        s0 = fmla(si1, T::R23_ODD_TWIDDLE_5, s0);
        c0 = fmla(ci2, T::R23_EVEN_TWIDDLE_9, c0);
        s0 = fmla(si2, -T::R23_ODD_TWIDDLE_1, s0);
        c0 = fmla(ci3, T::R23_EVEN_TWIDDLE_0, c0);
        s0 = fmla(si3, T::R23_ODD_TWIDDLE_2, s0);
        c0 = fmla(ci4, T::R23_EVEN_TWIDDLE_10, c0);
        s0 = fmla(si4, -T::R23_ODD_TWIDDLE_6, s0);
        c0 = fmla(ci5, T::R23_EVEN_TWIDDLE_2, c0);
        s0 = fmla(si5, T::R23_ODD_TWIDDLE_10, s0);
        c0 = fmla(ci6, T::R23_EVEN_TWIDDLE_7, c0);
        s0 = fmla(si6, T::R23_ODD_TWIDDLE_8, s0);
        c0 = fmla(ci7, T::R23_EVEN_TWIDDLE_5, c0);
        s0 = fmla(si7, -T::R23_ODD_TWIDDLE_4, s0);
        c0 = fmla(ci8, T::R23_EVEN_TWIDDLE_4, c0);
        s0 = fmla(si8, T::R23_ODD_TWIDDLE_0, s0);
        c0 = fmla(ci9, T::R23_EVEN_TWIDDLE_8, c0);
        s0 = fmla(si9, -T::R23_ODD_TWIDDLE_3, s0);
        c0 = fmla(ci10, T::R23_EVEN_TWIDDLE_1, c0);
        s0 = fmla(si10, T::R23_ODD_TWIDDLE_7, s0);
        let dc = c0 + a0;
        data[20] = dc;
        data[19] = -s0;
        let mut c0 = qc * T::R23_EVEN_TWIDDLE_5;
        let mut s0 = s_buffer[0];
        s0 *= T::R23_ODD_TWIDDLE_10;
        c0 = fmla(ci1, T::R23_EVEN_TWIDDLE_6, c0);
        s0 = fmla(si1, -T::R23_ODD_TWIDDLE_8, s0);
        c0 = fmla(ci2, T::R23_EVEN_TWIDDLE_4, c0);
        s0 = fmla(si2, T::R23_ODD_TWIDDLE_6, s0);
        c0 = fmla(ci3, T::R23_EVEN_TWIDDLE_7, c0);
        s0 = fmla(si3, -T::R23_ODD_TWIDDLE_4, s0);
        c0 = fmla(ci4, T::R23_EVEN_TWIDDLE_3, c0);
        s0 = fmla(si4, T::R23_ODD_TWIDDLE_2, s0);
        c0 = fmla(ci5, T::R23_EVEN_TWIDDLE_8, c0);
        s0 = fmla(si5, -T::R23_ODD_TWIDDLE_0, s0);
        c0 = fmla(ci6, T::R23_EVEN_TWIDDLE_2, c0);
        s0 = fmla(si6, T::R23_ODD_TWIDDLE_1, s0);
        c0 = fmla(ci7, T::R23_EVEN_TWIDDLE_9, c0);
        s0 = fmla(si7, -T::R23_ODD_TWIDDLE_3, s0);
        c0 = fmla(ci8, T::R23_EVEN_TWIDDLE_1, c0);
        s0 = fmla(si8, T::R23_ODD_TWIDDLE_5, s0);
        c0 = fmla(ci9, T::R23_EVEN_TWIDDLE_10, c0);
        s0 = fmla(si9, -T::R23_ODD_TWIDDLE_7, s0);
        c0 = fmla(ci10, T::R23_EVEN_TWIDDLE_0, c0);
        s0 = fmla(si10, T::R23_ODD_TWIDDLE_9, s0);
        let dc = c0 + a0;
        data[22] = -dc;
        data[21] = s0;
    }
}

define_in_place_butterfly!(Dct2Butterfly23, 23);

pub(crate) trait MixedRadix29Sample {
    const R29_EVEN_TWIDDLE_0: Self;
    const R29_EVEN_TWIDDLE_1: Self;
    const R29_EVEN_TWIDDLE_2: Self;
    const R29_EVEN_TWIDDLE_3: Self;
    const R29_EVEN_TWIDDLE_4: Self;
    const R29_EVEN_TWIDDLE_5: Self;
    const R29_EVEN_TWIDDLE_6: Self;
    const R29_EVEN_TWIDDLE_7: Self;
    const R29_EVEN_TWIDDLE_8: Self;
    const R29_EVEN_TWIDDLE_9: Self;
    const R29_EVEN_TWIDDLE_10: Self;
    const R29_EVEN_TWIDDLE_11: Self;
    const R29_EVEN_TWIDDLE_12: Self;
    const R29_EVEN_TWIDDLE_13: Self;
    const R29_ODD_TWIDDLE_0: Self;
    const R29_ODD_TWIDDLE_1: Self;
    const R29_ODD_TWIDDLE_2: Self;
    const R29_ODD_TWIDDLE_3: Self;
    const R29_ODD_TWIDDLE_4: Self;
    const R29_ODD_TWIDDLE_5: Self;
    const R29_ODD_TWIDDLE_6: Self;
    const R29_ODD_TWIDDLE_7: Self;
    const R29_ODD_TWIDDLE_8: Self;
    const R29_ODD_TWIDDLE_9: Self;
    const R29_ODD_TWIDDLE_10: Self;
    const R29_ODD_TWIDDLE_11: Self;
    const R29_ODD_TWIDDLE_12: Self;
    const R29_ODD_TWIDDLE_13: Self;
}

impl MixedRadix29Sample for f32 {
    const R29_EVEN_TWIDDLE_0: f32 = f32::from_bits(0xbf7e7fd3);
    const R29_EVEN_TWIDDLE_1: f32 = f32::from_bits(0xbf729966);
    const R29_EVEN_TWIDDLE_2: f32 = f32::from_bits(0xbf5b5afe);
    const R29_EVEN_TWIDDLE_3: f32 = f32::from_bits(0xbf39dad7);
    const R29_EVEN_TWIDDLE_4: f32 = f32::from_bits(0xbf0fa9f5);
    const R29_EVEN_TWIDDLE_5: f32 = f32::from_bits(0xbebd82c0);
    const R29_EVEN_TWIDDLE_6: f32 = f32::from_bits(0xbe25aa2e);
    const R29_EVEN_TWIDDLE_7: f32 = f32::from_bits(0x3d5dc0c3);
    const R29_EVEN_TWIDDLE_8: f32 = f32::from_bits(0x3e88f979);
    const R29_EVEN_TWIDDLE_9: f32 = f32::from_bits(0x3eefd33b);
    const R29_EVEN_TWIDDLE_10: f32 = f32::from_bits(0x3f25bb1c);
    const R29_EVEN_TWIDDLE_11: f32 = f32::from_bits(0x3f4bccc1);
    const R29_EVEN_TWIDDLE_12: f32 = f32::from_bits(0x3f6856dd);
    const R29_EVEN_TWIDDLE_13: f32 = f32::from_bits(0x3f7a03ce);
    const R29_ODD_TWIDDLE_0: f32 = f32::from_bits(0x3f7f9fe3);
    const R29_ODD_TWIDDLE_1: f32 = f32::from_bits(0x3f7ca0aa);
    const R29_ODD_TWIDDLE_2: f32 = f32::from_bits(0x3f76ab36);
    const R29_ODD_TWIDDLE_3: f32 = f32::from_bits(0x3f6dd16b);
    const R29_ODD_TWIDDLE_4: f32 = f32::from_bits(0x3f622dd8);
    const R29_ODD_TWIDDLE_5: f32 = f32::from_bits(0x3f53e36d);
    const R29_ODD_TWIDDLE_6: f32 = f32::from_bits(0x3f431d0d);
    const R29_ODD_TWIDDLE_7: f32 = f32::from_bits(0x3f300d12);
    const R29_ODD_TWIDDLE_8: f32 = f32::from_bits(0x3f1aecb3);
    const R29_ODD_TWIDDLE_9: f32 = f32::from_bits(0x3f03fb56);
    const R29_ODD_TWIDDLE_10: f32 = f32::from_bits(0x3ed6fbb4);
    const R29_ODD_TWIDDLE_11: f32 = f32::from_bits(0x3ea37b7d);
    const R29_ODD_TWIDDLE_12: f32 = f32::from_bits(0x3e5c2136);
    const R29_ODD_TWIDDLE_13: f32 = f32::from_bits(0x3ddd6d81);
}

impl MixedRadix29Sample for f64 {
    const R29_EVEN_TWIDDLE_0: f64 = f64::from_bits(0xbfefcffa67b61650);
    const R29_EVEN_TWIDDLE_1: f64 = f64::from_bits(0xbfee532cbe45c954);
    const R29_EVEN_TWIDDLE_2: f64 = f64::from_bits(0xbfeb6b5fbd9f7255);
    const R29_EVEN_TWIDDLE_3: f64 = f64::from_bits(0xbfe73b5ae5db4e0f);
    const R29_EVEN_TWIDDLE_4: f64 = f64::from_bits(0xbfe1f53e93956dc0);
    const R29_EVEN_TWIDDLE_5: f64 = f64::from_bits(0xbfd7b057f20bf2e5);
    const R29_EVEN_TWIDDLE_6: f64 = f64::from_bits(0xbfc4b545c0234a70);
    const R29_EVEN_TWIDDLE_7: f64 = f64::from_bits(0x3fabb81853a1896d);
    const R29_EVEN_TWIDDLE_8: f64 = f64::from_bits(0x3fd11f2f2e2f1e3c);
    const R29_EVEN_TWIDDLE_9: f64 = f64::from_bits(0x3fddfa67657e7607);
    const R29_EVEN_TWIDDLE_10: f64 = f64::from_bits(0x3fe4b76371208a62);
    const R29_EVEN_TWIDDLE_11: f64 = f64::from_bits(0x3fe979982a38e65a);
    const R29_EVEN_TWIDDLE_12: f64 = f64::from_bits(0x3fed0adb9b447ccf);
    const R29_EVEN_TWIDDLE_13: f64 = f64::from_bits(0x3fef4079c06c0992);
    const R29_ODD_TWIDDLE_0: f64 = f64::from_bits(0x3feff3fc588e859d);
    const R29_ODD_TWIDDLE_1: f64 = f64::from_bits(0x3fef941537248537);
    const R29_ODD_TWIDDLE_2: f64 = f64::from_bits(0x3feed566cb3dcba1);
    const R29_ODD_TWIDDLE_3: f64 = f64::from_bits(0x3fedba2d62cb789f);
    const R29_ODD_TWIDDLE_4: f64 = f64::from_bits(0x3fec45bb0d10918c);
    const R29_ODD_TWIDDLE_5: f64 = f64::from_bits(0x3fea7c6da34af89f);
    const R29_ODD_TWIDDLE_6: f64 = f64::from_bits(0x3fe863a1ada0cfa6);
    const R29_ODD_TWIDDLE_7: f64 = f64::from_bits(0x3fe601a24ba81342);
    const R29_ODD_TWIDDLE_8: f64 = f64::from_bits(0x3fe35d9650d47852);
    const R29_ODD_TWIDDLE_9: f64 = f64::from_bits(0x3fe07f6acd7cdce2);
    const R29_ODD_TWIDDLE_10: f64 = f64::from_bits(0x3fdadf7689c97b6f);
    const R29_ODD_TWIDDLE_11: f64 = f64::from_bits(0x3fd46f6faf5fcb72);
    const R29_ODD_TWIDDLE_12: f64 = f64::from_bits(0x3fcb8426c12812bc);
    const R29_ODD_TWIDDLE_13: f64 = f64::from_bits(0x3fbbadb02034d9ff);
}

#[derive(Debug, Clone)]
pub(crate) struct Dct2Butterfly29<T: DctSample> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T: DctSample> Default for Dct2Butterfly29<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: DctSample> Dct2Butterfly29<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        // This is autogenerated Radix-Q where Q = 29.
        let a_buffer = [data[14]];
        let c_buffer = [
            data[0] + data[28],
            data[1] + data[27],
            data[2] + data[26],
            data[3] + data[25],
            data[4] + data[24],
            data[5] + data[23],
            data[6] + data[22],
            data[7] + data[21],
            data[8] + data[20],
            data[9] + data[19],
            data[10] + data[18],
            data[11] + data[17],
            data[12] + data[16],
            data[13] + data[15],
        ];
        let s_buffer = [
            data[0] - data[28],
            data[1] - data[27],
            data[2] - data[26],
            data[3] - data[25],
            data[4] - data[24],
            data[5] - data[23],
            data[6] - data[22],
            data[7] - data[21],
            data[8] - data[20],
            data[9] - data[19],
            data[10] - data[18],
            data[11] - data[17],
            data[12] - data[16],
            data[13] - data[15],
        ];
        let qc = c_buffer[0];
        let mut c0 = qc;
        let mut c1 = qc * T::R29_EVEN_TWIDDLE_0;
        let mut s0 = s_buffer[0];
        s0 *= T::R29_ODD_TWIDDLE_0;
        let ci1 = c_buffer[1];
        let si1 = s_buffer[1];
        c0 = ci1 + c0;
        c1 = ci1 * T::R29_EVEN_TWIDDLE_1 + c1;
        s0 = fmla(si1, T::R29_ODD_TWIDDLE_1, s0);
        let ci2 = c_buffer[2];
        let si2 = s_buffer[2];
        c0 = ci2 + c0;
        c1 = ci2 * T::R29_EVEN_TWIDDLE_2 + c1;
        s0 = fmla(si2, T::R29_ODD_TWIDDLE_2, s0);
        let ci3 = c_buffer[3];
        let si3 = s_buffer[3];
        c0 = ci3 + c0;
        c1 = ci3 * T::R29_EVEN_TWIDDLE_3 + c1;
        s0 = fmla(si3, T::R29_ODD_TWIDDLE_3, s0);
        let ci4 = c_buffer[4];
        let si4 = s_buffer[4];
        c0 = ci4 + c0;
        c1 = ci4 * T::R29_EVEN_TWIDDLE_4 + c1;
        s0 = fmla(si4, T::R29_ODD_TWIDDLE_4, s0);
        let ci5 = c_buffer[5];
        let si5 = s_buffer[5];
        c0 = ci5 + c0;
        c1 = ci5 * T::R29_EVEN_TWIDDLE_5 + c1;
        s0 = fmla(si5, T::R29_ODD_TWIDDLE_5, s0);
        let ci6 = c_buffer[6];
        let si6 = s_buffer[6];
        c0 = ci6 + c0;
        c1 = ci6 * T::R29_EVEN_TWIDDLE_6 + c1;
        s0 = fmla(si6, T::R29_ODD_TWIDDLE_6, s0);
        let ci7 = c_buffer[7];
        let si7 = s_buffer[7];
        c0 = ci7 + c0;
        c1 = ci7 * T::R29_EVEN_TWIDDLE_7 + c1;
        s0 = fmla(si7, T::R29_ODD_TWIDDLE_7, s0);
        let ci8 = c_buffer[8];
        let si8 = s_buffer[8];
        c0 = ci8 + c0;
        c1 = ci8 * T::R29_EVEN_TWIDDLE_8 + c1;
        s0 = fmla(si8, T::R29_ODD_TWIDDLE_8, s0);
        let ci9 = c_buffer[9];
        let si9 = s_buffer[9];
        c0 = ci9 + c0;
        c1 = ci9 * T::R29_EVEN_TWIDDLE_9 + c1;
        s0 = fmla(si9, T::R29_ODD_TWIDDLE_9, s0);
        let ci10 = c_buffer[10];
        let si10 = s_buffer[10];
        c0 = ci10 + c0;
        c1 = ci10 * T::R29_EVEN_TWIDDLE_10 + c1;
        s0 = fmla(si10, T::R29_ODD_TWIDDLE_10, s0);
        let ci11 = c_buffer[11];
        let si11 = s_buffer[11];
        c0 = ci11 + c0;
        c1 = ci11 * T::R29_EVEN_TWIDDLE_11 + c1;
        s0 = fmla(si11, T::R29_ODD_TWIDDLE_11, s0);
        let ci12 = c_buffer[12];
        let si12 = s_buffer[12];
        c0 = ci12 + c0;
        c1 = ci12 * T::R29_EVEN_TWIDDLE_12 + c1;
        s0 = fmla(si12, T::R29_ODD_TWIDDLE_12, s0);
        let ci13 = c_buffer[13];
        let si13 = s_buffer[13];
        c0 = ci13 + c0;
        c1 = ci13 * T::R29_EVEN_TWIDDLE_13 + c1;
        s0 = fmla(si13, T::R29_ODD_TWIDDLE_13, s0);
        let a0 = a_buffer[0];
        let dc = c0 + a0;
        data[0] = dc;
        data[1] = s0;
        data[2] = -(c1 + a0);
        let mut c0 = qc * T::R29_EVEN_TWIDDLE_13;
        let mut s0 = s_buffer[0];
        s0 *= -T::R29_ODD_TWIDDLE_1;
        c0 = fmla(ci1, T::R29_EVEN_TWIDDLE_11, c0);
        s0 = fmla(si1, -T::R29_ODD_TWIDDLE_4, s0);
        c0 = fmla(ci2, T::R29_EVEN_TWIDDLE_9, c0);
        s0 = fmla(si2, -T::R29_ODD_TWIDDLE_7, s0);
        c0 = fmla(ci3, T::R29_EVEN_TWIDDLE_7, c0);
        s0 = fmla(si3, -T::R29_ODD_TWIDDLE_10, s0);
        c0 = fmla(ci4, T::R29_EVEN_TWIDDLE_5, c0);
        s0 = fmla(si4, -T::R29_ODD_TWIDDLE_13, s0);
        c0 = fmla(ci5, T::R29_EVEN_TWIDDLE_3, c0);
        s0 = fmla(si5, T::R29_ODD_TWIDDLE_12, s0);
        c0 = fmla(ci6, T::R29_EVEN_TWIDDLE_1, c0);
        s0 = fmla(si6, T::R29_ODD_TWIDDLE_9, s0);
        c0 = fmla(ci7, T::R29_EVEN_TWIDDLE_0, c0);
        s0 = fmla(si7, T::R29_ODD_TWIDDLE_6, s0);
        c0 = fmla(ci8, T::R29_EVEN_TWIDDLE_2, c0);
        s0 = fmla(si8, T::R29_ODD_TWIDDLE_3, s0);
        c0 = fmla(ci9, T::R29_EVEN_TWIDDLE_4, c0);
        s0 = fmla(si9, T::R29_ODD_TWIDDLE_0, s0);
        c0 = fmla(ci10, T::R29_EVEN_TWIDDLE_6, c0);
        s0 = fmla(si10, T::R29_ODD_TWIDDLE_2, s0);
        c0 = fmla(ci11, T::R29_EVEN_TWIDDLE_8, c0);
        s0 = fmla(si11, T::R29_ODD_TWIDDLE_5, s0);
        c0 = fmla(ci12, T::R29_EVEN_TWIDDLE_10, c0);
        s0 = fmla(si12, T::R29_ODD_TWIDDLE_8, s0);
        c0 = fmla(ci13, T::R29_EVEN_TWIDDLE_12, c0);
        s0 = fmla(si13, T::R29_ODD_TWIDDLE_11, s0);
        let dc = c0 + a0;
        data[4] = dc;
        data[3] = -s0;
        let mut c0 = qc * T::R29_EVEN_TWIDDLE_1;
        let mut s0 = s_buffer[0];
        s0 *= T::R29_ODD_TWIDDLE_2;
        c0 = fmla(ci1, T::R29_EVEN_TWIDDLE_4, c0);
        s0 = fmla(si1, T::R29_ODD_TWIDDLE_7, s0);
        c0 = fmla(ci2, T::R29_EVEN_TWIDDLE_7, c0);
        s0 = fmla(si2, T::R29_ODD_TWIDDLE_12, s0);
        c0 = fmla(ci3, T::R29_EVEN_TWIDDLE_10, c0);
        s0 = fmla(si3, -T::R29_ODD_TWIDDLE_11, s0);
        c0 = fmla(ci4, T::R29_EVEN_TWIDDLE_13, c0);
        s0 = fmla(si4, -T::R29_ODD_TWIDDLE_6, s0);
        c0 = fmla(ci5, T::R29_EVEN_TWIDDLE_12, c0);
        s0 = fmla(si5, -T::R29_ODD_TWIDDLE_1, s0);
        c0 = fmla(ci6, T::R29_EVEN_TWIDDLE_9, c0);
        s0 = fmla(si6, -T::R29_ODD_TWIDDLE_3, s0);
        c0 = fmla(ci7, T::R29_EVEN_TWIDDLE_6, c0);
        s0 = fmla(si7, -T::R29_ODD_TWIDDLE_8, s0);
        c0 = fmla(ci8, T::R29_EVEN_TWIDDLE_3, c0);
        s0 = fmla(si8, -T::R29_ODD_TWIDDLE_13, s0);
        c0 = fmla(ci9, T::R29_EVEN_TWIDDLE_0, c0);
        s0 = fmla(si9, T::R29_ODD_TWIDDLE_10, s0);
        c0 = fmla(ci10, T::R29_EVEN_TWIDDLE_2, c0);
        s0 = fmla(si10, T::R29_ODD_TWIDDLE_5, s0);
        c0 = fmla(ci11, T::R29_EVEN_TWIDDLE_5, c0);
        s0 = fmla(si11, T::R29_ODD_TWIDDLE_0, s0);
        c0 = fmla(ci12, T::R29_EVEN_TWIDDLE_8, c0);
        s0 = fmla(si12, T::R29_ODD_TWIDDLE_4, s0);
        c0 = fmla(ci13, T::R29_EVEN_TWIDDLE_11, c0);
        s0 = fmla(si13, T::R29_ODD_TWIDDLE_9, s0);
        let dc = c0 + a0;
        data[6] = -dc;
        data[5] = s0;
        let mut c0 = qc * T::R29_EVEN_TWIDDLE_12;
        let mut s0 = s_buffer[0];
        s0 *= -T::R29_ODD_TWIDDLE_3;
        c0 = fmla(ci1, T::R29_EVEN_TWIDDLE_8, c0);
        s0 = fmla(si1, -T::R29_ODD_TWIDDLE_10, s0);
        c0 = fmla(ci2, T::R29_EVEN_TWIDDLE_4, c0);
        s0 = fmla(si2, T::R29_ODD_TWIDDLE_11, s0);
        c0 = fmla(ci3, T::R29_EVEN_TWIDDLE_0, c0);
        s0 = fmla(si3, T::R29_ODD_TWIDDLE_4, s0);
        c0 = fmla(ci4, T::R29_EVEN_TWIDDLE_3, c0);
        s0 = fmla(si4, T::R29_ODD_TWIDDLE_2, s0);
        c0 = fmla(ci5, T::R29_EVEN_TWIDDLE_7, c0);
        s0 = fmla(si5, T::R29_ODD_TWIDDLE_9, s0);
        c0 = fmla(ci6, T::R29_EVEN_TWIDDLE_11, c0);
        s0 = fmla(si6, -T::R29_ODD_TWIDDLE_12, s0);
        c0 = fmla(ci7, T::R29_EVEN_TWIDDLE_13, c0);
        s0 = fmla(si7, -T::R29_ODD_TWIDDLE_5, s0);
        c0 = fmla(ci8, T::R29_EVEN_TWIDDLE_9, c0);
        s0 = fmla(si8, -T::R29_ODD_TWIDDLE_1, s0);
        c0 = fmla(ci9, T::R29_EVEN_TWIDDLE_5, c0);
        s0 = fmla(si9, -T::R29_ODD_TWIDDLE_8, s0);
        c0 = fmla(ci10, T::R29_EVEN_TWIDDLE_1, c0);
        s0 = fmla(si10, T::R29_ODD_TWIDDLE_13, s0);
        c0 = fmla(ci11, T::R29_EVEN_TWIDDLE_2, c0);
        s0 = fmla(si11, T::R29_ODD_TWIDDLE_6, s0);
        c0 = fmla(ci12, T::R29_EVEN_TWIDDLE_6, c0);
        s0 = fmla(si12, T::R29_ODD_TWIDDLE_0, s0);
        c0 = fmla(ci13, T::R29_EVEN_TWIDDLE_10, c0);
        s0 = fmla(si13, T::R29_ODD_TWIDDLE_7, s0);
        let dc = c0 + a0;
        data[8] = dc;
        data[7] = -s0;
        let mut c0 = qc * T::R29_EVEN_TWIDDLE_2;
        let mut s0 = s_buffer[0];
        s0 *= T::R29_ODD_TWIDDLE_4;
        c0 = fmla(ci1, T::R29_EVEN_TWIDDLE_7, c0);
        s0 = fmla(si1, T::R29_ODD_TWIDDLE_13, s0);
        c0 = fmla(ci2, T::R29_EVEN_TWIDDLE_12, c0);
        s0 = fmla(si2, -T::R29_ODD_TWIDDLE_6, s0);
        c0 = fmla(ci3, T::R29_EVEN_TWIDDLE_11, c0);
        s0 = fmla(si3, -T::R29_ODD_TWIDDLE_2, s0);
        c0 = fmla(ci4, T::R29_EVEN_TWIDDLE_6, c0);
        s0 = fmla(si4, -T::R29_ODD_TWIDDLE_11, s0);
        c0 = fmla(ci5, T::R29_EVEN_TWIDDLE_1, c0);
        s0 = fmla(si5, T::R29_ODD_TWIDDLE_8, s0);
        c0 = fmla(ci6, T::R29_EVEN_TWIDDLE_3, c0);
        s0 = fmla(si6, T::R29_ODD_TWIDDLE_0, s0);
        c0 = fmla(ci7, T::R29_EVEN_TWIDDLE_8, c0);
        s0 = fmla(si7, T::R29_ODD_TWIDDLE_9, s0);
        c0 = fmla(ci8, T::R29_EVEN_TWIDDLE_13, c0);
        s0 = fmla(si8, -T::R29_ODD_TWIDDLE_10, s0);
        c0 = fmla(ci9, T::R29_EVEN_TWIDDLE_10, c0);
        s0 = fmla(si9, -T::R29_ODD_TWIDDLE_1, s0);
        c0 = fmla(ci10, T::R29_EVEN_TWIDDLE_5, c0);
        s0 = fmla(si10, -T::R29_ODD_TWIDDLE_7, s0);
        c0 = fmla(ci11, T::R29_EVEN_TWIDDLE_0, c0);
        s0 = fmla(si11, T::R29_ODD_TWIDDLE_12, s0);
        c0 = fmla(ci12, T::R29_EVEN_TWIDDLE_4, c0);
        s0 = fmla(si12, T::R29_ODD_TWIDDLE_3, s0);
        c0 = fmla(ci13, T::R29_EVEN_TWIDDLE_9, c0);
        s0 = fmla(si13, T::R29_ODD_TWIDDLE_5, s0);
        let dc = c0 + a0;
        data[10] = -dc;
        data[9] = s0;
        let mut c0 = qc * T::R29_EVEN_TWIDDLE_11;
        let mut s0 = s_buffer[0];
        s0 *= -T::R29_ODD_TWIDDLE_5;
        c0 = fmla(ci1, T::R29_EVEN_TWIDDLE_5, c0);
        s0 = fmla(si1, T::R29_ODD_TWIDDLE_12, s0);
        c0 = fmla(ci2, T::R29_EVEN_TWIDDLE_0, c0);
        s0 = fmla(si2, T::R29_ODD_TWIDDLE_1, s0);
        c0 = fmla(ci3, T::R29_EVEN_TWIDDLE_6, c0);
        s0 = fmla(si3, T::R29_ODD_TWIDDLE_9, s0);
        c0 = fmla(ci4, T::R29_EVEN_TWIDDLE_12, c0);
        s0 = fmla(si4, -T::R29_ODD_TWIDDLE_8, s0);
        c0 = fmla(ci5, T::R29_EVEN_TWIDDLE_10, c0);
        s0 = fmla(si5, -T::R29_ODD_TWIDDLE_2, s0);
        c0 = fmla(ci6, T::R29_EVEN_TWIDDLE_4, c0);
        s0 = fmla(si6, -T::R29_ODD_TWIDDLE_13, s0);
        c0 = fmla(ci7, T::R29_EVEN_TWIDDLE_1, c0);
        s0 = fmla(si7, T::R29_ODD_TWIDDLE_4, s0);
        c0 = fmla(ci8, T::R29_EVEN_TWIDDLE_7, c0);
        s0 = fmla(si8, T::R29_ODD_TWIDDLE_6, s0);
        c0 = fmla(ci9, T::R29_EVEN_TWIDDLE_13, c0);
        s0 = fmla(si9, -T::R29_ODD_TWIDDLE_11, s0);
        c0 = fmla(ci10, T::R29_EVEN_TWIDDLE_9, c0);
        s0 = fmla(si10, -T::R29_ODD_TWIDDLE_0, s0);
        c0 = fmla(ci11, T::R29_EVEN_TWIDDLE_3, c0);
        s0 = fmla(si11, -T::R29_ODD_TWIDDLE_10, s0);
        c0 = fmla(ci12, T::R29_EVEN_TWIDDLE_2, c0);
        s0 = fmla(si12, T::R29_ODD_TWIDDLE_7, s0);
        c0 = fmla(ci13, T::R29_EVEN_TWIDDLE_8, c0);
        s0 = fmla(si13, T::R29_ODD_TWIDDLE_3, s0);
        let dc = c0 + a0;
        data[12] = dc;
        data[11] = -s0;
        let mut c0 = qc * T::R29_EVEN_TWIDDLE_3;
        let mut s0 = s_buffer[0];
        s0 *= T::R29_ODD_TWIDDLE_6;
        c0 = fmla(ci1, T::R29_EVEN_TWIDDLE_10, c0);
        s0 = fmla(si1, -T::R29_ODD_TWIDDLE_9, s0);
        c0 = fmla(ci2, T::R29_EVEN_TWIDDLE_11, c0);
        s0 = fmla(si2, -T::R29_ODD_TWIDDLE_3, s0);
        c0 = fmla(ci3, T::R29_EVEN_TWIDDLE_4, c0);
        s0 = fmla(si3, T::R29_ODD_TWIDDLE_12, s0);
        c0 = fmla(ci4, T::R29_EVEN_TWIDDLE_2, c0);
        s0 = fmla(si4, T::R29_ODD_TWIDDLE_0, s0);
        c0 = fmla(ci5, T::R29_EVEN_TWIDDLE_9, c0);
        s0 = fmla(si5, T::R29_ODD_TWIDDLE_13, s0);
        c0 = fmla(ci6, T::R29_EVEN_TWIDDLE_12, c0);
        s0 = fmla(si6, -T::R29_ODD_TWIDDLE_2, s0);
        c0 = fmla(ci7, T::R29_EVEN_TWIDDLE_5, c0);
        s0 = fmla(si7, -T::R29_ODD_TWIDDLE_10, s0);
        c0 = fmla(ci8, T::R29_EVEN_TWIDDLE_1, c0);
        s0 = fmla(si8, T::R29_ODD_TWIDDLE_5, s0);
        c0 = fmla(ci9, T::R29_EVEN_TWIDDLE_8, c0);
        s0 = fmla(si9, T::R29_ODD_TWIDDLE_7, s0);
        c0 = fmla(ci10, T::R29_EVEN_TWIDDLE_13, c0);
        s0 = fmla(si10, -T::R29_ODD_TWIDDLE_8, s0);
        c0 = fmla(ci11, T::R29_EVEN_TWIDDLE_6, c0);
        s0 = fmla(si11, -T::R29_ODD_TWIDDLE_4, s0);
        c0 = fmla(ci12, T::R29_EVEN_TWIDDLE_0, c0);
        s0 = fmla(si12, T::R29_ODD_TWIDDLE_11, s0);
        c0 = fmla(ci13, T::R29_EVEN_TWIDDLE_7, c0);
        s0 = fmla(si13, T::R29_ODD_TWIDDLE_1, s0);
        let dc = c0 + a0;
        data[14] = -dc;
        data[13] = s0;
        let mut c0 = qc * T::R29_EVEN_TWIDDLE_10;
        let mut s0 = s_buffer[0];
        s0 *= -T::R29_ODD_TWIDDLE_7;
        c0 = fmla(ci1, T::R29_EVEN_TWIDDLE_2, c0);
        s0 = fmla(si1, T::R29_ODD_TWIDDLE_6, s0);
        c0 = fmla(ci2, T::R29_EVEN_TWIDDLE_5, c0);
        s0 = fmla(si2, T::R29_ODD_TWIDDLE_8, s0);
        c0 = fmla(ci3, T::R29_EVEN_TWIDDLE_13, c0);
        s0 = fmla(si3, -T::R29_ODD_TWIDDLE_5, s0);
        c0 = fmla(ci4, T::R29_EVEN_TWIDDLE_7, c0);
        s0 = fmla(si4, -T::R29_ODD_TWIDDLE_9, s0);
        c0 = fmla(ci5, T::R29_EVEN_TWIDDLE_0, c0);
        s0 = fmla(si5, T::R29_ODD_TWIDDLE_4, s0);
        c0 = fmla(ci6, T::R29_EVEN_TWIDDLE_8, c0);
        s0 = fmla(si6, T::R29_ODD_TWIDDLE_10, s0);
        c0 = fmla(ci7, T::R29_EVEN_TWIDDLE_12, c0);
        s0 = fmla(si7, -T::R29_ODD_TWIDDLE_3, s0);
        c0 = fmla(ci8, T::R29_EVEN_TWIDDLE_4, c0);
        s0 = fmla(si8, -T::R29_ODD_TWIDDLE_11, s0);
        c0 = fmla(ci9, T::R29_EVEN_TWIDDLE_3, c0);
        s0 = fmla(si9, T::R29_ODD_TWIDDLE_2, s0);
        c0 = fmla(ci10, T::R29_EVEN_TWIDDLE_11, c0);
        s0 = fmla(si10, T::R29_ODD_TWIDDLE_12, s0);
        c0 = fmla(ci11, T::R29_EVEN_TWIDDLE_9, c0);
        s0 = fmla(si11, -T::R29_ODD_TWIDDLE_1, s0);
        c0 = fmla(ci12, T::R29_EVEN_TWIDDLE_1, c0);
        s0 = fmla(si12, -T::R29_ODD_TWIDDLE_13, s0);
        c0 = fmla(ci13, T::R29_EVEN_TWIDDLE_6, c0);
        s0 = fmla(si13, T::R29_ODD_TWIDDLE_0, s0);
        let dc = c0 + a0;
        data[16] = dc;
        data[15] = -s0;
        let mut c0 = qc * T::R29_EVEN_TWIDDLE_4;
        let mut s0 = s_buffer[0];
        s0 *= T::R29_ODD_TWIDDLE_8;
        c0 = fmla(ci1, T::R29_EVEN_TWIDDLE_13, c0);
        s0 = fmla(si1, -T::R29_ODD_TWIDDLE_3, s0);
        c0 = fmla(ci2, T::R29_EVEN_TWIDDLE_6, c0);
        s0 = fmla(si2, -T::R29_ODD_TWIDDLE_13, s0);
        c0 = fmla(ci3, T::R29_EVEN_TWIDDLE_2, c0);
        s0 = fmla(si3, T::R29_ODD_TWIDDLE_1, s0);
        c0 = fmla(ci4, T::R29_EVEN_TWIDDLE_11, c0);
        s0 = fmla(si4, -T::R29_ODD_TWIDDLE_10, s0);
        c0 = fmla(ci5, T::R29_EVEN_TWIDDLE_8, c0);
        s0 = fmla(si5, -T::R29_ODD_TWIDDLE_6, s0);
        c0 = fmla(ci6, T::R29_EVEN_TWIDDLE_0, c0);
        s0 = fmla(si6, T::R29_ODD_TWIDDLE_5, s0);
        c0 = fmla(ci7, T::R29_EVEN_TWIDDLE_9, c0);
        s0 = fmla(si7, T::R29_ODD_TWIDDLE_11, s0);
        c0 = fmla(ci8, T::R29_EVEN_TWIDDLE_10, c0);
        s0 = fmla(si8, -T::R29_ODD_TWIDDLE_0, s0);
        c0 = fmla(ci9, T::R29_EVEN_TWIDDLE_1, c0);
        s0 = fmla(si9, T::R29_ODD_TWIDDLE_12, s0);
        c0 = fmla(ci10, T::R29_EVEN_TWIDDLE_7, c0);
        s0 = fmla(si10, T::R29_ODD_TWIDDLE_4, s0);
        c0 = fmla(ci11, T::R29_EVEN_TWIDDLE_12, c0);
        s0 = fmla(si11, -T::R29_ODD_TWIDDLE_7, s0);
        c0 = fmla(ci12, T::R29_EVEN_TWIDDLE_3, c0);
        s0 = fmla(si12, -T::R29_ODD_TWIDDLE_9, s0);
        c0 = fmla(ci13, T::R29_EVEN_TWIDDLE_5, c0);
        s0 = fmla(si13, T::R29_ODD_TWIDDLE_2, s0);
        let dc = c0 + a0;
        data[18] = -dc;
        data[17] = s0;
        let mut c0 = qc * T::R29_EVEN_TWIDDLE_9;
        let mut s0 = s_buffer[0];
        s0 *= -T::R29_ODD_TWIDDLE_9;
        c0 = fmla(ci1, T::R29_EVEN_TWIDDLE_0, c0);
        s0 = fmla(si1, T::R29_ODD_TWIDDLE_0, s0);
        c0 = fmla(ci2, T::R29_EVEN_TWIDDLE_10, c0);
        s0 = fmla(si2, -T::R29_ODD_TWIDDLE_10, s0);
        c0 = fmla(ci3, T::R29_EVEN_TWIDDLE_8, c0);
        s0 = fmla(si3, -T::R29_ODD_TWIDDLE_8, s0);
        c0 = fmla(ci4, T::R29_EVEN_TWIDDLE_1, c0);
        s0 = fmla(si4, T::R29_ODD_TWIDDLE_1, s0);
        c0 = fmla(ci5, T::R29_EVEN_TWIDDLE_11, c0);
        s0 = fmla(si5, -T::R29_ODD_TWIDDLE_11, s0);
        c0 = fmla(ci6, T::R29_EVEN_TWIDDLE_7, c0);
        s0 = fmla(si6, -T::R29_ODD_TWIDDLE_7, s0);
        c0 = fmla(ci7, T::R29_EVEN_TWIDDLE_2, c0);
        s0 = fmla(si7, T::R29_ODD_TWIDDLE_2, s0);
        c0 = fmla(ci8, T::R29_EVEN_TWIDDLE_12, c0);
        s0 = fmla(si8, -T::R29_ODD_TWIDDLE_12, s0);
        c0 = fmla(ci9, T::R29_EVEN_TWIDDLE_6, c0);
        s0 = fmla(si9, -T::R29_ODD_TWIDDLE_6, s0);
        c0 = fmla(ci10, T::R29_EVEN_TWIDDLE_3, c0);
        s0 = fmla(si10, T::R29_ODD_TWIDDLE_3, s0);
        c0 = fmla(ci11, T::R29_EVEN_TWIDDLE_13, c0);
        s0 = fmla(si11, -T::R29_ODD_TWIDDLE_13, s0);
        c0 = fmla(ci12, T::R29_EVEN_TWIDDLE_5, c0);
        s0 = fmla(si12, -T::R29_ODD_TWIDDLE_5, s0);
        c0 = fmla(ci13, T::R29_EVEN_TWIDDLE_4, c0);
        s0 = fmla(si13, T::R29_ODD_TWIDDLE_4, s0);
        let dc = c0 + a0;
        data[20] = dc;
        data[19] = -s0;
        let mut c0 = qc * T::R29_EVEN_TWIDDLE_5;
        let mut s0 = s_buffer[0];
        s0 *= T::R29_ODD_TWIDDLE_10;
        c0 = fmla(ci1, T::R29_EVEN_TWIDDLE_12, c0);
        s0 = fmla(si1, -T::R29_ODD_TWIDDLE_2, s0);
        c0 = fmla(ci2, T::R29_EVEN_TWIDDLE_1, c0);
        s0 = fmla(si2, T::R29_ODD_TWIDDLE_5, s0);
        c0 = fmla(ci3, T::R29_EVEN_TWIDDLE_9, c0);
        s0 = fmla(si3, -T::R29_ODD_TWIDDLE_13, s0);
        c0 = fmla(ci4, T::R29_EVEN_TWIDDLE_8, c0);
        s0 = fmla(si4, -T::R29_ODD_TWIDDLE_7, s0);
        c0 = fmla(ci5, T::R29_EVEN_TWIDDLE_2, c0);
        s0 = fmla(si5, T::R29_ODD_TWIDDLE_0, s0);
        c0 = fmla(ci6, T::R29_EVEN_TWIDDLE_13, c0);
        s0 = fmla(si6, -T::R29_ODD_TWIDDLE_8, s0);
        c0 = fmla(ci7, T::R29_EVEN_TWIDDLE_4, c0);
        s0 = fmla(si7, -T::R29_ODD_TWIDDLE_12, s0);
        c0 = fmla(ci8, T::R29_EVEN_TWIDDLE_6, c0);
        s0 = fmla(si8, T::R29_ODD_TWIDDLE_4, s0);
        c0 = fmla(ci9, T::R29_EVEN_TWIDDLE_11, c0);
        s0 = fmla(si9, -T::R29_ODD_TWIDDLE_3, s0);
        c0 = fmla(ci10, T::R29_EVEN_TWIDDLE_0, c0);
        s0 = fmla(si10, T::R29_ODD_TWIDDLE_11, s0);
        c0 = fmla(ci11, T::R29_EVEN_TWIDDLE_10, c0);
        s0 = fmla(si11, T::R29_ODD_TWIDDLE_9, s0);
        c0 = fmla(ci12, T::R29_EVEN_TWIDDLE_7, c0);
        s0 = fmla(si12, -T::R29_ODD_TWIDDLE_1, s0);
        c0 = fmla(ci13, T::R29_EVEN_TWIDDLE_3, c0);
        s0 = fmla(si13, T::R29_ODD_TWIDDLE_6, s0);
        let dc = c0 + a0;
        data[22] = -dc;
        data[21] = s0;
        let mut c0 = qc * T::R29_EVEN_TWIDDLE_8;
        let mut s0 = s_buffer[0];
        s0 *= -T::R29_ODD_TWIDDLE_11;
        c0 = fmla(ci1, T::R29_EVEN_TWIDDLE_3, c0);
        s0 = fmla(si1, T::R29_ODD_TWIDDLE_5, s0);
        c0 = fmla(ci2, T::R29_EVEN_TWIDDLE_13, c0);
        s0 = fmla(si2, -T::R29_ODD_TWIDDLE_0, s0);
        c0 = fmla(ci3, T::R29_EVEN_TWIDDLE_1, c0);
        s0 = fmla(si3, T::R29_ODD_TWIDDLE_6, s0);
        c0 = fmla(ci4, T::R29_EVEN_TWIDDLE_10, c0);
        s0 = fmla(si4, -T::R29_ODD_TWIDDLE_12, s0);
        c0 = fmla(ci5, T::R29_EVEN_TWIDDLE_6, c0);
        s0 = fmla(si5, -T::R29_ODD_TWIDDLE_10, s0);
        c0 = fmla(ci6, T::R29_EVEN_TWIDDLE_5, c0);
        s0 = fmla(si6, T::R29_ODD_TWIDDLE_4, s0);
        c0 = fmla(ci7, T::R29_EVEN_TWIDDLE_11, c0);
        s0 = fmla(si7, -T::R29_ODD_TWIDDLE_1, s0);
        c0 = fmla(ci8, T::R29_EVEN_TWIDDLE_0, c0);
        s0 = fmla(si8, T::R29_ODD_TWIDDLE_7, s0);
        c0 = fmla(ci9, T::R29_EVEN_TWIDDLE_12, c0);
        s0 = fmla(si9, -T::R29_ODD_TWIDDLE_13, s0);
        c0 = fmla(ci10, T::R29_EVEN_TWIDDLE_4, c0);
        s0 = fmla(si10, -T::R29_ODD_TWIDDLE_9, s0);
        c0 = fmla(ci11, T::R29_EVEN_TWIDDLE_7, c0);
        s0 = fmla(si11, T::R29_ODD_TWIDDLE_3, s0);
        c0 = fmla(ci12, T::R29_EVEN_TWIDDLE_9, c0);
        s0 = fmla(si12, -T::R29_ODD_TWIDDLE_2, s0);
        c0 = fmla(ci13, T::R29_EVEN_TWIDDLE_2, c0);
        s0 = fmla(si13, T::R29_ODD_TWIDDLE_8, s0);
        let dc = c0 + a0;
        data[24] = dc;
        data[23] = -s0;
        let mut c0 = qc * T::R29_EVEN_TWIDDLE_6;
        let mut s0 = s_buffer[0];
        s0 *= T::R29_ODD_TWIDDLE_12;
        c0 = fmla(ci1, T::R29_EVEN_TWIDDLE_9, c0);
        s0 = fmla(si1, -T::R29_ODD_TWIDDLE_8, s0);
        c0 = fmla(ci2, T::R29_EVEN_TWIDDLE_3, c0);
        s0 = fmla(si2, T::R29_ODD_TWIDDLE_4, s0);
        c0 = fmla(ci3, T::R29_EVEN_TWIDDLE_12, c0);
        s0 = fmla(si3, -T::R29_ODD_TWIDDLE_0, s0);
        c0 = fmla(ci4, T::R29_EVEN_TWIDDLE_0, c0);
        s0 = fmla(si4, T::R29_ODD_TWIDDLE_3, s0);
        c0 = fmla(ci5, T::R29_EVEN_TWIDDLE_13, c0);
        s0 = fmla(si5, -T::R29_ODD_TWIDDLE_7, s0);
        c0 = fmla(ci6, T::R29_EVEN_TWIDDLE_2, c0);
        s0 = fmla(si6, T::R29_ODD_TWIDDLE_11, s0);
        c0 = fmla(ci7, T::R29_EVEN_TWIDDLE_10, c0);
        s0 = fmla(si7, T::R29_ODD_TWIDDLE_13, s0);
        c0 = fmla(ci8, T::R29_EVEN_TWIDDLE_5, c0);
        s0 = fmla(si8, -T::R29_ODD_TWIDDLE_9, s0);
        c0 = fmla(ci9, T::R29_EVEN_TWIDDLE_7, c0);
        s0 = fmla(si9, T::R29_ODD_TWIDDLE_5, s0);
        c0 = fmla(ci10, T::R29_EVEN_TWIDDLE_8, c0);
        s0 = fmla(si10, -T::R29_ODD_TWIDDLE_1, s0);
        c0 = fmla(ci11, T::R29_EVEN_TWIDDLE_4, c0);
        s0 = fmla(si11, T::R29_ODD_TWIDDLE_2, s0);
        c0 = fmla(ci12, T::R29_EVEN_TWIDDLE_11, c0);
        s0 = fmla(si12, -T::R29_ODD_TWIDDLE_6, s0);
        c0 = fmla(ci13, T::R29_EVEN_TWIDDLE_1, c0);
        s0 = fmla(si13, T::R29_ODD_TWIDDLE_10, s0);
        let dc = c0 + a0;
        data[26] = -dc;
        data[25] = s0;
        let mut c0 = qc * T::R29_EVEN_TWIDDLE_7;
        let mut s0 = s_buffer[0];
        s0 *= -T::R29_ODD_TWIDDLE_13;
        c0 = fmla(ci1, T::R29_EVEN_TWIDDLE_6, c0);
        s0 = fmla(si1, T::R29_ODD_TWIDDLE_11, s0);
        c0 = fmla(ci2, T::R29_EVEN_TWIDDLE_8, c0);
        s0 = fmla(si2, -T::R29_ODD_TWIDDLE_9, s0);
        c0 = fmla(ci3, T::R29_EVEN_TWIDDLE_5, c0);
        s0 = fmla(si3, T::R29_ODD_TWIDDLE_7, s0);
        c0 = fmla(ci4, T::R29_EVEN_TWIDDLE_9, c0);
        s0 = fmla(si4, -T::R29_ODD_TWIDDLE_5, s0);
        c0 = fmla(ci5, T::R29_EVEN_TWIDDLE_4, c0);
        s0 = fmla(si5, T::R29_ODD_TWIDDLE_3, s0);
        c0 = fmla(ci6, T::R29_EVEN_TWIDDLE_10, c0);
        s0 = fmla(si6, -T::R29_ODD_TWIDDLE_1, s0);
        c0 = fmla(ci7, T::R29_EVEN_TWIDDLE_3, c0);
        s0 = fmla(si7, T::R29_ODD_TWIDDLE_0, s0);
        c0 = fmla(ci8, T::R29_EVEN_TWIDDLE_11, c0);
        s0 = fmla(si8, -T::R29_ODD_TWIDDLE_2, s0);
        c0 = fmla(ci9, T::R29_EVEN_TWIDDLE_2, c0);
        s0 = fmla(si9, T::R29_ODD_TWIDDLE_4, s0);
        c0 = fmla(ci10, T::R29_EVEN_TWIDDLE_12, c0);
        s0 = fmla(si10, -T::R29_ODD_TWIDDLE_6, s0);
        c0 = fmla(ci11, T::R29_EVEN_TWIDDLE_1, c0);
        s0 = fmla(si11, T::R29_ODD_TWIDDLE_8, s0);
        c0 = fmla(ci12, T::R29_EVEN_TWIDDLE_13, c0);
        s0 = fmla(si12, -T::R29_ODD_TWIDDLE_10, s0);
        c0 = fmla(ci13, T::R29_EVEN_TWIDDLE_0, c0);
        s0 = fmla(si13, T::R29_ODD_TWIDDLE_12, s0);
        let dc = c0 + a0;
        data[28] = dc;
        data[27] = -s0;
    }
}

define_in_place_butterfly!(Dct2Butterfly29, 29);

pub(crate) trait MixedRadix31Sample {
    const R31_EVEN_TWIDDLE_0: Self;
    const R31_EVEN_TWIDDLE_1: Self;
    const R31_EVEN_TWIDDLE_2: Self;
    const R31_EVEN_TWIDDLE_3: Self;
    const R31_EVEN_TWIDDLE_4: Self;
    const R31_EVEN_TWIDDLE_5: Self;
    const R31_EVEN_TWIDDLE_6: Self;
    const R31_EVEN_TWIDDLE_7: Self;
    const R31_EVEN_TWIDDLE_8: Self;
    const R31_EVEN_TWIDDLE_9: Self;
    const R31_EVEN_TWIDDLE_10: Self;
    const R31_EVEN_TWIDDLE_11: Self;
    const R31_EVEN_TWIDDLE_12: Self;
    const R31_EVEN_TWIDDLE_13: Self;
    const R31_EVEN_TWIDDLE_14: Self;
    const R31_ODD_TWIDDLE_0: Self;
    const R31_ODD_TWIDDLE_1: Self;
    const R31_ODD_TWIDDLE_2: Self;
    const R31_ODD_TWIDDLE_3: Self;
    const R31_ODD_TWIDDLE_4: Self;
    const R31_ODD_TWIDDLE_5: Self;
    const R31_ODD_TWIDDLE_6: Self;
    const R31_ODD_TWIDDLE_7: Self;
    const R31_ODD_TWIDDLE_8: Self;
    const R31_ODD_TWIDDLE_9: Self;
    const R31_ODD_TWIDDLE_10: Self;
    const R31_ODD_TWIDDLE_11: Self;
    const R31_ODD_TWIDDLE_12: Self;
    const R31_ODD_TWIDDLE_13: Self;
    const R31_ODD_TWIDDLE_14: Self;
}

impl MixedRadix31Sample for f32 {
    const R31_EVEN_TWIDDLE_0: f32 = f32::from_bits(0xbf7eafc2);
    const R31_EVEN_TWIDDLE_1: f32 = f32::from_bits(0xbf744278);
    const R31_EVEN_TWIDDLE_2: f32 = f32::from_bits(0xbf5fd52e);
    const R31_EVEN_TWIDDLE_3: f32 = f32::from_bits(0xbf423df9);
    const R31_EVEN_TWIDDLE_4: f32 = f32::from_bits(0xbf1cb2fa);
    const R31_EVEN_TWIDDLE_5: f32 = f32::from_bits(0xbee17b58);
    const R31_EVEN_TWIDDLE_6: f32 = f32::from_bits(0xbe805587);
    const R31_EVEN_TWIDDLE_7: f32 = f32::from_bits(0xbd4f7581);
    const R31_EVEN_TWIDDLE_8: f32 = f32::from_bits(0x3e1b0fe2);
    const R31_EVEN_TWIDDLE_9: f32 = f32::from_bits(0x3eb1d1fe);
    const R31_EVEN_TWIDDLE_10: f32 = f32::from_bits(0x3f076a2f);
    const R31_EVEN_TWIDDLE_11: f32 = f32::from_bits(0x3f306023);
    const R31_EVEN_TWIDDLE_12: f32 = f32::from_bits(0x3f521d8e);
    const R31_EVEN_TWIDDLE_13: f32 = f32::from_bits(0x3f6b40d2);
    const R31_EVEN_TWIDDLE_14: f32 = f32::from_bits(0x3f7ac279);
    const R31_ODD_TWIDDLE_0: f32 = f32::from_bits(0x3f7fabe3);
    const R31_ODD_TWIDDLE_1: f32 = f32::from_bits(0x3f7d0c43);
    const R31_ODD_TWIDDLE_2: f32 = f32::from_bits(0x3f77d3e7);
    const R31_ODD_TWIDDLE_3: f32 = f32::from_bits(0x3f701086);
    const R31_ODD_TWIDDLE_4: f32 = f32::from_bits(0x3f65d685);
    const R31_ODD_TWIDDLE_5: f32 = f32::from_bits(0x3f5940c0);
    const R31_ODD_TWIDDLE_6: f32 = f32::from_bits(0x3f4a7047);
    const R31_ODD_TWIDDLE_7: f32 = f32::from_bits(0x3f398c05);
    const R31_ODD_TWIDDLE_8: f32 = f32::from_bits(0x3f26c059);
    const R31_ODD_TWIDDLE_9: f32 = f32::from_bits(0x3f123ea2);
    const R31_ODD_TWIDDLE_10: f32 = f32::from_bits(0x3ef87980);
    const R31_ODD_TWIDDLE_11: f32 = f32::from_bits(0x3ec9e903);
    const R31_ODD_TWIDDLE_12: f32 = f32::from_bits(0x3e994620);
    const R31_ODD_TWIDDLE_13: f32 = f32::from_bits(0x3e4e2133);
    const R31_ODD_TWIDDLE_14: f32 = f32::from_bits(0x3dcf3156);
}

impl MixedRadix31Sample for f64 {
    const R31_EVEN_TWIDDLE_0: f64 = f64::from_bits(0xbfefd5f830f860f9);
    const R31_EVEN_TWIDDLE_1: f64 = f64::from_bits(0xbfee884f0cc22ccc);
    const R31_EVEN_TWIDDLE_2: f64 = f64::from_bits(0xbfebfaa5c136b225);
    const R31_EVEN_TWIDDLE_3: f64 = f64::from_bits(0xbfe847bf1d5146cc);
    const R31_EVEN_TWIDDLE_4: f64 = f64::from_bits(0xbfe3965f49174d14);
    const R31_EVEN_TWIDDLE_5: f64 = f64::from_bits(0xbfdc2f6af3928a8e);
    const R31_EVEN_TWIDDLE_6: f64 = f64::from_bits(0xbfd00ab0eb2d7d96);
    const R31_EVEN_TWIDDLE_7: f64 = f64::from_bits(0xbfa9eeb01776b577);
    const R31_EVEN_TWIDDLE_8: f64 = f64::from_bits(0x3fc361fc440b4790);
    const R31_EVEN_TWIDDLE_9: f64 = f64::from_bits(0x3fd63a3fcfaca413);
    const R31_EVEN_TWIDDLE_10: f64 = f64::from_bits(0x3fe0ed45eea3b09f);
    const R31_EVEN_TWIDDLE_11: f64 = f64::from_bits(0x3fe60c045a2e972a);
    const R31_EVEN_TWIDDLE_12: f64 = f64::from_bits(0x3fea43b1b1379aff);
    const R31_EVEN_TWIDDLE_13: f64 = f64::from_bits(0x3fed681a366a00fa);
    const R31_EVEN_TWIDDLE_14: f64 = f64::from_bits(0x3fef584f2ce43b84);
    const R31_ODD_TWIDDLE_0: f64 = f64::from_bits(0x3feff57c5208ccf9);
    const R31_ODD_TWIDDLE_1: f64 = f64::from_bits(0x3fefa18852c3e08a);
    const R31_ODD_TWIDDLE_2: f64 = f64::from_bits(0x3feefa7cddb128fa);
    const R31_ODD_TWIDDLE_3: f64 = f64::from_bits(0x3fee0210c26a6e6f);
    const R31_ODD_TWIDDLE_4: f64 = f64::from_bits(0x3fecbad095f50378);
    const R31_ODD_TWIDDLE_5: f64 = f64::from_bits(0x3feb2818007c19df);
    const R31_ODD_TWIDDLE_6: f64 = f64::from_bits(0x3fe94e08eb13c452);
    const R31_ODD_TWIDDLE_7: f64 = f64::from_bits(0x3fe73180a4b0d300);
    const R31_ODD_TWIDDLE_8: f64 = f64::from_bits(0x3fe4d80b1ad9ccf5);
    const R31_ODD_TWIDDLE_9: f64 = f64::from_bits(0x3fe247d447a27216);
    const R31_ODD_TWIDDLE_10: f64 = f64::from_bits(0x3fdf0f2ff6705beb);
    const R31_ODD_TWIDDLE_11: f64 = f64::from_bits(0x3fd93d20572ca90b);
    const R31_ODD_TWIDDLE_12: f64 = f64::from_bits(0x3fd328c3f1b322cb);
    const R31_ODD_TWIDDLE_13: f64 = f64::from_bits(0x3fc9c4266041ca8e);
    const R31_ODD_TWIDDLE_14: f64 = f64::from_bits(0x3fb9e62aca53c49e);
}

#[derive(Debug, Clone)]
pub(crate) struct Dct2Butterfly31<T: DctSample> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T: DctSample> Default for Dct2Butterfly31<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: DctSample + MixedRadix31Sample> Dct2Butterfly31<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        // This is autogenerated Radix-Q where Q = 31.
        let a_buffer = [data[15]];
        let c_buffer = [
            data[0] + data[30],
            data[1] + data[29],
            data[2] + data[28],
            data[3] + data[27],
            data[4] + data[26],
            data[5] + data[25],
            data[6] + data[24],
            data[7] + data[23],
            data[8] + data[22],
            data[9] + data[21],
            data[10] + data[20],
            data[11] + data[19],
            data[12] + data[18],
            data[13] + data[17],
            data[14] + data[16],
        ];
        let s_buffer = [
            data[0] - data[30],
            data[1] - data[29],
            data[2] - data[28],
            data[3] - data[27],
            data[4] - data[26],
            data[5] - data[25],
            data[6] - data[24],
            data[7] - data[23],
            data[8] - data[22],
            data[9] - data[21],
            data[10] - data[20],
            data[11] - data[19],
            data[12] - data[18],
            data[13] - data[17],
            data[14] - data[16],
        ];
        let qc = c_buffer[0];
        let mut c0 = qc;
        let mut c1 = qc * T::R31_EVEN_TWIDDLE_0;
        let mut s0 = s_buffer[0];
        s0 *= T::R31_ODD_TWIDDLE_0;
        let ci1 = c_buffer[1];
        let si1 = s_buffer[1];
        c0 = ci1 + c0;
        c1 = ci1 * T::R31_EVEN_TWIDDLE_1 + c1;
        s0 = fmla(si1, T::R31_ODD_TWIDDLE_1, s0);
        let ci2 = c_buffer[2];
        let si2 = s_buffer[2];
        c0 = ci2 + c0;
        c1 = ci2 * T::R31_EVEN_TWIDDLE_2 + c1;
        s0 = fmla(si2, T::R31_ODD_TWIDDLE_2, s0);
        let ci3 = c_buffer[3];
        let si3 = s_buffer[3];
        c0 = ci3 + c0;
        c1 = ci3 * T::R31_EVEN_TWIDDLE_3 + c1;
        s0 = fmla(si3, T::R31_ODD_TWIDDLE_3, s0);
        let ci4 = c_buffer[4];
        let si4 = s_buffer[4];
        c0 = ci4 + c0;
        c1 = ci4 * T::R31_EVEN_TWIDDLE_4 + c1;
        s0 = fmla(si4, T::R31_ODD_TWIDDLE_4, s0);
        let ci5 = c_buffer[5];
        let si5 = s_buffer[5];
        c0 = ci5 + c0;
        c1 = ci5 * T::R31_EVEN_TWIDDLE_5 + c1;
        s0 = fmla(si5, T::R31_ODD_TWIDDLE_5, s0);
        let ci6 = c_buffer[6];
        let si6 = s_buffer[6];
        c0 = ci6 + c0;
        c1 = ci6 * T::R31_EVEN_TWIDDLE_6 + c1;
        s0 = fmla(si6, T::R31_ODD_TWIDDLE_6, s0);
        let ci7 = c_buffer[7];
        let si7 = s_buffer[7];
        c0 = ci7 + c0;
        c1 = ci7 * T::R31_EVEN_TWIDDLE_7 + c1;
        s0 = fmla(si7, T::R31_ODD_TWIDDLE_7, s0);
        let ci8 = c_buffer[8];
        let si8 = s_buffer[8];
        c0 = ci8 + c0;
        c1 = ci8 * T::R31_EVEN_TWIDDLE_8 + c1;
        s0 = fmla(si8, T::R31_ODD_TWIDDLE_8, s0);
        let ci9 = c_buffer[9];
        let si9 = s_buffer[9];
        c0 = ci9 + c0;
        c1 = ci9 * T::R31_EVEN_TWIDDLE_9 + c1;
        s0 = fmla(si9, T::R31_ODD_TWIDDLE_9, s0);
        let ci10 = c_buffer[10];
        let si10 = s_buffer[10];
        c0 = ci10 + c0;
        c1 = ci10 * T::R31_EVEN_TWIDDLE_10 + c1;
        s0 = fmla(si10, T::R31_ODD_TWIDDLE_10, s0);
        let ci11 = c_buffer[11];
        let si11 = s_buffer[11];
        c0 = ci11 + c0;
        c1 = ci11 * T::R31_EVEN_TWIDDLE_11 + c1;
        s0 = fmla(si11, T::R31_ODD_TWIDDLE_11, s0);
        let ci12 = c_buffer[12];
        let si12 = s_buffer[12];
        c0 = ci12 + c0;
        c1 = ci12 * T::R31_EVEN_TWIDDLE_12 + c1;
        s0 = fmla(si12, T::R31_ODD_TWIDDLE_12, s0);
        let ci13 = c_buffer[13];
        let si13 = s_buffer[13];
        c0 = ci13 + c0;
        c1 = ci13 * T::R31_EVEN_TWIDDLE_13 + c1;
        s0 = fmla(si13, T::R31_ODD_TWIDDLE_13, s0);
        let ci14 = c_buffer[14];
        let si14 = s_buffer[14];
        c0 = ci14 + c0;
        c1 = ci14 * T::R31_EVEN_TWIDDLE_14 + c1;
        s0 = fmla(si14, T::R31_ODD_TWIDDLE_14, s0);
        let a0 = a_buffer[0];
        let dc = c0 + a0;
        data[0] = dc;
        data[1] = s0;
        data[2] = -(c1 + a0);
        let mut c0 = qc * T::R31_EVEN_TWIDDLE_14;
        let mut s0 = s_buffer[0];
        s0 *= -T::R31_ODD_TWIDDLE_1;
        c0 = fmla(ci1, T::R31_EVEN_TWIDDLE_12, c0);
        s0 = fmla(si1, -T::R31_ODD_TWIDDLE_4, s0);
        c0 = fmla(ci2, T::R31_EVEN_TWIDDLE_10, c0);
        s0 = fmla(si2, -T::R31_ODD_TWIDDLE_7, s0);
        c0 = fmla(ci3, T::R31_EVEN_TWIDDLE_8, c0);
        s0 = fmla(si3, -T::R31_ODD_TWIDDLE_10, s0);
        c0 = fmla(ci4, T::R31_EVEN_TWIDDLE_6, c0);
        s0 = fmla(si4, -T::R31_ODD_TWIDDLE_13, s0);
        c0 = fmla(ci5, T::R31_EVEN_TWIDDLE_4, c0);
        s0 = fmla(si5, T::R31_ODD_TWIDDLE_14, s0);
        c0 = fmla(ci6, T::R31_EVEN_TWIDDLE_2, c0);
        s0 = fmla(si6, T::R31_ODD_TWIDDLE_11, s0);
        c0 = fmla(ci7, T::R31_EVEN_TWIDDLE_0, c0);
        s0 = fmla(si7, T::R31_ODD_TWIDDLE_8, s0);
        c0 = fmla(ci8, T::R31_EVEN_TWIDDLE_1, c0);
        s0 = fmla(si8, T::R31_ODD_TWIDDLE_5, s0);
        c0 = fmla(ci9, T::R31_EVEN_TWIDDLE_3, c0);
        s0 = fmla(si9, T::R31_ODD_TWIDDLE_2, s0);
        c0 = fmla(ci10, T::R31_EVEN_TWIDDLE_5, c0);
        s0 = fmla(si10, T::R31_ODD_TWIDDLE_0, s0);
        c0 = fmla(ci11, T::R31_EVEN_TWIDDLE_7, c0);
        s0 = fmla(si11, T::R31_ODD_TWIDDLE_3, s0);
        c0 = fmla(ci12, T::R31_EVEN_TWIDDLE_9, c0);
        s0 = fmla(si12, T::R31_ODD_TWIDDLE_6, s0);
        c0 = fmla(ci13, T::R31_EVEN_TWIDDLE_11, c0);
        s0 = fmla(si13, T::R31_ODD_TWIDDLE_9, s0);
        c0 = fmla(ci14, T::R31_EVEN_TWIDDLE_13, c0);
        s0 = fmla(si14, T::R31_ODD_TWIDDLE_12, s0);
        let dc = c0 + a0;
        data[4] = dc;
        data[3] = -s0;
        let mut c0 = qc * T::R31_EVEN_TWIDDLE_1;
        let mut s0 = s_buffer[0];
        s0 *= T::R31_ODD_TWIDDLE_2;
        c0 = fmla(ci1, T::R31_EVEN_TWIDDLE_4, c0);
        s0 = fmla(si1, T::R31_ODD_TWIDDLE_7, s0);
        c0 = fmla(ci2, T::R31_EVEN_TWIDDLE_7, c0);
        s0 = fmla(si2, T::R31_ODD_TWIDDLE_12, s0);
        c0 = fmla(ci3, T::R31_EVEN_TWIDDLE_10, c0);
        s0 = fmla(si3, -T::R31_ODD_TWIDDLE_13, s0);
        c0 = fmla(ci4, T::R31_EVEN_TWIDDLE_13, c0);
        s0 = fmla(si4, -T::R31_ODD_TWIDDLE_8, s0);
        c0 = fmla(ci5, T::R31_EVEN_TWIDDLE_14, c0);
        s0 = fmla(si5, -T::R31_ODD_TWIDDLE_3, s0);
        c0 = fmla(ci6, T::R31_EVEN_TWIDDLE_11, c0);
        s0 = fmla(si6, -T::R31_ODD_TWIDDLE_1, s0);
        c0 = fmla(ci7, T::R31_EVEN_TWIDDLE_8, c0);
        s0 = fmla(si7, -T::R31_ODD_TWIDDLE_6, s0);
        c0 = fmla(ci8, T::R31_EVEN_TWIDDLE_5, c0);
        s0 = fmla(si8, -T::R31_ODD_TWIDDLE_11, s0);
        c0 = fmla(ci9, T::R31_EVEN_TWIDDLE_2, c0);
        s0 = fmla(si9, T::R31_ODD_TWIDDLE_14, s0);
        c0 = fmla(ci10, T::R31_EVEN_TWIDDLE_0, c0);
        s0 = fmla(si10, T::R31_ODD_TWIDDLE_9, s0);
        c0 = fmla(ci11, T::R31_EVEN_TWIDDLE_3, c0);
        s0 = fmla(si11, T::R31_ODD_TWIDDLE_4, s0);
        c0 = fmla(ci12, T::R31_EVEN_TWIDDLE_6, c0);
        s0 = fmla(si12, T::R31_ODD_TWIDDLE_0, s0);
        c0 = fmla(ci13, T::R31_EVEN_TWIDDLE_9, c0);
        s0 = fmla(si13, T::R31_ODD_TWIDDLE_5, s0);
        c0 = fmla(ci14, T::R31_EVEN_TWIDDLE_12, c0);
        s0 = fmla(si14, T::R31_ODD_TWIDDLE_10, s0);
        let dc = c0 + a0;
        data[6] = -dc;
        data[5] = s0;
        let mut c0 = qc * T::R31_EVEN_TWIDDLE_13;
        let mut s0 = s_buffer[0];
        s0 *= -T::R31_ODD_TWIDDLE_3;
        c0 = fmla(ci1, T::R31_EVEN_TWIDDLE_9, c0);
        s0 = fmla(si1, -T::R31_ODD_TWIDDLE_10, s0);
        c0 = fmla(ci2, T::R31_EVEN_TWIDDLE_5, c0);
        s0 = fmla(si2, T::R31_ODD_TWIDDLE_13, s0);
        c0 = fmla(ci3, T::R31_EVEN_TWIDDLE_1, c0);
        s0 = fmla(si3, T::R31_ODD_TWIDDLE_6, s0);
        c0 = fmla(ci4, T::R31_EVEN_TWIDDLE_2, c0);
        s0 = fmla(si4, T::R31_ODD_TWIDDLE_0, s0);
        c0 = fmla(ci5, T::R31_EVEN_TWIDDLE_6, c0);
        s0 = fmla(si5, T::R31_ODD_TWIDDLE_7, s0);
        c0 = fmla(ci6, T::R31_EVEN_TWIDDLE_10, c0);
        s0 = fmla(si6, T::R31_ODD_TWIDDLE_14, s0);
        c0 = fmla(ci7, T::R31_EVEN_TWIDDLE_14, c0);
        s0 = fmla(si7, -T::R31_ODD_TWIDDLE_9, s0);
        c0 = fmla(ci8, T::R31_EVEN_TWIDDLE_12, c0);
        s0 = fmla(si8, -T::R31_ODD_TWIDDLE_2, s0);
        c0 = fmla(ci9, T::R31_EVEN_TWIDDLE_8, c0);
        s0 = fmla(si9, -T::R31_ODD_TWIDDLE_4, s0);
        c0 = fmla(ci10, T::R31_EVEN_TWIDDLE_4, c0);
        s0 = fmla(si10, -T::R31_ODD_TWIDDLE_11, s0);
        c0 = fmla(ci11, T::R31_EVEN_TWIDDLE_0, c0);
        s0 = fmla(si11, T::R31_ODD_TWIDDLE_12, s0);
        c0 = fmla(ci12, T::R31_EVEN_TWIDDLE_3, c0);
        s0 = fmla(si12, T::R31_ODD_TWIDDLE_5, s0);
        c0 = fmla(ci13, T::R31_EVEN_TWIDDLE_7, c0);
        s0 = fmla(si13, T::R31_ODD_TWIDDLE_1, s0);
        c0 = fmla(ci14, T::R31_EVEN_TWIDDLE_11, c0);
        s0 = fmla(si14, T::R31_ODD_TWIDDLE_8, s0);
        let dc = c0 + a0;
        data[8] = dc;
        data[7] = -s0;
        let mut c0 = qc * T::R31_EVEN_TWIDDLE_2;
        let mut s0 = s_buffer[0];
        s0 *= T::R31_ODD_TWIDDLE_4;
        c0 = fmla(ci1, T::R31_EVEN_TWIDDLE_7, c0);
        s0 = fmla(si1, T::R31_ODD_TWIDDLE_13, s0);
        c0 = fmla(ci2, T::R31_EVEN_TWIDDLE_12, c0);
        s0 = fmla(si2, -T::R31_ODD_TWIDDLE_8, s0);
        c0 = fmla(ci3, T::R31_EVEN_TWIDDLE_13, c0);
        s0 = fmla(si3, -T::R31_ODD_TWIDDLE_0, s0);
        c0 = fmla(ci4, T::R31_EVEN_TWIDDLE_8, c0);
        s0 = fmla(si4, -T::R31_ODD_TWIDDLE_9, s0);
        c0 = fmla(ci5, T::R31_EVEN_TWIDDLE_3, c0);
        s0 = fmla(si5, T::R31_ODD_TWIDDLE_12, s0);
        c0 = fmla(ci6, T::R31_EVEN_TWIDDLE_1, c0);
        s0 = fmla(si6, T::R31_ODD_TWIDDLE_3, s0);
        c0 = fmla(ci7, T::R31_EVEN_TWIDDLE_6, c0);
        s0 = fmla(si7, T::R31_ODD_TWIDDLE_5, s0);
        c0 = fmla(ci8, T::R31_EVEN_TWIDDLE_11, c0);
        s0 = fmla(si8, T::R31_ODD_TWIDDLE_14, s0);
        c0 = fmla(ci9, T::R31_EVEN_TWIDDLE_14, c0);
        s0 = fmla(si9, -T::R31_ODD_TWIDDLE_7, s0);
        c0 = fmla(ci10, T::R31_EVEN_TWIDDLE_9, c0);
        s0 = fmla(si10, -T::R31_ODD_TWIDDLE_1, s0);
        c0 = fmla(ci11, T::R31_EVEN_TWIDDLE_4, c0);
        s0 = fmla(si11, -T::R31_ODD_TWIDDLE_10, s0);
        c0 = fmla(ci12, T::R31_EVEN_TWIDDLE_0, c0);
        s0 = fmla(si12, T::R31_ODD_TWIDDLE_11, s0);
        c0 = fmla(ci13, T::R31_EVEN_TWIDDLE_5, c0);
        s0 = fmla(si13, T::R31_ODD_TWIDDLE_2, s0);
        c0 = fmla(ci14, T::R31_EVEN_TWIDDLE_10, c0);
        s0 = fmla(si14, T::R31_ODD_TWIDDLE_6, s0);
        let dc = c0 + a0;
        data[10] = -dc;
        data[9] = s0;
        let mut c0 = qc * T::R31_EVEN_TWIDDLE_12;
        let mut s0 = s_buffer[0];
        s0 *= -T::R31_ODD_TWIDDLE_5;
        c0 = fmla(ci1, T::R31_EVEN_TWIDDLE_6, c0);
        s0 = fmla(si1, T::R31_ODD_TWIDDLE_14, s0);
        c0 = fmla(ci2, T::R31_EVEN_TWIDDLE_0, c0);
        s0 = fmla(si2, T::R31_ODD_TWIDDLE_3, s0);
        c0 = fmla(ci3, T::R31_EVEN_TWIDDLE_5, c0);
        s0 = fmla(si3, T::R31_ODD_TWIDDLE_7, s0);
        c0 = fmla(ci4, T::R31_EVEN_TWIDDLE_11, c0);
        s0 = fmla(si4, -T::R31_ODD_TWIDDLE_12, s0);
        c0 = fmla(ci5, T::R31_EVEN_TWIDDLE_13, c0);
        s0 = fmla(si5, -T::R31_ODD_TWIDDLE_1, s0);
        c0 = fmla(ci6, T::R31_EVEN_TWIDDLE_7, c0);
        s0 = fmla(si6, -T::R31_ODD_TWIDDLE_9, s0);
        c0 = fmla(ci7, T::R31_EVEN_TWIDDLE_1, c0);
        s0 = fmla(si7, T::R31_ODD_TWIDDLE_10, s0);
        c0 = fmla(ci8, T::R31_EVEN_TWIDDLE_4, c0);
        s0 = fmla(si8, T::R31_ODD_TWIDDLE_0, s0);
        c0 = fmla(ci9, T::R31_EVEN_TWIDDLE_10, c0);
        s0 = fmla(si9, T::R31_ODD_TWIDDLE_11, s0);
        c0 = fmla(ci10, T::R31_EVEN_TWIDDLE_14, c0);
        s0 = fmla(si10, -T::R31_ODD_TWIDDLE_8, s0);
        c0 = fmla(ci11, T::R31_EVEN_TWIDDLE_8, c0);
        s0 = fmla(si11, -T::R31_ODD_TWIDDLE_2, s0);
        c0 = fmla(ci12, T::R31_EVEN_TWIDDLE_2, c0);
        s0 = fmla(si12, -T::R31_ODD_TWIDDLE_13, s0);
        c0 = fmla(ci13, T::R31_EVEN_TWIDDLE_3, c0);
        s0 = fmla(si13, T::R31_ODD_TWIDDLE_6, s0);
        c0 = fmla(ci14, T::R31_EVEN_TWIDDLE_9, c0);
        s0 = fmla(si14, T::R31_ODD_TWIDDLE_4, s0);
        let dc = c0 + a0;
        data[12] = dc;
        data[11] = -s0;
        let mut c0 = qc * T::R31_EVEN_TWIDDLE_3;
        let mut s0 = s_buffer[0];
        s0 *= T::R31_ODD_TWIDDLE_6;
        c0 = fmla(ci1, T::R31_EVEN_TWIDDLE_10, c0);
        s0 = fmla(si1, -T::R31_ODD_TWIDDLE_11, s0);
        c0 = fmla(ci2, T::R31_EVEN_TWIDDLE_13, c0);
        s0 = fmla(si2, -T::R31_ODD_TWIDDLE_1, s0);
        c0 = fmla(ci3, T::R31_EVEN_TWIDDLE_6, c0);
        s0 = fmla(si3, -T::R31_ODD_TWIDDLE_14, s0);
        c0 = fmla(ci4, T::R31_EVEN_TWIDDLE_0, c0);
        s0 = fmla(si4, T::R31_ODD_TWIDDLE_3, s0);
        c0 = fmla(ci5, T::R31_EVEN_TWIDDLE_7, c0);
        s0 = fmla(si5, T::R31_ODD_TWIDDLE_9, s0);
        c0 = fmla(ci6, T::R31_EVEN_TWIDDLE_14, c0);
        s0 = fmla(si6, -T::R31_ODD_TWIDDLE_8, s0);
        c0 = fmla(ci7, T::R31_EVEN_TWIDDLE_9, c0);
        s0 = fmla(si7, -T::R31_ODD_TWIDDLE_4, s0);
        c0 = fmla(ci8, T::R31_EVEN_TWIDDLE_2, c0);
        s0 = fmla(si8, T::R31_ODD_TWIDDLE_13, s0);
        c0 = fmla(ci9, T::R31_EVEN_TWIDDLE_4, c0);
        s0 = fmla(si9, T::R31_ODD_TWIDDLE_0, s0);
        c0 = fmla(ci10, T::R31_EVEN_TWIDDLE_11, c0);
        s0 = fmla(si10, T::R31_ODD_TWIDDLE_12, s0);
        c0 = fmla(ci11, T::R31_EVEN_TWIDDLE_12, c0);
        s0 = fmla(si11, -T::R31_ODD_TWIDDLE_5, s0);
        c0 = fmla(ci12, T::R31_EVEN_TWIDDLE_5, c0);
        s0 = fmla(si12, -T::R31_ODD_TWIDDLE_7, s0);
        c0 = fmla(ci13, T::R31_EVEN_TWIDDLE_1, c0);
        s0 = fmla(si13, T::R31_ODD_TWIDDLE_10, s0);
        c0 = fmla(ci14, T::R31_EVEN_TWIDDLE_8, c0);
        s0 = fmla(si14, T::R31_ODD_TWIDDLE_2, s0);
        let dc = c0 + a0;
        data[14] = -dc;
        data[13] = s0;
        let mut c0 = qc * T::R31_EVEN_TWIDDLE_11;
        let mut s0 = s_buffer[0];
        s0 *= -T::R31_ODD_TWIDDLE_7;
        c0 = fmla(ci1, T::R31_EVEN_TWIDDLE_3, c0);
        s0 = fmla(si1, T::R31_ODD_TWIDDLE_8, s0);
        c0 = fmla(ci2, T::R31_EVEN_TWIDDLE_4, c0);
        s0 = fmla(si2, T::R31_ODD_TWIDDLE_6, s0);
        c0 = fmla(ci3, T::R31_EVEN_TWIDDLE_12, c0);
        s0 = fmla(si3, -T::R31_ODD_TWIDDLE_9, s0);
        c0 = fmla(ci4, T::R31_EVEN_TWIDDLE_10, c0);
        s0 = fmla(si4, -T::R31_ODD_TWIDDLE_5, s0);
        c0 = fmla(ci5, T::R31_EVEN_TWIDDLE_2, c0);
        s0 = fmla(si5, T::R31_ODD_TWIDDLE_10, s0);
        c0 = fmla(ci6, T::R31_EVEN_TWIDDLE_5, c0);
        s0 = fmla(si6, T::R31_ODD_TWIDDLE_4, s0);
        c0 = fmla(ci7, T::R31_EVEN_TWIDDLE_13, c0);
        s0 = fmla(si7, -T::R31_ODD_TWIDDLE_11, s0);
        c0 = fmla(ci8, T::R31_EVEN_TWIDDLE_9, c0);
        s0 = fmla(si8, -T::R31_ODD_TWIDDLE_3, s0);
        c0 = fmla(ci9, T::R31_EVEN_TWIDDLE_1, c0);
        s0 = fmla(si9, T::R31_ODD_TWIDDLE_12, s0);
        c0 = fmla(ci10, T::R31_EVEN_TWIDDLE_6, c0);
        s0 = fmla(si10, T::R31_ODD_TWIDDLE_2, s0);
        c0 = fmla(ci11, T::R31_EVEN_TWIDDLE_14, c0);
        s0 = fmla(si11, -T::R31_ODD_TWIDDLE_13, s0);
        c0 = fmla(ci12, T::R31_EVEN_TWIDDLE_8, c0);
        s0 = fmla(si12, -T::R31_ODD_TWIDDLE_1, s0);
        c0 = fmla(ci13, T::R31_EVEN_TWIDDLE_0, c0);
        s0 = fmla(si13, T::R31_ODD_TWIDDLE_14, s0);
        c0 = fmla(ci14, T::R31_EVEN_TWIDDLE_7, c0);
        s0 = fmla(si14, T::R31_ODD_TWIDDLE_0, s0);
        let dc = c0 + a0;
        data[16] = dc;
        data[15] = -s0;
        let mut c0 = qc * T::R31_EVEN_TWIDDLE_4;
        let mut s0 = s_buffer[0];
        s0 *= T::R31_ODD_TWIDDLE_8;
        c0 = fmla(ci1, T::R31_EVEN_TWIDDLE_13, c0);
        s0 = fmla(si1, -T::R31_ODD_TWIDDLE_5, s0);
        c0 = fmla(ci2, T::R31_EVEN_TWIDDLE_8, c0);
        s0 = fmla(si2, -T::R31_ODD_TWIDDLE_11, s0);
        c0 = fmla(ci3, T::R31_EVEN_TWIDDLE_0, c0);
        s0 = fmla(si3, T::R31_ODD_TWIDDLE_2, s0);
        c0 = fmla(ci4, T::R31_EVEN_TWIDDLE_9, c0);
        s0 = fmla(si4, T::R31_ODD_TWIDDLE_14, s0);
        c0 = fmla(ci5, T::R31_EVEN_TWIDDLE_12, c0);
        s0 = fmla(si5, -T::R31_ODD_TWIDDLE_0, s0);
        c0 = fmla(ci6, T::R31_EVEN_TWIDDLE_3, c0);
        s0 = fmla(si6, T::R31_ODD_TWIDDLE_13, s0);
        c0 = fmla(ci7, T::R31_EVEN_TWIDDLE_5, c0);
        s0 = fmla(si7, T::R31_ODD_TWIDDLE_3, s0);
        c0 = fmla(ci8, T::R31_EVEN_TWIDDLE_14, c0);
        s0 = fmla(si8, -T::R31_ODD_TWIDDLE_10, s0);
        c0 = fmla(ci9, T::R31_EVEN_TWIDDLE_7, c0);
        s0 = fmla(si9, -T::R31_ODD_TWIDDLE_6, s0);
        c0 = fmla(ci10, T::R31_EVEN_TWIDDLE_1, c0);
        s0 = fmla(si10, T::R31_ODD_TWIDDLE_7, s0);
        c0 = fmla(ci11, T::R31_EVEN_TWIDDLE_10, c0);
        s0 = fmla(si11, T::R31_ODD_TWIDDLE_9, s0);
        c0 = fmla(ci12, T::R31_EVEN_TWIDDLE_11, c0);
        s0 = fmla(si12, -T::R31_ODD_TWIDDLE_4, s0);
        c0 = fmla(ci13, T::R31_EVEN_TWIDDLE_2, c0);
        s0 = fmla(si13, -T::R31_ODD_TWIDDLE_12, s0);
        c0 = fmla(ci14, T::R31_EVEN_TWIDDLE_6, c0);
        s0 = fmla(si14, T::R31_ODD_TWIDDLE_1, s0);
        let dc = c0 + a0;
        data[18] = -dc;
        data[17] = s0;
        let mut c0 = qc * T::R31_EVEN_TWIDDLE_10;
        let mut s0 = s_buffer[0];
        s0 *= -T::R31_ODD_TWIDDLE_9;
        c0 = fmla(ci1, T::R31_EVEN_TWIDDLE_0, c0);
        s0 = fmla(si1, T::R31_ODD_TWIDDLE_2, s0);
        c0 = fmla(ci2, T::R31_EVEN_TWIDDLE_9, c0);
        s0 = fmla(si2, -T::R31_ODD_TWIDDLE_14, s0);
        c0 = fmla(ci3, T::R31_EVEN_TWIDDLE_11, c0);
        s0 = fmla(si3, -T::R31_ODD_TWIDDLE_4, s0);
        c0 = fmla(ci4, T::R31_EVEN_TWIDDLE_1, c0);
        s0 = fmla(si4, T::R31_ODD_TWIDDLE_7, s0);
        c0 = fmla(ci5, T::R31_EVEN_TWIDDLE_8, c0);
        s0 = fmla(si5, T::R31_ODD_TWIDDLE_11, s0);
        c0 = fmla(ci6, T::R31_EVEN_TWIDDLE_12, c0);
        s0 = fmla(si6, -T::R31_ODD_TWIDDLE_0, s0);
        c0 = fmla(ci7, T::R31_EVEN_TWIDDLE_2, c0);
        s0 = fmla(si7, T::R31_ODD_TWIDDLE_12, s0);
        c0 = fmla(ci8, T::R31_EVEN_TWIDDLE_7, c0);
        s0 = fmla(si8, T::R31_ODD_TWIDDLE_6, s0);
        c0 = fmla(ci9, T::R31_EVEN_TWIDDLE_13, c0);
        s0 = fmla(si9, -T::R31_ODD_TWIDDLE_5, s0);
        c0 = fmla(ci10, T::R31_EVEN_TWIDDLE_3, c0);
        s0 = fmla(si10, -T::R31_ODD_TWIDDLE_13, s0);
        c0 = fmla(ci11, T::R31_EVEN_TWIDDLE_6, c0);
        s0 = fmla(si11, T::R31_ODD_TWIDDLE_1, s0);
        c0 = fmla(ci12, T::R31_EVEN_TWIDDLE_14, c0);
        s0 = fmla(si12, -T::R31_ODD_TWIDDLE_10, s0);
        c0 = fmla(ci13, T::R31_EVEN_TWIDDLE_4, c0);
        s0 = fmla(si13, -T::R31_ODD_TWIDDLE_8, s0);
        c0 = fmla(ci14, T::R31_EVEN_TWIDDLE_5, c0);
        s0 = fmla(si14, T::R31_ODD_TWIDDLE_3, s0);
        let dc = c0 + a0;
        data[20] = dc;
        data[19] = -s0;
        let mut c0 = qc * T::R31_EVEN_TWIDDLE_5;
        let mut s0 = s_buffer[0];
        s0 *= T::R31_ODD_TWIDDLE_10;
        c0 = fmla(ci1, T::R31_EVEN_TWIDDLE_14, c0);
        s0 = fmla(si1, -T::R31_ODD_TWIDDLE_0, s0);
        c0 = fmla(ci2, T::R31_EVEN_TWIDDLE_3, c0);
        s0 = fmla(si2, T::R31_ODD_TWIDDLE_9, s0);
        c0 = fmla(ci3, T::R31_EVEN_TWIDDLE_7, c0);
        s0 = fmla(si3, T::R31_ODD_TWIDDLE_11, s0);
        c0 = fmla(ci4, T::R31_EVEN_TWIDDLE_12, c0);
        s0 = fmla(si4, -T::R31_ODD_TWIDDLE_1, s0);
        c0 = fmla(ci5, T::R31_EVEN_TWIDDLE_1, c0);
        s0 = fmla(si5, T::R31_ODD_TWIDDLE_8, s0);
        c0 = fmla(ci6, T::R31_EVEN_TWIDDLE_9, c0);
        s0 = fmla(si6, T::R31_ODD_TWIDDLE_12, s0);
        c0 = fmla(ci7, T::R31_EVEN_TWIDDLE_10, c0);
        s0 = fmla(si7, -T::R31_ODD_TWIDDLE_2, s0);
        c0 = fmla(ci8, T::R31_EVEN_TWIDDLE_0, c0);
        s0 = fmla(si8, T::R31_ODD_TWIDDLE_7, s0);
        c0 = fmla(ci9, T::R31_EVEN_TWIDDLE_11, c0);
        s0 = fmla(si9, T::R31_ODD_TWIDDLE_13, s0);
        c0 = fmla(ci10, T::R31_EVEN_TWIDDLE_8, c0);
        s0 = fmla(si10, -T::R31_ODD_TWIDDLE_3, s0);
        c0 = fmla(ci11, T::R31_EVEN_TWIDDLE_2, c0);
        s0 = fmla(si11, T::R31_ODD_TWIDDLE_6, s0);
        c0 = fmla(ci12, T::R31_EVEN_TWIDDLE_13, c0);
        s0 = fmla(si12, T::R31_ODD_TWIDDLE_14, s0);
        c0 = fmla(ci13, T::R31_EVEN_TWIDDLE_6, c0);
        s0 = fmla(si13, -T::R31_ODD_TWIDDLE_4, s0);
        c0 = fmla(ci14, T::R31_EVEN_TWIDDLE_4, c0);
        s0 = fmla(si14, T::R31_ODD_TWIDDLE_5, s0);
        let dc = c0 + a0;
        data[22] = -dc;
        data[21] = s0;
        let mut c0 = qc * T::R31_EVEN_TWIDDLE_9;
        let mut s0 = s_buffer[0];
        s0 *= -T::R31_ODD_TWIDDLE_11;
        c0 = fmla(ci1, T::R31_EVEN_TWIDDLE_2, c0);
        s0 = fmla(si1, T::R31_ODD_TWIDDLE_3, s0);
        c0 = fmla(ci2, T::R31_EVEN_TWIDDLE_14, c0);
        s0 = fmla(si2, -T::R31_ODD_TWIDDLE_4, s0);
        c0 = fmla(ci3, T::R31_EVEN_TWIDDLE_4, c0);
        s0 = fmla(si3, T::R31_ODD_TWIDDLE_12, s0);
        c0 = fmla(ci4, T::R31_EVEN_TWIDDLE_7, c0);
        s0 = fmla(si4, T::R31_ODD_TWIDDLE_10, s0);
        c0 = fmla(ci5, T::R31_EVEN_TWIDDLE_11, c0);
        s0 = fmla(si5, -T::R31_ODD_TWIDDLE_2, s0);
        c0 = fmla(ci6, T::R31_EVEN_TWIDDLE_0, c0);
        s0 = fmla(si6, T::R31_ODD_TWIDDLE_5, s0);
        c0 = fmla(ci7, T::R31_EVEN_TWIDDLE_12, c0);
        s0 = fmla(si7, -T::R31_ODD_TWIDDLE_13, s0);
        c0 = fmla(ci8, T::R31_EVEN_TWIDDLE_6, c0);
        s0 = fmla(si8, -T::R31_ODD_TWIDDLE_9, s0);
        c0 = fmla(ci9, T::R31_EVEN_TWIDDLE_5, c0);
        s0 = fmla(si9, T::R31_ODD_TWIDDLE_1, s0);
        c0 = fmla(ci10, T::R31_EVEN_TWIDDLE_13, c0);
        s0 = fmla(si10, -T::R31_ODD_TWIDDLE_6, s0);
        c0 = fmla(ci11, T::R31_EVEN_TWIDDLE_1, c0);
        s0 = fmla(si11, T::R31_ODD_TWIDDLE_14, s0);
        c0 = fmla(ci12, T::R31_EVEN_TWIDDLE_10, c0);
        s0 = fmla(si12, T::R31_ODD_TWIDDLE_8, s0);
        c0 = fmla(ci13, T::R31_EVEN_TWIDDLE_8, c0);
        s0 = fmla(si13, -T::R31_ODD_TWIDDLE_0, s0);
        c0 = fmla(ci14, T::R31_EVEN_TWIDDLE_3, c0);
        s0 = fmla(si14, T::R31_ODD_TWIDDLE_7, s0);
        let dc = c0 + a0;
        data[24] = dc;
        data[23] = -s0;
        let mut c0 = qc * T::R31_EVEN_TWIDDLE_6;
        let mut s0 = s_buffer[0];
        s0 *= T::R31_ODD_TWIDDLE_12;
        c0 = fmla(ci1, T::R31_EVEN_TWIDDLE_11, c0);
        s0 = fmla(si1, -T::R31_ODD_TWIDDLE_6, s0);
        c0 = fmla(ci2, T::R31_EVEN_TWIDDLE_1, c0);
        s0 = fmla(si2, T::R31_ODD_TWIDDLE_0, s0);
        c0 = fmla(ci3, T::R31_EVEN_TWIDDLE_14, c0);
        s0 = fmla(si3, -T::R31_ODD_TWIDDLE_5, s0);
        c0 = fmla(ci4, T::R31_EVEN_TWIDDLE_3, c0);
        s0 = fmla(si4, T::R31_ODD_TWIDDLE_11, s0);
        c0 = fmla(ci5, T::R31_EVEN_TWIDDLE_9, c0);
        s0 = fmla(si5, T::R31_ODD_TWIDDLE_13, s0);
        c0 = fmla(ci6, T::R31_EVEN_TWIDDLE_8, c0);
        s0 = fmla(si6, -T::R31_ODD_TWIDDLE_7, s0);
        c0 = fmla(ci7, T::R31_EVEN_TWIDDLE_4, c0);
        s0 = fmla(si7, T::R31_ODD_TWIDDLE_1, s0);
        c0 = fmla(ci8, T::R31_EVEN_TWIDDLE_13, c0);
        s0 = fmla(si8, -T::R31_ODD_TWIDDLE_4, s0);
        c0 = fmla(ci9, T::R31_EVEN_TWIDDLE_0, c0);
        s0 = fmla(si9, T::R31_ODD_TWIDDLE_10, s0);
        c0 = fmla(ci10, T::R31_EVEN_TWIDDLE_12, c0);
        s0 = fmla(si10, T::R31_ODD_TWIDDLE_14, s0);
        c0 = fmla(ci11, T::R31_EVEN_TWIDDLE_5, c0);
        s0 = fmla(si11, -T::R31_ODD_TWIDDLE_8, s0);
        c0 = fmla(ci12, T::R31_EVEN_TWIDDLE_7, c0);
        s0 = fmla(si12, T::R31_ODD_TWIDDLE_2, s0);
        c0 = fmla(ci13, T::R31_EVEN_TWIDDLE_10, c0);
        s0 = fmla(si13, -T::R31_ODD_TWIDDLE_3, s0);
        c0 = fmla(ci14, T::R31_EVEN_TWIDDLE_2, c0);
        s0 = fmla(si14, T::R31_ODD_TWIDDLE_9, s0);
        let dc = c0 + a0;
        data[26] = -dc;
        data[25] = s0;
        let mut c0 = qc * T::R31_EVEN_TWIDDLE_8;
        let mut s0 = s_buffer[0];
        s0 *= -T::R31_ODD_TWIDDLE_13;
        c0 = fmla(ci1, T::R31_EVEN_TWIDDLE_5, c0);
        s0 = fmla(si1, T::R31_ODD_TWIDDLE_9, s0);
        c0 = fmla(ci2, T::R31_EVEN_TWIDDLE_11, c0);
        s0 = fmla(si2, -T::R31_ODD_TWIDDLE_5, s0);
        c0 = fmla(ci3, T::R31_EVEN_TWIDDLE_2, c0);
        s0 = fmla(si3, T::R31_ODD_TWIDDLE_1, s0);
        c0 = fmla(ci4, T::R31_EVEN_TWIDDLE_14, c0);
        s0 = fmla(si4, -T::R31_ODD_TWIDDLE_2, s0);
        c0 = fmla(ci5, T::R31_EVEN_TWIDDLE_0, c0);
        s0 = fmla(si5, T::R31_ODD_TWIDDLE_6, s0);
        c0 = fmla(ci6, T::R31_EVEN_TWIDDLE_13, c0);
        s0 = fmla(si6, -T::R31_ODD_TWIDDLE_10, s0);
        c0 = fmla(ci7, T::R31_EVEN_TWIDDLE_3, c0);
        s0 = fmla(si7, T::R31_ODD_TWIDDLE_14, s0);
        c0 = fmla(ci8, T::R31_EVEN_TWIDDLE_10, c0);
        s0 = fmla(si8, T::R31_ODD_TWIDDLE_12, s0);
        c0 = fmla(ci9, T::R31_EVEN_TWIDDLE_6, c0);
        s0 = fmla(si9, -T::R31_ODD_TWIDDLE_8, s0);
        c0 = fmla(ci10, T::R31_EVEN_TWIDDLE_7, c0);
        s0 = fmla(si10, T::R31_ODD_TWIDDLE_4, s0);
        c0 = fmla(ci11, T::R31_EVEN_TWIDDLE_9, c0);
        s0 = fmla(si11, -T::R31_ODD_TWIDDLE_0, s0);
        c0 = fmla(ci12, T::R31_EVEN_TWIDDLE_4, c0);
        s0 = fmla(si12, T::R31_ODD_TWIDDLE_3, s0);
        c0 = fmla(ci13, T::R31_EVEN_TWIDDLE_12, c0);
        s0 = fmla(si13, -T::R31_ODD_TWIDDLE_7, s0);
        c0 = fmla(ci14, T::R31_EVEN_TWIDDLE_1, c0);
        s0 = fmla(si14, T::R31_ODD_TWIDDLE_11, s0);
        let dc = c0 + a0;
        data[28] = dc;
        data[27] = -s0;
        let mut c0 = qc * T::R31_EVEN_TWIDDLE_7;
        let mut s0 = s_buffer[0];
        s0 *= T::R31_ODD_TWIDDLE_14;
        c0 = fmla(ci1, T::R31_EVEN_TWIDDLE_8, c0);
        s0 = fmla(si1, -T::R31_ODD_TWIDDLE_12, s0);
        c0 = fmla(ci2, T::R31_EVEN_TWIDDLE_6, c0);
        s0 = fmla(si2, T::R31_ODD_TWIDDLE_10, s0);
        c0 = fmla(ci3, T::R31_EVEN_TWIDDLE_9, c0);
        s0 = fmla(si3, -T::R31_ODD_TWIDDLE_8, s0);
        c0 = fmla(ci4, T::R31_EVEN_TWIDDLE_5, c0);
        s0 = fmla(si4, T::R31_ODD_TWIDDLE_6, s0);
        c0 = fmla(ci5, T::R31_EVEN_TWIDDLE_10, c0);
        s0 = fmla(si5, -T::R31_ODD_TWIDDLE_4, s0);
        c0 = fmla(ci6, T::R31_EVEN_TWIDDLE_4, c0);
        s0 = fmla(si6, T::R31_ODD_TWIDDLE_2, s0);
        c0 = fmla(ci7, T::R31_EVEN_TWIDDLE_11, c0);
        s0 = fmla(si7, -T::R31_ODD_TWIDDLE_0, s0);
        c0 = fmla(ci8, T::R31_EVEN_TWIDDLE_3, c0);
        s0 = fmla(si8, T::R31_ODD_TWIDDLE_1, s0);
        c0 = fmla(ci9, T::R31_EVEN_TWIDDLE_12, c0);
        s0 = fmla(si9, -T::R31_ODD_TWIDDLE_3, s0);
        c0 = fmla(ci10, T::R31_EVEN_TWIDDLE_2, c0);
        s0 = fmla(si10, T::R31_ODD_TWIDDLE_5, s0);
        c0 = fmla(ci11, T::R31_EVEN_TWIDDLE_13, c0);
        s0 = fmla(si11, -T::R31_ODD_TWIDDLE_7, s0);
        c0 = fmla(ci12, T::R31_EVEN_TWIDDLE_1, c0);
        s0 = fmla(si12, T::R31_ODD_TWIDDLE_9, s0);
        c0 = fmla(ci13, T::R31_EVEN_TWIDDLE_14, c0);
        s0 = fmla(si13, -T::R31_ODD_TWIDDLE_11, s0);
        c0 = fmla(ci14, T::R31_EVEN_TWIDDLE_0, c0);
        s0 = fmla(si14, T::R31_ODD_TWIDDLE_13, s0);
        let dc = c0 + a0;
        data[30] = -dc;
        data[29] = s0;
    }
}

impl<T: DctSample + MixedRadix31Sample> PxdctExecutor<T> for Dct2Butterfly31<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, data: &mut [T]) -> Result<(), PxdctError> {
        if !data.len().is_multiple_of(31) {
            return Err(PxdctError::InvalidSizeMultiplier(data.len(), self.length()));
        }
        for chunk in data.chunks_exact_mut(31) {
            self.exec(&mut InPlaceStore::new(chunk));
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
        validate_oof_sizes!(input, output, 31);
        use crate::bidirectional::BiStore;
        for (src, dst) in input.chunks_exact(31).zip(output.chunks_exact_mut(31)) {
            self.exec(&mut BiStore::new(src, dst));
        }
        Ok(())
    }

    fn length(&self) -> usize {
        31
    }

    fn scratch_size(&self) -> usize {
        0
    }
}

pub(crate) trait MixedRadix37Sample {
    const R37_EVEN_TWIDDLE_0: Self;
    const R37_EVEN_TWIDDLE_1: Self;
    const R37_EVEN_TWIDDLE_2: Self;
    const R37_EVEN_TWIDDLE_3: Self;
    const R37_EVEN_TWIDDLE_4: Self;
    const R37_EVEN_TWIDDLE_5: Self;
    const R37_EVEN_TWIDDLE_6: Self;
    const R37_EVEN_TWIDDLE_7: Self;
    const R37_EVEN_TWIDDLE_8: Self;
    const R37_EVEN_TWIDDLE_9: Self;
    const R37_EVEN_TWIDDLE_10: Self;
    const R37_EVEN_TWIDDLE_11: Self;
    const R37_EVEN_TWIDDLE_12: Self;
    const R37_EVEN_TWIDDLE_13: Self;
    const R37_EVEN_TWIDDLE_14: Self;
    const R37_EVEN_TWIDDLE_15: Self;
    const R37_EVEN_TWIDDLE_16: Self;
    const R37_EVEN_TWIDDLE_17: Self;
    const R37_ODD_TWIDDLE_0: Self;
    const R37_ODD_TWIDDLE_1: Self;
    const R37_ODD_TWIDDLE_2: Self;
    const R37_ODD_TWIDDLE_3: Self;
    const R37_ODD_TWIDDLE_4: Self;
    const R37_ODD_TWIDDLE_5: Self;
    const R37_ODD_TWIDDLE_6: Self;
    const R37_ODD_TWIDDLE_7: Self;
    const R37_ODD_TWIDDLE_8: Self;
    const R37_ODD_TWIDDLE_9: Self;
    const R37_ODD_TWIDDLE_10: Self;
    const R37_ODD_TWIDDLE_11: Self;
    const R37_ODD_TWIDDLE_12: Self;
    const R37_ODD_TWIDDLE_13: Self;
    const R37_ODD_TWIDDLE_14: Self;
    const R37_ODD_TWIDDLE_15: Self;
    const R37_ODD_TWIDDLE_16: Self;
    const R37_ODD_TWIDDLE_17: Self;
}

impl MixedRadix37Sample for f32 {
    const R37_EVEN_TWIDDLE_0: f32 = f32::from_bits(0xbf7f13e8);
    const R37_EVEN_TWIDDLE_1: f32 = f32::from_bits(0xbf77bd59);
    const R37_EVEN_TWIDDLE_2: f32 = f32::from_bits(0xbf694645);
    const R37_EVEN_TWIDDLE_3: f32 = f32::from_bits(0xbf541935);
    const R37_EVEN_TWIDDLE_4: f32 = f32::from_bits(0xbf38d21d);
    const R37_EVEN_TWIDDLE_5: f32 = f32::from_bits(0xbf1839e1);
    const R37_EVEN_TWIDDLE_6: f32 = f32::from_bits(0xbee6811b);
    const R37_EVEN_TWIDDLE_7: f32 = f32::from_bits(0xbe95ecde);
    const R37_EVEN_TWIDDLE_8: f32 = f32::from_bits(0xbe0210f6);
    const R37_EVEN_TWIDDLE_9: f32 = f32::from_bits(0x3d2dd6d4);
    const R37_EVEN_TWIDDLE_10: f32 = f32::from_bits(0x3e57bc4e);
    const R37_EVEN_TWIDDLE_11: f32 = f32::from_bits(0x3ebee70b);
    const R37_EVEN_TWIDDLE_12: f32 = f32::from_bits(0x3f063901);
    const R37_EVEN_TWIDDLE_13: f32 = f32::from_bits(0x3f2921fb);
    const R37_EVEN_TWIDDLE_14: f32 = f32::from_bits(0x3f472d5a);
    const R37_EVEN_TWIDDLE_15: f32 = f32::from_bits(0x3f5f7dda);
    const R37_EVEN_TWIDDLE_16: f32 = f32::from_bits(0x3f71606b);
    const R37_EVEN_TWIDDLE_17: f32 = f32::from_bits(0x3f7c5153);
    const R37_ODD_TWIDDLE_0: f32 = f32::from_bits(0x3f7fc4f3);
    const R37_ODD_TWIDDLE_1: f32 = f32::from_bits(0x3f7ded30);
    const R37_ODD_TWIDDLE_2: f32 = f32::from_bits(0x3f7a410f);
    const R37_ODD_TWIDDLE_3: f32 = f32::from_bits(0x3f74c758);
    const R37_ODD_TWIDDLE_4: f32 = f32::from_bits(0x3f6d8a23);
    const R37_ODD_TWIDDLE_5: f32 = f32::from_bits(0x3f6496ca);
    const R37_ODD_TWIDDLE_6: f32 = f32::from_bits(0x3f59fdd0);
    const R37_ODD_TWIDDLE_7: f32 = f32::from_bits(0x3f4dd2c2);
    const R37_ODD_TWIDDLE_8: f32 = f32::from_bits(0x3f402c0f);
    const R37_ODD_TWIDDLE_9: f32 = f32::from_bits(0x3f3122e8);
    const R37_ODD_TWIDDLE_10: f32 = f32::from_bits(0x3f20d307);
    const R37_ODD_TWIDDLE_11: f32 = f32::from_bits(0x3f0f5a82);
    const R37_ODD_TWIDDLE_12: f32 = f32::from_bits(0x3ef9b327);
    const R37_ODD_TWIDDLE_13: f32 = f32::from_bits(0x3ed2e4b8);
    const R37_ODD_TWIDDLE_14: f32 = f32::from_bits(0x3eaa914d);
    const R37_ODD_TWIDDLE_15: f32 = f32::from_bits(0x3e810345);
    const R37_ODD_TWIDDLE_16: f32 = f32::from_bits(0x3e2d0e8d);
    const R37_ODD_TWIDDLE_17: f32 = f32::from_bits(0x3dadaebb);
}

impl MixedRadix37Sample for f64 {
    const R37_EVEN_TWIDDLE_0: f64 = f64::from_bits(0xbfefe27cfc6369f9);
    const R37_EVEN_TWIDDLE_1: f64 = f64::from_bits(0xbfeef7ab15caea23);
    const R37_EVEN_TWIDDLE_2: f64 = f64::from_bits(0xbfed28c8a6acd0d8);
    const R37_EVEN_TWIDDLE_3: f64 = f64::from_bits(0xbfea8326aafd8708);
    const R37_EVEN_TWIDDLE_4: f64 = f64::from_bits(0xbfe71a43aa95d7bf);
    const R37_EVEN_TWIDDLE_5: f64 = f64::from_bits(0xbfe3073c27aafe10);
    const R37_EVEN_TWIDDLE_6: f64 = f64::from_bits(0xbfdcd0235e21d52f);
    const R37_EVEN_TWIDDLE_7: f64 = f64::from_bits(0xbfd2bd9bb88a0a33);
    const R37_EVEN_TWIDDLE_8: f64 = f64::from_bits(0xbfc0421eb2e93983);
    const R37_EVEN_TWIDDLE_9: f64 = f64::from_bits(0x3fa5bada7775f014);
    const R37_EVEN_TWIDDLE_10: f64 = f64::from_bits(0x3fcaf789cf49d487);
    const R37_EVEN_TWIDDLE_11: f64 = f64::from_bits(0x3fd7dce16a8ac365);
    const R37_EVEN_TWIDDLE_12: f64 = f64::from_bits(0x3fe0c720117dda63);
    const R37_EVEN_TWIDDLE_13: f64 = f64::from_bits(0x3fe5243f514822c8);
    const R37_EVEN_TWIDDLE_14: f64 = f64::from_bits(0x3fe8e5ab3cfd530e);
    const R37_EVEN_TWIDDLE_15: f64 = f64::from_bits(0x3febefbb4b1f28ee);
    const R37_EVEN_TWIDDLE_16: f64 = f64::from_bits(0x3fee2c0d520c8630);
    const R37_EVEN_TWIDDLE_17: f64 = f64::from_bits(0x3fef8a2a60ab8aa8);
    const R37_ODD_TWIDDLE_0: f64 = f64::from_bits(0x3feff89e652a20e7);
    const R37_ODD_TWIDDLE_1: f64 = f64::from_bits(0x3fefbda5fb4a66be);
    const R37_ODD_TWIDDLE_2: f64 = f64::from_bits(0x3fef4821ecaa0a78);
    const R37_ODD_TWIDDLE_3: f64 = f64::from_bits(0x3fee98eafae74d55);
    const R37_ODD_TWIDDLE_4: f64 = f64::from_bits(0x3fedb1445451c8c7);
    const R37_ODD_TWIDDLE_5: f64 = f64::from_bits(0x3fec92d93fd02a0a);
    const R37_ODD_TWIDDLE_6: f64 = f64::from_bits(0x3feb3fba08c4fb19);
    const R37_ODD_TWIDDLE_7: f64 = f64::from_bits(0x3fe9ba5830a01f9b);
    const R37_ODD_TWIDDLE_8: f64 = f64::from_bits(0x3fe80581ed225922);
    const R37_ODD_TWIDDLE_9: f64 = f64::from_bits(0x3fe6245cfba2df8f);
    const R37_ODD_TWIDDLE_10: f64 = f64::from_bits(0x3fe41a60d2e27eea);
    const R37_ODD_TWIDDLE_11: f64 = f64::from_bits(0x3fe1eb503e217548);
    const R37_ODD_TWIDDLE_12: f64 = f64::from_bits(0x3fdf3664da86a9df);
    const R37_ODD_TWIDDLE_13: f64 = f64::from_bits(0x3fda5c970d98edff);
    const R37_ODD_TWIDDLE_14: f64 = f64::from_bits(0x3fd5522992da7358);
    const R37_ODD_TWIDDLE_15: f64 = f64::from_bits(0x3fd02068973d5997);
    const R37_ODD_TWIDDLE_16: f64 = f64::from_bits(0x3fc5a1d1a223c71d);
    const R37_ODD_TWIDDLE_17: f64 = f64::from_bits(0x3fb5b5d750211d9a);
}

#[derive(Debug, Clone)]
pub(crate) struct Dct2Butterfly37<T: DctSample> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T: DctSample> Default for Dct2Butterfly37<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: DctSample + MixedRadix37Sample> Dct2Butterfly37<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        // This is autogenerated Radix-Q where Q = 37.
        let a_buffer = [data[18]];
        let c_buffer = [
            data[0] + data[36],
            data[1] + data[35],
            data[2] + data[34],
            data[3] + data[33],
            data[4] + data[32],
            data[5] + data[31],
            data[6] + data[30],
            data[7] + data[29],
            data[8] + data[28],
            data[9] + data[27],
            data[10] + data[26],
            data[11] + data[25],
            data[12] + data[24],
            data[13] + data[23],
            data[14] + data[22],
            data[15] + data[21],
            data[16] + data[20],
            data[17] + data[19],
        ];
        let s_buffer = [
            data[0] - data[36],
            data[1] - data[35],
            data[2] - data[34],
            data[3] - data[33],
            data[4] - data[32],
            data[5] - data[31],
            data[6] - data[30],
            data[7] - data[29],
            data[8] - data[28],
            data[9] - data[27],
            data[10] - data[26],
            data[11] - data[25],
            data[12] - data[24],
            data[13] - data[23],
            data[14] - data[22],
            data[15] - data[21],
            data[16] - data[20],
            data[17] - data[19],
        ];
        let qc = c_buffer[0];
        let mut c0 = qc;
        let mut c1 = qc * T::R37_EVEN_TWIDDLE_0;
        let mut s0 = s_buffer[0];
        s0 *= T::R37_ODD_TWIDDLE_0;
        let ci1 = c_buffer[1];
        let si1 = s_buffer[1];
        c0 = ci1 + c0;
        c1 = ci1 * T::R37_EVEN_TWIDDLE_1 + c1;
        s0 = fmla(si1, T::R37_ODD_TWIDDLE_1, s0);
        let ci2 = c_buffer[2];
        let si2 = s_buffer[2];
        c0 = ci2 + c0;
        c1 = ci2 * T::R37_EVEN_TWIDDLE_2 + c1;
        s0 = fmla(si2, T::R37_ODD_TWIDDLE_2, s0);
        let ci3 = c_buffer[3];
        let si3 = s_buffer[3];
        c0 = ci3 + c0;
        c1 = ci3 * T::R37_EVEN_TWIDDLE_3 + c1;
        s0 = fmla(si3, T::R37_ODD_TWIDDLE_3, s0);
        let ci4 = c_buffer[4];
        let si4 = s_buffer[4];
        c0 = ci4 + c0;
        c1 = ci4 * T::R37_EVEN_TWIDDLE_4 + c1;
        s0 = fmla(si4, T::R37_ODD_TWIDDLE_4, s0);
        let ci5 = c_buffer[5];
        let si5 = s_buffer[5];
        c0 = ci5 + c0;
        c1 = ci5 * T::R37_EVEN_TWIDDLE_5 + c1;
        s0 = fmla(si5, T::R37_ODD_TWIDDLE_5, s0);
        let ci6 = c_buffer[6];
        let si6 = s_buffer[6];
        c0 = ci6 + c0;
        c1 = ci6 * T::R37_EVEN_TWIDDLE_6 + c1;
        s0 = fmla(si6, T::R37_ODD_TWIDDLE_6, s0);
        let ci7 = c_buffer[7];
        let si7 = s_buffer[7];
        c0 = ci7 + c0;
        c1 = ci7 * T::R37_EVEN_TWIDDLE_7 + c1;
        s0 = fmla(si7, T::R37_ODD_TWIDDLE_7, s0);
        let ci8 = c_buffer[8];
        let si8 = s_buffer[8];
        c0 = ci8 + c0;
        c1 = ci8 * T::R37_EVEN_TWIDDLE_8 + c1;
        s0 = fmla(si8, T::R37_ODD_TWIDDLE_8, s0);
        let ci9 = c_buffer[9];
        let si9 = s_buffer[9];
        c0 = ci9 + c0;
        c1 = ci9 * T::R37_EVEN_TWIDDLE_9 + c1;
        s0 = fmla(si9, T::R37_ODD_TWIDDLE_9, s0);
        let ci10 = c_buffer[10];
        let si10 = s_buffer[10];
        c0 = ci10 + c0;
        c1 = ci10 * T::R37_EVEN_TWIDDLE_10 + c1;
        s0 = fmla(si10, T::R37_ODD_TWIDDLE_10, s0);
        let ci11 = c_buffer[11];
        let si11 = s_buffer[11];
        c0 = ci11 + c0;
        c1 = ci11 * T::R37_EVEN_TWIDDLE_11 + c1;
        s0 = fmla(si11, T::R37_ODD_TWIDDLE_11, s0);
        let ci12 = c_buffer[12];
        let si12 = s_buffer[12];
        c0 = ci12 + c0;
        c1 = ci12 * T::R37_EVEN_TWIDDLE_12 + c1;
        s0 = fmla(si12, T::R37_ODD_TWIDDLE_12, s0);
        let ci13 = c_buffer[13];
        let si13 = s_buffer[13];
        c0 = ci13 + c0;
        c1 = ci13 * T::R37_EVEN_TWIDDLE_13 + c1;
        s0 = fmla(si13, T::R37_ODD_TWIDDLE_13, s0);
        let ci14 = c_buffer[14];
        let si14 = s_buffer[14];
        c0 = ci14 + c0;
        c1 = ci14 * T::R37_EVEN_TWIDDLE_14 + c1;
        s0 = fmla(si14, T::R37_ODD_TWIDDLE_14, s0);
        let ci15 = c_buffer[15];
        let si15 = s_buffer[15];
        c0 = ci15 + c0;
        c1 = ci15 * T::R37_EVEN_TWIDDLE_15 + c1;
        s0 = fmla(si15, T::R37_ODD_TWIDDLE_15, s0);
        let ci16 = c_buffer[16];
        let si16 = s_buffer[16];
        c0 = ci16 + c0;
        c1 = ci16 * T::R37_EVEN_TWIDDLE_16 + c1;
        s0 = fmla(si16, T::R37_ODD_TWIDDLE_16, s0);
        let ci17 = c_buffer[17];
        let si17 = s_buffer[17];
        c0 = ci17 + c0;
        c1 = ci17 * T::R37_EVEN_TWIDDLE_17 + c1;
        s0 = fmla(si17, T::R37_ODD_TWIDDLE_17, s0);
        let a0 = a_buffer[0];
        let dc = c0 + a0;
        data[0] = dc;
        data[1] = s0;
        data[2] = -(c1 + a0);
        let mut c0 = qc * T::R37_EVEN_TWIDDLE_17;
        let mut s0 = s_buffer[0];
        s0 *= -T::R37_ODD_TWIDDLE_1;
        c0 = fmla(ci1, T::R37_EVEN_TWIDDLE_15, c0);
        s0 = fmla(si1, -T::R37_ODD_TWIDDLE_4, s0);
        c0 = fmla(ci2, T::R37_EVEN_TWIDDLE_13, c0);
        s0 = fmla(si2, -T::R37_ODD_TWIDDLE_7, s0);
        c0 = fmla(ci3, T::R37_EVEN_TWIDDLE_11, c0);
        s0 = fmla(si3, -T::R37_ODD_TWIDDLE_10, s0);
        c0 = fmla(ci4, T::R37_EVEN_TWIDDLE_9, c0);
        s0 = fmla(si4, -T::R37_ODD_TWIDDLE_13, s0);
        c0 = fmla(ci5, T::R37_EVEN_TWIDDLE_7, c0);
        s0 = fmla(si5, -T::R37_ODD_TWIDDLE_16, s0);
        c0 = fmla(ci6, T::R37_EVEN_TWIDDLE_5, c0);
        s0 = fmla(si6, T::R37_ODD_TWIDDLE_17, s0);
        c0 = fmla(ci7, T::R37_EVEN_TWIDDLE_3, c0);
        s0 = fmla(si7, T::R37_ODD_TWIDDLE_14, s0);
        c0 = fmla(ci8, T::R37_EVEN_TWIDDLE_1, c0);
        s0 = fmla(si8, T::R37_ODD_TWIDDLE_11, s0);
        c0 = fmla(ci9, T::R37_EVEN_TWIDDLE_0, c0);
        s0 = fmla(si9, T::R37_ODD_TWIDDLE_8, s0);
        c0 = fmla(ci10, T::R37_EVEN_TWIDDLE_2, c0);
        s0 = fmla(si10, T::R37_ODD_TWIDDLE_5, s0);
        c0 = fmla(ci11, T::R37_EVEN_TWIDDLE_4, c0);
        s0 = fmla(si11, T::R37_ODD_TWIDDLE_2, s0);
        c0 = fmla(ci12, T::R37_EVEN_TWIDDLE_6, c0);
        s0 = fmla(si12, T::R37_ODD_TWIDDLE_0, s0);
        c0 = fmla(ci13, T::R37_EVEN_TWIDDLE_8, c0);
        s0 = fmla(si13, T::R37_ODD_TWIDDLE_3, s0);
        c0 = fmla(ci14, T::R37_EVEN_TWIDDLE_10, c0);
        s0 = fmla(si14, T::R37_ODD_TWIDDLE_6, s0);
        c0 = fmla(ci15, T::R37_EVEN_TWIDDLE_12, c0);
        s0 = fmla(si15, T::R37_ODD_TWIDDLE_9, s0);
        c0 = fmla(ci16, T::R37_EVEN_TWIDDLE_14, c0);
        s0 = fmla(si16, T::R37_ODD_TWIDDLE_12, s0);
        c0 = fmla(ci17, T::R37_EVEN_TWIDDLE_16, c0);
        s0 = fmla(si17, T::R37_ODD_TWIDDLE_15, s0);
        let dc = c0 + a0;
        data[4] = dc;
        data[3] = -s0;
        let mut c0 = qc * T::R37_EVEN_TWIDDLE_1;
        let mut s0 = s_buffer[0];
        s0 *= T::R37_ODD_TWIDDLE_2;
        c0 = fmla(ci1, T::R37_EVEN_TWIDDLE_4, c0);
        s0 = fmla(si1, T::R37_ODD_TWIDDLE_7, s0);
        c0 = fmla(ci2, T::R37_EVEN_TWIDDLE_7, c0);
        s0 = fmla(si2, T::R37_ODD_TWIDDLE_12, s0);
        c0 = fmla(ci3, T::R37_EVEN_TWIDDLE_10, c0);
        s0 = fmla(si3, T::R37_ODD_TWIDDLE_17, s0);
        c0 = fmla(ci4, T::R37_EVEN_TWIDDLE_13, c0);
        s0 = fmla(si4, -T::R37_ODD_TWIDDLE_14, s0);
        c0 = fmla(ci5, T::R37_EVEN_TWIDDLE_16, c0);
        s0 = fmla(si5, -T::R37_ODD_TWIDDLE_9, s0);
        c0 = fmla(ci6, T::R37_EVEN_TWIDDLE_17, c0);
        s0 = fmla(si6, -T::R37_ODD_TWIDDLE_4, s0);
        c0 = fmla(ci7, T::R37_EVEN_TWIDDLE_14, c0);
        s0 = fmla(si7, -T::R37_ODD_TWIDDLE_0, s0);
        c0 = fmla(ci8, T::R37_EVEN_TWIDDLE_11, c0);
        s0 = fmla(si8, -T::R37_ODD_TWIDDLE_5, s0);
        c0 = fmla(ci9, T::R37_EVEN_TWIDDLE_8, c0);
        s0 = fmla(si9, -T::R37_ODD_TWIDDLE_10, s0);
        c0 = fmla(ci10, T::R37_EVEN_TWIDDLE_5, c0);
        s0 = fmla(si10, -T::R37_ODD_TWIDDLE_15, s0);
        c0 = fmla(ci11, T::R37_EVEN_TWIDDLE_2, c0);
        s0 = fmla(si11, T::R37_ODD_TWIDDLE_16, s0);
        c0 = fmla(ci12, T::R37_EVEN_TWIDDLE_0, c0);
        s0 = fmla(si12, T::R37_ODD_TWIDDLE_11, s0);
        c0 = fmla(ci13, T::R37_EVEN_TWIDDLE_3, c0);
        s0 = fmla(si13, T::R37_ODD_TWIDDLE_6, s0);
        c0 = fmla(ci14, T::R37_EVEN_TWIDDLE_6, c0);
        s0 = fmla(si14, T::R37_ODD_TWIDDLE_1, s0);
        c0 = fmla(ci15, T::R37_EVEN_TWIDDLE_9, c0);
        s0 = fmla(si15, T::R37_ODD_TWIDDLE_3, s0);
        c0 = fmla(ci16, T::R37_EVEN_TWIDDLE_12, c0);
        s0 = fmla(si16, T::R37_ODD_TWIDDLE_8, s0);
        c0 = fmla(ci17, T::R37_EVEN_TWIDDLE_15, c0);
        s0 = fmla(si17, T::R37_ODD_TWIDDLE_13, s0);
        let dc = c0 + a0;
        data[6] = -dc;
        data[5] = s0;
        let mut c0 = qc * T::R37_EVEN_TWIDDLE_16;
        let mut s0 = s_buffer[0];
        s0 *= -T::R37_ODD_TWIDDLE_3;
        c0 = fmla(ci1, T::R37_EVEN_TWIDDLE_12, c0);
        s0 = fmla(si1, -T::R37_ODD_TWIDDLE_10, s0);
        c0 = fmla(ci2, T::R37_EVEN_TWIDDLE_8, c0);
        s0 = fmla(si2, -T::R37_ODD_TWIDDLE_17, s0);
        c0 = fmla(ci3, T::R37_EVEN_TWIDDLE_4, c0);
        s0 = fmla(si3, T::R37_ODD_TWIDDLE_12, s0);
        c0 = fmla(ci4, T::R37_EVEN_TWIDDLE_0, c0);
        s0 = fmla(si4, T::R37_ODD_TWIDDLE_5, s0);
        c0 = fmla(ci5, T::R37_EVEN_TWIDDLE_3, c0);
        s0 = fmla(si5, T::R37_ODD_TWIDDLE_1, s0);
        c0 = fmla(ci6, T::R37_EVEN_TWIDDLE_7, c0);
        s0 = fmla(si6, T::R37_ODD_TWIDDLE_8, s0);
        c0 = fmla(ci7, T::R37_EVEN_TWIDDLE_11, c0);
        s0 = fmla(si7, T::R37_ODD_TWIDDLE_15, s0);
        c0 = fmla(ci8, T::R37_EVEN_TWIDDLE_15, c0);
        s0 = fmla(si8, -T::R37_ODD_TWIDDLE_14, s0);
        c0 = fmla(ci9, T::R37_EVEN_TWIDDLE_17, c0);
        s0 = fmla(si9, -T::R37_ODD_TWIDDLE_7, s0);
        c0 = fmla(ci10, T::R37_EVEN_TWIDDLE_13, c0);
        s0 = fmla(si10, -T::R37_ODD_TWIDDLE_0, s0);
        c0 = fmla(ci11, T::R37_EVEN_TWIDDLE_9, c0);
        s0 = fmla(si11, -T::R37_ODD_TWIDDLE_6, s0);
        c0 = fmla(ci12, T::R37_EVEN_TWIDDLE_5, c0);
        s0 = fmla(si12, -T::R37_ODD_TWIDDLE_13, s0);
        c0 = fmla(ci13, T::R37_EVEN_TWIDDLE_1, c0);
        s0 = fmla(si13, T::R37_ODD_TWIDDLE_16, s0);
        c0 = fmla(ci14, T::R37_EVEN_TWIDDLE_2, c0);
        s0 = fmla(si14, T::R37_ODD_TWIDDLE_9, s0);
        c0 = fmla(ci15, T::R37_EVEN_TWIDDLE_6, c0);
        s0 = fmla(si15, T::R37_ODD_TWIDDLE_2, s0);
        c0 = fmla(ci16, T::R37_EVEN_TWIDDLE_10, c0);
        s0 = fmla(si16, T::R37_ODD_TWIDDLE_4, s0);
        c0 = fmla(ci17, T::R37_EVEN_TWIDDLE_14, c0);
        s0 = fmla(si17, T::R37_ODD_TWIDDLE_11, s0);
        let dc = c0 + a0;
        data[8] = dc;
        data[7] = -s0;
        let mut c0 = qc * T::R37_EVEN_TWIDDLE_2;
        let mut s0 = s_buffer[0];
        s0 *= T::R37_ODD_TWIDDLE_4;
        c0 = fmla(ci1, T::R37_EVEN_TWIDDLE_7, c0);
        s0 = fmla(si1, T::R37_ODD_TWIDDLE_13, s0);
        c0 = fmla(ci2, T::R37_EVEN_TWIDDLE_12, c0);
        s0 = fmla(si2, -T::R37_ODD_TWIDDLE_14, s0);
        c0 = fmla(ci3, T::R37_EVEN_TWIDDLE_17, c0);
        s0 = fmla(si3, -T::R37_ODD_TWIDDLE_5, s0);
        c0 = fmla(ci4, T::R37_EVEN_TWIDDLE_14, c0);
        s0 = fmla(si4, -T::R37_ODD_TWIDDLE_3, s0);
        c0 = fmla(ci5, T::R37_EVEN_TWIDDLE_9, c0);
        s0 = fmla(si5, -T::R37_ODD_TWIDDLE_12, s0);
        c0 = fmla(ci6, T::R37_EVEN_TWIDDLE_4, c0);
        s0 = fmla(si6, T::R37_ODD_TWIDDLE_15, s0);
        c0 = fmla(ci7, T::R37_EVEN_TWIDDLE_0, c0);
        s0 = fmla(si7, T::R37_ODD_TWIDDLE_6, s0);
        c0 = fmla(ci8, T::R37_EVEN_TWIDDLE_5, c0);
        s0 = fmla(si8, T::R37_ODD_TWIDDLE_2, s0);
        c0 = fmla(ci9, T::R37_EVEN_TWIDDLE_10, c0);
        s0 = fmla(si9, T::R37_ODD_TWIDDLE_11, s0);
        c0 = fmla(ci10, T::R37_EVEN_TWIDDLE_15, c0);
        s0 = fmla(si10, -T::R37_ODD_TWIDDLE_16, s0);
        c0 = fmla(ci11, T::R37_EVEN_TWIDDLE_16, c0);
        s0 = fmla(si11, -T::R37_ODD_TWIDDLE_7, s0);
        c0 = fmla(ci12, T::R37_EVEN_TWIDDLE_11, c0);
        s0 = fmla(si12, -T::R37_ODD_TWIDDLE_1, s0);
        c0 = fmla(ci13, T::R37_EVEN_TWIDDLE_6, c0);
        s0 = fmla(si13, -T::R37_ODD_TWIDDLE_10, s0);
        c0 = fmla(ci14, T::R37_EVEN_TWIDDLE_1, c0);
        s0 = fmla(si14, T::R37_ODD_TWIDDLE_17, s0);
        c0 = fmla(ci15, T::R37_EVEN_TWIDDLE_3, c0);
        s0 = fmla(si15, T::R37_ODD_TWIDDLE_8, s0);
        c0 = fmla(ci16, T::R37_EVEN_TWIDDLE_8, c0);
        s0 = fmla(si16, T::R37_ODD_TWIDDLE_0, s0);
        c0 = fmla(ci17, T::R37_EVEN_TWIDDLE_13, c0);
        s0 = fmla(si17, T::R37_ODD_TWIDDLE_9, s0);
        let dc = c0 + a0;
        data[10] = -dc;
        data[9] = s0;
        let mut c0 = qc * T::R37_EVEN_TWIDDLE_15;
        let mut s0 = s_buffer[0];
        s0 *= -T::R37_ODD_TWIDDLE_5;
        c0 = fmla(ci1, T::R37_EVEN_TWIDDLE_9, c0);
        s0 = fmla(si1, -T::R37_ODD_TWIDDLE_16, s0);
        c0 = fmla(ci2, T::R37_EVEN_TWIDDLE_3, c0);
        s0 = fmla(si2, T::R37_ODD_TWIDDLE_9, s0);
        c0 = fmla(ci3, T::R37_EVEN_TWIDDLE_2, c0);
        s0 = fmla(si3, T::R37_ODD_TWIDDLE_1, s0);
        c0 = fmla(ci4, T::R37_EVEN_TWIDDLE_8, c0);
        s0 = fmla(si4, T::R37_ODD_TWIDDLE_12, s0);
        c0 = fmla(ci5, T::R37_EVEN_TWIDDLE_14, c0);
        s0 = fmla(si5, -T::R37_ODD_TWIDDLE_13, s0);
        c0 = fmla(ci6, T::R37_EVEN_TWIDDLE_16, c0);
        s0 = fmla(si6, -T::R37_ODD_TWIDDLE_2, s0);
        c0 = fmla(ci7, T::R37_EVEN_TWIDDLE_10, c0);
        s0 = fmla(si7, -T::R37_ODD_TWIDDLE_8, s0);
        c0 = fmla(ci8, T::R37_EVEN_TWIDDLE_4, c0);
        s0 = fmla(si8, T::R37_ODD_TWIDDLE_17, s0);
        c0 = fmla(ci9, T::R37_EVEN_TWIDDLE_1, c0);
        s0 = fmla(si9, T::R37_ODD_TWIDDLE_6, s0);
        c0 = fmla(ci10, T::R37_EVEN_TWIDDLE_7, c0);
        s0 = fmla(si10, T::R37_ODD_TWIDDLE_4, s0);
        c0 = fmla(ci11, T::R37_EVEN_TWIDDLE_13, c0);
        s0 = fmla(si11, T::R37_ODD_TWIDDLE_15, s0);
        c0 = fmla(ci12, T::R37_EVEN_TWIDDLE_17, c0);
        s0 = fmla(si12, -T::R37_ODD_TWIDDLE_10, s0);
        c0 = fmla(ci13, T::R37_EVEN_TWIDDLE_11, c0);
        s0 = fmla(si13, -T::R37_ODD_TWIDDLE_0, s0);
        c0 = fmla(ci14, T::R37_EVEN_TWIDDLE_5, c0);
        s0 = fmla(si14, -T::R37_ODD_TWIDDLE_11, s0);
        c0 = fmla(ci15, T::R37_EVEN_TWIDDLE_0, c0);
        s0 = fmla(si15, T::R37_ODD_TWIDDLE_14, s0);
        c0 = fmla(ci16, T::R37_EVEN_TWIDDLE_6, c0);
        s0 = fmla(si16, T::R37_ODD_TWIDDLE_3, s0);
        c0 = fmla(ci17, T::R37_EVEN_TWIDDLE_12, c0);
        s0 = fmla(si17, T::R37_ODD_TWIDDLE_7, s0);
        let dc = c0 + a0;
        data[12] = dc;
        data[11] = -s0;
        let mut c0 = qc * T::R37_EVEN_TWIDDLE_3;
        let mut s0 = s_buffer[0];
        s0 *= T::R37_ODD_TWIDDLE_6;
        c0 = fmla(ci1, T::R37_EVEN_TWIDDLE_10, c0);
        s0 = fmla(si1, -T::R37_ODD_TWIDDLE_17, s0);
        c0 = fmla(ci2, T::R37_EVEN_TWIDDLE_17, c0);
        s0 = fmla(si2, -T::R37_ODD_TWIDDLE_4, s0);
        c0 = fmla(ci3, T::R37_EVEN_TWIDDLE_12, c0);
        s0 = fmla(si3, -T::R37_ODD_TWIDDLE_8, s0);
        c0 = fmla(ci4, T::R37_EVEN_TWIDDLE_5, c0);
        s0 = fmla(si4, T::R37_ODD_TWIDDLE_15, s0);
        c0 = fmla(ci5, T::R37_EVEN_TWIDDLE_1, c0);
        s0 = fmla(si5, T::R37_ODD_TWIDDLE_2, s0);
        c0 = fmla(ci6, T::R37_EVEN_TWIDDLE_8, c0);
        s0 = fmla(si6, T::R37_ODD_TWIDDLE_10, s0);
        c0 = fmla(ci7, T::R37_EVEN_TWIDDLE_15, c0);
        s0 = fmla(si7, -T::R37_ODD_TWIDDLE_13, s0);
        c0 = fmla(ci8, T::R37_EVEN_TWIDDLE_14, c0);
        s0 = fmla(si8, -T::R37_ODD_TWIDDLE_0, s0);
        c0 = fmla(ci9, T::R37_EVEN_TWIDDLE_7, c0);
        s0 = fmla(si9, -T::R37_ODD_TWIDDLE_12, s0);
        c0 = fmla(ci10, T::R37_EVEN_TWIDDLE_0, c0);
        s0 = fmla(si10, T::R37_ODD_TWIDDLE_11, s0);
        c0 = fmla(ci11, T::R37_EVEN_TWIDDLE_6, c0);
        s0 = fmla(si11, T::R37_ODD_TWIDDLE_1, s0);
        c0 = fmla(ci12, T::R37_EVEN_TWIDDLE_13, c0);
        s0 = fmla(si12, T::R37_ODD_TWIDDLE_14, s0);
        c0 = fmla(ci13, T::R37_EVEN_TWIDDLE_16, c0);
        s0 = fmla(si13, -T::R37_ODD_TWIDDLE_9, s0);
        c0 = fmla(ci14, T::R37_EVEN_TWIDDLE_9, c0);
        s0 = fmla(si14, -T::R37_ODD_TWIDDLE_3, s0);
        c0 = fmla(ci15, T::R37_EVEN_TWIDDLE_2, c0);
        s0 = fmla(si15, -T::R37_ODD_TWIDDLE_16, s0);
        c0 = fmla(ci16, T::R37_EVEN_TWIDDLE_4, c0);
        s0 = fmla(si16, T::R37_ODD_TWIDDLE_7, s0);
        c0 = fmla(ci17, T::R37_EVEN_TWIDDLE_11, c0);
        s0 = fmla(si17, T::R37_ODD_TWIDDLE_5, s0);
        let dc = c0 + a0;
        data[14] = -dc;
        data[13] = s0;
        let mut c0 = qc * T::R37_EVEN_TWIDDLE_14;
        let mut s0 = s_buffer[0];
        s0 *= -T::R37_ODD_TWIDDLE_7;
        c0 = fmla(ci1, T::R37_EVEN_TWIDDLE_6, c0);
        s0 = fmla(si1, T::R37_ODD_TWIDDLE_14, s0);
        c0 = fmla(ci2, T::R37_EVEN_TWIDDLE_1, c0);
        s0 = fmla(si2, T::R37_ODD_TWIDDLE_0, s0);
        c0 = fmla(ci3, T::R37_EVEN_TWIDDLE_9, c0);
        s0 = fmla(si3, T::R37_ODD_TWIDDLE_15, s0);
        c0 = fmla(ci4, T::R37_EVEN_TWIDDLE_17, c0);
        s0 = fmla(si4, -T::R37_ODD_TWIDDLE_6, s0);
        c0 = fmla(ci5, T::R37_EVEN_TWIDDLE_11, c0);
        s0 = fmla(si5, -T::R37_ODD_TWIDDLE_8, s0);
        c0 = fmla(ci6, T::R37_EVEN_TWIDDLE_3, c0);
        s0 = fmla(si6, T::R37_ODD_TWIDDLE_13, s0);
        c0 = fmla(ci7, T::R37_EVEN_TWIDDLE_4, c0);
        s0 = fmla(si7, T::R37_ODD_TWIDDLE_1, s0);
        c0 = fmla(ci8, T::R37_EVEN_TWIDDLE_12, c0);
        s0 = fmla(si8, T::R37_ODD_TWIDDLE_16, s0);
        c0 = fmla(ci9, T::R37_EVEN_TWIDDLE_16, c0);
        s0 = fmla(si9, -T::R37_ODD_TWIDDLE_5, s0);
        c0 = fmla(ci10, T::R37_EVEN_TWIDDLE_8, c0);
        s0 = fmla(si10, -T::R37_ODD_TWIDDLE_9, s0);
        c0 = fmla(ci11, T::R37_EVEN_TWIDDLE_0, c0);
        s0 = fmla(si11, T::R37_ODD_TWIDDLE_12, s0);
        c0 = fmla(ci12, T::R37_EVEN_TWIDDLE_7, c0);
        s0 = fmla(si12, T::R37_ODD_TWIDDLE_2, s0);
        c0 = fmla(ci13, T::R37_EVEN_TWIDDLE_15, c0);
        s0 = fmla(si13, T::R37_ODD_TWIDDLE_17, s0);
        c0 = fmla(ci14, T::R37_EVEN_TWIDDLE_13, c0);
        s0 = fmla(si14, -T::R37_ODD_TWIDDLE_4, s0);
        c0 = fmla(ci15, T::R37_EVEN_TWIDDLE_5, c0);
        s0 = fmla(si15, -T::R37_ODD_TWIDDLE_10, s0);
        c0 = fmla(ci16, T::R37_EVEN_TWIDDLE_2, c0);
        s0 = fmla(si16, T::R37_ODD_TWIDDLE_11, s0);
        c0 = fmla(ci17, T::R37_EVEN_TWIDDLE_10, c0);
        s0 = fmla(si17, T::R37_ODD_TWIDDLE_3, s0);
        let dc = c0 + a0;
        data[16] = dc;
        data[15] = -s0;
        let mut c0 = qc * T::R37_EVEN_TWIDDLE_4;
        let mut s0 = s_buffer[0];
        s0 *= T::R37_ODD_TWIDDLE_8;
        c0 = fmla(ci1, T::R37_EVEN_TWIDDLE_13, c0);
        s0 = fmla(si1, -T::R37_ODD_TWIDDLE_11, s0);
        c0 = fmla(ci2, T::R37_EVEN_TWIDDLE_14, c0);
        s0 = fmla(si2, -T::R37_ODD_TWIDDLE_5, s0);
        c0 = fmla(ci3, T::R37_EVEN_TWIDDLE_5, c0);
        s0 = fmla(si3, T::R37_ODD_TWIDDLE_14, s0);
        c0 = fmla(ci4, T::R37_EVEN_TWIDDLE_3, c0);
        s0 = fmla(si4, T::R37_ODD_TWIDDLE_2, s0);
        c0 = fmla(ci5, T::R37_EVEN_TWIDDLE_12, c0);
        s0 = fmla(si5, -T::R37_ODD_TWIDDLE_17, s0);
        c0 = fmla(ci6, T::R37_EVEN_TWIDDLE_15, c0);
        s0 = fmla(si6, -T::R37_ODD_TWIDDLE_0, s0);
        c0 = fmla(ci7, T::R37_EVEN_TWIDDLE_6, c0);
        s0 = fmla(si7, -T::R37_ODD_TWIDDLE_16, s0);
        c0 = fmla(ci8, T::R37_EVEN_TWIDDLE_2, c0);
        s0 = fmla(si8, T::R37_ODD_TWIDDLE_3, s0);
        c0 = fmla(ci9, T::R37_EVEN_TWIDDLE_11, c0);
        s0 = fmla(si9, T::R37_ODD_TWIDDLE_13, s0);
        c0 = fmla(ci10, T::R37_EVEN_TWIDDLE_16, c0);
        s0 = fmla(si10, -T::R37_ODD_TWIDDLE_6, s0);
        c0 = fmla(ci11, T::R37_EVEN_TWIDDLE_7, c0);
        s0 = fmla(si11, -T::R37_ODD_TWIDDLE_10, s0);
        c0 = fmla(ci12, T::R37_EVEN_TWIDDLE_1, c0);
        s0 = fmla(si12, T::R37_ODD_TWIDDLE_9, s0);
        c0 = fmla(ci13, T::R37_EVEN_TWIDDLE_10, c0);
        s0 = fmla(si13, T::R37_ODD_TWIDDLE_7, s0);
        c0 = fmla(ci14, T::R37_EVEN_TWIDDLE_17, c0);
        s0 = fmla(si14, -T::R37_ODD_TWIDDLE_12, s0);
        c0 = fmla(ci15, T::R37_EVEN_TWIDDLE_8, c0);
        s0 = fmla(si15, -T::R37_ODD_TWIDDLE_4, s0);
        c0 = fmla(ci16, T::R37_EVEN_TWIDDLE_0, c0);
        s0 = fmla(si16, T::R37_ODD_TWIDDLE_15, s0);
        c0 = fmla(ci17, T::R37_EVEN_TWIDDLE_9, c0);
        s0 = fmla(si17, T::R37_ODD_TWIDDLE_1, s0);
        let dc = c0 + a0;
        data[18] = -dc;
        data[17] = s0;
        let mut c0 = qc * T::R37_EVEN_TWIDDLE_13;
        let mut s0 = s_buffer[0];
        s0 *= -T::R37_ODD_TWIDDLE_9;
        c0 = fmla(ci1, T::R37_EVEN_TWIDDLE_3, c0);
        s0 = fmla(si1, T::R37_ODD_TWIDDLE_8, s0);
        c0 = fmla(ci2, T::R37_EVEN_TWIDDLE_6, c0);
        s0 = fmla(si2, T::R37_ODD_TWIDDLE_10, s0);
        c0 = fmla(ci3, T::R37_EVEN_TWIDDLE_16, c0);
        s0 = fmla(si3, -T::R37_ODD_TWIDDLE_7, s0);
        c0 = fmla(ci4, T::R37_EVEN_TWIDDLE_10, c0);
        s0 = fmla(si4, -T::R37_ODD_TWIDDLE_11, s0);
        c0 = fmla(ci5, T::R37_EVEN_TWIDDLE_0, c0);
        s0 = fmla(si5, T::R37_ODD_TWIDDLE_6, s0);
        c0 = fmla(ci6, T::R37_EVEN_TWIDDLE_9, c0);
        s0 = fmla(si6, T::R37_ODD_TWIDDLE_12, s0);
        c0 = fmla(ci7, T::R37_EVEN_TWIDDLE_17, c0);
        s0 = fmla(si7, -T::R37_ODD_TWIDDLE_5, s0);
        c0 = fmla(ci8, T::R37_EVEN_TWIDDLE_7, c0);
        s0 = fmla(si8, -T::R37_ODD_TWIDDLE_13, s0);
        c0 = fmla(ci9, T::R37_EVEN_TWIDDLE_2, c0);
        s0 = fmla(si9, T::R37_ODD_TWIDDLE_4, s0);
        c0 = fmla(ci10, T::R37_EVEN_TWIDDLE_12, c0);
        s0 = fmla(si10, T::R37_ODD_TWIDDLE_14, s0);
        c0 = fmla(ci11, T::R37_EVEN_TWIDDLE_14, c0);
        s0 = fmla(si11, -T::R37_ODD_TWIDDLE_3, s0);
        c0 = fmla(ci12, T::R37_EVEN_TWIDDLE_4, c0);
        s0 = fmla(si12, -T::R37_ODD_TWIDDLE_15, s0);
        c0 = fmla(ci13, T::R37_EVEN_TWIDDLE_5, c0);
        s0 = fmla(si13, T::R37_ODD_TWIDDLE_2, s0);
        c0 = fmla(ci14, T::R37_EVEN_TWIDDLE_15, c0);
        s0 = fmla(si14, T::R37_ODD_TWIDDLE_16, s0);
        c0 = fmla(ci15, T::R37_EVEN_TWIDDLE_11, c0);
        s0 = fmla(si15, -T::R37_ODD_TWIDDLE_1, s0);
        c0 = fmla(ci16, T::R37_EVEN_TWIDDLE_1, c0);
        s0 = fmla(si16, -T::R37_ODD_TWIDDLE_17, s0);
        c0 = fmla(ci17, T::R37_EVEN_TWIDDLE_8, c0);
        s0 = fmla(si17, T::R37_ODD_TWIDDLE_0, s0);
        let dc = c0 + a0;
        data[20] = dc;
        data[19] = -s0;
        let mut c0 = qc * T::R37_EVEN_TWIDDLE_5;
        let mut s0 = s_buffer[0];
        s0 *= T::R37_ODD_TWIDDLE_10;
        c0 = fmla(ci1, T::R37_EVEN_TWIDDLE_16, c0);
        s0 = fmla(si1, -T::R37_ODD_TWIDDLE_5, s0);
        c0 = fmla(ci2, T::R37_EVEN_TWIDDLE_9, c0);
        s0 = fmla(si2, -T::R37_ODD_TWIDDLE_15, s0);
        c0 = fmla(ci3, T::R37_EVEN_TWIDDLE_1, c0);
        s0 = fmla(si3, T::R37_ODD_TWIDDLE_0, s0);
        c0 = fmla(ci4, T::R37_EVEN_TWIDDLE_12, c0);
        s0 = fmla(si4, -T::R37_ODD_TWIDDLE_16, s0);
        c0 = fmla(ci5, T::R37_EVEN_TWIDDLE_13, c0);
        s0 = fmla(si5, -T::R37_ODD_TWIDDLE_4, s0);
        c0 = fmla(ci6, T::R37_EVEN_TWIDDLE_2, c0);
        s0 = fmla(si6, T::R37_ODD_TWIDDLE_11, s0);
        c0 = fmla(ci7, T::R37_EVEN_TWIDDLE_8, c0);
        s0 = fmla(si7, T::R37_ODD_TWIDDLE_9, s0);
        c0 = fmla(ci8, T::R37_EVEN_TWIDDLE_17, c0);
        s0 = fmla(si8, -T::R37_ODD_TWIDDLE_6, s0);
        c0 = fmla(ci9, T::R37_EVEN_TWIDDLE_6, c0);
        s0 = fmla(si9, -T::R37_ODD_TWIDDLE_14, s0);
        c0 = fmla(ci10, T::R37_EVEN_TWIDDLE_4, c0);
        s0 = fmla(si10, T::R37_ODD_TWIDDLE_1, s0);
        c0 = fmla(ci11, T::R37_EVEN_TWIDDLE_15, c0);
        s0 = fmla(si11, -T::R37_ODD_TWIDDLE_17, s0);
        c0 = fmla(ci12, T::R37_EVEN_TWIDDLE_10, c0);
        s0 = fmla(si12, -T::R37_ODD_TWIDDLE_3, s0);
        c0 = fmla(ci13, T::R37_EVEN_TWIDDLE_0, c0);
        s0 = fmla(si13, T::R37_ODD_TWIDDLE_12, s0);
        c0 = fmla(ci14, T::R37_EVEN_TWIDDLE_11, c0);
        s0 = fmla(si14, T::R37_ODD_TWIDDLE_8, s0);
        c0 = fmla(ci15, T::R37_EVEN_TWIDDLE_14, c0);
        s0 = fmla(si15, -T::R37_ODD_TWIDDLE_7, s0);
        c0 = fmla(ci16, T::R37_EVEN_TWIDDLE_3, c0);
        s0 = fmla(si16, -T::R37_ODD_TWIDDLE_13, s0);
        c0 = fmla(ci17, T::R37_EVEN_TWIDDLE_7, c0);
        s0 = fmla(si17, T::R37_ODD_TWIDDLE_2, s0);
        let dc = c0 + a0;
        data[22] = -dc;
        data[21] = s0;
        let mut c0 = qc * T::R37_EVEN_TWIDDLE_12;
        let mut s0 = s_buffer[0];
        s0 *= -T::R37_ODD_TWIDDLE_11;
        c0 = fmla(ci1, T::R37_EVEN_TWIDDLE_0, c0);
        s0 = fmla(si1, T::R37_ODD_TWIDDLE_2, s0);
        c0 = fmla(ci2, T::R37_EVEN_TWIDDLE_11, c0);
        s0 = fmla(si2, -T::R37_ODD_TWIDDLE_16, s0);
        c0 = fmla(ci3, T::R37_EVEN_TWIDDLE_13, c0);
        s0 = fmla(si3, -T::R37_ODD_TWIDDLE_6, s0);
        c0 = fmla(ci4, T::R37_EVEN_TWIDDLE_1, c0);
        s0 = fmla(si4, T::R37_ODD_TWIDDLE_7, s0);
        c0 = fmla(ci5, T::R37_EVEN_TWIDDLE_10, c0);
        s0 = fmla(si5, T::R37_ODD_TWIDDLE_15, s0);
        c0 = fmla(ci6, T::R37_EVEN_TWIDDLE_14, c0);
        s0 = fmla(si6, -T::R37_ODD_TWIDDLE_1, s0);
        c0 = fmla(ci7, T::R37_EVEN_TWIDDLE_2, c0);
        s0 = fmla(si7, T::R37_ODD_TWIDDLE_12, s0);
        c0 = fmla(ci8, T::R37_EVEN_TWIDDLE_9, c0);
        s0 = fmla(si8, T::R37_ODD_TWIDDLE_10, s0);
        c0 = fmla(ci9, T::R37_EVEN_TWIDDLE_15, c0);
        s0 = fmla(si9, -T::R37_ODD_TWIDDLE_3, s0);
        c0 = fmla(ci10, T::R37_EVEN_TWIDDLE_3, c0);
        s0 = fmla(si10, T::R37_ODD_TWIDDLE_17, s0);
        c0 = fmla(ci11, T::R37_EVEN_TWIDDLE_8, c0);
        s0 = fmla(si11, T::R37_ODD_TWIDDLE_5, s0);
        c0 = fmla(ci12, T::R37_EVEN_TWIDDLE_16, c0);
        s0 = fmla(si12, -T::R37_ODD_TWIDDLE_8, s0);
        c0 = fmla(ci13, T::R37_EVEN_TWIDDLE_4, c0);
        s0 = fmla(si13, -T::R37_ODD_TWIDDLE_14, s0);
        c0 = fmla(ci14, T::R37_EVEN_TWIDDLE_7, c0);
        s0 = fmla(si14, T::R37_ODD_TWIDDLE_0, s0);
        c0 = fmla(ci15, T::R37_EVEN_TWIDDLE_17, c0);
        s0 = fmla(si15, -T::R37_ODD_TWIDDLE_13, s0);
        c0 = fmla(ci16, T::R37_EVEN_TWIDDLE_5, c0);
        s0 = fmla(si16, -T::R37_ODD_TWIDDLE_9, s0);
        c0 = fmla(ci17, T::R37_EVEN_TWIDDLE_6, c0);
        s0 = fmla(si17, T::R37_ODD_TWIDDLE_4, s0);
        let dc = c0 + a0;
        data[24] = dc;
        data[23] = -s0;
        let mut c0 = qc * T::R37_EVEN_TWIDDLE_6;
        let mut s0 = s_buffer[0];
        s0 *= T::R37_ODD_TWIDDLE_12;
        c0 = fmla(ci1, T::R37_EVEN_TWIDDLE_17, c0);
        s0 = fmla(si1, -T::R37_ODD_TWIDDLE_0, s0);
        c0 = fmla(ci2, T::R37_EVEN_TWIDDLE_4, c0);
        s0 = fmla(si2, T::R37_ODD_TWIDDLE_11, s0);
        c0 = fmla(ci3, T::R37_EVEN_TWIDDLE_8, c0);
        s0 = fmla(si3, T::R37_ODD_TWIDDLE_13, s0);
        c0 = fmla(ci4, T::R37_EVEN_TWIDDLE_15, c0);
        s0 = fmla(si4, -T::R37_ODD_TWIDDLE_1, s0);
        c0 = fmla(ci5, T::R37_EVEN_TWIDDLE_2, c0);
        s0 = fmla(si5, T::R37_ODD_TWIDDLE_10, s0);
        c0 = fmla(ci6, T::R37_EVEN_TWIDDLE_10, c0);
        s0 = fmla(si6, T::R37_ODD_TWIDDLE_14, s0);
        c0 = fmla(ci7, T::R37_EVEN_TWIDDLE_13, c0);
        s0 = fmla(si7, -T::R37_ODD_TWIDDLE_2, s0);
        c0 = fmla(ci8, T::R37_EVEN_TWIDDLE_0, c0);
        s0 = fmla(si8, T::R37_ODD_TWIDDLE_9, s0);
        c0 = fmla(ci9, T::R37_EVEN_TWIDDLE_12, c0);
        s0 = fmla(si9, T::R37_ODD_TWIDDLE_15, s0);
        c0 = fmla(ci10, T::R37_EVEN_TWIDDLE_11, c0);
        s0 = fmla(si10, -T::R37_ODD_TWIDDLE_3, s0);
        c0 = fmla(ci11, T::R37_EVEN_TWIDDLE_1, c0);
        s0 = fmla(si11, T::R37_ODD_TWIDDLE_8, s0);
        c0 = fmla(ci12, T::R37_EVEN_TWIDDLE_14, c0);
        s0 = fmla(si12, T::R37_ODD_TWIDDLE_16, s0);
        c0 = fmla(ci13, T::R37_EVEN_TWIDDLE_9, c0);
        s0 = fmla(si13, -T::R37_ODD_TWIDDLE_4, s0);
        c0 = fmla(ci14, T::R37_EVEN_TWIDDLE_3, c0);
        s0 = fmla(si14, T::R37_ODD_TWIDDLE_7, s0);
        c0 = fmla(ci15, T::R37_EVEN_TWIDDLE_16, c0);
        s0 = fmla(si15, T::R37_ODD_TWIDDLE_17, s0);
        c0 = fmla(ci16, T::R37_EVEN_TWIDDLE_7, c0);
        s0 = fmla(si16, -T::R37_ODD_TWIDDLE_5, s0);
        c0 = fmla(ci17, T::R37_EVEN_TWIDDLE_5, c0);
        s0 = fmla(si17, T::R37_ODD_TWIDDLE_6, s0);
        let dc = c0 + a0;
        data[26] = -dc;
        data[25] = s0;
        let mut c0 = qc * T::R37_EVEN_TWIDDLE_11;
        let mut s0 = s_buffer[0];
        s0 *= -T::R37_ODD_TWIDDLE_13;
        c0 = fmla(ci1, T::R37_EVEN_TWIDDLE_2, c0);
        s0 = fmla(si1, T::R37_ODD_TWIDDLE_3, s0);
        c0 = fmla(ci2, T::R37_EVEN_TWIDDLE_16, c0);
        s0 = fmla(si2, -T::R37_ODD_TWIDDLE_6, s0);
        c0 = fmla(ci3, T::R37_EVEN_TWIDDLE_6, c0);
        s0 = fmla(si3, T::R37_ODD_TWIDDLE_16, s0);
        c0 = fmla(ci4, T::R37_EVEN_TWIDDLE_7, c0);
        s0 = fmla(si4, T::R37_ODD_TWIDDLE_10, s0);
        c0 = fmla(ci5, T::R37_EVEN_TWIDDLE_15, c0);
        s0 = fmla(si5, -T::R37_ODD_TWIDDLE_0, s0);
        c0 = fmla(ci6, T::R37_EVEN_TWIDDLE_1, c0);
        s0 = fmla(si6, T::R37_ODD_TWIDDLE_9, s0);
        c0 = fmla(ci7, T::R37_EVEN_TWIDDLE_12, c0);
        s0 = fmla(si7, T::R37_ODD_TWIDDLE_17, s0);
        c0 = fmla(ci8, T::R37_EVEN_TWIDDLE_10, c0);
        s0 = fmla(si8, -T::R37_ODD_TWIDDLE_7, s0);
        c0 = fmla(ci9, T::R37_EVEN_TWIDDLE_3, c0);
        s0 = fmla(si9, T::R37_ODD_TWIDDLE_2, s0);
        c0 = fmla(ci10, T::R37_EVEN_TWIDDLE_17, c0);
        s0 = fmla(si10, -T::R37_ODD_TWIDDLE_12, s0);
        c0 = fmla(ci11, T::R37_EVEN_TWIDDLE_5, c0);
        s0 = fmla(si11, -T::R37_ODD_TWIDDLE_14, s0);
        c0 = fmla(ci12, T::R37_EVEN_TWIDDLE_8, c0);
        s0 = fmla(si12, T::R37_ODD_TWIDDLE_4, s0);
        c0 = fmla(ci13, T::R37_EVEN_TWIDDLE_14, c0);
        s0 = fmla(si13, -T::R37_ODD_TWIDDLE_5, s0);
        c0 = fmla(ci14, T::R37_EVEN_TWIDDLE_0, c0);
        s0 = fmla(si14, T::R37_ODD_TWIDDLE_15, s0);
        c0 = fmla(ci15, T::R37_EVEN_TWIDDLE_13, c0);
        s0 = fmla(si15, T::R37_ODD_TWIDDLE_11, s0);
        c0 = fmla(ci16, T::R37_EVEN_TWIDDLE_9, c0);
        s0 = fmla(si16, -T::R37_ODD_TWIDDLE_1, s0);
        c0 = fmla(ci17, T::R37_EVEN_TWIDDLE_4, c0);
        s0 = fmla(si17, T::R37_ODD_TWIDDLE_8, s0);
        let dc = c0 + a0;
        data[28] = dc;
        data[27] = -s0;
        let mut c0 = qc * T::R37_EVEN_TWIDDLE_7;
        let mut s0 = s_buffer[0];
        s0 *= T::R37_ODD_TWIDDLE_14;
        c0 = fmla(ci1, T::R37_EVEN_TWIDDLE_14, c0);
        s0 = fmla(si1, -T::R37_ODD_TWIDDLE_6, s0);
        c0 = fmla(ci2, T::R37_EVEN_TWIDDLE_0, c0);
        s0 = fmla(si2, T::R37_ODD_TWIDDLE_1, s0);
        c0 = fmla(ci3, T::R37_EVEN_TWIDDLE_15, c0);
        s0 = fmla(si3, -T::R37_ODD_TWIDDLE_9, s0);
        c0 = fmla(ci4, T::R37_EVEN_TWIDDLE_6, c0);
        s0 = fmla(si4, T::R37_ODD_TWIDDLE_17, s0);
        c0 = fmla(ci5, T::R37_EVEN_TWIDDLE_8, c0);
        s0 = fmla(si5, T::R37_ODD_TWIDDLE_11, s0);
        c0 = fmla(ci6, T::R37_EVEN_TWIDDLE_13, c0);
        s0 = fmla(si6, -T::R37_ODD_TWIDDLE_3, s0);
        c0 = fmla(ci7, T::R37_EVEN_TWIDDLE_1, c0);
        s0 = fmla(si7, T::R37_ODD_TWIDDLE_4, s0);
        c0 = fmla(ci8, T::R37_EVEN_TWIDDLE_16, c0);
        s0 = fmla(si8, -T::R37_ODD_TWIDDLE_12, s0);
        c0 = fmla(ci9, T::R37_EVEN_TWIDDLE_5, c0);
        s0 = fmla(si9, -T::R37_ODD_TWIDDLE_16, s0);
        c0 = fmla(ci10, T::R37_EVEN_TWIDDLE_9, c0);
        s0 = fmla(si10, T::R37_ODD_TWIDDLE_8, s0);
        c0 = fmla(ci11, T::R37_EVEN_TWIDDLE_12, c0);
        s0 = fmla(si11, -T::R37_ODD_TWIDDLE_0, s0);
        c0 = fmla(ci12, T::R37_EVEN_TWIDDLE_2, c0);
        s0 = fmla(si12, T::R37_ODD_TWIDDLE_7, s0);
        c0 = fmla(ci13, T::R37_EVEN_TWIDDLE_17, c0);
        s0 = fmla(si13, -T::R37_ODD_TWIDDLE_15, s0);
        c0 = fmla(ci14, T::R37_EVEN_TWIDDLE_4, c0);
        s0 = fmla(si14, -T::R37_ODD_TWIDDLE_13, s0);
        c0 = fmla(ci15, T::R37_EVEN_TWIDDLE_10, c0);
        s0 = fmla(si15, T::R37_ODD_TWIDDLE_5, s0);
        c0 = fmla(ci16, T::R37_EVEN_TWIDDLE_11, c0);
        s0 = fmla(si16, -T::R37_ODD_TWIDDLE_2, s0);
        c0 = fmla(ci17, T::R37_EVEN_TWIDDLE_3, c0);
        s0 = fmla(si17, T::R37_ODD_TWIDDLE_10, s0);
        let dc = c0 + a0;
        data[30] = -dc;
        data[29] = s0;
        let mut c0 = qc * T::R37_EVEN_TWIDDLE_10;
        let mut s0 = s_buffer[0];
        s0 *= -T::R37_ODD_TWIDDLE_15;
        c0 = fmla(ci1, T::R37_EVEN_TWIDDLE_5, c0);
        s0 = fmla(si1, T::R37_ODD_TWIDDLE_9, s0);
        c0 = fmla(ci2, T::R37_EVEN_TWIDDLE_15, c0);
        s0 = fmla(si2, -T::R37_ODD_TWIDDLE_3, s0);
        c0 = fmla(ci3, T::R37_EVEN_TWIDDLE_0, c0);
        s0 = fmla(si3, T::R37_ODD_TWIDDLE_2, s0);
        c0 = fmla(ci4, T::R37_EVEN_TWIDDLE_16, c0);
        s0 = fmla(si4, -T::R37_ODD_TWIDDLE_8, s0);
        c0 = fmla(ci5, T::R37_EVEN_TWIDDLE_4, c0);
        s0 = fmla(si5, T::R37_ODD_TWIDDLE_14, s0);
        c0 = fmla(ci6, T::R37_EVEN_TWIDDLE_11, c0);
        s0 = fmla(si6, T::R37_ODD_TWIDDLE_16, s0);
        c0 = fmla(ci7, T::R37_EVEN_TWIDDLE_9, c0);
        s0 = fmla(si7, -T::R37_ODD_TWIDDLE_10, s0);
        c0 = fmla(ci8, T::R37_EVEN_TWIDDLE_6, c0);
        s0 = fmla(si8, T::R37_ODD_TWIDDLE_4, s0);
        c0 = fmla(ci9, T::R37_EVEN_TWIDDLE_14, c0);
        s0 = fmla(si9, -T::R37_ODD_TWIDDLE_1, s0);
        c0 = fmla(ci10, T::R37_EVEN_TWIDDLE_1, c0);
        s0 = fmla(si10, T::R37_ODD_TWIDDLE_7, s0);
        c0 = fmla(ci11, T::R37_EVEN_TWIDDLE_17, c0);
        s0 = fmla(si11, -T::R37_ODD_TWIDDLE_13, s0);
        c0 = fmla(ci12, T::R37_EVEN_TWIDDLE_3, c0);
        s0 = fmla(si12, -T::R37_ODD_TWIDDLE_17, s0);
        c0 = fmla(ci13, T::R37_EVEN_TWIDDLE_12, c0);
        s0 = fmla(si13, T::R37_ODD_TWIDDLE_11, s0);
        c0 = fmla(ci14, T::R37_EVEN_TWIDDLE_8, c0);
        s0 = fmla(si14, -T::R37_ODD_TWIDDLE_5, s0);
        c0 = fmla(ci15, T::R37_EVEN_TWIDDLE_7, c0);
        s0 = fmla(si15, T::R37_ODD_TWIDDLE_0, s0);
        c0 = fmla(ci16, T::R37_EVEN_TWIDDLE_13, c0);
        s0 = fmla(si16, -T::R37_ODD_TWIDDLE_6, s0);
        c0 = fmla(ci17, T::R37_EVEN_TWIDDLE_2, c0);
        s0 = fmla(si17, T::R37_ODD_TWIDDLE_12, s0);
        let dc = c0 + a0;
        data[32] = dc;
        data[31] = -s0;
        let mut c0 = qc * T::R37_EVEN_TWIDDLE_8;
        let mut s0 = s_buffer[0];
        s0 *= T::R37_ODD_TWIDDLE_16;
        c0 = fmla(ci1, T::R37_EVEN_TWIDDLE_11, c0);
        s0 = fmla(si1, -T::R37_ODD_TWIDDLE_12, s0);
        c0 = fmla(ci2, T::R37_EVEN_TWIDDLE_5, c0);
        s0 = fmla(si2, T::R37_ODD_TWIDDLE_8, s0);
        c0 = fmla(ci3, T::R37_EVEN_TWIDDLE_14, c0);
        s0 = fmla(si3, -T::R37_ODD_TWIDDLE_4, s0);
        c0 = fmla(ci4, T::R37_EVEN_TWIDDLE_2, c0);
        s0 = fmla(si4, T::R37_ODD_TWIDDLE_0, s0);
        c0 = fmla(ci5, T::R37_EVEN_TWIDDLE_17, c0);
        s0 = fmla(si5, -T::R37_ODD_TWIDDLE_3, s0);
        c0 = fmla(ci6, T::R37_EVEN_TWIDDLE_0, c0);
        s0 = fmla(si6, T::R37_ODD_TWIDDLE_7, s0);
        c0 = fmla(ci7, T::R37_EVEN_TWIDDLE_16, c0);
        s0 = fmla(si7, -T::R37_ODD_TWIDDLE_11, s0);
        c0 = fmla(ci8, T::R37_EVEN_TWIDDLE_3, c0);
        s0 = fmla(si8, T::R37_ODD_TWIDDLE_15, s0);
        c0 = fmla(ci9, T::R37_EVEN_TWIDDLE_13, c0);
        s0 = fmla(si9, T::R37_ODD_TWIDDLE_17, s0);
        c0 = fmla(ci10, T::R37_EVEN_TWIDDLE_6, c0);
        s0 = fmla(si10, -T::R37_ODD_TWIDDLE_13, s0);
        c0 = fmla(ci11, T::R37_EVEN_TWIDDLE_10, c0);
        s0 = fmla(si11, T::R37_ODD_TWIDDLE_9, s0);
        c0 = fmla(ci12, T::R37_EVEN_TWIDDLE_9, c0);
        s0 = fmla(si12, -T::R37_ODD_TWIDDLE_5, s0);
        c0 = fmla(ci13, T::R37_EVEN_TWIDDLE_7, c0);
        s0 = fmla(si13, T::R37_ODD_TWIDDLE_1, s0);
        c0 = fmla(ci14, T::R37_EVEN_TWIDDLE_12, c0);
        s0 = fmla(si14, -T::R37_ODD_TWIDDLE_2, s0);
        c0 = fmla(ci15, T::R37_EVEN_TWIDDLE_4, c0);
        s0 = fmla(si15, T::R37_ODD_TWIDDLE_6, s0);
        c0 = fmla(ci16, T::R37_EVEN_TWIDDLE_15, c0);
        s0 = fmla(si16, -T::R37_ODD_TWIDDLE_10, s0);
        c0 = fmla(ci17, T::R37_EVEN_TWIDDLE_1, c0);
        s0 = fmla(si17, T::R37_ODD_TWIDDLE_14, s0);
        let dc = c0 + a0;
        data[34] = -dc;
        data[33] = s0;
        let mut c0 = qc * T::R37_EVEN_TWIDDLE_9;
        let mut s0 = s_buffer[0];
        s0 *= -T::R37_ODD_TWIDDLE_17;
        c0 = fmla(ci1, T::R37_EVEN_TWIDDLE_8, c0);
        s0 = fmla(si1, T::R37_ODD_TWIDDLE_15, s0);
        c0 = fmla(ci2, T::R37_EVEN_TWIDDLE_10, c0);
        s0 = fmla(si2, -T::R37_ODD_TWIDDLE_13, s0);
        c0 = fmla(ci3, T::R37_EVEN_TWIDDLE_7, c0);
        s0 = fmla(si3, T::R37_ODD_TWIDDLE_11, s0);
        c0 = fmla(ci4, T::R37_EVEN_TWIDDLE_11, c0);
        s0 = fmla(si4, -T::R37_ODD_TWIDDLE_9, s0);
        c0 = fmla(ci5, T::R37_EVEN_TWIDDLE_6, c0);
        s0 = fmla(si5, T::R37_ODD_TWIDDLE_7, s0);
        c0 = fmla(ci6, T::R37_EVEN_TWIDDLE_12, c0);
        s0 = fmla(si6, -T::R37_ODD_TWIDDLE_5, s0);
        c0 = fmla(ci7, T::R37_EVEN_TWIDDLE_5, c0);
        s0 = fmla(si7, T::R37_ODD_TWIDDLE_3, s0);
        c0 = fmla(ci8, T::R37_EVEN_TWIDDLE_13, c0);
        s0 = fmla(si8, -T::R37_ODD_TWIDDLE_1, s0);
        c0 = fmla(ci9, T::R37_EVEN_TWIDDLE_4, c0);
        s0 = fmla(si9, T::R37_ODD_TWIDDLE_0, s0);
        c0 = fmla(ci10, T::R37_EVEN_TWIDDLE_14, c0);
        s0 = fmla(si10, -T::R37_ODD_TWIDDLE_2, s0);
        c0 = fmla(ci11, T::R37_EVEN_TWIDDLE_3, c0);
        s0 = fmla(si11, T::R37_ODD_TWIDDLE_4, s0);
        c0 = fmla(ci12, T::R37_EVEN_TWIDDLE_15, c0);
        s0 = fmla(si12, -T::R37_ODD_TWIDDLE_6, s0);
        c0 = fmla(ci13, T::R37_EVEN_TWIDDLE_2, c0);
        s0 = fmla(si13, T::R37_ODD_TWIDDLE_8, s0);
        c0 = fmla(ci14, T::R37_EVEN_TWIDDLE_16, c0);
        s0 = fmla(si14, -T::R37_ODD_TWIDDLE_10, s0);
        c0 = fmla(ci15, T::R37_EVEN_TWIDDLE_1, c0);
        s0 = fmla(si15, T::R37_ODD_TWIDDLE_12, s0);
        c0 = fmla(ci16, T::R37_EVEN_TWIDDLE_17, c0);
        s0 = fmla(si16, -T::R37_ODD_TWIDDLE_14, s0);
        c0 = fmla(ci17, T::R37_EVEN_TWIDDLE_0, c0);
        s0 = fmla(si17, T::R37_ODD_TWIDDLE_16, s0);
        let dc = c0 + a0;
        data[36] = dc;
        data[35] = -s0;
    }
}

impl<T: DctSample + MixedRadix37Sample> PxdctExecutor<T> for Dct2Butterfly37<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, data: &mut [T]) -> Result<(), PxdctError> {
        if !data.len().is_multiple_of(37) {
            return Err(PxdctError::InvalidSizeMultiplier(data.len(), self.length()));
        }
        for chunk in data.chunks_exact_mut(37) {
            self.exec(&mut InPlaceStore::new(chunk));
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
        validate_oof_sizes!(input, output, 37);
        use crate::bidirectional::BiStore;
        for (src, dst) in input.chunks_exact(37).zip(output.chunks_exact_mut(37)) {
            self.exec(&mut BiStore::new(src, dst));
        }
        Ok(())
    }

    fn length(&self) -> usize {
        37
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
    use rand::RngExt;

    gen_test_butterfly!(test_bf3, f64, Dct2Butterfly3, 3, 1e-7, naive_dct2);
    gen_test_butterfly!(test_bf5, f64, Dct2Butterfly5, 5, 1e-7, naive_dct2);
    gen_test_butterfly!(test_bf7, f64, Dct2Butterfly7, 7, 1e-7, naive_dct2);
    gen_test_butterfly!(test_bf11, f64, Dct2Butterfly11, 11, 1e-7, naive_dct2);
    gen_test_butterfly!(test_bf13, f64, Dct2Butterfly13, 13, 1e-7, naive_dct2);
    gen_test_butterfly!(test_bf17, f64, Dct2Butterfly17, 17, 1e-7, naive_dct2);
    gen_test_butterfly!(test_bf19, f64, Dct2Butterfly19, 19, 1e-7, naive_dct2);
    gen_test_butterfly!(test_bf23, f64, Dct2Butterfly23, 23, 1e-7, naive_dct2);
    gen_test_butterfly!(test_bf29, f64, Dct2Butterfly29, 29, 1e-7, naive_dct2);
    gen_test_butterfly!(test_bf37, f64, Dct2Butterfly37, 37, 1e-7, naive_dct2);
}
