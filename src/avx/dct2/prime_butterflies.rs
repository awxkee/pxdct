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
use crate::avx::util::{define_avx_butterfly, fma};
use crate::bidirectional::{BidirectionalStore, InPlaceStore};
use crate::dct2::prime_butterflies::{MixedRadix11Sample, MixedRadix31Sample, MixedRadix37Sample};
use crate::twiddles::compute_twiddle;
use crate::util::DctSample;
use crate::{PxdctError, PxdctExecutor};
use num_traits::AsPrimitive;

#[derive(Debug, Clone)]
pub(crate) struct AvxDct2Butterfly3<T: DctSample> {
    twiddle: T,
}

impl<T: DctSample> Default for AvxDct2Butterfly3<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            twiddle: compute_twiddle(1, 12).re,
        }
    }
}

impl<T: DctSample> AvxDct2Butterfly3<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        let u0 = data[0];
        let u1 = data[1];
        let u2 = data[2];

        data[0] = u0 + u1 + u2;
        data[1] = (u0 - u2) * self.twiddle;
        data[2] = fma(u0 + u2, T::HALF, -u1);
    }
}

define_avx_butterfly!(AvxDct2Butterfly3, 3);

#[derive(Debug, Clone)]
pub(crate) struct AvxDct2Butterfly5<T: DctSample> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T: DctSample> Default for AvxDct2Butterfly5<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: DctSample> AvxDct2Butterfly5<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        // Radix-q for size 5
        let a_buffer = [data[2]];
        let c_buffer = [data[0] + data[4], data[1] + data[3]];
        let s_buffer = [data[0] - data[4], data[1] - data[3]];

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
            c1 = fma(twiddle_ci, T::R5_COS_EVEN4_M0, c1);
            c2 = fma(twiddle_ci, T::R5_COS_EVEN2_M0, c2);
            s0 = fma(twiddle_si, -T::R5_SIN_ODD1_M0, s0);
            s1 = fma(twiddle_si, T::R5_SIN_ODD_M0, s1);
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

define_avx_butterfly!(AvxDct2Butterfly5, 5);

#[derive(Debug, Clone)]
pub(crate) struct AvxDct2Butterfly7<T: DctSample> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T: DctSample> Default for AvxDct2Butterfly7<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: DctSample> AvxDct2Butterfly7<T>
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
        let mut c3 = qc * T::R7_COS_EVEN2_M1; // Component C6 (position 4, uses j=6)

        let s0_twiddled = s_buffer[0];

        let mut s0 = s0_twiddled * T::R7_SIN_ODD0_M0;
        let mut s1 = s0_twiddled * T::R7_SIN_ODD1_M0;
        let mut s2 = s0_twiddled * T::R7_SIN_ODD2_M0;

        let ci = c_buffer[1];
        let si = s_buffer[1];

        let ci2 = c_buffer[2];
        let si2 = s_buffer[2];

        c0 = ci + c0 + ci2;

        c1 = fma(ci, T::R7_COS_EVEN2_M1, c1);
        c1 = fma(ci2, T::R7_COS_EVEN2_M2, c1);

        c2 = fma(ci, T::R7_COS_EVEN2_M0, c2);
        c2 = fma(ci2, T::R7_COS_EVEN2_M1, c2);

        c3 = fma(ci, T::R7_COS_EVEN2_M2, c3);
        c3 = fma(ci2, T::R7_COS_EVEN2_M0, c3);

        s0 = fma(si, T::R7_SIN_ODD0_M1, s0);
        s0 = fma(si2, T::R7_SIN_ODD0_M2, s0);

        s1 = fma(si, T::R7_SIN_ODD1_M1, s1);
        s1 = fma(si2, T::R7_SIN_ODD1_M2, s1);

        s2 = fma(si, T::R7_SIN_ODD2_M1, s2);
        s2 = fma(si2, T::R7_SIN_ODD2_M2, s2);

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

define_avx_butterfly!(AvxDct2Butterfly7, 7);

#[derive(Debug, Clone)]
pub(crate) struct AvxDct2Butterfly11<T: DctSample> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T: DctSample> Default for AvxDct2Butterfly11<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: DctSample + MixedRadix11Sample> AvxDct2Butterfly11<T>
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
        c1 = fma(ci1, T::R11_EVEN_TWIDDLE_1, c1);
        s0 = fma(si1, T::R11_ODD_TWIDDLE_1, s0);
        let ci2 = c_buffer[2];
        let si2 = s_buffer[2];
        c0 = ci2 + c0;
        c1 = fma(ci2, T::R11_EVEN_TWIDDLE_2, c1);
        s0 = fma(si2, T::R11_ODD_TWIDDLE_2, s0);
        let ci3 = c_buffer[3];
        let si3 = s_buffer[3];
        c0 = ci3 + c0;
        c1 = fma(ci3, T::R11_EVEN_TWIDDLE_3, c1);
        s0 = fma(si3, T::R11_ODD_TWIDDLE_3, s0);
        let ci4 = c_buffer[4];
        let si4 = s_buffer[4];
        c0 = ci4 + c0;
        c1 = fma(ci4, T::R11_EVEN_TWIDDLE_4, c1);
        s0 = fma(si4, T::R11_ODD_TWIDDLE_4, s0);
        let a0 = a_buffer[0];
        let dc = c0 + a0;
        data[0] = dc;
        data[1] = s0;
        data[2] = -(c1 + a0);
        let mut c0 = qc * T::R11_EVEN_TWIDDLE_4;
        let mut s0 = s_buffer[0];
        s0 *= -T::R11_ODD_TWIDDLE_1;
        c0 = fma(ci1, T::R11_EVEN_TWIDDLE_2, c0);
        s0 = fma(si1, -T::R11_ODD_TWIDDLE_4, s0);
        c0 = fma(ci2, T::R11_EVEN_TWIDDLE_0, c0);
        s0 = fma(si2, T::R11_ODD_TWIDDLE_3, s0);
        c0 = fma(ci3, T::R11_EVEN_TWIDDLE_1, c0);
        s0 = fma(si3, T::R11_ODD_TWIDDLE_0, s0);
        c0 = fma(ci4, T::R11_EVEN_TWIDDLE_3, c0);
        s0 = fma(si4, T::R11_ODD_TWIDDLE_2, s0);
        let dc = c0 + a0;
        data[4] = dc;
        data[3] = -s0;
        let mut c0 = qc * T::R11_EVEN_TWIDDLE_1;
        let mut s0 = s_buffer[0];
        s0 *= T::R11_ODD_TWIDDLE_2;
        c0 = fma(ci1, T::R11_EVEN_TWIDDLE_4, c0);
        s0 = fma(si1, -T::R11_ODD_TWIDDLE_3, s0);
        c0 = fma(ci2, T::R11_EVEN_TWIDDLE_3, c0);
        s0 = fma(si2, -T::R11_ODD_TWIDDLE_1, s0);
        c0 = fma(ci3, T::R11_EVEN_TWIDDLE_0, c0);
        s0 = fma(si3, T::R11_ODD_TWIDDLE_4, s0);
        c0 = fma(ci4, T::R11_EVEN_TWIDDLE_2, c0);
        s0 = fma(si4, T::R11_ODD_TWIDDLE_0, s0);
        let dc = c0 + a0;
        data[6] = -dc;
        data[5] = s0;
        let mut c0 = qc * T::R11_EVEN_TWIDDLE_3;
        let mut s0 = s_buffer[0];
        s0 *= -T::R11_ODD_TWIDDLE_3;
        c0 = fma(ci1, T::R11_EVEN_TWIDDLE_0, c0);
        s0 = fma(si1, T::R11_ODD_TWIDDLE_0, s0);
        c0 = fma(ci2, T::R11_EVEN_TWIDDLE_4, c0);
        s0 = fma(si2, -T::R11_ODD_TWIDDLE_4, s0);
        c0 = fma(ci3, T::R11_EVEN_TWIDDLE_2, c0);
        s0 = fma(si3, -T::R11_ODD_TWIDDLE_2, s0);
        c0 = fma(ci4, T::R11_EVEN_TWIDDLE_1, c0);
        s0 = fma(si4, T::R11_ODD_TWIDDLE_1, s0);
        let dc = c0 + a0;
        data[8] = dc;
        data[7] = -s0;
        let mut c0 = qc * T::R11_EVEN_TWIDDLE_2;
        let mut s0 = s_buffer[0];
        s0 *= T::R11_ODD_TWIDDLE_4;
        c0 = fma(ci1, T::R11_EVEN_TWIDDLE_3, c0);
        s0 = fma(si1, -T::R11_ODD_TWIDDLE_2, s0);
        c0 = fma(ci2, T::R11_EVEN_TWIDDLE_1, c0);
        s0 = fma(si2, T::R11_ODD_TWIDDLE_0, s0);
        c0 = fma(ci3, T::R11_EVEN_TWIDDLE_4, c0);
        s0 = fma(si3, -T::R11_ODD_TWIDDLE_1, s0);
        c0 = fma(ci4, T::R11_EVEN_TWIDDLE_0, c0);
        s0 = fma(si4, T::R11_ODD_TWIDDLE_3, s0);
        let dc = c0 + a0;
        data[10] = -dc;
        data[9] = s0;
    }
}

impl<T: DctSample + MixedRadix11Sample> AvxDct2Butterfly11<T>
where
    f64: AsPrimitive<T>,
{
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_impl(&self, data: &mut [T]) -> Result<(), PxdctError> {
        if !data.len().is_multiple_of(11) {
            return Err(PxdctError::InvalidSizeMultiplier(data.len(), self.length()));
        }
        for chunk in data.chunks_exact_mut(11) {
            self.exec(&mut InPlaceStore::new(chunk));
        }
        Ok(())
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_into_impl(&self, input: &[T], output: &mut [T]) -> Result<(), PxdctError> {
        use crate::util::validate_oof_sizes;
        validate_oof_sizes!(input, output, 11);
        use crate::bidirectional::BiStore;
        for (src, dst) in input.chunks_exact(11).zip(output.chunks_exact_mut(11)) {
            self.exec(&mut BiStore::new(src, dst));
        }
        Ok(())
    }
}

impl<T: DctSample + MixedRadix11Sample> PxdctExecutor<T> for AvxDct2Butterfly11<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, data: &mut [T]) -> Result<(), PxdctError> {
        unsafe { self.execute_impl(data) }
    }

    fn execute_with_scratch(&self, data: &mut [T], _: &mut [T]) -> Result<(), PxdctError> {
        unsafe { self.execute_impl(data) }
    }

    fn execute_into(&self, input: &[T], output: &mut [T]) -> Result<(), PxdctError> {
        unsafe { self.execute_into_impl(input, output) }
    }

    fn execute_into_with_scratch(
        &self,
        input: &[T],
        output: &mut [T],
        _: &mut [T],
    ) -> Result<(), PxdctError> {
        unsafe { self.execute_into_impl(input, output) }
    }

    fn length(&self) -> usize {
        11
    }

    fn scratch_size(&self) -> usize {
        0
    }
}

#[derive(Debug, Clone)]
pub(crate) struct AvxDct2Butterfly13<T: DctSample> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T: DctSample> Default for AvxDct2Butterfly13<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: DctSample> AvxDct2Butterfly13<T>
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
        c1 = fma(ci1, T::R13_EVEN_TWIDDLE_1, c1);
        s0 = fma(si1, T::R13_ODD_TWIDDLE_1, s0);
        let ci2 = c_buffer[2];
        let si2 = s_buffer[2];
        c0 = ci2 + c0;
        c1 = fma(ci2, T::R13_EVEN_TWIDDLE_2, c1);
        s0 = fma(si2, T::R13_ODD_TWIDDLE_2, s0);
        let ci3 = c_buffer[3];
        let si3 = s_buffer[3];
        c0 = ci3 + c0;
        c1 = fma(ci3, T::R13_EVEN_TWIDDLE_3, c1);
        s0 = fma(si3, T::R13_ODD_TWIDDLE_3, s0);
        let ci4 = c_buffer[4];
        let si4 = s_buffer[4];
        c0 = ci4 + c0;
        c1 = fma(ci4, T::R13_EVEN_TWIDDLE_4, c1);
        s0 = fma(si4, T::R13_ODD_TWIDDLE_4, s0);
        let ci5 = c_buffer[5];
        let si5 = s_buffer[5];
        c0 = ci5 + c0;
        c1 = fma(ci5, T::R13_EVEN_TWIDDLE_5, c1);
        s0 = fma(si5, T::R13_ODD_TWIDDLE_5, s0);
        let a0 = a_buffer[0];
        let dc = c0 + a0;
        data[0] = dc;
        data[1] = s0;
        data[2] = -(c1 + a0);
        let mut c0 = qc * T::R13_EVEN_TWIDDLE_5;
        let mut s0 = s_buffer[0];
        s0 *= -T::R13_ODD_TWIDDLE_1;
        c0 = fma(ci1, T::R13_EVEN_TWIDDLE_3, c0);
        s0 = fma(si1, -T::R13_ODD_TWIDDLE_4, s0);
        c0 = fma(ci2, T::R13_EVEN_TWIDDLE_1, c0);
        s0 = fma(si2, T::R13_ODD_TWIDDLE_5, s0);
        c0 = fma(ci3, T::R13_EVEN_TWIDDLE_0, c0);
        s0 = fma(si3, T::R13_ODD_TWIDDLE_2, s0);
        c0 = fma(ci4, T::R13_EVEN_TWIDDLE_2, c0);
        s0 = fma(si4, T::R13_ODD_TWIDDLE_0, s0);
        c0 = fma(ci5, T::R13_EVEN_TWIDDLE_4, c0);
        s0 = fma(si5, T::R13_ODD_TWIDDLE_3, s0);
        let dc = c0 + a0;
        data[4] = dc;
        data[3] = -s0;
        let mut c0 = qc * T::R13_EVEN_TWIDDLE_1;
        let mut s0 = s_buffer[0];
        s0 *= T::R13_ODD_TWIDDLE_2;
        c0 = fma(ci1, T::R13_EVEN_TWIDDLE_4, c0);
        s0 = fma(si1, -T::R13_ODD_TWIDDLE_5, s0);
        c0 = fma(ci2, T::R13_EVEN_TWIDDLE_5, c0);
        s0 = fma(si2, -T::R13_ODD_TWIDDLE_0, s0);
        c0 = fma(ci3, T::R13_EVEN_TWIDDLE_2, c0);
        s0 = fma(si3, -T::R13_ODD_TWIDDLE_4, s0);
        c0 = fma(ci4, T::R13_EVEN_TWIDDLE_0, c0);
        s0 = fma(si4, T::R13_ODD_TWIDDLE_3, s0);
        c0 = fma(ci5, T::R13_EVEN_TWIDDLE_3, c0);
        s0 = fma(si5, T::R13_ODD_TWIDDLE_1, s0);
        let dc = c0 + a0;
        data[6] = -dc;
        data[5] = s0;
        let mut c0 = qc * T::R13_EVEN_TWIDDLE_4;
        let mut s0 = s_buffer[0];
        s0 *= -T::R13_ODD_TWIDDLE_3;
        c0 = fma(ci1, T::R13_EVEN_TWIDDLE_0, c0);
        s0 = fma(si1, T::R13_ODD_TWIDDLE_2, s0);
        c0 = fma(ci2, T::R13_EVEN_TWIDDLE_3, c0);
        s0 = fma(si2, T::R13_ODD_TWIDDLE_4, s0);
        c0 = fma(ci3, T::R13_EVEN_TWIDDLE_5, c0);
        s0 = fma(si3, -T::R13_ODD_TWIDDLE_1, s0);
        c0 = fma(ci4, T::R13_EVEN_TWIDDLE_1, c0);
        s0 = fma(si4, -T::R13_ODD_TWIDDLE_5, s0);
        c0 = fma(ci5, T::R13_EVEN_TWIDDLE_2, c0);
        s0 = fma(si5, T::R13_ODD_TWIDDLE_0, s0);
        let dc = c0 + a0;
        data[8] = dc;
        data[7] = -s0;
        let mut c0 = qc * T::R13_EVEN_TWIDDLE_2;
        let mut s0 = s_buffer[0];
        s0 *= T::R13_ODD_TWIDDLE_4;
        c0 = fma(ci1, T::R13_EVEN_TWIDDLE_5, c0);
        s0 = fma(si1, -T::R13_ODD_TWIDDLE_0, s0);
        c0 = fma(ci2, T::R13_EVEN_TWIDDLE_0, c0);
        s0 = fma(si2, T::R13_ODD_TWIDDLE_3, s0);
        c0 = fma(ci3, T::R13_EVEN_TWIDDLE_4, c0);
        s0 = fma(si3, T::R13_ODD_TWIDDLE_5, s0);
        c0 = fma(ci4, T::R13_EVEN_TWIDDLE_3, c0);
        s0 = fma(si4, -T::R13_ODD_TWIDDLE_1, s0);
        c0 = fma(ci5, T::R13_EVEN_TWIDDLE_1, c0);
        s0 = fma(si5, T::R13_ODD_TWIDDLE_2, s0);
        let dc = c0 + a0;
        data[10] = -dc;
        data[9] = s0;
        let mut c0 = qc * T::R13_EVEN_TWIDDLE_3;
        let mut s0 = s_buffer[0];
        s0 *= -T::R13_ODD_TWIDDLE_5;
        c0 = fma(ci1, T::R13_EVEN_TWIDDLE_2, c0);
        s0 = fma(si1, T::R13_ODD_TWIDDLE_3, s0);
        c0 = fma(ci2, T::R13_EVEN_TWIDDLE_4, c0);
        s0 = fma(si2, -T::R13_ODD_TWIDDLE_1, s0);
        c0 = fma(ci3, T::R13_EVEN_TWIDDLE_1, c0);
        s0 = fma(si3, T::R13_ODD_TWIDDLE_0, s0);
        c0 = fma(ci4, T::R13_EVEN_TWIDDLE_5, c0);
        s0 = fma(si4, -T::R13_ODD_TWIDDLE_2, s0);
        c0 = fma(ci5, T::R13_EVEN_TWIDDLE_0, c0);
        s0 = fma(si5, T::R13_ODD_TWIDDLE_4, s0);
        let dc = c0 + a0;
        data[12] = dc;
        data[11] = -s0;
    }
}

define_avx_butterfly!(AvxDct2Butterfly13, 13);

#[derive(Debug, Clone)]
pub(crate) struct AvxDct2Butterfly17<T: DctSample> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T: DctSample> Default for AvxDct2Butterfly17<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: DctSample> AvxDct2Butterfly17<T>
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
        s0 = fma(si1, T::R17_ODD_TWIDDLE_1, s0);
        let ci2 = c_buffer[2];
        let si2 = s_buffer[2];
        c0 = ci2 + c0;
        c1 = ci2 * T::R17_EVEN_TWIDDLE_2 + c1;
        s0 = fma(si2, T::R17_ODD_TWIDDLE_2, s0);
        let ci3 = c_buffer[3];
        let si3 = s_buffer[3];
        c0 = ci3 + c0;
        c1 = ci3 * T::R17_EVEN_TWIDDLE_3 + c1;
        s0 = fma(si3, T::R17_ODD_TWIDDLE_3, s0);
        let ci4 = c_buffer[4];
        let si4 = s_buffer[4];
        c0 = ci4 + c0;
        c1 = ci4 * T::R17_EVEN_TWIDDLE_4 + c1;
        s0 = fma(si4, T::R17_ODD_TWIDDLE_4, s0);
        let ci5 = c_buffer[5];
        let si5 = s_buffer[5];
        c0 = ci5 + c0;
        c1 = ci5 * T::R17_EVEN_TWIDDLE_5 + c1;
        s0 = fma(si5, T::R17_ODD_TWIDDLE_5, s0);
        let ci6 = c_buffer[6];
        let si6 = s_buffer[6];
        c0 = ci6 + c0;
        c1 = ci6 * T::R17_EVEN_TWIDDLE_6 + c1;
        s0 = fma(si6, T::R17_ODD_TWIDDLE_6, s0);
        let ci7 = c_buffer[7];
        let si7 = s_buffer[7];
        c0 = ci7 + c0;
        c1 = ci7 * T::R17_EVEN_TWIDDLE_7 + c1;
        s0 = fma(si7, T::R17_ODD_TWIDDLE_7, s0);
        let a0 = a_buffer[0];
        let dc = c0 + a0;
        data[0] = dc;
        data[1] = s0;
        data[2] = -(c1 + a0);
        let mut c0 = qc * T::R17_EVEN_TWIDDLE_7;
        let mut s0 = s_buffer[0];
        s0 *= -T::R17_ODD_TWIDDLE_1;
        c0 = fma(ci1, T::R17_EVEN_TWIDDLE_5, c0);
        s0 = fma(si1, -T::R17_ODD_TWIDDLE_4, s0);
        c0 = fma(ci2, T::R17_EVEN_TWIDDLE_3, c0);
        s0 = fma(si2, -T::R17_ODD_TWIDDLE_7, s0);
        c0 = fma(ci3, T::R17_EVEN_TWIDDLE_1, c0);
        s0 = fma(si3, T::R17_ODD_TWIDDLE_6, s0);
        c0 = fma(ci4, T::R17_EVEN_TWIDDLE_0, c0);
        s0 = fma(si4, T::R17_ODD_TWIDDLE_3, s0);
        c0 = fma(ci5, T::R17_EVEN_TWIDDLE_2, c0);
        s0 = fma(si5, T::R17_ODD_TWIDDLE_0, s0);
        c0 = fma(ci6, T::R17_EVEN_TWIDDLE_4, c0);
        s0 = fma(si6, T::R17_ODD_TWIDDLE_2, s0);
        c0 = fma(ci7, T::R17_EVEN_TWIDDLE_6, c0);
        s0 = fma(si7, T::R17_ODD_TWIDDLE_5, s0);
        let dc = c0 + a0;
        data[4] = dc;
        data[3] = -s0;
        let mut c0 = qc * T::R17_EVEN_TWIDDLE_1;
        let mut s0 = s_buffer[0];
        s0 *= T::R17_ODD_TWIDDLE_2;
        c0 = fma(ci1, T::R17_EVEN_TWIDDLE_4, c0);
        s0 = fma(si1, T::R17_ODD_TWIDDLE_7, s0);
        c0 = fma(ci2, T::R17_EVEN_TWIDDLE_7, c0);
        s0 = fma(si2, -T::R17_ODD_TWIDDLE_4, s0);
        c0 = fma(ci3, T::R17_EVEN_TWIDDLE_6, c0);
        s0 = fma(si3, -T::R17_ODD_TWIDDLE_0, s0);
        c0 = fma(ci4, T::R17_EVEN_TWIDDLE_3, c0);
        s0 = fma(si4, -T::R17_ODD_TWIDDLE_5, s0);
        c0 = fma(ci5, T::R17_EVEN_TWIDDLE_0, c0);
        s0 = fma(si5, T::R17_ODD_TWIDDLE_6, s0);
        c0 = fma(ci6, T::R17_EVEN_TWIDDLE_2, c0);
        s0 = fma(si6, T::R17_ODD_TWIDDLE_1, s0);
        c0 = fma(ci7, T::R17_EVEN_TWIDDLE_5, c0);
        s0 = fma(si7, T::R17_ODD_TWIDDLE_3, s0);
        let dc = c0 + a0;
        data[6] = -dc;
        data[5] = s0;
        let mut c0 = qc * T::R17_EVEN_TWIDDLE_6;
        let mut s0 = s_buffer[0];
        s0 *= -T::R17_ODD_TWIDDLE_3;
        c0 = fma(ci1, T::R17_EVEN_TWIDDLE_2, c0);
        s0 = fma(si1, T::R17_ODD_TWIDDLE_6, s0);
        c0 = fma(ci2, T::R17_EVEN_TWIDDLE_1, c0);
        s0 = fma(si2, T::R17_ODD_TWIDDLE_0, s0);
        c0 = fma(ci3, T::R17_EVEN_TWIDDLE_5, c0);
        s0 = fma(si3, T::R17_ODD_TWIDDLE_7, s0);
        c0 = fma(ci4, T::R17_EVEN_TWIDDLE_7, c0);
        s0 = fma(si4, -T::R17_ODD_TWIDDLE_2, s0);
        c0 = fma(ci5, T::R17_EVEN_TWIDDLE_3, c0);
        s0 = fma(si5, -T::R17_ODD_TWIDDLE_4, s0);
        c0 = fma(ci6, T::R17_EVEN_TWIDDLE_0, c0);
        s0 = fma(si6, T::R17_ODD_TWIDDLE_5, s0);
        c0 = fma(ci7, T::R17_EVEN_TWIDDLE_4, c0);
        s0 = fma(si7, T::R17_ODD_TWIDDLE_1, s0);
        let dc = c0 + a0;
        data[8] = dc;
        data[7] = -s0;
        let mut c0 = qc * T::R17_EVEN_TWIDDLE_2;
        let mut s0 = s_buffer[0];
        s0 *= T::R17_ODD_TWIDDLE_4;
        c0 = fma(ci1, T::R17_EVEN_TWIDDLE_7, c0);
        s0 = fma(si1, -T::R17_ODD_TWIDDLE_3, s0);
        c0 = fma(ci2, T::R17_EVEN_TWIDDLE_4, c0);
        s0 = fma(si2, -T::R17_ODD_TWIDDLE_5, s0);
        c0 = fma(ci3, T::R17_EVEN_TWIDDLE_0, c0);
        s0 = fma(si3, T::R17_ODD_TWIDDLE_2, s0);
        c0 = fma(ci4, T::R17_EVEN_TWIDDLE_5, c0);
        s0 = fma(si4, T::R17_ODD_TWIDDLE_6, s0);
        c0 = fma(ci5, T::R17_EVEN_TWIDDLE_6, c0);
        s0 = fma(si5, -T::R17_ODD_TWIDDLE_1, s0);
        c0 = fma(ci6, T::R17_EVEN_TWIDDLE_1, c0);
        s0 = fma(si6, -T::R17_ODD_TWIDDLE_7, s0);
        c0 = fma(ci7, T::R17_EVEN_TWIDDLE_3, c0);
        s0 = fma(si7, T::R17_ODD_TWIDDLE_0, s0);
        let dc = c0 + a0;
        data[10] = -dc;
        data[9] = s0;
        let mut c0 = qc * T::R17_EVEN_TWIDDLE_5;
        let mut s0 = s_buffer[0];
        s0 *= -T::R17_ODD_TWIDDLE_5;
        c0 = fma(ci1, T::R17_EVEN_TWIDDLE_0, c0);
        s0 = fma(si1, T::R17_ODD_TWIDDLE_0, s0);
        c0 = fma(ci2, T::R17_EVEN_TWIDDLE_6, c0);
        s0 = fma(si2, -T::R17_ODD_TWIDDLE_6, s0);
        c0 = fma(ci3, T::R17_EVEN_TWIDDLE_4, c0);
        s0 = fma(si3, -T::R17_ODD_TWIDDLE_4, s0);
        c0 = fma(ci4, T::R17_EVEN_TWIDDLE_1, c0);
        s0 = fma(si4, T::R17_ODD_TWIDDLE_1, s0);
        c0 = fma(ci5, T::R17_EVEN_TWIDDLE_7, c0);
        s0 = fma(si5, -T::R17_ODD_TWIDDLE_7, s0);
        c0 = fma(ci6, T::R17_EVEN_TWIDDLE_3, c0);
        s0 = fma(si6, -T::R17_ODD_TWIDDLE_3, s0);
        c0 = fma(ci7, T::R17_EVEN_TWIDDLE_2, c0);
        s0 = fma(si7, T::R17_ODD_TWIDDLE_2, s0);
        let dc = c0 + a0;
        data[12] = dc;
        data[11] = -s0;
        let mut c0 = qc * T::R17_EVEN_TWIDDLE_3;
        let mut s0 = s_buffer[0];
        s0 *= T::R17_ODD_TWIDDLE_6;
        c0 = fma(ci1, T::R17_EVEN_TWIDDLE_6, c0);
        s0 = fma(si1, -T::R17_ODD_TWIDDLE_2, s0);
        c0 = fma(ci2, T::R17_EVEN_TWIDDLE_0, c0);
        s0 = fma(si2, T::R17_ODD_TWIDDLE_1, s0);
        c0 = fma(ci3, T::R17_EVEN_TWIDDLE_7, c0);
        s0 = fma(si3, -T::R17_ODD_TWIDDLE_5, s0);
        c0 = fma(ci4, T::R17_EVEN_TWIDDLE_2, c0);
        s0 = fma(si4, -T::R17_ODD_TWIDDLE_7, s0);
        c0 = fma(ci5, T::R17_EVEN_TWIDDLE_4, c0);
        s0 = fma(si5, T::R17_ODD_TWIDDLE_3, s0);
        c0 = fma(ci6, T::R17_EVEN_TWIDDLE_5, c0);
        s0 = fma(si6, -T::R17_ODD_TWIDDLE_0, s0);
        c0 = fma(ci7, T::R17_EVEN_TWIDDLE_1, c0);
        s0 = fma(si7, T::R17_ODD_TWIDDLE_4, s0);
        let dc = c0 + a0;
        data[14] = -dc;
        data[13] = s0;
        let mut c0 = qc * T::R17_EVEN_TWIDDLE_4;
        let mut s0 = s_buffer[0];
        s0 *= -T::R17_ODD_TWIDDLE_7;
        c0 = fma(ci1, T::R17_EVEN_TWIDDLE_3, c0);
        s0 = fma(si1, T::R17_ODD_TWIDDLE_5, s0);
        c0 = fma(ci2, T::R17_EVEN_TWIDDLE_5, c0);
        s0 = fma(si2, -T::R17_ODD_TWIDDLE_3, s0);
        c0 = fma(ci3, T::R17_EVEN_TWIDDLE_2, c0);
        s0 = fma(si3, T::R17_ODD_TWIDDLE_1, s0);
        c0 = fma(ci4, T::R17_EVEN_TWIDDLE_6, c0);
        s0 = fma(si4, -T::R17_ODD_TWIDDLE_0, s0);
        c0 = fma(ci5, T::R17_EVEN_TWIDDLE_1, c0);
        s0 = fma(si5, T::R17_ODD_TWIDDLE_2, s0);
        c0 = fma(ci6, T::R17_EVEN_TWIDDLE_7, c0);
        s0 = fma(si6, -T::R17_ODD_TWIDDLE_4, s0);
        c0 = fma(ci7, T::R17_EVEN_TWIDDLE_0, c0);
        s0 = fma(si7, T::R17_ODD_TWIDDLE_6, s0);
        let dc = c0 + a0;
        data[16] = dc;
        data[15] = -s0;
    }
}

define_avx_butterfly!(AvxDct2Butterfly17, 17);

#[derive(Debug, Clone)]
pub(crate) struct AvxDct2Butterfly19<T: DctSample> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T: DctSample> Default for AvxDct2Butterfly19<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: DctSample> AvxDct2Butterfly19<T>
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
        s0 = fma(si1, T::R19_ODD_TWIDDLE_1, s0);
        let ci2 = c_buffer[2];
        let si2 = s_buffer[2];
        c0 = ci2 + c0;
        c1 = ci2 * T::R19_EVEN_TWIDDLE_2 + c1;
        s0 = fma(si2, T::R19_ODD_TWIDDLE_2, s0);
        let ci3 = c_buffer[3];
        let si3 = s_buffer[3];
        c0 = ci3 + c0;
        c1 = ci3 * T::R19_EVEN_TWIDDLE_3 + c1;
        s0 = fma(si3, T::R19_ODD_TWIDDLE_3, s0);
        let ci4 = c_buffer[4];
        let si4 = s_buffer[4];
        c0 = ci4 + c0;
        c1 = ci4 * T::R19_EVEN_TWIDDLE_4 + c1;
        s0 = fma(si4, T::R19_ODD_TWIDDLE_4, s0);
        let ci5 = c_buffer[5];
        let si5 = s_buffer[5];
        c0 = ci5 + c0;
        c1 = ci5 * T::R19_EVEN_TWIDDLE_5 + c1;
        s0 = fma(si5, T::R19_ODD_TWIDDLE_5, s0);
        let ci6 = c_buffer[6];
        let si6 = s_buffer[6];
        c0 = ci6 + c0;
        c1 = ci6 * T::R19_EVEN_TWIDDLE_6 + c1;
        s0 = fma(si6, T::R19_ODD_TWIDDLE_6, s0);
        let ci7 = c_buffer[7];
        let si7 = s_buffer[7];
        c0 = ci7 + c0;
        c1 = ci7 * T::R19_EVEN_TWIDDLE_7 + c1;
        s0 = fma(si7, T::R19_ODD_TWIDDLE_7, s0);
        let ci8 = c_buffer[8];
        let si8 = s_buffer[8];
        c0 = ci8 + c0;
        c1 = ci8 * T::R19_EVEN_TWIDDLE_8 + c1;
        s0 = fma(si8, T::R19_ODD_TWIDDLE_8, s0);
        let a0 = a_buffer[0];
        let dc = c0 + a0;
        data[0] = dc;
        data[1] = s0;
        data[2] = -(c1 + a0);
        let mut c0 = qc * T::R19_EVEN_TWIDDLE_8;
        let mut s0 = s_buffer[0];
        s0 *= -T::R19_ODD_TWIDDLE_1;
        c0 = fma(ci1, T::R19_EVEN_TWIDDLE_6, c0);
        s0 = fma(si1, -T::R19_ODD_TWIDDLE_4, s0);
        c0 = fma(ci2, T::R19_EVEN_TWIDDLE_4, c0);
        s0 = fma(si2, -T::R19_ODD_TWIDDLE_7, s0);
        c0 = fma(ci3, T::R19_EVEN_TWIDDLE_2, c0);
        s0 = fma(si3, T::R19_ODD_TWIDDLE_8, s0);
        c0 = fma(ci4, T::R19_EVEN_TWIDDLE_0, c0);
        s0 = fma(si4, T::R19_ODD_TWIDDLE_5, s0);
        c0 = fma(ci5, T::R19_EVEN_TWIDDLE_1, c0);
        s0 = fma(si5, T::R19_ODD_TWIDDLE_2, s0);
        c0 = fma(ci6, T::R19_EVEN_TWIDDLE_3, c0);
        s0 = fma(si6, T::R19_ODD_TWIDDLE_0, s0);
        c0 = fma(ci7, T::R19_EVEN_TWIDDLE_5, c0);
        s0 = fma(si7, T::R19_ODD_TWIDDLE_3, s0);
        c0 = fma(ci8, T::R19_EVEN_TWIDDLE_7, c0);
        s0 = fma(si8, T::R19_ODD_TWIDDLE_6, s0);
        let dc = c0 + a0;
        data[4] = dc;
        data[3] = -s0;
        let mut c0 = qc * T::R19_EVEN_TWIDDLE_1;
        let mut s0 = s_buffer[0];
        s0 *= T::R19_ODD_TWIDDLE_2;
        c0 = fma(ci1, T::R19_EVEN_TWIDDLE_4, c0);
        s0 = fma(si1, T::R19_ODD_TWIDDLE_7, s0);
        c0 = fma(ci2, T::R19_EVEN_TWIDDLE_7, c0);
        s0 = fma(si2, -T::R19_ODD_TWIDDLE_6, s0);
        c0 = fma(ci3, T::R19_EVEN_TWIDDLE_8, c0);
        s0 = fma(si3, -T::R19_ODD_TWIDDLE_1, s0);
        c0 = fma(ci4, T::R19_EVEN_TWIDDLE_5, c0);
        s0 = fma(si4, -T::R19_ODD_TWIDDLE_3, s0);
        c0 = fma(ci5, T::R19_EVEN_TWIDDLE_2, c0);
        s0 = fma(si5, -T::R19_ODD_TWIDDLE_8, s0);
        c0 = fma(ci6, T::R19_EVEN_TWIDDLE_0, c0);
        s0 = fma(si6, T::R19_ODD_TWIDDLE_5, s0);
        c0 = fma(ci7, T::R19_EVEN_TWIDDLE_3, c0);
        s0 = fma(si7, T::R19_ODD_TWIDDLE_0, s0);
        c0 = fma(ci8, T::R19_EVEN_TWIDDLE_6, c0);
        s0 = fma(si8, T::R19_ODD_TWIDDLE_4, s0);
        let dc = c0 + a0;
        data[6] = -dc;
        data[5] = s0;
        let mut c0 = qc * T::R19_EVEN_TWIDDLE_7;
        let mut s0 = s_buffer[0];
        s0 *= -T::R19_ODD_TWIDDLE_3;
        c0 = fma(ci1, T::R19_EVEN_TWIDDLE_3, c0);
        s0 = fma(si1, T::R19_ODD_TWIDDLE_8, s0);
        c0 = fma(ci2, T::R19_EVEN_TWIDDLE_0, c0);
        s0 = fma(si2, T::R19_ODD_TWIDDLE_1, s0);
        c0 = fma(ci3, T::R19_EVEN_TWIDDLE_4, c0);
        s0 = fma(si3, T::R19_ODD_TWIDDLE_5, s0);
        c0 = fma(ci4, T::R19_EVEN_TWIDDLE_8, c0);
        s0 = fma(si4, -T::R19_ODD_TWIDDLE_6, s0);
        c0 = fma(ci5, T::R19_EVEN_TWIDDLE_6, c0);
        s0 = fma(si5, -T::R19_ODD_TWIDDLE_0, s0);
        c0 = fma(ci6, T::R19_EVEN_TWIDDLE_2, c0);
        s0 = fma(si6, -T::R19_ODD_TWIDDLE_7, s0);
        c0 = fma(ci7, T::R19_EVEN_TWIDDLE_1, c0);
        s0 = fma(si7, T::R19_ODD_TWIDDLE_4, s0);
        c0 = fma(ci8, T::R19_EVEN_TWIDDLE_5, c0);
        s0 = fma(si8, T::R19_ODD_TWIDDLE_2, s0);
        let dc = c0 + a0;
        data[8] = dc;
        data[7] = -s0;
        let mut c0 = qc * T::R19_EVEN_TWIDDLE_2;
        let mut s0 = s_buffer[0];
        s0 *= T::R19_ODD_TWIDDLE_4;
        c0 = fma(ci1, T::R19_EVEN_TWIDDLE_7, c0);
        s0 = fma(si1, -T::R19_ODD_TWIDDLE_5, s0);
        c0 = fma(ci2, T::R19_EVEN_TWIDDLE_6, c0);
        s0 = fma(si2, -T::R19_ODD_TWIDDLE_3, s0);
        c0 = fma(ci3, T::R19_EVEN_TWIDDLE_1, c0);
        s0 = fma(si3, T::R19_ODD_TWIDDLE_6, s0);
        c0 = fma(ci4, T::R19_EVEN_TWIDDLE_3, c0);
        s0 = fma(si4, T::R19_ODD_TWIDDLE_2, s0);
        c0 = fma(ci5, T::R19_EVEN_TWIDDLE_8, c0);
        s0 = fma(si5, -T::R19_ODD_TWIDDLE_7, s0);
        c0 = fma(ci6, T::R19_EVEN_TWIDDLE_5, c0);
        s0 = fma(si6, -T::R19_ODD_TWIDDLE_1, s0);
        c0 = fma(ci7, T::R19_EVEN_TWIDDLE_0, c0);
        s0 = fma(si7, T::R19_ODD_TWIDDLE_8, s0);
        c0 = fma(ci8, T::R19_EVEN_TWIDDLE_4, c0);
        s0 = fma(si8, T::R19_ODD_TWIDDLE_0, s0);
        let dc = c0 + a0;
        data[10] = -dc;
        data[9] = s0;
        let mut c0 = qc * T::R19_EVEN_TWIDDLE_6;
        let mut s0 = s_buffer[0];
        s0 *= -T::R19_ODD_TWIDDLE_5;
        c0 = fma(ci1, T::R19_EVEN_TWIDDLE_0, c0);
        s0 = fma(si1, T::R19_ODD_TWIDDLE_2, s0);
        c0 = fma(ci2, T::R19_EVEN_TWIDDLE_5, c0);
        s0 = fma(si2, T::R19_ODD_TWIDDLE_8, s0);
        c0 = fma(ci3, T::R19_EVEN_TWIDDLE_7, c0);
        s0 = fma(si3, -T::R19_ODD_TWIDDLE_0, s0);
        c0 = fma(ci4, T::R19_EVEN_TWIDDLE_1, c0);
        s0 = fma(si4, T::R19_ODD_TWIDDLE_7, s0);
        c0 = fma(ci5, T::R19_EVEN_TWIDDLE_4, c0);
        s0 = fma(si5, T::R19_ODD_TWIDDLE_3, s0);
        c0 = fma(ci6, T::R19_EVEN_TWIDDLE_8, c0);
        s0 = fma(si6, -T::R19_ODD_TWIDDLE_4, s0);
        c0 = fma(ci7, T::R19_EVEN_TWIDDLE_2, c0);
        s0 = fma(si7, -T::R19_ODD_TWIDDLE_6, s0);
        c0 = fma(ci8, T::R19_EVEN_TWIDDLE_3, c0);
        s0 = fma(si8, T::R19_ODD_TWIDDLE_1, s0);
        let dc = c0 + a0;
        data[12] = dc;
        data[11] = -s0;
        let mut c0 = qc * T::R19_EVEN_TWIDDLE_3;
        let mut s0 = s_buffer[0];
        s0 *= T::R19_ODD_TWIDDLE_6;
        c0 = fma(ci1, T::R19_EVEN_TWIDDLE_8, c0);
        s0 = fma(si1, -T::R19_ODD_TWIDDLE_0, s0);
        c0 = fma(ci2, T::R19_EVEN_TWIDDLE_1, c0);
        s0 = fma(si2, T::R19_ODD_TWIDDLE_5, s0);
        c0 = fma(ci3, T::R19_EVEN_TWIDDLE_5, c0);
        s0 = fma(si3, T::R19_ODD_TWIDDLE_7, s0);
        c0 = fma(ci4, T::R19_EVEN_TWIDDLE_6, c0);
        s0 = fma(si4, -T::R19_ODD_TWIDDLE_1, s0);
        c0 = fma(ci5, T::R19_EVEN_TWIDDLE_0, c0);
        s0 = fma(si5, T::R19_ODD_TWIDDLE_4, s0);
        c0 = fma(ci6, T::R19_EVEN_TWIDDLE_7, c0);
        s0 = fma(si6, T::R19_ODD_TWIDDLE_8, s0);
        c0 = fma(ci7, T::R19_EVEN_TWIDDLE_4, c0);
        s0 = fma(si7, -T::R19_ODD_TWIDDLE_2, s0);
        c0 = fma(ci8, T::R19_EVEN_TWIDDLE_2, c0);
        s0 = fma(si8, T::R19_ODD_TWIDDLE_3, s0);
        let dc = c0 + a0;
        data[14] = -dc;
        data[13] = s0;
        let mut c0 = qc * T::R19_EVEN_TWIDDLE_5;
        let mut s0 = s_buffer[0];
        s0 *= -T::R19_ODD_TWIDDLE_7;
        c0 = fma(ci1, T::R19_EVEN_TWIDDLE_2, c0);
        s0 = fma(si1, T::R19_ODD_TWIDDLE_3, s0);
        c0 = fma(ci2, T::R19_EVEN_TWIDDLE_8, c0);
        s0 = fma(si2, -T::R19_ODD_TWIDDLE_0, s0);
        c0 = fma(ci3, T::R19_EVEN_TWIDDLE_0, c0);
        s0 = fma(si3, T::R19_ODD_TWIDDLE_4, s0);
        c0 = fma(ci4, T::R19_EVEN_TWIDDLE_7, c0);
        s0 = fma(si4, -T::R19_ODD_TWIDDLE_8, s0);
        c0 = fma(ci5, T::R19_EVEN_TWIDDLE_3, c0);
        s0 = fma(si5, -T::R19_ODD_TWIDDLE_6, s0);
        c0 = fma(ci6, T::R19_EVEN_TWIDDLE_4, c0);
        s0 = fma(si6, T::R19_ODD_TWIDDLE_2, s0);
        c0 = fma(ci7, T::R19_EVEN_TWIDDLE_6, c0);
        s0 = fma(si7, -T::R19_ODD_TWIDDLE_1, s0);
        c0 = fma(ci8, T::R19_EVEN_TWIDDLE_1, c0);
        s0 = fma(si8, T::R19_ODD_TWIDDLE_5, s0);
        let dc = c0 + a0;
        data[16] = dc;
        data[15] = -s0;
        let mut c0 = qc * T::R19_EVEN_TWIDDLE_4;
        let mut s0 = s_buffer[0];
        s0 *= T::R19_ODD_TWIDDLE_8;
        c0 = fma(ci1, T::R19_EVEN_TWIDDLE_5, c0);
        s0 = fma(si1, -T::R19_ODD_TWIDDLE_6, s0);
        c0 = fma(ci2, T::R19_EVEN_TWIDDLE_3, c0);
        s0 = fma(si2, T::R19_ODD_TWIDDLE_4, s0);
        c0 = fma(ci3, T::R19_EVEN_TWIDDLE_6, c0);
        s0 = fma(si3, -T::R19_ODD_TWIDDLE_2, s0);
        c0 = fma(ci4, T::R19_EVEN_TWIDDLE_2, c0);
        s0 = fma(si4, T::R19_ODD_TWIDDLE_0, s0);
        c0 = fma(ci5, T::R19_EVEN_TWIDDLE_7, c0);
        s0 = fma(si5, -T::R19_ODD_TWIDDLE_1, s0);
        c0 = fma(ci6, T::R19_EVEN_TWIDDLE_1, c0);
        s0 = fma(si6, T::R19_ODD_TWIDDLE_3, s0);
        c0 = fma(ci7, T::R19_EVEN_TWIDDLE_8, c0);
        s0 = fma(si7, -T::R19_ODD_TWIDDLE_5, s0);
        c0 = fma(ci8, T::R19_EVEN_TWIDDLE_0, c0);
        s0 = fma(si8, T::R19_ODD_TWIDDLE_7, s0);
        let dc = c0 + a0;
        data[18] = -dc;
        data[17] = s0;
    }
}

define_avx_butterfly!(AvxDct2Butterfly19, 19);

#[derive(Debug, Clone)]
pub(crate) struct AvxDct2Butterfly23<T: DctSample> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T: DctSample> Default for AvxDct2Butterfly23<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: DctSample> AvxDct2Butterfly23<T>
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
        s0 = fma(si1, T::R23_ODD_TWIDDLE_1, s0);
        let ci2 = c_buffer[2];
        let si2 = s_buffer[2];
        c0 = ci2 + c0;
        c1 = ci2 * T::R23_EVEN_TWIDDLE_2 + c1;
        s0 = fma(si2, T::R23_ODD_TWIDDLE_2, s0);
        let ci3 = c_buffer[3];
        let si3 = s_buffer[3];
        c0 = ci3 + c0;
        c1 = ci3 * T::R23_EVEN_TWIDDLE_3 + c1;
        s0 = fma(si3, T::R23_ODD_TWIDDLE_3, s0);
        let ci4 = c_buffer[4];
        let si4 = s_buffer[4];
        c0 = ci4 + c0;
        c1 = ci4 * T::R23_EVEN_TWIDDLE_4 + c1;
        s0 = fma(si4, T::R23_ODD_TWIDDLE_4, s0);
        let ci5 = c_buffer[5];
        let si5 = s_buffer[5];
        c0 = ci5 + c0;
        c1 = ci5 * T::R23_EVEN_TWIDDLE_5 + c1;
        s0 = fma(si5, T::R23_ODD_TWIDDLE_5, s0);
        let ci6 = c_buffer[6];
        let si6 = s_buffer[6];
        c0 = ci6 + c0;
        c1 = ci6 * T::R23_EVEN_TWIDDLE_6 + c1;
        s0 = fma(si6, T::R23_ODD_TWIDDLE_6, s0);
        let ci7 = c_buffer[7];
        let si7 = s_buffer[7];
        c0 = ci7 + c0;
        c1 = ci7 * T::R23_EVEN_TWIDDLE_7 + c1;
        s0 = fma(si7, T::R23_ODD_TWIDDLE_7, s0);
        let ci8 = c_buffer[8];
        let si8 = s_buffer[8];
        c0 = ci8 + c0;
        c1 = ci8 * T::R23_EVEN_TWIDDLE_8 + c1;
        s0 = fma(si8, T::R23_ODD_TWIDDLE_8, s0);
        let ci9 = c_buffer[9];
        let si9 = s_buffer[9];
        c0 = ci9 + c0;
        c1 = ci9 * T::R23_EVEN_TWIDDLE_9 + c1;
        s0 = fma(si9, T::R23_ODD_TWIDDLE_9, s0);
        let ci10 = c_buffer[10];
        let si10 = s_buffer[10];
        c0 = ci10 + c0;
        c1 = ci10 * T::R23_EVEN_TWIDDLE_10 + c1;
        s0 = fma(si10, T::R23_ODD_TWIDDLE_10, s0);
        let a0 = a_buffer[0];
        let dc = c0 + a0;
        data[0] = dc;
        data[1] = s0;
        data[2] = -(c1 + a0);
        let mut c0 = qc * T::R23_EVEN_TWIDDLE_10;
        let mut s0 = s_buffer[0];
        s0 *= -T::R23_ODD_TWIDDLE_1;
        c0 = fma(ci1, T::R23_EVEN_TWIDDLE_8, c0);
        s0 = fma(si1, -T::R23_ODD_TWIDDLE_4, s0);
        c0 = fma(ci2, T::R23_EVEN_TWIDDLE_6, c0);
        s0 = fma(si2, -T::R23_ODD_TWIDDLE_7, s0);
        c0 = fma(ci3, T::R23_EVEN_TWIDDLE_4, c0);
        s0 = fma(si3, -T::R23_ODD_TWIDDLE_10, s0);
        c0 = fma(ci4, T::R23_EVEN_TWIDDLE_2, c0);
        s0 = fma(si4, T::R23_ODD_TWIDDLE_9, s0);
        c0 = fma(ci5, T::R23_EVEN_TWIDDLE_0, c0);
        s0 = fma(si5, T::R23_ODD_TWIDDLE_6, s0);
        c0 = fma(ci6, T::R23_EVEN_TWIDDLE_1, c0);
        s0 = fma(si6, T::R23_ODD_TWIDDLE_3, s0);
        c0 = fma(ci7, T::R23_EVEN_TWIDDLE_3, c0);
        s0 = fma(si7, T::R23_ODD_TWIDDLE_0, s0);
        c0 = fma(ci8, T::R23_EVEN_TWIDDLE_5, c0);
        s0 = fma(si8, T::R23_ODD_TWIDDLE_2, s0);
        c0 = fma(ci9, T::R23_EVEN_TWIDDLE_7, c0);
        s0 = fma(si9, T::R23_ODD_TWIDDLE_5, s0);
        c0 = fma(ci10, T::R23_EVEN_TWIDDLE_9, c0);
        s0 = fma(si10, T::R23_ODD_TWIDDLE_8, s0);
        let dc = c0 + a0;
        data[4] = dc;
        data[3] = -s0;
        let mut c0 = qc * T::R23_EVEN_TWIDDLE_1;
        let mut s0 = s_buffer[0];
        s0 *= T::R23_ODD_TWIDDLE_2;
        c0 = fma(ci1, T::R23_EVEN_TWIDDLE_4, c0);
        s0 = fma(si1, T::R23_ODD_TWIDDLE_7, s0);
        c0 = fma(ci2, T::R23_EVEN_TWIDDLE_7, c0);
        s0 = fma(si2, -T::R23_ODD_TWIDDLE_10, s0);
        c0 = fma(ci3, T::R23_EVEN_TWIDDLE_10, c0);
        s0 = fma(si3, -T::R23_ODD_TWIDDLE_5, s0);
        c0 = fma(ci4, T::R23_EVEN_TWIDDLE_9, c0);
        s0 = fma(si4, -T::R23_ODD_TWIDDLE_0, s0);
        c0 = fma(ci5, T::R23_EVEN_TWIDDLE_6, c0);
        s0 = fma(si5, -T::R23_ODD_TWIDDLE_4, s0);
        c0 = fma(ci6, T::R23_EVEN_TWIDDLE_3, c0);
        s0 = fma(si6, -T::R23_ODD_TWIDDLE_9, s0);
        c0 = fma(ci7, T::R23_EVEN_TWIDDLE_0, c0);
        s0 = fma(si7, T::R23_ODD_TWIDDLE_8, s0);
        c0 = fma(ci8, T::R23_EVEN_TWIDDLE_2, c0);
        s0 = fma(si8, T::R23_ODD_TWIDDLE_3, s0);
        c0 = fma(ci9, T::R23_EVEN_TWIDDLE_5, c0);
        s0 = fma(si9, T::R23_ODD_TWIDDLE_1, s0);
        c0 = fma(ci10, T::R23_EVEN_TWIDDLE_8, c0);
        s0 = fma(si10, T::R23_ODD_TWIDDLE_6, s0);
        let dc = c0 + a0;
        data[6] = -dc;
        data[5] = s0;
        let mut c0 = qc * T::R23_EVEN_TWIDDLE_9;
        let mut s0 = s_buffer[0];
        s0 *= -T::R23_ODD_TWIDDLE_3;
        c0 = fma(ci1, T::R23_EVEN_TWIDDLE_5, c0);
        s0 = fma(si1, -T::R23_ODD_TWIDDLE_10, s0);
        c0 = fma(ci2, T::R23_EVEN_TWIDDLE_1, c0);
        s0 = fma(si2, T::R23_ODD_TWIDDLE_5, s0);
        c0 = fma(ci3, T::R23_EVEN_TWIDDLE_2, c0);
        s0 = fma(si3, T::R23_ODD_TWIDDLE_1, s0);
        c0 = fma(ci4, T::R23_EVEN_TWIDDLE_6, c0);
        s0 = fma(si4, T::R23_ODD_TWIDDLE_8, s0);
        c0 = fma(ci5, T::R23_EVEN_TWIDDLE_10, c0);
        s0 = fma(si5, -T::R23_ODD_TWIDDLE_7, s0);
        c0 = fma(ci6, T::R23_EVEN_TWIDDLE_8, c0);
        s0 = fma(si6, -T::R23_ODD_TWIDDLE_0, s0);
        c0 = fma(ci7, T::R23_EVEN_TWIDDLE_4, c0);
        s0 = fma(si7, -T::R23_ODD_TWIDDLE_6, s0);
        c0 = fma(ci8, T::R23_EVEN_TWIDDLE_0, c0);
        s0 = fma(si8, T::R23_ODD_TWIDDLE_9, s0);
        c0 = fma(ci9, T::R23_EVEN_TWIDDLE_3, c0);
        s0 = fma(si9, T::R23_ODD_TWIDDLE_2, s0);
        c0 = fma(ci10, T::R23_EVEN_TWIDDLE_7, c0);
        s0 = fma(si10, T::R23_ODD_TWIDDLE_4, s0);
        let dc = c0 + a0;
        data[8] = dc;
        data[7] = -s0;
        let mut c0 = qc * T::R23_EVEN_TWIDDLE_2;
        let mut s0 = s_buffer[0];
        s0 *= T::R23_ODD_TWIDDLE_4;
        c0 = fma(ci1, T::R23_EVEN_TWIDDLE_7, c0);
        s0 = fma(si1, -T::R23_ODD_TWIDDLE_9, s0);
        c0 = fma(ci2, T::R23_EVEN_TWIDDLE_10, c0);
        s0 = fma(si2, -T::R23_ODD_TWIDDLE_0, s0);
        c0 = fma(ci3, T::R23_EVEN_TWIDDLE_5, c0);
        s0 = fma(si3, -T::R23_ODD_TWIDDLE_8, s0);
        c0 = fma(ci4, T::R23_EVEN_TWIDDLE_0, c0);
        s0 = fma(si4, T::R23_ODD_TWIDDLE_5, s0);
        c0 = fma(ci5, T::R23_EVEN_TWIDDLE_4, c0);
        s0 = fma(si5, T::R23_ODD_TWIDDLE_3, s0);
        c0 = fma(ci6, T::R23_EVEN_TWIDDLE_9, c0);
        s0 = fma(si6, -T::R23_ODD_TWIDDLE_10, s0);
        c0 = fma(ci7, T::R23_EVEN_TWIDDLE_8, c0);
        s0 = fma(si7, -T::R23_ODD_TWIDDLE_1, s0);
        c0 = fma(ci8, T::R23_EVEN_TWIDDLE_3, c0);
        s0 = fma(si8, -T::R23_ODD_TWIDDLE_7, s0);
        c0 = fma(ci9, T::R23_EVEN_TWIDDLE_1, c0);
        s0 = fma(si9, T::R23_ODD_TWIDDLE_6, s0);
        c0 = fma(ci10, T::R23_EVEN_TWIDDLE_6, c0);
        s0 = fma(si10, T::R23_ODD_TWIDDLE_2, s0);
        let dc = c0 + a0;
        data[10] = -dc;
        data[9] = s0;
        let mut c0 = qc * T::R23_EVEN_TWIDDLE_8;
        let mut s0 = s_buffer[0];
        s0 *= -T::R23_ODD_TWIDDLE_5;
        c0 = fma(ci1, T::R23_EVEN_TWIDDLE_2, c0);
        s0 = fma(si1, T::R23_ODD_TWIDDLE_6, s0);
        c0 = fma(ci2, T::R23_EVEN_TWIDDLE_3, c0);
        s0 = fma(si2, T::R23_ODD_TWIDDLE_4, s0);
        c0 = fma(ci3, T::R23_EVEN_TWIDDLE_9, c0);
        s0 = fma(si3, -T::R23_ODD_TWIDDLE_7, s0);
        c0 = fma(ci4, T::R23_EVEN_TWIDDLE_7, c0);
        s0 = fma(si4, -T::R23_ODD_TWIDDLE_3, s0);
        c0 = fma(ci5, T::R23_EVEN_TWIDDLE_1, c0);
        s0 = fma(si5, T::R23_ODD_TWIDDLE_8, s0);
        c0 = fma(ci6, T::R23_EVEN_TWIDDLE_4, c0);
        s0 = fma(si6, T::R23_ODD_TWIDDLE_2, s0);
        c0 = fma(ci7, T::R23_EVEN_TWIDDLE_10, c0);
        s0 = fma(si7, -T::R23_ODD_TWIDDLE_9, s0);
        c0 = fma(ci8, T::R23_EVEN_TWIDDLE_6, c0);
        s0 = fma(si8, -T::R23_ODD_TWIDDLE_1, s0);
        c0 = fma(ci9, T::R23_EVEN_TWIDDLE_0, c0);
        s0 = fma(si9, T::R23_ODD_TWIDDLE_10, s0);
        c0 = fma(ci10, T::R23_EVEN_TWIDDLE_5, c0);
        s0 = fma(si10, T::R23_ODD_TWIDDLE_0, s0);
        let dc = c0 + a0;
        data[12] = dc;
        data[11] = -s0;
        let mut c0 = qc * T::R23_EVEN_TWIDDLE_3;
        let mut s0 = s_buffer[0];
        s0 *= T::R23_ODD_TWIDDLE_6;
        c0 = fma(ci1, T::R23_EVEN_TWIDDLE_10, c0);
        s0 = fma(si1, -T::R23_ODD_TWIDDLE_3, s0);
        c0 = fma(ci2, T::R23_EVEN_TWIDDLE_5, c0);
        s0 = fma(si2, -T::R23_ODD_TWIDDLE_9, s0);
        c0 = fma(ci3, T::R23_EVEN_TWIDDLE_1, c0);
        s0 = fma(si3, T::R23_ODD_TWIDDLE_0, s0);
        c0 = fma(ci4, T::R23_EVEN_TWIDDLE_8, c0);
        s0 = fma(si4, -T::R23_ODD_TWIDDLE_10, s0);
        c0 = fma(ci5, T::R23_EVEN_TWIDDLE_7, c0);
        s0 = fma(si5, -T::R23_ODD_TWIDDLE_2, s0);
        c0 = fma(ci6, T::R23_EVEN_TWIDDLE_0, c0);
        s0 = fma(si6, T::R23_ODD_TWIDDLE_7, s0);
        c0 = fma(ci7, T::R23_EVEN_TWIDDLE_6, c0);
        s0 = fma(si7, T::R23_ODD_TWIDDLE_5, s0);
        c0 = fma(ci8, T::R23_EVEN_TWIDDLE_9, c0);
        s0 = fma(si8, -T::R23_ODD_TWIDDLE_4, s0);
        c0 = fma(ci9, T::R23_EVEN_TWIDDLE_2, c0);
        s0 = fma(si9, -T::R23_ODD_TWIDDLE_8, s0);
        c0 = fma(ci10, T::R23_EVEN_TWIDDLE_4, c0);
        s0 = fma(si10, T::R23_ODD_TWIDDLE_1, s0);
        let dc = c0 + a0;
        data[14] = -dc;
        data[13] = s0;
        let mut c0 = qc * T::R23_EVEN_TWIDDLE_7;
        let mut s0 = s_buffer[0];
        s0 *= -T::R23_ODD_TWIDDLE_7;
        c0 = fma(ci1, T::R23_EVEN_TWIDDLE_0, c0);
        s0 = fma(si1, T::R23_ODD_TWIDDLE_0, s0);
        c0 = fma(ci2, T::R23_EVEN_TWIDDLE_8, c0);
        s0 = fma(si2, -T::R23_ODD_TWIDDLE_8, s0);
        c0 = fma(ci3, T::R23_EVEN_TWIDDLE_6, c0);
        s0 = fma(si3, -T::R23_ODD_TWIDDLE_6, s0);
        c0 = fma(ci4, T::R23_EVEN_TWIDDLE_1, c0);
        s0 = fma(si4, T::R23_ODD_TWIDDLE_1, s0);
        c0 = fma(ci5, T::R23_EVEN_TWIDDLE_9, c0);
        s0 = fma(si5, -T::R23_ODD_TWIDDLE_9, s0);
        c0 = fma(ci6, T::R23_EVEN_TWIDDLE_5, c0);
        s0 = fma(si6, -T::R23_ODD_TWIDDLE_5, s0);
        c0 = fma(ci7, T::R23_EVEN_TWIDDLE_2, c0);
        s0 = fma(si7, T::R23_ODD_TWIDDLE_2, s0);
        c0 = fma(ci8, T::R23_EVEN_TWIDDLE_10, c0);
        s0 = fma(si8, -T::R23_ODD_TWIDDLE_10, s0);
        c0 = fma(ci9, T::R23_EVEN_TWIDDLE_4, c0);
        s0 = fma(si9, -T::R23_ODD_TWIDDLE_4, s0);
        c0 = fma(ci10, T::R23_EVEN_TWIDDLE_3, c0);
        s0 = fma(si10, T::R23_ODD_TWIDDLE_3, s0);
        let dc = c0 + a0;
        data[16] = dc;
        data[15] = -s0;
        let mut c0 = qc * T::R23_EVEN_TWIDDLE_4;
        let mut s0 = s_buffer[0];
        s0 *= T::R23_ODD_TWIDDLE_8;
        c0 = fma(ci1, T::R23_EVEN_TWIDDLE_9, c0);
        s0 = fma(si1, -T::R23_ODD_TWIDDLE_2, s0);
        c0 = fma(ci2, T::R23_EVEN_TWIDDLE_0, c0);
        s0 = fma(si2, T::R23_ODD_TWIDDLE_3, s0);
        c0 = fma(ci3, T::R23_EVEN_TWIDDLE_8, c0);
        s0 = fma(si3, -T::R23_ODD_TWIDDLE_9, s0);
        c0 = fma(ci4, T::R23_EVEN_TWIDDLE_5, c0);
        s0 = fma(si4, -T::R23_ODD_TWIDDLE_7, s0);
        c0 = fma(ci5, T::R23_EVEN_TWIDDLE_3, c0);
        s0 = fma(si5, T::R23_ODD_TWIDDLE_1, s0);
        c0 = fma(ci6, T::R23_EVEN_TWIDDLE_10, c0);
        s0 = fma(si6, -T::R23_ODD_TWIDDLE_4, s0);
        c0 = fma(ci7, T::R23_EVEN_TWIDDLE_1, c0);
        s0 = fma(si7, T::R23_ODD_TWIDDLE_10, s0);
        c0 = fma(ci8, T::R23_EVEN_TWIDDLE_7, c0);
        s0 = fma(si8, T::R23_ODD_TWIDDLE_6, s0);
        c0 = fma(ci9, T::R23_EVEN_TWIDDLE_6, c0);
        s0 = fma(si9, -T::R23_ODD_TWIDDLE_0, s0);
        c0 = fma(ci10, T::R23_EVEN_TWIDDLE_2, c0);
        s0 = fma(si10, T::R23_ODD_TWIDDLE_5, s0);
        let dc = c0 + a0;
        data[18] = -dc;
        data[17] = s0;
        let mut c0 = qc * T::R23_EVEN_TWIDDLE_6;
        let mut s0 = s_buffer[0];
        s0 *= -T::R23_ODD_TWIDDLE_9;
        c0 = fma(ci1, T::R23_EVEN_TWIDDLE_3, c0);
        s0 = fma(si1, T::R23_ODD_TWIDDLE_5, s0);
        c0 = fma(ci2, T::R23_EVEN_TWIDDLE_9, c0);
        s0 = fma(si2, -T::R23_ODD_TWIDDLE_1, s0);
        c0 = fma(ci3, T::R23_EVEN_TWIDDLE_0, c0);
        s0 = fma(si3, T::R23_ODD_TWIDDLE_2, s0);
        c0 = fma(ci4, T::R23_EVEN_TWIDDLE_10, c0);
        s0 = fma(si4, -T::R23_ODD_TWIDDLE_6, s0);
        c0 = fma(ci5, T::R23_EVEN_TWIDDLE_2, c0);
        s0 = fma(si5, T::R23_ODD_TWIDDLE_10, s0);
        c0 = fma(ci6, T::R23_EVEN_TWIDDLE_7, c0);
        s0 = fma(si6, T::R23_ODD_TWIDDLE_8, s0);
        c0 = fma(ci7, T::R23_EVEN_TWIDDLE_5, c0);
        s0 = fma(si7, -T::R23_ODD_TWIDDLE_4, s0);
        c0 = fma(ci8, T::R23_EVEN_TWIDDLE_4, c0);
        s0 = fma(si8, T::R23_ODD_TWIDDLE_0, s0);
        c0 = fma(ci9, T::R23_EVEN_TWIDDLE_8, c0);
        s0 = fma(si9, -T::R23_ODD_TWIDDLE_3, s0);
        c0 = fma(ci10, T::R23_EVEN_TWIDDLE_1, c0);
        s0 = fma(si10, T::R23_ODD_TWIDDLE_7, s0);
        let dc = c0 + a0;
        data[20] = dc;
        data[19] = -s0;
        let mut c0 = qc * T::R23_EVEN_TWIDDLE_5;
        let mut s0 = s_buffer[0];
        s0 *= T::R23_ODD_TWIDDLE_10;
        c0 = fma(ci1, T::R23_EVEN_TWIDDLE_6, c0);
        s0 = fma(si1, -T::R23_ODD_TWIDDLE_8, s0);
        c0 = fma(ci2, T::R23_EVEN_TWIDDLE_4, c0);
        s0 = fma(si2, T::R23_ODD_TWIDDLE_6, s0);
        c0 = fma(ci3, T::R23_EVEN_TWIDDLE_7, c0);
        s0 = fma(si3, -T::R23_ODD_TWIDDLE_4, s0);
        c0 = fma(ci4, T::R23_EVEN_TWIDDLE_3, c0);
        s0 = fma(si4, T::R23_ODD_TWIDDLE_2, s0);
        c0 = fma(ci5, T::R23_EVEN_TWIDDLE_8, c0);
        s0 = fma(si5, -T::R23_ODD_TWIDDLE_0, s0);
        c0 = fma(ci6, T::R23_EVEN_TWIDDLE_2, c0);
        s0 = fma(si6, T::R23_ODD_TWIDDLE_1, s0);
        c0 = fma(ci7, T::R23_EVEN_TWIDDLE_9, c0);
        s0 = fma(si7, -T::R23_ODD_TWIDDLE_3, s0);
        c0 = fma(ci8, T::R23_EVEN_TWIDDLE_1, c0);
        s0 = fma(si8, T::R23_ODD_TWIDDLE_5, s0);
        c0 = fma(ci9, T::R23_EVEN_TWIDDLE_10, c0);
        s0 = fma(si9, -T::R23_ODD_TWIDDLE_7, s0);
        c0 = fma(ci10, T::R23_EVEN_TWIDDLE_0, c0);
        s0 = fma(si10, T::R23_ODD_TWIDDLE_9, s0);
        let dc = c0 + a0;
        data[22] = -dc;
        data[21] = s0;
    }
}

define_avx_butterfly!(AvxDct2Butterfly23, 23);

#[derive(Debug, Clone)]
pub(crate) struct AvxDct2Butterfly29<T: DctSample> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T: DctSample> Default for AvxDct2Butterfly29<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: DctSample> AvxDct2Butterfly29<T>
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
        s0 = fma(si1, T::R29_ODD_TWIDDLE_1, s0);
        let ci2 = c_buffer[2];
        let si2 = s_buffer[2];
        c0 = ci2 + c0;
        c1 = ci2 * T::R29_EVEN_TWIDDLE_2 + c1;
        s0 = fma(si2, T::R29_ODD_TWIDDLE_2, s0);
        let ci3 = c_buffer[3];
        let si3 = s_buffer[3];
        c0 = ci3 + c0;
        c1 = ci3 * T::R29_EVEN_TWIDDLE_3 + c1;
        s0 = fma(si3, T::R29_ODD_TWIDDLE_3, s0);
        let ci4 = c_buffer[4];
        let si4 = s_buffer[4];
        c0 = ci4 + c0;
        c1 = ci4 * T::R29_EVEN_TWIDDLE_4 + c1;
        s0 = fma(si4, T::R29_ODD_TWIDDLE_4, s0);
        let ci5 = c_buffer[5];
        let si5 = s_buffer[5];
        c0 = ci5 + c0;
        c1 = ci5 * T::R29_EVEN_TWIDDLE_5 + c1;
        s0 = fma(si5, T::R29_ODD_TWIDDLE_5, s0);
        let ci6 = c_buffer[6];
        let si6 = s_buffer[6];
        c0 = ci6 + c0;
        c1 = ci6 * T::R29_EVEN_TWIDDLE_6 + c1;
        s0 = fma(si6, T::R29_ODD_TWIDDLE_6, s0);
        let ci7 = c_buffer[7];
        let si7 = s_buffer[7];
        c0 = ci7 + c0;
        c1 = ci7 * T::R29_EVEN_TWIDDLE_7 + c1;
        s0 = fma(si7, T::R29_ODD_TWIDDLE_7, s0);
        let ci8 = c_buffer[8];
        let si8 = s_buffer[8];
        c0 = ci8 + c0;
        c1 = ci8 * T::R29_EVEN_TWIDDLE_8 + c1;
        s0 = fma(si8, T::R29_ODD_TWIDDLE_8, s0);
        let ci9 = c_buffer[9];
        let si9 = s_buffer[9];
        c0 = ci9 + c0;
        c1 = ci9 * T::R29_EVEN_TWIDDLE_9 + c1;
        s0 = fma(si9, T::R29_ODD_TWIDDLE_9, s0);
        let ci10 = c_buffer[10];
        let si10 = s_buffer[10];
        c0 = ci10 + c0;
        c1 = ci10 * T::R29_EVEN_TWIDDLE_10 + c1;
        s0 = fma(si10, T::R29_ODD_TWIDDLE_10, s0);
        let ci11 = c_buffer[11];
        let si11 = s_buffer[11];
        c0 = ci11 + c0;
        c1 = ci11 * T::R29_EVEN_TWIDDLE_11 + c1;
        s0 = fma(si11, T::R29_ODD_TWIDDLE_11, s0);
        let ci12 = c_buffer[12];
        let si12 = s_buffer[12];
        c0 = ci12 + c0;
        c1 = ci12 * T::R29_EVEN_TWIDDLE_12 + c1;
        s0 = fma(si12, T::R29_ODD_TWIDDLE_12, s0);
        let ci13 = c_buffer[13];
        let si13 = s_buffer[13];
        c0 = ci13 + c0;
        c1 = ci13 * T::R29_EVEN_TWIDDLE_13 + c1;
        s0 = fma(si13, T::R29_ODD_TWIDDLE_13, s0);
        let a0 = a_buffer[0];
        let dc = c0 + a0;
        data[0] = dc;
        data[1] = s0;
        data[2] = -(c1 + a0);
        let mut c0 = qc * T::R29_EVEN_TWIDDLE_13;
        let mut s0 = s_buffer[0];
        s0 *= -T::R29_ODD_TWIDDLE_1;
        c0 = fma(ci1, T::R29_EVEN_TWIDDLE_11, c0);
        s0 = fma(si1, -T::R29_ODD_TWIDDLE_4, s0);
        c0 = fma(ci2, T::R29_EVEN_TWIDDLE_9, c0);
        s0 = fma(si2, -T::R29_ODD_TWIDDLE_7, s0);
        c0 = fma(ci3, T::R29_EVEN_TWIDDLE_7, c0);
        s0 = fma(si3, -T::R29_ODD_TWIDDLE_10, s0);
        c0 = fma(ci4, T::R29_EVEN_TWIDDLE_5, c0);
        s0 = fma(si4, -T::R29_ODD_TWIDDLE_13, s0);
        c0 = fma(ci5, T::R29_EVEN_TWIDDLE_3, c0);
        s0 = fma(si5, T::R29_ODD_TWIDDLE_12, s0);
        c0 = fma(ci6, T::R29_EVEN_TWIDDLE_1, c0);
        s0 = fma(si6, T::R29_ODD_TWIDDLE_9, s0);
        c0 = fma(ci7, T::R29_EVEN_TWIDDLE_0, c0);
        s0 = fma(si7, T::R29_ODD_TWIDDLE_6, s0);
        c0 = fma(ci8, T::R29_EVEN_TWIDDLE_2, c0);
        s0 = fma(si8, T::R29_ODD_TWIDDLE_3, s0);
        c0 = fma(ci9, T::R29_EVEN_TWIDDLE_4, c0);
        s0 = fma(si9, T::R29_ODD_TWIDDLE_0, s0);
        c0 = fma(ci10, T::R29_EVEN_TWIDDLE_6, c0);
        s0 = fma(si10, T::R29_ODD_TWIDDLE_2, s0);
        c0 = fma(ci11, T::R29_EVEN_TWIDDLE_8, c0);
        s0 = fma(si11, T::R29_ODD_TWIDDLE_5, s0);
        c0 = fma(ci12, T::R29_EVEN_TWIDDLE_10, c0);
        s0 = fma(si12, T::R29_ODD_TWIDDLE_8, s0);
        c0 = fma(ci13, T::R29_EVEN_TWIDDLE_12, c0);
        s0 = fma(si13, T::R29_ODD_TWIDDLE_11, s0);
        let dc = c0 + a0;
        data[4] = dc;
        data[3] = -s0;
        let mut c0 = qc * T::R29_EVEN_TWIDDLE_1;
        let mut s0 = s_buffer[0];
        s0 *= T::R29_ODD_TWIDDLE_2;
        c0 = fma(ci1, T::R29_EVEN_TWIDDLE_4, c0);
        s0 = fma(si1, T::R29_ODD_TWIDDLE_7, s0);
        c0 = fma(ci2, T::R29_EVEN_TWIDDLE_7, c0);
        s0 = fma(si2, T::R29_ODD_TWIDDLE_12, s0);
        c0 = fma(ci3, T::R29_EVEN_TWIDDLE_10, c0);
        s0 = fma(si3, -T::R29_ODD_TWIDDLE_11, s0);
        c0 = fma(ci4, T::R29_EVEN_TWIDDLE_13, c0);
        s0 = fma(si4, -T::R29_ODD_TWIDDLE_6, s0);
        c0 = fma(ci5, T::R29_EVEN_TWIDDLE_12, c0);
        s0 = fma(si5, -T::R29_ODD_TWIDDLE_1, s0);
        c0 = fma(ci6, T::R29_EVEN_TWIDDLE_9, c0);
        s0 = fma(si6, -T::R29_ODD_TWIDDLE_3, s0);
        c0 = fma(ci7, T::R29_EVEN_TWIDDLE_6, c0);
        s0 = fma(si7, -T::R29_ODD_TWIDDLE_8, s0);
        c0 = fma(ci8, T::R29_EVEN_TWIDDLE_3, c0);
        s0 = fma(si8, -T::R29_ODD_TWIDDLE_13, s0);
        c0 = fma(ci9, T::R29_EVEN_TWIDDLE_0, c0);
        s0 = fma(si9, T::R29_ODD_TWIDDLE_10, s0);
        c0 = fma(ci10, T::R29_EVEN_TWIDDLE_2, c0);
        s0 = fma(si10, T::R29_ODD_TWIDDLE_5, s0);
        c0 = fma(ci11, T::R29_EVEN_TWIDDLE_5, c0);
        s0 = fma(si11, T::R29_ODD_TWIDDLE_0, s0);
        c0 = fma(ci12, T::R29_EVEN_TWIDDLE_8, c0);
        s0 = fma(si12, T::R29_ODD_TWIDDLE_4, s0);
        c0 = fma(ci13, T::R29_EVEN_TWIDDLE_11, c0);
        s0 = fma(si13, T::R29_ODD_TWIDDLE_9, s0);
        let dc = c0 + a0;
        data[6] = -dc;
        data[5] = s0;
        let mut c0 = qc * T::R29_EVEN_TWIDDLE_12;
        let mut s0 = s_buffer[0];
        s0 *= -T::R29_ODD_TWIDDLE_3;
        c0 = fma(ci1, T::R29_EVEN_TWIDDLE_8, c0);
        s0 = fma(si1, -T::R29_ODD_TWIDDLE_10, s0);
        c0 = fma(ci2, T::R29_EVEN_TWIDDLE_4, c0);
        s0 = fma(si2, T::R29_ODD_TWIDDLE_11, s0);
        c0 = fma(ci3, T::R29_EVEN_TWIDDLE_0, c0);
        s0 = fma(si3, T::R29_ODD_TWIDDLE_4, s0);
        c0 = fma(ci4, T::R29_EVEN_TWIDDLE_3, c0);
        s0 = fma(si4, T::R29_ODD_TWIDDLE_2, s0);
        c0 = fma(ci5, T::R29_EVEN_TWIDDLE_7, c0);
        s0 = fma(si5, T::R29_ODD_TWIDDLE_9, s0);
        c0 = fma(ci6, T::R29_EVEN_TWIDDLE_11, c0);
        s0 = fma(si6, -T::R29_ODD_TWIDDLE_12, s0);
        c0 = fma(ci7, T::R29_EVEN_TWIDDLE_13, c0);
        s0 = fma(si7, -T::R29_ODD_TWIDDLE_5, s0);
        c0 = fma(ci8, T::R29_EVEN_TWIDDLE_9, c0);
        s0 = fma(si8, -T::R29_ODD_TWIDDLE_1, s0);
        c0 = fma(ci9, T::R29_EVEN_TWIDDLE_5, c0);
        s0 = fma(si9, -T::R29_ODD_TWIDDLE_8, s0);
        c0 = fma(ci10, T::R29_EVEN_TWIDDLE_1, c0);
        s0 = fma(si10, T::R29_ODD_TWIDDLE_13, s0);
        c0 = fma(ci11, T::R29_EVEN_TWIDDLE_2, c0);
        s0 = fma(si11, T::R29_ODD_TWIDDLE_6, s0);
        c0 = fma(ci12, T::R29_EVEN_TWIDDLE_6, c0);
        s0 = fma(si12, T::R29_ODD_TWIDDLE_0, s0);
        c0 = fma(ci13, T::R29_EVEN_TWIDDLE_10, c0);
        s0 = fma(si13, T::R29_ODD_TWIDDLE_7, s0);
        let dc = c0 + a0;
        data[8] = dc;
        data[7] = -s0;
        let mut c0 = qc * T::R29_EVEN_TWIDDLE_2;
        let mut s0 = s_buffer[0];
        s0 *= T::R29_ODD_TWIDDLE_4;
        c0 = fma(ci1, T::R29_EVEN_TWIDDLE_7, c0);
        s0 = fma(si1, T::R29_ODD_TWIDDLE_13, s0);
        c0 = fma(ci2, T::R29_EVEN_TWIDDLE_12, c0);
        s0 = fma(si2, -T::R29_ODD_TWIDDLE_6, s0);
        c0 = fma(ci3, T::R29_EVEN_TWIDDLE_11, c0);
        s0 = fma(si3, -T::R29_ODD_TWIDDLE_2, s0);
        c0 = fma(ci4, T::R29_EVEN_TWIDDLE_6, c0);
        s0 = fma(si4, -T::R29_ODD_TWIDDLE_11, s0);
        c0 = fma(ci5, T::R29_EVEN_TWIDDLE_1, c0);
        s0 = fma(si5, T::R29_ODD_TWIDDLE_8, s0);
        c0 = fma(ci6, T::R29_EVEN_TWIDDLE_3, c0);
        s0 = fma(si6, T::R29_ODD_TWIDDLE_0, s0);
        c0 = fma(ci7, T::R29_EVEN_TWIDDLE_8, c0);
        s0 = fma(si7, T::R29_ODD_TWIDDLE_9, s0);
        c0 = fma(ci8, T::R29_EVEN_TWIDDLE_13, c0);
        s0 = fma(si8, -T::R29_ODD_TWIDDLE_10, s0);
        c0 = fma(ci9, T::R29_EVEN_TWIDDLE_10, c0);
        s0 = fma(si9, -T::R29_ODD_TWIDDLE_1, s0);
        c0 = fma(ci10, T::R29_EVEN_TWIDDLE_5, c0);
        s0 = fma(si10, -T::R29_ODD_TWIDDLE_7, s0);
        c0 = fma(ci11, T::R29_EVEN_TWIDDLE_0, c0);
        s0 = fma(si11, T::R29_ODD_TWIDDLE_12, s0);
        c0 = fma(ci12, T::R29_EVEN_TWIDDLE_4, c0);
        s0 = fma(si12, T::R29_ODD_TWIDDLE_3, s0);
        c0 = fma(ci13, T::R29_EVEN_TWIDDLE_9, c0);
        s0 = fma(si13, T::R29_ODD_TWIDDLE_5, s0);
        let dc = c0 + a0;
        data[10] = -dc;
        data[9] = s0;
        let mut c0 = qc * T::R29_EVEN_TWIDDLE_11;
        let mut s0 = s_buffer[0];
        s0 *= -T::R29_ODD_TWIDDLE_5;
        c0 = fma(ci1, T::R29_EVEN_TWIDDLE_5, c0);
        s0 = fma(si1, T::R29_ODD_TWIDDLE_12, s0);
        c0 = fma(ci2, T::R29_EVEN_TWIDDLE_0, c0);
        s0 = fma(si2, T::R29_ODD_TWIDDLE_1, s0);
        c0 = fma(ci3, T::R29_EVEN_TWIDDLE_6, c0);
        s0 = fma(si3, T::R29_ODD_TWIDDLE_9, s0);
        c0 = fma(ci4, T::R29_EVEN_TWIDDLE_12, c0);
        s0 = fma(si4, -T::R29_ODD_TWIDDLE_8, s0);
        c0 = fma(ci5, T::R29_EVEN_TWIDDLE_10, c0);
        s0 = fma(si5, -T::R29_ODD_TWIDDLE_2, s0);
        c0 = fma(ci6, T::R29_EVEN_TWIDDLE_4, c0);
        s0 = fma(si6, -T::R29_ODD_TWIDDLE_13, s0);
        c0 = fma(ci7, T::R29_EVEN_TWIDDLE_1, c0);
        s0 = fma(si7, T::R29_ODD_TWIDDLE_4, s0);
        c0 = fma(ci8, T::R29_EVEN_TWIDDLE_7, c0);
        s0 = fma(si8, T::R29_ODD_TWIDDLE_6, s0);
        c0 = fma(ci9, T::R29_EVEN_TWIDDLE_13, c0);
        s0 = fma(si9, -T::R29_ODD_TWIDDLE_11, s0);
        c0 = fma(ci10, T::R29_EVEN_TWIDDLE_9, c0);
        s0 = fma(si10, -T::R29_ODD_TWIDDLE_0, s0);
        c0 = fma(ci11, T::R29_EVEN_TWIDDLE_3, c0);
        s0 = fma(si11, -T::R29_ODD_TWIDDLE_10, s0);
        c0 = fma(ci12, T::R29_EVEN_TWIDDLE_2, c0);
        s0 = fma(si12, T::R29_ODD_TWIDDLE_7, s0);
        c0 = fma(ci13, T::R29_EVEN_TWIDDLE_8, c0);
        s0 = fma(si13, T::R29_ODD_TWIDDLE_3, s0);
        let dc = c0 + a0;
        data[12] = dc;
        data[11] = -s0;
        let mut c0 = qc * T::R29_EVEN_TWIDDLE_3;
        let mut s0 = s_buffer[0];
        s0 *= T::R29_ODD_TWIDDLE_6;
        c0 = fma(ci1, T::R29_EVEN_TWIDDLE_10, c0);
        s0 = fma(si1, -T::R29_ODD_TWIDDLE_9, s0);
        c0 = fma(ci2, T::R29_EVEN_TWIDDLE_11, c0);
        s0 = fma(si2, -T::R29_ODD_TWIDDLE_3, s0);
        c0 = fma(ci3, T::R29_EVEN_TWIDDLE_4, c0);
        s0 = fma(si3, T::R29_ODD_TWIDDLE_12, s0);
        c0 = fma(ci4, T::R29_EVEN_TWIDDLE_2, c0);
        s0 = fma(si4, T::R29_ODD_TWIDDLE_0, s0);
        c0 = fma(ci5, T::R29_EVEN_TWIDDLE_9, c0);
        s0 = fma(si5, T::R29_ODD_TWIDDLE_13, s0);
        c0 = fma(ci6, T::R29_EVEN_TWIDDLE_12, c0);
        s0 = fma(si6, -T::R29_ODD_TWIDDLE_2, s0);
        c0 = fma(ci7, T::R29_EVEN_TWIDDLE_5, c0);
        s0 = fma(si7, -T::R29_ODD_TWIDDLE_10, s0);
        c0 = fma(ci8, T::R29_EVEN_TWIDDLE_1, c0);
        s0 = fma(si8, T::R29_ODD_TWIDDLE_5, s0);
        c0 = fma(ci9, T::R29_EVEN_TWIDDLE_8, c0);
        s0 = fma(si9, T::R29_ODD_TWIDDLE_7, s0);
        c0 = fma(ci10, T::R29_EVEN_TWIDDLE_13, c0);
        s0 = fma(si10, -T::R29_ODD_TWIDDLE_8, s0);
        c0 = fma(ci11, T::R29_EVEN_TWIDDLE_6, c0);
        s0 = fma(si11, -T::R29_ODD_TWIDDLE_4, s0);
        c0 = fma(ci12, T::R29_EVEN_TWIDDLE_0, c0);
        s0 = fma(si12, T::R29_ODD_TWIDDLE_11, s0);
        c0 = fma(ci13, T::R29_EVEN_TWIDDLE_7, c0);
        s0 = fma(si13, T::R29_ODD_TWIDDLE_1, s0);
        let dc = c0 + a0;
        data[14] = -dc;
        data[13] = s0;
        let mut c0 = qc * T::R29_EVEN_TWIDDLE_10;
        let mut s0 = s_buffer[0];
        s0 *= -T::R29_ODD_TWIDDLE_7;
        c0 = fma(ci1, T::R29_EVEN_TWIDDLE_2, c0);
        s0 = fma(si1, T::R29_ODD_TWIDDLE_6, s0);
        c0 = fma(ci2, T::R29_EVEN_TWIDDLE_5, c0);
        s0 = fma(si2, T::R29_ODD_TWIDDLE_8, s0);
        c0 = fma(ci3, T::R29_EVEN_TWIDDLE_13, c0);
        s0 = fma(si3, -T::R29_ODD_TWIDDLE_5, s0);
        c0 = fma(ci4, T::R29_EVEN_TWIDDLE_7, c0);
        s0 = fma(si4, -T::R29_ODD_TWIDDLE_9, s0);
        c0 = fma(ci5, T::R29_EVEN_TWIDDLE_0, c0);
        s0 = fma(si5, T::R29_ODD_TWIDDLE_4, s0);
        c0 = fma(ci6, T::R29_EVEN_TWIDDLE_8, c0);
        s0 = fma(si6, T::R29_ODD_TWIDDLE_10, s0);
        c0 = fma(ci7, T::R29_EVEN_TWIDDLE_12, c0);
        s0 = fma(si7, -T::R29_ODD_TWIDDLE_3, s0);
        c0 = fma(ci8, T::R29_EVEN_TWIDDLE_4, c0);
        s0 = fma(si8, -T::R29_ODD_TWIDDLE_11, s0);
        c0 = fma(ci9, T::R29_EVEN_TWIDDLE_3, c0);
        s0 = fma(si9, T::R29_ODD_TWIDDLE_2, s0);
        c0 = fma(ci10, T::R29_EVEN_TWIDDLE_11, c0);
        s0 = fma(si10, T::R29_ODD_TWIDDLE_12, s0);
        c0 = fma(ci11, T::R29_EVEN_TWIDDLE_9, c0);
        s0 = fma(si11, -T::R29_ODD_TWIDDLE_1, s0);
        c0 = fma(ci12, T::R29_EVEN_TWIDDLE_1, c0);
        s0 = fma(si12, -T::R29_ODD_TWIDDLE_13, s0);
        c0 = fma(ci13, T::R29_EVEN_TWIDDLE_6, c0);
        s0 = fma(si13, T::R29_ODD_TWIDDLE_0, s0);
        let dc = c0 + a0;
        data[16] = dc;
        data[15] = -s0;
        let mut c0 = qc * T::R29_EVEN_TWIDDLE_4;
        let mut s0 = s_buffer[0];
        s0 *= T::R29_ODD_TWIDDLE_8;
        c0 = fma(ci1, T::R29_EVEN_TWIDDLE_13, c0);
        s0 = fma(si1, -T::R29_ODD_TWIDDLE_3, s0);
        c0 = fma(ci2, T::R29_EVEN_TWIDDLE_6, c0);
        s0 = fma(si2, -T::R29_ODD_TWIDDLE_13, s0);
        c0 = fma(ci3, T::R29_EVEN_TWIDDLE_2, c0);
        s0 = fma(si3, T::R29_ODD_TWIDDLE_1, s0);
        c0 = fma(ci4, T::R29_EVEN_TWIDDLE_11, c0);
        s0 = fma(si4, -T::R29_ODD_TWIDDLE_10, s0);
        c0 = fma(ci5, T::R29_EVEN_TWIDDLE_8, c0);
        s0 = fma(si5, -T::R29_ODD_TWIDDLE_6, s0);
        c0 = fma(ci6, T::R29_EVEN_TWIDDLE_0, c0);
        s0 = fma(si6, T::R29_ODD_TWIDDLE_5, s0);
        c0 = fma(ci7, T::R29_EVEN_TWIDDLE_9, c0);
        s0 = fma(si7, T::R29_ODD_TWIDDLE_11, s0);
        c0 = fma(ci8, T::R29_EVEN_TWIDDLE_10, c0);
        s0 = fma(si8, -T::R29_ODD_TWIDDLE_0, s0);
        c0 = fma(ci9, T::R29_EVEN_TWIDDLE_1, c0);
        s0 = fma(si9, T::R29_ODD_TWIDDLE_12, s0);
        c0 = fma(ci10, T::R29_EVEN_TWIDDLE_7, c0);
        s0 = fma(si10, T::R29_ODD_TWIDDLE_4, s0);
        c0 = fma(ci11, T::R29_EVEN_TWIDDLE_12, c0);
        s0 = fma(si11, -T::R29_ODD_TWIDDLE_7, s0);
        c0 = fma(ci12, T::R29_EVEN_TWIDDLE_3, c0);
        s0 = fma(si12, -T::R29_ODD_TWIDDLE_9, s0);
        c0 = fma(ci13, T::R29_EVEN_TWIDDLE_5, c0);
        s0 = fma(si13, T::R29_ODD_TWIDDLE_2, s0);
        let dc = c0 + a0;
        data[18] = -dc;
        data[17] = s0;
        let mut c0 = qc * T::R29_EVEN_TWIDDLE_9;
        let mut s0 = s_buffer[0];
        s0 *= -T::R29_ODD_TWIDDLE_9;
        c0 = fma(ci1, T::R29_EVEN_TWIDDLE_0, c0);
        s0 = fma(si1, T::R29_ODD_TWIDDLE_0, s0);
        c0 = fma(ci2, T::R29_EVEN_TWIDDLE_10, c0);
        s0 = fma(si2, -T::R29_ODD_TWIDDLE_10, s0);
        c0 = fma(ci3, T::R29_EVEN_TWIDDLE_8, c0);
        s0 = fma(si3, -T::R29_ODD_TWIDDLE_8, s0);
        c0 = fma(ci4, T::R29_EVEN_TWIDDLE_1, c0);
        s0 = fma(si4, T::R29_ODD_TWIDDLE_1, s0);
        c0 = fma(ci5, T::R29_EVEN_TWIDDLE_11, c0);
        s0 = fma(si5, -T::R29_ODD_TWIDDLE_11, s0);
        c0 = fma(ci6, T::R29_EVEN_TWIDDLE_7, c0);
        s0 = fma(si6, -T::R29_ODD_TWIDDLE_7, s0);
        c0 = fma(ci7, T::R29_EVEN_TWIDDLE_2, c0);
        s0 = fma(si7, T::R29_ODD_TWIDDLE_2, s0);
        c0 = fma(ci8, T::R29_EVEN_TWIDDLE_12, c0);
        s0 = fma(si8, -T::R29_ODD_TWIDDLE_12, s0);
        c0 = fma(ci9, T::R29_EVEN_TWIDDLE_6, c0);
        s0 = fma(si9, -T::R29_ODD_TWIDDLE_6, s0);
        c0 = fma(ci10, T::R29_EVEN_TWIDDLE_3, c0);
        s0 = fma(si10, T::R29_ODD_TWIDDLE_3, s0);
        c0 = fma(ci11, T::R29_EVEN_TWIDDLE_13, c0);
        s0 = fma(si11, -T::R29_ODD_TWIDDLE_13, s0);
        c0 = fma(ci12, T::R29_EVEN_TWIDDLE_5, c0);
        s0 = fma(si12, -T::R29_ODD_TWIDDLE_5, s0);
        c0 = fma(ci13, T::R29_EVEN_TWIDDLE_4, c0);
        s0 = fma(si13, T::R29_ODD_TWIDDLE_4, s0);
        let dc = c0 + a0;
        data[20] = dc;
        data[19] = -s0;
        let mut c0 = qc * T::R29_EVEN_TWIDDLE_5;
        let mut s0 = s_buffer[0];
        s0 *= T::R29_ODD_TWIDDLE_10;
        c0 = fma(ci1, T::R29_EVEN_TWIDDLE_12, c0);
        s0 = fma(si1, -T::R29_ODD_TWIDDLE_2, s0);
        c0 = fma(ci2, T::R29_EVEN_TWIDDLE_1, c0);
        s0 = fma(si2, T::R29_ODD_TWIDDLE_5, s0);
        c0 = fma(ci3, T::R29_EVEN_TWIDDLE_9, c0);
        s0 = fma(si3, -T::R29_ODD_TWIDDLE_13, s0);
        c0 = fma(ci4, T::R29_EVEN_TWIDDLE_8, c0);
        s0 = fma(si4, -T::R29_ODD_TWIDDLE_7, s0);
        c0 = fma(ci5, T::R29_EVEN_TWIDDLE_2, c0);
        s0 = fma(si5, T::R29_ODD_TWIDDLE_0, s0);
        c0 = fma(ci6, T::R29_EVEN_TWIDDLE_13, c0);
        s0 = fma(si6, -T::R29_ODD_TWIDDLE_8, s0);
        c0 = fma(ci7, T::R29_EVEN_TWIDDLE_4, c0);
        s0 = fma(si7, -T::R29_ODD_TWIDDLE_12, s0);
        c0 = fma(ci8, T::R29_EVEN_TWIDDLE_6, c0);
        s0 = fma(si8, T::R29_ODD_TWIDDLE_4, s0);
        c0 = fma(ci9, T::R29_EVEN_TWIDDLE_11, c0);
        s0 = fma(si9, -T::R29_ODD_TWIDDLE_3, s0);
        c0 = fma(ci10, T::R29_EVEN_TWIDDLE_0, c0);
        s0 = fma(si10, T::R29_ODD_TWIDDLE_11, s0);
        c0 = fma(ci11, T::R29_EVEN_TWIDDLE_10, c0);
        s0 = fma(si11, T::R29_ODD_TWIDDLE_9, s0);
        c0 = fma(ci12, T::R29_EVEN_TWIDDLE_7, c0);
        s0 = fma(si12, -T::R29_ODD_TWIDDLE_1, s0);
        c0 = fma(ci13, T::R29_EVEN_TWIDDLE_3, c0);
        s0 = fma(si13, T::R29_ODD_TWIDDLE_6, s0);
        let dc = c0 + a0;
        data[22] = -dc;
        data[21] = s0;
        let mut c0 = qc * T::R29_EVEN_TWIDDLE_8;
        let mut s0 = s_buffer[0];
        s0 *= -T::R29_ODD_TWIDDLE_11;
        c0 = fma(ci1, T::R29_EVEN_TWIDDLE_3, c0);
        s0 = fma(si1, T::R29_ODD_TWIDDLE_5, s0);
        c0 = fma(ci2, T::R29_EVEN_TWIDDLE_13, c0);
        s0 = fma(si2, -T::R29_ODD_TWIDDLE_0, s0);
        c0 = fma(ci3, T::R29_EVEN_TWIDDLE_1, c0);
        s0 = fma(si3, T::R29_ODD_TWIDDLE_6, s0);
        c0 = fma(ci4, T::R29_EVEN_TWIDDLE_10, c0);
        s0 = fma(si4, -T::R29_ODD_TWIDDLE_12, s0);
        c0 = fma(ci5, T::R29_EVEN_TWIDDLE_6, c0);
        s0 = fma(si5, -T::R29_ODD_TWIDDLE_10, s0);
        c0 = fma(ci6, T::R29_EVEN_TWIDDLE_5, c0);
        s0 = fma(si6, T::R29_ODD_TWIDDLE_4, s0);
        c0 = fma(ci7, T::R29_EVEN_TWIDDLE_11, c0);
        s0 = fma(si7, -T::R29_ODD_TWIDDLE_1, s0);
        c0 = fma(ci8, T::R29_EVEN_TWIDDLE_0, c0);
        s0 = fma(si8, T::R29_ODD_TWIDDLE_7, s0);
        c0 = fma(ci9, T::R29_EVEN_TWIDDLE_12, c0);
        s0 = fma(si9, -T::R29_ODD_TWIDDLE_13, s0);
        c0 = fma(ci10, T::R29_EVEN_TWIDDLE_4, c0);
        s0 = fma(si10, -T::R29_ODD_TWIDDLE_9, s0);
        c0 = fma(ci11, T::R29_EVEN_TWIDDLE_7, c0);
        s0 = fma(si11, T::R29_ODD_TWIDDLE_3, s0);
        c0 = fma(ci12, T::R29_EVEN_TWIDDLE_9, c0);
        s0 = fma(si12, -T::R29_ODD_TWIDDLE_2, s0);
        c0 = fma(ci13, T::R29_EVEN_TWIDDLE_2, c0);
        s0 = fma(si13, T::R29_ODD_TWIDDLE_8, s0);
        let dc = c0 + a0;
        data[24] = dc;
        data[23] = -s0;
        let mut c0 = qc * T::R29_EVEN_TWIDDLE_6;
        let mut s0 = s_buffer[0];
        s0 *= T::R29_ODD_TWIDDLE_12;
        c0 = fma(ci1, T::R29_EVEN_TWIDDLE_9, c0);
        s0 = fma(si1, -T::R29_ODD_TWIDDLE_8, s0);
        c0 = fma(ci2, T::R29_EVEN_TWIDDLE_3, c0);
        s0 = fma(si2, T::R29_ODD_TWIDDLE_4, s0);
        c0 = fma(ci3, T::R29_EVEN_TWIDDLE_12, c0);
        s0 = fma(si3, -T::R29_ODD_TWIDDLE_0, s0);
        c0 = fma(ci4, T::R29_EVEN_TWIDDLE_0, c0);
        s0 = fma(si4, T::R29_ODD_TWIDDLE_3, s0);
        c0 = fma(ci5, T::R29_EVEN_TWIDDLE_13, c0);
        s0 = fma(si5, -T::R29_ODD_TWIDDLE_7, s0);
        c0 = fma(ci6, T::R29_EVEN_TWIDDLE_2, c0);
        s0 = fma(si6, T::R29_ODD_TWIDDLE_11, s0);
        c0 = fma(ci7, T::R29_EVEN_TWIDDLE_10, c0);
        s0 = fma(si7, T::R29_ODD_TWIDDLE_13, s0);
        c0 = fma(ci8, T::R29_EVEN_TWIDDLE_5, c0);
        s0 = fma(si8, -T::R29_ODD_TWIDDLE_9, s0);
        c0 = fma(ci9, T::R29_EVEN_TWIDDLE_7, c0);
        s0 = fma(si9, T::R29_ODD_TWIDDLE_5, s0);
        c0 = fma(ci10, T::R29_EVEN_TWIDDLE_8, c0);
        s0 = fma(si10, -T::R29_ODD_TWIDDLE_1, s0);
        c0 = fma(ci11, T::R29_EVEN_TWIDDLE_4, c0);
        s0 = fma(si11, T::R29_ODD_TWIDDLE_2, s0);
        c0 = fma(ci12, T::R29_EVEN_TWIDDLE_11, c0);
        s0 = fma(si12, -T::R29_ODD_TWIDDLE_6, s0);
        c0 = fma(ci13, T::R29_EVEN_TWIDDLE_1, c0);
        s0 = fma(si13, T::R29_ODD_TWIDDLE_10, s0);
        let dc = c0 + a0;
        data[26] = -dc;
        data[25] = s0;
        let mut c0 = qc * T::R29_EVEN_TWIDDLE_7;
        let mut s0 = s_buffer[0];
        s0 *= -T::R29_ODD_TWIDDLE_13;
        c0 = fma(ci1, T::R29_EVEN_TWIDDLE_6, c0);
        s0 = fma(si1, T::R29_ODD_TWIDDLE_11, s0);
        c0 = fma(ci2, T::R29_EVEN_TWIDDLE_8, c0);
        s0 = fma(si2, -T::R29_ODD_TWIDDLE_9, s0);
        c0 = fma(ci3, T::R29_EVEN_TWIDDLE_5, c0);
        s0 = fma(si3, T::R29_ODD_TWIDDLE_7, s0);
        c0 = fma(ci4, T::R29_EVEN_TWIDDLE_9, c0);
        s0 = fma(si4, -T::R29_ODD_TWIDDLE_5, s0);
        c0 = fma(ci5, T::R29_EVEN_TWIDDLE_4, c0);
        s0 = fma(si5, T::R29_ODD_TWIDDLE_3, s0);
        c0 = fma(ci6, T::R29_EVEN_TWIDDLE_10, c0);
        s0 = fma(si6, -T::R29_ODD_TWIDDLE_1, s0);
        c0 = fma(ci7, T::R29_EVEN_TWIDDLE_3, c0);
        s0 = fma(si7, T::R29_ODD_TWIDDLE_0, s0);
        c0 = fma(ci8, T::R29_EVEN_TWIDDLE_11, c0);
        s0 = fma(si8, -T::R29_ODD_TWIDDLE_2, s0);
        c0 = fma(ci9, T::R29_EVEN_TWIDDLE_2, c0);
        s0 = fma(si9, T::R29_ODD_TWIDDLE_4, s0);
        c0 = fma(ci10, T::R29_EVEN_TWIDDLE_12, c0);
        s0 = fma(si10, -T::R29_ODD_TWIDDLE_6, s0);
        c0 = fma(ci11, T::R29_EVEN_TWIDDLE_1, c0);
        s0 = fma(si11, T::R29_ODD_TWIDDLE_8, s0);
        c0 = fma(ci12, T::R29_EVEN_TWIDDLE_13, c0);
        s0 = fma(si12, -T::R29_ODD_TWIDDLE_10, s0);
        c0 = fma(ci13, T::R29_EVEN_TWIDDLE_0, c0);
        s0 = fma(si13, T::R29_ODD_TWIDDLE_12, s0);
        let dc = c0 + a0;
        data[28] = dc;
        data[27] = -s0;
    }
}

define_avx_butterfly!(AvxDct2Butterfly29, 29);

#[derive(Debug, Clone)]
pub(crate) struct AvxDct2Butterfly31<T: DctSample> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T: DctSample> Default for AvxDct2Butterfly31<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: DctSample + MixedRadix31Sample> AvxDct2Butterfly31<T>
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
        s0 = fma(si1, T::R31_ODD_TWIDDLE_1, s0);
        let ci2 = c_buffer[2];
        let si2 = s_buffer[2];
        c0 = ci2 + c0;
        c1 = ci2 * T::R31_EVEN_TWIDDLE_2 + c1;
        s0 = fma(si2, T::R31_ODD_TWIDDLE_2, s0);
        let ci3 = c_buffer[3];
        let si3 = s_buffer[3];
        c0 = ci3 + c0;
        c1 = ci3 * T::R31_EVEN_TWIDDLE_3 + c1;
        s0 = fma(si3, T::R31_ODD_TWIDDLE_3, s0);
        let ci4 = c_buffer[4];
        let si4 = s_buffer[4];
        c0 = ci4 + c0;
        c1 = ci4 * T::R31_EVEN_TWIDDLE_4 + c1;
        s0 = fma(si4, T::R31_ODD_TWIDDLE_4, s0);
        let ci5 = c_buffer[5];
        let si5 = s_buffer[5];
        c0 = ci5 + c0;
        c1 = ci5 * T::R31_EVEN_TWIDDLE_5 + c1;
        s0 = fma(si5, T::R31_ODD_TWIDDLE_5, s0);
        let ci6 = c_buffer[6];
        let si6 = s_buffer[6];
        c0 = ci6 + c0;
        c1 = ci6 * T::R31_EVEN_TWIDDLE_6 + c1;
        s0 = fma(si6, T::R31_ODD_TWIDDLE_6, s0);
        let ci7 = c_buffer[7];
        let si7 = s_buffer[7];
        c0 = ci7 + c0;
        c1 = ci7 * T::R31_EVEN_TWIDDLE_7 + c1;
        s0 = fma(si7, T::R31_ODD_TWIDDLE_7, s0);
        let ci8 = c_buffer[8];
        let si8 = s_buffer[8];
        c0 = ci8 + c0;
        c1 = ci8 * T::R31_EVEN_TWIDDLE_8 + c1;
        s0 = fma(si8, T::R31_ODD_TWIDDLE_8, s0);
        let ci9 = c_buffer[9];
        let si9 = s_buffer[9];
        c0 = ci9 + c0;
        c1 = ci9 * T::R31_EVEN_TWIDDLE_9 + c1;
        s0 = fma(si9, T::R31_ODD_TWIDDLE_9, s0);
        let ci10 = c_buffer[10];
        let si10 = s_buffer[10];
        c0 = ci10 + c0;
        c1 = ci10 * T::R31_EVEN_TWIDDLE_10 + c1;
        s0 = fma(si10, T::R31_ODD_TWIDDLE_10, s0);
        let ci11 = c_buffer[11];
        let si11 = s_buffer[11];
        c0 = ci11 + c0;
        c1 = ci11 * T::R31_EVEN_TWIDDLE_11 + c1;
        s0 = fma(si11, T::R31_ODD_TWIDDLE_11, s0);
        let ci12 = c_buffer[12];
        let si12 = s_buffer[12];
        c0 = ci12 + c0;
        c1 = ci12 * T::R31_EVEN_TWIDDLE_12 + c1;
        s0 = fma(si12, T::R31_ODD_TWIDDLE_12, s0);
        let ci13 = c_buffer[13];
        let si13 = s_buffer[13];
        c0 = ci13 + c0;
        c1 = ci13 * T::R31_EVEN_TWIDDLE_13 + c1;
        s0 = fma(si13, T::R31_ODD_TWIDDLE_13, s0);
        let ci14 = c_buffer[14];
        let si14 = s_buffer[14];
        c0 = ci14 + c0;
        c1 = ci14 * T::R31_EVEN_TWIDDLE_14 + c1;
        s0 = fma(si14, T::R31_ODD_TWIDDLE_14, s0);
        let a0 = a_buffer[0];
        let dc = c0 + a0;
        data[0] = dc;
        data[1] = s0;
        data[2] = -(c1 + a0);
        let mut c0 = qc * T::R31_EVEN_TWIDDLE_14;
        let mut s0 = s_buffer[0];
        s0 *= -T::R31_ODD_TWIDDLE_1;
        c0 = fma(ci1, T::R31_EVEN_TWIDDLE_12, c0);
        s0 = fma(si1, -T::R31_ODD_TWIDDLE_4, s0);
        c0 = fma(ci2, T::R31_EVEN_TWIDDLE_10, c0);
        s0 = fma(si2, -T::R31_ODD_TWIDDLE_7, s0);
        c0 = fma(ci3, T::R31_EVEN_TWIDDLE_8, c0);
        s0 = fma(si3, -T::R31_ODD_TWIDDLE_10, s0);
        c0 = fma(ci4, T::R31_EVEN_TWIDDLE_6, c0);
        s0 = fma(si4, -T::R31_ODD_TWIDDLE_13, s0);
        c0 = fma(ci5, T::R31_EVEN_TWIDDLE_4, c0);
        s0 = fma(si5, T::R31_ODD_TWIDDLE_14, s0);
        c0 = fma(ci6, T::R31_EVEN_TWIDDLE_2, c0);
        s0 = fma(si6, T::R31_ODD_TWIDDLE_11, s0);
        c0 = fma(ci7, T::R31_EVEN_TWIDDLE_0, c0);
        s0 = fma(si7, T::R31_ODD_TWIDDLE_8, s0);
        c0 = fma(ci8, T::R31_EVEN_TWIDDLE_1, c0);
        s0 = fma(si8, T::R31_ODD_TWIDDLE_5, s0);
        c0 = fma(ci9, T::R31_EVEN_TWIDDLE_3, c0);
        s0 = fma(si9, T::R31_ODD_TWIDDLE_2, s0);
        c0 = fma(ci10, T::R31_EVEN_TWIDDLE_5, c0);
        s0 = fma(si10, T::R31_ODD_TWIDDLE_0, s0);
        c0 = fma(ci11, T::R31_EVEN_TWIDDLE_7, c0);
        s0 = fma(si11, T::R31_ODD_TWIDDLE_3, s0);
        c0 = fma(ci12, T::R31_EVEN_TWIDDLE_9, c0);
        s0 = fma(si12, T::R31_ODD_TWIDDLE_6, s0);
        c0 = fma(ci13, T::R31_EVEN_TWIDDLE_11, c0);
        s0 = fma(si13, T::R31_ODD_TWIDDLE_9, s0);
        c0 = fma(ci14, T::R31_EVEN_TWIDDLE_13, c0);
        s0 = fma(si14, T::R31_ODD_TWIDDLE_12, s0);
        let dc = c0 + a0;
        data[4] = dc;
        data[3] = -s0;
        let mut c0 = qc * T::R31_EVEN_TWIDDLE_1;
        let mut s0 = s_buffer[0];
        s0 *= T::R31_ODD_TWIDDLE_2;
        c0 = fma(ci1, T::R31_EVEN_TWIDDLE_4, c0);
        s0 = fma(si1, T::R31_ODD_TWIDDLE_7, s0);
        c0 = fma(ci2, T::R31_EVEN_TWIDDLE_7, c0);
        s0 = fma(si2, T::R31_ODD_TWIDDLE_12, s0);
        c0 = fma(ci3, T::R31_EVEN_TWIDDLE_10, c0);
        s0 = fma(si3, -T::R31_ODD_TWIDDLE_13, s0);
        c0 = fma(ci4, T::R31_EVEN_TWIDDLE_13, c0);
        s0 = fma(si4, -T::R31_ODD_TWIDDLE_8, s0);
        c0 = fma(ci5, T::R31_EVEN_TWIDDLE_14, c0);
        s0 = fma(si5, -T::R31_ODD_TWIDDLE_3, s0);
        c0 = fma(ci6, T::R31_EVEN_TWIDDLE_11, c0);
        s0 = fma(si6, -T::R31_ODD_TWIDDLE_1, s0);
        c0 = fma(ci7, T::R31_EVEN_TWIDDLE_8, c0);
        s0 = fma(si7, -T::R31_ODD_TWIDDLE_6, s0);
        c0 = fma(ci8, T::R31_EVEN_TWIDDLE_5, c0);
        s0 = fma(si8, -T::R31_ODD_TWIDDLE_11, s0);
        c0 = fma(ci9, T::R31_EVEN_TWIDDLE_2, c0);
        s0 = fma(si9, T::R31_ODD_TWIDDLE_14, s0);
        c0 = fma(ci10, T::R31_EVEN_TWIDDLE_0, c0);
        s0 = fma(si10, T::R31_ODD_TWIDDLE_9, s0);
        c0 = fma(ci11, T::R31_EVEN_TWIDDLE_3, c0);
        s0 = fma(si11, T::R31_ODD_TWIDDLE_4, s0);
        c0 = fma(ci12, T::R31_EVEN_TWIDDLE_6, c0);
        s0 = fma(si12, T::R31_ODD_TWIDDLE_0, s0);
        c0 = fma(ci13, T::R31_EVEN_TWIDDLE_9, c0);
        s0 = fma(si13, T::R31_ODD_TWIDDLE_5, s0);
        c0 = fma(ci14, T::R31_EVEN_TWIDDLE_12, c0);
        s0 = fma(si14, T::R31_ODD_TWIDDLE_10, s0);
        let dc = c0 + a0;
        data[6] = -dc;
        data[5] = s0;
        let mut c0 = qc * T::R31_EVEN_TWIDDLE_13;
        let mut s0 = s_buffer[0];
        s0 *= -T::R31_ODD_TWIDDLE_3;
        c0 = fma(ci1, T::R31_EVEN_TWIDDLE_9, c0);
        s0 = fma(si1, -T::R31_ODD_TWIDDLE_10, s0);
        c0 = fma(ci2, T::R31_EVEN_TWIDDLE_5, c0);
        s0 = fma(si2, T::R31_ODD_TWIDDLE_13, s0);
        c0 = fma(ci3, T::R31_EVEN_TWIDDLE_1, c0);
        s0 = fma(si3, T::R31_ODD_TWIDDLE_6, s0);
        c0 = fma(ci4, T::R31_EVEN_TWIDDLE_2, c0);
        s0 = fma(si4, T::R31_ODD_TWIDDLE_0, s0);
        c0 = fma(ci5, T::R31_EVEN_TWIDDLE_6, c0);
        s0 = fma(si5, T::R31_ODD_TWIDDLE_7, s0);
        c0 = fma(ci6, T::R31_EVEN_TWIDDLE_10, c0);
        s0 = fma(si6, T::R31_ODD_TWIDDLE_14, s0);
        c0 = fma(ci7, T::R31_EVEN_TWIDDLE_14, c0);
        s0 = fma(si7, -T::R31_ODD_TWIDDLE_9, s0);
        c0 = fma(ci8, T::R31_EVEN_TWIDDLE_12, c0);
        s0 = fma(si8, -T::R31_ODD_TWIDDLE_2, s0);
        c0 = fma(ci9, T::R31_EVEN_TWIDDLE_8, c0);
        s0 = fma(si9, -T::R31_ODD_TWIDDLE_4, s0);
        c0 = fma(ci10, T::R31_EVEN_TWIDDLE_4, c0);
        s0 = fma(si10, -T::R31_ODD_TWIDDLE_11, s0);
        c0 = fma(ci11, T::R31_EVEN_TWIDDLE_0, c0);
        s0 = fma(si11, T::R31_ODD_TWIDDLE_12, s0);
        c0 = fma(ci12, T::R31_EVEN_TWIDDLE_3, c0);
        s0 = fma(si12, T::R31_ODD_TWIDDLE_5, s0);
        c0 = fma(ci13, T::R31_EVEN_TWIDDLE_7, c0);
        s0 = fma(si13, T::R31_ODD_TWIDDLE_1, s0);
        c0 = fma(ci14, T::R31_EVEN_TWIDDLE_11, c0);
        s0 = fma(si14, T::R31_ODD_TWIDDLE_8, s0);
        let dc = c0 + a0;
        data[8] = dc;
        data[7] = -s0;
        let mut c0 = qc * T::R31_EVEN_TWIDDLE_2;
        let mut s0 = s_buffer[0];
        s0 *= T::R31_ODD_TWIDDLE_4;
        c0 = fma(ci1, T::R31_EVEN_TWIDDLE_7, c0);
        s0 = fma(si1, T::R31_ODD_TWIDDLE_13, s0);
        c0 = fma(ci2, T::R31_EVEN_TWIDDLE_12, c0);
        s0 = fma(si2, -T::R31_ODD_TWIDDLE_8, s0);
        c0 = fma(ci3, T::R31_EVEN_TWIDDLE_13, c0);
        s0 = fma(si3, -T::R31_ODD_TWIDDLE_0, s0);
        c0 = fma(ci4, T::R31_EVEN_TWIDDLE_8, c0);
        s0 = fma(si4, -T::R31_ODD_TWIDDLE_9, s0);
        c0 = fma(ci5, T::R31_EVEN_TWIDDLE_3, c0);
        s0 = fma(si5, T::R31_ODD_TWIDDLE_12, s0);
        c0 = fma(ci6, T::R31_EVEN_TWIDDLE_1, c0);
        s0 = fma(si6, T::R31_ODD_TWIDDLE_3, s0);
        c0 = fma(ci7, T::R31_EVEN_TWIDDLE_6, c0);
        s0 = fma(si7, T::R31_ODD_TWIDDLE_5, s0);
        c0 = fma(ci8, T::R31_EVEN_TWIDDLE_11, c0);
        s0 = fma(si8, T::R31_ODD_TWIDDLE_14, s0);
        c0 = fma(ci9, T::R31_EVEN_TWIDDLE_14, c0);
        s0 = fma(si9, -T::R31_ODD_TWIDDLE_7, s0);
        c0 = fma(ci10, T::R31_EVEN_TWIDDLE_9, c0);
        s0 = fma(si10, -T::R31_ODD_TWIDDLE_1, s0);
        c0 = fma(ci11, T::R31_EVEN_TWIDDLE_4, c0);
        s0 = fma(si11, -T::R31_ODD_TWIDDLE_10, s0);
        c0 = fma(ci12, T::R31_EVEN_TWIDDLE_0, c0);
        s0 = fma(si12, T::R31_ODD_TWIDDLE_11, s0);
        c0 = fma(ci13, T::R31_EVEN_TWIDDLE_5, c0);
        s0 = fma(si13, T::R31_ODD_TWIDDLE_2, s0);
        c0 = fma(ci14, T::R31_EVEN_TWIDDLE_10, c0);
        s0 = fma(si14, T::R31_ODD_TWIDDLE_6, s0);
        let dc = c0 + a0;
        data[10] = -dc;
        data[9] = s0;
        let mut c0 = qc * T::R31_EVEN_TWIDDLE_12;
        let mut s0 = s_buffer[0];
        s0 *= -T::R31_ODD_TWIDDLE_5;
        c0 = fma(ci1, T::R31_EVEN_TWIDDLE_6, c0);
        s0 = fma(si1, T::R31_ODD_TWIDDLE_14, s0);
        c0 = fma(ci2, T::R31_EVEN_TWIDDLE_0, c0);
        s0 = fma(si2, T::R31_ODD_TWIDDLE_3, s0);
        c0 = fma(ci3, T::R31_EVEN_TWIDDLE_5, c0);
        s0 = fma(si3, T::R31_ODD_TWIDDLE_7, s0);
        c0 = fma(ci4, T::R31_EVEN_TWIDDLE_11, c0);
        s0 = fma(si4, -T::R31_ODD_TWIDDLE_12, s0);
        c0 = fma(ci5, T::R31_EVEN_TWIDDLE_13, c0);
        s0 = fma(si5, -T::R31_ODD_TWIDDLE_1, s0);
        c0 = fma(ci6, T::R31_EVEN_TWIDDLE_7, c0);
        s0 = fma(si6, -T::R31_ODD_TWIDDLE_9, s0);
        c0 = fma(ci7, T::R31_EVEN_TWIDDLE_1, c0);
        s0 = fma(si7, T::R31_ODD_TWIDDLE_10, s0);
        c0 = fma(ci8, T::R31_EVEN_TWIDDLE_4, c0);
        s0 = fma(si8, T::R31_ODD_TWIDDLE_0, s0);
        c0 = fma(ci9, T::R31_EVEN_TWIDDLE_10, c0);
        s0 = fma(si9, T::R31_ODD_TWIDDLE_11, s0);
        c0 = fma(ci10, T::R31_EVEN_TWIDDLE_14, c0);
        s0 = fma(si10, -T::R31_ODD_TWIDDLE_8, s0);
        c0 = fma(ci11, T::R31_EVEN_TWIDDLE_8, c0);
        s0 = fma(si11, -T::R31_ODD_TWIDDLE_2, s0);
        c0 = fma(ci12, T::R31_EVEN_TWIDDLE_2, c0);
        s0 = fma(si12, -T::R31_ODD_TWIDDLE_13, s0);
        c0 = fma(ci13, T::R31_EVEN_TWIDDLE_3, c0);
        s0 = fma(si13, T::R31_ODD_TWIDDLE_6, s0);
        c0 = fma(ci14, T::R31_EVEN_TWIDDLE_9, c0);
        s0 = fma(si14, T::R31_ODD_TWIDDLE_4, s0);
        let dc = c0 + a0;
        data[12] = dc;
        data[11] = -s0;
        let mut c0 = qc * T::R31_EVEN_TWIDDLE_3;
        let mut s0 = s_buffer[0];
        s0 *= T::R31_ODD_TWIDDLE_6;
        c0 = fma(ci1, T::R31_EVEN_TWIDDLE_10, c0);
        s0 = fma(si1, -T::R31_ODD_TWIDDLE_11, s0);
        c0 = fma(ci2, T::R31_EVEN_TWIDDLE_13, c0);
        s0 = fma(si2, -T::R31_ODD_TWIDDLE_1, s0);
        c0 = fma(ci3, T::R31_EVEN_TWIDDLE_6, c0);
        s0 = fma(si3, -T::R31_ODD_TWIDDLE_14, s0);
        c0 = fma(ci4, T::R31_EVEN_TWIDDLE_0, c0);
        s0 = fma(si4, T::R31_ODD_TWIDDLE_3, s0);
        c0 = fma(ci5, T::R31_EVEN_TWIDDLE_7, c0);
        s0 = fma(si5, T::R31_ODD_TWIDDLE_9, s0);
        c0 = fma(ci6, T::R31_EVEN_TWIDDLE_14, c0);
        s0 = fma(si6, -T::R31_ODD_TWIDDLE_8, s0);
        c0 = fma(ci7, T::R31_EVEN_TWIDDLE_9, c0);
        s0 = fma(si7, -T::R31_ODD_TWIDDLE_4, s0);
        c0 = fma(ci8, T::R31_EVEN_TWIDDLE_2, c0);
        s0 = fma(si8, T::R31_ODD_TWIDDLE_13, s0);
        c0 = fma(ci9, T::R31_EVEN_TWIDDLE_4, c0);
        s0 = fma(si9, T::R31_ODD_TWIDDLE_0, s0);
        c0 = fma(ci10, T::R31_EVEN_TWIDDLE_11, c0);
        s0 = fma(si10, T::R31_ODD_TWIDDLE_12, s0);
        c0 = fma(ci11, T::R31_EVEN_TWIDDLE_12, c0);
        s0 = fma(si11, -T::R31_ODD_TWIDDLE_5, s0);
        c0 = fma(ci12, T::R31_EVEN_TWIDDLE_5, c0);
        s0 = fma(si12, -T::R31_ODD_TWIDDLE_7, s0);
        c0 = fma(ci13, T::R31_EVEN_TWIDDLE_1, c0);
        s0 = fma(si13, T::R31_ODD_TWIDDLE_10, s0);
        c0 = fma(ci14, T::R31_EVEN_TWIDDLE_8, c0);
        s0 = fma(si14, T::R31_ODD_TWIDDLE_2, s0);
        let dc = c0 + a0;
        data[14] = -dc;
        data[13] = s0;
        let mut c0 = qc * T::R31_EVEN_TWIDDLE_11;
        let mut s0 = s_buffer[0];
        s0 *= -T::R31_ODD_TWIDDLE_7;
        c0 = fma(ci1, T::R31_EVEN_TWIDDLE_3, c0);
        s0 = fma(si1, T::R31_ODD_TWIDDLE_8, s0);
        c0 = fma(ci2, T::R31_EVEN_TWIDDLE_4, c0);
        s0 = fma(si2, T::R31_ODD_TWIDDLE_6, s0);
        c0 = fma(ci3, T::R31_EVEN_TWIDDLE_12, c0);
        s0 = fma(si3, -T::R31_ODD_TWIDDLE_9, s0);
        c0 = fma(ci4, T::R31_EVEN_TWIDDLE_10, c0);
        s0 = fma(si4, -T::R31_ODD_TWIDDLE_5, s0);
        c0 = fma(ci5, T::R31_EVEN_TWIDDLE_2, c0);
        s0 = fma(si5, T::R31_ODD_TWIDDLE_10, s0);
        c0 = fma(ci6, T::R31_EVEN_TWIDDLE_5, c0);
        s0 = fma(si6, T::R31_ODD_TWIDDLE_4, s0);
        c0 = fma(ci7, T::R31_EVEN_TWIDDLE_13, c0);
        s0 = fma(si7, -T::R31_ODD_TWIDDLE_11, s0);
        c0 = fma(ci8, T::R31_EVEN_TWIDDLE_9, c0);
        s0 = fma(si8, -T::R31_ODD_TWIDDLE_3, s0);
        c0 = fma(ci9, T::R31_EVEN_TWIDDLE_1, c0);
        s0 = fma(si9, T::R31_ODD_TWIDDLE_12, s0);
        c0 = fma(ci10, T::R31_EVEN_TWIDDLE_6, c0);
        s0 = fma(si10, T::R31_ODD_TWIDDLE_2, s0);
        c0 = fma(ci11, T::R31_EVEN_TWIDDLE_14, c0);
        s0 = fma(si11, -T::R31_ODD_TWIDDLE_13, s0);
        c0 = fma(ci12, T::R31_EVEN_TWIDDLE_8, c0);
        s0 = fma(si12, -T::R31_ODD_TWIDDLE_1, s0);
        c0 = fma(ci13, T::R31_EVEN_TWIDDLE_0, c0);
        s0 = fma(si13, T::R31_ODD_TWIDDLE_14, s0);
        c0 = fma(ci14, T::R31_EVEN_TWIDDLE_7, c0);
        s0 = fma(si14, T::R31_ODD_TWIDDLE_0, s0);
        let dc = c0 + a0;
        data[16] = dc;
        data[15] = -s0;
        let mut c0 = qc * T::R31_EVEN_TWIDDLE_4;
        let mut s0 = s_buffer[0];
        s0 *= T::R31_ODD_TWIDDLE_8;
        c0 = fma(ci1, T::R31_EVEN_TWIDDLE_13, c0);
        s0 = fma(si1, -T::R31_ODD_TWIDDLE_5, s0);
        c0 = fma(ci2, T::R31_EVEN_TWIDDLE_8, c0);
        s0 = fma(si2, -T::R31_ODD_TWIDDLE_11, s0);
        c0 = fma(ci3, T::R31_EVEN_TWIDDLE_0, c0);
        s0 = fma(si3, T::R31_ODD_TWIDDLE_2, s0);
        c0 = fma(ci4, T::R31_EVEN_TWIDDLE_9, c0);
        s0 = fma(si4, T::R31_ODD_TWIDDLE_14, s0);
        c0 = fma(ci5, T::R31_EVEN_TWIDDLE_12, c0);
        s0 = fma(si5, -T::R31_ODD_TWIDDLE_0, s0);
        c0 = fma(ci6, T::R31_EVEN_TWIDDLE_3, c0);
        s0 = fma(si6, T::R31_ODD_TWIDDLE_13, s0);
        c0 = fma(ci7, T::R31_EVEN_TWIDDLE_5, c0);
        s0 = fma(si7, T::R31_ODD_TWIDDLE_3, s0);
        c0 = fma(ci8, T::R31_EVEN_TWIDDLE_14, c0);
        s0 = fma(si8, -T::R31_ODD_TWIDDLE_10, s0);
        c0 = fma(ci9, T::R31_EVEN_TWIDDLE_7, c0);
        s0 = fma(si9, -T::R31_ODD_TWIDDLE_6, s0);
        c0 = fma(ci10, T::R31_EVEN_TWIDDLE_1, c0);
        s0 = fma(si10, T::R31_ODD_TWIDDLE_7, s0);
        c0 = fma(ci11, T::R31_EVEN_TWIDDLE_10, c0);
        s0 = fma(si11, T::R31_ODD_TWIDDLE_9, s0);
        c0 = fma(ci12, T::R31_EVEN_TWIDDLE_11, c0);
        s0 = fma(si12, -T::R31_ODD_TWIDDLE_4, s0);
        c0 = fma(ci13, T::R31_EVEN_TWIDDLE_2, c0);
        s0 = fma(si13, -T::R31_ODD_TWIDDLE_12, s0);
        c0 = fma(ci14, T::R31_EVEN_TWIDDLE_6, c0);
        s0 = fma(si14, T::R31_ODD_TWIDDLE_1, s0);
        let dc = c0 + a0;
        data[18] = -dc;
        data[17] = s0;
        let mut c0 = qc * T::R31_EVEN_TWIDDLE_10;
        let mut s0 = s_buffer[0];
        s0 *= -T::R31_ODD_TWIDDLE_9;
        c0 = fma(ci1, T::R31_EVEN_TWIDDLE_0, c0);
        s0 = fma(si1, T::R31_ODD_TWIDDLE_2, s0);
        c0 = fma(ci2, T::R31_EVEN_TWIDDLE_9, c0);
        s0 = fma(si2, -T::R31_ODD_TWIDDLE_14, s0);
        c0 = fma(ci3, T::R31_EVEN_TWIDDLE_11, c0);
        s0 = fma(si3, -T::R31_ODD_TWIDDLE_4, s0);
        c0 = fma(ci4, T::R31_EVEN_TWIDDLE_1, c0);
        s0 = fma(si4, T::R31_ODD_TWIDDLE_7, s0);
        c0 = fma(ci5, T::R31_EVEN_TWIDDLE_8, c0);
        s0 = fma(si5, T::R31_ODD_TWIDDLE_11, s0);
        c0 = fma(ci6, T::R31_EVEN_TWIDDLE_12, c0);
        s0 = fma(si6, -T::R31_ODD_TWIDDLE_0, s0);
        c0 = fma(ci7, T::R31_EVEN_TWIDDLE_2, c0);
        s0 = fma(si7, T::R31_ODD_TWIDDLE_12, s0);
        c0 = fma(ci8, T::R31_EVEN_TWIDDLE_7, c0);
        s0 = fma(si8, T::R31_ODD_TWIDDLE_6, s0);
        c0 = fma(ci9, T::R31_EVEN_TWIDDLE_13, c0);
        s0 = fma(si9, -T::R31_ODD_TWIDDLE_5, s0);
        c0 = fma(ci10, T::R31_EVEN_TWIDDLE_3, c0);
        s0 = fma(si10, -T::R31_ODD_TWIDDLE_13, s0);
        c0 = fma(ci11, T::R31_EVEN_TWIDDLE_6, c0);
        s0 = fma(si11, T::R31_ODD_TWIDDLE_1, s0);
        c0 = fma(ci12, T::R31_EVEN_TWIDDLE_14, c0);
        s0 = fma(si12, -T::R31_ODD_TWIDDLE_10, s0);
        c0 = fma(ci13, T::R31_EVEN_TWIDDLE_4, c0);
        s0 = fma(si13, -T::R31_ODD_TWIDDLE_8, s0);
        c0 = fma(ci14, T::R31_EVEN_TWIDDLE_5, c0);
        s0 = fma(si14, T::R31_ODD_TWIDDLE_3, s0);
        let dc = c0 + a0;
        data[20] = dc;
        data[19] = -s0;
        let mut c0 = qc * T::R31_EVEN_TWIDDLE_5;
        let mut s0 = s_buffer[0];
        s0 *= T::R31_ODD_TWIDDLE_10;
        c0 = fma(ci1, T::R31_EVEN_TWIDDLE_14, c0);
        s0 = fma(si1, -T::R31_ODD_TWIDDLE_0, s0);
        c0 = fma(ci2, T::R31_EVEN_TWIDDLE_3, c0);
        s0 = fma(si2, T::R31_ODD_TWIDDLE_9, s0);
        c0 = fma(ci3, T::R31_EVEN_TWIDDLE_7, c0);
        s0 = fma(si3, T::R31_ODD_TWIDDLE_11, s0);
        c0 = fma(ci4, T::R31_EVEN_TWIDDLE_12, c0);
        s0 = fma(si4, -T::R31_ODD_TWIDDLE_1, s0);
        c0 = fma(ci5, T::R31_EVEN_TWIDDLE_1, c0);
        s0 = fma(si5, T::R31_ODD_TWIDDLE_8, s0);
        c0 = fma(ci6, T::R31_EVEN_TWIDDLE_9, c0);
        s0 = fma(si6, T::R31_ODD_TWIDDLE_12, s0);
        c0 = fma(ci7, T::R31_EVEN_TWIDDLE_10, c0);
        s0 = fma(si7, -T::R31_ODD_TWIDDLE_2, s0);
        c0 = fma(ci8, T::R31_EVEN_TWIDDLE_0, c0);
        s0 = fma(si8, T::R31_ODD_TWIDDLE_7, s0);
        c0 = fma(ci9, T::R31_EVEN_TWIDDLE_11, c0);
        s0 = fma(si9, T::R31_ODD_TWIDDLE_13, s0);
        c0 = fma(ci10, T::R31_EVEN_TWIDDLE_8, c0);
        s0 = fma(si10, -T::R31_ODD_TWIDDLE_3, s0);
        c0 = fma(ci11, T::R31_EVEN_TWIDDLE_2, c0);
        s0 = fma(si11, T::R31_ODD_TWIDDLE_6, s0);
        c0 = fma(ci12, T::R31_EVEN_TWIDDLE_13, c0);
        s0 = fma(si12, T::R31_ODD_TWIDDLE_14, s0);
        c0 = fma(ci13, T::R31_EVEN_TWIDDLE_6, c0);
        s0 = fma(si13, -T::R31_ODD_TWIDDLE_4, s0);
        c0 = fma(ci14, T::R31_EVEN_TWIDDLE_4, c0);
        s0 = fma(si14, T::R31_ODD_TWIDDLE_5, s0);
        let dc = c0 + a0;
        data[22] = -dc;
        data[21] = s0;
        let mut c0 = qc * T::R31_EVEN_TWIDDLE_9;
        let mut s0 = s_buffer[0];
        s0 *= -T::R31_ODD_TWIDDLE_11;
        c0 = fma(ci1, T::R31_EVEN_TWIDDLE_2, c0);
        s0 = fma(si1, T::R31_ODD_TWIDDLE_3, s0);
        c0 = fma(ci2, T::R31_EVEN_TWIDDLE_14, c0);
        s0 = fma(si2, -T::R31_ODD_TWIDDLE_4, s0);
        c0 = fma(ci3, T::R31_EVEN_TWIDDLE_4, c0);
        s0 = fma(si3, T::R31_ODD_TWIDDLE_12, s0);
        c0 = fma(ci4, T::R31_EVEN_TWIDDLE_7, c0);
        s0 = fma(si4, T::R31_ODD_TWIDDLE_10, s0);
        c0 = fma(ci5, T::R31_EVEN_TWIDDLE_11, c0);
        s0 = fma(si5, -T::R31_ODD_TWIDDLE_2, s0);
        c0 = fma(ci6, T::R31_EVEN_TWIDDLE_0, c0);
        s0 = fma(si6, T::R31_ODD_TWIDDLE_5, s0);
        c0 = fma(ci7, T::R31_EVEN_TWIDDLE_12, c0);
        s0 = fma(si7, -T::R31_ODD_TWIDDLE_13, s0);
        c0 = fma(ci8, T::R31_EVEN_TWIDDLE_6, c0);
        s0 = fma(si8, -T::R31_ODD_TWIDDLE_9, s0);
        c0 = fma(ci9, T::R31_EVEN_TWIDDLE_5, c0);
        s0 = fma(si9, T::R31_ODD_TWIDDLE_1, s0);
        c0 = fma(ci10, T::R31_EVEN_TWIDDLE_13, c0);
        s0 = fma(si10, -T::R31_ODD_TWIDDLE_6, s0);
        c0 = fma(ci11, T::R31_EVEN_TWIDDLE_1, c0);
        s0 = fma(si11, T::R31_ODD_TWIDDLE_14, s0);
        c0 = fma(ci12, T::R31_EVEN_TWIDDLE_10, c0);
        s0 = fma(si12, T::R31_ODD_TWIDDLE_8, s0);
        c0 = fma(ci13, T::R31_EVEN_TWIDDLE_8, c0);
        s0 = fma(si13, -T::R31_ODD_TWIDDLE_0, s0);
        c0 = fma(ci14, T::R31_EVEN_TWIDDLE_3, c0);
        s0 = fma(si14, T::R31_ODD_TWIDDLE_7, s0);
        let dc = c0 + a0;
        data[24] = dc;
        data[23] = -s0;
        let mut c0 = qc * T::R31_EVEN_TWIDDLE_6;
        let mut s0 = s_buffer[0];
        s0 *= T::R31_ODD_TWIDDLE_12;
        c0 = fma(ci1, T::R31_EVEN_TWIDDLE_11, c0);
        s0 = fma(si1, -T::R31_ODD_TWIDDLE_6, s0);
        c0 = fma(ci2, T::R31_EVEN_TWIDDLE_1, c0);
        s0 = fma(si2, T::R31_ODD_TWIDDLE_0, s0);
        c0 = fma(ci3, T::R31_EVEN_TWIDDLE_14, c0);
        s0 = fma(si3, -T::R31_ODD_TWIDDLE_5, s0);
        c0 = fma(ci4, T::R31_EVEN_TWIDDLE_3, c0);
        s0 = fma(si4, T::R31_ODD_TWIDDLE_11, s0);
        c0 = fma(ci5, T::R31_EVEN_TWIDDLE_9, c0);
        s0 = fma(si5, T::R31_ODD_TWIDDLE_13, s0);
        c0 = fma(ci6, T::R31_EVEN_TWIDDLE_8, c0);
        s0 = fma(si6, -T::R31_ODD_TWIDDLE_7, s0);
        c0 = fma(ci7, T::R31_EVEN_TWIDDLE_4, c0);
        s0 = fma(si7, T::R31_ODD_TWIDDLE_1, s0);
        c0 = fma(ci8, T::R31_EVEN_TWIDDLE_13, c0);
        s0 = fma(si8, -T::R31_ODD_TWIDDLE_4, s0);
        c0 = fma(ci9, T::R31_EVEN_TWIDDLE_0, c0);
        s0 = fma(si9, T::R31_ODD_TWIDDLE_10, s0);
        c0 = fma(ci10, T::R31_EVEN_TWIDDLE_12, c0);
        s0 = fma(si10, T::R31_ODD_TWIDDLE_14, s0);
        c0 = fma(ci11, T::R31_EVEN_TWIDDLE_5, c0);
        s0 = fma(si11, -T::R31_ODD_TWIDDLE_8, s0);
        c0 = fma(ci12, T::R31_EVEN_TWIDDLE_7, c0);
        s0 = fma(si12, T::R31_ODD_TWIDDLE_2, s0);
        c0 = fma(ci13, T::R31_EVEN_TWIDDLE_10, c0);
        s0 = fma(si13, -T::R31_ODD_TWIDDLE_3, s0);
        c0 = fma(ci14, T::R31_EVEN_TWIDDLE_2, c0);
        s0 = fma(si14, T::R31_ODD_TWIDDLE_9, s0);
        let dc = c0 + a0;
        data[26] = -dc;
        data[25] = s0;
        let mut c0 = qc * T::R31_EVEN_TWIDDLE_8;
        let mut s0 = s_buffer[0];
        s0 *= -T::R31_ODD_TWIDDLE_13;
        c0 = fma(ci1, T::R31_EVEN_TWIDDLE_5, c0);
        s0 = fma(si1, T::R31_ODD_TWIDDLE_9, s0);
        c0 = fma(ci2, T::R31_EVEN_TWIDDLE_11, c0);
        s0 = fma(si2, -T::R31_ODD_TWIDDLE_5, s0);
        c0 = fma(ci3, T::R31_EVEN_TWIDDLE_2, c0);
        s0 = fma(si3, T::R31_ODD_TWIDDLE_1, s0);
        c0 = fma(ci4, T::R31_EVEN_TWIDDLE_14, c0);
        s0 = fma(si4, -T::R31_ODD_TWIDDLE_2, s0);
        c0 = fma(ci5, T::R31_EVEN_TWIDDLE_0, c0);
        s0 = fma(si5, T::R31_ODD_TWIDDLE_6, s0);
        c0 = fma(ci6, T::R31_EVEN_TWIDDLE_13, c0);
        s0 = fma(si6, -T::R31_ODD_TWIDDLE_10, s0);
        c0 = fma(ci7, T::R31_EVEN_TWIDDLE_3, c0);
        s0 = fma(si7, T::R31_ODD_TWIDDLE_14, s0);
        c0 = fma(ci8, T::R31_EVEN_TWIDDLE_10, c0);
        s0 = fma(si8, T::R31_ODD_TWIDDLE_12, s0);
        c0 = fma(ci9, T::R31_EVEN_TWIDDLE_6, c0);
        s0 = fma(si9, -T::R31_ODD_TWIDDLE_8, s0);
        c0 = fma(ci10, T::R31_EVEN_TWIDDLE_7, c0);
        s0 = fma(si10, T::R31_ODD_TWIDDLE_4, s0);
        c0 = fma(ci11, T::R31_EVEN_TWIDDLE_9, c0);
        s0 = fma(si11, -T::R31_ODD_TWIDDLE_0, s0);
        c0 = fma(ci12, T::R31_EVEN_TWIDDLE_4, c0);
        s0 = fma(si12, T::R31_ODD_TWIDDLE_3, s0);
        c0 = fma(ci13, T::R31_EVEN_TWIDDLE_12, c0);
        s0 = fma(si13, -T::R31_ODD_TWIDDLE_7, s0);
        c0 = fma(ci14, T::R31_EVEN_TWIDDLE_1, c0);
        s0 = fma(si14, T::R31_ODD_TWIDDLE_11, s0);
        let dc = c0 + a0;
        data[28] = dc;
        data[27] = -s0;
        let mut c0 = qc * T::R31_EVEN_TWIDDLE_7;
        let mut s0 = s_buffer[0];
        s0 *= T::R31_ODD_TWIDDLE_14;
        c0 = fma(ci1, T::R31_EVEN_TWIDDLE_8, c0);
        s0 = fma(si1, -T::R31_ODD_TWIDDLE_12, s0);
        c0 = fma(ci2, T::R31_EVEN_TWIDDLE_6, c0);
        s0 = fma(si2, T::R31_ODD_TWIDDLE_10, s0);
        c0 = fma(ci3, T::R31_EVEN_TWIDDLE_9, c0);
        s0 = fma(si3, -T::R31_ODD_TWIDDLE_8, s0);
        c0 = fma(ci4, T::R31_EVEN_TWIDDLE_5, c0);
        s0 = fma(si4, T::R31_ODD_TWIDDLE_6, s0);
        c0 = fma(ci5, T::R31_EVEN_TWIDDLE_10, c0);
        s0 = fma(si5, -T::R31_ODD_TWIDDLE_4, s0);
        c0 = fma(ci6, T::R31_EVEN_TWIDDLE_4, c0);
        s0 = fma(si6, T::R31_ODD_TWIDDLE_2, s0);
        c0 = fma(ci7, T::R31_EVEN_TWIDDLE_11, c0);
        s0 = fma(si7, -T::R31_ODD_TWIDDLE_0, s0);
        c0 = fma(ci8, T::R31_EVEN_TWIDDLE_3, c0);
        s0 = fma(si8, T::R31_ODD_TWIDDLE_1, s0);
        c0 = fma(ci9, T::R31_EVEN_TWIDDLE_12, c0);
        s0 = fma(si9, -T::R31_ODD_TWIDDLE_3, s0);
        c0 = fma(ci10, T::R31_EVEN_TWIDDLE_2, c0);
        s0 = fma(si10, T::R31_ODD_TWIDDLE_5, s0);
        c0 = fma(ci11, T::R31_EVEN_TWIDDLE_13, c0);
        s0 = fma(si11, -T::R31_ODD_TWIDDLE_7, s0);
        c0 = fma(ci12, T::R31_EVEN_TWIDDLE_1, c0);
        s0 = fma(si12, T::R31_ODD_TWIDDLE_9, s0);
        c0 = fma(ci13, T::R31_EVEN_TWIDDLE_14, c0);
        s0 = fma(si13, -T::R31_ODD_TWIDDLE_11, s0);
        c0 = fma(ci14, T::R31_EVEN_TWIDDLE_0, c0);
        s0 = fma(si14, T::R31_ODD_TWIDDLE_13, s0);
        let dc = c0 + a0;
        data[30] = -dc;
        data[29] = s0;
    }
}

impl<T: DctSample + MixedRadix31Sample> AvxDct2Butterfly31<T>
where
    f64: AsPrimitive<T>,
{
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_impl(&self, data: &mut [T]) -> Result<(), PxdctError> {
        if !data.len().is_multiple_of(31) {
            return Err(PxdctError::InvalidSizeMultiplier(data.len(), self.length()));
        }
        for chunk in data.chunks_exact_mut(31) {
            self.exec(&mut InPlaceStore::new(chunk));
        }
        Ok(())
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_into_impl(&self, input: &[T], output: &mut [T]) -> Result<(), PxdctError> {
        use crate::util::validate_oof_sizes;
        validate_oof_sizes!(input, output, 31);
        use crate::bidirectional::BiStore;
        for (src, dst) in input.chunks_exact(31).zip(output.chunks_exact_mut(31)) {
            self.exec(&mut BiStore::new(src, dst));
        }
        Ok(())
    }
}

impl<T: DctSample + MixedRadix31Sample> PxdctExecutor<T> for AvxDct2Butterfly31<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, data: &mut [T]) -> Result<(), PxdctError> {
        unsafe { self.execute_impl(data) }
    }

    fn execute_with_scratch(&self, data: &mut [T], _: &mut [T]) -> Result<(), PxdctError> {
        unsafe { self.execute_impl(data) }
    }

    fn execute_into(&self, input: &[T], output: &mut [T]) -> Result<(), PxdctError> {
        unsafe { self.execute_into_impl(input, output) }
    }

    fn execute_into_with_scratch(
        &self,
        input: &[T],
        output: &mut [T],
        _: &mut [T],
    ) -> Result<(), PxdctError> {
        unsafe { self.execute_into_impl(input, output) }
    }

    fn length(&self) -> usize {
        31
    }

    fn scratch_size(&self) -> usize {
        0
    }
}

#[derive(Debug, Clone)]
pub(crate) struct AvxDct2Butterfly37<T: DctSample> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T: DctSample> Default for AvxDct2Butterfly37<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: DctSample + MixedRadix37Sample> AvxDct2Butterfly37<T>
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
        s0 = fma(si1, T::R37_ODD_TWIDDLE_1, s0);
        let ci2 = c_buffer[2];
        let si2 = s_buffer[2];
        c0 = ci2 + c0;
        c1 = ci2 * T::R37_EVEN_TWIDDLE_2 + c1;
        s0 = fma(si2, T::R37_ODD_TWIDDLE_2, s0);
        let ci3 = c_buffer[3];
        let si3 = s_buffer[3];
        c0 = ci3 + c0;
        c1 = ci3 * T::R37_EVEN_TWIDDLE_3 + c1;
        s0 = fma(si3, T::R37_ODD_TWIDDLE_3, s0);
        let ci4 = c_buffer[4];
        let si4 = s_buffer[4];
        c0 = ci4 + c0;
        c1 = ci4 * T::R37_EVEN_TWIDDLE_4 + c1;
        s0 = fma(si4, T::R37_ODD_TWIDDLE_4, s0);
        let ci5 = c_buffer[5];
        let si5 = s_buffer[5];
        c0 = ci5 + c0;
        c1 = ci5 * T::R37_EVEN_TWIDDLE_5 + c1;
        s0 = fma(si5, T::R37_ODD_TWIDDLE_5, s0);
        let ci6 = c_buffer[6];
        let si6 = s_buffer[6];
        c0 = ci6 + c0;
        c1 = ci6 * T::R37_EVEN_TWIDDLE_6 + c1;
        s0 = fma(si6, T::R37_ODD_TWIDDLE_6, s0);
        let ci7 = c_buffer[7];
        let si7 = s_buffer[7];
        c0 = ci7 + c0;
        c1 = ci7 * T::R37_EVEN_TWIDDLE_7 + c1;
        s0 = fma(si7, T::R37_ODD_TWIDDLE_7, s0);
        let ci8 = c_buffer[8];
        let si8 = s_buffer[8];
        c0 = ci8 + c0;
        c1 = ci8 * T::R37_EVEN_TWIDDLE_8 + c1;
        s0 = fma(si8, T::R37_ODD_TWIDDLE_8, s0);
        let ci9 = c_buffer[9];
        let si9 = s_buffer[9];
        c0 = ci9 + c0;
        c1 = ci9 * T::R37_EVEN_TWIDDLE_9 + c1;
        s0 = fma(si9, T::R37_ODD_TWIDDLE_9, s0);
        let ci10 = c_buffer[10];
        let si10 = s_buffer[10];
        c0 = ci10 + c0;
        c1 = ci10 * T::R37_EVEN_TWIDDLE_10 + c1;
        s0 = fma(si10, T::R37_ODD_TWIDDLE_10, s0);
        let ci11 = c_buffer[11];
        let si11 = s_buffer[11];
        c0 = ci11 + c0;
        c1 = ci11 * T::R37_EVEN_TWIDDLE_11 + c1;
        s0 = fma(si11, T::R37_ODD_TWIDDLE_11, s0);
        let ci12 = c_buffer[12];
        let si12 = s_buffer[12];
        c0 = ci12 + c0;
        c1 = ci12 * T::R37_EVEN_TWIDDLE_12 + c1;
        s0 = fma(si12, T::R37_ODD_TWIDDLE_12, s0);
        let ci13 = c_buffer[13];
        let si13 = s_buffer[13];
        c0 = ci13 + c0;
        c1 = ci13 * T::R37_EVEN_TWIDDLE_13 + c1;
        s0 = fma(si13, T::R37_ODD_TWIDDLE_13, s0);
        let ci14 = c_buffer[14];
        let si14 = s_buffer[14];
        c0 = ci14 + c0;
        c1 = ci14 * T::R37_EVEN_TWIDDLE_14 + c1;
        s0 = fma(si14, T::R37_ODD_TWIDDLE_14, s0);
        let ci15 = c_buffer[15];
        let si15 = s_buffer[15];
        c0 = ci15 + c0;
        c1 = ci15 * T::R37_EVEN_TWIDDLE_15 + c1;
        s0 = fma(si15, T::R37_ODD_TWIDDLE_15, s0);
        let ci16 = c_buffer[16];
        let si16 = s_buffer[16];
        c0 = ci16 + c0;
        c1 = ci16 * T::R37_EVEN_TWIDDLE_16 + c1;
        s0 = fma(si16, T::R37_ODD_TWIDDLE_16, s0);
        let ci17 = c_buffer[17];
        let si17 = s_buffer[17];
        c0 = ci17 + c0;
        c1 = ci17 * T::R37_EVEN_TWIDDLE_17 + c1;
        s0 = fma(si17, T::R37_ODD_TWIDDLE_17, s0);
        let a0 = a_buffer[0];
        let dc = c0 + a0;
        data[0] = dc;
        data[1] = s0;
        data[2] = -(c1 + a0);
        let mut c0 = qc * T::R37_EVEN_TWIDDLE_17;
        let mut s0 = s_buffer[0];
        s0 *= -T::R37_ODD_TWIDDLE_1;
        c0 = fma(ci1, T::R37_EVEN_TWIDDLE_15, c0);
        s0 = fma(si1, -T::R37_ODD_TWIDDLE_4, s0);
        c0 = fma(ci2, T::R37_EVEN_TWIDDLE_13, c0);
        s0 = fma(si2, -T::R37_ODD_TWIDDLE_7, s0);
        c0 = fma(ci3, T::R37_EVEN_TWIDDLE_11, c0);
        s0 = fma(si3, -T::R37_ODD_TWIDDLE_10, s0);
        c0 = fma(ci4, T::R37_EVEN_TWIDDLE_9, c0);
        s0 = fma(si4, -T::R37_ODD_TWIDDLE_13, s0);
        c0 = fma(ci5, T::R37_EVEN_TWIDDLE_7, c0);
        s0 = fma(si5, -T::R37_ODD_TWIDDLE_16, s0);
        c0 = fma(ci6, T::R37_EVEN_TWIDDLE_5, c0);
        s0 = fma(si6, T::R37_ODD_TWIDDLE_17, s0);
        c0 = fma(ci7, T::R37_EVEN_TWIDDLE_3, c0);
        s0 = fma(si7, T::R37_ODD_TWIDDLE_14, s0);
        c0 = fma(ci8, T::R37_EVEN_TWIDDLE_1, c0);
        s0 = fma(si8, T::R37_ODD_TWIDDLE_11, s0);
        c0 = fma(ci9, T::R37_EVEN_TWIDDLE_0, c0);
        s0 = fma(si9, T::R37_ODD_TWIDDLE_8, s0);
        c0 = fma(ci10, T::R37_EVEN_TWIDDLE_2, c0);
        s0 = fma(si10, T::R37_ODD_TWIDDLE_5, s0);
        c0 = fma(ci11, T::R37_EVEN_TWIDDLE_4, c0);
        s0 = fma(si11, T::R37_ODD_TWIDDLE_2, s0);
        c0 = fma(ci12, T::R37_EVEN_TWIDDLE_6, c0);
        s0 = fma(si12, T::R37_ODD_TWIDDLE_0, s0);
        c0 = fma(ci13, T::R37_EVEN_TWIDDLE_8, c0);
        s0 = fma(si13, T::R37_ODD_TWIDDLE_3, s0);
        c0 = fma(ci14, T::R37_EVEN_TWIDDLE_10, c0);
        s0 = fma(si14, T::R37_ODD_TWIDDLE_6, s0);
        c0 = fma(ci15, T::R37_EVEN_TWIDDLE_12, c0);
        s0 = fma(si15, T::R37_ODD_TWIDDLE_9, s0);
        c0 = fma(ci16, T::R37_EVEN_TWIDDLE_14, c0);
        s0 = fma(si16, T::R37_ODD_TWIDDLE_12, s0);
        c0 = fma(ci17, T::R37_EVEN_TWIDDLE_16, c0);
        s0 = fma(si17, T::R37_ODD_TWIDDLE_15, s0);
        let dc = c0 + a0;
        data[4] = dc;
        data[3] = -s0;
        let mut c0 = qc * T::R37_EVEN_TWIDDLE_1;
        let mut s0 = s_buffer[0];
        s0 *= T::R37_ODD_TWIDDLE_2;
        c0 = fma(ci1, T::R37_EVEN_TWIDDLE_4, c0);
        s0 = fma(si1, T::R37_ODD_TWIDDLE_7, s0);
        c0 = fma(ci2, T::R37_EVEN_TWIDDLE_7, c0);
        s0 = fma(si2, T::R37_ODD_TWIDDLE_12, s0);
        c0 = fma(ci3, T::R37_EVEN_TWIDDLE_10, c0);
        s0 = fma(si3, T::R37_ODD_TWIDDLE_17, s0);
        c0 = fma(ci4, T::R37_EVEN_TWIDDLE_13, c0);
        s0 = fma(si4, -T::R37_ODD_TWIDDLE_14, s0);
        c0 = fma(ci5, T::R37_EVEN_TWIDDLE_16, c0);
        s0 = fma(si5, -T::R37_ODD_TWIDDLE_9, s0);
        c0 = fma(ci6, T::R37_EVEN_TWIDDLE_17, c0);
        s0 = fma(si6, -T::R37_ODD_TWIDDLE_4, s0);
        c0 = fma(ci7, T::R37_EVEN_TWIDDLE_14, c0);
        s0 = fma(si7, -T::R37_ODD_TWIDDLE_0, s0);
        c0 = fma(ci8, T::R37_EVEN_TWIDDLE_11, c0);
        s0 = fma(si8, -T::R37_ODD_TWIDDLE_5, s0);
        c0 = fma(ci9, T::R37_EVEN_TWIDDLE_8, c0);
        s0 = fma(si9, -T::R37_ODD_TWIDDLE_10, s0);
        c0 = fma(ci10, T::R37_EVEN_TWIDDLE_5, c0);
        s0 = fma(si10, -T::R37_ODD_TWIDDLE_15, s0);
        c0 = fma(ci11, T::R37_EVEN_TWIDDLE_2, c0);
        s0 = fma(si11, T::R37_ODD_TWIDDLE_16, s0);
        c0 = fma(ci12, T::R37_EVEN_TWIDDLE_0, c0);
        s0 = fma(si12, T::R37_ODD_TWIDDLE_11, s0);
        c0 = fma(ci13, T::R37_EVEN_TWIDDLE_3, c0);
        s0 = fma(si13, T::R37_ODD_TWIDDLE_6, s0);
        c0 = fma(ci14, T::R37_EVEN_TWIDDLE_6, c0);
        s0 = fma(si14, T::R37_ODD_TWIDDLE_1, s0);
        c0 = fma(ci15, T::R37_EVEN_TWIDDLE_9, c0);
        s0 = fma(si15, T::R37_ODD_TWIDDLE_3, s0);
        c0 = fma(ci16, T::R37_EVEN_TWIDDLE_12, c0);
        s0 = fma(si16, T::R37_ODD_TWIDDLE_8, s0);
        c0 = fma(ci17, T::R37_EVEN_TWIDDLE_15, c0);
        s0 = fma(si17, T::R37_ODD_TWIDDLE_13, s0);
        let dc = c0 + a0;
        data[6] = -dc;
        data[5] = s0;
        let mut c0 = qc * T::R37_EVEN_TWIDDLE_16;
        let mut s0 = s_buffer[0];
        s0 *= -T::R37_ODD_TWIDDLE_3;
        c0 = fma(ci1, T::R37_EVEN_TWIDDLE_12, c0);
        s0 = fma(si1, -T::R37_ODD_TWIDDLE_10, s0);
        c0 = fma(ci2, T::R37_EVEN_TWIDDLE_8, c0);
        s0 = fma(si2, -T::R37_ODD_TWIDDLE_17, s0);
        c0 = fma(ci3, T::R37_EVEN_TWIDDLE_4, c0);
        s0 = fma(si3, T::R37_ODD_TWIDDLE_12, s0);
        c0 = fma(ci4, T::R37_EVEN_TWIDDLE_0, c0);
        s0 = fma(si4, T::R37_ODD_TWIDDLE_5, s0);
        c0 = fma(ci5, T::R37_EVEN_TWIDDLE_3, c0);
        s0 = fma(si5, T::R37_ODD_TWIDDLE_1, s0);
        c0 = fma(ci6, T::R37_EVEN_TWIDDLE_7, c0);
        s0 = fma(si6, T::R37_ODD_TWIDDLE_8, s0);
        c0 = fma(ci7, T::R37_EVEN_TWIDDLE_11, c0);
        s0 = fma(si7, T::R37_ODD_TWIDDLE_15, s0);
        c0 = fma(ci8, T::R37_EVEN_TWIDDLE_15, c0);
        s0 = fma(si8, -T::R37_ODD_TWIDDLE_14, s0);
        c0 = fma(ci9, T::R37_EVEN_TWIDDLE_17, c0);
        s0 = fma(si9, -T::R37_ODD_TWIDDLE_7, s0);
        c0 = fma(ci10, T::R37_EVEN_TWIDDLE_13, c0);
        s0 = fma(si10, -T::R37_ODD_TWIDDLE_0, s0);
        c0 = fma(ci11, T::R37_EVEN_TWIDDLE_9, c0);
        s0 = fma(si11, -T::R37_ODD_TWIDDLE_6, s0);
        c0 = fma(ci12, T::R37_EVEN_TWIDDLE_5, c0);
        s0 = fma(si12, -T::R37_ODD_TWIDDLE_13, s0);
        c0 = fma(ci13, T::R37_EVEN_TWIDDLE_1, c0);
        s0 = fma(si13, T::R37_ODD_TWIDDLE_16, s0);
        c0 = fma(ci14, T::R37_EVEN_TWIDDLE_2, c0);
        s0 = fma(si14, T::R37_ODD_TWIDDLE_9, s0);
        c0 = fma(ci15, T::R37_EVEN_TWIDDLE_6, c0);
        s0 = fma(si15, T::R37_ODD_TWIDDLE_2, s0);
        c0 = fma(ci16, T::R37_EVEN_TWIDDLE_10, c0);
        s0 = fma(si16, T::R37_ODD_TWIDDLE_4, s0);
        c0 = fma(ci17, T::R37_EVEN_TWIDDLE_14, c0);
        s0 = fma(si17, T::R37_ODD_TWIDDLE_11, s0);
        let dc = c0 + a0;
        data[8] = dc;
        data[7] = -s0;
        let mut c0 = qc * T::R37_EVEN_TWIDDLE_2;
        let mut s0 = s_buffer[0];
        s0 *= T::R37_ODD_TWIDDLE_4;
        c0 = fma(ci1, T::R37_EVEN_TWIDDLE_7, c0);
        s0 = fma(si1, T::R37_ODD_TWIDDLE_13, s0);
        c0 = fma(ci2, T::R37_EVEN_TWIDDLE_12, c0);
        s0 = fma(si2, -T::R37_ODD_TWIDDLE_14, s0);
        c0 = fma(ci3, T::R37_EVEN_TWIDDLE_17, c0);
        s0 = fma(si3, -T::R37_ODD_TWIDDLE_5, s0);
        c0 = fma(ci4, T::R37_EVEN_TWIDDLE_14, c0);
        s0 = fma(si4, -T::R37_ODD_TWIDDLE_3, s0);
        c0 = fma(ci5, T::R37_EVEN_TWIDDLE_9, c0);
        s0 = fma(si5, -T::R37_ODD_TWIDDLE_12, s0);
        c0 = fma(ci6, T::R37_EVEN_TWIDDLE_4, c0);
        s0 = fma(si6, T::R37_ODD_TWIDDLE_15, s0);
        c0 = fma(ci7, T::R37_EVEN_TWIDDLE_0, c0);
        s0 = fma(si7, T::R37_ODD_TWIDDLE_6, s0);
        c0 = fma(ci8, T::R37_EVEN_TWIDDLE_5, c0);
        s0 = fma(si8, T::R37_ODD_TWIDDLE_2, s0);
        c0 = fma(ci9, T::R37_EVEN_TWIDDLE_10, c0);
        s0 = fma(si9, T::R37_ODD_TWIDDLE_11, s0);
        c0 = fma(ci10, T::R37_EVEN_TWIDDLE_15, c0);
        s0 = fma(si10, -T::R37_ODD_TWIDDLE_16, s0);
        c0 = fma(ci11, T::R37_EVEN_TWIDDLE_16, c0);
        s0 = fma(si11, -T::R37_ODD_TWIDDLE_7, s0);
        c0 = fma(ci12, T::R37_EVEN_TWIDDLE_11, c0);
        s0 = fma(si12, -T::R37_ODD_TWIDDLE_1, s0);
        c0 = fma(ci13, T::R37_EVEN_TWIDDLE_6, c0);
        s0 = fma(si13, -T::R37_ODD_TWIDDLE_10, s0);
        c0 = fma(ci14, T::R37_EVEN_TWIDDLE_1, c0);
        s0 = fma(si14, T::R37_ODD_TWIDDLE_17, s0);
        c0 = fma(ci15, T::R37_EVEN_TWIDDLE_3, c0);
        s0 = fma(si15, T::R37_ODD_TWIDDLE_8, s0);
        c0 = fma(ci16, T::R37_EVEN_TWIDDLE_8, c0);
        s0 = fma(si16, T::R37_ODD_TWIDDLE_0, s0);
        c0 = fma(ci17, T::R37_EVEN_TWIDDLE_13, c0);
        s0 = fma(si17, T::R37_ODD_TWIDDLE_9, s0);
        let dc = c0 + a0;
        data[10] = -dc;
        data[9] = s0;
        let mut c0 = qc * T::R37_EVEN_TWIDDLE_15;
        let mut s0 = s_buffer[0];
        s0 *= -T::R37_ODD_TWIDDLE_5;
        c0 = fma(ci1, T::R37_EVEN_TWIDDLE_9, c0);
        s0 = fma(si1, -T::R37_ODD_TWIDDLE_16, s0);
        c0 = fma(ci2, T::R37_EVEN_TWIDDLE_3, c0);
        s0 = fma(si2, T::R37_ODD_TWIDDLE_9, s0);
        c0 = fma(ci3, T::R37_EVEN_TWIDDLE_2, c0);
        s0 = fma(si3, T::R37_ODD_TWIDDLE_1, s0);
        c0 = fma(ci4, T::R37_EVEN_TWIDDLE_8, c0);
        s0 = fma(si4, T::R37_ODD_TWIDDLE_12, s0);
        c0 = fma(ci5, T::R37_EVEN_TWIDDLE_14, c0);
        s0 = fma(si5, -T::R37_ODD_TWIDDLE_13, s0);
        c0 = fma(ci6, T::R37_EVEN_TWIDDLE_16, c0);
        s0 = fma(si6, -T::R37_ODD_TWIDDLE_2, s0);
        c0 = fma(ci7, T::R37_EVEN_TWIDDLE_10, c0);
        s0 = fma(si7, -T::R37_ODD_TWIDDLE_8, s0);
        c0 = fma(ci8, T::R37_EVEN_TWIDDLE_4, c0);
        s0 = fma(si8, T::R37_ODD_TWIDDLE_17, s0);
        c0 = fma(ci9, T::R37_EVEN_TWIDDLE_1, c0);
        s0 = fma(si9, T::R37_ODD_TWIDDLE_6, s0);
        c0 = fma(ci10, T::R37_EVEN_TWIDDLE_7, c0);
        s0 = fma(si10, T::R37_ODD_TWIDDLE_4, s0);
        c0 = fma(ci11, T::R37_EVEN_TWIDDLE_13, c0);
        s0 = fma(si11, T::R37_ODD_TWIDDLE_15, s0);
        c0 = fma(ci12, T::R37_EVEN_TWIDDLE_17, c0);
        s0 = fma(si12, -T::R37_ODD_TWIDDLE_10, s0);
        c0 = fma(ci13, T::R37_EVEN_TWIDDLE_11, c0);
        s0 = fma(si13, -T::R37_ODD_TWIDDLE_0, s0);
        c0 = fma(ci14, T::R37_EVEN_TWIDDLE_5, c0);
        s0 = fma(si14, -T::R37_ODD_TWIDDLE_11, s0);
        c0 = fma(ci15, T::R37_EVEN_TWIDDLE_0, c0);
        s0 = fma(si15, T::R37_ODD_TWIDDLE_14, s0);
        c0 = fma(ci16, T::R37_EVEN_TWIDDLE_6, c0);
        s0 = fma(si16, T::R37_ODD_TWIDDLE_3, s0);
        c0 = fma(ci17, T::R37_EVEN_TWIDDLE_12, c0);
        s0 = fma(si17, T::R37_ODD_TWIDDLE_7, s0);
        let dc = c0 + a0;
        data[12] = dc;
        data[11] = -s0;
        let mut c0 = qc * T::R37_EVEN_TWIDDLE_3;
        let mut s0 = s_buffer[0];
        s0 *= T::R37_ODD_TWIDDLE_6;
        c0 = fma(ci1, T::R37_EVEN_TWIDDLE_10, c0);
        s0 = fma(si1, -T::R37_ODD_TWIDDLE_17, s0);
        c0 = fma(ci2, T::R37_EVEN_TWIDDLE_17, c0);
        s0 = fma(si2, -T::R37_ODD_TWIDDLE_4, s0);
        c0 = fma(ci3, T::R37_EVEN_TWIDDLE_12, c0);
        s0 = fma(si3, -T::R37_ODD_TWIDDLE_8, s0);
        c0 = fma(ci4, T::R37_EVEN_TWIDDLE_5, c0);
        s0 = fma(si4, T::R37_ODD_TWIDDLE_15, s0);
        c0 = fma(ci5, T::R37_EVEN_TWIDDLE_1, c0);
        s0 = fma(si5, T::R37_ODD_TWIDDLE_2, s0);
        c0 = fma(ci6, T::R37_EVEN_TWIDDLE_8, c0);
        s0 = fma(si6, T::R37_ODD_TWIDDLE_10, s0);
        c0 = fma(ci7, T::R37_EVEN_TWIDDLE_15, c0);
        s0 = fma(si7, -T::R37_ODD_TWIDDLE_13, s0);
        c0 = fma(ci8, T::R37_EVEN_TWIDDLE_14, c0);
        s0 = fma(si8, -T::R37_ODD_TWIDDLE_0, s0);
        c0 = fma(ci9, T::R37_EVEN_TWIDDLE_7, c0);
        s0 = fma(si9, -T::R37_ODD_TWIDDLE_12, s0);
        c0 = fma(ci10, T::R37_EVEN_TWIDDLE_0, c0);
        s0 = fma(si10, T::R37_ODD_TWIDDLE_11, s0);
        c0 = fma(ci11, T::R37_EVEN_TWIDDLE_6, c0);
        s0 = fma(si11, T::R37_ODD_TWIDDLE_1, s0);
        c0 = fma(ci12, T::R37_EVEN_TWIDDLE_13, c0);
        s0 = fma(si12, T::R37_ODD_TWIDDLE_14, s0);
        c0 = fma(ci13, T::R37_EVEN_TWIDDLE_16, c0);
        s0 = fma(si13, -T::R37_ODD_TWIDDLE_9, s0);
        c0 = fma(ci14, T::R37_EVEN_TWIDDLE_9, c0);
        s0 = fma(si14, -T::R37_ODD_TWIDDLE_3, s0);
        c0 = fma(ci15, T::R37_EVEN_TWIDDLE_2, c0);
        s0 = fma(si15, -T::R37_ODD_TWIDDLE_16, s0);
        c0 = fma(ci16, T::R37_EVEN_TWIDDLE_4, c0);
        s0 = fma(si16, T::R37_ODD_TWIDDLE_7, s0);
        c0 = fma(ci17, T::R37_EVEN_TWIDDLE_11, c0);
        s0 = fma(si17, T::R37_ODD_TWIDDLE_5, s0);
        let dc = c0 + a0;
        data[14] = -dc;
        data[13] = s0;
        let mut c0 = qc * T::R37_EVEN_TWIDDLE_14;
        let mut s0 = s_buffer[0];
        s0 *= -T::R37_ODD_TWIDDLE_7;
        c0 = fma(ci1, T::R37_EVEN_TWIDDLE_6, c0);
        s0 = fma(si1, T::R37_ODD_TWIDDLE_14, s0);
        c0 = fma(ci2, T::R37_EVEN_TWIDDLE_1, c0);
        s0 = fma(si2, T::R37_ODD_TWIDDLE_0, s0);
        c0 = fma(ci3, T::R37_EVEN_TWIDDLE_9, c0);
        s0 = fma(si3, T::R37_ODD_TWIDDLE_15, s0);
        c0 = fma(ci4, T::R37_EVEN_TWIDDLE_17, c0);
        s0 = fma(si4, -T::R37_ODD_TWIDDLE_6, s0);
        c0 = fma(ci5, T::R37_EVEN_TWIDDLE_11, c0);
        s0 = fma(si5, -T::R37_ODD_TWIDDLE_8, s0);
        c0 = fma(ci6, T::R37_EVEN_TWIDDLE_3, c0);
        s0 = fma(si6, T::R37_ODD_TWIDDLE_13, s0);
        c0 = fma(ci7, T::R37_EVEN_TWIDDLE_4, c0);
        s0 = fma(si7, T::R37_ODD_TWIDDLE_1, s0);
        c0 = fma(ci8, T::R37_EVEN_TWIDDLE_12, c0);
        s0 = fma(si8, T::R37_ODD_TWIDDLE_16, s0);
        c0 = fma(ci9, T::R37_EVEN_TWIDDLE_16, c0);
        s0 = fma(si9, -T::R37_ODD_TWIDDLE_5, s0);
        c0 = fma(ci10, T::R37_EVEN_TWIDDLE_8, c0);
        s0 = fma(si10, -T::R37_ODD_TWIDDLE_9, s0);
        c0 = fma(ci11, T::R37_EVEN_TWIDDLE_0, c0);
        s0 = fma(si11, T::R37_ODD_TWIDDLE_12, s0);
        c0 = fma(ci12, T::R37_EVEN_TWIDDLE_7, c0);
        s0 = fma(si12, T::R37_ODD_TWIDDLE_2, s0);
        c0 = fma(ci13, T::R37_EVEN_TWIDDLE_15, c0);
        s0 = fma(si13, T::R37_ODD_TWIDDLE_17, s0);
        c0 = fma(ci14, T::R37_EVEN_TWIDDLE_13, c0);
        s0 = fma(si14, -T::R37_ODD_TWIDDLE_4, s0);
        c0 = fma(ci15, T::R37_EVEN_TWIDDLE_5, c0);
        s0 = fma(si15, -T::R37_ODD_TWIDDLE_10, s0);
        c0 = fma(ci16, T::R37_EVEN_TWIDDLE_2, c0);
        s0 = fma(si16, T::R37_ODD_TWIDDLE_11, s0);
        c0 = fma(ci17, T::R37_EVEN_TWIDDLE_10, c0);
        s0 = fma(si17, T::R37_ODD_TWIDDLE_3, s0);
        let dc = c0 + a0;
        data[16] = dc;
        data[15] = -s0;
        let mut c0 = qc * T::R37_EVEN_TWIDDLE_4;
        let mut s0 = s_buffer[0];
        s0 *= T::R37_ODD_TWIDDLE_8;
        c0 = fma(ci1, T::R37_EVEN_TWIDDLE_13, c0);
        s0 = fma(si1, -T::R37_ODD_TWIDDLE_11, s0);
        c0 = fma(ci2, T::R37_EVEN_TWIDDLE_14, c0);
        s0 = fma(si2, -T::R37_ODD_TWIDDLE_5, s0);
        c0 = fma(ci3, T::R37_EVEN_TWIDDLE_5, c0);
        s0 = fma(si3, T::R37_ODD_TWIDDLE_14, s0);
        c0 = fma(ci4, T::R37_EVEN_TWIDDLE_3, c0);
        s0 = fma(si4, T::R37_ODD_TWIDDLE_2, s0);
        c0 = fma(ci5, T::R37_EVEN_TWIDDLE_12, c0);
        s0 = fma(si5, -T::R37_ODD_TWIDDLE_17, s0);
        c0 = fma(ci6, T::R37_EVEN_TWIDDLE_15, c0);
        s0 = fma(si6, -T::R37_ODD_TWIDDLE_0, s0);
        c0 = fma(ci7, T::R37_EVEN_TWIDDLE_6, c0);
        s0 = fma(si7, -T::R37_ODD_TWIDDLE_16, s0);
        c0 = fma(ci8, T::R37_EVEN_TWIDDLE_2, c0);
        s0 = fma(si8, T::R37_ODD_TWIDDLE_3, s0);
        c0 = fma(ci9, T::R37_EVEN_TWIDDLE_11, c0);
        s0 = fma(si9, T::R37_ODD_TWIDDLE_13, s0);
        c0 = fma(ci10, T::R37_EVEN_TWIDDLE_16, c0);
        s0 = fma(si10, -T::R37_ODD_TWIDDLE_6, s0);
        c0 = fma(ci11, T::R37_EVEN_TWIDDLE_7, c0);
        s0 = fma(si11, -T::R37_ODD_TWIDDLE_10, s0);
        c0 = fma(ci12, T::R37_EVEN_TWIDDLE_1, c0);
        s0 = fma(si12, T::R37_ODD_TWIDDLE_9, s0);
        c0 = fma(ci13, T::R37_EVEN_TWIDDLE_10, c0);
        s0 = fma(si13, T::R37_ODD_TWIDDLE_7, s0);
        c0 = fma(ci14, T::R37_EVEN_TWIDDLE_17, c0);
        s0 = fma(si14, -T::R37_ODD_TWIDDLE_12, s0);
        c0 = fma(ci15, T::R37_EVEN_TWIDDLE_8, c0);
        s0 = fma(si15, -T::R37_ODD_TWIDDLE_4, s0);
        c0 = fma(ci16, T::R37_EVEN_TWIDDLE_0, c0);
        s0 = fma(si16, T::R37_ODD_TWIDDLE_15, s0);
        c0 = fma(ci17, T::R37_EVEN_TWIDDLE_9, c0);
        s0 = fma(si17, T::R37_ODD_TWIDDLE_1, s0);
        let dc = c0 + a0;
        data[18] = -dc;
        data[17] = s0;
        let mut c0 = qc * T::R37_EVEN_TWIDDLE_13;
        let mut s0 = s_buffer[0];
        s0 *= -T::R37_ODD_TWIDDLE_9;
        c0 = fma(ci1, T::R37_EVEN_TWIDDLE_3, c0);
        s0 = fma(si1, T::R37_ODD_TWIDDLE_8, s0);
        c0 = fma(ci2, T::R37_EVEN_TWIDDLE_6, c0);
        s0 = fma(si2, T::R37_ODD_TWIDDLE_10, s0);
        c0 = fma(ci3, T::R37_EVEN_TWIDDLE_16, c0);
        s0 = fma(si3, -T::R37_ODD_TWIDDLE_7, s0);
        c0 = fma(ci4, T::R37_EVEN_TWIDDLE_10, c0);
        s0 = fma(si4, -T::R37_ODD_TWIDDLE_11, s0);
        c0 = fma(ci5, T::R37_EVEN_TWIDDLE_0, c0);
        s0 = fma(si5, T::R37_ODD_TWIDDLE_6, s0);
        c0 = fma(ci6, T::R37_EVEN_TWIDDLE_9, c0);
        s0 = fma(si6, T::R37_ODD_TWIDDLE_12, s0);
        c0 = fma(ci7, T::R37_EVEN_TWIDDLE_17, c0);
        s0 = fma(si7, -T::R37_ODD_TWIDDLE_5, s0);
        c0 = fma(ci8, T::R37_EVEN_TWIDDLE_7, c0);
        s0 = fma(si8, -T::R37_ODD_TWIDDLE_13, s0);
        c0 = fma(ci9, T::R37_EVEN_TWIDDLE_2, c0);
        s0 = fma(si9, T::R37_ODD_TWIDDLE_4, s0);
        c0 = fma(ci10, T::R37_EVEN_TWIDDLE_12, c0);
        s0 = fma(si10, T::R37_ODD_TWIDDLE_14, s0);
        c0 = fma(ci11, T::R37_EVEN_TWIDDLE_14, c0);
        s0 = fma(si11, -T::R37_ODD_TWIDDLE_3, s0);
        c0 = fma(ci12, T::R37_EVEN_TWIDDLE_4, c0);
        s0 = fma(si12, -T::R37_ODD_TWIDDLE_15, s0);
        c0 = fma(ci13, T::R37_EVEN_TWIDDLE_5, c0);
        s0 = fma(si13, T::R37_ODD_TWIDDLE_2, s0);
        c0 = fma(ci14, T::R37_EVEN_TWIDDLE_15, c0);
        s0 = fma(si14, T::R37_ODD_TWIDDLE_16, s0);
        c0 = fma(ci15, T::R37_EVEN_TWIDDLE_11, c0);
        s0 = fma(si15, -T::R37_ODD_TWIDDLE_1, s0);
        c0 = fma(ci16, T::R37_EVEN_TWIDDLE_1, c0);
        s0 = fma(si16, -T::R37_ODD_TWIDDLE_17, s0);
        c0 = fma(ci17, T::R37_EVEN_TWIDDLE_8, c0);
        s0 = fma(si17, T::R37_ODD_TWIDDLE_0, s0);
        let dc = c0 + a0;
        data[20] = dc;
        data[19] = -s0;
        let mut c0 = qc * T::R37_EVEN_TWIDDLE_5;
        let mut s0 = s_buffer[0];
        s0 *= T::R37_ODD_TWIDDLE_10;
        c0 = fma(ci1, T::R37_EVEN_TWIDDLE_16, c0);
        s0 = fma(si1, -T::R37_ODD_TWIDDLE_5, s0);
        c0 = fma(ci2, T::R37_EVEN_TWIDDLE_9, c0);
        s0 = fma(si2, -T::R37_ODD_TWIDDLE_15, s0);
        c0 = fma(ci3, T::R37_EVEN_TWIDDLE_1, c0);
        s0 = fma(si3, T::R37_ODD_TWIDDLE_0, s0);
        c0 = fma(ci4, T::R37_EVEN_TWIDDLE_12, c0);
        s0 = fma(si4, -T::R37_ODD_TWIDDLE_16, s0);
        c0 = fma(ci5, T::R37_EVEN_TWIDDLE_13, c0);
        s0 = fma(si5, -T::R37_ODD_TWIDDLE_4, s0);
        c0 = fma(ci6, T::R37_EVEN_TWIDDLE_2, c0);
        s0 = fma(si6, T::R37_ODD_TWIDDLE_11, s0);
        c0 = fma(ci7, T::R37_EVEN_TWIDDLE_8, c0);
        s0 = fma(si7, T::R37_ODD_TWIDDLE_9, s0);
        c0 = fma(ci8, T::R37_EVEN_TWIDDLE_17, c0);
        s0 = fma(si8, -T::R37_ODD_TWIDDLE_6, s0);
        c0 = fma(ci9, T::R37_EVEN_TWIDDLE_6, c0);
        s0 = fma(si9, -T::R37_ODD_TWIDDLE_14, s0);
        c0 = fma(ci10, T::R37_EVEN_TWIDDLE_4, c0);
        s0 = fma(si10, T::R37_ODD_TWIDDLE_1, s0);
        c0 = fma(ci11, T::R37_EVEN_TWIDDLE_15, c0);
        s0 = fma(si11, -T::R37_ODD_TWIDDLE_17, s0);
        c0 = fma(ci12, T::R37_EVEN_TWIDDLE_10, c0);
        s0 = fma(si12, -T::R37_ODD_TWIDDLE_3, s0);
        c0 = fma(ci13, T::R37_EVEN_TWIDDLE_0, c0);
        s0 = fma(si13, T::R37_ODD_TWIDDLE_12, s0);
        c0 = fma(ci14, T::R37_EVEN_TWIDDLE_11, c0);
        s0 = fma(si14, T::R37_ODD_TWIDDLE_8, s0);
        c0 = fma(ci15, T::R37_EVEN_TWIDDLE_14, c0);
        s0 = fma(si15, -T::R37_ODD_TWIDDLE_7, s0);
        c0 = fma(ci16, T::R37_EVEN_TWIDDLE_3, c0);
        s0 = fma(si16, -T::R37_ODD_TWIDDLE_13, s0);
        c0 = fma(ci17, T::R37_EVEN_TWIDDLE_7, c0);
        s0 = fma(si17, T::R37_ODD_TWIDDLE_2, s0);
        let dc = c0 + a0;
        data[22] = -dc;
        data[21] = s0;
        let mut c0 = qc * T::R37_EVEN_TWIDDLE_12;
        let mut s0 = s_buffer[0];
        s0 *= -T::R37_ODD_TWIDDLE_11;
        c0 = fma(ci1, T::R37_EVEN_TWIDDLE_0, c0);
        s0 = fma(si1, T::R37_ODD_TWIDDLE_2, s0);
        c0 = fma(ci2, T::R37_EVEN_TWIDDLE_11, c0);
        s0 = fma(si2, -T::R37_ODD_TWIDDLE_16, s0);
        c0 = fma(ci3, T::R37_EVEN_TWIDDLE_13, c0);
        s0 = fma(si3, -T::R37_ODD_TWIDDLE_6, s0);
        c0 = fma(ci4, T::R37_EVEN_TWIDDLE_1, c0);
        s0 = fma(si4, T::R37_ODD_TWIDDLE_7, s0);
        c0 = fma(ci5, T::R37_EVEN_TWIDDLE_10, c0);
        s0 = fma(si5, T::R37_ODD_TWIDDLE_15, s0);
        c0 = fma(ci6, T::R37_EVEN_TWIDDLE_14, c0);
        s0 = fma(si6, -T::R37_ODD_TWIDDLE_1, s0);
        c0 = fma(ci7, T::R37_EVEN_TWIDDLE_2, c0);
        s0 = fma(si7, T::R37_ODD_TWIDDLE_12, s0);
        c0 = fma(ci8, T::R37_EVEN_TWIDDLE_9, c0);
        s0 = fma(si8, T::R37_ODD_TWIDDLE_10, s0);
        c0 = fma(ci9, T::R37_EVEN_TWIDDLE_15, c0);
        s0 = fma(si9, -T::R37_ODD_TWIDDLE_3, s0);
        c0 = fma(ci10, T::R37_EVEN_TWIDDLE_3, c0);
        s0 = fma(si10, T::R37_ODD_TWIDDLE_17, s0);
        c0 = fma(ci11, T::R37_EVEN_TWIDDLE_8, c0);
        s0 = fma(si11, T::R37_ODD_TWIDDLE_5, s0);
        c0 = fma(ci12, T::R37_EVEN_TWIDDLE_16, c0);
        s0 = fma(si12, -T::R37_ODD_TWIDDLE_8, s0);
        c0 = fma(ci13, T::R37_EVEN_TWIDDLE_4, c0);
        s0 = fma(si13, -T::R37_ODD_TWIDDLE_14, s0);
        c0 = fma(ci14, T::R37_EVEN_TWIDDLE_7, c0);
        s0 = fma(si14, T::R37_ODD_TWIDDLE_0, s0);
        c0 = fma(ci15, T::R37_EVEN_TWIDDLE_17, c0);
        s0 = fma(si15, -T::R37_ODD_TWIDDLE_13, s0);
        c0 = fma(ci16, T::R37_EVEN_TWIDDLE_5, c0);
        s0 = fma(si16, -T::R37_ODD_TWIDDLE_9, s0);
        c0 = fma(ci17, T::R37_EVEN_TWIDDLE_6, c0);
        s0 = fma(si17, T::R37_ODD_TWIDDLE_4, s0);
        let dc = c0 + a0;
        data[24] = dc;
        data[23] = -s0;
        let mut c0 = qc * T::R37_EVEN_TWIDDLE_6;
        let mut s0 = s_buffer[0];
        s0 *= T::R37_ODD_TWIDDLE_12;
        c0 = fma(ci1, T::R37_EVEN_TWIDDLE_17, c0);
        s0 = fma(si1, -T::R37_ODD_TWIDDLE_0, s0);
        c0 = fma(ci2, T::R37_EVEN_TWIDDLE_4, c0);
        s0 = fma(si2, T::R37_ODD_TWIDDLE_11, s0);
        c0 = fma(ci3, T::R37_EVEN_TWIDDLE_8, c0);
        s0 = fma(si3, T::R37_ODD_TWIDDLE_13, s0);
        c0 = fma(ci4, T::R37_EVEN_TWIDDLE_15, c0);
        s0 = fma(si4, -T::R37_ODD_TWIDDLE_1, s0);
        c0 = fma(ci5, T::R37_EVEN_TWIDDLE_2, c0);
        s0 = fma(si5, T::R37_ODD_TWIDDLE_10, s0);
        c0 = fma(ci6, T::R37_EVEN_TWIDDLE_10, c0);
        s0 = fma(si6, T::R37_ODD_TWIDDLE_14, s0);
        c0 = fma(ci7, T::R37_EVEN_TWIDDLE_13, c0);
        s0 = fma(si7, -T::R37_ODD_TWIDDLE_2, s0);
        c0 = fma(ci8, T::R37_EVEN_TWIDDLE_0, c0);
        s0 = fma(si8, T::R37_ODD_TWIDDLE_9, s0);
        c0 = fma(ci9, T::R37_EVEN_TWIDDLE_12, c0);
        s0 = fma(si9, T::R37_ODD_TWIDDLE_15, s0);
        c0 = fma(ci10, T::R37_EVEN_TWIDDLE_11, c0);
        s0 = fma(si10, -T::R37_ODD_TWIDDLE_3, s0);
        c0 = fma(ci11, T::R37_EVEN_TWIDDLE_1, c0);
        s0 = fma(si11, T::R37_ODD_TWIDDLE_8, s0);
        c0 = fma(ci12, T::R37_EVEN_TWIDDLE_14, c0);
        s0 = fma(si12, T::R37_ODD_TWIDDLE_16, s0);
        c0 = fma(ci13, T::R37_EVEN_TWIDDLE_9, c0);
        s0 = fma(si13, -T::R37_ODD_TWIDDLE_4, s0);
        c0 = fma(ci14, T::R37_EVEN_TWIDDLE_3, c0);
        s0 = fma(si14, T::R37_ODD_TWIDDLE_7, s0);
        c0 = fma(ci15, T::R37_EVEN_TWIDDLE_16, c0);
        s0 = fma(si15, T::R37_ODD_TWIDDLE_17, s0);
        c0 = fma(ci16, T::R37_EVEN_TWIDDLE_7, c0);
        s0 = fma(si16, -T::R37_ODD_TWIDDLE_5, s0);
        c0 = fma(ci17, T::R37_EVEN_TWIDDLE_5, c0);
        s0 = fma(si17, T::R37_ODD_TWIDDLE_6, s0);
        let dc = c0 + a0;
        data[26] = -dc;
        data[25] = s0;
        let mut c0 = qc * T::R37_EVEN_TWIDDLE_11;
        let mut s0 = s_buffer[0];
        s0 *= -T::R37_ODD_TWIDDLE_13;
        c0 = fma(ci1, T::R37_EVEN_TWIDDLE_2, c0);
        s0 = fma(si1, T::R37_ODD_TWIDDLE_3, s0);
        c0 = fma(ci2, T::R37_EVEN_TWIDDLE_16, c0);
        s0 = fma(si2, -T::R37_ODD_TWIDDLE_6, s0);
        c0 = fma(ci3, T::R37_EVEN_TWIDDLE_6, c0);
        s0 = fma(si3, T::R37_ODD_TWIDDLE_16, s0);
        c0 = fma(ci4, T::R37_EVEN_TWIDDLE_7, c0);
        s0 = fma(si4, T::R37_ODD_TWIDDLE_10, s0);
        c0 = fma(ci5, T::R37_EVEN_TWIDDLE_15, c0);
        s0 = fma(si5, -T::R37_ODD_TWIDDLE_0, s0);
        c0 = fma(ci6, T::R37_EVEN_TWIDDLE_1, c0);
        s0 = fma(si6, T::R37_ODD_TWIDDLE_9, s0);
        c0 = fma(ci7, T::R37_EVEN_TWIDDLE_12, c0);
        s0 = fma(si7, T::R37_ODD_TWIDDLE_17, s0);
        c0 = fma(ci8, T::R37_EVEN_TWIDDLE_10, c0);
        s0 = fma(si8, -T::R37_ODD_TWIDDLE_7, s0);
        c0 = fma(ci9, T::R37_EVEN_TWIDDLE_3, c0);
        s0 = fma(si9, T::R37_ODD_TWIDDLE_2, s0);
        c0 = fma(ci10, T::R37_EVEN_TWIDDLE_17, c0);
        s0 = fma(si10, -T::R37_ODD_TWIDDLE_12, s0);
        c0 = fma(ci11, T::R37_EVEN_TWIDDLE_5, c0);
        s0 = fma(si11, -T::R37_ODD_TWIDDLE_14, s0);
        c0 = fma(ci12, T::R37_EVEN_TWIDDLE_8, c0);
        s0 = fma(si12, T::R37_ODD_TWIDDLE_4, s0);
        c0 = fma(ci13, T::R37_EVEN_TWIDDLE_14, c0);
        s0 = fma(si13, -T::R37_ODD_TWIDDLE_5, s0);
        c0 = fma(ci14, T::R37_EVEN_TWIDDLE_0, c0);
        s0 = fma(si14, T::R37_ODD_TWIDDLE_15, s0);
        c0 = fma(ci15, T::R37_EVEN_TWIDDLE_13, c0);
        s0 = fma(si15, T::R37_ODD_TWIDDLE_11, s0);
        c0 = fma(ci16, T::R37_EVEN_TWIDDLE_9, c0);
        s0 = fma(si16, -T::R37_ODD_TWIDDLE_1, s0);
        c0 = fma(ci17, T::R37_EVEN_TWIDDLE_4, c0);
        s0 = fma(si17, T::R37_ODD_TWIDDLE_8, s0);
        let dc = c0 + a0;
        data[28] = dc;
        data[27] = -s0;
        let mut c0 = qc * T::R37_EVEN_TWIDDLE_7;
        let mut s0 = s_buffer[0];
        s0 *= T::R37_ODD_TWIDDLE_14;
        c0 = fma(ci1, T::R37_EVEN_TWIDDLE_14, c0);
        s0 = fma(si1, -T::R37_ODD_TWIDDLE_6, s0);
        c0 = fma(ci2, T::R37_EVEN_TWIDDLE_0, c0);
        s0 = fma(si2, T::R37_ODD_TWIDDLE_1, s0);
        c0 = fma(ci3, T::R37_EVEN_TWIDDLE_15, c0);
        s0 = fma(si3, -T::R37_ODD_TWIDDLE_9, s0);
        c0 = fma(ci4, T::R37_EVEN_TWIDDLE_6, c0);
        s0 = fma(si4, T::R37_ODD_TWIDDLE_17, s0);
        c0 = fma(ci5, T::R37_EVEN_TWIDDLE_8, c0);
        s0 = fma(si5, T::R37_ODD_TWIDDLE_11, s0);
        c0 = fma(ci6, T::R37_EVEN_TWIDDLE_13, c0);
        s0 = fma(si6, -T::R37_ODD_TWIDDLE_3, s0);
        c0 = fma(ci7, T::R37_EVEN_TWIDDLE_1, c0);
        s0 = fma(si7, T::R37_ODD_TWIDDLE_4, s0);
        c0 = fma(ci8, T::R37_EVEN_TWIDDLE_16, c0);
        s0 = fma(si8, -T::R37_ODD_TWIDDLE_12, s0);
        c0 = fma(ci9, T::R37_EVEN_TWIDDLE_5, c0);
        s0 = fma(si9, -T::R37_ODD_TWIDDLE_16, s0);
        c0 = fma(ci10, T::R37_EVEN_TWIDDLE_9, c0);
        s0 = fma(si10, T::R37_ODD_TWIDDLE_8, s0);
        c0 = fma(ci11, T::R37_EVEN_TWIDDLE_12, c0);
        s0 = fma(si11, -T::R37_ODD_TWIDDLE_0, s0);
        c0 = fma(ci12, T::R37_EVEN_TWIDDLE_2, c0);
        s0 = fma(si12, T::R37_ODD_TWIDDLE_7, s0);
        c0 = fma(ci13, T::R37_EVEN_TWIDDLE_17, c0);
        s0 = fma(si13, -T::R37_ODD_TWIDDLE_15, s0);
        c0 = fma(ci14, T::R37_EVEN_TWIDDLE_4, c0);
        s0 = fma(si14, -T::R37_ODD_TWIDDLE_13, s0);
        c0 = fma(ci15, T::R37_EVEN_TWIDDLE_10, c0);
        s0 = fma(si15, T::R37_ODD_TWIDDLE_5, s0);
        c0 = fma(ci16, T::R37_EVEN_TWIDDLE_11, c0);
        s0 = fma(si16, -T::R37_ODD_TWIDDLE_2, s0);
        c0 = fma(ci17, T::R37_EVEN_TWIDDLE_3, c0);
        s0 = fma(si17, T::R37_ODD_TWIDDLE_10, s0);
        let dc = c0 + a0;
        data[30] = -dc;
        data[29] = s0;
        let mut c0 = qc * T::R37_EVEN_TWIDDLE_10;
        let mut s0 = s_buffer[0];
        s0 *= -T::R37_ODD_TWIDDLE_15;
        c0 = fma(ci1, T::R37_EVEN_TWIDDLE_5, c0);
        s0 = fma(si1, T::R37_ODD_TWIDDLE_9, s0);
        c0 = fma(ci2, T::R37_EVEN_TWIDDLE_15, c0);
        s0 = fma(si2, -T::R37_ODD_TWIDDLE_3, s0);
        c0 = fma(ci3, T::R37_EVEN_TWIDDLE_0, c0);
        s0 = fma(si3, T::R37_ODD_TWIDDLE_2, s0);
        c0 = fma(ci4, T::R37_EVEN_TWIDDLE_16, c0);
        s0 = fma(si4, -T::R37_ODD_TWIDDLE_8, s0);
        c0 = fma(ci5, T::R37_EVEN_TWIDDLE_4, c0);
        s0 = fma(si5, T::R37_ODD_TWIDDLE_14, s0);
        c0 = fma(ci6, T::R37_EVEN_TWIDDLE_11, c0);
        s0 = fma(si6, T::R37_ODD_TWIDDLE_16, s0);
        c0 = fma(ci7, T::R37_EVEN_TWIDDLE_9, c0);
        s0 = fma(si7, -T::R37_ODD_TWIDDLE_10, s0);
        c0 = fma(ci8, T::R37_EVEN_TWIDDLE_6, c0);
        s0 = fma(si8, T::R37_ODD_TWIDDLE_4, s0);
        c0 = fma(ci9, T::R37_EVEN_TWIDDLE_14, c0);
        s0 = fma(si9, -T::R37_ODD_TWIDDLE_1, s0);
        c0 = fma(ci10, T::R37_EVEN_TWIDDLE_1, c0);
        s0 = fma(si10, T::R37_ODD_TWIDDLE_7, s0);
        c0 = fma(ci11, T::R37_EVEN_TWIDDLE_17, c0);
        s0 = fma(si11, -T::R37_ODD_TWIDDLE_13, s0);
        c0 = fma(ci12, T::R37_EVEN_TWIDDLE_3, c0);
        s0 = fma(si12, -T::R37_ODD_TWIDDLE_17, s0);
        c0 = fma(ci13, T::R37_EVEN_TWIDDLE_12, c0);
        s0 = fma(si13, T::R37_ODD_TWIDDLE_11, s0);
        c0 = fma(ci14, T::R37_EVEN_TWIDDLE_8, c0);
        s0 = fma(si14, -T::R37_ODD_TWIDDLE_5, s0);
        c0 = fma(ci15, T::R37_EVEN_TWIDDLE_7, c0);
        s0 = fma(si15, T::R37_ODD_TWIDDLE_0, s0);
        c0 = fma(ci16, T::R37_EVEN_TWIDDLE_13, c0);
        s0 = fma(si16, -T::R37_ODD_TWIDDLE_6, s0);
        c0 = fma(ci17, T::R37_EVEN_TWIDDLE_2, c0);
        s0 = fma(si17, T::R37_ODD_TWIDDLE_12, s0);
        let dc = c0 + a0;
        data[32] = dc;
        data[31] = -s0;
        let mut c0 = qc * T::R37_EVEN_TWIDDLE_8;
        let mut s0 = s_buffer[0];
        s0 *= T::R37_ODD_TWIDDLE_16;
        c0 = fma(ci1, T::R37_EVEN_TWIDDLE_11, c0);
        s0 = fma(si1, -T::R37_ODD_TWIDDLE_12, s0);
        c0 = fma(ci2, T::R37_EVEN_TWIDDLE_5, c0);
        s0 = fma(si2, T::R37_ODD_TWIDDLE_8, s0);
        c0 = fma(ci3, T::R37_EVEN_TWIDDLE_14, c0);
        s0 = fma(si3, -T::R37_ODD_TWIDDLE_4, s0);
        c0 = fma(ci4, T::R37_EVEN_TWIDDLE_2, c0);
        s0 = fma(si4, T::R37_ODD_TWIDDLE_0, s0);
        c0 = fma(ci5, T::R37_EVEN_TWIDDLE_17, c0);
        s0 = fma(si5, -T::R37_ODD_TWIDDLE_3, s0);
        c0 = fma(ci6, T::R37_EVEN_TWIDDLE_0, c0);
        s0 = fma(si6, T::R37_ODD_TWIDDLE_7, s0);
        c0 = fma(ci7, T::R37_EVEN_TWIDDLE_16, c0);
        s0 = fma(si7, -T::R37_ODD_TWIDDLE_11, s0);
        c0 = fma(ci8, T::R37_EVEN_TWIDDLE_3, c0);
        s0 = fma(si8, T::R37_ODD_TWIDDLE_15, s0);
        c0 = fma(ci9, T::R37_EVEN_TWIDDLE_13, c0);
        s0 = fma(si9, T::R37_ODD_TWIDDLE_17, s0);
        c0 = fma(ci10, T::R37_EVEN_TWIDDLE_6, c0);
        s0 = fma(si10, -T::R37_ODD_TWIDDLE_13, s0);
        c0 = fma(ci11, T::R37_EVEN_TWIDDLE_10, c0);
        s0 = fma(si11, T::R37_ODD_TWIDDLE_9, s0);
        c0 = fma(ci12, T::R37_EVEN_TWIDDLE_9, c0);
        s0 = fma(si12, -T::R37_ODD_TWIDDLE_5, s0);
        c0 = fma(ci13, T::R37_EVEN_TWIDDLE_7, c0);
        s0 = fma(si13, T::R37_ODD_TWIDDLE_1, s0);
        c0 = fma(ci14, T::R37_EVEN_TWIDDLE_12, c0);
        s0 = fma(si14, -T::R37_ODD_TWIDDLE_2, s0);
        c0 = fma(ci15, T::R37_EVEN_TWIDDLE_4, c0);
        s0 = fma(si15, T::R37_ODD_TWIDDLE_6, s0);
        c0 = fma(ci16, T::R37_EVEN_TWIDDLE_15, c0);
        s0 = fma(si16, -T::R37_ODD_TWIDDLE_10, s0);
        c0 = fma(ci17, T::R37_EVEN_TWIDDLE_1, c0);
        s0 = fma(si17, T::R37_ODD_TWIDDLE_14, s0);
        let dc = c0 + a0;
        data[34] = -dc;
        data[33] = s0;
        let mut c0 = qc * T::R37_EVEN_TWIDDLE_9;
        let mut s0 = s_buffer[0];
        s0 *= -T::R37_ODD_TWIDDLE_17;
        c0 = fma(ci1, T::R37_EVEN_TWIDDLE_8, c0);
        s0 = fma(si1, T::R37_ODD_TWIDDLE_15, s0);
        c0 = fma(ci2, T::R37_EVEN_TWIDDLE_10, c0);
        s0 = fma(si2, -T::R37_ODD_TWIDDLE_13, s0);
        c0 = fma(ci3, T::R37_EVEN_TWIDDLE_7, c0);
        s0 = fma(si3, T::R37_ODD_TWIDDLE_11, s0);
        c0 = fma(ci4, T::R37_EVEN_TWIDDLE_11, c0);
        s0 = fma(si4, -T::R37_ODD_TWIDDLE_9, s0);
        c0 = fma(ci5, T::R37_EVEN_TWIDDLE_6, c0);
        s0 = fma(si5, T::R37_ODD_TWIDDLE_7, s0);
        c0 = fma(ci6, T::R37_EVEN_TWIDDLE_12, c0);
        s0 = fma(si6, -T::R37_ODD_TWIDDLE_5, s0);
        c0 = fma(ci7, T::R37_EVEN_TWIDDLE_5, c0);
        s0 = fma(si7, T::R37_ODD_TWIDDLE_3, s0);
        c0 = fma(ci8, T::R37_EVEN_TWIDDLE_13, c0);
        s0 = fma(si8, -T::R37_ODD_TWIDDLE_1, s0);
        c0 = fma(ci9, T::R37_EVEN_TWIDDLE_4, c0);
        s0 = fma(si9, T::R37_ODD_TWIDDLE_0, s0);
        c0 = fma(ci10, T::R37_EVEN_TWIDDLE_14, c0);
        s0 = fma(si10, -T::R37_ODD_TWIDDLE_2, s0);
        c0 = fma(ci11, T::R37_EVEN_TWIDDLE_3, c0);
        s0 = fma(si11, T::R37_ODD_TWIDDLE_4, s0);
        c0 = fma(ci12, T::R37_EVEN_TWIDDLE_15, c0);
        s0 = fma(si12, -T::R37_ODD_TWIDDLE_6, s0);
        c0 = fma(ci13, T::R37_EVEN_TWIDDLE_2, c0);
        s0 = fma(si13, T::R37_ODD_TWIDDLE_8, s0);
        c0 = fma(ci14, T::R37_EVEN_TWIDDLE_16, c0);
        s0 = fma(si14, -T::R37_ODD_TWIDDLE_10, s0);
        c0 = fma(ci15, T::R37_EVEN_TWIDDLE_1, c0);
        s0 = fma(si15, T::R37_ODD_TWIDDLE_12, s0);
        c0 = fma(ci16, T::R37_EVEN_TWIDDLE_17, c0);
        s0 = fma(si16, -T::R37_ODD_TWIDDLE_14, s0);
        c0 = fma(ci17, T::R37_EVEN_TWIDDLE_0, c0);
        s0 = fma(si17, T::R37_ODD_TWIDDLE_16, s0);
        let dc = c0 + a0;
        data[36] = dc;
        data[35] = -s0;
    }
}

impl<T: DctSample + MixedRadix37Sample> AvxDct2Butterfly37<T>
where
    f64: AsPrimitive<T>,
{
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_impl(&self, data: &mut [T]) -> Result<(), PxdctError> {
        if !data.len().is_multiple_of(37) {
            return Err(PxdctError::InvalidSizeMultiplier(data.len(), self.length()));
        }
        for chunk in data.chunks_exact_mut(37) {
            self.exec(&mut InPlaceStore::new(chunk));
        }
        Ok(())
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_into_impl(&self, input: &[T], output: &mut [T]) -> Result<(), PxdctError> {
        use crate::util::validate_oof_sizes;
        validate_oof_sizes!(input, output, 37);
        use crate::bidirectional::BiStore;
        for (src, dst) in input.chunks_exact(37).zip(output.chunks_exact_mut(37)) {
            self.exec(&mut BiStore::new(src, dst));
        }
        Ok(())
    }
}

impl<T: DctSample + MixedRadix37Sample> PxdctExecutor<T> for AvxDct2Butterfly37<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, data: &mut [T]) -> Result<(), PxdctError> {
        unsafe { self.execute_impl(data) }
    }

    fn execute_with_scratch(&self, data: &mut [T], _: &mut [T]) -> Result<(), PxdctError> {
        unsafe { self.execute_impl(data) }
    }

    fn execute_into(&self, input: &[T], output: &mut [T]) -> Result<(), PxdctError> {
        unsafe { self.execute_into_impl(input, output) }
    }

    fn execute_into_with_scratch(
        &self,
        input: &[T],
        output: &mut [T],
        _: &mut [T],
    ) -> Result<(), PxdctError> {
        unsafe { self.execute_into_impl(input, output) }
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
    use crate::avx::dct2_bf_power2::gen_test_avx_butterfly;
    use crate::tests::naive_dct2;

    gen_test_avx_butterfly!(test_avx_bf3, AvxDct2Butterfly3, 3, 1e-7, naive_dct2);
    gen_test_avx_butterfly!(test_avx_bf5, AvxDct2Butterfly5, 5, 1e-7, naive_dct2);
    gen_test_avx_butterfly!(test_avx_bf7, AvxDct2Butterfly7, 7, 1e-7, naive_dct2);
    gen_test_avx_butterfly!(test_avx_bf11, AvxDct2Butterfly11, 11, 1e-7, naive_dct2);
    gen_test_avx_butterfly!(test_avx_bf13, AvxDct2Butterfly13, 13, 1e-7, naive_dct2);
    gen_test_avx_butterfly!(test_avx_bf17, AvxDct2Butterfly17, 17, 1e-7, naive_dct2);
    gen_test_avx_butterfly!(test_avx_bf19, AvxDct2Butterfly19, 19, 1e-7, naive_dct2);
    gen_test_avx_butterfly!(test_avx_bf23, AvxDct2Butterfly23, 23, 1e-7, naive_dct2);
    gen_test_avx_butterfly!(test_avx_bf29, AvxDct2Butterfly29, 29, 1e-7, naive_dct2);
    gen_test_avx_butterfly!(test_avx_bf31, AvxDct2Butterfly31, 31, 1e-7, naive_dct2);
    gen_test_avx_butterfly!(test_avx_bf37, AvxDct2Butterfly37, 37, 1e-7, naive_dct2);
}
