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
use crate::mla::fmla;
use crate::util::{
    DctSample, define_in_place_butterfly, mixed_radix_inner_twiddle, mixed_radix3_twiddles,
    validate_scratch,
};
use crate::{PxdctError, PxdctExecutor};
use num_complex::Complex;
use num_traits::AsPrimitive;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub(crate) struct Dct2Butterfly6<T: DctSample> {
    bf3: Dct2Butterfly3<T>,
    bf2: Dct2Butterfly2<T>,
}

impl<T: DctSample> Default for Dct2Butterfly6<T>
where
    f64: AsPrimitive<T>,
    usize: AsPrimitive<T>,
{
    fn default() -> Self {
        let mut inner_layer = [Complex::<T>::default(); 4];
        for (i, layer) in inner_layer.chunks_exact_mut(4).enumerate() {
            let angle = (2. * i as f64 + 1.).as_();
            layer[0] = mixed_radix_inner_twiddle(angle, 6);
            layer[0].im *= T::SQRT_3;
            layer[1] = mixed_radix_inner_twiddle(2f64.as_() * angle, 6);
            layer[1].im *= T::SQRT_3;
            layer[2] = mixed_radix_inner_twiddle(3f64.as_() * angle, 6);
            layer[2].im *= T::SQRT_3;
            layer[3] = mixed_radix_inner_twiddle(5f64.as_() * angle, 6);
            layer[3].im = -layer[3].im * T::SQRT_3;
        }

        Self {
            bf3: Dct2Butterfly3::default(),
            bf2: Dct2Butterfly2::default(),
        }
    }
}

impl<T: DctSample> Dct2Butterfly6<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        // co-prime 2x3 DCT-II algorithm

        let mut c0 = [data[0], data[5]];
        let mut c1 = [data[4], data[1]];
        let mut c2 = [data[3], data[2]];

        self.bf2.exec(&mut InPlaceStore::new(&mut c0));
        self.bf2.exec(&mut InPlaceStore::new(&mut c1));
        self.bf2.exec(&mut InPlaceStore::new(&mut c2));

        let mut rows0 = [c0[0], c1[0], c2[0]];
        let mut rows1 = [c0[1], c1[1], c2[1]];

        self.bf3.exec(&mut InPlaceStore::new(&mut rows0));
        self.bf3.exec(&mut InPlaceStore::new(&mut rows1));

        data[0] = rows0[0];
        data[1] = rows1[1] + rows1[2];
        data[2] = rows0[1];
        data[3] = rows1[0];
        data[4] = rows0[2];
        data[5] = rows1[1] - rows1[2];
    }
}

define_in_place_butterfly!(Dct2Butterfly6, 6);

pub(crate) trait MixedRadix9Sample {
    const R9_EVEN_TWIDDLE_0: Self;
    const R9_EVEN_TWIDDLE_1: Self;
    const R9_EVEN_TWIDDLE_2: Self;
    const R9_ODD_TWIDDLE_0: Self;
    const R9_ODD_TWIDDLE_1: Self;
    const R9_ODD_TWIDDLE_2: Self;
    const R9_ODD_TWIDDLE_3: Self;
}

impl MixedRadix9Sample for f32 {
    const R9_EVEN_TWIDDLE_0: f32 = f32::from_bits(0xbf708fb2);
    const R9_EVEN_TWIDDLE_1: f32 = f32::from_bits(0x3e31d0d4);
    const R9_EVEN_TWIDDLE_2: f32 = f32::from_bits(0x3f441b7d);
    const R9_ODD_TWIDDLE_0: f32 = f32::from_bits(0x3f7c1c5c);
    const R9_ODD_TWIDDLE_1: f32 = f32::from_bits(0x3f5db3d7);
    const R9_ODD_TWIDDLE_2: f32 = f32::from_bits(0x3f248dbb);
    const R9_ODD_TWIDDLE_3: f32 = f32::from_bits(0x3eaf1d44);
}

impl MixedRadix9Sample for f64 {
    const R9_EVEN_TWIDDLE_0: f64 = f64::from_bits(0xbfee11f642522d1b);
    const R9_EVEN_TWIDDLE_1: f64 = f64::from_bits(0x3fc63a1a7e0b738c);
    const R9_EVEN_TWIDDLE_2: f64 = f64::from_bits(0x3fe8836fa2cf5039);
    const R9_ODD_TWIDDLE_0: f64 = f64::from_bits(0x3fef838b8c811c17);
    const R9_ODD_TWIDDLE_1: f64 = f64::from_bits(0x3febb67ae8584caa);
    const R9_ODD_TWIDDLE_2: f64 = f64::from_bits(0x3fe491b7523c161c);
    const R9_ODD_TWIDDLE_3: f64 = f64::from_bits(0x3fd5e3a8748a0bf5);
}

#[derive(Debug, Clone)]
pub(crate) struct Dct2Butterfly9<T: DctSample> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T: DctSample> Default for Dct2Butterfly9<T>
where
    f64: AsPrimitive<T>,
    usize: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: DctSample> Dct2Butterfly9<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        // This is autogenerated Radix-Q where Q = 9.
        let a_buffer = [data[4]];
        let c_buffer = [
            data[0] + data[8],
            data[1] + data[7],
            data[2] + data[6],
            data[3] + data[5],
        ];
        let s_buffer = [
            data[0] - data[8],
            data[1] - data[7],
            data[2] - data[6],
            data[3] - data[5],
        ];
        let qc = c_buffer[0];
        let mut c0 = qc;
        let mut c1 = qc * T::R9_EVEN_TWIDDLE_0;
        let mut s0 = s_buffer[0];
        s0 *= T::R9_ODD_TWIDDLE_0;
        let ci1 = c_buffer[1];
        let si1 = s_buffer[1];
        c0 = ci1 + c0;
        c1 = fmla(ci1, -T::HALF, c1);
        s0 = fmla(si1, T::R9_ODD_TWIDDLE_1, s0);
        let ci2 = c_buffer[2];
        let si2 = s_buffer[2];
        c0 = ci2 + c0;
        c1 = fmla(ci2, T::R9_EVEN_TWIDDLE_1, c1);
        s0 = fmla(si2, T::R9_ODD_TWIDDLE_2, s0);
        let ci3 = c_buffer[3];
        let si3 = s_buffer[3];
        c0 = ci3 + c0;
        c1 = fmla(ci3, T::R9_EVEN_TWIDDLE_2, c1);
        s0 = fmla(si3, T::R9_ODD_TWIDDLE_3, s0);
        let a0 = a_buffer[0];
        let dc = c0 + a0;
        data[0] = dc;
        data[1] = s0;
        data[2] = -(c1 + a0);
        let mut c0 = qc * T::R9_EVEN_TWIDDLE_2;
        let mut s0 = s_buffer[0];
        s0 *= -T::R9_ODD_TWIDDLE_1;
        c0 = fmla(ci1, -T::HALF, c0);
        c0 = fmla(ci2, T::R9_EVEN_TWIDDLE_0, c0);
        s0 = fmla(si2, T::R9_ODD_TWIDDLE_1, s0);
        c0 = fmla(ci3, T::R9_EVEN_TWIDDLE_1, c0);
        s0 = fmla(si3, T::R9_ODD_TWIDDLE_1, s0);
        let dc = c0 + a0;
        data[4] = dc;
        data[3] = -s0;
        let mut c0 = qc * -T::HALF;
        let mut s0 = s_buffer[0];
        s0 *= T::R9_ODD_TWIDDLE_2;
        c0 += ci1;
        s0 = fmla(si1, -T::R9_ODD_TWIDDLE_1, s0);
        c0 = fmla(ci2, -T::HALF, c0);
        s0 = fmla(si2, -T::R9_ODD_TWIDDLE_3, s0);
        c0 = fmla(ci3, -T::HALF, c0);
        s0 = fmla(si3, T::R9_ODD_TWIDDLE_0, s0);
        let dc = c0 + a0;
        data[6] = -dc;
        data[5] = s0;
        let mut c0 = qc * T::R9_EVEN_TWIDDLE_1;
        let mut s0 = s_buffer[0];
        s0 *= -T::R9_ODD_TWIDDLE_3;
        c0 = fmla(ci1, -T::HALF, c0);
        s0 = fmla(si1, T::R9_ODD_TWIDDLE_1, s0);
        c0 = fmla(ci2, T::R9_EVEN_TWIDDLE_2, c0);
        s0 = fmla(si2, -T::R9_ODD_TWIDDLE_0, s0);
        c0 = fmla(ci3, T::R9_EVEN_TWIDDLE_0, c0);
        s0 = fmla(si3, T::R9_ODD_TWIDDLE_2, s0);
        let dc = c0 + a0;
        data[8] = dc;
        data[7] = -s0;
    }
}

define_in_place_butterfly!(Dct2Butterfly9, 9);

#[derive(Debug, Clone)]
pub(crate) struct Dct2Butterfly10<T: DctSample> {
    bf5: Dct2Butterfly5<T>,
    bf2: Dct2Butterfly2<T>,
}

impl<T: DctSample> Default for Dct2Butterfly10<T>
where
    f64: AsPrimitive<T>,
    usize: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            bf5: Dct2Butterfly5::default(),
            bf2: Dct2Butterfly2::default(),
        }
    }
}

impl<T: DctSample> Dct2Butterfly10<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        let mut c0 = [data[0], data[9]];
        let mut c1 = [data[8], data[1]];
        let mut c2 = [data[7], data[2]];
        let mut c3 = [data[3], data[6]];
        let mut c4 = [data[4], data[5]];

        self.bf2.exec(&mut InPlaceStore::new(&mut c0));
        self.bf2.exec(&mut InPlaceStore::new(&mut c1));
        self.bf2.exec(&mut InPlaceStore::new(&mut c2));
        self.bf2.exec(&mut InPlaceStore::new(&mut c3));
        self.bf2.exec(&mut InPlaceStore::new(&mut c4));

        let mut rows0 = [c0[0], c1[0], c2[0], c3[0], c4[0]];
        let mut rows1 = [c0[1], c1[1], c2[1], c3[1], c4[1]];

        self.bf5.exec(&mut InPlaceStore::new(&mut rows0));
        self.bf5.exec(&mut InPlaceStore::new(&mut rows1));

        data[0] = rows0[0];
        data[1] = rows1[2] + rows1[3];
        data[2] = rows0[1];
        data[3] = rows1[1] + rows1[4];
        data[4] = rows0[2];
        data[5] = rows1[0];
        data[6] = rows0[3];
        data[7] = rows1[1] - rows1[4];
        data[8] = rows0[4];
        data[9] = rows1[2] - rows1[3];
    }
}

define_in_place_butterfly!(Dct2Butterfly10, 10);

#[derive(Debug, Clone)]
pub(crate) struct Dct2Butterfly12<T: DctSample> {
    bf4: Dct2Butterfly4<T>,
    bf3: Dct2Butterfly3<T>,
}

impl<T: DctSample> Default for Dct2Butterfly12<T>
where
    f64: AsPrimitive<T>,
    usize: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            bf4: Dct2Butterfly4::default(),
            bf3: Dct2Butterfly3::default(),
        }
    }
}

impl<T: DctSample> Dct2Butterfly12<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        // co-prime 3x4 DCT-II algorithm

        let mut c0 = [data[0], data[7], data[8]];
        let mut c1 = [data[6], data[1], data[9]];
        let mut c2 = [data[5], data[10], data[2]];
        let mut c3 = [data[11], data[4], data[3]];

        self.bf3.exec(&mut InPlaceStore::new(&mut c0));
        self.bf3.exec(&mut InPlaceStore::new(&mut c1));
        self.bf3.exec(&mut InPlaceStore::new(&mut c2));
        self.bf3.exec(&mut InPlaceStore::new(&mut c3));

        let mut rows0 = [c0[0], c1[0], c2[0], c3[0]];
        let mut rows1 = [c0[1], c1[1], c2[1], c3[1]];
        let mut rows2 = [c0[2], c1[2], c2[2], c3[2]];

        self.bf4.exec(&mut InPlaceStore::new(&mut rows0));
        self.bf4.exec(&mut InPlaceStore::new(&mut rows1));
        self.bf4.exec(&mut InPlaceStore::new(&mut rows2));

        data[0] = rows0[0];
        data[1] = rows1[1] + rows2[3];
        data[2] = rows1[2] + rows2[2];
        data[3] = rows0[1];
        data[4] = rows1[0];
        data[5] = rows1[3] + rows2[1];
        data[6] = rows0[2];
        data[7] = -rows2[3] + rows1[1];
        data[8] = rows2[0];
        data[9] = rows0[3];
        data[10] = -rows2[2] + rows1[2];
        data[11] = rows2[1] - rows1[3];
    }
}

define_in_place_butterfly!(Dct2Butterfly12, 12);

#[derive(Debug, Clone)]
pub(crate) struct Dct2Butterfly14<T: DctSample> {
    bf7: Dct2Butterfly7<T>,
    bf2: Dct2Butterfly2<T>,
}

impl<T: DctSample> Default for Dct2Butterfly14<T>
where
    f64: AsPrimitive<T>,
    usize: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            bf7: Dct2Butterfly7::default(),
            bf2: Dct2Butterfly2::default(),
        }
    }
}

impl<T: DctSample + MixedRadix7Sample> Dct2Butterfly14<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        // co-prime 2x7 DCT-II algorithm
        let mut c0 = [data[0], data[13]];
        let mut c1 = [data[12], data[1]];
        let mut c2 = [data[11], data[2]];
        let mut c3 = [data[3], data[10]];
        let mut c4 = [data[4], data[9]];
        let mut c5 = [data[8], data[5]];
        let mut c6 = [data[7], data[6]];

        self.bf2.exec(&mut InPlaceStore::new(&mut c0));
        self.bf2.exec(&mut InPlaceStore::new(&mut c1));
        self.bf2.exec(&mut InPlaceStore::new(&mut c2));
        self.bf2.exec(&mut InPlaceStore::new(&mut c3));
        self.bf2.exec(&mut InPlaceStore::new(&mut c4));
        self.bf2.exec(&mut InPlaceStore::new(&mut c5));
        self.bf2.exec(&mut InPlaceStore::new(&mut c6));

        let mut rows0 = [c0[0], c1[0], c2[0], c3[0], c4[0], c5[0], c6[0]];
        let mut rows1 = [c0[1], c1[1], c2[1], c3[1], c4[1], c5[1], c6[1]];

        self.bf7.exec(&mut InPlaceStore::new(&mut rows0));
        self.bf7.exec(&mut InPlaceStore::new(&mut rows1));

        data[0] = rows0[0];
        data[1] = rows1[3] + rows1[4];
        data[2] = rows0[1];
        data[3] = rows1[2] + rows1[5];
        data[4] = rows0[2];
        data[5] = rows1[1] + rows1[6];
        data[6] = rows0[3];
        data[7] = rows1[0];
        data[8] = rows0[4];
        data[9] = rows1[1] - rows1[6];
        data[10] = rows0[5];
        data[11] = rows1[2] - rows1[5];
        data[12] = rows0[6];
        data[13] = rows1[3] - rows1[4];
    }
}

define_in_place_butterfly!(Dct2Butterfly14, 14);

#[derive(Debug, Clone)]
pub(crate) struct Dct2Butterfly15<T: DctSample> {
    bf5: Dct2Butterfly5<T>,
    bf3: Dct2Butterfly3<T>,
}

impl<T: DctSample> Default for Dct2Butterfly15<T>
where
    f64: AsPrimitive<T>,
    usize: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            bf5: Dct2Butterfly5::default(),
            bf3: Dct2Butterfly3::default(),
        }
    }
}

impl<T: DctSample> Dct2Butterfly15<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        let mut c0 = [data[0], data[10], data[9]];
        let mut c1 = [data[11], data[1], data[8]];
        let mut c2 = [data[12], data[7], data[2]];
        let mut c3 = [data[6], data[13], data[3]];
        let mut c4 = [data[5], data[4], data[14]];

        self.bf3.exec(&mut InPlaceStore::new(&mut c0));
        self.bf3.exec(&mut InPlaceStore::new(&mut c1));
        self.bf3.exec(&mut InPlaceStore::new(&mut c2));
        self.bf3.exec(&mut InPlaceStore::new(&mut c3));
        self.bf3.exec(&mut InPlaceStore::new(&mut c4));

        let mut rows0 = [c0[0], c1[0], c2[0], c3[0], c4[0]];
        let mut rows1 = [c0[1], c1[1], c2[1], c3[1], c4[1]];
        let mut rows2 = [c0[2], c1[2], c2[2], c3[2], c4[2]];

        self.bf5.exec(&mut InPlaceStore::new(&mut rows0));
        self.bf5.exec(&mut InPlaceStore::new(&mut rows1));
        self.bf5.exec(&mut InPlaceStore::new(&mut rows2));

        data[0] = rows0[0];
        data[1] = rows1[2] + rows2[3];
        data[2] = rows1[1] + rows2[4];
        data[3] = rows0[1];
        data[4] = rows1[3] + rows2[2];
        data[5] = rows1[0];
        data[6] = rows0[2];
        data[7] = rows2[1] + rows1[4];
        data[8] = rows1[1] - rows2[4];
        data[9] = rows0[3];
        data[10] = rows2[0];
        data[11] = rows1[2] - rows2[3];
        data[12] = rows0[4];
        data[13] = rows2[1] - rows1[4];
        data[14] = rows1[3] - rows2[2];
    }
}

define_in_place_butterfly!(Dct2Butterfly15, 15);

#[derive(Debug, Clone)]
pub(crate) struct Dct2Butterfly18<T: DctSample> {
    bf9: Dct2Butterfly9<T>,
    bf2: Dct2Butterfly2<T>,
}

impl<T: DctSample> Default for Dct2Butterfly18<T>
where
    f64: AsPrimitive<T>,
    usize: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            bf9: Dct2Butterfly9::default(),
            bf2: Dct2Butterfly2::default(),
        }
    }
}

impl<T: DctSample> Dct2Butterfly18<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        let mut c0 = [data[0], data[17]];
        let mut c1 = [data[16], data[1]];
        let mut c2 = [data[15], data[2]];
        let mut c3 = [data[3], data[14]];

        self.bf2.exec(&mut InPlaceStore::new(&mut c0));
        self.bf2.exec(&mut InPlaceStore::new(&mut c1));
        self.bf2.exec(&mut InPlaceStore::new(&mut c2));
        self.bf2.exec(&mut InPlaceStore::new(&mut c3));

        let mut c4 = [data[4], data[13]];
        let mut c5 = [data[12], data[5]];
        let mut c6 = [data[11], data[6]];
        let mut c7 = [data[7], data[10]];
        let mut c8 = [data[8], data[9]];

        self.bf2.exec(&mut InPlaceStore::new(&mut c4));
        self.bf2.exec(&mut InPlaceStore::new(&mut c5));
        self.bf2.exec(&mut InPlaceStore::new(&mut c6));
        self.bf2.exec(&mut InPlaceStore::new(&mut c7));
        self.bf2.exec(&mut InPlaceStore::new(&mut c8));

        let mut rows0 = [
            c0[0], c1[0], c2[0], c3[0], c4[0], c5[0], c6[0], c7[0], c8[0],
        ];

        self.bf9.execute(&mut rows0).unwrap();

        let mut rows1 = [
            c0[1], c1[1], c2[1], c3[1], c4[1], c5[1], c6[1], c7[1], c8[1],
        ];

        self.bf9.execute(&mut rows1).unwrap();

        //
        data[0] = rows0[0];
        data[1] = rows1[4] + rows1[5];
        data[2] = rows0[1];
        data[3] = rows1[3] + rows1[6];
        data[4] = rows0[2];
        data[5] = rows1[2] + rows1[7];
        data[6] = rows0[3];
        data[7] = rows1[1] + rows1[8];
        data[8] = rows0[4];
        data[9] = rows1[0];
        data[10] = rows0[5];
        data[11] = rows1[1] - rows1[8];
        data[12] = rows0[6];
        data[13] = rows1[2] - rows1[7];
        data[14] = rows0[7];
        data[15] = rows1[3] - rows1[6];
        data[16] = rows0[8];
        data[17] = rows1[4] - rows1[5];
    }
}

define_in_place_butterfly!(Dct2Butterfly18, 18);

#[derive(Debug, Clone)]
pub(crate) struct Dct2Butterfly20<T: DctSample> {
    bf4: Dct2Butterfly4<T>,
    bf5: Dct2Butterfly5<T>,
}

impl<T: DctSample> Default for Dct2Butterfly20<T>
where
    f64: AsPrimitive<T>,
    usize: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            bf4: Dct2Butterfly4::default(),
            bf5: Dct2Butterfly5::default(),
        }
    }
}

impl<T: DctSample> Dct2Butterfly20<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        let mut col0 = [data[0], data[9], data[10], data[19]];
        let mut col1 = [data[8], data[1], data[18], data[11]];
        self.bf4.exec(&mut InPlaceStore::new(&mut col0));
        self.bf4.exec(&mut InPlaceStore::new(&mut col1));
        let mut col2 = [data[7], data[17], data[2], data[12]];
        let mut col3 = [data[16], data[6], data[13], data[3]];
        self.bf4.exec(&mut InPlaceStore::new(&mut col2));
        self.bf4.exec(&mut InPlaceStore::new(&mut col3));
        let mut col4 = [data[15], data[14], data[5], data[4]];
        self.bf4.exec(&mut InPlaceStore::new(&mut col4));
        let mut row0 = [col0[0], col1[0], col2[0], col3[0], col4[0]];
        self.bf5.exec(&mut InPlaceStore::new(&mut row0));
        data[0] = row0[0];
        data[4] = row0[1];
        data[8] = row0[2];
        data[12] = row0[3];
        data[16] = row0[4];
        let mut row1 = [col0[1], col1[1], col2[1], col3[1], col4[1]];
        self.bf5.exec(&mut InPlaceStore::new(&mut row1));
        data[5] = row1[0];
        let mut row2 = [col0[2], col1[2], col2[2], col3[2], col4[2]];
        self.bf5.exec(&mut InPlaceStore::new(&mut row2));
        data[10] = row2[0];
        let mut row3 = [col0[3], col1[3], col2[3], col3[3], col4[3]];
        self.bf5.exec(&mut InPlaceStore::new(&mut row3));
        data[15] = row3[0];

        data[1] = row3[4] + row1[1];
        data[3] = row3[3] + row1[2];
        data[7] = row3[2] + row1[3];
        data[11] = row3[1] + row1[4];
        data[6] = row2[4] + row2[1];
        data[2] = row2[3] + row2[2];
        data[18] = row2[2] - row2[3];
        data[14] = row2[1] - row2[4];
        data[19] = row3[1] - row1[4];
        data[17] = row1[3] - row3[2];
        data[13] = row1[2] - row3[3];
        data[9] = row1[1] - row3[4];
    }
}

define_in_place_butterfly!(Dct2Butterfly20, 20);

#[derive(Debug, Clone)]
pub(crate) struct Dct2Butterfly21<T: DctSample> {
    bf7: Dct2Butterfly7<T>,
    bf3: Dct2Butterfly3<T>,
}

impl<T: DctSample> Default for Dct2Butterfly21<T>
where
    f64: AsPrimitive<T>,
    usize: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            bf7: Dct2Butterfly7::default(),
            bf3: Dct2Butterfly3::default(),
        }
    }
}

impl<T: DctSample> Dct2Butterfly21<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        let mut c0 = [data[0], data[13], data[14]];
        let mut c1 = [data[12], data[1], data[15]];

        self.bf3.exec(&mut InPlaceStore::new(&mut c0));
        self.bf3.exec(&mut InPlaceStore::new(&mut c1));

        let mut c2 = [data[11], data[16], data[2]];
        let mut c3 = [data[17], data[10], data[3]];

        self.bf3.exec(&mut InPlaceStore::new(&mut c2));
        self.bf3.exec(&mut InPlaceStore::new(&mut c3));

        let mut c4 = [data[18], data[4], data[9]];
        let mut c5 = [data[5], data[19], data[8]];

        self.bf3.exec(&mut InPlaceStore::new(&mut c4));
        self.bf3.exec(&mut InPlaceStore::new(&mut c5));

        let mut c6 = [data[6], data[7], data[20]];

        self.bf3.exec(&mut InPlaceStore::new(&mut c6));

        let mut rows0 = [c0[0], c1[0], c2[0], c3[0], c4[0], c5[0], c6[0]];

        self.bf7.exec(&mut InPlaceStore::new(&mut rows0));

        let mut rows1 = [c0[1], c1[1], c2[1], c3[1], c4[1], c5[1], c6[1]];

        self.bf7.exec(&mut InPlaceStore::new(&mut rows1));

        let mut rows2 = [c0[2], c1[2], c2[2], c3[2], c4[2], c5[2], c6[2]];

        self.bf7.exec(&mut InPlaceStore::new(&mut rows2));

        data[0] = rows0[0];
        data[1] = rows1[2] + rows2[5];
        data[2] = rows1[3] + rows2[4];
        data[3] = rows0[1];
        data[4] = rows1[1] + rows2[6];
        data[5] = rows1[4] + rows2[3];
        data[6] = rows0[2];
        data[7] = rows1[0];
        data[8] = rows2[2] + rows1[5];
        data[9] = rows0[3];
        data[10] = rows1[1] - rows2[6];
        data[11] = rows2[1] + rows1[6];
        data[12] = rows0[4];
        data[13] = rows1[2] - rows2[5];
        data[14] = rows2[0];
        data[15] = rows0[5];
        data[16] = rows1[3] - rows2[4];
        data[17] = -(rows1[6] - rows2[1]);
        data[18] = rows0[6];
        data[19] = rows1[4] - rows2[3];
        data[20] = -(rows1[5] - rows2[2]);
    }
}

define_in_place_butterfly!(Dct2Butterfly21, 21);

#[derive(Debug, Clone)]
pub(crate) struct Dct2Butterfly22<T: DctSample> {
    bf11: Dct2Butterfly11<T>,
    bf2: Dct2Butterfly2<T>,
}

impl<T: DctSample + MixedRadix11Sample> Default for Dct2Butterfly22<T>
where
    f64: AsPrimitive<T>,
    usize: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            bf11: Dct2Butterfly11::default(),
            bf2: Dct2Butterfly2::default(),
        }
    }
}

impl<T: DctSample + MixedRadix11Sample> Dct2Butterfly22<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        let mut col0 = [data[0], data[21]];
        let mut col1 = [data[20], data[1]];
        let mut col2 = [data[19], data[2]];
        let mut col3 = [data[3], data[18]];

        self.bf2.exec(&mut InPlaceStore::new(&mut col0));
        self.bf2.exec(&mut InPlaceStore::new(&mut col1));
        self.bf2.exec(&mut InPlaceStore::new(&mut col2));
        self.bf2.exec(&mut InPlaceStore::new(&mut col3));

        let mut col4 = [data[4], data[17]];
        let mut col5 = [data[16], data[5]];
        let mut col6 = [data[15], data[6]];
        let mut col7 = [data[7], data[14]];

        self.bf2.exec(&mut InPlaceStore::new(&mut col4));
        self.bf2.exec(&mut InPlaceStore::new(&mut col5));
        self.bf2.exec(&mut InPlaceStore::new(&mut col6));
        self.bf2.exec(&mut InPlaceStore::new(&mut col7));

        let mut col8 = [data[8], data[13]];
        let mut col9 = [data[12], data[9]];
        let mut col10 = [data[11], data[10]];

        self.bf2.exec(&mut InPlaceStore::new(&mut col8));
        self.bf2.exec(&mut InPlaceStore::new(&mut col9));
        self.bf2.exec(&mut InPlaceStore::new(&mut col10));

        let mut row0 = [
            col0[0], col1[0], col2[0], col3[0], col4[0], col5[0], col6[0], col7[0], col8[0],
            col9[0], col10[0],
        ];

        self.bf11.exec(&mut InPlaceStore::new(&mut row0));

        data[0] = row0[0];
        data[2] = row0[1];
        data[4] = row0[2];
        data[6] = row0[3];
        data[8] = row0[4];
        data[10] = row0[5];
        data[12] = row0[6];
        data[14] = row0[7];
        data[16] = row0[8];
        data[18] = row0[9];
        data[20] = row0[10];

        let mut row1 = [
            col0[1], col1[1], col2[1], col3[1], col4[1], col5[1], col6[1], col7[1], col8[1],
            col9[1], col10[1],
        ];

        self.bf11.exec(&mut InPlaceStore::new(&mut row1));

        data[11] = row1[0];
        data[9] = row1[1] + row1[10];
        data[7] = row1[2] + row1[9];
        data[5] = row1[3] + row1[8];
        data[3] = row1[4] + row1[7];
        data[1] = row1[5] + row1[6];
        data[21] = row1[5] - row1[6];
        data[19] = row1[4] - row1[7];
        data[17] = row1[3] - row1[8];
        data[15] = row1[2] - row1[9];
        data[13] = row1[1] - row1[10];
    }
}

impl<T: DctSample + MixedRadix11Sample> PxdctExecutor<T> for Dct2Butterfly22<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, data: &mut [T]) -> Result<(), PxdctError> {
        if !data.len().is_multiple_of(22) {
            return Err(PxdctError::InvalidSizeMultiplier(data.len(), self.length()));
        }
        for chunk in data.chunks_exact_mut(22) {
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
        validate_oof_sizes!(input, output, 22);
        use crate::bidirectional::BiStore;
        for (src, dst) in input.chunks_exact(22).zip(output.chunks_exact_mut(22)) {
            self.exec(&mut BiStore::new(src, dst));
        }
        Ok(())
    }

    fn length(&self) -> usize {
        22
    }

    fn scratch_size(&self) -> usize {
        0
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Dct2Butterfly24<T: DctSample> {
    bf3: Dct2Butterfly3<T>,
    bf8: Dct2Butterfly8<T>,
}

impl<T: DctSample> Default for Dct2Butterfly24<T>
where
    f64: AsPrimitive<T>,
    usize: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            bf3: Dct2Butterfly3::default(),
            bf8: Dct2Butterfly8::default(),
        }
    }
}

impl<T: DctSample> Dct2Butterfly24<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        let mut col0 = [data[0], data[16], data[15]];
        let mut col1 = [data[17], data[1], data[14]];

        self.bf3.exec(&mut InPlaceStore::new(&mut col0));
        self.bf3.exec(&mut InPlaceStore::new(&mut col1));

        let mut col2 = [data[18], data[13], data[2]];
        let mut col3 = [data[12], data[19], data[3]];

        self.bf3.exec(&mut InPlaceStore::new(&mut col2));
        self.bf3.exec(&mut InPlaceStore::new(&mut col3));

        let mut col4 = [data[11], data[4], data[20]];
        let mut col5 = [data[5], data[10], data[21]];

        self.bf3.exec(&mut InPlaceStore::new(&mut col4));
        self.bf3.exec(&mut InPlaceStore::new(&mut col5));

        let mut col6 = [data[6], data[22], data[9]];
        let mut col7 = [data[23], data[7], data[8]];

        self.bf3.exec(&mut InPlaceStore::new(&mut col6));
        self.bf3.exec(&mut InPlaceStore::new(&mut col7));

        let mut row0 = [
            col0[0], col1[0], col2[0], col3[0], col4[0], col5[0], col6[0], col7[0],
        ];

        self.bf8.exec(&mut InPlaceStore::new(&mut row0));

        data[0] = row0[0];
        data[3] = row0[1];
        data[6] = row0[2];
        data[9] = row0[3];
        data[12] = row0[4];
        data[15] = row0[5];
        data[18] = row0[6];
        data[21] = row0[7];

        let mut row1 = [
            col0[1], col1[1], col2[1], col3[1], col4[1], col5[1], col6[1], col7[1],
        ];

        self.bf8.exec(&mut InPlaceStore::new(&mut row1));

        data[8] = row1[0];

        let mut row2 = [
            col0[2], col1[2], col2[2], col3[2], col4[2], col5[2], col6[2], col7[2],
        ];

        self.bf8.exec(&mut InPlaceStore::new(&mut row2));

        data[5] = row1[1] + row2[7];
        data[2] = row1[2] + row2[6];
        data[1] = row1[3] + row2[5];
        data[4] = row1[4] + row2[4];
        data[7] = row1[5] + row2[3];
        data[10] = row1[6] + row2[2];
        data[13] = row1[7] + row2[1];
        data[16] = row2[0];
        data[19] = row2[1] - row1[7];
        data[22] = row2[2] - row1[6];
        data[23] = row1[5] - row2[3];
        data[20] = row1[4] - row2[4];
        data[17] = row1[3] - row2[5];
        data[14] = row1[2] - row2[6];
        data[11] = row1[1] - row2[7];
    }
}

define_in_place_butterfly!(Dct2Butterfly24, 24);

#[derive(Debug, Clone)]
pub(crate) struct Dct2Butterfly26<T: DctSample> {
    bf13: Dct2Butterfly13<T>,
    bf2: Dct2Butterfly2<T>,
}

impl<T: DctSample> Default for Dct2Butterfly26<T>
where
    f64: AsPrimitive<T>,
    usize: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            bf13: Dct2Butterfly13::default(),
            bf2: Dct2Butterfly2::default(),
        }
    }
}

impl<T: DctSample> Dct2Butterfly26<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        let mut col0 = [data[0], data[25]];
        let mut col1 = [data[24], data[1]];
        let mut col2 = [data[23], data[2]];
        let mut col3 = [data[3], data[22]];

        self.bf2.exec(&mut InPlaceStore::new(&mut col0));
        self.bf2.exec(&mut InPlaceStore::new(&mut col1));
        self.bf2.exec(&mut InPlaceStore::new(&mut col2));
        self.bf2.exec(&mut InPlaceStore::new(&mut col3));

        let mut col4 = [data[4], data[21]];
        let mut col5 = [data[20], data[5]];
        let mut col6 = [data[19], data[6]];
        let mut col7 = [data[7], data[18]];

        self.bf2.exec(&mut InPlaceStore::new(&mut col4));
        self.bf2.exec(&mut InPlaceStore::new(&mut col5));
        self.bf2.exec(&mut InPlaceStore::new(&mut col6));
        self.bf2.exec(&mut InPlaceStore::new(&mut col7));

        let mut col8 = [data[8], data[17]];
        let mut col9 = [data[16], data[9]];
        let mut col10 = [data[15], data[10]];
        let mut col11 = [data[11], data[14]];
        let mut col12 = [data[12], data[13]];

        self.bf2.exec(&mut InPlaceStore::new(&mut col8));
        self.bf2.exec(&mut InPlaceStore::new(&mut col9));
        self.bf2.exec(&mut InPlaceStore::new(&mut col10));
        self.bf2.exec(&mut InPlaceStore::new(&mut col11));
        self.bf2.exec(&mut InPlaceStore::new(&mut col12));

        let mut row0 = [
            col0[0], col1[0], col2[0], col3[0], col4[0], col5[0], col6[0], col7[0], col8[0],
            col9[0], col10[0], col11[0], col12[0],
        ];
        self.bf13.exec(&mut InPlaceStore::new(&mut row0));

        data[0] = row0[0];
        data[2] = row0[1];
        data[4] = row0[2];
        data[6] = row0[3];
        data[8] = row0[4];
        data[10] = row0[5];
        data[12] = row0[6];
        data[14] = row0[7];
        data[16] = row0[8];
        data[18] = row0[9];
        data[20] = row0[10];
        data[22] = row0[11];
        data[24] = row0[12];
        let mut row1 = [
            col0[1], col1[1], col2[1], col3[1], col4[1], col5[1], col6[1], col7[1], col8[1],
            col9[1], col10[1], col11[1], col12[1],
        ];
        self.bf13.exec(&mut InPlaceStore::new(&mut row1));
        data[13] = row1[0];
        data[11] = row1[12] + row1[1];
        data[9] = row1[11] + row1[2];
        data[7] = row1[10] + row1[3];
        data[5] = row1[9] + row1[4];
        data[3] = row1[8] + row1[5];
        data[1] = row1[7] + row1[6];
        data[25] = row1[6] - row1[7];
        data[23] = row1[5] - row1[8];
        data[21] = row1[4] - row1[9];
        data[19] = row1[3] - row1[10];
        data[17] = row1[2] - row1[11];
        data[15] = row1[1] - row1[12];
    }
}

define_in_place_butterfly!(Dct2Butterfly26, 26);

#[derive(Debug, Clone)]
pub(crate) struct Dct2Butterfly27<T: DctSample> {
    bf9: Dct2Butterfly9<T>,
    inner_layer: [Complex<T>; 18],
}

impl<T: DctSample> Default for Dct2Butterfly27<T>
where
    f64: AsPrimitive<T>,
    usize: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            bf9: Dct2Butterfly9::default(),
            inner_layer: mixed_radix3_twiddles(27),
        }
    }
}

impl<T: DctSample> Dct2Butterfly27<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<T>>(
        &self,
        data: &mut S,
        a_buffer: &mut [T; 9],
        b_buffer: &mut [T; 9],
        c_buffer: &mut [T; 9],
    ) {
        for i in 0..9 {
            let ai = data[i];
            let bi = data[18 + i];
            let ci = data[18 - i - 1];

            let cos_sin_ai = self.inner_layer[i * 2];
            let cos_sin_2ai = self.inner_layer[i * 2 + 1];

            let bici = bi + ci;
            let a_comp = ai + bici;
            let second_layer_comp0 = fmla(2f64.as_(), ai, -bici);

            let d_ci_bi = ci - bi;

            let b0_b = fmla(second_layer_comp0, cos_sin_ai.re, d_ci_bi * cos_sin_ai.im);
            let c0_b = fmla(second_layer_comp0, cos_sin_2ai.re, d_ci_bi * cos_sin_2ai.im);

            a_buffer[i] = a_comp;
            b_buffer[i] = b0_b;
            c_buffer[i] = c0_b;
        }

        self.bf9.exec(&mut InPlaceStore::new(a_buffer));
        self.bf9.exec(&mut InPlaceStore::new(b_buffer));
        self.bf9.exec(&mut InPlaceStore::new(c_buffer));

        data[0] = a_buffer[0];
        let b0 = b_buffer[0] * T::HALF;
        data[1] = b0;
        let c0 = c_buffer[0] * T::HALF;
        data[2] = c0;

        let mut last_b = c0;
        let mut last_c = b0;

        for k in 1..9 {
            data[3 * k] = a_buffer[k];

            let deferred_c = b_buffer[k] - last_b;
            data[3 * k + 1] = deferred_c;

            last_b = c_buffer[k] - last_c;
            data[3 * k + 2] = last_b;
            last_c = deferred_c;
        }
    }
}

impl<T: DctSample> PxdctExecutor<T> for Dct2Butterfly27<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, data: &mut [T]) -> Result<(), PxdctError> {
        if !data.len().is_multiple_of(27) {
            return Err(PxdctError::InvalidSizeMultiplier(data.len(), self.length()));
        }

        let mut a_buffer = [T::zero(); 9];
        let mut b_buffer = [T::zero(); 9];
        let mut c_buffer = [T::zero(); 9];

        for chunk in data.chunks_exact_mut(27) {
            self.exec(
                &mut InPlaceStore::new(chunk),
                &mut a_buffer,
                &mut b_buffer,
                &mut c_buffer,
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
        validate_oof_sizes!(input, output, 27);

        let mut a_buffer = [T::zero(); 9];
        let mut b_buffer = [T::zero(); 9];
        let mut c_buffer = [T::zero(); 9];

        use crate::bidirectional::BiStore;
        for (src, dst) in input.chunks_exact(27).zip(output.chunks_exact_mut(27)) {
            self.exec(
                &mut BiStore::new(src, dst),
                &mut a_buffer,
                &mut b_buffer,
                &mut c_buffer,
            );
        }
        Ok(())
    }

    fn length(&self) -> usize {
        27
    }

    fn scratch_size(&self) -> usize {
        27
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Dct2Butterfly30<T: DctSample> {
    bf6: Dct2Butterfly6<T>,
    bf5: Dct2Butterfly5<T>,
}

impl<T: DctSample> Default for Dct2Butterfly30<T>
where
    f64: AsPrimitive<T>,
    usize: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            bf6: Dct2Butterfly6::default(),
            bf5: Dct2Butterfly5::default(),
        }
    }
}

impl<T: DctSample> Dct2Butterfly30<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        let mut col0 = [data[0], data[11], data[12], data[23], data[24]];
        let mut col1 = [data[10], data[1], data[22], data[13], data[25]];

        self.bf5.exec(&mut InPlaceStore::new(&mut col0));
        self.bf5.exec(&mut InPlaceStore::new(&mut col1));

        let mut col2 = [data[9], data[21], data[2], data[26], data[14]];
        let mut col3 = [data[20], data[8], data[27], data[3], data[15]];

        self.bf5.exec(&mut InPlaceStore::new(&mut col2));
        self.bf5.exec(&mut InPlaceStore::new(&mut col3));

        let mut col4 = [data[19], data[28], data[7], data[16], data[4]];
        let mut col5 = [data[29], data[18], data[17], data[6], data[5]];

        self.bf5.exec(&mut InPlaceStore::new(&mut col4));
        self.bf5.exec(&mut InPlaceStore::new(&mut col5));

        let mut row0 = [col0[0], col1[0], col2[0], col3[0], col4[0], col5[0]];

        self.bf6.exec(&mut InPlaceStore::new(&mut row0));

        data[0] = row0[0];
        data[5] = row0[1];
        data[10] = row0[2];
        data[15] = row0[3];
        data[20] = row0[4];
        data[25] = row0[5];

        let mut row1 = [col0[1], col1[1], col2[1], col3[1], col4[1], col5[1]];

        self.bf6.exec(&mut InPlaceStore::new(&mut row1));
        data[6] = row1[0];

        let mut row2 = [col0[2], col1[2], col2[2], col3[2], col4[2], col5[2]];

        self.bf6.exec(&mut InPlaceStore::new(&mut row2));

        data[12] = row2[0];

        let mut row3 = [col0[3], col1[3], col2[3], col3[3], col4[3], col5[3]];

        self.bf6.exec(&mut InPlaceStore::new(&mut row3));
        data[18] = row3[0];

        let mut row4 = [col0[4], col1[4], col2[4], col3[4], col4[4], col5[4]];

        self.bf6.exec(&mut InPlaceStore::new(&mut row4));
        data[24] = row4[0];

        data[1] = row1[1] + row4[5];
        data[4] = row1[2] + row4[4];
        data[9] = row1[3] + row4[3];
        data[14] = row1[4] + row4[2];
        data[19] = row1[5] + row4[1];
        data[7] = row2[1] + row3[5];
        data[2] = row2[2] + row3[4];
        data[3] = row2[3] + row3[3];
        data[8] = row2[4] + row3[2];
        data[13] = row2[5] + row3[1];
        data[23] = row3[1] - row2[5];
        data[28] = row3[2] - row2[4];
        data[27] = row2[3] - row3[3];
        data[22] = row2[2] - row3[4];
        data[17] = row2[1] - row3[5];
        data[29] = row4[1] - row1[5];
        data[26] = row1[4] - row4[2];
        data[21] = row1[3] - row4[3];
        data[16] = row1[2] - row4[4];
        data[11] = row1[1] - row4[5];
    }
}

define_in_place_butterfly!(Dct2Butterfly30, 30);

#[derive(Debug, Clone)]
pub(crate) struct Dct2Butterfly35<T: DctSample> {
    bf7: Dct2Butterfly7<T>,
    bf5: Dct2Butterfly5<T>,
}

impl<T: DctSample> Default for Dct2Butterfly35<T>
where
    f64: AsPrimitive<T>,
    usize: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            bf7: Dct2Butterfly7::default(),
            bf5: Dct2Butterfly5::default(),
        }
    }
}

impl<T: DctSample> Dct2Butterfly35<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        let mut col0 = [data[0], data[28], data[27], data[13], data[14]];
        self.bf5.exec(&mut InPlaceStore::new(&mut col0));
        let mut col1 = [data[29], data[1], data[12], data[26], data[15]];
        self.bf5.exec(&mut InPlaceStore::new(&mut col1));
        let mut col2 = [data[30], data[11], data[2], data[16], data[25]];
        self.bf5.exec(&mut InPlaceStore::new(&mut col2));
        let mut col3 = [data[10], data[31], data[17], data[3], data[24]];
        self.bf5.exec(&mut InPlaceStore::new(&mut col3));
        let mut col4 = [data[9], data[18], data[32], data[23], data[4]];
        self.bf5.exec(&mut InPlaceStore::new(&mut col4));
        let mut col5 = [data[19], data[8], data[22], data[33], data[5]];
        self.bf5.exec(&mut InPlaceStore::new(&mut col5));
        let mut col6 = [data[20], data[21], data[7], data[6], data[34]];
        self.bf5.exec(&mut InPlaceStore::new(&mut col6));
        let mut row0 = [
            col0[0], col1[0], col2[0], col3[0], col4[0], col5[0], col6[0],
        ];
        self.bf7.exec(&mut InPlaceStore::new(&mut row0));

        data[0] = row0[0];
        data[5] = row0[1];
        data[10] = row0[2];
        data[15] = row0[3];
        data[20] = row0[4];
        data[25] = row0[5];
        data[30] = row0[6];

        let mut row1 = [
            col0[1], col1[1], col2[1], col3[1], col4[1], col5[1], col6[1],
        ];
        self.bf7.exec(&mut InPlaceStore::new(&mut row1));
        data[7] = row1[0];
        let mut row2 = [
            col0[2], col1[2], col2[2], col3[2], col4[2], col5[2], col6[2],
        ];
        self.bf7.exec(&mut InPlaceStore::new(&mut row2));
        data[14] = row2[0];
        let mut row3 = [
            col0[3], col1[3], col2[3], col3[3], col4[3], col5[3], col6[3],
        ];
        self.bf7.exec(&mut InPlaceStore::new(&mut row3));
        data[21] = row3[0];
        let mut row4 = [
            col0[4], col1[4], col2[4], col3[4], col4[4], col5[4], col6[4],
        ];
        self.bf7.exec(&mut InPlaceStore::new(&mut row4));
        data[28] = row4[0];

        data[2] = row1[1] + row4[6];
        data[3] = row1[2] + row4[5];
        data[8] = row1[3] + row4[4];
        data[13] = row1[4] + row4[3];
        data[18] = row1[5] + row4[2];
        data[23] = row1[6] + row4[1];
        data[9] = row2[1] + row3[6];
        data[4] = row2[2] + row3[5];
        data[1] = row2[3] + row3[4];
        data[6] = row2[4] + row3[3];
        data[11] = row2[5] + row3[2];
        data[16] = row2[6] + row3[1];
        data[26] = row3[1] - row2[6];
        data[31] = row3[2] - row2[5];
        data[34] = row2[4] - row3[3];
        data[29] = row2[3] - row3[4];
        data[24] = row2[2] - row3[5];
        data[19] = row2[1] - row3[6];
        data[33] = row4[1] - row1[6];
        data[32] = row1[5] - row4[2];
        data[27] = row1[4] - row4[3];
        data[22] = row1[3] - row4[4];
        data[17] = row1[2] - row4[5];
        data[12] = row1[1] - row4[6];
    }
}

define_in_place_butterfly!(Dct2Butterfly35, 35);

#[allow(unused)]
#[derive(Debug, Clone)]
pub(crate) struct Dct2Butterfly36<T: DctSample> {
    inner_layer: [Complex<T>; 24],
    bf6: Dct2Butterfly6<T>,
}

impl<T: DctSample> Default for Dct2Butterfly36<T>
where
    f64: AsPrimitive<T>,
    usize: AsPrimitive<T>,
{
    fn default() -> Self {
        let mut inner_layer = [Complex::<T>::default(); 6 * 4];
        for (i, layer) in inner_layer.chunks_exact_mut(4).enumerate() {
            let angle = (2. * i as f64 + 1.).as_();
            layer[0] = mixed_radix_inner_twiddle(angle, 36);
            layer[0].im *= T::SQRT_3;
            layer[1] = mixed_radix_inner_twiddle(2f64.as_() * angle, 36);
            layer[1].im *= T::SQRT_3;
            layer[2] = mixed_radix_inner_twiddle(3f64.as_() * angle, 36);
            layer[2].im *= T::SQRT_3;
            layer[3] = mixed_radix_inner_twiddle(5f64.as_() * angle, 36);
            layer[3].im = -layer[3].im * T::SQRT_3;
        }

        Self {
            inner_layer,
            bf6: Dct2Butterfly6::default(),
        }
    }
}

impl<T: DctSample> Dct2Butterfly36<T>
where
    f64: AsPrimitive<T>,
{
    #[allow(unused)]
    #[inline(always)]
    fn exec<S: BidirectionalStore<T>>(
        &self,
        data: &mut S,
        a_buffer: &mut [T; 6],
        b_buffer: &mut [T; 6],
        c_buffer: &mut [T; 6],
        d_buffer: &mut [T; 6],
        e_buffer: &mut [T; 6],
        f_buffer: &mut [T; 6],
    ) {
        let s_n = 36 / 3;
        let s_2n = 2 * 36 / 3;

        for i in 0..6 {
            let ai = data[i];
            let bi = data[s_n - i - 1];
            let ci = data[s_n + i];
            let di = data[s_2n - i - 1];
            let ei = data[s_2n + i];
            let fi = data[36 - i - 1];

            let cos_sin_ai = self.inner_layer[i * 4];
            let cos_sin_2ai = self.inner_layer[i * 4 + 1];
            let cos_sin_3ai = self.inner_layer[i * 4 + 2];
            let cos_sin_5ai = self.inner_layer[i * 4 + 3];

            let s2 = bi + ei;
            let dcd = ci - di;
            let dbe = bi - ei;

            let ai2 = T::TWO * ai;
            let fi2 = T::TWO * fi;
            let scd = ci + di;

            let sdbedcd = dbe + dcd;
            let ai2dbedcd = ai2 + sdbedcd - fi2;

            let s2scd = s2 + scd;

            let a_comp = ai + s2scd + fi;
            let c_comp = ai2 - s2scd + fi2;
            let d_comp = T::TWO * (ai - sdbedcd - fi);

            let dbedcd = dbe - dcd;

            let c_img = s2 - scd;
            let b_zet = dbedcd * cos_sin_ai.im;
            let c_zet = c_img * cos_sin_2ai.im;
            let f_zet = dbedcd * cos_sin_5ai.im;

            let e_comp = fmla(
                T::TWO * cos_sin_2ai.re,
                fmla(c_comp, cos_sin_2ai.re, -c_zet),
                -c_comp,
            );

            a_buffer[i] = a_comp;
            b_buffer[i] = fmla(ai2dbedcd, cos_sin_ai.re, b_zet);
            c_buffer[i] = fmla(c_comp, cos_sin_2ai.re, c_zet);
            d_buffer[i] = d_comp * cos_sin_3ai.re;
            e_buffer[i] = e_comp;
            f_buffer[i] = fmla(ai2dbedcd, cos_sin_5ai.re, f_zet);
        }

        self.bf6.exec(&mut InPlaceStore::new(a_buffer));
        self.bf6.exec(&mut InPlaceStore::new(b_buffer));
        self.bf6.exec(&mut InPlaceStore::new(c_buffer));
        self.bf6.exec(&mut InPlaceStore::new(d_buffer));
        self.bf6.exec(&mut InPlaceStore::new(e_buffer));
        self.bf6.exec(&mut InPlaceStore::new(f_buffer));

        data[0] = a_buffer[0];
        let b0 = b_buffer[0] * T::HALF;
        data[1] = b0;
        let c0 = c_buffer[0] * T::HALF;
        data[2] = c0;
        let d0 = d_buffer[0] * T::HALF;
        data[3] = d0;
        let e0 = e_buffer[0] * T::HALF;
        data[4] = e0;
        let f0 = f_buffer[0] * T::HALF;
        data[5] = f0;

        let mut b_diff = f0;
        let mut c_diff = e0;
        let mut e_diff = d0;
        let mut d_diff = c0;
        let mut f_diff = b0;

        for k in 1..6 {
            data[6 * k] = a_buffer[k];
            let deferred_f_diff = b_buffer[k] - b_diff;
            data[6 * k + 1] = deferred_f_diff;
            let deferred_d_diff = c_buffer[k] - c_diff;
            data[6 * k + 2] = deferred_d_diff;
            e_diff = d_buffer[k] - e_diff;
            data[6 * k + 3] = e_diff;
            let new_d = e_buffer[k] - d_diff;
            data[6 * k + 4] = new_d;
            c_diff = new_d;
            d_diff = deferred_d_diff;
            let new_f = f_buffer[k] - f_diff;
            b_diff = new_f;
            f_diff = deferred_f_diff;
            data[6 * k + 5] = new_f;
        }
    }
}

impl<T: DctSample> PxdctExecutor<T> for Dct2Butterfly36<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, data: &mut [T]) -> Result<(), PxdctError> {
        if !data.len().is_multiple_of(36) {
            return Err(PxdctError::InvalidSizeMultiplier(data.len(), self.length()));
        }

        let mut a_buffer = [T::default(); 6];
        let mut b_buffer = [T::default(); 6];
        let mut c_buffer = [T::default(); 6];
        let mut d_buffer = [T::default(); 6];
        let mut e_buffer = [T::default(); 6];
        let mut f_buffer = [T::default(); 6];

        for chunk in data.chunks_exact_mut(36) {
            self.exec(
                &mut InPlaceStore::new(chunk),
                &mut a_buffer,
                &mut b_buffer,
                &mut c_buffer,
                &mut d_buffer,
                &mut e_buffer,
                &mut f_buffer,
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
        validate_oof_sizes!(input, output, 36);

        let mut a_buffer = [T::default(); 6];
        let mut b_buffer = [T::default(); 6];
        let mut c_buffer = [T::default(); 6];
        let mut d_buffer = [T::default(); 6];
        let mut e_buffer = [T::default(); 6];
        let mut f_buffer = [T::default(); 6];

        use crate::bidirectional::BiStore;
        for (src, dst) in input.chunks_exact(36).zip(output.chunks_exact_mut(36)) {
            self.exec(
                &mut BiStore::new(src, dst),
                &mut a_buffer,
                &mut b_buffer,
                &mut c_buffer,
                &mut d_buffer,
                &mut e_buffer,
                &mut f_buffer,
            );
        }
        Ok(())
    }

    fn length(&self) -> usize {
        36
    }

    fn scratch_size(&self) -> usize {
        0
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Dct2Butterfly42<T: DctSample> {
    bf7: Dct2Butterfly7<T>,
    bf6: Dct2Butterfly6<T>,
}

impl<T: DctSample> Default for Dct2Butterfly42<T>
where
    f64: AsPrimitive<T>,
    usize: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            bf7: Dct2Butterfly7::default(),
            bf6: Dct2Butterfly6::default(),
        }
    }
}

impl<T: DctSample + MixedRadix7Sample> Dct2Butterfly42<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        let mut col0 = [data[0], data[13], data[14], data[27], data[28], data[41]];
        self.bf6.exec(&mut InPlaceStore::new(&mut col0));
        let mut col1 = [data[12], data[1], data[26], data[15], data[40], data[29]];
        self.bf6.exec(&mut InPlaceStore::new(&mut col1));
        let mut col2 = [data[11], data[25], data[2], data[39], data[16], data[30]];
        self.bf6.exec(&mut InPlaceStore::new(&mut col2));
        let mut col3 = [data[24], data[10], data[38], data[3], data[31], data[17]];
        self.bf6.exec(&mut InPlaceStore::new(&mut col3));
        let mut col4 = [data[23], data[37], data[9], data[32], data[4], data[18]];
        self.bf6.exec(&mut InPlaceStore::new(&mut col4));
        let mut col5 = [data[36], data[22], data[33], data[8], data[19], data[5]];
        self.bf6.exec(&mut InPlaceStore::new(&mut col5));
        let mut col6 = [data[35], data[34], data[21], data[20], data[7], data[6]];
        self.bf6.exec(&mut InPlaceStore::new(&mut col6));
        let mut row0 = [
            col0[0], col1[0], col2[0], col3[0], col4[0], col5[0], col6[0],
        ];
        self.bf7.exec(&mut InPlaceStore::new(&mut row0));
        data[0] = row0[0];
        data[6] = row0[1];
        data[12] = row0[2];
        data[18] = row0[3];
        data[24] = row0[4];
        data[30] = row0[5];
        data[36] = row0[6];
        let mut row1 = [
            col0[1], col1[1], col2[1], col3[1], col4[1], col5[1], col6[1],
        ];
        self.bf7.exec(&mut InPlaceStore::new(&mut row1));
        data[7] = row1[0];
        let mut row2 = [
            col0[2], col1[2], col2[2], col3[2], col4[2], col5[2], col6[2],
        ];
        self.bf7.exec(&mut InPlaceStore::new(&mut row2));
        data[14] = row2[0];
        let mut row3 = [
            col0[3], col1[3], col2[3], col3[3], col4[3], col5[3], col6[3],
        ];
        self.bf7.exec(&mut InPlaceStore::new(&mut row3));
        data[21] = row3[0];
        let mut row4 = [
            col0[4], col1[4], col2[4], col3[4], col4[4], col5[4], col6[4],
        ];
        self.bf7.exec(&mut InPlaceStore::new(&mut row4));
        data[28] = row4[0];
        let mut row5 = [
            col0[5], col1[5], col2[5], col3[5], col4[5], col5[5], col6[5],
        ];
        self.bf7.exec(&mut InPlaceStore::new(&mut row5));
        data[35] = row5[0];

        data[1] = row5[6] + row1[1];
        data[5] = row5[5] + row1[2];
        data[11] = row5[4] + row1[3];
        data[17] = row5[3] + row1[4];
        data[23] = row5[2] + row1[5];
        data[29] = row5[1] + row1[6];
        data[8] = row4[6] + row2[1];
        data[2] = row4[5] + row2[2];
        data[4] = row4[4] + row2[3];
        data[10] = row4[3] + row2[4];
        data[16] = row4[2] + row2[5];
        data[22] = row4[1] + row2[6];
        data[15] = row3[6] + row3[1];
        data[9] = row3[5] + row3[2];
        data[3] = row3[4] + row3[3];
        data[39] = row3[3] - row3[4];
        data[33] = row3[2] - row3[5];
        data[27] = row3[1] - row3[6];
        data[34] = row4[1] - row2[6];
        data[40] = row4[2] - row2[5];
        data[38] = row2[4] - row4[3];
        data[32] = row2[3] - row4[4];
        data[26] = row2[2] - row4[5];
        data[20] = row2[1] - row4[6];
        data[41] = row5[1] - row1[6];
        data[37] = row1[5] - row5[2];
        data[31] = row1[4] - row5[3];
        data[25] = row1[3] - row5[4];
        data[19] = row1[2] - row5[5];
        data[13] = row1[1] - row5[6];
    }
}

define_in_place_butterfly!(Dct2Butterfly42, 42);

#[derive(Debug, Clone)]
pub(crate) struct Dct2Butterfly48<T: DctSample> {
    bf16: Dct2Butterfly16<T>,
    bf3: Dct2Butterfly3<T>,
}

impl<T: DctSample> Default for Dct2Butterfly48<T>
where
    f64: AsPrimitive<T>,
    usize: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            bf16: Dct2Butterfly16::default(),
            bf3: Dct2Butterfly3::default(),
        }
    }
}

impl<T: DctSample> Dct2Butterfly48<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    fn exec<S: BidirectionalStore<T>>(&self, data: &mut S) {
        let mut col0 = [data[0], data[31], data[32]];
        let mut col1 = [data[30], data[1], data[33]];
        self.bf3.exec(&mut InPlaceStore::new(&mut col0));
        self.bf3.exec(&mut InPlaceStore::new(&mut col1));
        let mut col2 = [data[29], data[34], data[2]];
        let mut col3 = [data[35], data[28], data[3]];
        self.bf3.exec(&mut InPlaceStore::new(&mut col2));
        self.bf3.exec(&mut InPlaceStore::new(&mut col3));
        let mut col4 = [data[36], data[4], data[27]];
        let mut col5 = [data[5], data[37], data[26]];
        self.bf3.exec(&mut InPlaceStore::new(&mut col4));
        self.bf3.exec(&mut InPlaceStore::new(&mut col5));
        let mut col6 = [data[6], data[25], data[38]];
        let mut col7 = [data[24], data[7], data[39]];
        self.bf3.exec(&mut InPlaceStore::new(&mut col6));
        self.bf3.exec(&mut InPlaceStore::new(&mut col7));
        let mut col8 = [data[23], data[40], data[8]];
        let mut col9 = [data[41], data[22], data[9]];
        self.bf3.exec(&mut InPlaceStore::new(&mut col8));
        self.bf3.exec(&mut InPlaceStore::new(&mut col9));
        let mut col10 = [data[42], data[10], data[21]];
        let mut col11 = [data[11], data[43], data[20]];
        self.bf3.exec(&mut InPlaceStore::new(&mut col10));
        self.bf3.exec(&mut InPlaceStore::new(&mut col11));
        let mut col12 = [data[12], data[19], data[44]];
        let mut col13 = [data[18], data[13], data[45]];
        self.bf3.exec(&mut InPlaceStore::new(&mut col12));
        self.bf3.exec(&mut InPlaceStore::new(&mut col13));
        let mut col14 = [data[17], data[46], data[14]];
        let mut col15 = [data[47], data[16], data[15]];
        self.bf3.exec(&mut InPlaceStore::new(&mut col14));
        self.bf3.exec(&mut InPlaceStore::new(&mut col15));
        let mut row0 = [
            col0[0], col1[0], col2[0], col3[0], col4[0], col5[0], col6[0], col7[0], col8[0],
            col9[0], col10[0], col11[0], col12[0], col13[0], col14[0], col15[0],
        ];
        self.bf16.exec(&mut InPlaceStore::new(&mut row0));
        data[0] = row0[0];
        data[3] = row0[1];
        data[6] = row0[2];
        data[9] = row0[3];
        data[12] = row0[4];
        data[15] = row0[5];
        data[18] = row0[6];
        data[21] = row0[7];
        data[24] = row0[8];
        data[27] = row0[9];
        data[30] = row0[10];
        data[33] = row0[11];
        data[36] = row0[12];
        data[39] = row0[13];
        data[42] = row0[14];
        data[45] = row0[15];
        let mut row1 = [
            col0[1], col1[1], col2[1], col3[1], col4[1], col5[1], col6[1], col7[1], col8[1],
            col9[1], col10[1], col11[1], col12[1], col13[1], col14[1], col15[1],
        ];
        self.bf16.exec(&mut InPlaceStore::new(&mut row1));
        data[16] = row1[0];
        let mut row2 = [
            col0[2], col1[2], col2[2], col3[2], col4[2], col5[2], col6[2], col7[2], col8[2],
            col9[2], col10[2], col11[2], col12[2], col13[2], col14[2], col15[2],
        ];
        self.bf16.exec(&mut InPlaceStore::new(&mut row2));
        data[32] = row2[0];
        data[13] = row2[15] + row1[1];
        data[10] = row2[14] + row1[2];
        data[7] = row2[13] + row1[3];
        data[4] = row2[12] + row1[4];
        data[1] = row2[11] + row1[5];
        data[2] = row2[10] + row1[6];
        data[5] = row2[9] + row1[7];
        data[8] = row2[8] + row1[8];
        data[11] = row2[7] + row1[9];
        data[14] = row2[6] + row1[10];
        data[17] = row2[5] + row1[11];
        data[20] = row2[4] + row1[12];
        data[23] = row2[3] + row1[13];
        data[26] = row2[2] + row1[14];
        data[29] = row2[1] + row1[15];
        data[35] = row2[1] - row1[15];
        data[38] = row2[2] - row1[14];
        data[41] = row2[3] - row1[13];
        data[44] = row2[4] - row1[12];
        data[47] = row2[5] - row1[11];
        data[46] = row1[10] - row2[6];
        data[43] = row1[9] - row2[7];
        data[40] = row1[8] - row2[8];
        data[37] = row1[7] - row2[9];
        data[34] = row1[6] - row2[10];
        data[31] = row1[5] - row2[11];
        data[28] = row1[4] - row2[12];
        data[25] = row1[3] - row2[13];
        data[22] = row1[2] - row2[14];
        data[19] = row1[1] - row2[15];
    }
}

define_in_place_butterfly!(Dct2Butterfly48, 48);

#[derive(Debug, Clone)]
pub(crate) struct Dct2Butterfly81<T: DctSample> {
    bf27: Dct2Butterfly27<T>,
    inner_layer: [Complex<T>; 54],
}

impl<T: DctSample> Default for Dct2Butterfly81<T>
where
    f64: AsPrimitive<T>,
    usize: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            bf27: Dct2Butterfly27::default(),
            inner_layer: mixed_radix3_twiddles(81),
        }
    }
}

impl<T: DctSample> Dct2Butterfly81<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    fn exec<S: BidirectionalStore<T>>(
        &self,
        data: &mut S,
        a_buffer: &mut [T; 27],
        b_buffer: &mut [T; 27],
        c_buffer: &mut [T; 27],
        a_buffer1: &mut [T; 9],
        b_buffer1: &mut [T; 9],
        c_buffer1: &mut [T; 9],
    ) {
        for i in 0..27 {
            let ai = data[i];
            let bi = data[54 + i];
            let ci = data[54 - i - 1];

            let cos_sin_ai = self.inner_layer[i * 2];
            let cos_sin_2ai = self.inner_layer[i * 2 + 1];

            let a_comp = ai + bi + ci;
            let second_layer_comp0 = fmla(2f64.as_(), ai, -bi - ci);

            let d_ci_bi = ci - bi;

            let b0_b = fmla(second_layer_comp0, cos_sin_ai.re, d_ci_bi * cos_sin_ai.im);
            let c0_b = fmla(second_layer_comp0, cos_sin_2ai.re, d_ci_bi * cos_sin_2ai.im);

            a_buffer[i] = a_comp;
            b_buffer[i] = b0_b;
            c_buffer[i] = c0_b;
        }

        self.bf27.exec(
            &mut InPlaceStore::new(a_buffer),
            a_buffer1,
            b_buffer1,
            c_buffer1,
        );
        self.bf27.exec(
            &mut InPlaceStore::new(b_buffer),
            a_buffer1,
            b_buffer1,
            c_buffer1,
        );
        self.bf27.exec(
            &mut InPlaceStore::new(c_buffer),
            a_buffer1,
            b_buffer1,
            c_buffer1,
        );

        data[0] = a_buffer[0];
        let b0 = b_buffer[0] * T::HALF;
        data[1] = b0;
        let c0 = c_buffer[0] * T::HALF;
        data[2] = c0;

        let mut last_b = c0;
        let mut last_c = b0;

        for k in 1..27 {
            data[3 * k] = a_buffer[k];

            let deferred_c = b_buffer[k] - last_b;
            data[3 * k + 1] = deferred_c;

            last_b = c_buffer[k] - last_c;
            data[3 * k + 2] = last_b;
            last_c = deferred_c;
        }
    }
}

impl<T: DctSample> PxdctExecutor<T> for Dct2Butterfly81<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, data: &mut [T]) -> Result<(), PxdctError> {
        if !data.len().is_multiple_of(81) {
            return Err(PxdctError::InvalidSizeMultiplier(data.len(), self.length()));
        }

        let mut a_buffer = [T::default(); 27];
        let mut b_buffer = [T::default(); 27];
        let mut c_buffer = [T::default(); 27];

        let mut a_buffer1 = [T::default(); 9];
        let mut b_buffer1 = [T::default(); 9];
        let mut c_buffer1 = [T::default(); 9];

        for chunk in data.chunks_exact_mut(81) {
            self.exec(
                &mut InPlaceStore::new(chunk),
                &mut a_buffer,
                &mut b_buffer,
                &mut c_buffer,
                &mut a_buffer1,
                &mut b_buffer1,
                &mut c_buffer1,
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
        validate_oof_sizes!(input, output, 81);

        let mut a_buffer = [T::default(); 27];
        let mut b_buffer = [T::default(); 27];
        let mut c_buffer = [T::default(); 27];

        let mut a_buffer1 = [T::default(); 9];
        let mut b_buffer1 = [T::default(); 9];
        let mut c_buffer1 = [T::default(); 9];

        use crate::bidirectional::BiStore;
        for (src, dst) in input.chunks_exact(81).zip(output.chunks_exact_mut(81)) {
            self.exec(
                &mut BiStore::new(src, dst),
                &mut a_buffer,
                &mut b_buffer,
                &mut c_buffer,
                &mut a_buffer1,
                &mut b_buffer1,
                &mut c_buffer1,
            );
        }
        Ok(())
    }

    fn length(&self) -> usize {
        81
    }

    fn scratch_size(&self) -> usize {
        81
    }
}

#[allow(unused)]
#[derive(Clone)]
pub(crate) struct Dct2Butterfly216<T: DctSample> {
    inner_layer: [Complex<T>; 144],
    bf36: Arc<dyn PxdctExecutor<T> + Send + Sync>,
}

impl<T: DctSample + Dct2Factory> Default for Dct2Butterfly216<T>
where
    f64: AsPrimitive<T>,
    usize: AsPrimitive<T>,
{
    fn default() -> Self {
        let mut inner_layer = [Complex::<T>::default(); 144];
        for (i, layer) in inner_layer.chunks_exact_mut(4).enumerate() {
            let angle = (2. * i as f64 + 1.).as_();
            layer[0] = mixed_radix_inner_twiddle(angle, 216);
            layer[0].im *= T::SQRT_3;
            layer[1] = mixed_radix_inner_twiddle(2f64.as_() * angle, 216);
            layer[1].im *= T::SQRT_3;
            layer[2] = mixed_radix_inner_twiddle(3f64.as_() * angle, 216);
            layer[2].im *= T::SQRT_3;
            layer[3] = mixed_radix_inner_twiddle(5f64.as_() * angle, 216);
            layer[3].im = -layer[3].im * T::SQRT_3;
        }

        Self {
            inner_layer,
            bf36: T::dct2_butterfly36(),
        }
    }
}

impl<T: DctSample> Dct2Butterfly216<T>
where
    f64: AsPrimitive<T>,
{
    #[allow(unused)]
    #[inline(always)]
    fn exec<S: BidirectionalStore<T>>(&self, data: &mut S, scratch: &mut [T; 216]) {
        let s_n = 216 / 3;
        let s_2n = 2 * 216 / 3;

        let (a_buffer, rem) = scratch.split_at_mut(36);
        let (b_buffer, rem) = rem.split_at_mut(36);
        let (c_buffer, rem) = rem.split_at_mut(36);
        let (d_buffer, rem) = rem.split_at_mut(36);
        let (e_buffer, rem) = rem.split_at_mut(36);
        let (f_buffer, _) = rem.split_at_mut(36);

        for i in 0..36 {
            let ai = data[i];
            let bi = data[s_n - i - 1];
            let ci = data[s_n + i];
            let di = data[s_2n - i - 1];
            let ei = data[s_2n + i];
            let fi = data[216 - i - 1];

            let cos_sin_ai = self.inner_layer[i * 4];
            let cos_sin_2ai = self.inner_layer[i * 4 + 1];
            let cos_sin_3ai = self.inner_layer[i * 4 + 2];
            let cos_sin_5ai = self.inner_layer[i * 4 + 3];

            let s2 = bi + ei;
            let dcd = ci - di;
            let dbe = bi - ei;

            let ai2 = T::TWO * ai;
            let fi2 = T::TWO * fi;
            let scd = ci + di;

            let sdbedcd = dbe + dcd;
            let ai2dbedcd = ai2 + sdbedcd - fi2;

            let s2scd = s2 + scd;

            let a_comp = ai + s2scd + fi;
            let c_comp = ai2 - s2scd + fi2;
            let d_comp = T::TWO * (ai - sdbedcd - fi);

            let dbedcd = dbe - dcd;

            let c_img = s2 - scd;
            let b_zet = dbedcd * cos_sin_ai.im;
            let c_zet = c_img * cos_sin_2ai.im;
            let f_zet = dbedcd * cos_sin_5ai.im;

            let e_comp = fmla(
                T::TWO * cos_sin_2ai.re,
                fmla(c_comp, cos_sin_2ai.re, -c_zet),
                -c_comp,
            );

            unsafe {
                *a_buffer.get_unchecked_mut(i) = a_comp;
                *b_buffer.get_unchecked_mut(i) = fmla(ai2dbedcd, cos_sin_ai.re, b_zet);
                *c_buffer.get_unchecked_mut(i) = fmla(c_comp, cos_sin_2ai.re, c_zet);
                *d_buffer.get_unchecked_mut(i) = d_comp * cos_sin_3ai.re;
                *e_buffer.get_unchecked_mut(i) = e_comp;
                *f_buffer.get_unchecked_mut(i) = fmla(ai2dbedcd, cos_sin_5ai.re, f_zet);
            }
        }

        _ = self.bf36.execute(scratch);

        let (a_buffer, rem) = scratch.split_at_mut(36);
        let (b_buffer, rem) = rem.split_at_mut(36);
        let (c_buffer, rem) = rem.split_at_mut(36);
        let (d_buffer, rem) = rem.split_at_mut(36);
        let (e_buffer, rem) = rem.split_at_mut(36);
        let (f_buffer, _) = rem.split_at_mut(36);

        data[0] = a_buffer[0];
        let b0 = b_buffer[0] * T::HALF;
        data[1] = b0;
        let c0 = c_buffer[0] * T::HALF;
        data[2] = c0;
        let d0 = d_buffer[0] * T::HALF;
        data[3] = d0;
        let e0 = e_buffer[0] * T::HALF;
        data[4] = e0;
        let f0 = f_buffer[0] * T::HALF;
        data[5] = f0;

        let mut b_diff = f0;
        let mut c_diff = e0;
        let mut e_diff = d0;
        let mut d_diff = c0;
        let mut f_diff = b0;

        for k in 1..36 {
            data[6 * k] = a_buffer[k];
            let deferred_f_diff = b_buffer[k] - b_diff;
            data[6 * k + 1] = deferred_f_diff;
            let deferred_d_diff = c_buffer[k] - c_diff;
            data[6 * k + 2] = deferred_d_diff;
            e_diff = d_buffer[k] - e_diff;
            data[6 * k + 3] = e_diff;
            let new_d = e_buffer[k] - d_diff;
            data[6 * k + 4] = new_d;
            c_diff = new_d;
            d_diff = deferred_d_diff;
            let new_f = f_buffer[k] - f_diff;
            b_diff = new_f;
            f_diff = deferred_f_diff;
            data[6 * k + 5] = new_f;
        }
    }
}

impl<T: DctSample> PxdctExecutor<T> for Dct2Butterfly216<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, data: &mut [T]) -> Result<(), PxdctError> {
        let mut scratch = [T::default(); 216];
        self.execute_with_scratch(data, &mut scratch)
    }

    fn execute_with_scratch(&self, data: &mut [T], scratch: &mut [T]) -> Result<(), PxdctError> {
        if !data.len().is_multiple_of(216) {
            return Err(PxdctError::InvalidSizeMultiplier(data.len(), self.length()));
        }

        let full_scratch = validate_scratch!(scratch, 216);

        let scratch: &mut [T; 216] = (&mut full_scratch[..216]).try_into().unwrap();

        for chunk in data.chunks_exact_mut(216) {
            self.exec(&mut InPlaceStore::new(chunk), scratch);
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
        validate_oof_sizes!(input, output, 216);

        let mut scratch = [T::default(); 216];

        use crate::bidirectional::BiStore;
        for (src, dst) in input.chunks_exact(216).zip(output.chunks_exact_mut(216)) {
            self.exec(&mut BiStore::new(src, dst), &mut scratch);
        }
        Ok(())
    }

    fn length(&self) -> usize {
        216
    }

    fn scratch_size(&self) -> usize {
        216
    }
}

#[cfg(test)]
macro_rules! gen_test_butterfly {
    ($test_name: ident, $f_typ: ident, $bf_name: ident, $size:expr, $cutoff: expr, $naive_reference: ident) => {
        #[test]
        fn $test_name() {
            let mut input = vec![0.; $size];
            use rand::RngExt;
            for z in input.iter_mut() {
                *z = rand::rng().random_range(1.0..2.0);
            }
            use crate::tests::$naive_reference;
            let reference_input = input.clone();
            let reference_input = $naive_reference(&reference_input);
            use crate::PxdctExecutor;
            let bf = $bf_name::<$f_typ>::default();
            bf.execute(&mut input).unwrap();
            assert_eq!(bf.length(), $size);
            input
                .iter()
                .zip(reference_input.iter())
                .enumerate()
                .for_each(|(i, (&src, &r0))| {
                    assert!(
                        (src - r0).abs() < $cutoff,
                        "Difference must be < {}, but it was {}, at position {i} seq ref - {:?} seq n - {:?}",
                        $cutoff,
                        (src - r0).abs(),
                        reference_input,
                        input
                    )
                });
        }
    };
}

#[derive(Clone)]
pub(crate) struct Dct2Butterfly243<T: DctSample> {
    bf81: Arc<dyn PxdctExecutor<T> + Send + Sync>,
    inner_layer: [Complex<T>; 162],
}

impl<T: DctSample + Dct2Factory> Default for Dct2Butterfly243<T>
where
    f64: AsPrimitive<T>,
    usize: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            bf81: T::dct2_butterfly81(),
            inner_layer: mixed_radix3_twiddles(243),
        }
    }
}

impl<T: DctSample> Dct2Butterfly243<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    fn exec<S: BidirectionalStore<T>>(
        &self,
        data: &mut S,
        a_buffer: &mut [T; 81],
        b_buffer: &mut [T; 81],
        c_buffer: &mut [T; 81],
    ) {
        for i in 0..81 {
            let ai = data[i];
            let bi = data[162 + i];
            let ci = data[162 - i - 1];

            let cos_sin_ai = self.inner_layer[i * 2];
            let cos_sin_2ai = self.inner_layer[i * 2 + 1];

            let a_comp = ai + bi + ci;
            let second_layer_comp0 = fmla(2f64.as_(), ai, -bi - ci);

            let d_ci_bi = ci - bi;

            let b0_b = fmla(second_layer_comp0, cos_sin_ai.re, d_ci_bi * cos_sin_ai.im);
            let c0_b = fmla(second_layer_comp0, cos_sin_2ai.re, d_ci_bi * cos_sin_2ai.im);

            a_buffer[i] = a_comp;
            b_buffer[i] = b0_b;
            c_buffer[i] = c0_b;
        }

        _ = self.bf81.execute(a_buffer);
        _ = self.bf81.execute(b_buffer);
        _ = self.bf81.execute(c_buffer);

        data[0] = a_buffer[0];
        let b_value = b_buffer[0] * T::HALF;
        data[1] = b_value;
        let c_value = c_buffer[0] * T::HALF;
        data[2] = c_value;

        let mut last_b = c_value;
        let mut last_c = b_value;

        for k in 1..81 {
            data[3 * k] = a_buffer[k];

            let deferred_c = b_buffer[k] - last_b;
            data[3 * k + 1] = deferred_c;

            last_b = c_buffer[k] - last_c;
            data[3 * k + 2] = last_b;
            last_c = deferred_c;
        }
    }
}

impl<T: DctSample> PxdctExecutor<T> for Dct2Butterfly243<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, data: &mut [T]) -> Result<(), PxdctError> {
        let mut scratch = [T::default(); 243];
        self.execute_with_scratch(data, &mut scratch)
    }

    fn execute_with_scratch(&self, data: &mut [T], _: &mut [T]) -> Result<(), PxdctError> {
        if !data.len().is_multiple_of(243) {
            return Err(PxdctError::InvalidSizeMultiplier(data.len(), self.length()));
        }

        let mut a_buffer = [T::zero(); 81];
        let mut b_buffer = [T::zero(); 81];
        let mut c_buffer = [T::zero(); 81];

        for chunk in data.chunks_exact_mut(243) {
            self.exec(
                &mut InPlaceStore::new(chunk),
                &mut a_buffer,
                &mut b_buffer,
                &mut c_buffer,
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
        validate_oof_sizes!(input, output, 243);

        let mut a_buffer = [T::zero(); 81];
        let mut b_buffer = [T::zero(); 81];
        let mut c_buffer = [T::zero(); 81];

        use crate::bidirectional::BiStore;
        for (src, dst) in input.chunks_exact(243).zip(output.chunks_exact_mut(243)) {
            self.exec(
                &mut BiStore::new(src, dst),
                &mut a_buffer,
                &mut b_buffer,
                &mut c_buffer,
            );
        }
        Ok(())
    }

    fn length(&self) -> usize {
        243
    }

    fn scratch_size(&self) -> usize {
        0
    }
}

#[cfg(test)]
pub(super) use gen_test_butterfly;

#[cfg(test)]
#[allow(unused)]
macro_rules! gen_test_butterfly_f {
    ($test_name: ident, $bf_name: ident, $size:expr, $cutoff: expr, $naive_reference: ident) => {
        #[test]
        fn $test_name() {
            let mut input = vec![0.; $size];
            for z in input.iter_mut() {
                use rand::RngExt;
                *z = rand::rng().random_range(1.0..2.0);
            }
            let reference_input = input.clone();
            let reference_input = $naive_reference(&reference_input);
            let bf = $bf_name::default();
            bf.execute(&mut input).unwrap();
            assert_eq!(bf.length(), $size);
            input
                .iter()
                .zip(reference_input.iter())
                .enumerate()
                .for_each(|(i, (&src, &r0))| {
                    assert!(
                        (src - r0).abs() < $cutoff,
                        "Difference must be < {}, but it was {}, at position {i}",
                        $cutoff,
                        (src - r0).abs()
                    )
                });
        }
    };
}

use crate::bidirectional::{BidirectionalStore, InPlaceStore};
use crate::dct2::MixedRadix7Sample;
use crate::dct2::power2_butterflies::{
    Dct2Butterfly2, Dct2Butterfly4, Dct2Butterfly8, Dct2Butterfly16,
};
use crate::dct2::prime_butterflies::{
    Dct2Butterfly3, Dct2Butterfly5, Dct2Butterfly7, Dct2Butterfly11, Dct2Butterfly13,
    MixedRadix11Sample,
};
use crate::factory_dct2::Dct2Factory;
#[cfg(test)]
#[allow(unused)]
pub(super) use gen_test_butterfly_f;

#[cfg(test)]
mod tests {
    use super::*;

    gen_test_butterfly!(test_bf3, f64, Dct2Butterfly3, 3, 1e-7, naive_dct2);
    gen_test_butterfly!(test_bf4, f64, Dct2Butterfly4, 4, 1e-7, naive_dct2);
    gen_test_butterfly!(test_bf6, f64, Dct2Butterfly6, 6, 1e-7, naive_dct2);
    gen_test_butterfly!(test_bf9, f64, Dct2Butterfly9, 9, 1e-7, naive_dct2);
    gen_test_butterfly!(test_bf10, f64, Dct2Butterfly10, 10, 1e-7, naive_dct2);
    gen_test_butterfly!(test_bf12, f64, Dct2Butterfly12, 12, 1e-7, naive_dct2);
    gen_test_butterfly!(test_bf14, f64, Dct2Butterfly14, 14, 1e-7, naive_dct2);
    gen_test_butterfly!(test_bf15n, f64, Dct2Butterfly15, 15, 1e-7, naive_dct2);
    gen_test_butterfly!(test_bf18, f64, Dct2Butterfly18, 18, 1e-7, naive_dct2);
    gen_test_butterfly!(test_bf20, f64, Dct2Butterfly20, 20, 1e-7, naive_dct2);
    gen_test_butterfly!(test_bf21, f64, Dct2Butterfly21, 21, 1e-7, naive_dct2);
    gen_test_butterfly!(test_bf22, f64, Dct2Butterfly22, 22, 1e-7, naive_dct2);
    gen_test_butterfly!(test_bf24, f64, Dct2Butterfly24, 24, 1e-7, naive_dct2);
    gen_test_butterfly!(test_bf26, f64, Dct2Butterfly26, 26, 1e-7, naive_dct2);
    gen_test_butterfly!(test_bf27, f64, Dct2Butterfly27, 27, 1e-7, naive_dct2);
    gen_test_butterfly!(test_bf30, f64, Dct2Butterfly30, 30, 1e-7, naive_dct2);
    gen_test_butterfly!(test_bf35, f64, Dct2Butterfly35, 35, 1e-7, naive_dct2);
    gen_test_butterfly!(test_bf36, f64, Dct2Butterfly36, 36, 1e-7, naive_dct2);
    gen_test_butterfly!(test_bf42, f64, Dct2Butterfly42, 42, 1e-7, naive_dct2);
    gen_test_butterfly!(test_bf48, f64, Dct2Butterfly48, 48, 1e-7, naive_dct2);
    gen_test_butterfly!(test_bf81, f64, Dct2Butterfly81, 81, 1e-7, naive_dct2);
    gen_test_butterfly!(test_bf243, f64, Dct2Butterfly243, 243, 1e-7, naive_dct2);
    gen_test_butterfly!(test_bf216, f64, Dct2Butterfly216, 216, 1e-7, naive_dct2);
}
