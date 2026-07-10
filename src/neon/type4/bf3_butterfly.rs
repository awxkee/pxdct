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
use crate::neon::type4::mixed_radix3f::dct4_radix_n_rotation_twiddles_neon;
use crate::neon::util::NeonStoreF;
use crate::type4::Dct4Butterfly9;
use crate::util::DctConstants;
use crate::{PxdctError, PxdctExecutor};

#[derive(Debug, Clone)]
pub(crate) struct NeonDct4Butterfly27f {
    twiddle: [NeonStoreF; 6],
    bf9: Dct4Butterfly9<f32>,
}

impl Default for NeonDct4Butterfly27f {
    fn default() -> Self {
        Self {
            twiddle: dct4_radix_n_rotation_twiddles_neon(3, 9, 27)
                .try_into()
                .unwrap(),
            bf9: Dct4Butterfly9::default(),
        }
    }
}

impl NeonDct4Butterfly27f {
    #[inline(always)]
    pub(crate) fn exec<S: BidirectionalStore<f32>>(&self, data: &mut S) {
        let mut a_buffer = [
            data[1], data[4], data[7], data[10], data[13], data[16], data[19], data[22], data[25],
        ];
        let mut c_buffer = [
            data[0] + data[2],
            data[3] + data[5],
            data[6] + data[8],
            data[9] + data[11],
            data[12] + data[14],
            data[15] + data[17],
            data[18] + data[20],
            data[21] + data[23],
            data[24] + data[26],
        ];
        let mut s_buffer = [
            data[0] - data[2],
            data[5] - data[3],
            data[6] - data[8],
            data[11] - data[9],
            data[12] - data[14],
            data[17] - data[15],
            data[18] - data[20],
            data[23] - data[21],
            data[24] - data[26],
        ];

        self.bf9
            .exec(&mut InPlaceStore::new(a_buffer.as_mut_slice()));
        self.bf9
            .exec(&mut InPlaceStore::new(c_buffer.as_mut_slice()));
        self.bf9
            .exec(&mut InPlaceStore::new(s_buffer.as_mut_slice()));

        let mut k = 0usize;
        let mut uk = 0usize;

        let q_modules = 9;
        let s = 2 * 27 / 3;

        while k + 4 <= q_modules {
            const S: usize = 4;
            let c_v = NeonStoreF::load(unsafe { c_buffer.get_unchecked(k..) });
            let s_v =
                NeonStoreF::load(unsafe { s_buffer.get_unchecked(q_modules - k - S..) }).reverse();
            let a_v = NeonStoreF::load(unsafe { a_buffer.get_unchecked(k..) });

            let twiddle_re = unsafe { *self.twiddle.get_unchecked(uk) };
            let twiddle_im = unsafe { *self.twiddle.get_unchecked(uk + 1) };

            let mut u0 = fmla(c_v, twiddle_re, s_v * twiddle_im);
            let mut u1 = u0;
            let mut v0 = fmla(c_v, twiddle_im, -s_v * twiddle_re);

            u0 += a_v;
            u1 *= f32::HALF;
            v0 *= f32::SQRT_3_OVER_2;
            u1 = u1 - a_v;

            let uc0 = u1 - v0;
            let uc1 = u1 + v0;

            u0.write(data.slice_from_mut(k..));
            uc1.write(data.slice_from_mut(s + k..));
            uc0.reverse().write(data.slice_from_mut(s - S - k..));

            k += 4;
            uk += 2;
        }

        {
            const S: usize = 1;
            let c_v = NeonStoreF::load1(unsafe { c_buffer.get_unchecked(k..) });
            let s_v = NeonStoreF::load1(unsafe { s_buffer.get_unchecked(q_modules - k - S..) });
            let a_v = NeonStoreF::load1(unsafe { a_buffer.get_unchecked(k..) });

            let twiddle_re = unsafe { *self.twiddle.get_unchecked(uk) };
            let twiddle_im = unsafe { *self.twiddle.get_unchecked(uk + 1) };

            let mut u0 = fmla(c_v, twiddle_re, s_v * twiddle_im);
            let mut u1 = u0;
            let mut v0 = fmla(c_v, twiddle_im, -s_v * twiddle_re);

            u0 += a_v;
            u1 *= f32::HALF;
            v0 *= f32::SQRT_3_OVER_2;
            u1 = u1 - a_v;

            let uc0 = u1 - v0;
            let uc1 = u1 + v0;

            u0.write1(data.slice_from_mut(k..));
            uc1.write1(data.slice_from_mut(s + k..));
            uc0.write1(data.slice_from_mut(s - S - k..));
        }
    }
}

impl PxdctExecutor<f32> for NeonDct4Butterfly27f {
    fn execute(&self, data: &mut [f32]) -> Result<(), PxdctError> {
        if !data.len().is_multiple_of(27) {
            return Err(PxdctError::InvalidSizeMultiplier(data.len(), self.length()));
        }

        use crate::bidirectional::InPlaceStore;

        for chunk in data.as_chunks_mut::<27>().0.iter_mut() {
            self.exec(&mut InPlaceStore::new(chunk));
        }
        Ok(())
    }

    fn execute_with_scratch(&self, data: &mut [f32], _: &mut [f32]) -> Result<(), PxdctError> {
        self.execute(data)
    }

    fn execute_into(&self, input: &[f32], output: &mut [f32]) -> Result<(), PxdctError> {
        self.execute_into_with_scratch(input, output, &mut [])
    }

    fn execute_into_with_scratch(
        &self,
        input: &[f32],
        output: &mut [f32],
        _: &mut [f32],
    ) -> Result<(), PxdctError> {
        use crate::util::validate_oof_sizes;
        validate_oof_sizes!(input, output, 27);
        use crate::bidirectional::BiStore;
        for (src, dst) in input
            .as_chunks::<27>()
            .0
            .iter()
            .zip(output.as_chunks_mut::<27>().0.iter_mut())
        {
            self.exec(&mut BiStore::new(src, dst));
        }
        Ok(())
    }

    fn length(&self) -> usize {
        27
    }

    fn scratch_size(&self) -> usize {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PxdctExecutor;
    use crate::butterflies::gen_test_butterfly_f;
    use crate::tests::naive_dct4_f32;

    gen_test_butterfly_f!(
        test_bf_dct4_27f,
        NeonDct4Butterfly27f,
        27,
        1e-3,
        naive_dct4_f32
    );
}
