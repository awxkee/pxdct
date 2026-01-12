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
use crate::mla::fmla;
use crate::util::DctSample;
use crate::{PxdctError, PxdctExecutor};
use num_traits::AsPrimitive;
use std::marker::PhantomData;

#[derive(Debug, Clone)]
pub(crate) struct Dct4Butterfly3<T> {
    _phantom_data: PhantomData<T>,
}

impl<T: DctSample> Default for Dct4Butterfly3<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            _phantom_data: PhantomData {},
        }
    }
}

impl<T: DctSample> Dct4Butterfly3<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    pub(crate) fn exec(&self, data: &mut [T; 3]) {
        // DCT-IV Radix-Q for Q = 3
        let identity = T::FRAC_1_SQRT_2;
        let a = data[1] * identity;
        let c = (data[0] + data[2]) * identity;
        let s = (data[0] - data[2]) * identity;

        let u0 = fmla(c, T::SQRT_3_OVER_2, s * T::HALF);
        let mut v0 = fmla(c, T::HALF, -s * T::SQRT_3_OVER_2);

        let mut u1 = u0 * T::HALF;
        v0 *= T::SQRT_3_OVER_2;
        u1 = u1 - a;

        data[0] = u0 + a;
        data[1] = u1 - v0;
        data[2] = v0 + u1;
    }
}

impl<T: DctSample> PxdctExecutor<T> for Dct4Butterfly3<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, data: &mut [T]) -> Result<(), PxdctError> {
        if !data.len().is_multiple_of(3) {
            return Err(PxdctError::InvalidSizeMultiplier(data.len(), self.length()));
        }
        for chunk in data.chunks_exact_mut(3) {
            self.exec((&mut chunk[..3]).try_into().unwrap());
        }
        Ok(())
    }

    fn length(&self) -> usize {
        3
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PxdctExecutor;
    use crate::butterflies::gen_test_butterfly;
    use crate::tests::naive_dct4;
    use rand::Rng;

    gen_test_butterfly!(test_bf_dct4_3, f64, Dct4Butterfly3, 3, 1e-7, naive_dct4);
}
