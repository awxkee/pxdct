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
use crate::avx::{AvxDct2Butterfly3, AvxDct2Butterfly4};
use crate::util::DctSample;
use crate::{PxdctError, PxdctExecutor};
use num_traits::AsPrimitive;

#[derive(Debug, Clone)]
pub(crate) struct AvxDct2Butterfly12<T: DctSample> {
    bf4: AvxDct2Butterfly4<T>,
    bf3: AvxDct2Butterfly3<T>,
}

impl<T: DctSample> Default for AvxDct2Butterfly12<T>
where
    f64: AsPrimitive<T>,
    usize: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            bf4: AvxDct2Butterfly4::default(),
            bf3: AvxDct2Butterfly3::default(),
        }
    }
}

impl<T: DctSample> AvxDct2Butterfly12<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    fn exec(&self, data: &mut [T; 12]) {
        // co-prime 3x4 DCT-II algorithm

        let mut c0 = [data[0], data[7], data[8]];
        let mut c1 = [data[6], data[1], data[9]];
        let mut c2 = [data[5], data[10], data[2]];
        let mut c3 = [data[11], data[4], data[3]];

        self.bf3.exec(&mut c0);
        self.bf3.exec(&mut c1);
        self.bf3.exec(&mut c2);
        self.bf3.exec(&mut c3);

        let mut rows0 = [c0[0], c1[0], c2[0], c3[0]];
        let mut rows1 = [c0[1], c1[1], c2[1], c3[1]];
        let mut rows2 = [c0[2], c1[2], c2[2], c3[2]];

        self.bf4.exec(&mut rows0);
        self.bf4.exec(&mut rows1);
        self.bf4.exec(&mut rows2);

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

impl<T: DctSample> AvxDct2Butterfly12<T>
where
    f64: AsPrimitive<T>,
{
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_impl(&self, data: &mut [T]) -> Result<(), PxdctError> {
        if !data.len().is_multiple_of(12) {
            return Err(PxdctError::InvalidSizeMultiplier(data.len(), self.length()));
        }
        for chunk in data.chunks_exact_mut(12) {
            self.exec((&mut chunk[..12]).try_into().unwrap());
        }
        Ok(())
    }
}

impl<T: DctSample> PxdctExecutor<T> for AvxDct2Butterfly12<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, data: &mut [T]) -> Result<(), PxdctError> {
        unsafe { self.execute_impl(data) }
    }

    fn length(&self) -> usize {
        12
    }
}
