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
use crate::util::DctSample;
use crate::{PxdctError, PxdctExecutor};
use num_traits::AsPrimitive;

#[derive(Debug, Clone)]
pub(crate) struct Dct4Identity<T: DctSample> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T: DctSample> Default for Dct4Identity<T> {
    fn default() -> Self {
        Dct4Identity {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: DctSample> PxdctExecutor<T> for Dct4Identity<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, in_place: &mut [T]) -> Result<(), PxdctError> {
        if in_place.is_empty() {
            return Err(PxdctError::InvalidSizeMultiplier(1, 0));
        }
        for sample in in_place.iter_mut() {
            *sample *= T::FRAC_1_SQRT_2;
        }
        Ok(())
    }

    fn execute_with_scratch(&self, data: &mut [T], _: &mut [T]) -> Result<(), PxdctError> {
        if data.is_empty() {
            return Err(PxdctError::InvalidSizeMultiplier(1, 0));
        }
        for sample in data.iter_mut() {
            *sample *= T::FRAC_1_SQRT_2;
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
        validate_oof_sizes!(input, output, 1);
        for (&src, dst) in input.iter().zip(output.iter_mut()) {
            *dst = src * T::FRAC_1_SQRT_2;
        }
        Ok(())
    }

    fn length(&self) -> usize {
        1
    }

    fn scratch_size(&self) -> usize {
        0
    }
}
