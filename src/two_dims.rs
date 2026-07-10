/*
 * // Copyright (c) Radzivon Bartoshyk 2/2026. All rights reserved.
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
use crate::transpose::{TransposeFactory, Transposition};
use crate::util::{DctSample, try_vec, validate_scratch};
use crate::{PxdctError, PxdctExecutor};
use std::sync::Arc;

pub trait MultidimensionalDctExecutor<T> {
    /// Executes a 2D DCT on source and writes the result into output.
    ///
    /// Both source and output must have length equal to width() * height().
    ///
    /// # Errors
    /// Returns a [PxdctError] if the execution fails.
    fn execute(&self, data: &mut [T]) -> Result<(), PxdctError>;
    /// Executes a 2D DCT using a pre-allocated `scratch` buffer for temporary storage.
    /// This can reduce allocations and improve performance for repeated calls.
    /// Both `source`, `output`, and `scratch` must have sufficient length.
    ///
    /// # Errors
    /// Returns a [`PxdctError`] if the execution fails.
    fn execute_with_scratch(&self, data: &mut [T], scratch: &mut [T]) -> Result<(), PxdctError>;
    /// Returns the **width** (number of columns, X-dimension) of the 2D input data grid.
    fn width(&self) -> usize;
    /// Returns the **height** (number of rows, Y-dimension) of the 2D input data grid.
    fn height(&self) -> usize;
    /// Required scratch size
    fn scratch_size(&self) -> usize;
}

pub(crate) struct TwoDimensionalDct<T> {
    pub(crate) width_executor: Arc<dyn PxdctExecutor<T> + Send + Sync>,
    pub(crate) height_executor: Arc<dyn PxdctExecutor<T> + Send + Sync>,
    pub(crate) transpose_width_to_height: Arc<dyn Transposition<T> + Send + Sync>,
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) width_scratch_size: usize,
    pub(crate) height_scratch_size: usize,
    pub(crate) full_size: usize,
    pub(crate) scratch_size: usize,
}

impl<T: DctSample + TransposeFactory> TwoDimensionalDct<T> {
    pub(crate) fn new(
        width_executor: Arc<dyn PxdctExecutor<T> + Send + Sync>,
        height_executor: Arc<dyn PxdctExecutor<T> + Send + Sync>,
    ) -> Result<Self, PxdctError> {
        let width = width_executor.length();
        let height = height_executor.length();
        if width == 0 || height == 0 {
            return Err(PxdctError::ZeroSizedDct);
        }

        let full_size = width
            .checked_mul(height)
            .ok_or(PxdctError::SizeOverflow(width, height))?;
        let width_scratch_size = width_executor.scratch_size();
        let height_scratch_size = height_executor.scratch_size();
        let transform_scratch_size = width_scratch_size.max(height_scratch_size);
        let scratch_size = full_size
            .checked_add(transform_scratch_size)
            .ok_or(PxdctError::SizeOverflow(full_size, transform_scratch_size))?;

        Ok(Self {
            width_executor,
            height_executor,
            transpose_width_to_height: T::make_transpose(width, height),
            width,
            height,
            width_scratch_size,
            height_scratch_size,
            full_size,
            scratch_size,
        })
    }
}

impl<T: DctSample> MultidimensionalDctExecutor<T> for TwoDimensionalDct<T> {
    fn execute(&self, data: &mut [T]) -> Result<(), PxdctError> {
        let mut scratch = try_vec![T::default(); self.scratch_size()];
        self.execute_with_scratch(data, &mut scratch)
    }

    fn execute_with_scratch(&self, data: &mut [T], scratch: &mut [T]) -> Result<(), PxdctError> {
        if !data.len().is_multiple_of(self.full_size) {
            return Err(PxdctError::InvalidSizeMultiplier(
                data.len(),
                self.full_size,
            ));
        }

        let scratch = validate_scratch!(scratch, self.scratch_size());
        let (scratch, rem_scratch) = scratch.split_at_mut(self.full_size);

        for chunk in data.chunks_exact_mut(self.full_size) {
            let (width_scratch, _) = rem_scratch.split_at_mut(self.width_scratch_size);
            self.width_executor
                .execute_into_with_scratch(chunk, scratch, width_scratch)?;

            self.transpose_width_to_height.transpose(scratch, chunk);

            let (height_scratch, _) = rem_scratch.split_at_mut(self.height_scratch_size);
            self.height_executor
                .execute_with_scratch(chunk, height_scratch)?;
        }
        Ok(())
    }

    #[inline]
    fn width(&self) -> usize {
        self.width
    }

    #[inline]
    fn height(&self) -> usize {
        self.height
    }

    #[inline]
    fn scratch_size(&self) -> usize {
        self.scratch_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FakeExecutor {
        length: usize,
        scratch_size: usize,
    }

    impl PxdctExecutor<f64> for FakeExecutor {
        fn execute(&self, _data: &mut [f64]) -> Result<(), PxdctError> {
            Ok(())
        }

        fn execute_with_scratch(
            &self,
            _data: &mut [f64],
            _scratch: &mut [f64],
        ) -> Result<(), PxdctError> {
            Ok(())
        }

        fn execute_into(&self, _input: &[f64], _output: &mut [f64]) -> Result<(), PxdctError> {
            Ok(())
        }

        fn execute_into_with_scratch(
            &self,
            _input: &[f64],
            _output: &mut [f64],
            _scratch: &mut [f64],
        ) -> Result<(), PxdctError> {
            Ok(())
        }

        fn length(&self) -> usize {
            self.length
        }

        fn scratch_size(&self) -> usize {
            self.scratch_size
        }
    }

    fn fake(length: usize, scratch_size: usize) -> Arc<dyn PxdctExecutor<f64> + Send + Sync> {
        Arc::new(FakeExecutor {
            length,
            scratch_size,
        })
    }

    #[test]
    fn rejects_dimension_product_overflow() {
        let result = TwoDimensionalDct::<f64>::new(fake(usize::MAX, 0), fake(usize::MAX, 0));
        assert!(matches!(result, Err(PxdctError::SizeOverflow(_, _))));
    }

    #[test]
    fn rejects_scratch_size_overflow() {
        let result = TwoDimensionalDct::<f64>::new(fake(usize::MAX, 1), fake(1, 0));
        assert!(matches!(result, Err(PxdctError::SizeOverflow(_, _))));
    }

    #[test]
    fn rejects_zero_dimension() {
        let result = TwoDimensionalDct::<f64>::new(fake(0, 0), fake(1, 0));
        assert!(matches!(result, Err(PxdctError::ZeroSizedDct)));
    }
}
