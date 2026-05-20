/*
 * // Copyright (c) Radzivon Bartoshyk 5/2026. All rights reserved.
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
use crate::util::{DctSample, force_cast_real_scratch_to_complex, try_vec, validate_scratch};
use crate::{PxdctError, PxdctExecutor};
use num_traits::AsPrimitive;
use std::sync::Arc;
use zaft::R2CFftExecutor;

pub(crate) struct Dct1Fft<T> {
    fft_executor: Arc<dyn R2CFftExecutor<T> + Send + Sync>,
    execution_length: usize,
    fft_scratch_size: usize,
}

impl<T: DctSample> Dct1Fft<T>
where
    f64: AsPrimitive<T>,
{
    pub(crate) fn new(len: usize) -> Result<Dct1Fft<T>, PxdctError> {
        let inner_fft = T::make_fft_r2c((len - 1) * 2)?;

        let inner_fft_scratch = inner_fft.complex_scratch_length();

        Ok(Dct1Fft {
            fft_executor: inner_fft,
            fft_scratch_size: inner_fft_scratch,
            execution_length: len,
        })
    }
}

impl<T: DctSample> PxdctExecutor<T> for Dct1Fft<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, data: &mut [T]) -> Result<(), PxdctError> {
        let mut scratch = try_vec![T::default(); self.scratch_size()];
        self.execute_with_scratch(data, &mut scratch)
    }

    fn execute_with_scratch(&self, data: &mut [T], scratch: &mut [T]) -> Result<(), PxdctError> {
        if !data.len().is_multiple_of(self.execution_length) {
            return Err(PxdctError::InvalidSizeMultiplier(
                data.len(),
                self.execution_length,
            ));
        }

        let full_scratch = validate_scratch!(scratch, self.scratch_size());

        let in_real = (self.execution_length - 1) * 2;
        let complex_len = ((self.execution_length - 1) * 2) / 2 + 1;

        let (scratch_real, rem_scratch) = full_scratch.split_at_mut(in_real);
        let (c1, c2) = rem_scratch.split_at_mut(self.fft_scratch_size * 2);
        let scratch_complex = force_cast_real_scratch_to_complex(c2, complex_len);
        let scratch_fft = force_cast_real_scratch_to_complex(c1, self.fft_scratch_size);

        for chunk in data.chunks_exact_mut(self.execution_length) {
            scratch_real[..self.execution_length].copy_from_slice(chunk);

            scratch_real[self.execution_length..]
                .iter_mut()
                .zip(chunk[1..self.execution_length - 1].iter().rev())
                .for_each(|(fft_val, &data_val)| *fft_val = data_val);

            self.fft_executor
                .execute_with_scratch(scratch_real, scratch_complex, scratch_fft)
                .map_err(|x| PxdctError::FftError(x.to_string()))?;

            for (d, f) in chunk.iter_mut().zip(scratch_complex.iter()) {
                *d = f.re;
            }
        }

        Ok(())
    }

    fn execute_into(&self, input: &[T], output: &mut [T]) -> Result<(), PxdctError> {
        let mut scratch = try_vec![T::default(); self.scratch_size()];
        self.execute_into_with_scratch(input, output, &mut scratch)
    }

    fn execute_into_with_scratch(
        &self,
        input: &[T],
        output: &mut [T],
        scratch: &mut [T],
    ) -> Result<(), PxdctError> {
        use crate::util::validate_oof_sizes;
        validate_oof_sizes!(input, output, self.execution_length);

        let full_scratch = validate_scratch!(scratch, self.scratch_size());

        let in_real = (self.execution_length - 1) * 2;
        let complex_len = ((self.execution_length - 1) * 2) / 2 + 1;

        let (scratch_real, rem_scratch) = full_scratch.split_at_mut(in_real);
        let (c1, c2) = rem_scratch.split_at_mut(self.fft_scratch_size * 2);
        let scratch_complex = force_cast_real_scratch_to_complex(c2, complex_len);
        let scratch_fft = force_cast_real_scratch_to_complex(c1, self.fft_scratch_size);

        for (src, dst) in input
            .chunks_exact(self.execution_length)
            .zip(output.chunks_exact_mut(self.execution_length))
        {
            scratch_real[..self.execution_length].copy_from_slice(src);

            scratch_real[self.execution_length..]
                .iter_mut()
                .zip(src[1..self.execution_length - 1].iter().rev())
                .for_each(|(fft_val, &data_val)| *fft_val = data_val);

            self.fft_executor
                .execute_with_scratch(scratch_real, scratch_complex, scratch_fft)
                .map_err(|x| PxdctError::FftError(x.to_string()))?;

            for (d, f) in dst.iter_mut().zip(scratch_complex.iter()) {
                *d = f.re;
            }
        }
        Ok(())
    }

    fn length(&self) -> usize {
        self.execution_length
    }

    #[inline]
    fn scratch_size(&self) -> usize {
        let in_real = (self.execution_length - 1) * 2;
        let out_complex = ((self.execution_length - 1) * 2) / 2 + 1;
        self.fft_scratch_size * 2 + out_complex * 2 + in_real
    }
}

#[cfg(test)]
mod tests {
    use crate::PxdctExecutor;
    use crate::tests::naive_dct1;
    use crate::type1::fft::Dct1Fft;

    #[test]
    fn test_14() {
        let mut array = vec![0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13.];
        let dct0 = Dct1Fft::<f64>::new(array.len()).unwrap();

        let control = naive_dct1(&array);

        dct0.execute(&mut array).unwrap();
        array.iter().zip(control.iter()).enumerate().for_each(|(i, (&x, &c))| {
            assert!((x - c).abs() < 1e-4, "Difference to control values exceeded 1e-4 when it shouldn't, value {x}, control {c} at {i}");
        });
    }

    #[test]
    fn test_15() {
        let mut array = vec![
            0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14.,
        ];
        let dct0 = Dct1Fft::<f64>::new(array.len()).unwrap();

        let control = naive_dct1(&array);

        dct0.execute(&mut array).unwrap();
        array.iter().zip(control.iter()).enumerate().for_each(|(i, (&x, &c))| {
            assert!((x - c).abs() < 1e-4, "Difference to control values exceeded 1e-4 when it shouldn't, value {x}, control {c} at {i}");
        });
    }
}
