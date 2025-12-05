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
use crate::twiddles::compute_twiddle;
use crate::util::DctSample;
use crate::{PxdctError, PxdctExecutor};
use num_complex::Complex;
use num_traits::AsPrimitive;

#[derive(Debug, Clone, Default)]
pub(crate) struct Dst2Butterfly2 {}

impl Dst2Butterfly2 {
    #[inline(always)]
    pub(crate) fn exec<T: DctSample>(data: &[T; 2]) -> [T; 2] {
        let u0 = data[0];
        let u1 = data[1];
        let sum = u0 - u1;
        let v0 = (u0 + u1) * T::FRAC_1_SQRT_2;
        let v1 = sum;
        [v0, v1]
    }
}

impl<T: DctSample> PxdctExecutor<T> for Dst2Butterfly2
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, data: &mut [T]) -> Result<(), PxdctError> {
        if !data.len().is_multiple_of(2) {
            return Err(PxdctError::InvalidSizeMultiplier(data.len(), self.length()));
        }
        for chunk in data.chunks_exact_mut(2) {
            let u0 = chunk[0];
            let u1 = chunk[1];
            let sum = u0 - u1;
            chunk[0] = (u0 + u1) * T::FRAC_1_SQRT_2;
            chunk[1] = sum;
        }
        Ok(())
    }

    fn length(&self) -> usize {
        2
    }
}

#[derive(Debug, Clone, Default)]
pub(crate) struct Dct2Butterfly2 {}

impl Dct2Butterfly2 {
    #[inline(always)]
    pub(crate) fn exec<T: DctSample>(data: &[T; 2]) -> [T; 2] {
        let sum = data[0] + data[1];
        let v1 = (data[0] - data[1]) * T::FRAC_1_SQRT_2;
        let v0 = sum;
        [v0, v1]
    }
}

impl<T: DctSample> PxdctExecutor<T> for Dct2Butterfly2
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, data: &mut [T]) -> Result<(), PxdctError> {
        if !data.len().is_multiple_of(2) {
            return Err(PxdctError::InvalidSizeMultiplier(data.len(), self.length()));
        }
        for chunk in data.chunks_exact_mut(2) {
            let sum = chunk[0] + chunk[1];
            chunk[1] = (chunk[0] - chunk[1]) * T::FRAC_1_SQRT_2;
            chunk[0] = sum;
        }
        Ok(())
    }

    fn length(&self) -> usize {
        2
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Dct2Butterfly3<T: DctSample> {
    twiddle: T,
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

impl<T: DctSample> PxdctExecutor<T> for Dct2Butterfly3<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, data: &mut [T]) -> Result<(), PxdctError> {
        if !data.len().is_multiple_of(3) {
            return Err(PxdctError::InvalidSizeMultiplier(data.len(), self.length()));
        }
        for chunk in data.chunks_exact_mut(3) {
            let buffer_0 = chunk[0];
            let buffer_1 = chunk[1];
            let buffer_2 = chunk[2];

            chunk[0] = buffer_0 + buffer_1 + buffer_2;
            chunk[1] = (buffer_0 - buffer_2) * self.twiddle;
            chunk[2] = fmla(buffer_0 + buffer_2, T::HALF, -buffer_1);
        }
        Ok(())
    }

    fn length(&self) -> usize {
        3
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Dct2Butterfly4<T: DctSample> {
    twiddle: Complex<T>,
}

impl<T: DctSample> Default for Dct2Butterfly4<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            twiddle: compute_twiddle(1, 16).conj(),
        }
    }
}

impl<T: DctSample> Dct2Butterfly4<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    fn exec(&self, data: &[T; 4]) -> [T; 4] {
        let u0 = data[0];
        let u1 = data[1];
        let u2 = data[2];
        let u3 = data[3];

        let lower_dct4 = u0 - u3;
        let upper_dct4 = u2 - u1;

        let [v0, v2] = Dct2Butterfly2::exec(&[u0 + u3, u2 + u1]);

        let v1 = fmla(lower_dct4, self.twiddle.re, -upper_dct4 * self.twiddle.im);
        let v3 = fmla(upper_dct4, self.twiddle.re, lower_dct4 * self.twiddle.im);
        [v0, v1, v2, v3]
    }
}

impl<T: DctSample> PxdctExecutor<T> for Dct2Butterfly4<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, data: &mut [T]) -> Result<(), PxdctError> {
        if !data.len().is_multiple_of(4) {
            return Err(PxdctError::InvalidSizeMultiplier(data.len(), self.length()));
        }
        for chunk in data.chunks_exact_mut(4) {
            let u0 = chunk[0];
            let u1 = chunk[1];
            let u2 = chunk[2];
            let u3 = chunk[3];

            let lower_dct4 = u0 - u3;
            let upper_dct4 = u2 - u1;

            let [v0, v2] = Dct2Butterfly2::exec(&[u0 + u3, u2 + u1]);

            chunk[0] = v0;
            chunk[2] = v2;

            chunk[1] = fmla(lower_dct4, self.twiddle.re, -upper_dct4 * self.twiddle.im);
            chunk[3] = fmla(upper_dct4, self.twiddle.re, lower_dct4 * self.twiddle.im);
        }
        Ok(())
    }

    fn length(&self) -> usize {
        4
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Dst2Butterfly4<T: DctSample> {
    twiddle: Complex<T>,
}

impl<T: DctSample> Default for Dst2Butterfly4<T>
where
    f64: AsPrimitive<T>,
{
    fn default() -> Self {
        Self {
            twiddle: compute_twiddle(1, 16).conj(),
        }
    }
}

impl<T: DctSample> Dst2Butterfly4<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    fn exec(&self, data: &[T; 4]) -> [T; 4] {
        let u0 = data[0];
        let u1 = data[1];
        let u2 = data[2];
        let u3 = data[3];

        let lower_dct4 = u0 + u3;
        let upper_dct4 = u2 + u1;

        let q = Dct2Butterfly2::exec(&[u0 - u3, u2 - u1]);

        let v2 = fmla(lower_dct4, self.twiddle.re, -upper_dct4 * self.twiddle.im);
        let v0 = fmla(upper_dct4, self.twiddle.re, lower_dct4 * self.twiddle.im);
        [v0, q[1], v2, q[0]]
    }
}

impl<T: DctSample> PxdctExecutor<T> for Dst2Butterfly4<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, data: &mut [T]) -> Result<(), PxdctError> {
        if !data.len().is_multiple_of(4) {
            return Err(PxdctError::InvalidSizeMultiplier(data.len(), self.length()));
        }
        for chunk in data.chunks_exact_mut(4) {
            let u0 = chunk[0];
            let u1 = chunk[1];
            let u2 = chunk[2];
            let u3 = chunk[3];

            let w = self.exec(&[u0, u1, u2, u3]);
            chunk[0] = w[0];
            chunk[1] = w[1];
            chunk[2] = w[2];
            chunk[3] = w[3];
        }
        Ok(())
    }

    fn length(&self) -> usize {
        4
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Dct2Butterfly8<T: DctSample> {
    bf4: Dct2Butterfly4<T>,
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
        }
    }
}

impl<T: DctSample> Dct2Butterfly8<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    fn exec(&self, data: &[T; 8]) -> [T; 8] {
        let u0 = data[0];
        let u1 = data[1];
        let u2 = data[2];
        let u3 = data[3];
        let u4 = data[4];
        let u5 = data[5];
        let u6 = data[6];
        let u7 = data[7];

        let dct2_buffer = self.bf4.exec(&[u0 + u7, u1 + u6, u2 + u5, u3 + u4]);

        // odds
        let differences = [u0 - u7, u3 - u4, u1 - u6, u2 - u5];

        let dct4_even_buffer = Dct2Butterfly2::exec(&[
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
        ]);
        let dct4_odd_buffer = Dst2Butterfly2::exec(&[
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
        ]);

        // combine the results
        [
            dct2_buffer[0],
            dct4_even_buffer[0],
            dct2_buffer[1],
            dct4_even_buffer[1] - dct4_odd_buffer[0],
            dct2_buffer[2],
            dct4_even_buffer[1] + dct4_odd_buffer[0],
            dct2_buffer[3],
            dct4_odd_buffer[1],
        ]
    }
}

impl<T: DctSample> PxdctExecutor<T> for Dct2Butterfly8<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, data: &mut [T]) -> Result<(), PxdctError> {
        if !data.len().is_multiple_of(8) {
            return Err(PxdctError::InvalidSizeMultiplier(data.len(), self.length()));
        }
        for chunk in data.chunks_exact_mut(8) {
            let u0 = chunk[0];
            let u1 = chunk[1];
            let u2 = chunk[2];
            let u3 = chunk[3];
            let u4 = chunk[4];
            let u5 = chunk[5];
            let u6 = chunk[6];
            let u7 = chunk[7];

            let w = self.exec(&[u0, u1, u2, u3, u4, u5, u6, u7]);

            chunk[0] = w[0];
            chunk[1] = w[1];
            chunk[2] = w[2];
            chunk[3] = w[3];
            chunk[4] = w[4];
            chunk[5] = w[5];
            chunk[6] = w[6];
            chunk[7] = w[7];
        }
        Ok(())
    }

    fn length(&self) -> usize {
        8
    }
}

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

impl<T: DctSample> PxdctExecutor<T> for Dct2Butterfly16<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, data: &mut [T]) -> Result<(), PxdctError> {
        if !data.len().is_multiple_of(16) {
            return Err(PxdctError::InvalidSizeMultiplier(data.len(), self.length()));
        }
        for chunk in data.chunks_exact_mut(16) {
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
            let dct2_buffer = [
                u0 + u15,
                u1 + u14,
                u2 + u13,
                u3 + u12,
                u4 + u11,
                u5 + u10,
                u6 + u9,
                u7 + u8,
            ];
            let dct2_buffer = self.bf8.exec(&dct2_buffer);

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

            let dct4_even_buffer = [
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
            let dct4_odd_buffer = [
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

            let dct4_even_buffer = self.bf8.bf4.exec(&dct4_even_buffer);
            let dct4_odd_buffer = self.bf4_dst.exec(&dct4_odd_buffer);

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
        Ok(())
    }

    fn length(&self) -> usize {
        16
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PxdctExecutor;
    use crate::tests::{naive_dct2, naive_dst2};
    use rand::Rng;

    macro_rules! gen_test_butterfly {
        ($test_name: ident, $bf_name: ident, $size:expr, $cutoff: expr, $naive_reference: ident) => {
            #[test]
            fn $test_name() {
                let mut input = vec![0.; $size];
                for z in input.iter_mut() {
                    *z = rand::rng().random_range(1.0..2.0);
                }
                let reference_input = input.clone();
                let reference_input = $naive_reference(&reference_input);
                let bf = $bf_name::default();
                bf.execute(&mut input).unwrap();
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

    gen_test_butterfly!(test_bf_dst2, Dst2Butterfly2, 2, 1e-7, naive_dst2);
    gen_test_butterfly!(test_bf_dst4, Dst2Butterfly4, 4, 1e-7, naive_dst2);

    gen_test_butterfly!(test_bf2, Dct2Butterfly2, 2, 1e-7, naive_dct2);
    gen_test_butterfly!(test_bf3, Dct2Butterfly3, 3, 1e-7, naive_dct2);
    gen_test_butterfly!(test_bf4, Dct2Butterfly4, 4, 1e-7, naive_dct2);
    gen_test_butterfly!(test_bf8, Dct2Butterfly8, 8, 1e-7, naive_dct2);
    gen_test_butterfly!(test_bf16, Dct2Butterfly16, 16, 1e-7, naive_dct2);
}
