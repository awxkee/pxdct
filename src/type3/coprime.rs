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
use crate::transpose::Transposition;
use crate::util::{DctSample, try_vec, validate_scratch};
use crate::{PxdctError, PxdctExecutor, SpectralExecutor};
use num_traits::AsPrimitive;
use std::sync::Arc;

struct Dct3PfaIndexer {
    /// Gain (G) with sign encoded in sign bit; addresses original input data.
    in_gains: Vec<isize>,
    /// Modulation (M) with sign encoded in sign bit; addresses original input data.
    in_modulation: Vec<isize>,
    /// Output scatter: destination[0] = data[out_gather] in natural-order output
    out_gather: Vec<usize>,
    /// n2 = w = column count of the input butterfly buffer = row stride
    n2: usize,
}

struct PfaInputPlan {
    gains: Vec<isize>,
    modulation: Vec<isize>,
}

impl Dct3PfaIndexer {
    /// Unity-gain (G side) addressing.
    /// n1 = number of rows, n2 = row stride (cols).
    fn pfa_unity_gain(n1: usize, n2: usize) -> Result<Vec<isize>, PxdctError> {
        let length = n1 * n2;
        let mut out = try_vec![0isize; length];
        let mut i = 0usize;
        for r in 0..n1 {
            for c in 0..n2 {
                let idx = c * n1 + r * n2;
                out[i] = if idx < length {
                    idx as isize
                } else {
                    -((2 * length - idx) as isize)
                };
                i += 1;
            }
        }
        Ok(out)
    }

    /// Modulation offsets: M = |c·n1 − r·n2|
    fn pfa_modulation(n1: usize, n2: usize) -> Result<Vec<isize>, PxdctError> {
        let length = n1 * n2;
        let mut out = try_vec![0isize; length];
        let mut i = 0usize;
        for r in 0..n1 {
            for c in 0..n2 {
                out[i] =
                    (c as isize * n1 as isize - r as isize * n2 as isize).unsigned_abs() as isize;
                i += 1;
            }
        }
        Ok(out)
    }

    /// Build the input butterfly plan.
    /// n1 = number of rows, n2 = row stride.
    fn pfa_input_plan(n1: usize, n2: usize) -> Result<PfaInputPlan, PxdctError> {
        let gains = Self::pfa_unity_gain(n1, n2)?;
        let modulation = Self::pfa_modulation(n1, n2)?;
        Ok(PfaInputPlan { gains, modulation })
    }

    /// CRT folding permutation — used for the output scatter.
    fn pfa_crt_permutation(n1: usize, n2: usize) -> Result<Vec<usize>, PxdctError> {
        let mut indices = try_vec![0usize; n1 * n2];
        let mut index = 0usize;
        #[allow(clippy::explicit_counter_loop)]
        for _ in 0..(n1 * n2) {
            let mut k1 = index % (2 * n1);
            k1 = if k1 < n1 { k1 } else { 2 * n1 - k1 - 1 };
            let mut k2 = index % (2 * n2);
            k2 = if k2 < n2 { k2 } else { 2 * n2 - k2 - 1 };
            indices[k1 * n2 + k2] = index;
            index += 1;
        }
        Ok(indices)
    }

    fn pfa_crt_permutation_gather(w: usize, h: usize) -> Result<Vec<usize>, PxdctError> {
        let scatter = Self::pfa_crt_permutation(w, h)?;
        let mut gather = try_vec![0; scatter.len()];
        for (dst, &src) in scatter.iter().enumerate() {
            gather[src] = dst;
        }
        Ok(gather)
    }

    /// w = larger length (used as inner DCT length and as row stride in the butterfly buffer)
    /// h = smaller length (used as outer DCT length and as number of rows in the buffer)
    pub(crate) fn new(w: usize, h: usize) -> Result<Self, PxdctError> {
        // Matches generator: pfa_input_indices(n2=w, n1=h) -> pfa_unity_gain(n1=h, n2=w)
        let plan = Self::pfa_input_plan(h, w)?;

        // Output scatter: generator uses pfa_output_indices(w, h) = pfa_crt_permutation(n1=w, n2=h)
        let out_scatter = Self::pfa_crt_permutation_gather(w, h)?;

        Ok(Self {
            in_gains: plan.gains,
            in_modulation: plan.modulation,
            out_gather: out_scatter,
            n2: w, // row stride of the input butterfly buffer
        })
    }
}

pub(crate) struct Dct3Coprime<T> {
    width_dct: SpectralExecutor<T>,
    height_dct: SpectralExecutor<T>,
    transposition: Arc<dyn Transposition<T> + Send + Sync>,
    w: usize,
    h: usize,
    execution_length: usize,
    indexer: Dct3PfaIndexer,
    remapper: Arc<dyn PfaRemapper<T> + Send + Sync>,
    width_dct_scratch: usize,
    height_dct_scratch: usize,
}

impl<T: DctSample + PfaRemapperFactory> Dct3Coprime<T>
where
    f64: AsPrimitive<T>,
{
    pub(crate) fn new(
        mut width_dct: SpectralExecutor<T>,
        mut height_dct: SpectralExecutor<T>,
    ) -> Result<Self, PxdctError> {
        // Ensure w >= h
        if height_dct.length() > width_dct.length() {
            std::mem::swap(&mut width_dct, &mut height_dct);
        }
        let w = width_dct.length();
        let h = height_dct.length();
        if num_integer::gcd(w, h) != 1 {
            panic!("Internal error: co-prime DCT-III called on non-coprime dimensions {w}x{h}");
        }

        Ok(Self {
            width_dct_scratch: width_dct.scratch_size(),
            height_dct_scratch: height_dct.scratch_size(),
            w,
            h,
            execution_length: w * h,
            indexer: Dct3PfaIndexer::new(w, h)?,
            transposition: T::make_transpose(w, h),
            remapper: T::make_remapper(),
            width_dct,
            height_dct,
        })
    }
}

/// Butterfly kernel:  dst[i] = ±src[|G[i]|] ± src[|M[i]|]
/// `dst` is written linearly by `i`, the generator's row-major (n1=h, n2=w) layout.
pub(crate) trait PfaRemapper<T> {
    fn remap(
        &self,
        src: &[T],
        dst: &mut [T],
        gains: &[isize],
        modulation: &[isize],
        row_stride: usize, // = w
    );
}

struct DefaultPfaRemapper;

impl<T: DctSample> PfaRemapper<T> for DefaultPfaRemapper {
    fn remap(
        &self,
        src: &[T],
        dst: &mut [T],
        gains: &[isize],
        modulation: &[isize],
        row_stride: usize,
    ) {
        // Row 0: pass-through, dst[i] = src[|gain[i]|]
        let row0_gain = &gains[..row_stride];
        for (i, &g) in row0_gain.iter().enumerate() {
            unsafe {
                *dst.get_unchecked_mut(i) = *src.get_unchecked(g.unsigned_abs());
            }
        }

        // Remaining rows: row stride = w cols each
        let q_gain = &gains[row_stride..];
        let q_mod = &modulation[row_stride..];

        for (row_idx, (gain_row, mod_row)) in q_gain
            .chunks_exact(row_stride)
            .zip(q_mod.chunks_exact(row_stride))
            .enumerate()
        {
            let row_base = (row_idx + 1) * row_stride;

            // Column 0: pass-through, dst[base] = src[|gain[0]|]
            unsafe {
                *dst.get_unchecked_mut(row_base) =
                    *src.get_unchecked(gain_row.get_unchecked(0).unsigned_abs());
            }

            let gain_rest = &gain_row[1..];
            let mod_rest = &mod_row[1..];

            let (gain_chunks, gain_tail) = gain_rest.as_chunks::<4>();
            let (mod_chunks, mod_tail) = mod_rest.as_chunks::<4>();

            for (chunk_idx, (gain, modulation)) in
                gain_chunks.iter().zip(mod_chunks.iter()).enumerate()
            {
                let base = row_base + 1 + chunk_idx * 4;
                let g0 = gain[0];
                let m0 = modulation[0];
                let g1 = gain[1];
                let m1 = modulation[1];
                let g2 = gain[2];
                let m2 = modulation[2];
                let g3 = gain[3];
                let m3 = modulation[3];
                unsafe {
                    *dst.get_unchecked_mut(base) =
                        src.get_unchecked(m0.unsigned_abs()).mulsigni(m0)
                            + src.get_unchecked(g0.unsigned_abs()).mulsigni(g0);
                    *dst.get_unchecked_mut(base + 1) =
                        src.get_unchecked(m1.unsigned_abs()).mulsigni(m1)
                            + src.get_unchecked(g1.unsigned_abs()).mulsigni(g1);
                    *dst.get_unchecked_mut(base + 2) =
                        src.get_unchecked(m2.unsigned_abs()).mulsigni(m2)
                            + src.get_unchecked(g2.unsigned_abs()).mulsigni(g2);
                    *dst.get_unchecked_mut(base + 3) =
                        src.get_unchecked(m3.unsigned_abs()).mulsigni(m3)
                            + src.get_unchecked(g3.unsigned_abs()).mulsigni(g3);
                }
            }

            let tail_base = row_base + 1 + gain_chunks.len() * 4;
            for (i, (&g, &m)) in gain_tail.iter().zip(mod_tail.iter()).enumerate() {
                unsafe {
                    *dst.get_unchecked_mut(tail_base + i) =
                        src.get_unchecked(m.unsigned_abs()).mulsigni(m)
                            + src.get_unchecked(g.unsigned_abs()).mulsigni(g);
                }
            }
        }
    }
}

pub(crate) trait PfaRemapperFactory {
    fn make_remapper() -> Arc<dyn PfaRemapper<Self> + Send + Sync>;
}

impl PfaRemapperFactory for f32 {
    fn make_remapper() -> Arc<dyn PfaRemapper<Self> + Send + Sync> {
        Arc::new(DefaultPfaRemapper)
    }
}

impl PfaRemapperFactory for f64 {
    fn make_remapper() -> Arc<dyn PfaRemapper<Self> + Send + Sync> {
        Arc::new(DefaultPfaRemapper)
    }
}

impl<T: DctSample> Dct3Coprime<T>
where
    f64: AsPrimitive<T>,
{
    #[inline]
    fn do_remap_input(&self, src: &[T], dst: &mut [T]) {
        self.remapper.remap(
            src,
            dst,
            &self.indexer.in_gains,
            &self.indexer.in_modulation,
            self.indexer.n2, // = w = row stride of the butterfly buffer
        );
    }

    /// Output stage: scatter — dst[out_scatter[i]] = src[i]
    #[inline]
    fn do_remap_output(&self, src: &[T], dst: &mut [T]) {
        for (dst, &gather_src) in dst.iter_mut().zip(self.indexer.out_gather.iter()) {
            unsafe {
                *dst = *src.get_unchecked(gather_src);
            }
        }
    }
}

impl<T: DctSample> PxdctExecutor<T> for Dct3Coprime<T>
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
        let full = validate_scratch!(scratch, self.scratch_size());
        let (left, right) = full.split_at_mut(self.execution_length * 2);
        let (scratch0, scratch1) = left.split_at_mut(self.execution_length);
        let dct_scratch = &mut right[..self.height_dct_scratch.max(self.width_dct_scratch)];

        // in-place: use a temp copy of the chunk as src
        for chunk in data.chunks_exact_mut(self.execution_length) {
            self.do_remap_input(chunk, scratch0);

            for row in scratch0.chunks_exact_mut(self.w).skip(1) {
                row[0] = row[0] + row[0];
            }

            self.width_dct.execute_with_scratch(scratch0, dct_scratch)?;

            self.transposition.transpose(scratch0, scratch1);

            for row in scratch1.chunks_exact_mut(self.h) {
                row[0] = row[0] + row[0];
            }

            self.height_dct
                .execute_with_scratch(scratch1, dct_scratch)?;

            self.do_remap_output(scratch1, chunk);
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

        let full = validate_scratch!(scratch, self.scratch_size());
        let (left, right) = full.split_at_mut(self.execution_length * 2);
        let (scratch0, scratch1) = left.split_at_mut(self.execution_length);
        let dct_scratch = &mut right[..self.height_dct_scratch.max(self.width_dct_scratch)];

        for (src, dst) in input
            .chunks_exact(self.execution_length)
            .zip(output.chunks_exact_mut(self.execution_length))
        {
            self.do_remap_input(src, scratch0);

            for row in scratch0.chunks_exact_mut(self.w).skip(1) {
                row[0] = row[0] + row[0];
            }

            self.width_dct.execute_with_scratch(scratch0, dct_scratch)?;

            self.transposition.transpose(scratch0, scratch1);

            for row in scratch1.chunks_exact_mut(self.h) {
                row[0] = row[0] + row[0];
            }

            self.height_dct
                .execute_with_scratch(scratch1, dct_scratch)?;

            self.do_remap_output(scratch1, dst);
        }
        Ok(())
    }

    fn length(&self) -> usize {
        self.execution_length
    }

    fn scratch_size(&self) -> usize {
        self.execution_length * 3 + self.height_dct_scratch.max(self.width_dct_scratch)
    }
}

#[cfg(test)]
mod tests {
    use crate::PxdctExecutor;
    use crate::tests::naive_dct3_f32;
    use crate::type3::{Dct3Butterfly3, Dct3Butterfly4, Dct3Coprime};
    use std::sync::Arc;

    #[test]
    fn test_coprime_dct3_3x4() {
        let mut input: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let reference = naive_dct3_f32(&input);
        let bf = Dct3Coprime::new(
            Arc::new(Dct3Butterfly3::default()),
            Arc::new(Dct3Butterfly4::default()),
        )
        .unwrap();
        bf.execute(&mut input).unwrap();
        for (i, (&got, &exp)) in input.iter().zip(reference.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-3,
                "index {i}: got {got}, expected {exp}, diff {}",
                (got - exp).abs()
            );
        }
    }
}
