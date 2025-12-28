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
use crate::transpose::Transposition;
use crate::util::{DctSample, try_vec};
use crate::{PxdctError, PxdctExecutor};
use num_traits::AsPrimitive;
use std::sync::Arc;

/// Precomputed index/sign/modulation tables for PFA-DCT2
///
/// This structure fully encodes:
///   • input permutation
///   • output butterfly addressing
///   • sign handling
///   • modulation index mapping
///
/// All expensive address math is done once here.
struct Dct2PfaIndexer {
    /// Input permutation indices
    input: Vec<usize>,

    /// Gain (G) addressing with sign encoded in sign bit
    gains: Vec<isize>,
    /// Modulation (M) addressing with sign encoded in sign bit
    modulation: Vec<isize>,
    /// Final output scatter indices
    indices: Vec<isize>,
}

struct PfaOutput {
    gains: Vec<isize>,
    modulation: Vec<isize>,
    indices: Vec<isize>,
}

impl Dct2PfaIndexer {
    /// Generates the Chinese-Remainder folded input permutation.
    ///
    /// This converts a linear DCT-II input into the order required by
    /// two coprime 1-D transforms (n1 × n2).
    ///
    /// Folding (mirror wrap) is used to implement the even extension of DCT-II.
    fn pfa_input_indices(n1: usize, n2: usize) -> Result<Vec<usize>, PxdctError> {
        // this is technically output for DCT-III ( IDCT ) but was adapted in inverse order
        // for DCT-II
        let mut indices = try_vec![0usize; n1 * n2];
        let mut index = 0usize;
        #[allow(clippy::explicit_counter_loop)]
        for _ in 0..(n1 * n2) {
            let mut k1 = index % (2 * n1);
            k1 = if k1 < n1 { k1 } else { 2 * n1 - k1 - 1 };

            let mut k2 = index % (2 * n2);
            k2 = if k2 < n2 { k2 } else { 2 * n2 - k2 - 1 };

            let new_idx = k1 * n2 + k2;

            indices[new_idx] = index;

            index += 1;
        }
        Ok(indices)
    }

    /// Generates the unity-gain addressing for output butterflies.
    ///
    /// Produces the G side of (X = G + M) butterflies.
    /// Negative values encode subtraction.
    fn pfa_unity_gain(n1: usize, n2: usize) -> Result<Vec<isize>, PxdctError> {
        let length = n1 * n2;
        let mut indices = try_vec![0isize; length];
        let mut i = 0usize;
        for r in 0..n1 {
            for c in 0..n2 {
                let idx = c * n1 + r * n2;
                if idx < length {
                    indices[i] = idx as isize;
                } else {
                    let idx = 2 * length - idx;
                    indices[i] = -(idx as isize);
                }
                i += 1;
            }
        }

        Ok(indices)
    }

    /// Computes modulation offsets for each butterfly:
    /// M = |c·n1 − r·n2|
    fn pfa_modulation(n1: usize, n2: usize) -> Result<Vec<isize>, PxdctError> {
        let length = n1 * n2;
        let mut indices = try_vec![0isize; length];
        let mut i = 0usize;
        for r in 0..n1 {
            for c in 0..n2 {
                indices[i] =
                    (c as isize * n1 as isize - r as isize * n2 as isize).unsigned_abs() as isize;
                i += 1;
            }
        }
        Ok(indices)
    }

    /// Builds final output butterfly plan.
    /// For first row/column butterflies are skipped.
    fn pfa_output_indices(n1: usize, n2: usize) -> Result<PfaOutput, PxdctError> {
        let gains = Self::pfa_unity_gain(n1, n2)?;
        let modulation = Self::pfa_modulation(n1, n2)?;
        let mut indices = try_vec![0isize; n1 * n2];
        let modulation_cutoff = (n1 - 1) * (n2 - 1) / 2;
        let mut cumulative_modulation = 0usize;

        let mut i = 0usize;

        for r in 0..n1 {
            for c in 0..n2 {
                if r == 0 || c == 0 {
                    indices[i] = gains[r * n2 + c];
                } else {
                    if cumulative_modulation < modulation_cutoff {
                        let k = modulation[r * n2 + c];
                        indices[i] = k;
                    } else {
                        indices[i] = gains[r * n2 + c].abs();
                    }
                    cumulative_modulation += 1;
                }
                i += 1;
            }
        }

        Ok(PfaOutput {
            gains,
            modulation,
            indices,
        })
    }

    /// Encodes sign information directly into gain/modulation indices
    /// so that runtime butterfly has:
    ///     X = G + M
    ///     G = ±src[|G|]
    ///     M = ±src[|M|]
    fn pfa_encode_signs(v_gains: &mut [isize], v_modulation: &mut [isize]) {
        for (index, (gain, modulation)) in
            v_gains.iter_mut().zip(v_modulation.iter_mut()).enumerate()
        {
            if *gain != *modulation {
                // X = Gain + Modulation
                // hence if address = modulation -> Modulation = X - Gain
                // else if Gain = X - Modulation
                if index == *modulation as usize {
                    *gain = gain.abs();
                } else if *gain < 0 {
                    *gain = -gain.abs();
                } else {
                    *modulation = -*modulation;
                }
            }
        }
    }

    /// Converts CRT indices into final linear array positions
    /// so runtime never performs searches.
    fn pfa_remap_gain_modulation_to_linear_indices(
        indices: &[isize],
        gains: &mut [isize],
        modulations: &mut [isize],
    ) {
        for gain in gains.iter_mut() {
            let q = indices
                .iter()
                .position(|&x| x == gain.abs())
                .expect("PFA Algorithm doesn't converge") as isize;
            *gain = if gain.is_negative() { -q } else { q };
        }
        for modulation in modulations.iter_mut() {
            let q = indices
                .iter()
                .position(|&x| x == modulation.abs())
                .expect("PFA Algorithm doesn't converge") as isize;
            *modulation = if modulation.is_negative() { -q } else { q };
        }
    }

    pub(crate) fn new(width: usize, height: usize) -> Result<Self, PxdctError> {
        // algorithm is always configured here to run first cols, then rows, but
        // technically ordering doesn't matter
        let input = Self::pfa_input_indices(width, height)?;
        let mut output = Self::pfa_output_indices(height, width)?;

        // Since we're doing inversion of algorithm, in algorithm base we have an result index,
        // but we want to avoid search in runtime, so we're do inverse index remapping in
        // advance.
        Self::pfa_remap_gain_modulation_to_linear_indices(
            &output.indices,
            &mut output.gains,
            &mut output.modulation,
        );

        Self::pfa_encode_signs(&mut output.gains, &mut output.modulation);

        Ok(Self {
            input,
            gains: output.gains,
            modulation: output.modulation,
            indices: output.indices,
        })
    }
}

pub(crate) struct Dct2Coprime<T> {
    width_dct: Arc<dyn PxdctExecutor<T> + Send + Sync>,
    height_dct: Arc<dyn PxdctExecutor<T> + Send + Sync>,
    transposition: Arc<dyn Transposition<T> + Send + Sync>,
    width: usize,
    execution_length: usize,
    indexer: Dct2PfaIndexer,
    remapper: Arc<dyn Dct2OutputRemapper<T> + Send + Sync>,
}

impl<T: DctSample + Dct2RemapperFactory> Dct2Coprime<T>
where
    f64: AsPrimitive<T>,
{
    pub(crate) fn new(
        mut width_dct: Arc<dyn PxdctExecutor<T> + Send + Sync>,
        mut height_dct: Arc<dyn PxdctExecutor<T> + Send + Sync>,
    ) -> Result<Self, PxdctError> {
        if height_dct.length() > width_dct.length() {
            std::mem::swap(&mut width_dct, &mut height_dct);
        }
        let width = width_dct.length();
        let height = height_dct.length();
        if num_integer::gcd(width, height) != 1 {
            panic!(
                "This is an internal configuration error, co-prime algorithm is called on numbers {width}x{height} that are not relatively primes",
            );
        }
        let total_length = width * height;

        Ok(Self {
            width_dct,
            height_dct,
            width,
            execution_length: total_length,
            indexer: Dct2PfaIndexer::new(width, height)?,
            transposition: T::make_transpose(height, width),
            remapper: T::make_remapper(),
        })
    }
}

/// Scatter + butterfly fusion kernel
///
/// Executes:
///     dst[X] = ±src[G] ± src[M]
pub(crate) trait Dct2OutputRemapper<T> {
    fn remap_output(
        &self,
        src: &[T],
        dst: &mut [T],
        indices: &[isize],
        gains: &[isize],
        modulation: &[isize],
        width: usize,
    );
}

struct DefaultRemapper {}

impl<T: DctSample> Dct2OutputRemapper<T> for DefaultRemapper {
    fn remap_output(
        &self,
        src: &[T],
        dst: &mut [T],
        indices: &[isize],
        gains: &[isize],
        modulation: &[isize],
        width: usize,
    ) {
        let f_indices = &indices[..width];
        let f_gains = &gains[..width];

        // first row and first column is always itself and do not need butterflies

        for (&address, &gain) in f_indices.iter().zip(f_gains.iter()) {
            let r_gain = unsafe { *src.get_unchecked(gain.unsigned_abs()) };
            // X = Gain + Modulation
            // hence if address = modulation -> Modulation = X - Gain
            // else if Gain = X - Modulation
            unsafe {
                *dst.get_unchecked_mut(address as usize) = r_gain;
            }
        }

        let q_indices = &indices[width..];
        let q_gains = &gains[width..];
        let q_modulations = &modulation[width..];

        for ((address, gain), modulation) in q_indices
            .chunks_exact(width)
            .zip(q_gains.chunks_exact(width))
            .zip(q_modulations.chunks_exact(width))
        {
            unsafe {
                let r_gain = *src.get_unchecked(gain.get_unchecked(0).unsigned_abs());
                *dst.get_unchecked_mut(*address.get_unchecked(0) as usize) = r_gain;
            }

            let q_indices = &address[1..];
            let q_gains = &gain[1..];
            let q_modulations = &modulation[1..];

            for ((address, gain), modulation) in q_indices
                .chunks_exact(4)
                .zip(q_gains.chunks_exact(4))
                .zip(q_modulations.chunks_exact(4))
            {
                let g0 = gain[0];
                let m0 = modulation[0];
                let g1 = gain[1];
                let m1 = modulation[1];
                let g2 = gain[2];
                let m2 = modulation[2];
                let g3 = gain[3];
                let m3 = modulation[3];

                let r_gain0 = unsafe { *src.get_unchecked(g0.unsigned_abs()) };
                let r_modulation0 = unsafe { *src.get_unchecked(m0.unsigned_abs()) };
                let r_gain1 = unsafe { *src.get_unchecked(g1.unsigned_abs()) };
                let r_modulation1 = unsafe { *src.get_unchecked(m1.unsigned_abs()) };
                let r_gain2 = unsafe { *src.get_unchecked(g2.unsigned_abs()) };
                let r_modulation2 = unsafe { *src.get_unchecked(m2.unsigned_abs()) };
                let r_gain3 = unsafe { *src.get_unchecked(g3.unsigned_abs()) };
                let r_modulation3 = unsafe { *src.get_unchecked(m3.unsigned_abs()) };

                // X = Gain + Modulation
                // hence if address = modulation -> Modulation = X - Gain
                // else if Gain = X - Modulation
                unsafe {
                    *dst.get_unchecked_mut(address[0] as usize) =
                        r_modulation0.mulsigni(m0) + r_gain0.mulsigni(g0);
                    *dst.get_unchecked_mut(address[1] as usize) =
                        r_modulation1.mulsigni(m1) + r_gain1.mulsigni(g1);
                    *dst.get_unchecked_mut(address[2] as usize) =
                        r_modulation2.mulsigni(m2) + r_gain2.mulsigni(g2);
                    *dst.get_unchecked_mut(address[3] as usize) =
                        r_modulation3.mulsigni(m3) + r_gain3.mulsigni(g3);
                }
            }

            let q_indices = address.chunks_exact(4).remainder();
            let q_gains = gain.chunks_exact(4).remainder();
            let q_modulations = modulation.chunks_exact(4).remainder();

            for ((&address, &gain), &modulation) in q_indices
                .iter()
                .zip(q_gains.iter())
                .zip(q_modulations.iter())
            {
                let r_gain = unsafe { *src.get_unchecked(gain.unsigned_abs()) };
                let r_modulation = unsafe { *src.get_unchecked(modulation.unsigned_abs()) };

                // X = Gain + Modulation
                // hence if address = modulation -> Modulation = X - Gain
                // else if Gain = X - Modulation
                unsafe {
                    *dst.get_unchecked_mut(address as usize) =
                        r_modulation.mulsigni(modulation) + r_gain.mulsigni(gain);
                }
            }
        }
    }
}

pub(crate) trait Dct2RemapperFactory {
    fn make_remapper() -> Arc<dyn Dct2OutputRemapper<Self> + Send + Sync>;
}

impl Dct2RemapperFactory for f32 {
    fn make_remapper() -> Arc<dyn Dct2OutputRemapper<Self> + Send + Sync> {
        #[cfg(all(feature = "avx", target_pointer_width = "64", target_arch = "x86_64"))]
        {
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                use crate::avx::AvxPfaDct2Remapper;
                return Arc::new(AvxPfaDct2Remapper {});
            }
        }
        #[cfg(all(target_arch = "aarch64", feature = "neon", target_pointer_width = "64"))]
        {
            use crate::neon::NeonPfaDct2Remapper;
            Arc::new(NeonPfaDct2Remapper {})
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon", target_pointer_width = "64")))]
        {
            Arc::new(DefaultRemapper {})
        }
    }
}

impl Dct2RemapperFactory for f64 {
    fn make_remapper() -> Arc<dyn Dct2OutputRemapper<Self> + Send + Sync> {
        Arc::new(DefaultRemapper {})
    }
}

impl<T: DctSample> Dct2Coprime<T>
where
    f64: AsPrimitive<T>,
{
    fn remap_input(&self, src: &[T], dst: &mut [T]) {
        for (dst, index) in dst
            .chunks_exact_mut(4)
            .zip(self.indexer.input.chunks_exact(4))
        {
            unsafe {
                dst[0] = *src.get_unchecked(index[0]);
                dst[1] = *src.get_unchecked(index[1]);
                dst[2] = *src.get_unchecked(index[2]);
                dst[3] = *src.get_unchecked(index[3]);
            }
        }
        let dst = dst.chunks_exact_mut(4).into_remainder();
        let indexer = self.indexer.input.chunks_exact(4).remainder();
        for (dst, &index) in dst.iter_mut().zip(indexer.iter()) {
            unsafe {
                *dst = *src.get_unchecked(index);
            }
        }
    }

    fn remap_output(&self, src: &[T], dst: &mut [T]) {
        self.remapper.remap_output(
            src,
            dst,
            &self.indexer.indices,
            &self.indexer.gains,
            &self.indexer.modulation,
            self.width,
        );
    }
}

impl<T: DctSample> PxdctExecutor<T> for Dct2Coprime<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, data: &mut [T]) -> Result<(), PxdctError> {
        if !data.len().is_multiple_of(self.execution_length) {
            return Err(PxdctError::InvalidSizeMultiplier(
                data.len(),
                self.execution_length,
            ));
        }

        let mut f_scratch = try_vec![T::default(); self.execution_length * 2];
        let (scratch0, scratch1) = f_scratch.split_at_mut(self.execution_length);

        for chunk in data.chunks_exact_mut(self.execution_length) {
            self.remap_input(chunk, scratch0);
            self.height_dct.execute(scratch0)?;
            self.transposition.transpose(scratch0, scratch1);
            self.width_dct.execute(scratch1)?;
            self.remap_output(scratch1, chunk);
        }

        Ok(())
    }

    fn length(&self) -> usize {
        self.execution_length
    }
}

#[cfg(test)]
mod tests {}
