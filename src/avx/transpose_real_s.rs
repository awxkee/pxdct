/*
 * // Copyright (c) Radzivon Bartoshyk 02/2026. All rights reserved.
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
use crate::avx::storef::{AvxFullF, AvxLanesF, AvxStoreF, AvxTailF};
use crate::avx::util::shuffle;
use crate::transpose::{Transposition, validate_transpose_buffers};
use std::arch::x86_64::*;

#[inline]
#[target_feature(enable = "avx2")]
fn transpose_8x8_f32(store: [AvxStoreF; 8]) -> [AvxStoreF; 8] {
    let t0 = _mm256_unpacklo_ps(store[0].v, store[1].v);
    let t1 = _mm256_unpackhi_ps(store[0].v, store[1].v);
    let t2 = _mm256_unpacklo_ps(store[2].v, store[3].v);
    let t3 = _mm256_unpackhi_ps(store[2].v, store[3].v);
    let t4 = _mm256_unpacklo_ps(store[4].v, store[5].v);
    let t5 = _mm256_unpackhi_ps(store[4].v, store[5].v);
    let t6 = _mm256_unpacklo_ps(store[6].v, store[7].v);
    let t7 = _mm256_unpackhi_ps(store[6].v, store[7].v);
    const FLAG_1: i32 = shuffle(1, 0, 1, 0);
    let tt0 = _mm256_shuffle_ps::<FLAG_1>(t0, t2);
    const FLAG_2: i32 = shuffle(3, 2, 3, 2);
    let tt1 = _mm256_shuffle_ps::<FLAG_2>(t0, t2);
    const FLAG_3: i32 = shuffle(1, 0, 1, 0);
    let tt2 = _mm256_shuffle_ps::<FLAG_3>(t1, t3);
    const FLAG_4: i32 = shuffle(3, 2, 3, 2);
    let tt3 = _mm256_shuffle_ps::<FLAG_4>(t1, t3);
    const FLAG_5: i32 = shuffle(1, 0, 1, 0);
    let tt4 = _mm256_shuffle_ps::<FLAG_5>(t4, t6);
    const FLAG_6: i32 = shuffle(3, 2, 3, 2);
    let tt5 = _mm256_shuffle_ps::<FLAG_6>(t4, t6);
    const FLAG_7: i32 = shuffle(1, 0, 1, 0);
    let tt6 = _mm256_shuffle_ps::<FLAG_7>(t5, t7);
    const FLAG_8: i32 = shuffle(3, 2, 3, 2);
    let tt7 = _mm256_shuffle_ps::<FLAG_8>(t5, t7);
    let r0 = _mm256_permute2f128_ps::<0x20>(tt0, tt4);
    let r1 = _mm256_permute2f128_ps::<0x20>(tt1, tt5);
    let r2 = _mm256_permute2f128_ps::<0x20>(tt2, tt6);
    let r3 = _mm256_permute2f128_ps::<0x20>(tt3, tt7);
    let r4 = _mm256_permute2f128_ps::<0x31>(tt0, tt4);
    let r5 = _mm256_permute2f128_ps::<0x31>(tt1, tt5);
    let r6 = _mm256_permute2f128_ps::<0x31>(tt2, tt6);
    let r7 = _mm256_permute2f128_ps::<0x31>(tt3, tt7);

    [
        AvxStoreF::raw(r0),
        AvxStoreF::raw(r1),
        AvxStoreF::raw(r2),
        AvxStoreF::raw(r3),
        AvxStoreF::raw(r4),
        AvxStoreF::raw(r5),
        AvxStoreF::raw(r6),
        AvxStoreF::raw(r7),
    ]
}

pub(crate) struct AvxTransposeFReal4x4 {
    pub(crate) width: usize,
    pub(crate) height: usize,
}

impl AvxTransposeFReal4x4 {
    #[target_feature(enable = "avx2")]
    fn transpose_block<R: AvxLanesF, C: AvxLanesF>(
        &self,
        src: &[f32],
        dst: &mut [f32],
        y: usize,
        x: usize,
        width: usize,
        height: usize,
        rows: R,
        columns: C,
    ) {
        let input_stride = width;
        let output_stride = height;
        let row_count = rows.len();
        let column_count = columns.len();

        debug_assert!(y + row_count <= height);
        debug_assert!(x + column_count <= width);

        let src = unsafe { src.get_unchecked(input_stride * y..) };
        let block_src = unsafe { src.get_unchecked(x..) };
        let block_dst = unsafe { dst.get_unchecked_mut(y + output_stride * x..) };

        let zbuffer = std::array::from_fn(|row| {
            if row < row_count {
                AvxStoreF::load_lanes(columns, unsafe {
                    block_src.get_unchecked(row * input_stride..)
                })
            } else {
                AvxStoreF::zero()
            }
        });
        let transposed = transpose_8x8_f32(zbuffer);

        for (column, value) in transposed.into_iter().take(column_count).enumerate() {
            value.write_lanes(rows, unsafe {
                block_dst.get_unchecked_mut(output_stride * column..)
            });
        }
    }

    #[target_feature(enable = "avx2")]
    fn transpose_y<R: AvxLanesF>(
        &self,
        src: &[f32],
        dst: &mut [f32],
        y: usize,
        width: usize,
        height: usize,
        rows: R,
    ) {
        const BLOCK_SIZE_X: usize = 8;
        let mut x = 0usize;

        while x + BLOCK_SIZE_X <= width {
            self.transpose_block(src, dst, y, x, width, height, rows, AvxFullF);
            x += BLOCK_SIZE_X;
        }

        let remainder = width - x;
        if remainder != 0 {
            self.transpose_block(
                src,
                dst,
                y,
                x,
                width,
                height,
                rows,
                AvxTailF::new(remainder),
            );
        }
    }
}

impl Transposition<f32> for AvxTransposeFReal4x4 {
    fn transpose(&self, input: &[f32], output: &mut [f32]) {
        validate_transpose_buffers(input, output, self.width, self.height);

        const BLOCK_SIZE_Y: usize = 8;
        let mut y = 0usize;

        unsafe {
            while y + BLOCK_SIZE_Y <= self.height {
                self.transpose_y(input, output, y, self.width, self.height, AvxFullF);
                y += BLOCK_SIZE_Y;
            }

            let rem_y = self.height - y;
            if rem_y > 0 {
                self.transpose_y(
                    input,
                    output,
                    y,
                    self.width,
                    self.height,
                    AvxTailF::new(rem_y),
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transpose_handles_every_avx_tail_width() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }

        for height in 1..=17 {
            for width in 1..=17 {
                let input: Vec<f32> = (0..width * height).map(|x| x as f32).collect();
                let mut output = vec![-1.; input.len()];
                AvxTransposeFReal4x4 { width, height }.transpose(&input, &mut output);

                for y in 0..height {
                    for x in 0..width {
                        assert_eq!(output[x * height + y], input[y * width + x]);
                    }
                }
            }
        }
    }
}
