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
use crate::neon::transpose::transpose_4x4;
use crate::neon::util::NeonStoreF;
use crate::transpose::{Transposition, validate_transpose_buffers};
use std::arch::aarch64::{float32x4x4_t, vdupq_n_f32};

#[inline(always)]
fn transpose_4x11_f32(store: [NeonStoreF; 11]) -> [NeonStoreF; 12] {
    let q0 = transpose_4x4(float32x4x4_t(
        store[0].v, store[1].v, store[2].v, store[3].v,
    ));
    let q1 = transpose_4x4(float32x4x4_t(
        store[4].v, store[5].v, store[6].v, store[7].v,
    ));
    let q2 = transpose_4x4(float32x4x4_t(store[8].v, store[9].v, store[10].v, unsafe {
        vdupq_n_f32(0.)
    }));
    [
        NeonStoreF::raw(q0.0),
        NeonStoreF::raw(q0.1),
        NeonStoreF::raw(q0.2),
        NeonStoreF::raw(q0.3),
        NeonStoreF::raw(q1.0),
        NeonStoreF::raw(q1.1),
        NeonStoreF::raw(q1.2),
        NeonStoreF::raw(q1.3),
        NeonStoreF::raw(q2.0),
        NeonStoreF::raw(q2.1),
        NeonStoreF::raw(q2.2),
        NeonStoreF::raw(q2.3),
    ]
}

#[inline(always)]
fn transpose_4x6_f32(store: [NeonStoreF; 6]) -> [NeonStoreF; 8] {
    let q0 = transpose_4x4(float32x4x4_t(
        store[0].v, store[1].v, store[2].v, store[3].v,
    ));
    let q1 = transpose_4x4(float32x4x4_t(
        store[4].v,
        store[5].v,
        unsafe { vdupq_n_f32(0.) },
        unsafe { vdupq_n_f32(0.) },
    ));
    [
        NeonStoreF::raw(q0.0),
        NeonStoreF::raw(q0.1),
        NeonStoreF::raw(q0.2),
        NeonStoreF::raw(q0.3),
        NeonStoreF::raw(q1.0),
        NeonStoreF::raw(q1.1),
        NeonStoreF::raw(q1.2),
        NeonStoreF::raw(q1.3),
    ]
}

#[inline(always)]
fn transpose_4x7_f32(store: [NeonStoreF; 7]) -> [NeonStoreF; 8] {
    let q0 = transpose_4x4(float32x4x4_t(
        store[0].v, store[1].v, store[2].v, store[3].v,
    ));
    let q1 = transpose_4x4(float32x4x4_t(store[4].v, store[5].v, store[6].v, unsafe {
        vdupq_n_f32(0.)
    }));
    [
        NeonStoreF::raw(q0.0),
        NeonStoreF::raw(q0.1),
        NeonStoreF::raw(q0.2),
        NeonStoreF::raw(q0.3),
        NeonStoreF::raw(q1.0),
        NeonStoreF::raw(q1.1),
        NeonStoreF::raw(q1.2),
        NeonStoreF::raw(q1.3),
    ]
}

#[inline(always)]
fn transpose_4x5_f32(store: [NeonStoreF; 5]) -> [NeonStoreF; 8] {
    let q0 = transpose_4x4(float32x4x4_t(
        store[0].v, store[1].v, store[2].v, store[3].v,
    ));
    let q1 = transpose_4x4(float32x4x4_t(
        store[4].v,
        unsafe { vdupq_n_f32(0.) },
        unsafe { vdupq_n_f32(0.) },
        unsafe { vdupq_n_f32(0.) },
    ));
    [
        NeonStoreF::raw(q0.0),
        NeonStoreF::raw(q0.1),
        NeonStoreF::raw(q0.2),
        NeonStoreF::raw(q0.3),
        NeonStoreF::raw(q1.0),
        NeonStoreF::raw(q1.1),
        NeonStoreF::raw(q1.2),
        NeonStoreF::raw(q1.3),
    ]
}

pub(crate) fn transpose_height_block_executor2_f32_odd<
    const X_BLOCK_SIZE: usize,
    const Y_BLOCK_SIZE: usize,
    const Y_ODD_BLOCK_SIZE: usize,
    E: Fn([NeonStoreF; Y_BLOCK_SIZE]) -> [NeonStoreF; Y_ODD_BLOCK_SIZE],
>(
    input: &[f32],
    input_stride: usize,
    output: &mut [f32],
    output_stride: usize,
    width: usize,
    height: usize,
    start_y: usize,
    exec: E,
) -> usize {
    let mut y = start_y;
    unsafe {
        let mut store = [NeonStoreF::default(); Y_BLOCK_SIZE];

        let rem = Y_BLOCK_SIZE % 4;
        let quo = Y_BLOCK_SIZE / 4;

        while y + Y_BLOCK_SIZE <= height {
            let input_y = y;

            let src = input.get_unchecked(input_stride * input_y..);

            let mut x = 0usize;

            if rem == 0 {
                while x + X_BLOCK_SIZE <= width {
                    let output_x = x;

                    let src = src.get_unchecked(x..);
                    let dst = output.get_unchecked_mut(y + output_stride * output_x..);

                    for i in 0..Y_BLOCK_SIZE {
                        store[i] = NeonStoreF::load(src.get_unchecked(i * input_stride..));
                    }

                    let q = exec(store);

                    for i in 0..quo {
                        q[i * 4].write(dst.get_unchecked_mut(i * 4..));
                        q[i * 4 + 1].write(dst.get_unchecked_mut(i * 4 + output_stride..));
                        q[i * 4 + 2].write(dst.get_unchecked_mut(i * 4 + output_stride * 2..));
                        q[i * 4 + 3].write(dst.get_unchecked_mut(i * 4 + output_stride * 3..));
                    }

                    x += X_BLOCK_SIZE;
                }
            } else if rem == 1 {
                while x + X_BLOCK_SIZE <= width {
                    let output_x = x;

                    let src = src.get_unchecked(x..);
                    let dst = output.get_unchecked_mut(y + output_stride * output_x..);

                    for i in 0..Y_BLOCK_SIZE {
                        store[i] = NeonStoreF::load(src.get_unchecked(i * input_stride..));
                    }

                    let q = exec(store);

                    for i in 0..quo {
                        q[i * 4].write(dst.get_unchecked_mut(i * 4..));
                        q[i * 4 + 1].write(dst.get_unchecked_mut(i * 4 + output_stride..));
                        q[i * 4 + 2].write(dst.get_unchecked_mut(i * 4 + output_stride * 2..));
                        q[i * 4 + 3].write(dst.get_unchecked_mut(i * 4 + output_stride * 3..));
                    }

                    q[quo * 4].write1(dst.get_unchecked_mut(quo * 4..));
                    q[quo * 4 + 1].write1(dst.get_unchecked_mut(quo * 4 + output_stride..));
                    q[quo * 4 + 2].write1(dst.get_unchecked_mut(quo * 4 + output_stride * 2..));
                    q[quo * 4 + 3].write1(dst.get_unchecked_mut(quo * 4 + output_stride * 3..));

                    x += X_BLOCK_SIZE;
                }
            } else if rem == 2 {
                while x + X_BLOCK_SIZE <= width {
                    let output_x = x;

                    let src = src.get_unchecked(x..);
                    let dst = output.get_unchecked_mut(y + output_stride * output_x..);

                    for i in 0..Y_BLOCK_SIZE {
                        store[i] = NeonStoreF::load(src.get_unchecked(i * input_stride..));
                    }

                    let q = exec(store);

                    for i in 0..quo {
                        q[i * 4].write(dst.get_unchecked_mut(i * 4..));
                        q[i * 4 + 1].write(dst.get_unchecked_mut(i * 4 + output_stride..));
                        q[i * 4 + 2].write(dst.get_unchecked_mut(i * 4 + output_stride * 2..));
                        q[i * 4 + 3].write(dst.get_unchecked_mut(i * 4 + output_stride * 3..));
                    }

                    q[quo * 4].write2(dst.get_unchecked_mut(quo * 4..));
                    q[quo * 4 + 1].write2(dst.get_unchecked_mut(quo * 4 + output_stride..));
                    q[quo * 4 + 2].write2(dst.get_unchecked_mut(quo * 4 + output_stride * 2..));
                    q[quo * 4 + 3].write2(dst.get_unchecked_mut(quo * 4 + output_stride * 3..));

                    x += X_BLOCK_SIZE;
                }
            } else if rem == 3 {
                while x + X_BLOCK_SIZE <= width {
                    let output_x = x;

                    let src = src.get_unchecked(x..);
                    let dst = output.get_unchecked_mut(y + output_stride * output_x..);

                    for i in 0..Y_BLOCK_SIZE {
                        store[i] = NeonStoreF::load(src.get_unchecked(i * input_stride..));
                    }

                    let q = exec(store);

                    for i in 0..quo {
                        q[i * 4].write(dst.get_unchecked_mut(i * 4..));
                        q[i * 4 + 1].write(dst.get_unchecked_mut(i * 4 + output_stride..));
                        q[i * 4 + 2].write(dst.get_unchecked_mut(i * 4 + output_stride * 2..));
                        q[i * 4 + 3].write(dst.get_unchecked_mut(i * 4 + output_stride * 3..));
                    }

                    q[quo * 4].write3(dst.get_unchecked_mut(quo * 4..));
                    q[quo * 4 + 1].write3(dst.get_unchecked_mut(quo * 4 + output_stride..));
                    q[quo * 4 + 2].write3(dst.get_unchecked_mut(quo * 4 + output_stride * 2..));
                    q[quo * 4 + 3].write3(dst.get_unchecked_mut(quo * 4 + output_stride * 3..));

                    x += X_BLOCK_SIZE;
                }
            }

            if x < width {
                let rem_x = width - x;
                let output_x = x;

                let src = src.get_unchecked(x..);
                let dst = output.get_unchecked_mut(y + output_stride * output_x..);

                if rem_x == 1 {
                    for i in 0..Y_BLOCK_SIZE {
                        store[i] = NeonStoreF::load1(src.get_unchecked(i * input_stride..));
                    }
                } else if rem_x == 2 {
                    for i in 0..Y_BLOCK_SIZE {
                        store[i] = NeonStoreF::load2(src.get_unchecked(i * input_stride..));
                    }
                } else if rem_x == 3 {
                    for i in 0..Y_BLOCK_SIZE {
                        store[i] = NeonStoreF::load3(src.get_unchecked(i * input_stride..));
                    }
                }

                let q = exec(store);

                if rem_x == 1 {
                    for i in 0..quo {
                        q[i * 4].write(dst.get_unchecked_mut(i * 4..));
                    }

                    if rem == 1 {
                        q[quo * 4].write1(dst.get_unchecked_mut(quo * 4..));
                    } else if rem == 2 {
                        q[quo * 4].write2(dst.get_unchecked_mut(quo * 4..));
                    } else if rem == 3 {
                        q[quo * 4].write3(dst.get_unchecked_mut(quo * 4..));
                    }
                } else if rem_x == 2 {
                    for i in 0..quo {
                        q[i * 4].write(dst.get_unchecked_mut(i * 4..));
                        q[i * 4 + 1].write(dst.get_unchecked_mut(i * 4 + output_stride..));
                    }

                    if rem == 1 {
                        q[quo * 4].write1(dst.get_unchecked_mut(quo * 4..));
                        q[quo * 4 + 1].write1(dst.get_unchecked_mut(quo * 4 + output_stride..));
                    } else if rem == 2 {
                        q[quo * 4].write2(dst.get_unchecked_mut(quo * 4..));
                        q[quo * 4 + 1].write2(dst.get_unchecked_mut(quo * 4 + output_stride..));
                    } else if rem == 3 {
                        q[quo * 4].write3(dst.get_unchecked_mut(quo * 4..));
                        q[quo * 4 + 1].write3(dst.get_unchecked_mut(quo * 4 + output_stride..));
                    }
                } else if rem_x == 3 {
                    for i in 0..quo {
                        q[i * 4].write(dst.get_unchecked_mut(i * 4..));
                        q[i * 4 + 1].write(dst.get_unchecked_mut(i * 4 + output_stride..));
                        q[i * 4 + 2].write(dst.get_unchecked_mut(i * 4 + output_stride * 2..));
                    }

                    if rem == 1 {
                        q[quo * 4].write1(dst.get_unchecked_mut(quo * 4..));
                        q[quo * 4 + 1].write1(dst.get_unchecked_mut(quo * 4 + output_stride..));
                        q[quo * 4 + 2].write1(dst.get_unchecked_mut(quo * 4 + output_stride * 2..));
                    } else if rem == 2 {
                        q[quo * 4].write2(dst.get_unchecked_mut(quo * 4..));
                        q[quo * 4 + 1].write2(dst.get_unchecked_mut(quo * 4 + output_stride..));
                        q[quo * 4 + 2].write2(dst.get_unchecked_mut(quo * 4 + output_stride * 2..));
                    } else if rem == 3 {
                        q[quo * 4].write3(dst.get_unchecked_mut(quo * 4..));
                        q[quo * 4 + 1].write3(dst.get_unchecked_mut(quo * 4 + output_stride..));
                        q[quo * 4 + 2].write3(dst.get_unchecked_mut(quo * 4 + output_stride * 2..));
                    }
                }
            }

            y += Y_BLOCK_SIZE;
        }
    }

    y
}

type FunctionOddF<const N: usize, const ODD: usize> = fn([NeonStoreF; N]) -> [NeonStoreF; ODD];

macro_rules! define_transpose_oddf {
    ($rule_name: ident, $complex_type: ident, $rot_name: ident, $block_width: expr, $block_height: expr) => {
        #[derive(Default)]
        pub(crate) struct $rule_name {
            pub(crate) width: usize,
            pub(crate) height: usize,
        }

        impl Transposition<$complex_type> for $rule_name {
            fn transpose(&self, input: &[$complex_type], output: &mut [$complex_type]) {
                validate_transpose_buffers(input, output, self.width, self.height);

                const R: usize = ($block_height as usize).div_ceil(4) * 4;
                transpose_height_block_executor2_f32_odd::<
                    $block_width,
                    $block_height,
                    R,
                    FunctionOddF<$block_height, R>,
                >(
                    input,
                    self.width,
                    output,
                    self.height,
                    self.width,
                    self.height,
                    0,
                    $rot_name,
                );
            }
        }
    };
}

define_transpose_oddf!(NeonTransposeNx11F32, f32, transpose_4x11_f32, 4, 11);
define_transpose_oddf!(NeonTransposeNx7F32, f32, transpose_4x7_f32, 4, 7);
define_transpose_oddf!(NeonTransposeNx6F32, f32, transpose_4x6_f32, 4, 6);
define_transpose_oddf!(NeonTransposeNx5F32, f32, transpose_4x5_f32, 4, 5);
