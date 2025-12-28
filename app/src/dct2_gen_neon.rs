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
use crate::dct3_gen::{almost_eq, almost_eq_exact, compute_twiddle};
use crate::solver::solve_expression;
use std::ops::Add;

pub(crate) fn generate_dct2_neon(len: usize) -> (String, Vec<String>) {
    let mut result = Vec::new();

    let mut twiddles = Vec::new();

    let mut builder = String::new();

    for output_index in 0..len {
        twiddles.push(compute_twiddle(output_index + 1, len * 4).conj());
    }

    let is_odd = !len.is_multiple_of(2);

    for output_index in 0..len / 4 {
        builder = builder.add(
            format!(
                "let block{} = vld1q_f32(chunk.get_unchecked({output_index} * 4..).as_ptr().cast());\n",
                output_index
            )
            .as_str(),
        );
    }
    let blocks_rem = len % 4;
    let full_blocks = len / 4;
    if blocks_rem == 1 {
        let final_block = full_blocks;
        builder = builder.add(
            format!(
                "let block{} = vcombine_f32(vld1_lane_f32::<0>(chunk.get_unchecked({final_block} * 4..).as_ptr().cast(), vdup_n_f32(0.)), vdup_n_f32(0.));;\n",
                final_block
            )
                .as_str(),
        );
    } else if blocks_rem == 2 {
        let final_block = full_blocks;
        builder = builder.add(
            format!(
                "let block{} = vcombine_f32(vld1_f32(chunk.get_unchecked({final_block} * 4..).as_ptr().cast()), vdup_n_f32(0.));\n",
                final_block
            )
                .as_str(),
        );
    } else if blocks_rem == 3 {
        let final_block = full_blocks;
        builder = builder.add(
            format!(
                "let block{} = vcombine_f32(vld1_f32(chunk.get_unchecked({final_block} * 4..).as_ptr().cast()), vld1_lane_f32::<0>(chunk.get_unchecked({final_block} * 4 + 2..).as_ptr().cast(), vdup_n_f32(0.)));\n",
                final_block
            )
                .as_str(),
        );
    }

    builder = builder.add("\n");
    let mut lanes = Vec::new();

    // reduce first
    builder = builder.add(
        "let mut y0 = vaddq_f32(block0, block1);\n"
            .to_string()
            .as_str(),
    );
    for i in 1..len / 4 {
        builder = builder.add(format!("y0 = vaddq_f32(y0, block{});\n", i + 1).as_str());
    }
    builder = builder.add(
        "let q = vpadd_f32(vget_low_f32(y0), vget_high_f32(y0));\n"
            .to_string()
            .as_str(),
    );
    builder = builder.add("let reduced_y0 = vpadd_f32(q, q);\n".to_string().as_str());
    builder = builder.add(
        "vst1_lane_f32::<0>(chunk.get_unchecked_mut(0..).as_mut_ptr().cast(), reduced_y0);\n\n"
            .to_string()
            .as_str(),
    );

    for i in 0..len {
        let block_index = i / 4;
        let block_position = i % 4;
        builder = builder.add(
            format!("let x{i} = vdup_laneq_f32::<{block_position}>( block{block_index});\n")
                .as_str(),
        );
    }

    for output_index in 1..len {
        let mut entry = 0.0;

        let mut line_builder = String::new();

        for input_index in 0..len {
            let cos_inner =
                (output_index as f64) * (input_index as f64 + 0.5) * std::f64::consts::PI
                    / (len as f64);
            let cos_twiddle = cos_inner.cos();

            let mut twiddle_idx: Option<usize> = None;
            let mut twiddle_img = false;
            let mut neg_twiddle = false;

            if output_index > 0 {
                for (i, twiddle) in twiddles.iter().enumerate() {
                    if almost_eq(twiddle.re, cos_twiddle) {
                        if almost_eq_exact(twiddle.re, -cos_twiddle) {
                            neg_twiddle = true;
                        }
                        twiddle_idx = Some(i);
                        twiddle_img = false;
                        break;
                    } else if almost_eq(twiddle.im, cos_twiddle) {
                        if almost_eq_exact(twiddle.im, -cos_twiddle) {
                            neg_twiddle = true;
                        }
                        twiddle_idx = Some(i);
                        twiddle_img = true;
                        break;
                    }
                }

                if twiddle_idx.is_none() {
                    panic!(
                        "Wasn't found required idx at output {output_index} input {input_index}"
                    );
                }

                if almost_eq(cos_twiddle, 0.) {
                    // do nothing
                } else if almost_eq(cos_twiddle, 1.) {
                    line_builder = line_builder.add(
                        format!(
                            "y{output_index} = {}(y{output_index}, x{input_index});\n",
                            if neg_twiddle { "vsub_f32" } else { "vadd_f32" },
                        )
                        .as_str(),
                    );
                } else {
                    if input_index == 0 {
                        line_builder = line_builder.add(
                                format!(
                                    "let mut y{output_index} = vmul_n_f32(x{input_index},{}self.twiddle{}.{});\n",
                                    if neg_twiddle { "-" } else { "" },
                                    twiddle_idx.unwrap(),
                                    if twiddle_img { "im" } else { "re" }
                                )
                                    .as_str(),
                            );
                    } else {
                        line_builder = line_builder.add(
                            format!(
                                "y{output_index} = {}(y{output_index}, x{input_index},self.twiddle{}.{});\n",
                                if neg_twiddle { "vfms_n_f32" } else { "vfma_n_f32" },
                                twiddle_idx.unwrap(),
                                if twiddle_img { "im" } else { "re" }
                            )
                                .as_str(),
                        );
                    }
                }
            }
        }

        line_builder = line_builder.add(format!("vst1_lane_f32::<0>(chunk.get_unchecked_mut({output_index}..).as_mut_ptr().cast(), y{output_index});\n").as_str());

        lanes.push(line_builder.to_string());

        if is_odd {
            builder = builder.add(&line_builder);
            builder = builder.add("\n");
        }

        result.push(entry);
    }

    if !is_odd {
        let solved_expressions = solve_expression(&lanes);

        builder = builder.add(solved_expressions.as_str());
    }

    builder = builder.add("\n");

    (builder, lanes)
}
