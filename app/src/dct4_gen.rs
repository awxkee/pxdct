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
use crate::dct3_gen::{almost_eq, almost_eq_exact};
use crate::solver::solve_expression_arr;
use primal_check::miller_rabin;
use pxfm::f_cospi;
use std::ops::Add;

pub(crate) fn generate_dct4(len: usize) -> (String, Vec<String>) {
    let mut result = Vec::new();

    let mut twiddles = Vec::new();

    let mut builder = String::new();

    for output_index in 0..len {
        twiddles.push(f_cospi((output_index as f64 + 0.5) * 0.5 / len as f64));
    }

    let is_odd = !len.is_multiple_of(2);

    for output_index in 0..len {
        builder = builder.add(format!("let x{} = data[{output_index}];\n", output_index).as_str());
    }

    builder = builder.add("\n");
    let mut lanes = Vec::new();

    let is_prime = miller_rabin(len as u64);
    let fma_cutoff = 5;
    let fuse_first = is_odd && is_prime;

    for output_index in 0..len {
        let mut entry = 0.0;

        let mut line_builder = String::new();

        let mut need_to_close = 0usize;

        if fuse_first {
            line_builder = line_builder.add(format!("let y{} = fmla(", output_index).as_str());
            need_to_close = 1;
        } else {
            line_builder = line_builder.add(format!("let y{} = ", output_index).as_str());
        }

        for input_index in 0..len {
            let cos_inner =
                (output_index as f64 + 0.5) * (input_index as f64 + 0.5) * std::f64::consts::PI
                    / (len as f64);
            let cos_twiddle = cos_inner.cos();

            let mut twiddle_idx: Option<usize> = None;
            let mut neg_twiddle = false;

            for (i, twiddle) in twiddles.iter().enumerate() {
                if almost_eq(*twiddle, cos_twiddle) {
                    if almost_eq_exact(*twiddle, -cos_twiddle) {
                        neg_twiddle = true;
                    }
                    twiddle_idx = Some(i);
                    break;
                }
            }

            if twiddle_idx.is_none() {
                panic!("Wasn't found required idx at output {output_index} input {input_index}");
            }

            if input_index != len - 1 {
                if almost_eq(cos_twiddle, 0.) {
                    // do nothing
                } else if almost_eq(cos_twiddle, 1.) {
                    if len < fma_cutoff || input_index == 1 {
                        line_builder = line_builder.add(
                            format!(
                                "{}1f64.as_(), x{input_index}, fmla(",
                                if cos_twiddle < 0. { "-" } else { "" },
                            )
                            .as_str(),
                        );
                        need_to_close += 1;
                    } else {
                        line_builder = line_builder.add(
                            format!(
                                " {} x{input_index}",
                                if cos_twiddle < 0. {
                                    "-"
                                } else {
                                    if input_index > 1 { "+" } else { "" }
                                },
                            )
                            .as_str(),
                        );
                    }
                } else {
                    if len < fma_cutoff || (input_index == 0 && fuse_first) {
                        if len >= fma_cutoff && fuse_first {
                            line_builder = line_builder.add(
                                format!(
                                    "{}self.twiddles[{}], x{input_index},",
                                    if neg_twiddle { "-" } else { "" },
                                    twiddle_idx.unwrap(),
                                )
                                .as_str(),
                            );
                        } else {
                            line_builder = line_builder.add(
                                format!(
                                    "{}self.twiddles[{}], x{input_index}, {}",
                                    if neg_twiddle { "-" } else { "" },
                                    twiddle_idx.unwrap(),
                                    if input_index + 2 < len { "fmla(" } else { "" }
                                )
                                .as_str(),
                            );
                            if input_index + 2 < len {
                                need_to_close += 1;
                            }
                        }
                    } else {
                        line_builder = line_builder.add(
                            format!(
                                " {} self.twiddles[{}] * x{input_index}",
                                if neg_twiddle {
                                    "-"
                                } else {
                                    if input_index > if fuse_first { 1 } else { 0 } {
                                        "+"
                                    } else {
                                        ""
                                    }
                                },
                                twiddle_idx.unwrap(),
                            )
                            .as_str(),
                        );
                    }
                }
            } else {
                // last item
                if almost_eq(cos_twiddle, 0.) {
                    // do nothing
                } else {
                    line_builder = line_builder.add(
                        format!(
                            "{}self.twiddles[{}] * x{input_index}",
                            if neg_twiddle {
                                "-"
                            } else {
                                if fma_cutoff < len { "+" } else { "" }
                            },
                            twiddle_idx.unwrap(),
                        )
                        .as_str(),
                    );
                }
            }
        }

        if need_to_close > 0 {
            for _ in 0..need_to_close {
                line_builder = line_builder.add(")");
            }
        }
        line_builder = line_builder.add(";");

        lanes.push(line_builder.to_string());

        if is_prime {
            builder = builder.add(&line_builder);
            builder = builder.add("\n");
        }

        result.push(entry);
    }

    if !is_prime {
        let solved_expressions = solve_expression_arr(&lanes);

        builder = builder.add(solved_expressions.as_str());
    }

    builder = builder.add("\n");

    for i in 0..len {
        builder = builder.add(format!("data[{i}] = y{i};\n").as_str());
    }

    (builder, lanes)
}
