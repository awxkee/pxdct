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
use num_complex::Complex;
use pxdct::{PxdctError, PxdctExecutor};
use rustdct::num_traits::{AsPrimitive, Float, One, Zero};
use std::fmt::format;
use std::ops::Add;
use std::sync::Arc;

pub(crate) struct Dct2RadixqGenerator {
    inner_layer: Vec<Complex<f32>>,
    fusion_layer: Vec<Complex<f32>>,
    execution_length: usize,
    q: usize,
    inner_blocks: usize,
}

impl Dct2RadixqGenerator {
    pub(crate) fn new(len: usize, q_len: usize) -> Dct2RadixqGenerator {
        // assert_eq!(
        //     len,
        //     q_dct.length() ,
        //     "Invalid DCT was received, third size is not multiple of full size"
        // );

        let inner_blocks = (len - q_len) / q_len;

        let subseq_q = len / q_len;

        let inner_groups = ((subseq_q.saturating_sub(3)) / 2) + 1;

        let mut inner_layer = vec![Complex::<f32>::zero(); (q_len - 1) * inner_groups];

        let mut fusion_layer =
            vec![Complex::<f32>::zero(); inner_blocks * q_len / 2 * inner_groups];

        Dct2RadixqGenerator {
            inner_layer,
            fusion_layer,
            execution_length: len,
            q: q_len,
            inner_blocks,
        }
    }
}

impl Dct2RadixqGenerator {
    pub fn execute(&self) {
        let mut scratchf = vec![f32::default(); self.execution_length * 2];

        let (scratch, scratch2) = scratchf.split_at_mut(self.execution_length);

        let modules = self.execution_length / self.q;

        let inner_groups = (modules.saturating_sub(3)) / 2 + 1;

        let a_module = ((modules - 1) / 2).max(1);
        let fusion_shift = inner_groups * self.q;
        let inner_shift = inner_groups;

        let (a_buffer, c_s_buffer) = scratch.split_at_mut(self.q);
        let (c_buffer, s_buffer) = c_s_buffer.split_at_mut(self.q * (self.inner_blocks / 2));

        let mut a_builder = "let mut a_buffer = [".to_string();

        for (n, dst) in a_buffer.iter_mut().enumerate() {
            unsafe {
                a_builder = a_builder.add(&format!("data[{}],", n * modules + a_module));
            }
        }

        a_builder = a_builder.add("];");
        println!("{}", a_builder);
        println!("self.bf{}.exec(&mut a_buffer);", self.q);

        for (m, (c_buffer, s_buffer)) in c_buffer
            .chunks_exact_mut(self.q)
            .zip(s_buffer.chunks_exact_mut(self.q))
            .enumerate()
        {
            let mut c_builder = format!("let mut c_buffer{} = [", m).to_string();
            let mut s_builder = format!("let mut s_buffer{} = [", m).to_string();

            let mut sign = f32::one();
            for (n, (c_dst, s_dst)) in c_buffer.iter_mut().zip(s_buffer.iter_mut()).enumerate() {
                c_builder = c_builder.add(&format!(
                    "data[{}] + data[{}],",
                    modules * n + m,
                    modules * n + modules - m - 1
                ));
                let expr = if sign < 0. {
                    format!(
                        "data[{}] - data[{}],",
                        modules * n + modules - m - 1,
                        modules * n + m
                    )
                } else {
                    format!(
                        "data[{}] - data[{}],",
                        modules * n + m,
                        modules * n + modules - m - 1
                    )
                };
                s_builder = s_builder.add(&expr);

                sign = -sign;
            }

            c_builder = c_builder.add("];");
            println!("{}", c_builder);

            println!("self.bf{}.exec(&mut c_buffer{m});", self.q);

            s_builder = s_builder.add("];");
            println!("{}", s_builder);

            println!("self.bf{}.exec(&mut s_buffer{m});", self.q);
        }

        let (a_buffer, c_s_buffer) = scratch.split_at_mut(self.q);
        let (c_buffer, s_buffer) = c_s_buffer.split_at_mut(self.q * (self.inner_blocks / 2));

        let mut j = 0usize;

        while j < a_module {
            let fusion_layer = unsafe {
                self.fusion_layer
                    .get_unchecked(fusion_shift * j..j * fusion_shift + fusion_shift)
            };

            let twiddles_start = fusion_shift * j;

            let mut c0 = format!(
                "let mut c0 = c_buffer0[0] * self.fusion_layer[{}].re;\n",
                twiddles_start
            );
            c0 = c0.add(&format!(
                "let mut s0 = s_buffer0[0] * self.fusion_layer[{}].im;\n",
                twiddles_start
            ));

            let mut buffer_offset = self.q;

            let mut m = 1usize;
            while m < inner_groups {
                let buffer_idx = buffer_offset / self.q;
                c0 = c0.add(&format!(
                    "c0 = fmla(c_buffer{buffer_idx}[0], self.fusion_layer[{m}].re, c0);\n"
                ));
                c0 = c0.add(&format!(
                    "s0 = fmla(s_buffer{buffer_idx}[0], self.fusion_layer[{m}].im, s0);\n"
                ));
                m += 1;
                buffer_offset += self.q;
            }

            println!("{}", c0);

            println!("let a0 = a_buffer[0];");

            if j == 0 {
                println!("data[0] = c0 + a0;");
            }

            unsafe {
                let idx = j * self.q * 2 + self.q;
                println!("data[{}] = s0;", idx);
            }
            unsafe {
                let idx = j * self.q * 2 + self.q + self.q;
                println!("data[{}] = -fmla(-0.5f64.as_(), c0, a0);", idx);
            }

            if j == 0 && a_module > 1 {
                self.fuse_layers(inner_groups, inner_shift, twiddles_start);
            } else {
                //     if a_module == 1 {
                //         self.fuse_layers_next::<true>(
                //             inner_groups,
                //             inner_shift,
                //             chunk,
                //             a_buffer,
                //             c_buffer,
                //             s_buffer,
                //             c_buffer2,
                //             s_buffer2,
                //             fusion_layer,
                //             j,
                //             true,
                //         );
                //     } else {
                self.fuse_layers_next::<false>(
                    inner_groups,
                    inner_shift,
                    twiddles_start,
                    j,
                    true,
                    // j == a_module - 1,
                );
                //     }
            }

            j += 1;
        }

        // self.combine_layers(chunk, a_buffer, c_buffer2, s_buffer2, a_module);
    }

    #[inline]
    fn length(&self) -> usize {
        self.execution_length
    }
}

impl Dct2RadixqGenerator {
    //     fn combine_layers(
    //         &self,
    //         chunk: &mut [f32],
    //         a_buffer: &mut [f32],
    //         c_buffer: &mut [f32],
    //         s_buffer: &mut [f32],
    //         modules: usize,
    //     ) {
    //         let mut i = 1usize;
    //         while i < self.q {
    //             let mut j = 0usize;
    //
    //             let a = unsafe { *a_buffer.get_unchecked(i) };
    //             let modulated_a = unsafe { *chunk.get_unchecked_mut(i) };
    //
    //             while j < modules {
    //                 let mut dc = unsafe { *c_buffer.get_unchecked(self.q * j + i) };
    //                 let mut ds = unsafe { *s_buffer.get_unchecked(self.q * j + self.q - i) };
    //
    //                 dc = -fmla(-0.5f64.as_(), dc, a);
    //                 ds = fmla(2f64.as_(), ds, -modulated_a);
    //                 dc = fmla(2f64.as_(), dc, -ds);
    //
    //                 unsafe {
    //                     let idx = j * self.q * 2 + self.q * 2 - i;
    //                     *chunk.get_unchecked_mut(idx) = ds;
    //                 }
    //                 unsafe {
    //                     let idx = j * self.q * 2 + self.q + self.q + i;
    //                     *chunk.get_unchecked_mut(idx) = dc;
    //                 }
    //
    //                 j += 1;
    //             }
    //
    //             i += 1;
    //         }
    //     }
    //
    #[inline]
    fn fuse_layers(&self, inner_groups: usize, inner_shift: usize, twiddles_start: usize) {
        let mut i = 1usize;
        while i < self.q {
            let mut buffer_offset = self.q;

            let mut c0 = format!("let mut dc = c_buffer0[{}];\n", i);
            c0 = c0.add(&format!("let mut ds = s_buffer0[{}];\n", self.q - i));

            c0 = c0.add(&format!(
                "let qc = fmla(ds, self.inner_layer[{}].re, dc) * self.fusion_layer[{}].re;\n",
                (i - 1) * inner_shift,
                twiddles_start + i * inner_shift
            ));
            c0 = c0.add(&format!(
                "let qs = fmla(dc, self.inner_layer[{}].im, ds) * self.fusion_layer[{}].im;\n",
                (i - 1) * inner_shift,
                twiddles_start + i * inner_shift
            ));

            c0 = c0.add("dc = qc;\n");
            c0 = c0.add("ds = qs;\n");

            let mut m = 1usize;

            while m < inner_groups {
                let buffer_idx = buffer_offset / self.q;

                c0 = c0.add(&format!("let c0 = c_buffer{}[{}];\n", buffer_idx, i));
                c0 = c0.add(&format!(
                    "let s0 = s_buffer{}[{}];\n",
                    buffer_idx,
                    self.q - i
                ));

                c0 = c0.add(&format!("dc = fmla(fmla(s0, self.inner_layer[{}].re, c0), self.fusion_layer[{}].re, dc);\n", (i - 1) * inner_shift +  m, twiddles_start + i * inner_shift + m));
                c0 = c0.add(&format!("ds = fmla(fmla(c0, self.inner_layer[{}].im, s0), self.fusion_layer[{}].im, ds);\n", (i - 1) * inner_shift + m, twiddles_start + i * inner_shift + m));

                m += 1;
                buffer_offset += self.q;
            }

            println!("{}", c0);

            println!("data[{}] = dc + a_buffer[{i}];\n", i);

            // unsafe {
            //     *chunk.get_unchecked_mut(i) = dc + *a_buffer.get_unchecked(i);
            //     *c_buffer2.get_unchecked_mut(i) = dc;
            //     *s_buffer2.get_unchecked_mut(self.q - i) = ds;
            // }

            i += 1;
        }
    }

    #[inline]
    fn fuse_layers_next<const AND_MODULATE: bool>(
        &self,
        inner_groups: usize,
        inner_shift: usize,
        twiddles_start: usize,
        j: usize,
        finish: bool,
    ) {
        let mut i = 1usize;
        while i < self.q {
            let mut buffer_offset = self.q;

            let mut c0 = format!("let mut dc = c_buffer0[{}];\n", i);
            c0 = c0.add(&format!("let mut ds = s_buffer0[{}];\n", self.q - i));

            c0 = c0.add(&format!(
                "let qc = fmla(ds, self.inner_layer[{}].re, dc) * self.fusion_layer[{}].re;\n",
                (i - 1) * inner_shift,
                twiddles_start + i * inner_shift
            ));
            c0 = c0.add(&format!(
                "let qs = fmla(dc, self.inner_layer[{}].im, ds) * self.fusion_layer[{}].im;\n",
                (i - 1) * inner_shift,
                twiddles_start + i * inner_shift
            ));

            c0 = c0.add("dc = qc;\n");
            c0 = c0.add("ds = qs;\n");

            let mut m = 1usize;

            while m < inner_groups {
                let buffer_idx = buffer_offset / self.q;

                c0 = c0.add(&format!("let c0 = c_buffer{}[{}];\n", buffer_idx, i));
                c0 = c0.add(&format!(
                    "let s0 = s_buffer{}[{}];\n",
                    buffer_idx,
                    self.q - i
                ));

                c0 = c0.add(&format!("dc = fmla(fmla(s0, self.inner_layer[{}].re, c0), self.fusion_layer[{}].re, dc);\n", (i - 1) * inner_shift +  m, twiddles_start + i * inner_shift + m));
                c0 = c0.add(&format!("ds = fmla(fmla(c0, self.inner_layer[{}].im, s0), self.fusion_layer[{}].im, ds);\n", (i - 1) * inner_shift + m, twiddles_start + i * inner_shift + m));

                m += 1;
                buffer_offset += self.q;
            }

            unsafe {
                if finish {
                    c0 = c0.add(&format!("dc = -fmla(-0.5f64.as_(), dc, a_buffer[{i}]);\n"));
                    c0 = c0.add(&format!("ds = fmla(2f64.as_(), ds, -data[{i}]);\n"));
                    c0 = c0.add("dc = fmla(2f64.as_(), dc, -ds);\n");

                    c0 = c0.add(&format!(
                        "data[{}] = ds;\n",
                        j * self.q * 2 + self.q * 2 - i
                    ));
                    c0 = c0.add(&format!(
                        "data[{}] = dc;\n",
                        j * self.q * 2 + self.q + self.q + i
                    ));
                    println!("{}", c0);
                } /*else {
                 *c_buffer2.get_unchecked_mut(self.q * j + i) = dc;
                 *s_buffer2.get_unchecked_mut(self.q * j + self.q - i) = ds;
                }*/
            }

            i += 1;
        }
    }
}
