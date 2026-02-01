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
use std::ops::{Add, BitXor};

pub(crate) fn transpose<T: Copy>(src: &[T], dst: &mut [T], width: usize, height: usize) {
    for x in 0..width {
        for y in 0..height {
            let input_index = x + y * width;
            let output_index = y + x * height;

            unsafe {
                *dst.get_unchecked_mut(output_index) = *src.get_unchecked(input_index);
            }
        }
    }
}

pub(crate) trait Mulsigni {
    fn mulsigni(self, other: isize) -> Self;
}

impl Mulsigni for f32 {
    #[inline(always)]
    fn mulsigni(self, other: isize) -> Self {
        let s_prec_size = other >> 31;
        f32::from_bits(self.to_bits().bitxor((s_prec_size & (1isize << 31)) as u32))
    }
}

#[allow(unused)]
pub(crate) fn naive_dct2_f32(input: &[f32]) -> Vec<f32> {
    let mut result = Vec::new();

    for output_index in 0..input.len() {
        let mut entry = 0.0;
        for input_index in 0..input.len() {
            let cos_inner =
                (output_index as f32) * (input_index as f32 + 0.5) * std::f32::consts::PI
                    / (input.len() as f32);
            let twiddle = cos_inner.cos();
            entry += input[input_index] * twiddle;
        }
        result.push(entry);
    }

    result
}

pub(crate) fn pfa_input_indices(n1: usize, n2: usize) -> Vec<isize> {
    let mut indices = vec![0isize; n1 * n2];
    let mut index = 0usize;
    println!("---");
    for _ in 0..(n1 * n2) {
        let mut k1 = index % (2 * n1);
        k1 = if k1 < n1 { k1 } else { 2 * n1 - k1 - 1 };

        let mut k2 = index % (2 * n2);
        k2 = if k2 < n2 { k2 } else { 2 * n2 - k2 - 1 };

        let new_idx = k1 * n2 + k2;

        indices[new_idx] = index as isize;

        index += 1;
    }
    for chunk in indices.chunks_exact(n2) {
        println!("{:?}", chunk);
    }
    indices
}

pub(crate) fn pfa_unity_gain(n1: usize, n2: usize) -> Vec<isize> {
    let length = n1 * n2;
    let mut indices = Vec::new();
    for r in 0..n1 {
        for c in 0..n2 {
            let mut idx = c * n1 + r * n2;
            if idx < length {
                indices.push(idx as isize);
            } else {
                let idx = 2 * length - idx;
                indices.push(-(idx as isize));
            }
        }
    }

    for chunk in indices.chunks_exact(n2) {
        println!("{:?}", chunk);
    }

    indices
}

pub(crate) fn pfa_modulation(n1: usize, n2: usize) -> Vec<isize> {
    let mut indices = Vec::new();
    for r in 0..n1 {
        for c in 0..n2 {
            indices.push(
                (c as isize * n1 as isize - r as isize * n2 as isize).unsigned_abs() as isize,
            );
        }
    }

    println!("modulation: ");

    for chunk in indices.chunks_exact(n2) {
        println!("{:?}", chunk);
    }

    indices
}

fn pfa_output_indices(n1: usize, n2: usize) -> (Vec<isize>, Vec<isize>, Vec<isize>) {
    println!("unity gain ---");
    let gains = pfa_unity_gain(n1, n2);
    println!("modulation ---");
    let modulation = pfa_modulation(n1, n2);
    println!("---");
    let mut indices = Vec::new();
    let modulation_cutoff = (n1 - 1) * (n2 - 1) / 2;
    let mut cumulative_modulation = 0usize;

    for r in 0..n1 {
        for c in 0..n2 {
            if r == 0 || c == 0 {
                indices.push(gains[r * n2 + c])
            } else {
                if cumulative_modulation < modulation_cutoff {
                    let k = modulation[r * n2 + c];
                    indices.push(k)
                } else {
                    indices.push(gains[r * n2 + c].abs());
                }
                cumulative_modulation += 1;
            }
        }
    }

    for (row, chunk) in indices.chunks_exact(n2).enumerate() {
        println!(
            "{:?}",
            chunk
                .iter()
                .enumerate()
                .map(|(i, x)| format!("{x}: rows{row}[{i}]"))
                .collect::<Vec<_>>()
        );
    }
    (gains, modulation, indices)
}

fn pfa_encode_signs(indices: &[isize], v_gains: &mut [isize], v_modulation: &mut [isize]) {
    for (index, &address) in indices.iter().enumerate() {
        let gain = v_gains[index];
        let modulation = v_modulation[index];

        if gain != modulation {
            // X = Gain + Modulation
            // hence if address = modulation -> Modulation = X - Gain
            // else if Gain = X - Modulation
            if index == modulation as usize {
                v_gains[index] = gain.abs();
            } else {
                if gain < 0 {
                    v_gains[index] = -gain.abs();
                } else {
                    v_modulation[index] = -modulation;
                }
            }
        }
    }
}

fn print_diff(row: usize, chunk0: &[f32], chunk1: &[f32]) {
    const RED: &str = "\x1b[31m";
    const RESET: &str = "\x1b[0m";
    let mut str = String::new();
    str = str.add("[");
    for i in 0..chunk0.len() {
        let va = chunk0[i];
        let vb = chunk1[i];

        str = str.add(" ");

        if (va - vb).abs() > 1e-3 {
            str = str.add(&format!(
                "({}) {RED}({:.6} → {:.6}){RESET} ",
                i + row,
                va,
                vb
            ));
        } else {
            str = str.add(&format!("({}) {:.6}", i + row, va));
        }

        if i + 1 < chunk0.len() {
            str = str.add(",");
        }
    }
    str = str.add("]");
    println!("{}", str.to_string());
}

pub(crate) fn gen_coprimes(w: usize, h: usize) {
    let mut qq = vec![0.; w * h];
    for (i, dst) in qq.iter_mut().enumerate() {
        *dst = i as f32 + rand::random::<f32>();
    }
    let mut qq_fixed = qq.clone();

    println!("---");

    let (mut gains, mut modulation, indices) = pfa_output_indices(h, w);
    let original_gains = gains.clone();
    for gain in gains.iter_mut() {
        let q = indices
            .iter()
            .position(|&x| x == gain.abs())
            .expect("Algorithm doesn't converge") as isize;
        *gain = if gain.is_negative() { -q } else { q };
    }
    println!("gains ---");
    for chunk in gains.chunks_exact(w) {
        println!("{:?}", chunk);
    }
    for modulation in modulation.iter_mut() {
        let q = indices
            .iter()
            .position(|&x| x == modulation.abs())
            .expect("Algorithm doesn't converge") as isize;
        *modulation = if modulation.is_negative() { -q } else { q };
    }
    println!("modulation ---");
    for chunk in modulation.chunks_exact(w) {
        println!("{:?}", chunk);
    }
    let input_indices = pfa_input_indices(w, h);

    let mut scratch = qq.clone();
    for (dst, &index) in scratch.iter_mut().zip(input_indices.iter()) {
        *dst = qq[index as usize];
    }

    for (i, cols) in input_indices.chunks(h).enumerate() {
        let mut str = String::new();

        str = str.add(&format!("let mut col{i} = ["));
        for (i, col) in cols.iter().enumerate() {
            if i + 1 < cols.len() {
                str = str.add(&format!("data[{col}],",));
            } else {
                str = str.add(&format!("data[{col}]",));
            }
        }
        str = str.add("];");
        println!("{}", str.to_string());
    }

    for (i, cols) in input_indices.chunks(h).enumerate() {
        println!("{}", format!("self.bf{h}.exec(&mut col{});", i));
    }

    for (c, cols) in input_indices.chunks(w).enumerate() {
        let mut str = String::new();

        str = str.add(&format!("let mut row{c} = ["));
        for (i, col) in cols.iter().enumerate() {
            if i + 1 < cols.len() {
                str = str.add(&format!("col{i}[{c}],",));
            } else {
                str = str.add(&format!("col{i}[{c}]",));
            }
        }
        str = str.add("];");
        println!("{}", str.to_string());
    }

    for (i, cols) in input_indices.chunks(w).enumerate() {
        println!("{}", format!("self.bf{w}.exec(&mut row{});", i));
    }

    let mut scratch2 = qq.clone();

    for row in scratch.chunks_exact_mut(h) {
        let q = naive_dct2_f32(row);
        row.copy_from_slice(&q);
    }

    transpose(&scratch, &mut scratch2, h, w);

    for row in scratch2.chunks_exact_mut(w) {
        let q = naive_dct2_f32(row);
        row.copy_from_slice(&q);
    }

    let sq = qq.clone();

    pfa_encode_signs(&indices, &mut gains, &mut modulation);

    println!("coded gains reference --");
    for chunk in gains.chunks_exact(w) {
        println!("{:?}", chunk);
    }
    println!("--");

    println!("coded modulations reference --");
    for chunk in modulation.chunks_exact(w) {
        println!("{:?}", chunk);
    }
    println!("--");

    let mut modulation_ref = String::new();

    for (index, &address) in indices.iter().enumerate() {
        let gain = gains[index];
        let modulation = modulation[index];

        let a_gain = gain.unsigned_abs();

        let r_gain = scratch2[a_gain];
        let r_modulation = scratch2[modulation.unsigned_abs()];

        if gain == modulation {
            qq[address as usize] = r_gain;
            let row = a_gain / w;
            let col = a_gain % w;
            modulation_ref = modulation_ref.add(&format!("data[{address}] = row{row}[{col}];\n"));
        } else {
            // X = Gain + Modulation
            // hence if address = modulation -> Modulation = X - Gain
            // else if Gain = X - Modulation
            qq[address as usize] = r_modulation.mulsigni(modulation) + r_gain.mulsigni(gain);

            let g_row = a_gain / w;
            let g_col = a_gain % w;

            let m_row = modulation.unsigned_abs() / w;
            let m_col = modulation.unsigned_abs() % w;

            if gain < 0 {
                modulation_ref = modulation_ref.add(&format!(
                    "data[{address}] = {}row{m_row}[{m_col}] {}row{g_row}[{g_col}];\n",
                    if modulation < 0 { "-" } else { "" },
                    if gain < 0 { "-" } else { "+" },
                ));
            } else {
                modulation_ref = modulation_ref.add(&format!(
                    "data[{address}] = {}row{g_row}[{g_col}] {}row{m_row}[{m_col}];\n",
                    if gain < 0 { "-" } else { "" },
                    if modulation < 0 { "-" } else { "+" },
                ));
            }
        }
    }

    println!("{}", modulation_ref);
}
