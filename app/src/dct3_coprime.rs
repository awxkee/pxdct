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
use crate::coprime_generator::{
    Mulsigni, naive_dct2_f32, pfa_modulation, pfa_unity_gain, transpose,
};
use pxdct::Pxdct;
use rand::Rng;
use std::fmt::format;
use std::ops::Add;

pub(crate) fn pfa_output_indices(n1: usize, n2: usize) -> Vec<isize> {
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

pub fn naive_dct3_f32(input: &[f32]) -> Vec<f32> {
    let mut result = Vec::new();

    for output_index in 0..input.len() {
        let mut entry = 0.0;
        for input_index in 0..input.len() {
            let multiplier = if input_index == 0 { 0.5 } else { 1.0 };
            let cos_inner =
                (output_index as f32 + 0.5) * (input_index as f32) * std::f32::consts::PI
                    / (input.len() as f32);
            let twiddle = cos_inner.cos();
            entry += input[input_index] * twiddle * multiplier;
        }
        result.push(entry);
    }

    result
}

pub fn naive_dct3_f32_orth(input: &[f32]) -> Vec<f32> {
    let mut result = Vec::new();

    for output_index in 0..input.len() {
        let mut entry = 0.0;
        for input_index in 0..input.len() {
            let multiplier = if input_index == 0 { 1.0 } else { 1.0 };
            let cos_inner =
                (output_index as f32 + 0.5) * (input_index as f32) * std::f32::consts::PI
                    / (input.len() as f32);
            let twiddle = cos_inner.cos();
            entry += input[input_index] * twiddle * multiplier;
        }
        result.push(entry);
    }

    result
}

fn pfa_input_indices(n2: usize, n1: usize) -> (Vec<isize>, Vec<isize>, Vec<isize>) {
    println!("unity gain ---");
    let gains = pfa_unity_gain(n1, n2);
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

    println!("indicies");
    println!("---");
    for (row, chunk) in indices.chunks_exact(n2).enumerate() {
        println!(
            "{:?}",
            chunk.iter().map(|&x| format!("{x}")).collect::<Vec<_>>()
        );
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

pub(crate) fn gen_dct3_coprimes(w: usize, h: usize) -> String {
    let mut pfa_builder = String::new();
    pfa_builder = pfa_builder
        + &format!("// This is auto-generated DCT-III Prime-Factor algorithm for size {w}x{h}\n");
    let mut qq = vec![0.; w * h];
    for (i, dst) in qq.iter_mut().enumerate() {
        *dst = i as f32 + 0.5;
    }
    let qq_fixed = qq.clone();
    let (mut gains, mut modulation, indices) = pfa_input_indices(w, h);

    let mut qq_original = qq_fixed.clone();

    let mut cols_ref = Vec::new();
    for _ in 0..h {
        cols_ref.push("".to_string());
    }

    for (index, &address) in indices.iter().enumerate() {
        assert!(address >= 0);
        let address = address.unsigned_abs();
        let gain = gains[index];
        let modulation = modulation[index];

        let a_gain = gain.unsigned_abs();

        let r_gain = qq_fixed[a_gain];
        let r_modulation = qq_fixed[modulation.unsigned_abs()];

        if gain == modulation {
            qq[index as usize] = r_gain;

            let m_row = index / w;
            let m_col = index % w;

            if m_col != 0 || m_row == 0 {
                cols_ref[m_row] = cols_ref[m_row].to_string() + &format!("data[{gain}],");
            } else {
                cols_ref[m_row] = cols_ref[m_row].to_string() + &format!("data[{gain}] * T::TWO,");
            }
        } else {
            qq[index] = r_modulation.mulsigni(modulation) + r_gain.mulsigni(gain);

            let m_row = index / w;
            let m_col = index % w;

            cols_ref[m_row] = (cols_ref[m_row].to_string()
                + &format!(
                    "{}data[{}] {}data[{}]",
                    if modulation < 0 { "-" } else { "" },
                    modulation.unsigned_abs(),
                    if gain < 0 { "-" } else { "+" },
                    a_gain
                ))
                .to_string()
                + if m_col == 0 { "T::TWO," } else { "," };
        }
    }

    for (i, col) in cols_ref.iter().enumerate() {
        pfa_builder = pfa_builder.add(&format!("let mut col{i} = [{}];\n", col));
    }

    pfa_builder = pfa_builder + "\n";

    for i in 0..h {
        pfa_builder = pfa_builder.add(&format!("self.bf{w}.exec(&mut col{});\n", i));
    }

    println!("new");
    for row in qq_fixed.chunks_exact(w) {
        println!("{:?}", row);
    }
    println!("f");

    for row in qq.chunks_exact_mut(w) {
        println!("{:?}", row);
    }

    for (i, row) in qq.chunks_exact_mut(w).enumerate() {
        if i > 0 {
            row[0] *= 2.0;
        }
        let mut q = naive_dct3_f32(row);
        row.copy_from_slice(&q);
    }

    pfa_builder = pfa_builder + "\n";

    for (c, cols) in qq_fixed.chunks(h).enumerate() {
        let mut str = String::new();

        str = str.add(&format!("let mut row{c} = ["));
        for (i, _) in cols.iter().enumerate() {
            str = str.add(&format!("col{i}[{c}]",));
            if i == 0 {
                str = str.add("*T::TWO");
            }
            if i + 1 < cols.len() {
                str = str + ",";
            }
        }
        str = str.add("];");
        pfa_builder = pfa_builder.add(&(str + "\n"));
    }

    pfa_builder = pfa_builder + "\n";

    for i in 0..w {
        pfa_builder = pfa_builder.add(&format!("self.bf{h}.exec(&mut row{});\n", i));
    }

    println!("w");
    for row in qq.chunks_exact_mut(w) {
        println!("{:?}", row);
    }

    let mut scratch2 = qq.clone();

    transpose(&qq, &mut scratch2, w, h);

    println!("t");
    for row in scratch2.chunks_exact_mut(h) {
        println!("{:?}", row);
    }
    println!("--");

    for row in scratch2.chunks_exact_mut(h) {
        row[0] *= 2.0;
        let q = naive_dct3_f32(row);
        row.copy_from_slice(&q);
    }

    let qq_fixed = naive_dct3_f32(&qq_fixed);

    let output_indices = pfa_output_indices(w, h);
    let mut output = qq_fixed.clone();

    for (i, index) in output_indices.iter().enumerate() {
        output[index.unsigned_abs()] = scratch2[i];
    }

    pfa_builder = pfa_builder + "\n";

    for (r, output) in output_indices.chunks(h).enumerate() {
        let mut str = String::new();

        for (c, output) in output.iter().enumerate() {
            str = str + &format!("data[{output}] = row{r}[{c}];\n");
        }
        pfa_builder = pfa_builder.add(&str);
    }

    println!("original {:?}", qq_fixed);
    println!("{:?}", output);
    let pxdct = Pxdct::make_dct3_f32(output.len());
    pxdct.unwrap().execute(&mut qq_original).unwrap();
    println!("pxdct {:?}", qq_original);

    pfa_builder
}
