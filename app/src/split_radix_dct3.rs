/*
 * // Copyright (c) Radzivon Bartoshyk 2/2026. All rights reserved.
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
use rustdct::num_traits::One;

pub(crate) fn split_radix_dct3(len: usize) -> String {
    let mut v = String::new();

    let quarter_len = len / 4;

    let mut recursive_input_evens = String::new();
    let mut recursive_input_n1 = String::new();
    let mut recursive_input_n3 = String::new();
    let mut recursive_input_n3_rev = vec![String::new(); quarter_len - 1];
    recursive_input_evens = recursive_input_evens + "data[0],";
    recursive_input_evens = recursive_input_evens + "data[2],";
    recursive_input_n1 = recursive_input_n1 + "data[1] * T::TWO,";
    recursive_input_n3 = recursive_input_n3 + &format!("data[{}] * T::TWO,", len - 1);

    // populate the recursive input arrays
    for i in 1..quarter_len {
        let k = 4 * i;

        // the evens are the easy ones - just copy straight over
        recursive_input_evens = recursive_input_evens + &format!("data[{k}],");
        recursive_input_evens = recursive_input_evens + &format!("data[{}],", k + 2);

        recursive_input_n1 = recursive_input_n1 + &format!("data[{}] + data[{}],", k - 1, k + 1);
        recursive_input_n3_rev[i - 1] = format!("data[{}] - data[{}],", k - 1, k + 1).to_string();
    }

    recursive_input_n3_rev.reverse();
    for k in recursive_input_n3_rev.iter() {
        recursive_input_n3 = recursive_input_n3 + &k;
    }

    v = v + &format!("let mut evens = [{recursive_input_evens}];\n");
    v = v + &format!("let mut recursive_input_n1 = [{recursive_input_n1}];\n");
    v = v + &format!("let mut recursive_input_n3 = [{recursive_input_n3}];\n");

    let mut phase_sign = f32::one();

    v = v + "\n";

    let half_len = len / 2;

    for i in 0..quarter_len {
        v = v + &format!("let tw{i} = self.twiddles[{i}];\n");
        v = v + &format!("let cosine_value{i} = recursive_input_n1[{i}];\n");
        v = v + &format!(
            "let sine_value{i} = {}recursive_input_n3[{i}];\n",
            if phase_sign.is_sign_negative() {
                "-"
            } else {
                ""
            }
        );
        v = v + &format!(
            "let lower_dct4{i} = fmla(cosine_value{i}, tw{i}.re, sine_value{i} * tw{i}.im);\n"
        );
        v = v + &format!(
            "let upper_dct4{i} = fmla(cosine_value{i}, tw{i}.im, -sine_value{i} * tw{i}.re);\n"
        );
        v = v + &format!("let lower_dct3{i} = evens[{i}];\n");
        v = v + &format!("let upper_dct3{i} = evens[{}];\n", half_len - i - 1);
        v = v + &format!("data[{i}] = lower_dct3{i} + lower_dct4{i};\n");
        v = v + &format!("data[{}] = lower_dct3{i} - lower_dct4{i};\n", len - i - 1);
        v = v + &format!(
            "data[{}] = upper_dct3{i} + upper_dct4{i};\n",
            half_len - i - 1
        );
        v = v + &format!("data[{}] = upper_dct3{i} - upper_dct4{i};\n", half_len + i);
        phase_sign = -phase_sign;
    }

    v
}
