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
use rustdct::num_traits::One;
use std::ops::Add;

pub(crate) fn generate_butterfly_dct4(n: usize) -> String {
    let mut builder = String::new();
    builder = builder.add(&format!(
        r#"#[derive(Debug, Clone)]
pub(crate) struct Dct4Butterfly{n}<T: DctSample> {{
    twiddles: [Complex<T>; {}],
    bf{}: Dct2Butterfly{}<T>,
}}
impl<T: DctSample> Default for Dct4Butterfly{n}<T>
where
    f64: AsPrimitive<T>,
{{
    fn default() -> Self {{
        Self {{
            twiddles: std::array::from_fn(|x| compute_twiddle::<T>(2 * x + 1, {n} * 8).conj()),
            bf{}: Dct2Butterfly{}::default(),
        }}
    }}
}}
"#,
        n / 2,
        n / 2,
        n / 2,
        n / 2,
        n / 2
    ));

    builder = builder.add(&format!(
        r#"
impl<T: DctSample> Dct4Butterfly{n}<T>
where
    f64: AsPrimitive<T>,
{{
    #[inline(always)]
    pub(crate) fn exec(&self, data: &mut [T; {n}]) {{
        // This is auto-generated factorization of DCT-IV butterfly for {n} points
        let mut left = [T; {}];
        let mut right = [T; {}];
        
        "#,
        n / 2,
        n / 2
    ));

    for i in 0..n / 2 {
        builder = builder.add(&format!(
            r#"
        left[{i}] = fmla(
            self.twiddles[{i}].re,
            data[{i}],
            self.twiddles[{i}].im * data[{}],
        );"#,
            n - i - 1
        ));
        if !i.is_multiple_of(2) {
            builder = builder.add(&format!(
                r#"
          right[{}] = fmla(
                self.twiddles[{i}].re,
                data[{}],
                -self.twiddles[{i}].im * data[{i}],
            );"#,
                n / 2 - i - 1,
                n - i - 1
            ));
        } else {
            builder = builder.add(&format!(
                r#"
          right[{}] = fmla(
                -self.twiddles[{i}].re,
                data[{}],
                self.twiddles[{i}].im * data[{i}],
            );"#,
                n / 2 - i - 1,
                n - i - 1
            ));
        }
    }

    builder = builder.add(&format!("\nself.bf{}.exec(left);\n", n / 2));
    builder = builder.add(&format!(" self.bf{}.exec(right);\n", n / 2));
    builder = builder.add("data[0] = left[0];\n");
    builder = builder.add(&format!("data[{}] = right[0];\n", n - 1));

    let mut sign_left = if (n / 2).is_multiple_of(2) {
        -f32::one()
    } else {
        f32::one()
    };
    let mut sign_right = if (n / 2).is_multiple_of(2) {
        f32::one()
    } else {
        -f32::one()
    };

    if n.is_power_of_two() {
        for i in 1..n / 4 {
            builder = builder.add(&format!(
                "data[{}] = left[{i}] {} right[{}];\n",
                (i - 1) * 2 + 1,
                if sign_left.is_sign_negative() {
                    "-"
                } else {
                    "+"
                },
                n / 2 - i
            ));
            builder = builder.add(&format!(
                "data[{}] = left[{i}] {} right[{}];\n",
                (i - 1) * 2 + 2,
                if sign_right.is_sign_negative() {
                    "-"
                } else {
                    "+"
                },
                n / 2 - i
            ));

            builder = builder.add(&format!(
                "data[{}] = left[{}] {} right[{i}];\n",
                n - (i - 1) * 2 - 3,
                n / 2 - i,
                if sign_left.is_sign_negative() {
                    "-"
                } else {
                    "+"
                },
            ));

            builder = builder.add(&format!(
                "data[{}] = left[{}] {} right[{i}];\n",
                n - (i - 1) * 2 - 2,
                n / 2 - i,
                if sign_right.is_sign_negative() {
                    "-"
                } else {
                    "+"
                },
            ));

            sign_left = -sign_left;
            sign_right = -sign_right;
        }

        builder = builder.add(&format!(
            "data[{}] = left[{}] {} right[{}];\n",
            n / 2 - 1,
            n / 4,
            if sign_left.is_sign_negative() {
                "-"
            } else {
                "+"
            },
            n / 4
        ));

        builder = builder.add(&format!(
            "data[{}] = left[{}] {} right[{}];\n",
            n / 2,
            n / 4,
            if sign_right.is_sign_negative() {
                "-"
            } else {
                "+"
            },
            n / 4
        ));
    } else {
        for i in 1..n / 2 {
            builder = builder.add(&format!(
                "data[{}] = left[{i}] {} right[{}];\n",
                (i - 1) * 2 + 1,
                if sign_left.is_sign_negative() {
                    "-"
                } else {
                    "+"
                },
                n / 2 - i
            ));
            builder = builder.add(&format!(
                "data[{}] = left[{i}] {} right[{}];\n",
                (i - 1) * 2 + 2,
                if sign_right.is_sign_negative() {
                    "-"
                } else {
                    "+"
                },
                n / 2 - i
            ));

            sign_left = -sign_left;
            sign_right = -sign_right;
        }
    }

    builder = builder.add(&format!(
        r#"
        }}
}}

impl<T: DctSample> PxdctExecutor<T> for Dct4Butterfly{n}<T>
where
    f64: AsPrimitive<T>,
{{
    fn execute(&self, data: &mut [T]) -> Result<(), PxdctError> {{
        if !data.len().is_multiple_of({n}) {{
            return Err(PxdctError::InvalidSizeMultiplier(data.len(), self.length()));
        }}
        for chunk in data.chunks_exact_mut({n}) {{
            self.exec((&mut chunk[..{n}]).try_into().unwrap());
        }}
        Ok(())
    }}

    fn length(&self) -> usize {{
        {n}
    }}
}}
    "#
    ));

    builder
}
