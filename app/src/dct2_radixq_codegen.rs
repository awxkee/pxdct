use pxfm::{f_cospi, f_cospif, f_sinpi, f_sinpif};
use rustdct::num_traits::{One, Zero};
use std::ops::Add;

pub(crate) fn radixq_phase_cos_twiddle(q: usize, m: usize, j: usize) -> f32 {
    let module = (q - 1 - 2 * m) as f64;
    let angle_a_phase = module * j as f64 / (2 * q) as f64;
    let a_phase = f_cospi(angle_a_phase);
    a_phase as f32
}

pub(crate) fn radixq_phase_cos_twiddled(q: usize, m: usize, j: usize) -> f64 {
    let module = (q - 1 - 2 * m) as f64;
    let angle_a_phase = module * j as f64 / (2 * q) as f64;
    let a_phase = f_cospi(angle_a_phase);
    a_phase
}

pub(crate) fn radixq_even_twiddle(q: usize, m: usize, j: usize, k: f32, fft_len: usize) -> f32 {
    let module = (q - 1 - 2 * m) as f32;
    let angle_a = module * k / (2. * fft_len as f32);
    let angle_a_phase = module * j as f32 / (2 * q) as f32;
    let a = f_cospif(angle_a);
    let a_phase = f_cospif(angle_a_phase);
    a * a_phase
}

pub(crate) fn radixq_odd_phase_twiddlej(q: usize, inv_m: usize, j: usize) -> f32 {
    let inv_module = q as f64 - 1f64 - 2f64 * inv_m as f64;
    let theta = inv_module * j as f64;
    let angle_b_phase = theta / q as f64;
    let lo = f_sinpi(angle_b_phase);
    lo as f32
}

pub(crate) fn radixq_odd_phase_twiddled(q: usize, inv_m: usize, j: usize) -> f64 {
    let inv_module = q as f64 - 1f64 - 2f64 * inv_m as f64;
    let theta = inv_module * (2 * j + 1) as f64;
    let angle_b_phase = theta / (2 * q) as f64;
    let lo = f_sinpi(angle_b_phase);
    lo
}

pub(crate) fn naive_dct2_f32(input: &mut [f32], l: usize) {
    for chunk in input.chunks_exact_mut(l) {
        let mut result = Vec::new();

        for output_index in 0..l {
            let mut entry = 0.0;
            for input_index in 0..l {
                let cos_inner =
                    (output_index as f32) * (input_index as f32 + 0.5) * std::f32::consts::PI
                        / (l as f32);
                let twiddle = cos_inner.cos();
                entry += chunk[input_index] * twiddle;
            }
            result.push(entry);
        }

        chunk.copy_from_slice(&result);
    }
}

pub(crate) fn radixq_odd_twiddle(q: usize, inv_m: usize, j: usize, k: f32, dct_len: usize) -> f32 {
    let inv_module = q as f32 - 1f64 as f32 - 2f64 as f32 * inv_m as f32;
    let angle_b = inv_module * k / (2. * dct_len as f64) as f32;
    let theta = inv_module * (2. * j as f64 + 1.) as f32;
    let angle_b_phase = theta / (2 * q) as f32;
    let b_phase = f_cospif(angle_b);
    let lo = f_sinpif(angle_b_phase);
    let prod = b_phase * lo;
    prod
}

#[derive(Default)]
struct FilterBank {
    even_twiddles: Vec<f64>,
    odd_twiddles: Vec<f64>,
}

impl FilterBank {
    fn even_phase_twiddle(&mut self, q: usize, inv_m: usize, j: usize) -> String {
        let even0 = radixq_phase_cos_twiddled(q, inv_m, j);
        for (i, &twiddle) in self.even_twiddles.iter().enumerate() {
            if (twiddle.abs() - even0.abs()).abs() < 1e-7 {
                if twiddle.signum() != even0.signum() {
                    return format!("-T::R{q}_EVEN_TWIDDLE_{i}").to_string();
                }
                return format!("T::R{q}_EVEN_TWIDDLE_{i}").to_string();
            }
        }
        self.even_twiddles.push(even0);
        format!("T::R{q}_EVEN_TWIDDLE_{}", self.even_twiddles.len() - 1).to_string()
    }

    fn odd_phase_twiddle(&mut self, q: usize, inv_m: usize, j: usize) -> String {
        let even0 = radixq_odd_phase_twiddled(q, inv_m, j);
        for (i, &twiddle) in self.odd_twiddles.iter().enumerate() {
            if (twiddle.abs() - even0.abs()).abs() < 1e-7 {
                if twiddle.signum() != even0.signum() {
                    return format!("-T::R{q}_ODD_TWIDDLE_{i}").to_string();
                }
                return format!("T::R{q}_ODD_TWIDDLE_{i}").to_string();
            }
        }
        self.odd_twiddles.push(even0);
        format!("T::R{q}_ODD_TWIDDLE_{}", self.odd_twiddles.len() - 1).to_string()
    }

    fn build(&self, main_q: usize) -> String {
        let mut filter_bank = String::new();
        let mut filter_bank_f32 = String::new();
        let mut filter_bank_f64 = String::new();
        filter_bank = filter_bank.add(&format!("pub(crate) trait MixedRadix{main_q}Sample {{\n"));
        filter_bank_f32 =
            filter_bank_f32.add(&format!("impl MixedRadix{main_q}Sample for f32 {{\n"));
        filter_bank_f64 =
            filter_bank_f64.add(&format!("impl MixedRadix{main_q}Sample for f64 {{\n"));

        for (i, &even) in self.even_twiddles.iter().enumerate() {
            filter_bank = filter_bank.add(&format!("const R{main_q}_EVEN_TWIDDLE_{i}: Self;\n"));
            filter_bank_f32 = filter_bank_f32.add(&format!(
                "const R{main_q}_EVEN_TWIDDLE_{i}: f32 = f32::from_bits(0x{:08x});\n",
                (even as f32).to_bits()
            ));
            filter_bank_f64 = filter_bank_f64.add(&format!(
                "const R{main_q}_EVEN_TWIDDLE_{i}: f64 = f64::from_bits(0x{:016x});\n",
                even.to_bits()
            ));
        }

        for (i, &even) in self.odd_twiddles.iter().enumerate() {
            filter_bank = filter_bank.add(&format!("const R{main_q}_ODD_TWIDDLE_{i}: Self;\n"));
            filter_bank_f32 = filter_bank_f32.add(&format!(
                "const R{main_q}_ODD_TWIDDLE_{i}: f32 = f32::from_bits(0x{:08x});\n",
                (even as f32).to_bits()
            ));
            filter_bank_f64 = filter_bank_f64.add(&format!(
                "const R{main_q}_ODD_TWIDDLE_{i}: f64 = f64::from_bits(0x{:016x});\n",
                even.to_bits()
            ));
        }

        filter_bank = filter_bank.add(&format!("}}\n"));
        filter_bank_f32 = filter_bank_f32.add(&format!("}}\n"));
        filter_bank_f64 = filter_bank_f64.add(&format!("}}\n"));

        filter_bank = filter_bank.add("\n");
        filter_bank = filter_bank.add(filter_bank_f32.as_str());

        filter_bank = filter_bank.add("\n");
        filter_bank = filter_bank.add(filter_bank_f64.as_str());

        filter_bank
    }
}

fn is_minus_one(x: f32) -> bool {
    (x + 1f32).abs() < 1e-5
}

fn is_one_or_minus_one(x: f32) -> bool {
    (x.abs() - 1f32).abs() < 1e-5
}

fn is_half_or_minus_half(x: f32) -> bool {
    (x.abs() - 0.5).abs() < 1e-5
}

pub(crate) fn generate_radixq(data: &mut [f32], main_q: usize) -> String {
    assert_eq!(data.len(), main_q);
    let mut main = String::new();
    let mut filter_bank = FilterBank::default();

    let q_modules = data.len() / main_q;
    let len = data.len();

    let inner_groups = (main_q.saturating_sub(3)) / 2 + 1;

    let j_modules = ((main_q - 1) / 2).max(1);

    let mut scratch = vec![0f32; data.len()];
    let inner_blocks = data.len() / q_modules;

    let (a_buffer, c_s_buffer) = scratch.split_at_mut(q_modules);
    let (c_buffer, s_buffer) = c_s_buffer.split_at_mut(q_modules * (inner_blocks / 2));

    main = main.add(format!("// This is autogenerated Radix-Q where Q = {main_q}.\n").as_str());

    let mut a_builder = "let a_buffer = [".to_string();

    for (n, dst) in a_buffer.iter_mut().enumerate() {
        *dst = data[n * main_q + j_modules];
        a_builder = a_builder.add(&format!(
            "{}data[{}]",
            if n > 0 { "," } else { "" },
            n * main_q + j_modules
        ));
    }

    a_builder = a_builder.add("];\n");

    main = main.add(a_builder.as_str());

    let mut c_builder = "let c_buffer = [".to_string();
    let mut s_builder = "let s_buffer = [".to_string();

    for (m, (c_buffer, s_buffer)) in c_buffer
        .chunks_exact_mut(q_modules)
        .zip(s_buffer.chunks_exact_mut(q_modules))
        .enumerate()
    {
        let mut sign = f32::one();
        for (n, (c_dst, s_dst)) in c_buffer.iter_mut().zip(s_buffer.iter_mut()).enumerate() {
            let u0 = data[main_q * n + m];
            let u1 = data[main_q * n + main_q - m - 1];

            *c_dst = u0 + u1;
            *s_dst = (u0 - u1) * sign;

            c_builder = c_builder.add(&format!(
                "{}data[{}] + data[{}]",
                if m > 0 { "," } else { "" },
                main_q * n + m,
                main_q * n + main_q - m - 1
            ));
            let expr = if sign < 0. {
                format!(
                    "{}data[{}] - data[{}]",
                    if m > 0 { "," } else { "" },
                    main_q * n + main_q - m - 1,
                    main_q * n + m
                )
            } else {
                format!(
                    "{}data[{}] - data[{}]",
                    if m > 0 { "," } else { "" },
                    main_q * n + m,
                    main_q * n + main_q - m - 1
                )
            };
            s_builder = s_builder.add(&expr);

            sign = -sign;
        }
    }

    c_builder = c_builder.add("];\n");
    s_builder = s_builder.add("];\n");

    main = main.add(c_builder.as_str());
    main = main.add(s_builder.as_str());

    naive_dct2_f32(&mut scratch, q_modules);

    data.fill(f32::zero());

    let (a_buffer, c_s_buffer) = scratch.split_at_mut(q_modules);
    let (c_buffer, s_buffer) = c_s_buffer.split_at_mut(q_modules * (inner_blocks / 2));

    {
        let qc = c_buffer[0];
        let even0 = radixq_even_twiddle(main_q, 0, 0, 0., len);
        let even1 = radixq_even_twiddle(main_q, 0, 2, 0., len);
        let mut c0 = qc * even0;
        main = main.add("let qc = c_buffer[0];\n");
        main = main.add(&"let mut c0 = qc;\n".to_string());
        let mut c1 = qc * even1;
        if is_minus_one(even1) {
            main = main.add(&"let mut c1 = qc * -1;\n".to_string());
        } else if is_half_or_minus_half(even1) {
            main = main.add(&format!(
                "let mut c1 = qc * {}T::HALF;\n",
                if even1.is_sign_negative() { "-" } else { "" }
            ));
        } else {
            main = main.add(&format!(
                "let mut c1 = qc * {};\n",
                filter_bank.even_phase_twiddle(main_q, 0, 2)
            ));
        }
        let odd_twiddle0 = radixq_odd_twiddle(main_q, 0, 0, 0., len);

        let mut s0 = s_buffer[0];

        main = main.add(&"let mut s0 = s_buffer[0];\n".to_string());

        main = main.add(&format!(
            "s0 *= {};\n",
            filter_bank.odd_phase_twiddle(main_q, 0, 0)
        ));

        s0 *= odd_twiddle0;

        let mut buffer_offset = q_modules;

        let mut m = 1usize;
        while m < inner_groups {
            main = main.add(&format!("let ci{m} = c_buffer[{}];\n", m * q_modules));
            main = main.add(&format!("let si{m} = s_buffer[{}];\n", m * q_modules));

            let ci = c_buffer[buffer_offset];
            let si = s_buffer[buffer_offset];
            let even_twiddle = radixq_even_twiddle(main_q, m, 0, 0., len);
            let even_twiddle1 = radixq_even_twiddle(main_q, m, 2, 0., len);
            let odd_twiddle1 = radixq_odd_twiddle(main_q, m, 0, 0., len);

            if !is_one_or_minus_one(even_twiddle) && !is_half_or_minus_half(even_twiddle) {
                if main_q > 13 {
                    main = main.add(&format!(
                        "c0 = ci{m} * {} + c0;\n",
                        filter_bank.even_phase_twiddle(main_q, m, 0)
                    ));
                } else {
                    main = main.add(&format!(
                        "c0 = fmla(ci{m}, {}, c0);\n",
                        filter_bank.even_phase_twiddle(main_q, m, 0)
                    ));
                }
            } else if is_half_or_minus_half(even_twiddle) {
                main = main.add(&format!(
                    "c0 = fmla(ci{m}, {}T::HALF, c0);\n",
                    if even_twiddle.is_sign_negative() {
                        "-"
                    } else {
                        "+"
                    }
                ));
            } else {
                main = main.add(&format!(
                    "c0 = ci{m} {} c0;\n",
                    if even_twiddle.is_sign_negative() {
                        "-"
                    } else {
                        "+"
                    }
                ));
            }
            if !is_one_or_minus_one(even_twiddle1) && !is_half_or_minus_half(even_twiddle1) {
                if main_q > 13 {
                    main = main.add(&format!(
                        "c1 = ci{m} * {} + c1;\n",
                        filter_bank.even_phase_twiddle(main_q, m, 2)
                    ));
                } else {
                    main = main.add(&format!(
                        "c1 = fmla(ci{m}, {}, c1);\n",
                        filter_bank.even_phase_twiddle(main_q, m, 2)
                    ));
                }
            } else if is_half_or_minus_half(even_twiddle1) {
                main = main.add(&format!(
                    "c1 = fmla(ci{m}, {}T::HALF, c1);\n",
                    if even_twiddle1.is_sign_negative() {
                        "-"
                    } else {
                        "+"
                    }
                ));
            } else {
                main = main.add(&format!(
                    "c1 = ci{m} {} c1;\n",
                    if even_twiddle1.is_sign_negative() {
                        "-"
                    } else {
                        "+"
                    }
                ));
            }
            if odd_twiddle1 != 0. {
                main = main.add(&format!(
                    "s0 = fmla(si{m}, {}, s0);\n",
                    filter_bank.odd_phase_twiddle(main_q, m, 0)
                ));
            }

            c0 = f32::mul_add(ci, even_twiddle, c0);
            c1 = f32::mul_add(ci, even_twiddle1, c1);
            s0 = f32::mul_add(si, odd_twiddle1, s0);
            m += 1;
            buffer_offset += q_modules;
        }

        let a0 = a_buffer[0];
        let dc = c0 + a0;
        data[0] = dc;

        main = main.add("let a0 = a_buffer[0];\n");
        main = main.add("let dc = c0 + a0;\n");
        main = main.add("data[0] = dc;\n");
        main = main.add(&format!("data[{q_modules}] = s0;\n"));
        main = main.add(&format!("data[{}] = -(c1 + a0);\n", q_modules * 2));

        let idx = q_modules;
        data[idx] = s0;

        let idx1 = q_modules * 2;
        let qid2 = -(c1 + a0); // negated 2j
        data[idx1] = qid2;
    }

    let mut start_j = 4usize;
    let mut odd_j = 1usize;

    let mut sign_c = f32::one();
    let mut sign_s = -f32::one();

    for j in 1..inner_groups {
        let qc = c_buffer[0];
        let even0 = radixq_even_twiddle(main_q, 0, start_j, 0., len);
        let mut c0 = qc * even0;
        let odd_twiddle0 = radixq_odd_twiddle(main_q, 0, odd_j, 0., len);

        if is_one_or_minus_one(even0) {
            if even0.is_sign_negative() {
                main = main.add(&"let mut c0 = -qc;\n".to_string());
            } else {
                main = main.add(&"let mut c0 = qc ;\n".to_string());
            }
        } else if is_half_or_minus_half(even0) {
            main = main.add(&format!(
                "let mut c0 = qc * {}T::HALF;\n",
                if even0.is_sign_negative() { "-" } else { "" }
            ));
        } else {
            main = main.add(&format!(
                "let mut c0 = qc * {};\n",
                filter_bank.even_phase_twiddle(main_q, 0, start_j)
            ));
        }
        main = main.add(&"let mut s0 = s_buffer[0];\n".to_string());

        main = main.add(&format!(
            "s0 *= {};\n",
            filter_bank.odd_phase_twiddle(main_q, 0, odd_j)
        ));

        let mut s0 = s_buffer[0];

        s0 *= odd_twiddle0;

        let mut buffer_offset = q_modules;

        let mut m = 1usize;
        while m < inner_groups {
            let ci = c_buffer[buffer_offset];
            let si = s_buffer[buffer_offset];
            let even4 = radixq_even_twiddle(main_q, m, start_j, 0., len);
            let even_twiddle = even4;
            let odd_twiddle1 = radixq_odd_twiddle(main_q, m, odd_j, 0., len);
            c0 = f32::mul_add(ci, even_twiddle, c0);
            s0 = f32::mul_add(si, odd_twiddle1, s0);

            // main = main.add(&format!("let ci{m} = c_buffer[{}];\n", m * q_modules));
            // main = main.add(&format!("let si{m} = s_buffer[{}];\n", m * q_modules));

            if !is_one_or_minus_one(even_twiddle) && !is_half_or_minus_half(even_twiddle) {
                // if main_q > 13 {
                //     main = main.add(&format!(
                //         "c0 = ci{m} * {} + c0;\n",
                //         filter_bank.even_phase_twiddle(main_q, m, start_j)
                //     ));
                // } else {
                main = main.add(&format!(
                    "c0 = fmla(ci{m}, {}, c0);\n",
                    filter_bank.even_phase_twiddle(main_q, m, start_j)
                ));
                // }
            } else if is_half_or_minus_half(even_twiddle) {
                main = main.add(&format!(
                    "c0 = fmla(ci{m}, {}T::HALF, c0);\n",
                    if even_twiddle.is_sign_negative() {
                        "-"
                    } else {
                        ""
                    }
                ));
            } else if is_one_or_minus_one(even_twiddle) {
                main = main.add(&format!(
                    "c0 = c0 {} ci{m};\n",
                    if even_twiddle.is_sign_negative() {
                        "-"
                    } else {
                        "+"
                    }
                ));
            } else {
                unreachable!("Should not happen")
            }
            if odd_twiddle1 != 0. {
                // if main_q > 13 {
                //     main = main.add(&format!(
                //         "s0 = si{m} * {} + s0;\n",
                //         filter_bank.odd_phase_twiddle(main_q, m, odd_j)
                //     ));
                // } else {
                main = main.add(&format!(
                    "s0 = fmla(si{m}, {}, s0);\n",
                    filter_bank.odd_phase_twiddle(main_q, m, odd_j)
                ));
                // }
            }

            m += 1;
            buffer_offset += q_modules;
        }

        let a0 = a_buffer[0];
        let dc = c0 + a0;
        let i0 = (j - 1) * q_modules * 2 + q_modules * 4;
        data[i0] = sign_c * dc;

        main = main.add("let dc = c0 + a0;\n");
        main = main.add(&format!(
            "data[{}] = {}dc;\n",
            (j - 1) * q_modules * 2 + q_modules * 4,
            if sign_c < 0. { "-" } else { "" }
        ));

        let idx = q_modules * (j - 1) * 2 + q_modules * 3;
        data[idx] = sign_s * s0;

        main = main.add(&format!(
            "data[{}] = {}s0;\n",
            q_modules * (j - 1) * 2 + q_modules * 3,
            if sign_s < 0. { "-" } else { "" }
        ));

        // start_j += 6;
        // start_j = start_j % 7;
        start_j += 2;
        odd_j += 1;
        sign_s = -sign_s;
        sign_c = -sign_c;
        // odd_j = odd_j % 7;
        // start_j = start_j % 7;
    }

    let mut finalizer = String::new();
    finalizer = finalizer.add(&filter_bank.build(main_q));
    finalizer = finalizer.add("\n");
    finalizer = finalizer.add(&main);
    finalizer
}
