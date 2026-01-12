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

use regex::Regex;
use std::collections::HashMap;
use std::ops::Add;

fn split_signed_terms(expr: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut chars = expr.chars().peekable();

    while let Some(c) = chars.next() {
        if (c == '+' || c == '-') && !cur.is_empty() {
            out.push(cur.trim().to_string());
            cur.clear();
            cur.push(c);
        } else {
            cur.push(c);
        }
    }

    if !cur.trim().is_empty() {
        out.push(cur.trim().to_string());
    }

    out
}

fn replace_terms_exact_split(terms: &mut [String], common_subs: &HashMap<String, String>) {
    for t in terms.iter_mut() {
        // Extract the "bare" term without leading '+' or '-' and trim spaces
        let z = t.clone();
        let bare = z.trim_start_matches(|c| c == '+' || c == '-').trim();

        for (term, var) in common_subs {
            if bare == term {
                // Preserve the original sign
                let sign = if t.trim_start().starts_with('-') {
                    "-"
                } else {
                    "+"
                };
                *t = format!("{}{}", sign, var);
            }
        }
    }

    // Optional: remove leading '+' from the first term
    if let Some(first) = terms.first_mut() {
        *first = first.trim_start_matches('+').trim().to_string();
    }
}

pub(crate) fn solve_expression(expressions: &Vec<String>) -> String {
    let term_regex = Regex::new(r"(self\.twiddle\d+\.(?:re|im)\s*\*\s*x\d+)").unwrap();
    let mut term_counts: HashMap<String, usize> = HashMap::new();

    // Count term occurrences
    for expr in expressions.iter() {
        for cap in term_regex.captures_iter(expr) {
            let term = cap.get(1).unwrap().as_str().to_string();
            *term_counts.entry(term).or_insert(0) += 1;
        }
    }

    // Assign temporary names to terms that occur more than once
    let mut common_subs: HashMap<String, String> = HashMap::new();
    let mut tmp_index = 0;
    for (term, count) in &term_counts {
        if *count > 1 {
            common_subs.insert(term.clone(), format!("tmp{}", tmp_index));
            tmp_index += 1;
        }
    }

    let mut optimized_expr = String::new();

    for (term, var) in &common_subs {
        optimized_expr = optimized_expr.add(format!("let {} = {};\n", var, term).as_str());
    }

    for expr in expressions.iter() {
        let parts: Vec<&str> = expr.split('=').collect();
        let lhs = parts[0].trim();
        let mut rhs = parts[1].trim().to_string();
        let mut splat = split_signed_terms(&rhs);
        replace_terms_exact_split(&mut splat, &common_subs);
        // Replace common subexpressions
        // for (term, var) in &common_subs {
        // rhs = rhs.replace(term, var);

        // }

        optimized_expr = optimized_expr.add(format!("{} = {}\n", lhs, splat.join("")).as_str());
    }

    optimized_expr
}

pub(crate) fn solve_expression_arr(expressions: &Vec<String>) -> String {
    let term_regex = Regex::new(r"(self\.twiddles\[\d+]\s*\*\s*x\d+)").unwrap();
    let mut term_counts: HashMap<String, usize> = HashMap::new();

    // Count term occurrences
    for expr in expressions.iter() {
        for cap in term_regex.captures_iter(expr) {
            let term = cap.get(1).unwrap().as_str().to_string();
            *term_counts.entry(term).or_insert(0) += 1;
        }
    }

    // Assign temporary names to terms that occur more than once
    let mut common_subs: HashMap<String, String> = HashMap::new();
    let mut tmp_index = 0;
    for (term, count) in &term_counts {
        if *count > 1 {
            common_subs.insert(term.clone(), format!("tmp{}", tmp_index));
            tmp_index += 1;
        }
    }

    let mut optimized_expr = String::new();

    for (term, var) in &common_subs {
        optimized_expr = optimized_expr.add(format!("let {} = {};\n", var, term).as_str());
    }

    for expr in expressions.iter() {
        let parts: Vec<&str> = expr.split('=').collect();
        let lhs = parts[0].trim();
        let rhs = parts[1].trim().to_string();
        let mut splat = split_signed_terms(&rhs);
        replace_terms_exact_split(&mut splat, &common_subs);

        if splat.len() < 7 {
            splat.sort_by_key(|s| s.contains('*'));
            splat.reverse();

            let mut direct_sums = splat
                .iter()
                .filter(|x| !x.contains("*"))
                .map(|x| x.to_string())
                .collect::<Vec<_>>();
            direct_sums[0] = direct_sums[0].replace("+", "").to_string();
            for i in 1..direct_sums.len() {
                if !direct_sums[i].clone().contains("+") && !direct_sums[i].clone().contains("-") {
                    direct_sums[i] = "+".to_string() + &*direct_sums[i].clone().to_string();
                }
            }
            for q in direct_sums.iter_mut() {
                *q = q.clone().replace(";", "");
            }
            let sum_opt = direct_sums.join("");
            let mut fmas = splat
                .iter()
                .filter(|x| x.contains("*"))
                .map(|x| x.to_string())
                .collect::<Vec<_>>();
            for q in fmas.iter_mut() {
                *q = q.clone().replace(";", "");
            }

            let mut closing_count = 1usize;

            let mut lane_builder = "fmla(".to_string();

            for (i, lane) in fmas.iter().enumerate() {
                if i + 1 < fmas.len() {
                    lane_builder = lane_builder
                        .add(lane.replace("*", ",").replace("+", "").as_str())
                        + ",fmla(";
                    closing_count += 1;
                } else {
                    lane_builder =
                        lane_builder.add(lane.replace("+", "").replace("*", ",").as_str())
                }
            }

            lane_builder = lane_builder.add(format!(", {sum_opt}").as_str());
            for _ in 0..closing_count {
                lane_builder += ")";
            }

            optimized_expr =
                optimized_expr.add(format!("{} = {};\n", lhs, lane_builder.as_str()).as_str());
        } else {
            optimized_expr = optimized_expr.add(format!("{} = {}\n", lhs, splat.join("")).as_str());
        }
    }

    optimized_expr
}
