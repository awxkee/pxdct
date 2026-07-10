/*
 * // Copyright (c) Radzivon Bartoshyk 9/2025. All rights reserved.
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

/// Return the prime factors of `n` as a sorted `Vec` with multiplicity.
/// For example: `prime_factors(360) -> [2, 2, 2, 3, 3, 5]`.
pub(crate) fn prime_factors(mut n: u64) -> Vec<u64> {
    static SMALL_PRIMES: [u64; 12] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];

    let mut factors = Vec::new();
    if n < 2 {
        return factors;
    }

    for prime in SMALL_PRIMES {
        while n.is_multiple_of(prime) {
            factors.push(prime);
            n /= prime;
        }
    }

    let mut pending = Vec::new();
    if n > 1 {
        pending.push(n);
    }

    while let Some(value) = pending.pop() {
        if is_prime_u64(value) {
            factors.push(value);
            continue;
        }

        let divisor = pollard_brent(value);
        pending.push(divisor);
        pending.push(value / divisor);
    }

    factors.sort_unstable();
    factors
}

#[inline]
fn gcd_u64(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let remainder = a % b;
        a = b;
        b = remainder;
    }
    a
}

#[inline]
fn mul_mod_u64(a: u64, b: u64, modulus: u64) -> u64 {
    ((a as u128 * b as u128) % modulus as u128) as u64
}

#[inline]
fn rho_step(value: u64, constant: u64, modulus: u64) -> u64 {
    ((mul_mod_u64(value, value, modulus) as u128 + constant as u128) % modulus as u128) as u64
}

#[inline]
fn pow_mod_u64(mut base: u64, mut exponent: u64, modulus: u64) -> u64 {
    let mut result = 1u64;
    base %= modulus;

    while exponent != 0 {
        if exponent & 1 != 0 {
            result = mul_mod_u64(result, base, modulus);
        }
        exponent >>= 1;
        base = mul_mod_u64(base, base, modulus);
    }

    result
}

/// Deterministic Miller-Rabin primality test for the complete `u64` domain.
fn is_prime_u64(n: u64) -> bool {
    static SMALL_PRIMES: [u64; 12] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];
    static WITNESSES: [u64; 7] = [2, 325, 9_375, 28_178, 450_775, 9_780_504, 1_795_265_022];

    if n < 2 {
        return false;
    }

    for prime in SMALL_PRIMES {
        if n.is_multiple_of(prime) {
            return n == prime;
        }
    }

    let powers_of_two = (n - 1).trailing_zeros();
    let odd_part = (n - 1) >> powers_of_two;

    'witness: for witness in WITNESSES {
        let base = witness % n;
        if base == 0 {
            continue;
        }

        let mut value = pow_mod_u64(base, odd_part, n);
        if value == 1 || value == n - 1 {
            continue;
        }

        for _ in 1..powers_of_two {
            value = mul_mod_u64(value, value, n);
            if value == n - 1 {
                continue 'witness;
            }
        }

        return false;
    }

    true
}

#[inline]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

/// Pollard's rho using Brent cycle detection and batched GCDs.
///
/// The caller supplies an odd composite with no factor at most 37.
fn pollard_brent(n: u64) -> u64 {
    const GCD_BATCH: usize = 128;

    debug_assert!(n > 1 && !n.is_multiple_of(2) && !is_prime_u64(n));

    let mut state = n ^ 0x243f_6a88_85a3_08d3;
    loop {
        // Deterministic, decorrelated retries avoid adding a random-number dependency.
        let mut y = 2 + splitmix64(&mut state) % (n - 3);
        let constant = 1 + splitmix64(&mut state) % (n - 1);
        let mut cycle_length = 1usize;
        let mut divisor = 1u64;
        let mut x = 0u64;
        let mut saved_y = 0u64;

        while divisor == 1 {
            x = y;
            for _ in 0..cycle_length {
                y = rho_step(y, constant, n);
            }

            let mut offset = 0usize;
            while offset < cycle_length && divisor == 1 {
                saved_y = y;
                let batch_length = (cycle_length - offset).min(GCD_BATCH);
                let mut product = 1u64;

                for _ in 0..batch_length {
                    y = rho_step(y, constant, n);
                    product = mul_mod_u64(product, x.abs_diff(y), n);
                }

                divisor = gcd_u64(product, n);
                offset += batch_length;
            }

            let Some(next_cycle_length) = cycle_length.checked_mul(2) else {
                divisor = n;
                break;
            };
            cycle_length = next_cycle_length;
        }

        if divisor == n {
            loop {
                saved_y = rho_step(saved_y, constant, n);
                divisor = gcd_u64(x.abs_diff(saved_y), n);
                if divisor != 1 {
                    break;
                }
            }
        }

        if divisor != n {
            return divisor;
        }
    }
}

/// Return the prime factorization as (prime, exponent) pairs.
/// Example: `prime_factorization(360) -> [(2,3), (3,2), (5,1)]`.
pub(crate) fn prime_factorization(n: u64) -> Vec<(u64, u32)> {
    let factors = prime_factors(n);
    let mut out = Vec::new();
    let mut iter = factors.into_iter();
    if let Some(mut cur) = iter.next() {
        let mut cnt: u32 = 1;
        for f in iter {
            if f == cur {
                cnt += 1;
            } else {
                out.push((cur, cnt));
                cur = f;
                cnt = 1;
            }
        }
        out.push((cur, cnt));
    }
    out
}

#[derive(Clone, Ord, PartialOrd, Eq, PartialEq, Debug)]
pub(crate) struct PrimeFactors {
    pub(crate) n: u64,
    pub(crate) is_power_of_two: bool,
    pub(crate) factorization: Vec<(u64, u32)>,
}

impl PrimeFactors {
    pub(crate) fn from_number(n: u64) -> PrimeFactors {
        let is_power_of_two = n.is_power_of_two();
        let factorization = prime_factorization(n);
        PrimeFactors {
            n,
            is_power_of_two,
            factorization,
        }
    }

    pub(crate) fn factor_of_2(&self) -> u32 {
        self.factorization
            .iter()
            .find(|p| p.0 == 2)
            .map(|x| x.1)
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_small() {
        assert_eq!(prime_factors(1), Vec::<u64>::new());
        assert_eq!(prime_factors(2), vec![2]);
        assert_eq!(prime_factors(3), vec![3]);
        assert_eq!(prime_factors(4), vec![2, 2]);
        assert_eq!(prime_factors(18), vec![2, 3, 3]);
        assert_eq!(prime_factorization(1296), vec![(2, 4), (3, 4)]);
        assert_eq!(prime_factorization(360), vec![(2, 3), (3, 2), (5, 1)]);
        assert_eq!(prime_factorization(20), vec![(2, 2), (5, 1)]);
        assert_eq!(prime_factorization(97), vec![(97, 1)]);
        assert_eq!(prime_factorization(36), vec![(2, 2), (3, 2)]);
        assert_eq!(prime_factorization(36 * 6), vec![(2, 3), (3, 3)]);
    }

    #[test]
    fn test_large_prime() {
        let p = 4_294_967_291u64; // this is prime
        assert_eq!(prime_factors(p), vec![p]);
        assert_eq!(prime_factorization(p), vec![(p, 1)]);
        assert_eq!(prime_factorization(2028), vec![(2, 2), (3, 1), (13, 2)]);
        assert_eq!(prime_factorization(900), vec![(2, 2), (3, 2), (5, 2)]);
        assert_eq!(prime_factorization(121), vec![(11, 2)]);
        assert_eq!(prime_factorization(1312), vec![(2, 5), (41, 1)]);
        assert_eq!(prime_factorization(1201), vec![(1201, 1)]);
        assert_eq!(prime_factorization(1200), vec![(2, 4), (3, 1), (5, 2)]);
        assert_eq!(prime_factorization(1295), vec![(5, 1), (7, 1), (37, 1)]);
        assert_eq!(prime_factorization(1859), vec![(11, 1), (13, 2)]);
    }

    #[test]
    fn test_full_u64_domain() {
        let largest_u64_prime = 18_446_744_073_709_551_557u64;
        assert_eq!(prime_factors(largest_u64_prime), vec![largest_u64_prime]);

        let p = 4_294_967_291u64;
        let q = 4_294_967_279u64;
        assert_eq!(prime_factors(p * q), vec![q, p]);

        assert_eq!(
            prime_factors(u64::MAX),
            vec![3, 5, 17, 257, 641, 65_537, 6_700_417]
        );
    }

    #[test]
    fn rejects_strong_pseudoprime() {
        assert_eq!(
            prime_factors(341_550_071_728_321),
            vec![10_670_053, 32_010_157]
        );
    }

    #[test]
    fn test_composite() {
        let n = 2u64.pow(10) * 3u64.pow(6) * 7u64;
        assert_eq!(prime_factorization(n), vec![(2, 10), (3, 6), (7, 1)]);
    }
}
