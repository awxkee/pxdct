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
/// Return the prime factors of `n` as a Vec with multiplicity.
/// For example: `prime_factors(360) -> [2,2,2,3,3,5]`.
/// Special cases:
///  - n == 0 -> returns empty vec (undefined factorization)
///  - n == 1 -> returns empty vec
pub(crate) fn prime_factors(mut n: u64) -> Vec<u64> {
    let mut res = Vec::new();
    if n < 2 {
        return res;
    }

    // factor out 2s
    while (n & 1) == 0 {
        res.push(2);
        n >>= 1;
    }

    // factor out 3s
    while n.is_multiple_of(3) {
        res.push(3);
        n /= 3;
    }

    // trial divide by 6k - 1 and 6k + 1
    let mut p: u64 = 5;
    while (p as u128) * (p as u128) <= n as u128 {
        while n.is_multiple_of(p) {
            res.push(p);
            n /= p;
        }
        let q = p + 2; // p = 6k-1, q = 6k+1
        while n.is_multiple_of(q) {
            res.push(q);
            n /= q;
        }
        p += 6;
    }

    // if remaining n > 1 it's prime
    if n > 1 {
        res.push(n);
    }
    res
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
    fn test_composite() {
        let n = 2u64.pow(10) * 3u64.pow(6) * 7u64;
        assert_eq!(prime_factorization(n), vec![(2, 10), (3, 6), (7, 1)]);
    }
}
