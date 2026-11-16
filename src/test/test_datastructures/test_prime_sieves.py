import unittest

from data_structures.prime_sieves import SimplePrimeSieve, PrimeSPFsieve

import math

class TestSimplePrimeSieve(unittest.TestCase):

    def test_SimplePrimeSieve_init_noerr(self):
        for init_params in [
            {"kwargs": {}, "sieve_max": 10 ** 7, "prime_search_cycle_size_max": 10 ** 5},

        ]:
            pass

    def test_SimplePrimeSieve_sieve_noerr(self):
        sieve_start = [False, False, True, True, False, True, False, True, False, False, False, True, False, True, False, False, False]
        for init_params in [
            {"kwargs": {}},
            {"kwargs": {"n_max": 1}},
            {"kwargs": {"n_max": 2}},
            {"kwargs": {"n_max": 3}},
            {"kwargs": {"n_max": 4}},
            {"kwargs": {"n_max": 5}},
            {"kwargs": {"n_max": 10}},
            {"kwargs": {"n_max": 100}},
            {"kwargs": {"n_max": 10 ** 3}},
            {"kwargs": {"n_max": 10, "sieve_max": 20}},
            {"kwargs": {"n_max": 10, "sieve_max": 5}},
            {"kwargs": {"n_max": 20, "sieve_max": 10}},
        ]:
            init_params.setdefault("sieve_max", init_params["kwargs"].get("sieve_max", 10 ** 7))
            for pscsm in ([None, 1, 2, 5, 20, 10 ** 4] if "prime_search_cycle_size_max" not in init_params["kwargs"].keys() else [None]):
                if pscsm is not None:
                    init_params["kwargs"]["prime_search_cycle_size_max"] = pscsm
                
                ps = SimplePrimeSieve(**init_params["kwargs"])
                exp_sieve_mn_sz = min(init_params["sieve_max"], init_params["kwargs"].get("n_max", -1)) + 1
                exp_sieve_mx_sz = min(init_params["sieve_max"], max(init_params["kwargs"].get("n_max", -1), 3)) + 1
                # Checking the sieve is the expected size for the given initialization
                self.assertGreaterEqual(len(ps.sieve), exp_sieve_mn_sz, f"n_max = {init_params['kwargs'].get('n_max', None)}, sieve_max = {init_params['sieve_max']}")
                self.assertLessEqual(len(ps.sieve), exp_sieve_mx_sz, f"n_max = {init_params['kwargs'].get('n_max', None)}, sieve_max = {init_params['sieve_max']}")
                # Checking the initial part of the sieve equals the expected values
                self.assertEqual(ps.sieve[:len(sieve_start)], sieve_start[:len(ps.sieve)], f"n_max = {init_params['kwargs'].get('n_max', None)}, sieve_max = {init_params['sieve_max']}")

    def test_SimplePrimeSieve_isPrime_noerr(self):
        nonprime_cases = [
            1,
            4,
            6,
            8,
            9,
            10,
            25,
            49,
            81,
            91,
            121,
            341,
            # Carmichael numbers
            561,
            1105,
            1729,
            # ****
            2047,
            123456789987654321,
        ]
        prime_cases = [
            2,
            3,
            5,
            7,
            11,
            13,
            17,
            19,
            23,
            8675309,
            10 ** 9 + 7,
            (1 << 61) - 1,
        ]

        #invalid_cases =[
        #    *list(range(-11, 1)),
        #    math.pi,
        #]
        for init_params in [
            {"kwargs": {}},
            {"kwargs": {"n_max": 10, "sieve_max": 20}},
            {"kwargs": {"n_max": 10, "sieve_max": 5}},
            {"kwargs": {"n_max": 20, "sieve_max": 10}},
        ]:
            init_params.setdefault("sieve_max", init_params["kwargs"].get("sieve_max", 10 ** 7))
            for pscsm in ([None, 1, 2, 5, 20, 10 ** 4] if "prime_search_cycle_size_max" not in init_params["kwargs"].keys() else [None]):
                if pscsm is not None:
                    init_params["kwargs"]["prime_search_cycle_size_max"] = pscsm
                
                for extend_sieve in [True, False]:
                    for extend_sieve_sqrt in [True, False]:
                        for use_miller_rabin_screening in [True, False]:
                            it = range(1, 11) if use_miller_rabin_screening else [3]
                            for n_miller_rabin_trials in it:
                                # Checking the method isPrime() returns False for non-primes
                                for n in nonprime_cases:
                                    if not use_miller_rabin_screening and n > 10 ** 16: continue
                                    ps = SimplePrimeSieve(**init_params["kwargs"])
                                    print(f"ps.isPrime({n}, extend_sieve={extend_sieve}, extend_sieve_sqrt={extend_sieve_sqrt}, use_miller_rabin_screening={use_miller_rabin_screening}, n_miller_rabin_trials={n_miller_rabin_trials})")
                                    
                                    ans = ps.isPrime(
                                        n,
                                        extend_sieve=extend_sieve,
                                        extend_sieve_sqrt=extend_sieve_sqrt,
                                        use_miller_rabin_screening=use_miller_rabin_screening,
                                        n_miller_rabin_trials=n_miller_rabin_trials
                                    )
                                    self.assertFalse(ans, msg=f"ps.isPrime({n}, extend_sieve={extend_sieve}, extend_sieve_sqrt={extend_sieve_sqrt}, use_miller_rabin_screening={use_miller_rabin_screening}, n_miller_rabin_trials={n_miller_rabin_trials}) returned True even though {n} is not prime.")
                                # Checking the method isPrime() returns True for primes
                                for n in prime_cases:
                                    if not use_miller_rabin_screening and n > 10 ** 16: continue
                                    ps = SimplePrimeSieve(**init_params["kwargs"])
                                    print(f"ps.isPrime({n}, extend_sieve={extend_sieve}, extend_sieve_sqrt={extend_sieve_sqrt}, use_miller_rabin_screening={use_miller_rabin_screening}, n_miller_rabin_trials={n_miller_rabin_trials})")
                                    ans = ps.isPrime(
                                        n,
                                        extend_sieve=extend_sieve,
                                        extend_sieve_sqrt=extend_sieve_sqrt,
                                        use_miller_rabin_screening=use_miller_rabin_screening,
                                        n_miller_rabin_trials=n_miller_rabin_trials,
                                    )
                                    self.assertTrue(ans, msg=f"ps.isPrime({n}, extend_sieve={extend_sieve}, extend_sieve_sqrt={extend_sieve_sqrt}, use_miller_rabin_screening={use_miller_rabin_screening}, n_miller_rabin_trials={n_miller_rabin_trials}) returned False even though {n} is prime.")
