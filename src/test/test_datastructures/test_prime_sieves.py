import unittest

from data_structures.prime_sieves import SimplePrimeSieve, PrimeSPFsieve

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
                self.assertGreaterEqual(len(ps.sieve), exp_sieve_mn_sz, f"n_max = {init_params['kwargs'].get('n_max', None)}, sieve_max = {init_params['sieve_max']}")
                self.assertLessEqual(len(ps.sieve), exp_sieve_mx_sz, f"n_max = {init_params['kwargs'].get('n_max', None)}, sieve_max = {init_params['sieve_max']}")
                self.assertEqual(ps.sieve[:len(sieve_start)], sieve_start[:len(ps.sieve)], f"n_max = {init_params['kwargs'].get('n_max', None)}, sieve_max = {init_params['sieve_max']}")


