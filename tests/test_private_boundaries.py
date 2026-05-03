from __future__ import annotations

import unittest


class PrivateBoundaryTests(unittest.TestCase):
    def test_lite_storage_name_is_public_boundary(self) -> None:
        import convera_store_lite

        self.assertTrue(convera_store_lite.__doc__)

    def test_core_api_exposes_only_public_contract(self) -> None:
        from convera_core_api import interface

        self.assertEqual(
            set(interface.__all__),
            {"ADVANCED_MODE", "load_tensor", "merge_states", "optimize_kv", "store_tensor"},
        )
        self.assertFalse(interface.ADVANCED_MODE)

    def test_core_api_lite_state_ops_return_minimal_results(self) -> None:
        from convera_core_api.interface import merge_states, optimize_kv

        state_a = {"id": "a"}
        state_b = {"id": "b"}
        self.assertIs(optimize_kv(state_a), state_a)
        self.assertIs(merge_states(state_a, state_b), state_b)


if __name__ == "__main__":
    unittest.main()
