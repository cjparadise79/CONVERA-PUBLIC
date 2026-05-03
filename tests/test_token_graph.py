from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from inference.token_graph import TokenGraph


class TokenGraphTests(unittest.TestCase):
    def test_longest_path_handles_large_token_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            graph = TokenGraph(Path(temp_dir) / "graph.json")
            graph.add_sequence([128000, 42, 9001], kv_ref="kv-a")

            state_hash, prefix_len = graph.find_longest_path([128000, 42, 9001, 7])
            self.assertIsNotNone(state_hash)
            self.assertEqual(prefix_len, 3)


if __name__ == "__main__":
    unittest.main()

