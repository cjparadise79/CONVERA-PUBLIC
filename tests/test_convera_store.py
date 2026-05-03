from __future__ import annotations

import tempfile
import unittest

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from convera_store_lite.store import ChunkStore


@unittest.skipIf(torch is None, "torch is required for tensor store tests")
class ConveraStoreTests(unittest.TestCase):
    def test_tensor_round_trip_is_exact_and_reuses_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ChunkStore(temp_dir, ram_cache_items=8)
            tensor = torch.arange(128, dtype=torch.float32).reshape(16, 8)

            manifest = store.store_tensor(tensor, chunk_size=64)
            restored = store.load_tensor(manifest)
            self.assertTrue(torch.equal(tensor, restored))

            store.store_tensor(tensor, chunk_size=64)
            self.assertGreater(store.reuse_ratio(), 0.0)


if __name__ == "__main__":
    unittest.main()
