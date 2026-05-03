# CONVERA Open-Core Boundary

This repository is intended to be safe for public release as CONVERA-OSS.

## Public

- CLI and local UI
- Hugging Face model loading
- backend abstraction
- prompt-level KV persistence
- local token graph
- fixed-size `convera_store_lite` chunk store
- benchmarks and metrics-only telemetry client
- adaptive public outcome tracking
- execution inspector and audit trace export

## Private

- runtime acceleration systems
- enterprise runtime extensions
- adaptive precision and execution planning systems
- advanced audit and validation backends

## Interface

All public/private integration goes through:

```python
from convera_core_api import interface
```

The interface exposes only:

- `store_tensor(tensor)`
- `load_tensor(refs)`
- `optimize_kv(kv_tensor)`
- `merge_states(state_a, state_b)`

These functions must return only tensors, reference IDs, or simple metadata.
They must not return scoring, intermediate states, or debug details.

Public code must not include non-public algorithms, research notes, heuristic
details, or non-public module names.

## Protection Model

Protected functionality should be offered through stable contracts, licensed
packages, or hosted capability services. Public code should remain clean and
auditable; do not make public source intentionally tangled as a primary
protection method.

Public APIs may expose high-level decisions such as:

- `cached`
- `computed`
- `fp16`
- `int8`
- `int4`
- `confidence`

Public APIs must not expose the reasoning machinery, scoring formulas,
intermediate states, or private decision traces behind those decisions.

Audit traces may expose only redacted public behavior: request IDs, prompt
hashes, route mode, precision, latency, token counts, and validation status.
They must not expose prompts, outputs, raw tokens, local paths, private formulas,
or private implementation details.
