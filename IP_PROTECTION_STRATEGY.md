# CONVERA IP Protection Strategy

This document defines how CONVERA can offer protected functionality without exposing non-public algorithms, research names, scoring logic, or implementation details in the public repository.

## Principle

Public code should be clean, useful, and inspectable. Protected functionality should live behind a stable boundary.

Do not rely on intentionally tangled source code as the main protection method. It is harder to maintain, weak against determined reverse engineering, and lowers trust in the public project. Use clean boundaries, separate distribution, license controls, and server-side execution instead.

## Public Repository Rules

The public repository may include:

- CLI and UI.
- Model loading.
- Public cache behavior.
- Basic fixed-size tensor storage.
- Public benchmark and visualization tools.
- Metrics-only telemetry.
- Redacted audit traces.
- Stable API contracts.
- Neutral names for extension points.

The public repository must not include:

- Non-public research names.
- Non-public algorithms.
- Routing internals.
- Heuristic weights.
- Scoring formulas.
- Token-level traces.
- Cache internals.
- Private endpoint URLs.
- Private API keys.
- Debug dumps that reveal decision paths.
- Audit exports that include prompts, outputs, raw tokens, local paths, or private decision data.

## Supported Protection Patterns

### 1. Hosted Capability Service

The strongest protection model is server-side execution.

Public CONVERA sends a minimal, privacy-safe feature request to a licensed service. The service returns a high-level execution decision.

Example response shape:

```json
{
  "mode": "reuse",
  "precision": "int8",
  "confidence": 0.84,
  "reason_code": "eligible"
}
```

The service must not return:

- Internal scores.
- Rule traces.
- Graphs.
- Intermediate states.
- Explanations of proprietary decision logic.

### 2. Licensed Private Package

Enterprise or private deployments may install a separate package that implements protected acceleration.

Public code should call only a narrow interface. The private package should be distributed outside the public repository.

Recommended interface style:

```python
plan = runtime_provider.plan_execution(public_features)
result = runtime_provider.execute(plan, request)
```

The public repository should not import the private package directly. Use dependency injection or deployment wiring outside the public tree.

### 3. Signed Binary Extension

For offline deployments, protected logic may be shipped as a signed binary extension or wheel.

Use this only as an additional layer. Binary packaging slows casual inspection but is not a substitute for a clean legal and architectural boundary.

### 4. License-Gated Features

Protected functionality can be activated by license tier without revealing implementation details.

The public project can display neutral feature availability:

```text
Lite Runtime: enabled
Advanced Runtime: unavailable
```

Avoid public labels that reveal internal research names.

## Adaptive Precision Boundary

CONVERA may expose runtime-aware precision selection publicly at the behavior level.

Safe public concepts:

- `fp16`
- `int8`
- `int4`
- `cached`
- `computed`
- `latency target`
- `confidence`

Keep private:

- How confidence is calculated.
- How reuse likelihood is scored.
- Any non-public decision process.
- Any layer-wise or token-wise importance logic.
- Any non-public routing formulas.

The public API should return decisions, not the reasoning machinery behind them.

Safe response shape:

```json
{
  "precision": "int8",
  "mode": "computed",
  "confidence": 0.72
}
```

## File And Naming Rules

Use neutral public names:

- `execution_router`
- `precision_policy`
- `runtime_provider`
- `response_cache`
- `kv_cache_store`

Avoid names that disclose non-public research systems, decision families, or internal project architecture.

## Review Checklist

Before pushing public updates:

1. Run a sensitive-name scan.
2. Scan comments and docstrings.
3. Scan telemetry payloads.
4. Scan imports for non-public modules.
5. Confirm public history is sanitized.
6. Confirm public docs explain behavior, not protected mechanisms.
7. Confirm benchmark output exposes only public-safe metrics.
8. Confirm audit exports are redacted.

Recommended scans:

```bash
rg -n -i "internal|private|proprietary|token trace|cache internals|raw token|embedding" .
rg -n "PRIVATE_ENDPOINT|API_KEY|SECRET|PASSWORD" .
```

Review results manually. Some words may be acceptable in protection documents, but implementation files should stay neutral.

## Positioning

Public message:

```text
CONVERA reduces redundant local inference work through reusable runtime state, public benchmarking, and optional advanced runtime extensions.
```

Do not claim that public CONVERA exposes the full protected runtime.
