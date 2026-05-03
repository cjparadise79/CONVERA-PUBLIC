# SERVER HAND OFF

This document is the handoff for the server-side Codex that will build the CONVERA telemetry receiver. It intentionally describes only the public metrics contract and does not include private runtime internals, private optimization logic, secrets, prompts, outputs, local paths, or model weights.

## Goal

Build a small telemetry service that accepts metrics-only reports from CONVERA runtimes and makes those metrics available for dashboards, health checks, and later billing or license enforcement.

The first server version should be boring and reliable:

- receive metrics
- validate privacy-safe payloads
- authenticate requests
- store accepted reports
- expose aggregate summaries
- never block local inference if the server is down

## Client Contract

Public CONVERA clients send telemetry only when both environment variables are configured:

```text
CONVERA_METRICS_API_URL
CONVERA_METRICS_API_KEY
```

Recommended URL path:

```text
POST /api/convera/metrics
```

The client sends:

- `Authorization: Bearer <api-key>`
- `Content-Type: application/json`
- timeout no higher than 3 seconds

Client failures must be non-fatal. If the server is offline, rejects the report, or times out, local inference continues.

## Privacy Contract

The telemetry server must reject any payload that contains private user content.

Never accept or store:

- prompts
- model outputs
- file names
- local file paths
- source code snippets
- model weights
- raw tokens
- raw embeddings
- stack traces containing local paths
- machine usernames
- API keys or auth headers

Allowed data:

- latency
- tokens per second
- generated token counts
- cache hit rate
- chunk reuse ratio
- approximate disk usage
- GPU name and VRAM usage
- model identifier string
- runtime mode
- OS family and architecture

## Payload Shape

Expected top-level shape:

```json
{
  "report_kind": "convera-metrics-only",
  "privacy_mode": "no-file-names-or-paths",
  "session_id": "uuid",
  "created_at": "2026-05-02T12:00:00Z",
  "customer_details": {},
  "file_count": 0,
  "bundle_count": 0,
  "failed_file_count": 0,
  "completed_with_warnings": false,
  "total_original_bytes": 0,
  "total_stored_bytes": 1024,
  "total_logical_encoded_bytes": 1024,
  "weighted_reduction_percent": 25.0,
  "all_hashes_match": true,
  "metrics": {
    "performance_metrics": {
      "latency_seconds": 1.2,
      "tokens_per_second": 55.0,
      "tokens_generated": 20
    },
    "inference_metrics": {
      "kv_cache_hit_rate": 0.5,
      "chunk_reuse_ratio": 0.25
    },
    "gpu_metrics": {
      "gpu_name": "cpu",
      "vram_used_mb": 0.0
    },
    "model_metrics": {
      "model_name": "llama3",
      "precision": "fp16",
      "quantized": false
    },
    "environment_tags": {
      "engine": "convera",
      "mode": "inference",
      "deployment": "local",
      "system": "Darwin",
      "machine": "arm64"
    }
  },
  "failure_summary": {
    "total_failures": 0,
    "by_stage": [],
    "by_error_type": [],
    "by_stage_and_error_type": []
  }
}
```

## Server Endpoints

Build these first:

```text
GET  /health
POST /api/convera/metrics
GET  /api/convera/metrics/summary
GET  /api/convera/metrics/recent
```

`GET /health` returns service status and schema version.

`POST /api/convera/metrics` validates, authenticates, stores, and returns:

```json
{
  "accepted": true,
  "report_id": "uuid"
}
```

`GET /api/convera/metrics/summary` returns aggregate counts and averages:

```json
{
  "reports": 120,
  "avg_latency_seconds": 1.35,
  "avg_tokens_per_second": 48.2,
  "avg_kv_cache_hit_rate": 0.31,
  "avg_chunk_reuse_ratio": 0.18
}
```

`GET /api/convera/metrics/recent` returns the latest redacted report summaries only. Do not return raw request headers.

## Validation Rules

Reject with `400` when:

- `report_kind` is not `convera-metrics-only`
- `privacy_mode` is not `no-file-names-or-paths`
- required metric objects are missing
- numeric fields contain NaN or Infinity
- payload size exceeds 64 KB
- forbidden keys appear anywhere in the JSON tree

Forbidden key fragments:

```text
prompt
output
completion
path
filename
file_name
source
code
token_ids
embedding
password
secret
api_key
authorization
```

Reject with `401` when the bearer token is missing.

Reject with `403` when the bearer token is invalid or disabled.

Reject with `429` when rate limits are exceeded.

## Authentication

Use bearer tokens for v1.

Implementation requirements:

- store only hashed API keys server-side
- support enabled/disabled keys
- support optional tier labels
- support key rotation
- log key ID or key name only, never the raw key

Suggested table:

```text
api_keys
- id
- name
- key_hash
- tier
- enabled
- created_at
- last_used_at
```

## Storage

Use a real database, not flat files, for the server implementation.

Suggested table:

```text
metric_reports
- id
- api_key_id
- report_kind
- privacy_mode
- session_id
- created_at_client
- received_at_server
- latency_seconds
- tokens_per_second
- tokens_generated
- kv_cache_hit_rate
- chunk_reuse_ratio
- disk_usage_bytes
- gpu_name
- vram_used_mb
- model_name
- precision
- quantized
- engine
- deployment
- system
- machine
- raw_metrics_json
```

Keep `raw_metrics_json` only after validation and redaction checks pass.

## Rate Limits

Initial defaults:

- free tier: 60 reports/minute/key
- pro tier: 600 reports/minute/key
- enterprise tier: configurable

Rate-limit by API key and source IP.

## Logging

Server logs must include:

- request ID
- API key ID or key name
- accepted/rejected status
- validation error code
- latency

Server logs must not include:

- Authorization header
- raw payload when rejected for privacy
- raw prompts or outputs
- local paths

## Dashboard Data

The server should prepare aggregate fields for future UI panels:

- report count over time
- average latency
- tokens per second
- cache hit rate
- reuse ratio
- model usage
- runtime deployment type
- GPU/CPU distribution
- warning/failure count

## Server Codex Build Checklist

1. Create telemetry service with `GET /health`.
2. Add bearer-token auth with hashed key storage.
3. Add payload schema validation.
4. Add forbidden-key recursive scanner.
5. Add `POST /api/convera/metrics`.
6. Add database persistence.
7. Add summary and recent-report endpoints.
8. Add rate limiting.
9. Add redacted logging.
10. Add tests for accepted payloads, rejected privacy violations, missing auth, invalid auth, and rate limiting.

## Client Update Points

If the server path changes, update `CONVERA_METRICS_API_URL` in the deployment environment. The CONVERA client code should not hardcode production telemetry URLs or API keys.

## Open Questions

- Final deployment host and path.
- Database choice for v1.
- API key issuance workflow.
- Whether summary endpoints are admin-only or key-scoped.
- Retention window for accepted metrics.
- Tier names and rate limits for production.
