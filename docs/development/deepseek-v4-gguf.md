# DeepSeek-V4 GGUF notes

## Status

This document records a **working DeepSeek-V4 GGUF contract** for ongoing `llama.cpp` support work. It is downstream development documentation for fork-side bring-up, not a claim that DeepSeek-V4 already has complete end-to-end runtime support in tree.

The first concrete target is **DeepSeek-V4-Pro**.

The goal is to keep conversion, calibration, and GGUF handoff semantics stable while implementation work is still evolving.

## Why this note exists

DeepSeek-V4-Pro sits at an awkward intersection:

- the official release is a mixed **FP8 + FP4** model,
- routed experts use grouped scales and require calibration-aware lossy conversions,
- prompt formatting and special-token behavior are model-specific,
- the safest transport format for deployment and handoff is still **GGUF**.

Without a written contract, temporary bridge decisions tend to become accidental long-term semantics.

## Keep three artifact layers distinct

For DeepSeek-V4-Pro work, keep these layers separate:

1. **Source checkpoint** - the original safetensors release plus the official config.
2. **Calibration record** - prompt selection, route coverage, shard manifests, quality gates, and quantization decisions.
3. **GGUF artifact** - the deployable single-file or split output.

GGUF is the portable deployment container. It is **not** the only source of truth.

## Compatibility stance

If current conversion or loader work has to bridge through the nearest supported DeepSeek-like architecture before a native `deepseek_v4` architecture exists, keep that bridge as an implementation detail.

The important rule is:

- do **not** pretend a temporary bridge means ordinary DeepSeek2 semantics,
- do **not** discard V4-specific metadata just because a transport bridge is used,
- do **not** claim runtime completeness for partial or calibration-only outputs.

When a native `deepseek_v4` architecture eventually lands, the metadata below should keep the same meaning.

## Recommended GGUF metadata

The following keys are recommended for DeepSeek-V4 GGUF artifacts:

| Key | Recommended value / shape | Notes |
| --- | --- | --- |
| `deepseek_v4.profile.id` | `DeepSeek-V4-GGUF-Profile-v0` | Stable profile identifier |
| `deepseek_v4.profile.version` | `0` | Integer profile version |
| `deepseek_v4.model.variant` | `DeepSeek-V4-Pro` | First concrete target |
| `deepseek_v4.artifact.kind` | `full-model`, `split-full-model`, `expert-shard`, or `merge-fragment` | Distinguishes complete vs partial outputs |
| `deepseek_v4.source.format` | `safetensors` | Canonical source format |
| `deepseek_v4.target.format` | `gguf` | Deployment / handoff container |
| `deepseek_v4.source.repo` | `deepseek-ai/DeepSeek-V4-Pro` | Source lineage |
| `deepseek_v4.source.dense_dtype` | `fp8` | Official dense-weight storage contract |
| `deepseek_v4.source.expert_dtype` | `fp4-e2m1` | Official routed-expert source contract |
| `deepseek_v4.source.expert_scale_block` | `32` | One scale per 32 K-elements |
| `deepseek_v4.storage.dense_qtype` | freeform stable string | Stored GGUF dense representation |
| `deepseek_v4.storage.expert_qtype` | freeform stable string | Stored GGUF expert representation |
| `deepseek_v4.compatibility.targets` | string list | Example: `llama.cpp` |
| `deepseek_v4.calibration.manifest` | path, URI, checksum, or `none` | Required for lossy expert quantization |
| `deepseek_v4.tensor.inventory` | checksum, manifest path, or both | Makes full/partial coverage auditable |

Useful additional keys include:

- `deepseek_v4.source.config_sha256`
- `deepseek_v4.source.checkpoint_id`
- `deepseek_v4.quant.imatrix`
- `deepseek_v4.quant.group_size`
- `deepseek_v4.quant.calibration_corpus_sha256`
- `deepseek_v4.quant.route_coverage_sha256`
- `deepseek_v4.partial.layer_range`
- `deepseek_v4.partial.expert_range`

## Baseline invariants for DeepSeek-V4-Pro

These values are the important architecture anchors to preserve for the baseline profile:

| Field | Value |
| --- | ---: |
| `vocab_size` | `129280` |
| `dim` | `7168` |
| `n_layers` | `61` |
| `n_mtp_layers` | `1` |
| `n_heads` | `128` |
| `n_routed_experts` | `384` |
| `n_shared_experts` | `1` |
| `n_activated_experts` | `6` |
| `score_func` | `sqrtsoftplus` |
| `route_scale` | `2.5` |
| `head_dim` | `512` |
| `rope_head_dim` | `64` |
| `original_seq_len` | `65536` |
| `rope_theta` | `10000` |
| `rope_factor` | `16` |
| `dtype` | `fp8` |
| `scale_fmt` | `ue8m0` |
| `expert_dtype` | `fp4` |
| `fp4_block_size` | `32` |
| `compress_rope_theta` | `160000` |

In addition, the canonical compression schedule for the baseline profile should be preserved either directly in metadata or by linking the exact source config.

## Tensor naming guidance

Reuse existing stable GGUF / DeepSeek mappings when the semantics are actually equivalent.

For V4-only tensors, prefer stable additive names over squeezing them into unrelated slots. In particular, do not silently drop or repurpose:

- `hc_*` tensors,
- `ape` tensors,
- `tie2eid` / early-hash routing tensors,
- `attn_sink`,
- `weights_proj`,
- MTP-only tensors.

Representative normalized names from existing V4 conversion work are:

| Source name | Canonical converted name |
| --- | --- |
| `embed_tokens`, `embed` | `embed` |
| `input_layernorm` | `attn_norm` |
| `post_attention_layernorm` | `ffn_norm` |
| `q_a_proj` | `wq_a` |
| `q_a_layernorm` | `q_norm` |
| `q_b_proj` | `wq_b` |
| `kv_a_proj_with_mqa` | `wkv_a` |
| `kv_a_layernorm` | `kv_norm` |
| `kv_b_proj` | `wkv_b` |
| `o_proj` | `wo` |
| `gate_proj` | `w1` |
| `down_proj` | `w2` |
| `up_proj` | `w3` |
| `lm_head`, `head` | `head` |

## Quantization and calibration rules

DeepSeek-V4-Pro expert FP4 should not be relabeled casually.

In practice this means:

- distinguish the **official source dtype** from the **stored GGUF qtype**,
- do not imply the source format was `MXFP4` or `MMFP4` unless that conversion actually happened,
- keep lossy expert quantization traceable back to calibration inputs.

If routed experts are stored in lossy GGUF formats such as `IQ3`, `IQ2`, or block-quant variants, the calibration record should preserve at least:

- prompt corpus hash,
- route-coverage summary or histogram hash,
- quantization scheme,
- group size / block size,
- compatibility targets,
- shard-plan identity when distributed calibration was used.

## Tokenizer and prompt contract

Runtime-ready artifacts must preserve or reference the tokenizer and prompt-format contract.

At minimum, preserve compatibility with:

- `vocab_size = 129280`,
- DeepSeek-V4 special tokens such as `<｜begin▁of▁sentence｜>`, `<｜end▁of▁sentence｜>`, `<｜User｜>`, `<｜Assistant｜>`, `<｜latest_reminder｜>`, `<think>`, `</think>`, and `｜DSML｜`,
- a marker that the prompt encoding is **DeepSeek-V4 encoding**, not a generic chat template.

Useful metadata:

- `deepseek_v4.prompt.encoding = dsv4`
- `deepseek_v4.prompt.reasoning_tags = <think>,</think>`
- `deepseek_v4.prompt.tool_format = dsml`

Artifacts that omit tokenizer or prompt-format information should not claim runtime-ready conformance.

## Split files, partial artifacts, and distributed handoff

Prefer a single GGUF file when practical.

If the output is split, use the standard suffix:

- `-00001-of-000NN.gguf`

All chunks in one logical artifact should share one stable prefix, for example:

- `DeepSeek-V4-Pro-IQ3-profile-v0.gguf`
- `DeepSeek-V4-Pro-IQ3-profile-v0-00001-of-00008.gguf`

Expert-only or shard-only GGUF outputs are acceptable as **transport** or **merge** artifacts, but they must be explicit about that status.

Any artifact that does not contain the full runnable tensor set should:

- set `deepseek_v4.artifact.kind` to `expert-shard` or `merge-fragment`,
- declare its layer / expert coverage,
- provide or reference an inventory manifest,
- avoid claiming runtime-ready conformance.

This matters for distributed calibration: the returned object may be a valid GGUF container and still be only part of the final model.

## Rule of thumb

If the answer is **yes** to all of the following, the artifact is probably specific enough to be useful:

- Can I tell that it is DeepSeek-V4-Pro rather than generic DeepSeek2?
- Can I tell whether it is full, split, or partial?
- Can I tell the source dtype separately from the stored GGUF qtype?
- Can I recover calibration lineage for lossy experts?
- Can I recover the tokenizer and prompt-format contract?

If not, the artifact is still too ambiguous to be a stable DeepSeek-V4 baseline.