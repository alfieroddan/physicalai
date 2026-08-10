# Fuzzing

## Intro

Physical AI Runtime is a robot control loop that loads an exported, fine-tuned VLA policy, reads sensor observations, performs inference, and sends action vectors to the hardware. Because the system consumes file-system artifacts (such as manifests, model weights, and statistics files) and NumPy array streams (including camera frames and robot joint readings), it must be able to handle malformed, adversarial, and out-of-range inputs without crashing or generating unsafe actions. Fuzzing is used to identify input-related issues that could cause crashes, unexpected behavior, or unsafe actions.

Fuzzing automation uses [Atheris](https://github.com/google/atheris) - a coverage-guided Python fuzzer backed by libFuzzer.
The GitHub Actions workflow `.github/workflows/fuzz.yml` runs all harnesses in parallel on a schedule and on every PR that touches `tests/fuzz/` or the workflow file itself. Crash artifacts are uploaded for the future analysis.

## Fuzz target inventory

Each row maps a harness file to the component it covers, its input space, and the oracle invariants it checks. Oracle numbers refer to the [Key invariants for fuzzing oracles](#key-invariants-for-fuzzing-oracles) section below.

| Harness                                                                  | Entry Point                                                                  | Input Space                                                                                                                                   | Oracles       |
| ------------------------------------------------------------------------ | ---------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- | ------------- |
| [`fuzz_manifest.py`](harnesses/fuzz_manifest.py)                         | `Manifest.load(path)`                                                        | Arbitrary JSON bytes (valid and malformed); extreme values in `shape`, `n_obs_steps`, unknown keys                                            | I-3           |
| [`fuzz_component.py`](harnesses/fuzz_component.py)                       | `instantiate_component(spec)`, `resolve_artifact(spec, export_dir)`          | Mutated `class_path` strings; crafted `artifact` paths (`../`, absolute, symlink-like); deep nested `init_args`; arbitrary `flat_params` keys | I-1, I-4, I-5 |
| [`fuzz_import_dotted_path.py`](harnesses/fuzz_import_dotted_path.py)     | `import_dotted_path(path)`                                                   | Arbitrary dotted strings (relative imports, no-dot strings, known-safe module roots + fuzz suffix)                                            | I-6           |
| [`fuzz_policy_name.py`](harnesses/fuzz_policy_name.py)                   | `_is_safe_policy_name(name)`, `InferenceModel(export_dir, policy_name=name)` | Arbitrary Unicode strings; strings that pass and fail the safety regex                                                                        | I-2           |
| [`fuzz_detect_backend.py`](harnesses/fuzz_detect_backend.py)             | `InferenceModel._detect_backend()`                                           | Export directories populated with arbitrary filename extensions and combinations                                                              | I-6           |
| [`fuzz_prepare_inputs.py`](harnesses/fuzz_prepare_inputs.py)             | `InferenceModel._prepare_inputs(inputs)`                                     | Flat dicts, nested dicts, mixed flat+nested collisions, arbitrary key names                                                                   | I-7           |
| [`fuzz_stats_normalizer.py`](harnesses/fuzz_stats_normalizer.py)         | `StatsNormalizer.__call__(inputs)`                                           | Arbitrary shaped float32 arrays; extreme stat values (inf, nan, zero std, inverted quantiles); all four modes                                 | I-8, I-9      |
| [`fuzz_resize_preprocessor.py`](harnesses/fuzz_resize_preprocessor.py)   | `ResizePreprocessor.__call__(inputs)`                                        | Channels-first and channels-last images; uint8 and float32; zero spatial dims; extreme target resolutions; stretch and letterbox modes        | I-10, I-11    |
| [`fuzz_resize_smolvla.py`](harnesses/fuzz_resize_smolvla.py)             | `ResizeSmolVLA.__call__(inputs)`                                             | Same image layouts as above; arbitrary target resolution                                                                                      | I-10, I-12    |
| [`fuzz_action_normalizer.py`](harnesses/fuzz_action_normalizer.py)       | `ActionNormalizer.__call__(outputs)`                                         | Output dicts with arbitrary key names and array shapes; missing `action` key                                                                  | I-13          |
| [`fuzz_action_chunk_trimmer.py`](harnesses/fuzz_action_chunk_trimmer.py) | `ActionChunkTrimmer.__call__(outputs)`                                       | Action arrays with arbitrary first and second dimensions; extreme `n_action_steps`                                                            | I-14          |
| [`fuzz_lerp_smoother.py`](harnesses/fuzz_lerp_smoother.py)               | `LerpSmoother.merge(remaining, incoming)`                                    | 2D float arrays with arbitrary row counts, zero rows, NaN/Inf values, mismatched dims                                                         | I-15, I-16    |
| [`fuzz_action_queue.py`](harnesses/fuzz_action_queue.py)                 | `ChunkedActionQueue.push_chunk(chunk, offset)` + `pop()`                     | Arbitrary chunk shapes, extreme offsets, concurrent push/pop sequences                                                                        | I-17          |

## Key invariants for fuzzing oracles

These are the security/safety and correctness properties each harness asserts. A violation is a bug.

**I-1 — No path traversal via artifact**  
`resolve_artifact()` must never return a path outside `export_dir`. Any crafted `artifact` value (including `../`, absolute paths, and null bytes) must either raise `ValueError` or produce a path that is a sub-path of the resolved `export_dir`.

**I-2 — Policy name safety**  
`_is_safe_policy_name(name)` and `InferenceModel` must agree: if the predicate returns `True`, the model constructor must not raise `ValueError` for the name. Names that pass the regex (`^[a-zA-Z0-9][a-zA-Z0-9_.\-]*$`) are safe and must be accepted.

**I-3 — No crash on malformed JSON**  
`Manifest.load()` must not raise an unhandled exception for any JSON input. Only `ValidationError` (Pydantic) and `FileNotFoundError` are acceptable.

**I-4 — Class resolution requires a type**  
`instantiate_component` on a `class_path` that resolves to a non-type (function, module, instance) must raise `TypeError` and must not call or execute the resolved object.

**I-5 — Depth limit enforced**  
`instantiate_component` on a manifest with `init_args` nested deeper than `_MAX_COMPONENT_DEPTH = 10` must raise `ValueError`. It must not recurse indefinitely.

**I-6 — No unexpected exception from import or detection**  
`import_dotted_path()` must raise only `ValueError` for paths that cannot be imported — never `TypeError`, `AttributeError`, or other undocumented exceptions. `_detect_backend()` must raise `ValueError` (not crash) when no model files are found.

**I-7 — Key collision in `_prepare_inputs` raises `ValueError`**  
When both a flat key `"prefix.suffix"` and a nested dict `{"prefix": {"suffix": v}}` are present, `_prepare_inputs` must raise `ValueError`. Silently picking a winner based on insertion order would allow a caller to substitute an arbitrary tensor into the model without any observable error.

**I-8 — Non-listed keys pass through unchanged**  
`StatsDenormalizer` and `StatsNormalizer` must not modify or drop keys that are not in the `features` list. The value at any non-listed key must be byte-for-byte identical in the output.

**I-9 — NaN/Inf stats propagate to output**  
When stats contain NaN or Inf and the input tensor is finite and non-empty, the output must also be non-finite. The denormalizer must not silently clamp or wrap around to produce a plausible-looking finite value.

**I-10 — Preprocessor output is float32 channels-first**  
`ResizePreprocessor` and `ResizeSmolVLA` must produce `float32` output in `(B, C, H, W)` layout when the output is non-empty, regardless of input dtype or channel layout.

**I-11 — No unhandled exception for zero-spatial images**
`ResizePreprocessor` must not raise `ZeroDivisionError` or any unhandled exception when the input image has zero height or width. It must either produce an empty output or raise `ValueError`.

**I-12 — SmolVLA pixel values in [-1, 1]**
All pixel values in the `IMAGES` output of `ResizeSmolVLA` must satisfy `-1.0 - ε ≤ value ≤ 1.0 + ε` (with `ε = 1e-5`). Values outside this range produce out-of-distribution model inputs that can drive the robot with incorrectly scaled actions.

**I-13 — ActionNormalizer always emits `"action"` key**
The output dict of `ActionNormalizer.__call__()` must contain the key `"action"` for any input, even if the input dict does not contain it. Other keys must pass through unchanged.

**I-14 — ActionChunkTrimmer reduces chunk to at most `n_action_steps`**
The second dimension of the output action array must be `≤ n_action_steps`. The trimmer must not increase the action horizon.

**I-15 — LerpSmoother output is float32**
`LerpSmoother.merge()` output dtype must always be `np.float32`, regardless of input dtypes.

**I-16 — LerpSmoother does not introduce NaN**
If neither `remaining` nor `incoming` contains NaN or Inf, the merged output must also not contain NaN or Inf.

**I-17 — ChunkedActionQueue is thread-safe**
Concurrent `push_chunk` and `pop` calls must not raise exceptions, corrupt the deque, or return arrays of unexpected shape. `pop()` returns either `None` or a 1-D array; it must never raise.

## Shared Test Utilities

[`harnesses/_helpers.py`](harnesses/_helpers.py) provides fuzz-data-driven constructors used across multiple harnesses:

| Function                                  | Returns                                                                            |
| ----------------------------------------- | ---------------------------------------------------------------------------------- |
| `make_float_array(fdp, ...)`              | `np.ndarray` of arbitrary shape and float32 values (including NaN/Inf)             |
| `make_image_array(fdp, ...)`              | Plausible image array — channels-first or channels-last, uint8 or float32          |
| `make_2d_float_array(fdp, ...)`           | 2-D float32 array with fuzz-derived row and column counts                          |
| `make_2d_same_cols(fdp, cols, ...)`       | 2-D float32 array with a fixed column count (for action-queue tests)               |
| `make_stats_dict(fdp, feature, mode=...)` | Pre-built stats dict (mean/std, min/max, q01/q99) with fuzz-derived float32 values |

All helpers pad short byte sequences with zeros so the harness never throws `IndexError` on exhausted fuzz data.

## Corpus and seeds

- `tests/fuzz/corpus/<harness>/` - working corpus; populated by libFuzzer during fuzzing runs.
- `tests/fuzz/seeds/fuzz_manifest/` - hand-curated valid `manifest.json` examples used to bootstrap the manifest harness.
