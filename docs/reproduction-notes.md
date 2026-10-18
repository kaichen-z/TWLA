# Reproduction Notes

These notes describe the interfaces and evaluation conventions in the current checkout. Start with the [main README](../README.md) for installation, weights, and training commands. They are not a re-evaluation of the paper's reported results.

## Checkpoint compatibility

- **Visual policy:** `T-WLA-FD.pt` produces visual memory and base actions. It is not a complete visuotactile-controller checkpoint.
- **RGB world head:** `T-WLA-WM.pt` provides `world.*` weights. `--init-world-from` initializes this head only; `--init-from` initializes model weights for a fresh run, while `--resume` restores an existing run. A checkpoint and its world-model configuration must match.
- **Formal tactile encoder:** the Hugging Face [MANO-aligned H8 release](https://huggingface.co/MIT-Media-Lab/egotouch-tactile-encoder-p1-v1/tree/main/mano-aligned-v1/h8) includes `best.pt` and its resolved configuration. The [formal loader](../tactile_editor/src/tactile_editor/encoder.py) expects a P1 config with eight tactile frames, a `model` state containing the complete `tactile.*` tower, and `pressure_masks`. Pass a downloaded local `.pt` path to `--encoder-checkpoint`, not an HF ID or URL. The published metadata matches the intended format; successful loading and identity with the paper's final encoder still require verification.
- **Standalone encoder:** the same HF repository also contains `seed-*/tactile_encoder.safetensors` and related exports. These use a different loader and cannot be passed directly to the formal-P1 argument.
- **Editor:** an encoder checkpoint initializes tactile features; it does not contain a trained W3 residual editor. No final W3 download is currently linked in this checkout.

The supplied visual configuration trains the controller and RGB head together. A separate RGB-head warm-up is supported with `config.world.freeze_v4=true` and `config.world.freeze_head=false`, using a matching initialized controller. This is distinct from joint visual–tactile fine-tuning, which the offline editor runners do not implement.

## Data and cache preparation

The visual JSON resolves data/output variables through the visual loader. Tactile configuration and command-line paths resolve from the working directory; absolute paths are recommended.

The encoder expects format-v4 data with `manifest.json`, `pressure_masks.npy`, and per-split `touch.npy`, `pose.npy`, `windows.npy`, and evaluation `eval_indices.npy`. Its supplied pretraining recipe starts from an external OpenTouch initialization checkpoint, not the trained H8 checkpoint used for editor initialization.

EgoTouch editor caches contain common targets/tactile streams and model-specific frozen base predictions. T-WLA additionally uses final visual memory and, when RGB supervision is enabled, image tokens, future clean latents, validity masks, and ROI weights. See the [cache and time contract](../tactile_editor/README.md#cache-and-time-contract).

Native Tachin requires `statistics.npz` and `audit/repartition.json` with `status: "PASS"` and a true `recording_disjoint` flag. It uses its own 880-cell sensor layout and encoder rather than converting data into an EgoTouch grid.

Statistics and fixed tactile references must come from the training split. Verify checkpoint/data provenance as well as cache-level test splits. Changing the visual backbone or memory configuration requires regenerating its cached outputs, not renaming existing files. Raw-recording preprocessing and cache-generation scripts are not bundled.

**Future-image supervision:** the visual loader's raw-video fallback reuses the current image as a future target. Supply valid `jpeg_root` frame stores when training future-image prediction.

## Tensor and execution contract

| Quantity | Supplied configuration |
| :--- | :--- |
| Visual state and validity | 212 dimensions each |
| Base action chunk | 16 × 192 |
| Valid hand-motion dimensions | 0:69 left hand; 69:138 right hand; 138:192 padding |
| Per-hand action | 3 wrist-translation increments, 3 Euler-rotation increments, 63 local joint coordinates |
| Visual memory | 8 × 960 |
| EgoTouch tactile history | `[B, 8, 2, 21, 21]` |
| Tachin tactile history | `[B, 8, 2, 880]` |

`VLAPolicy.act(..., steps=16, return_mf=True)` returns raw physical-space actions as a NumPy array and visual memory as a Torch tensor on the policy device. Omitting `steps=16` returns the configured execution prefix, normally eight actions. The [public adapter](../tactile_editor/src/tactile_editor/public_backbone.py) normalizes the full action chunk with explicitly supplied editor-training statistics and validity masks. Padding must remain invalid.

Only the tactile encoder and residual editor run between visual updates. Each correction is relative to the original base chunk, not previously corrected actions. Visual history updates once per RGB observation. Both predictive heads are training-only.

## Evaluation protocol

The paper's metrics are wrist translation error (cm), wrist rotation geodesic error (degrees), and wrist-relative MPJPE over the 20 non-root joints (cm). Use a declared common prediction horizon, compute per-sequence means, then aggregate. EgoTouch gives equal weight to its five environment means; Tachin averages sequences over its held-out split.

The [physical evaluation utilities](../tactile_editor/src/tactile_editor/table1_experiment.py) implement strict per-hand 8/16-step horizons and sequence/environment aggregation. The native Tachin runner uses them. The default EgoTouch runner and visual evaluator use separate reductions:

- [`train_paired30.py`](../tactile_editor/scripts/train_paired30.py) pools eligible hand–time errors and accepts individually valid steps from incomplete prefixes. Its default summaries are not the paper's sequence/environment-balanced metrics.
- [`eval_test.py`](../T-WLA_foundation_model/eval_test.py) produces configured visual-policy diagnostics, not automatically the paper's sequence-balanced evaluation.

Use matching sequence/environment metadata and validity masks with the strict utilities for paper-compatible aggregation. Visual-only comparisons additionally require that tactile observations and supervision were excluded from the evaluated pathway's training and evaluation.

Keep EgoTouch's `--validation-only` flag during model selection. Native Tachin has no equivalent flag and normally evaluates its test split after validation checkpoint selection. Match splits, queries, targets, horizons, sensor conventions, and auxiliary weights across comparisons. Example coefficients are not a manifest of the paper's final runs.

## Native Tachin timing

The native runner supports legacy single-decision and extended paired30 caches. In the current extended path, [`train_tachin.py`](../tactile_editor/scripts/train_tachin.py) advances action masks using elapsed decision time, but [`NativeTachinW3Editor`](../tactile_editor/src/tactile_editor/native_tachin.py) supplies zero elapsed time to action conditioning and always selects the first eight actions for pressure prediction.

At later tactile decisions, the mask advances but these conditions do not. Resolve this discrepancy before treating the extended path as the paper's fully time-conditioned multi-update implementation. It does not affect the legacy zero-time, single-decision setting.

## Deployment and throughput

The offline cache protocol uses 10 Hz plans, 30 Hz tactile decisions, and 80 Hz action positions, assuming zero compute latency during replay. These are protocol settings, not measured hardware throughput. The release does not include a benchmark or complete asynchronous robot scheduler reproducing the paper's reported rates.

Hardware tools cover calibration, teleoperation, recording, and device interfaces. End-to-end robot adaptation, action conversion, scheduling, and safety integration remain platform-specific. These tools do not provide collision checking; validate calibration, limits, and emergency-stop procedures before hardware use.

## Dependencies and attribution

- **Visual branch:** SmolVLM2 configuration/tokenizer for inference and pretrained assets for training; SANA and DC-AE for future-image supervision.
- **Tactile encoder:** EgoTouch data and OpenTouch initialization for the supplied formal pretraining recipe. The linked H8 weights are a separately downloadable pretrained option.
- **VITRA baseline:** gated PaliGemma 2 access, MANO assets, and `smplx`; follow the Gemma Terms of Use.
- **MemoryVLA baseline:** compatible CogACT/Llama assets and its separate dependency environment. Upstream provenance is recorded in [UPSTREAM.md](../T-WLA_foundation_model/baselines/memoryvla_official/upstream/UPSTREAM.md).
- **SmolVLA utilities:** adapted from LeRobot under Apache-2.0.
- **Robot platform:** A7 dual arms, L6 dexterous hands, RealSense D435i, and supported glove/tracker interfaces; see the [hardware guide](../robot_hardware/README.md).

The requirements are not a locked environment, and baseline/hardware integrations may need separate environments. This checkout has no repository-wide license file; third-party notices do not grant a blanket license for the full project.
