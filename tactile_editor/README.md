# Tactile editor

This directory contains code, tests, and an encoder config. It contains no results, logs, caches, datasets, or weights.
The foundation model and robot code stay unchanged.

## Model

```text
Frozen T-WLA -> base action [B,16,192] + memory [B,8,960]
8 tactile frames -> shared hand encoder -> two tactile tokens
Editor(base action, memory, tactile tokens) -> delta action
Final action = base action + masked delta action
```

EgoTouch uses `[B,8,2,21,21]` pressure histories. Tachin uses native `[B,8,2,880]` histories and sensor masks.
The Tachin encoder does not reshape its sensors into an EgoTouch grid.

W3 is the released method: action supervision plus RGB and tactile prediction losses.
RGB supervision uses a frozen world head and a trainable tactile condition modulator.
The tactile world head is trainable. Neither world head runs during action inference.
Standard baseline editors omit T-WLA memory. Their tactile trunks train with the editor rather than load the retrieval checkpoint.

## Install

From the repository root, install this package:

```bash
python -m pip install -e './tactile_editor[test]'
python -m pytest tactile_editor/tests
```

For RGB supervision or foundation inference, also install the foundation dependencies in a compatible environment:

```bash
python -m pip install -r T-WLA_foundation_model/requirements.txt
```

The offline editor does not need the foundation model in memory. It reads frozen predictions and features from a cache.
Training commands require CUDA. Unit tests run on CPU without downloads.

## Train the EgoTouch retrieval encoder

Edit the paths in `configs/encoder_p1_h8.json`. Then run:

```bash
python -m egotouch_formal.train --config tactile_editor/configs/encoder_p1_h8.json
```

This P1 config trains touch-to-pose retrieval with a GRU pose tower and OpenTouch initialization.
Its prepared cache must use format version 4 and match the recorded coordinate convention.
The loader requires `manifest.json`, `pressure_masks.npy`, and per-split `touch.npy`, `pose.npy`, `windows.npy`, and `eval_indices.npy`.
The cache defines the split. The trainer does not create a new split.

`encoder.py` loads the tactile tower from the resulting P1 checkpoint.
It also supports the standalone weights at [MIT-Media-Lab/egotouch-tactile-encoder-p1-v1](https://huggingface.co/MIT-Media-Lab/egotouch-tactile-encoder-p1-v1).
These formats use different loaders. A retrieval checkpoint cannot replace the foundation checkpoint.

## Train EgoTouch editors

Use an audited paired30 cache, the matching P1 encoder checkpoint, and the matching RGB world checkpoint and config:

```bash
python tactile_editor/scripts/train_paired30.py \
  --root /path/to/paired30_cache --variant w3 \
  --encoder-checkpoint /path/to/encoder_h8/best.pt \
  --world-checkpoint /path/to/world.pt --world-config /path/to/config_used.json \
  --lambda-rgb 0.1 --lambda-tactile 0.0001 \
  --epochs 5 --output-dir /path/to/runs/w3
```

The coefficients are example defaults, not a claim of optimal performance.
`w3` is the default. There are no separate W0, W1, or W2 command-line entries.
For loss ablations, keep `--variant w3` and set either world weight, or both weights, to zero.
The model structure and checkpoint keys stay unchanged. Disabled branches do not run or receive gradients.
If `--lambda-rgb 0`, the trainer does not load the RGB cache fields or the RGB world checkpoint.
The low-level legacy mode enum remains only for earlier checkpoints and regression tests.
For standard editors, set `--variant` to `smolvla`, `vitra`, `trex`, or `memoryvla`.
Those variants do not require an encoder checkpoint or a world checkpoint.

Use `--smoke-steps 2` for a short data-dependent test. Use `--validation-only` during coefficient selection to avoid test evaluation.
The trainer selects the checkpoint with the lowest validation action loss.

## Train native Tachin editors

```bash
python tactile_editor/scripts/train_tachin.py \
  --cache /path/to/tachin_cache --model smolvla \
  --output /path/to/runs/tachin_smolvla

python tactile_editor/scripts/train_tachin.py \
  --cache /path/to/tachin_cache --model twla \
  --world-checkpoint /path/to/world.pt --world-config /path/to/config_used.json \
  --lambda-rgb 0.1 --lambda-tactile 0.0001 \
  --output /path/to/runs/tachin_w3
```

The native runner supports four baseline names and T-WLA W3. It requires `statistics.npz` and an audited recording-disjoint cache.
The same zero-weight ablations apply to Tachin. With zero RGB weight, the world checkpoint and config are not required.
The audit file is `audit/repartition.json`. The loader accepts the legacy single-decision format and the extended three-decision format.
Equal coefficients do not make different cache protocols comparable. Each comparison needs the same split, queries, targets, and evaluation horizon.

## Cache and time contract

The EgoTouch cache layout is:

```text
statistics.pt
shared/{train,val,test}/
  complete.json, queries.npz
  target_action.npy, action_mask.npy
  touch_sequence.npy, touch_sequence_valid.npy
  future_touch_pressure.npy, future_touch_valid.npy
  eligible.npy, source_frame.npy
bases/{smolvla,vitra,trex,memoryvla}/{train,val,test}/
  base_action.npy, complete.json
bases/twla/{train,val,test}/shard_*/
  base_action.npy, final_memory.npy, query_index.npy, complete.json
  image_tokens.npy, clean_latent.npy, rgb_valid.npy, future_roi.npy  # RGB training only
```

`Paired30Cache` checks shapes, query coverage, and duplicate T-WLA indices. Completion markers do not establish upstream train/test isolation.
The statistics must come from the training split. A new backbone requires new predictions, not renamed cache files.

The offline protocol uses 10 Hz plans, 80 Hz actions, and three tactile decisions at 0, 33.3, and 66.7 ms.
Each plan contains 16 actions. Evaluation executes the first eight actions, with later tactile decisions replacing only uncommitted positions.
Each decision reads eight history frames ending at that decision. Each revision uses the immutable base plan, not the previous revision.
This code does not measure hardware latency or implement a robot scheduler.

Raw dataset preprocessing and legacy cluster launchers are not part of this code-only integration.
The cache loaders preserve the existing formats. They do not silently rebuild or reclassify data.

## Connect the public foundation model

`public_backbone.policy_context` calls the existing policy with `steps=16, return_mf=True`.
It converts raw actions into the editor's normalized space with explicit training statistics.
The caller supplies action validity. Padding dimensions 138:192 remain invalid.

Call the foundation policy once per visual observation. Reuse that context for the tactile decisions within the plan.
Calling it for every tactile update changes the memory update rate.

The RGB bridge loads the sibling `world_model` implementation. It adds two adapter methods without changing foundation files.
The checkpoint and config must match exactly. The loader rejects missing or incompatible weights.
If the package is outside this checkout, set `TWLA_FOUNDATION_ROOT` to the foundation directory.

## Code origin

The encoder and EgoTouch editor code comes from `hannah/tactile_editor` and the local training implementation.
The native Tachin encoder and tests come from the collaborator's native-sensor implementation used by the Tachin runner.
This integration removes machine-specific paths and adds the public foundation adapter.
It preserves checkpoint parameter names. The tactile world head accepts either 441 or 880 cells.

This code-only release does not claim a new full training run or benchmark result.
