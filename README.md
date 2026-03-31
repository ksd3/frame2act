<div align="center">
  <h2>driving-idm</h2>
</div>

We write and maintain driving-idm, an inverse dynamics model for autonomous driving

It's dead simple. Give it frame `t` and frame `t+1`, it tells you what action happened between them. Steering and acceleration. That's the whole output.

The model breaks down into 4 modules:

**models** — the neural networks. ResBlock backbone, two flavors.
`IDM` stacks frames into a 6-channel input. `IDMSiamese` encodes each frame with shared weights and diffs the features. ~5M params, 90x160 input, trained on an H100 in minutes.

**dataset** — loads `.npz` driving sequences and serves frame pairs.
Each sample: `(frame_t || frame_t+1)` -> `(steering, acceleration)`. Normalizes actions, resizes frames, nothing else.

**metrics** — one function. `compute_metrics` gives you MSE and MAE, normalized and unnormalized, split by steering vs acceleration.

**video_models** — interface for chaining video prediction with the IDM.
`LastFrameBaseline` predicts nothing moves. `TorchScriptVideoModel` loads any exported model. Subclass `CheckpointVideoModel` for your own architecture.

### use it

```python
from idm import IDM, IDMSiamese, IDMDataset, compute_metrics, preprocess
```

train:
```bash
uv run --with modal modal run --detach train.py [--run-name my-run]
```

rollout (video model -> IDM -> action):
```bash
uv run --with modal modal run --detach rollout.py \
    --idm-ckpt /path/to/idm_best.pt \
    [--video-model-ckpt /path/to/video_model.pt]
```

skip `--video-model-ckpt` to use LastFrameBaseline. three baselines logged to wandb:
- `vm_idm/*` — video model -> IDM vs ground truth. the one you care about
- `oracle_idm/*` — real next frame -> IDM. upper bound
- `baseline/*` — no motion -> IDM. lower bound

### datasets

format: `.npz` on huggingface. `frames` `(T, H, W, 3)` uint8, `actions` `(T, 2)` float32 `(steering, acceleration)`.

**works out of the box** — swap `REPO_ID` and train:

| dataset | what |
|---------|------|
| [`nebusoku14/comm_hack_parking_npz`](https://huggingface.co/datasets/nebusoku14/comm_hack_parking_npz) | comma parking sequences (default) |
| [`nebusoku14/comm_hack_parking_day`](https://huggingface.co/datasets/nebusoku14/comm_hack_parking_day) | daytime variant (`--curvature-comma`) |
| [`nebusoku14/tesla_parking_npz`](https://huggingface.co/datasets/nebusoku14/tesla_parking_npz) | tesla parking |
| [`nebusoku14/tesla_leadup_npz`](https://huggingface.co/datasets/nebusoku14/tesla_leadup_npz) | tesla lead-up sequences |
| [`nebusoku14/comma_hacks_tesla_parking`](https://huggingface.co/datasets/nebusoku14/comma_hacks_tesla_parking) | tesla parking (comma hacks) |
| [`nebusoku14/comma_hacks_tesla_parking_stage2`](https://huggingface.co/datasets/nebusoku14/comma_hacks_tesla_parking_stage2) | stage 2 |
| [`nebusoku14/stage2v2`](https://huggingface.co/datasets/nebusoku14/stage2v2) | stage 2 v2 |

**needs a conversion script**:

| dataset | deal |
|---------|------|
| [`commaai/commavq`](https://huggingface.co/datasets/commaai/commavq) | 100k segments. VQ-VAE compressed, need to decode tokens to frames and map pose to actions |
| [`commaai/comma2k19`](https://huggingface.co/datasets/commaai/comma2k19) | 33 hrs highway. HEVC video + CAN bus, decode and extract |
| [`immanuelpeter/carla-autopilot-multimodal-dataset`](https://huggingface.co/datasets/immanuelpeter/carla-autopilot-multimodal-dataset) | 76k simulated frames in parquet. easiest conversion, has everything |
| [`yaak-ai/L2D`](https://huggingface.co/datasets/yaak-ai/L2D) | 5000+ hrs real-world, 90+ TB. sample a subset unless you have serious storage |

### setup

two modal secrets:
- `huggingface-secret` — `HUGGING_FACE_HUB_TOKEN`
- `wandb-secret` — `WANDB_API_KEY`

```bash
modal app logs idm-training
```
