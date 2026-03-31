<div align="center">
  <h2>frame2act</h2>
</div>

### Motivation

comma.ai builds its self-driving system around a world model: given the current state of the road, the model predicts what the next frame will look like, and from that predicted transition, infers the appropriate driving action. This is a compelling decomposition of the problem, and it was demonstrated to great effect by the winning team at comma hack 6. Their solutions really impressed me, and I wanted to see if I could implement what they'd done — and perhaps push it a little further. This repository is the result of that effort.

The idea of learning an inverse dynamics model from observation is, of course, not new. The approach here draws on a line of work in both traffic modeling and representation learning:

- Treiber, Hennecke & Helbing, [*Congested Traffic States in Empirical Observations and Microscopic Simulations*](https://arxiv.org/abs/cond-mat/0002177) (2000) — the original Intelligent Driver Model, which introduced a clean parametric formulation for car-following dynamics.
- Kesting, Treiber & Helbing, [*General Lane-Changing Model MOBIL for Car-Following Models*](https://journals.sagepub.com/doi/10.3141/1999-10) (2007) — the MOBIL lane-change model, extending IDM to multi-lane settings.
- Treiber, Kesting & Helbing, [*Enhanced Intelligent Driver Model to Access the Impact of Driving Strategies on Traffic Capacity*](https://arxiv.org/abs/0912.3613) (2010) — fixes to IDM for adaptive cruise control and cut-in scenarios.
- Mo, Shi & Di, [*A Physics-Informed Deep Learning Paradigm for Car-Following Models*](https://arxiv.org/abs/2012.13376) (2021) — PIDL-CF, which encodes IDM structure into neural network loss functions; the hybrid physics+learning approach most relevant to what we do here.
- [*Limitations and Improvements of the Intelligent Driver Model*](https://epubs.siam.org/doi/10.1137/21M1406477) (2022) — mathematical well-posedness analysis and fixes for numerical stability.
- [*Twenty-Five Years of the Intelligent Driver Model*](https://arxiv.org/abs/2506.05909) (2025) — a comprehensive survey covering the full ecosystem of IDM variants, extensions, and applications.

The idea is to cleanly factor the pipeline into two stages. First, a video prediction model forecasts the next frame given a context window of past observations. Then, an inverse dynamics model (IDM) takes the current frame and the predicted next frame, and outputs the action — a two-dimensional vector of steering and acceleration — that would produce that transition. It turns out that training the IDM itself is relatively straightforward (a few minutes on an H100), and the quality of the overall system depends primarily on how well the video model can anticipate the future.

### Architecture

The library is organized into four modules, each of which is fairly self-contained.

**`idm.models`** contains the neural network architectures. There are two variants. `IDM` is the simpler one: it concatenates the two frames into a 6-channel tensor and passes them through a ResBlock encoder (~5M parameters, operating on 90x160 inputs). `IDMSiamese` is a more structured alternative, in which each frame is encoded separately using shared weights, and the prediction head receives the concatenation `[z0, z1, z1 - z0]`, making the feature-level difference signal explicit.

**`idm.dataset`** handles data loading. Each sample is a pair of consecutive frames together with the corresponding action vector. The dataset normalizes actions by their empirical mean and standard deviation, and resizes frames to 90x160 via bilinear interpolation.

**`idm.metrics`** provides a single function, `compute_metrics`, which returns MSE and MAE in both normalized and unnormalized coordinates, broken down by steering and acceleration.

**`idm.video_models`** defines the interface for plugging in a video prediction model. `LastFrameBaseline` is the trivial baseline (predicting that nothing changes). `TorchScriptVideoModel` wraps any exported TorchScript model. For custom architectures, one can subclass `CheckpointVideoModel` and implement `build_model()`.

### Usage

```python
from idm import IDM, IDMSiamese, IDMDataset, compute_metrics, preprocess
```

To train:
```bash
uv run --with modal modal run --detach train.py [--run-name my-run]
```

To evaluate the full pipeline (video model followed by IDM):
```bash
uv run --with modal modal run --detach rollout.py \
    --idm-ckpt /path/to/idm_best.pt \
    [--video-model-ckpt /path/to/video_model.pt]
```

If `--video-model-ckpt` is omitted, the rollout defaults to `LastFrameBaseline`. Three families of metrics are logged to wandb:
- `vm_idm/*` — the main pipeline (video model prediction fed to IDM) compared against ground truth
- `oracle_idm/*` — IDM given the real next frame, which serves as an upper bound on performance
- `baseline/*` — IDM given no motion (i.e., the current frame repeated), which serves as a lower bound

### Datasets

The expected format is `.npz` files hosted on HuggingFace, each containing `frames` of shape `(T, H, W, 3)` as uint8 and `actions` of shape `(T, 2)` as float32 (steering, acceleration).

The following datasets are compatible out of the box — one simply needs to change the `REPO_ID` variable:

| Dataset | Description |
|---------|-------------|
| [`nebusoku14/comm_hack_parking_npz`](https://huggingface.co/datasets/nebusoku14/comm_hack_parking_npz) | Comma parking sequences (default) |
| [`nebusoku14/comm_hack_parking_day`](https://huggingface.co/datasets/nebusoku14/comm_hack_parking_day) | Daytime variant (`--curvature-comma`) |
| [`nebusoku14/tesla_parking_npz`](https://huggingface.co/datasets/nebusoku14/tesla_parking_npz) | Tesla parking sequences |
| [`nebusoku14/tesla_leadup_npz`](https://huggingface.co/datasets/nebusoku14/tesla_leadup_npz) | Tesla lead-up sequences |
| [`nebusoku14/comma_hacks_tesla_parking`](https://huggingface.co/datasets/nebusoku14/comma_hacks_tesla_parking) | Tesla parking (comma hacks) |
| [`nebusoku14/comma_hacks_tesla_parking_stage2`](https://huggingface.co/datasets/nebusoku14/comma_hacks_tesla_parking_stage2) | Stage 2 processing |
| [`nebusoku14/stage2v2`](https://huggingface.co/datasets/nebusoku14/stage2v2) | Stage 2 v2 |

In principle, one could also convert any of the following datasets, though a conversion script would be needed:

| Dataset | Notes |
|---------|-------|
| [`commaai/commavq`](https://huggingface.co/datasets/commaai/commavq) | 100k one-minute segments. Frames are VQ-VAE compressed and would need to be decoded; the 6-DOF pose would need to be mapped to (steering, acceleration). |
| [`commaai/comma2k19`](https://huggingface.co/datasets/commaai/comma2k19) | 33 hours of highway driving. Video is HEVC-encoded; steering angle and speed are available from the CAN bus. |
| [`immanuelpeter/carla-autopilot-multimodal-dataset`](https://huggingface.co/datasets/immanuelpeter/carla-autopilot-multimodal-dataset) | 76k simulated frames in Parquet format. This is probably the easiest to convert, as it already contains front camera images, steering, throttle, and brake. |
| [`yaak-ai/L2D`](https://huggingface.co/datasets/yaak-ai/L2D) | Over 5000 hours of real-world driving data at 90+ TB. One would likely want to sample a manageable subset. |

### Setup

Training and evaluation run on [Modal](https://modal.com) and require two secrets:
- `huggingface-secret` containing `HUGGING_FACE_HUB_TOKEN`
- `wandb-secret` containing `WANDB_API_KEY`

To monitor a running job:
```bash
modal app logs idm-training
```
