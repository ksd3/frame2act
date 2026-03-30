"""
rollout.py -- video model + IDM inference pipeline

Video model:   past frames (B, T, 3, H, W) -> predicted frame_t+1 (B, 3, H, W)
IDM:           (frame_t || predicted_frame_t+1) -> action (B, 2)

Usage:
  uv run --with modal modal run --detach rollout.py \
      --idm-ckpt /path/to/idm_best.pt \
      --video-model-ckpt /path/to/video_model.pt \
      [--run-name my-rollout] \
      [--context-frames 8]
"""

import modal

app = modal.App("idm-rollout")

volume = modal.Volume.from_name("idm-dataset-cache", create_if_missing=True)
VOLUME_PATH = "/data"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "torchvision",
        "numpy",
        "wandb",
        "huggingface_hub",
        "tqdm",
    )
    .add_local_python_source("idm")
)


@app.function(
    image=image,
    gpu="H100",
    timeout=3600 * 2,
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("wandb-secret"),
    ],
    memory=32768,
    volumes={VOLUME_PATH: volume},
)
def rollout(
    idm_ckpt: str,
    video_model_ckpt: str = "",
    run_name: str | None = None,
    context_frames: int = 8,
    n_eval_sequences: int = 200,
):
    import os
    import numpy as np
    import torch
    import torch.nn.functional as F
    from huggingface_hub import hf_hub_download, list_repo_files
    import wandb
    from tqdm import tqdm

    from idm import IDM, TorchScriptVideoModel, LastFrameBaseline, preprocess, IMG_H, IMG_W

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    run = wandb.init(
        project="idm-parking",
        name=run_name,
        job_type="rollout",
        config=dict(
            idm_ckpt=idm_ckpt,
            video_model_ckpt=video_model_ckpt or "LastFrameBaseline",
            context_frames=context_frames,
            n_eval_sequences=n_eval_sequences,
            img_h=IMG_H, img_w=IMG_W,
        ),
    )
    print(f"wandb run: {run.url}")

    # ── Load IDM ──────────────────────────────────────────────────────────────
    idm_model = IDM(action_dim=2).to(DEVICE)
    ckpt = torch.load(idm_ckpt, map_location=DEVICE)
    state = ckpt.get("model_state_dict", ckpt)
    # strip torch.compile prefix if present
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    idm_model.load_state_dict(state)
    idm_model.eval()

    act_mean = torch.tensor(ckpt["act_mean"], dtype=torch.float32, device=DEVICE)
    act_std = torch.tensor(ckpt["act_std"], dtype=torch.float32, device=DEVICE)
    print(f"IDM loaded from {idm_ckpt}")

    # ── Load video model ──────────────────────────────────────────────────────
    if video_model_ckpt:
        try:
            vm = TorchScriptVideoModel(device=DEVICE)
            vm.load_weights(video_model_ckpt)
            print(f"Loaded TorchScript video model from {video_model_ckpt}")
        except Exception:
            raise ValueError(
                f"Could not load video model from {video_model_ckpt}. "
                "For a custom architecture, subclass CheckpointVideoModel and call rollout() directly."
            )
    else:
        vm = LastFrameBaseline()
        print("No video model provided — using LastFrameBaseline (next frame = current frame)")

    # ── Load dataset (from volume cache) ─────────────────────────────────────
    npz_dir = os.path.join(VOLUME_PATH, "npz")
    REPO_ID = "nebusoku14/comm_hack_parking_npz"
    os.makedirs(npz_dir, exist_ok=True)

    all_files = sorted(
        f for f in list_repo_files(REPO_ID, repo_type="dataset")
        if f.endswith(".npz")
    )
    existing = set(os.listdir(npz_dir))
    to_download = [f for f in all_files if os.path.basename(f) not in existing]
    if to_download:
        print(f"Downloading {len(to_download)} missing files...")
        for fname in tqdm(to_download, desc="download"):
            hf_hub_download(repo_id=REPO_ID, filename=fname,
                            repo_type="dataset", local_dir=npz_dir)
        volume.commit()

    local_paths = [os.path.join(npz_dir, os.path.basename(f)) for f in all_files]

    # ── Eval loop ─────────────────────────────────────────────────────────────
    mae_idm_gt = []    # IDM(real_t, real_t+1)   vs gt  — upper bound
    mae_vm_idm = []    # IDM(real_t, pred_t+1)   vs gt  — main metric
    mae_baseline = []  # IDM(real_t, real_t)      vs gt  — trivial baseline

    rng = np.random.default_rng(42)
    paths_eval = rng.choice(local_paths, size=min(n_eval_sequences, len(local_paths)), replace=False)

    for p in tqdm(paths_eval, desc="eval sequences"):
        try:
            data = np.load(p)
        except Exception:
            continue
        frames = data["frames"]    # (T, H, W, 3)
        actions = data["actions"]  # (T, 2)
        T = frames.shape[0]
        if T < context_frames + 1:
            continue

        for t in range(context_frames, T - 1):
            gt_action = torch.tensor(actions[t], dtype=torch.float32, device=DEVICE)

            f_t = preprocess(frames[t]).to(DEVICE)
            f_t1 = preprocess(frames[t + 1]).to(DEVICE)

            # Context window for video model: (1, context_frames, 3, H, W)
            ctx = torch.cat(
                [preprocess(frames[t - context_frames + i]) for i in range(context_frames)],
                dim=0
            ).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                # 1. Video model prediction
                f_pred = vm.predict_next_frame(ctx)

                # 2. IDM on (real_t, predicted_t+1) — main pipeline
                x_vm = torch.cat([f_t, f_pred], dim=1)
                pred_vm = idm_model(x_vm)[0] * act_std + act_mean

                # 3. IDM on (real_t, real_t+1) — oracle upper bound
                x_gt = torch.cat([f_t, f_t1], dim=1)
                pred_gt = idm_model(x_gt)[0] * act_std + act_mean

                # 4. IDM on (real_t, real_t) — "no motion" baseline
                x_base = torch.cat([f_t, f_t], dim=1)
                pred_base = idm_model(x_base)[0] * act_std + act_mean

            mae_vm_idm.append((pred_vm - gt_action).abs().cpu().numpy())
            mae_idm_gt.append((pred_gt - gt_action).abs().cpu().numpy())
            mae_baseline.append((pred_base - gt_action).abs().cpu().numpy())

    mae_vm_idm = np.stack(mae_vm_idm)
    mae_idm_gt = np.stack(mae_idm_gt)
    mae_baseline = np.stack(mae_baseline)

    def log_split(arr, prefix):
        return {
            f"{prefix}/mae": arr.mean(),
            f"{prefix}/mae_steer": arr[:, 0].mean(),
            f"{prefix}/mae_accel": arr[:, 1].mean(),
        }

    metrics = {
        **log_split(mae_vm_idm, "vm_idm"),
        **log_split(mae_idm_gt, "oracle_idm"),
        **log_split(mae_baseline, "baseline"),
        "n_samples": len(mae_vm_idm),
    }

    print("\n── Results ──────────────────────────────────────────────")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print("─────────────────────────────────────────────────────────\n")

    wandb.log(metrics)
    wandb.finish()


@app.local_entrypoint()
def main(
    idm_ckpt: str,
    video_model_ckpt: str = "",
    run_name: str = "",
    context_frames: int = 8,
    n_eval_sequences: int = 200,
):
    call = rollout.spawn(
        idm_ckpt=idm_ckpt,
        video_model_ckpt=video_model_ckpt or "",
        run_name=run_name or None,
        context_frames=context_frames,
        n_eval_sequences=n_eval_sequences,
    )
    print(f"Spawned: {call.object_id}")
    print("Logs: modal app logs idm-rollout")
