import torch


class VideoModel:
    """
    Interface for any video prediction model.

    Args:
        past_frames: (B, T, 3, H, W) float32 in [0, 1]
    Returns:
        predicted next frame: (B, 3, H, W) float32 in [0, 1]
    """

    def predict_next_frame(self, past_frames):
        raise NotImplementedError

    def load_weights(self, ckpt_path):
        raise NotImplementedError


class LastFrameBaseline(VideoModel):
    """Trivial baseline: predicts that the next frame == the last frame."""

    def predict_next_frame(self, past_frames):
        return past_frames[:, -1]  # (B, 3, H, W)

    def load_weights(self, ckpt_path):
        pass


class TorchScriptVideoModel(VideoModel):
    """
    Wraps any TorchScript-exported video model.
    Expects forward signature: (B, T, 3, H, W) -> (B, 3, H, W)
    """

    def __init__(self, device="cuda"):
        self.model = None
        self.device = device

    def load_weights(self, ckpt_path):
        self.model = torch.jit.load(ckpt_path, map_location=self.device)
        self.model.eval()

    def predict_next_frame(self, past_frames):
        with torch.no_grad():
            return self.model(past_frames.to(self.device))


class CheckpointVideoModel(VideoModel):
    """
    Wraps a nn.Module whose state_dict is stored in a standard torch checkpoint.
    Override `build_model` to return your architecture, then load_weights will
    fill in the state dict.
    """

    def __init__(self, device="cuda"):
        self.model = None
        self.device = device

    def build_model(self):
        raise NotImplementedError("Override build_model() with your architecture.")

    def load_weights(self, ckpt_path):
        self.model = self.build_model().to(self.device)
        ckpt = torch.load(ckpt_path, map_location=self.device)
        state = ckpt.get("model_state_dict", ckpt)
        self.model.load_state_dict(state)
        self.model.eval()

    def predict_next_frame(self, past_frames):
        with torch.no_grad():
            return self.model(past_frames.to(self.device))
