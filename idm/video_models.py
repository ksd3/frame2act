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
