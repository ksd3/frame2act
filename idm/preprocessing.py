import torch
import torch.nn.functional as F

IMG_H = 90
IMG_W = 160


def preprocess(frame_np, img_h=IMG_H, img_w=IMG_W):
    """(H, W, 3) uint8 -> (1, 3, img_h, img_w) float32"""
    t = torch.from_numpy(frame_np).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    return F.interpolate(t, size=(img_h, img_w), mode="bilinear", align_corners=False)
