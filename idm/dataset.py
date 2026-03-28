import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class IDMDataset(Dataset):
    def __init__(self, indices, frames_list, actions_list,
                 img_h, img_w, act_mean, act_std):
        self.indices = indices
        self.frames = frames_list
        self.actions = actions_list
        self.img_h, self.img_w = img_h, img_w
        self.act_mean = torch.from_numpy(act_mean)
        self.act_std = torch.from_numpy(act_std)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        fi, t = self.indices[idx]
        f0 = self.frames[fi][t]
        f1 = self.frames[fi][t + 1]
        act = self.actions[fi][t]

        f0 = torch.from_numpy(f0).permute(2, 0, 1).float() / 255.0
        f1 = torch.from_numpy(f1).permute(2, 0, 1).float() / 255.0

        f0 = F.interpolate(f0.unsqueeze(0), size=(self.img_h, self.img_w),
                           mode="bilinear", align_corners=False).squeeze(0)
        f1 = F.interpolate(f1.unsqueeze(0), size=(self.img_h, self.img_w),
                           mode="bilinear", align_corners=False).squeeze(0)

        x = torch.cat([f0, f1], dim=0)  # (6, H, W)
        y = (torch.from_numpy(act.copy()) - self.act_mean) / self.act_std
        return x, y
