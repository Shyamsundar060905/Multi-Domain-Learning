import torch.nn.functional as F
import torch.nn as nn
import torch
class ChangeDetectionModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

        # 🔥 reduce channels
        self.reduce = nn.Conv2d(2048, 256, kernel_size=1)

        # 🔥 segmentation head
        self.head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1)   # ✅ binary output
        )

    def forward(self, img1, img2, domain):
        f1 = self.backbone(img1, domain)
        f2 = self.backbone(img2, domain)

        # 🔥 feature difference
        diff = torch.abs(f1 - f2)

        x = self.reduce(diff)
        x = self.head(x)

        # 🔥 upsample to input size
        x = F.interpolate(x, size=img1.shape[-2:], mode='bilinear', align_corners=False)

        return x