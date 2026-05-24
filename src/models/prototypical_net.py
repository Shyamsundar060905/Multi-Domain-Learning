import torch
import torch.nn as nn
import torch.nn.functional as F

class PrototypicalNetwork(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    # ----------------------------
    # FEATURE EXTRACTION
    # ----------------------------
    def extract_features(self, img, domain):
        x = self.backbone.stem(img)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.adapters[domain]['layer3'](x)
        x = self.backbone.adapters[domain]['layer4'](x)
        return x   # [B, 2048, H/32, W/32]

    # ----------------------------
    # PROTOTYPE COMPUTATION
    # ----------------------------
    def compute_prototype(self, feat, mask):
        """
        feat: [B, C, H, W]
        mask: [B, 1, H, W] (0/1)
        """
        masked_feat = feat * mask
        proto = masked_feat.sum(dim=(0, 2, 3)) / (mask.sum() + 1e-6)
        return proto  # [C]

    # ----------------------------
    # COSINE SIMILARITY MAP
    # ----------------------------
    def cosine_similarity_map(self, feat, prototype):
        B, C, H, W = feat.shape

        feat_flat = feat.view(B, C, -1)             # [B, C, HW]
        proto = prototype.view(1, C, 1)             # [1, C, 1]

        sim = F.cosine_similarity(feat_flat, proto, dim=1)
        sim = sim.view(B, 1, H, W)

        return sim

    # ----------------------------
    # FORWARD (TRAINING EPISODE)
    # ----------------------------
    def forward(
        self,
        support_img1, support_img2, support_mask,
        query_img1, query_img2,
        domain
    ):
        """
        support_*: [Ns, ...]
        query_*: [Nq, ...]
        """

        # ---- SUPPORT ----
        f1_s = self.extract_features(support_img1, domain)
        f2_s = self.extract_features(support_img2, domain)

        diff_s = torch.abs(f1_s - f2_s)

        # Resize masks to feature size
        support_mask = F.interpolate(
            support_mask, size=diff_s.shape[-2:], mode='nearest'
        )

        # Change prototype
        change_proto = self.compute_prototype(diff_s, support_mask)

        # No-change prototype
        no_change_mask = 1 - support_mask
        no_change_proto = self.compute_prototype(diff_s, no_change_mask)

        # ---- QUERY ----
        f1_q = self.extract_features(query_img1, domain)
        f2_q = self.extract_features(query_img2, domain)

        diff_q = torch.abs(f1_q - f2_q)

        # Similarity maps
        sim_change = self.cosine_similarity_map(diff_q, change_proto)
        sim_no_change = self.cosine_similarity_map(diff_q, no_change_proto)

        # Stack logits
        logits = torch.cat([sim_no_change, sim_change], dim=1)

        # Upsample to original resolution
        logits = F.interpolate(
            logits,
            size=query_img1.shape[-2:],
            mode='bilinear',
            align_corners=False
        )

        return logits